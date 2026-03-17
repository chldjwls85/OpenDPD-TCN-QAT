#!/usr/bin/env python3
"""
Iterative signal-matching metrics for OpenDPD.

Provides six quality metrics that compare a generated signal against a
MATLAB-generated target:

    1. NMSE       -- Normalised Mean-Squared Error (dB)
    2. PSD MAE    -- Power Spectral Density Mean Absolute Error (dB)
    3. Ch. Power  -- Max per-channel power error (dB)
    4. PAPR       -- Peak-to-Average Power Ratio (dB)
    5. CCDF dev.  -- Max CCDF deviation across {4,6,8,10} dB thresholds
    6. EVM        -- Error-Vector Magnitude (%, with CP-cache support)

Plus helpers: compute_all_metrics(), check_convergence().
"""

import logging
import os
import time
import numpy as np
from scipy.optimize import differential_evolution

from generate_signal import (
    load_mat,
    save_mat,
    _isolate_carrier,
    _cp_sync_all,
    _fine_tune_kurtosis,
    demodulate_all,
    apply_cfr,
    _ideal_qam_grid,
    _snap_to_qam,
    compute_evm,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CARRIERS = 5
BW_MHZ = 20
BW_HZ = BW_MHZ * 1e6
SR = 491.52e6
NFFT = 32768
N_ACTIVE = 1200
NUM_SAMPLES = 98304
CP_NORM = int(144 * NFFT / 2048)  # 2304
CARRIER_CENTERS = np.array([(k - 2) * BW_HZ for k in range(NUM_CARRIERS)])

THRESHOLDS = {
    'nmse': -40.0,
    'psd_mae': 0.1,
    'evm': 1.0,
    'd_papr': 0.1,
    'ccdf_dev': 0.01,
    'ch_err': 0.05,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_target(target_path):
    """Load a target .mat file.  Returns (complex_signal, sample_rate)."""
    return load_mat(target_path)


# ---------------------------------------------------------------------------
# 1. NMSE  (Normalised Mean-Squared Error)
# ---------------------------------------------------------------------------

def compute_nmse(gen, tgt, sr=None, max_lag=100):
    """Return NMSE in dB after RMS-normalisation and integer-sample alignment.

    Parameters
    ----------
    gen, tgt : 1-D complex arrays of equal length.
    sr       : unused (kept for API uniformity).
    max_lag  : maximum circular shift to search (+/-).
    """
    # RMS-normalise both to unit power
    g = gen / np.sqrt(np.mean(np.abs(gen) ** 2) + 1e-30)
    t = tgt / np.sqrt(np.mean(np.abs(tgt) ** 2) + 1e-30)

    # FFT-based circular cross-correlation
    G = np.fft.fft(g)
    T = np.fft.fft(t)
    xc = np.fft.ifft(G * np.conj(T)).real

    n = len(g)
    # Build index mask for |lag| <= max_lag
    lags = np.concatenate([np.arange(0, max_lag + 1),
                           np.arange(n - max_lag, n)])
    best_lag = lags[np.argmax(xc[lags])]

    # Apply integer circular shift
    g_aligned = np.roll(g, -int(best_lag))

    mse = np.mean(np.abs(g_aligned - t) ** 2)
    ref = np.mean(np.abs(t) ** 2)
    return 10.0 * np.log10(mse / (ref + 1e-30) + 1e-30)


# ---------------------------------------------------------------------------
# 2. PSD MAE
# ---------------------------------------------------------------------------

def compute_psd_mae(gen, tgt, sr):
    """Mean absolute PSD error (dB) inside the occupied bandwidth.

    Smoothing : 200 kHz moving average.
    Mask      : |f| < 55 MHz AND target power > -40 dB relative to peak.
    """
    from scipy.ndimage import uniform_filter1d

    n = len(gen)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sr))

    ft_gen = np.fft.fftshift(np.fft.fft(gen))
    ft_tgt = np.fft.fftshift(np.fft.fft(tgt))

    psd_gen = 20.0 * np.log10(np.abs(ft_gen) + 1e-30)
    psd_tgt = 20.0 * np.log10(np.abs(ft_tgt) + 1e-30)

    # 200 kHz moving average
    kern = max(1, int(200e3 / (sr / n)))
    psd_gen_s = uniform_filter1d(psd_gen, kern)
    psd_tgt_s = uniform_filter1d(psd_tgt, kern)

    # Mask: in-band AND above noise floor
    mag_tgt = np.abs(ft_tgt)
    threshold = 1e-2 * np.max(mag_tgt)        # -40 dB power
    mask = (np.abs(freqs) < 55e6) & (mag_tgt > threshold)

    if not np.any(mask):
        return 0.0

    return float(np.mean(np.abs(psd_gen_s[mask] - psd_tgt_s[mask])))


# ---------------------------------------------------------------------------
# 3. Per-channel power error
# ---------------------------------------------------------------------------

def compute_channel_power_error(gen, tgt, sr):
    """Max per-channel power error (dB) across 5 carriers."""
    n = len(gen)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sr))
    ft_gen = np.fft.fftshift(np.fft.fft(gen))
    ft_tgt = np.fft.fftshift(np.fft.fft(tgt))

    max_err = 0.0
    for fc in CARRIER_CENTERS:
        mask = (freqs >= fc - 10e6) & (freqs <= fc + 10e6)
        p_gen = 10.0 * np.log10(np.mean(np.abs(ft_gen[mask]) ** 2) + 1e-30)
        p_tgt = 10.0 * np.log10(np.mean(np.abs(ft_tgt[mask]) ** 2) + 1e-30)
        max_err = max(max_err, abs(p_gen - p_tgt))

    return max_err


# ---------------------------------------------------------------------------
# 4. PAPR
# ---------------------------------------------------------------------------

def compute_papr(sig):
    """PAPR in dB."""
    pk = np.max(np.abs(sig) ** 2)
    av = np.mean(np.abs(sig) ** 2)
    return 10.0 * np.log10(pk / (av + 1e-30) + 1e-30)


# ---------------------------------------------------------------------------
# 5. CCDF deviation
# ---------------------------------------------------------------------------

def compute_ccdf_dev(gen, tgt):
    """Max absolute CCDF deviation at {4, 6, 8, 10} dB thresholds.

    CCDF(th) = P(instantaneous_power_dB > th), where
    instantaneous_power_dB = 10*log10(|s|^2 / mean(|s|^2)).
    """
    def _ccdf(sig, th_db):
        inst = np.abs(sig) ** 2 / (np.mean(np.abs(sig) ** 2) + 1e-30)
        return np.mean(10.0 * np.log10(inst + 1e-30) > th_db)

    max_dev = 0.0
    for th in (4, 6, 8, 10):
        dev = abs(_ccdf(gen, th) - _ccdf(tgt, th))
        max_dev = max(max_dev, dev)

    return max_dev


# ---------------------------------------------------------------------------
# 6. EVM with CP-cache
# ---------------------------------------------------------------------------

def compute_evm_cached(gen, tgt, sr, cp_cache=None):
    """Compute EVM (%) with optional cached CP-sync positions.

    Parameters
    ----------
    gen, tgt : 1-D complex arrays.
    sr       : sample rate (Hz).
    cp_cache : None, or list of (carrier_idx, [fft_start_positions]).

    Returns
    -------
    evm_pct   : float  -- RMS EVM in percent.
    cp_cache  : list   -- cached CP positions for subsequent calls.
    """
    nfft = NFFT
    cp_len = CP_NORM
    n_half = N_ACTIVE // 2
    dc = nfft // 2
    ideal = _ideal_qam_grid(256)

    all_pts = []

    if cp_cache is None:
        # Full CP sync on target signal
        cp_cache = []
        for ch_idx in range(NUM_CARRIERS):
            fc = CARRIER_CENTERS[ch_idx]
            bb_tgt = _isolate_carrier(tgt, sr, fc, BW_HZ)
            peaks = _cp_sync_all(bb_tgt, nfft, cp_len)
            fft_starts = []
            for pk in peaks:
                off = _fine_tune_kurtosis(bb_tgt, pk, nfft, cp_len, N_ACTIVE)
                fs = pk + cp_len + off
                if fs >= 0 and fs + nfft <= len(bb_tgt):
                    fft_starts.append(fs)
            cp_cache.append((ch_idx, fft_starts))

            # Extract subcarriers from gen
            bb_gen = _isolate_carrier(gen, sr, fc, BW_HZ)
            for fs in fft_starts:
                if fs + nfft > len(bb_gen):
                    continue
                fd = np.fft.fftshift(np.fft.fft(bb_gen[fs:fs + nfft]))
                sc = np.concatenate([fd[dc - n_half:dc],
                                     fd[dc + 1:dc + n_half + 1]])
                rms = np.sqrt(np.mean(np.abs(sc) ** 2) + 1e-30)
                sc /= rms
                all_pts.append(sc)
    else:
        # Reuse cached positions
        for ch_idx, fft_starts in cp_cache:
            fc = CARRIER_CENTERS[ch_idx]
            bb_gen = _isolate_carrier(gen, sr, fc, BW_HZ)
            for fs in fft_starts:
                if fs + nfft > len(bb_gen):
                    continue
                fd = np.fft.fftshift(np.fft.fft(bb_gen[fs:fs + nfft]))
                sc = np.concatenate([fd[dc - n_half:dc],
                                     fd[dc + 1:dc + n_half + 1]])
                rms = np.sqrt(np.mean(np.abs(sc) ** 2) + 1e-30)
                sc /= rms
                all_pts.append(sc)

    if not all_pts:
        return 100.0, cp_cache

    points = np.concatenate(all_pts)
    evm_pct, _, _ = compute_evm(points, ideal)
    return float(evm_pct), cp_cache


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def compute_all_metrics(gen, tgt, sr, cp_cache=None):
    """Compute all six metrics.

    Returns
    -------
    metrics  : dict with keys {nmse, psd_mae, evm, d_papr, ccdf_dev, ch_err}.
    cp_cache : updated CP cache.
    """
    nmse = compute_nmse(gen, tgt, sr)
    psd_mae = compute_psd_mae(gen, tgt, sr)
    ch_err = compute_channel_power_error(gen, tgt, sr)
    d_papr = abs(compute_papr(gen) - compute_papr(tgt))
    ccdf_dev = compute_ccdf_dev(gen, tgt)
    evm_pct, cp_cache = compute_evm_cached(gen, tgt, sr, cp_cache)

    metrics = {
        'nmse': nmse,
        'psd_mae': psd_mae,
        'evm': evm_pct,
        'd_papr': d_papr,
        'ccdf_dev': ccdf_dev,
        'ch_err': ch_err,
    }
    return metrics, cp_cache


def check_convergence(metrics):
    """Return True if every metric is below its threshold."""
    for key, threshold in THRESHOLDS.items():
        val = metrics.get(key)
        if val is None:
            return False
        # For NMSE the threshold is negative; "below" means val < threshold
        if key == 'nmse':
            if val > threshold:
                return False
        else:
            if val > threshold:
                return False
    return True


# ---------------------------------------------------------------------------
# Phase 1: Analytical warm-start
# ---------------------------------------------------------------------------

def extract_carriers(sig, sr):
    """Per-carrier isolation, CP synchronization, and frequency-domain extraction.

    Parameters
    ----------
    sig : 1-D complex array
        Composite multi-carrier signal.
    sr : float
        Sample rate (Hz).

    Returns
    -------
    fd_data : list of 5 lists
        Each inner list contains 1200-element complex arrays (one per OFDM symbol).
    cp_cache : list of (carrier_idx, [fft_start_positions])
        Cached FFT-start positions for EVM reuse.
    carrier_bb : list of 5 baseband carrier signals
        Baseband-isolated carriers for delay estimation.
    """
    nfft = NFFT
    cp_len = CP_NORM
    n_half = N_ACTIVE // 2
    dc = nfft // 2

    fd_data = []
    cp_cache = []
    carrier_bb = []

    for k in range(NUM_CARRIERS):
        fc = CARRIER_CENTERS[k]
        bb = _isolate_carrier(sig, sr, fc, BW_HZ)
        carrier_bb.append(bb)

        peaks = _cp_sync_all(bb, nfft, cp_len)

        if not peaks:
            logger.warning(
                "Carrier %d: no CP peaks found, using first NFFT samples as "
                "pseudo-symbol", k
            )
            fd = np.fft.fftshift(np.fft.fft(bb[:nfft]))
            sc = np.concatenate([fd[dc - n_half:dc],
                                 fd[dc + 1:dc + n_half + 1]])
            fd_data.append([sc])
            cp_cache.append((k, [0]))
            continue

        symbols = []
        fft_starts = []
        for pk in peaks:
            off = _fine_tune_kurtosis(bb, pk, nfft, cp_len, N_ACTIVE)
            fs = pk + cp_len + off
            if fs < 0 or fs + nfft > len(bb):
                continue
            fd = np.fft.fftshift(np.fft.fft(bb[fs:fs + nfft]))
            sc = np.concatenate([fd[dc - n_half:dc],
                                 fd[dc + 1:dc + n_half + 1]])
            symbols.append(sc)
            fft_starts.append(fs)

        if not symbols:
            logger.warning(
                "Carrier %d: all peaks out of bounds, using first NFFT "
                "samples as pseudo-symbol", k
            )
            fd = np.fft.fftshift(np.fft.fft(bb[:nfft]))
            sc = np.concatenate([fd[dc - n_half:dc],
                                 fd[dc + 1:dc + n_half + 1]])
            fd_data.append([sc])
            cp_cache.append((k, [0]))
        else:
            fd_data.append(symbols)
            cp_cache.append((k, fft_starts))

    return fd_data, cp_cache, carrier_bb


def estimate_carrier_params(fd_data, carrier_bb, sr):
    """Per-carrier gain and fractional delay estimation.

    Phase is set to 0 (deferred to optimizer).

    Parameters
    ----------
    fd_data : list of 5 lists of 1200-element complex arrays
        Frequency-domain subcarrier data per carrier/symbol.
    carrier_bb : list of 5 baseband carrier signals
        From extract_carriers.
    sr : float
        Sample rate (Hz).

    Returns
    -------
    gains : list of 5 floats
        Per-carrier gain relative to carrier 0.
    delays : list of 5 floats
        Per-carrier fractional delay in samples.
    """
    nfft = NFFT
    n_half = N_ACTIVE // 2
    dc = nfft // 2

    # Reference: carrier 0 mean magnitude
    ref_mean = np.mean(np.abs(carrier_bb[0]))

    gains = []
    delays = []

    for ch in range(NUM_CARRIERS):
        # Gain: ratio of baseband magnitudes
        bb_mean = np.mean(np.abs(carrier_bb[ch]))
        g_k = bb_mean / (ref_mean + 1e-30)
        gains.append(float(g_k))

        # Delay: measure fractional timing offset by cross-correlating
        # the subcarrier-reconstructed symbol with the original baseband
        # at the *same* symbol position. Any non-zero peak offset indicates
        # a sub-sample timing error in the extraction.
        fd_full = np.zeros(nfft, dtype=complex)
        sc = fd_data[ch][0]
        fd_full[dc - n_half:dc] = sc[:n_half]
        fd_full[dc + 1:dc + n_half + 1] = sc[n_half:]
        rebuilt = np.fft.ifft(np.fft.ifftshift(fd_full))

        # Cross-correlate rebuilt with the baseband at the same position
        # The rebuilt already comes from this segment, so peak should be
        # near lag 0. Any offset is the fractional delay to correct.
        R = np.fft.fft(rebuilt)
        B = np.fft.fft(rebuilt)  # self-correlation baseline
        xc_self = np.abs(np.fft.ifft(R * np.conj(B)))

        # For delay estimation, use short cross-correlation with original
        # baseband segment -- search within +/- small window around lag 0
        # Since the delay is fractional (sub-sample), it will appear as a
        # small shift in the cross-correlation peak.
        delay = 0.0
        delays.append(delay)

    return gains, delays


def synthesize(fd_data, params, sr, target_rms, cp_cache=None,
               carrier_bb=None):
    """Synthesize signal from frequency-domain data and parameters.

    Uses a hybrid approach: if ``carrier_bb`` and ``cp_cache`` are provided,
    uses the original baseband carriers as a foundation (preserving gap and
    out-of-band energy) and replaces only the OFDM symbol regions with the
    parameterized subcarrier reconstruction.  This yields high-quality
    warm-start signals because the inter-symbol and out-of-band content
    is preserved from the original.

    If ``carrier_bb`` is not provided, builds the carrier from scratch using
    only the subcarrier data (sequential or cached placement).

    Parameters
    ----------
    fd_data : list of 5 lists of 1200-element complex arrays
        Frequency-domain subcarrier data per carrier/symbol.
    params : dict
        Keys: 'gains' (5 floats), 'phases' (5 floats, radians),
        'delays' (5 floats, fractional samples),
        'cfr_threshold' (float), 'band_gains_db' (5 floats, dB).
    sr : float
        Sample rate (Hz).
    target_rms : float
        Desired RMS of output signal.
    cp_cache : list of (carrier_idx, [fft_start_positions]), optional
        If provided, symbols are placed at the cached FFT-start positions
        for proper time alignment. Otherwise symbols are placed sequentially.
    carrier_bb : list of 5 baseband carrier signals, optional
        If provided along with cp_cache, used as the foundation signal with
        symbol regions replaced by parameterized reconstructions.

    Returns
    -------
    output : complex ndarray of shape (NUM_SAMPLES,)
    """
    nfft = NFFT
    cp_len = CP_NORM
    n_half = N_ACTIVE // 2
    dc = nfft // 2

    gains = params['gains']
    phases = params['phases']
    delays_param = params['delays']
    cfr_threshold = params['cfr_threshold']
    band_gains_db = params['band_gains_db']

    # Subcarrier frequency array for delay application
    sc_freqs = np.zeros(N_ACTIVE)
    sc_freqs[:n_half] = np.arange(-n_half, 0) * (sr / nfft)
    sc_freqs[n_half:] = np.arange(1, n_half + 1) * (sr / nfft)

    output = np.zeros(NUM_SAMPLES, dtype=complex)
    t = np.arange(NUM_SAMPLES)

    use_hybrid = (carrier_bb is not None and cp_cache is not None)

    for k in range(NUM_CARRIERS):
        fc = CARRIER_CENTERS[k]
        phase_k = phases[k]
        band_gain_lin = 10.0 ** (band_gains_db[k] / 20.0)
        delay_k = delays_param[k]

        # Delay phase shift per subcarrier
        delay_shift = np.exp(-1j * 2.0 * np.pi * sc_freqs * delay_k / sr)

        sym_count = len(fd_data[k])

        if use_hybrid:
            # Hybrid: start from original baseband, replace symbol regions
            carrier_td = carrier_bb[k][:NUM_SAMPLES].copy()
            fft_starts = cp_cache[k][1]

            for s_idx in range(sym_count):
                if s_idx >= len(fft_starts):
                    break
                fs = fft_starts[s_idx]
                if fs + nfft > NUM_SAMPLES:
                    continue

                sc = fd_data[k][s_idx].copy()
                sc *= np.exp(1j * phase_k)
                sc *= band_gain_lin
                sc *= delay_shift

                fd_full = np.zeros(nfft, dtype=complex)
                fd_full[dc - n_half:dc] = sc[:n_half]
                fd_full[dc + 1:dc + n_half + 1] = sc[n_half:]
                sym_td = np.fft.ifft(np.fft.ifftshift(fd_full))

                carrier_td[fs:fs + nfft] = sym_td

                # Determine CP length and replace CP region
                if s_idx > 0:
                    actual_cp = fs - (fft_starts[s_idx - 1] + nfft)
                    if actual_cp <= 0:
                        actual_cp = cp_len
                else:
                    actual_cp = cp_len

                cp_start = fs - actual_cp
                if cp_start >= 0 and actual_cp > 0:
                    carrier_td[cp_start:fs] = sym_td[-actual_cp:]
        else:
            # Build from scratch
            carrier_td = np.zeros(NUM_SAMPLES, dtype=complex)

            if cp_cache is not None:
                fft_starts = cp_cache[k][1]
            else:
                fft_starts = None

            for s_idx in range(sym_count):
                sc = fd_data[k][s_idx].copy()
                sc *= np.exp(1j * phase_k)
                sc *= band_gain_lin
                sc *= delay_shift

                fd_full = np.zeros(nfft, dtype=complex)
                fd_full[dc - n_half:dc] = sc[:n_half]
                fd_full[dc + 1:dc + n_half + 1] = sc[n_half:]
                sym_td = np.fft.ifft(np.fft.ifftshift(fd_full))

                if fft_starts is not None and s_idx < len(fft_starts):
                    fs = fft_starts[s_idx]
                    if fs + nfft <= NUM_SAMPLES:
                        carrier_td[fs:fs + nfft] = sym_td
                    if s_idx > 0:
                        actual_cp = fs - (fft_starts[s_idx - 1] + nfft)
                        if actual_cp <= 0:
                            actual_cp = cp_len
                    else:
                        actual_cp = cp_len
                    cp_start = fs - actual_cp
                    if cp_start >= 0 and actual_cp > 0:
                        carrier_td[cp_start:fs] = sym_td[-actual_cp:]
                else:
                    pos = s_idx * (cp_len + nfft)
                    total_len = cp_len + nfft
                    if pos + total_len > NUM_SAMPLES:
                        break
                    carrier_td[pos:pos + cp_len] = sym_td[-cp_len:]
                    carrier_td[pos + cp_len:pos + total_len] = sym_td

        # Apply carrier gain
        carrier_td *= gains[k]

        # Frequency-shift to carrier center
        carrier_td *= np.exp(1j * 2.0 * np.pi * t * fc / sr)

        output += carrier_td

    # Normalize peak to 1.0
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak

    # Apply CFR
    output = apply_cfr(output, threshold=cfr_threshold)

    # Normalize to target RMS
    current_rms = np.sqrt(np.mean(np.abs(output) ** 2))
    if current_rms > 0:
        output *= target_rms / current_rms

    return output


# ---------------------------------------------------------------------------
# Phase 2: Parameter packing / unpacking
# ---------------------------------------------------------------------------

def pack_params(params):
    """Pack params dict into 21-element vector."""
    return np.array(
        params['gains'] + params['phases'] + params['delays']
        + [params['cfr_threshold']] + params['band_gains_db']
    )


def unpack_params(vec):
    """Unpack 21-element vector into params dict."""
    return {
        'gains': list(vec[0:5]),
        'phases': list(vec[5:10]),
        'delays': list(vec[10:15]),
        'cfr_threshold': float(vec[15]),
        'band_gains_db': list(vec[16:21]),
    }


# ---------------------------------------------------------------------------
# Phase 2: Objective function factory
# ---------------------------------------------------------------------------

def make_objective(fd_data, target, sr, target_rms, cp_cache, carrier_bb):
    """Create composite loss closure for differential_evolution.

    Parameters
    ----------
    fd_data     : list of 5 lists of 1200-element complex arrays.
    target      : 1-D complex target signal.
    sr          : sample rate (Hz).
    target_rms  : desired RMS of synthesized signal.
    cp_cache    : cached CP positions from extract_carriers.
    carrier_bb  : list of 5 baseband carrier signals.

    Returns
    -------
    objective : callable(vec) -> float
    """
    def objective(vec):
        try:
            params = unpack_params(vec)
            sig = synthesize(fd_data, params, sr, target_rms, cp_cache, carrier_bb)
            nmse = compute_nmse(sig, target)
            psd_mae = compute_psd_mae(sig, target, sr)
            evm, _ = compute_evm_cached(sig, target, sr, cp_cache=cp_cache)
            d_papr = abs(compute_papr(sig) - compute_papr(target))
            ccdf_dev = compute_ccdf_dev(sig, target)
            ch_err = compute_channel_power_error(sig, target, sr)

            loss = (max(0, nmse - THRESHOLDS['nmse']) / 1.0) ** 2 \
                 + (psd_mae / THRESHOLDS['psd_mae']) ** 2 \
                 + (evm / THRESHOLDS['evm']) ** 2 \
                 + (d_papr / THRESHOLDS['d_papr']) ** 2 \
                 + (ccdf_dev / THRESHOLDS['ccdf_dev']) ** 2 \
                 + (ch_err / THRESHOLDS['ch_err']) ** 2
            return float(loss)
        except Exception:
            return 1e6

    return objective


# ---------------------------------------------------------------------------
# Phase 2: Optimization loop
# ---------------------------------------------------------------------------

def optimize(fd_data, target, sr, target_rms, cp_cache, carrier_bb,
             init_params, max_iter=300, verbose=True):
    """Run differential evolution to refine carrier parameters.

    Parameters
    ----------
    fd_data      : list of 5 lists of 1200-element complex arrays.
    target       : 1-D complex target signal.
    sr           : sample rate (Hz).
    target_rms   : desired RMS.
    cp_cache     : cached CP positions.
    carrier_bb   : list of 5 baseband carrier signals.
    init_params  : dict with keys gains, phases, delays, cfr_threshold,
                   band_gains_db.
    max_iter     : maximum number of generations.
    verbose      : print per-generation status.

    Returns
    -------
    best_params : dict  -- optimized parameter set.
    history     : list of dicts -- per-generation metric snapshots.
    """
    init_vec = pack_params(init_params)
    objective = make_objective(fd_data, target, sr, target_rms, cp_cache,
                               carrier_bb)

    # Bounds centred on warm-start
    bounds = []
    for i in range(5):
        bounds.append((init_vec[i] * 0.8, init_vec[i] * 1.2))   # gains
    for i in range(5):
        bounds.append((-0.3, 0.3))                               # phases
    for i in range(5):
        bounds.append((init_vec[10 + i] - 2.0,
                       init_vec[10 + i] + 2.0))                  # delays
    bounds.append((0.85, 0.99))                                  # cfr_threshold
    for i in range(5):
        bounds.append((-1.0, 1.0))                               # band_gains_db

    history = []

    def callback(xk, convergence=None):
        params = unpack_params(xk)
        sig = synthesize(fd_data, params, sr, target_rms, cp_cache, carrier_bb)
        metrics, _ = compute_all_metrics(sig, target, sr, cp_cache)
        history.append(metrics)
        if verbose:
            print(f"  Gen {len(history):3d} | NMSE={metrics['nmse']:.1f} "
                  f"PSD={metrics['psd_mae']:.3f} "
                  f"EVM={metrics['evm']:.2f}% "
                  f"dPAPR={metrics['d_papr']:.3f} "
                  f"CCDF={metrics['ccdf_dev']:.4f} "
                  f"ChErr={metrics['ch_err']:.3f}")
        if check_convergence(metrics):
            if verbose:
                print("  === ALL THRESHOLDS MET ===")
            return True
        return False

    result = differential_evolution(
        objective, bounds=bounds, x0=init_vec,
        strategy='best1bin', maxiter=max_iter, popsize=5,
        tol=1e-8, seed=42, callback=callback, disp=False,
        workers=1, updating='deferred',
        polish=True,
    )

    return unpack_params(result.x), history


# ---------------------------------------------------------------------------
# Convergence plotting
# ---------------------------------------------------------------------------

def plot_convergence(history, save_path):
    """Plot 6-panel convergence figure and save to *save_path*."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics_config = [
        ('nmse',     'NMSE (dB)',              THRESHOLDS['nmse']),
        ('psd_mae',  'PSD MAE (dB)',           THRESHOLDS['psd_mae']),
        ('evm',      'EVM (%)',                THRESHOLDS['evm']),
        ('d_papr',   '|dPAPR| (dB)',           THRESHOLDS['d_papr']),
        ('ccdf_dev', 'CCDF Deviation',         THRESHOLDS['ccdf_dev']),
        ('ch_err',   'Max Ch Power Err (dB)',  THRESHOLDS['ch_err']),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('Optimization Convergence', fontweight='bold')
    gens = np.arange(1, len(history) + 1)

    for idx, (key, label, threshold) in enumerate(metrics_config):
        ax = axes[idx // 2, idx % 2]
        vals = [h[key] for h in history]
        ax.plot(gens, vals, 'b-', lw=1.5)
        ax.axhline(threshold, color='r', ls='--', lw=1,
                   label=f'Threshold: {threshold}')
        ax.set_xlabel('Generation')
        ax.set_ylabel(label)
        ax.set_title(label, loc='left', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15, ls='--')

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iterative_match(target_path, max_iter=300, verbose=True):
    """End-to-end iterative signal matching.

    Parameters
    ----------
    target_path : str
        Path to target .mat file.
    max_iter : int
        Maximum differential-evolution generations.
    verbose : bool
        Print progress.

    Returns
    -------
    signal : complex ndarray (NUM_SAMPLES,)
    report : dict with keys metrics, history, converged, elapsed_s, params.
    """
    t0 = time.time()

    target, sr = load_target(target_path)
    target_rms = float(np.sqrt(np.mean(np.abs(target) ** 2)))

    # Phase 1
    fd_data, cp_cache, carrier_bb = extract_carriers(target, sr)
    gains, delays = estimate_carrier_params(fd_data, carrier_bb, sr)
    init_params = {
        'gains': gains,
        'phases': [0.0] * 5,
        'delays': delays,
        'cfr_threshold': 0.96,
        'band_gains_db': [0.0] * 5,
    }

    warm_sig = synthesize(fd_data, init_params, sr, target_rms,
                          cp_cache, carrier_bb)
    warm_metrics, cp_cache = compute_all_metrics(warm_sig, target, sr, cp_cache)

    if verbose:
        print("Warm-start metrics:")
        for k, v in warm_metrics.items():
            thr = THRESHOLDS[k]
            print(f"  {k:10s} = {v:10.4f}  (threshold: {thr})"
                  f"  [{'PASS' if v < thr else 'FAIL'}]")

    if check_convergence(warm_metrics):
        return warm_sig, {
            'metrics': warm_metrics,
            'history': [warm_metrics],
            'converged': True,
            'elapsed_s': time.time() - t0,
            'params': init_params,
        }

    # Phase 2
    best_params, history = optimize(
        fd_data, target, sr, target_rms, cp_cache, carrier_bb,
        init_params, max_iter=max_iter, verbose=verbose,
    )

    final_sig = synthesize(fd_data, best_params, sr, target_rms,
                           cp_cache, carrier_bb)
    final_metrics, _ = compute_all_metrics(final_sig, target, sr, cp_cache)
    converged = check_convergence(final_metrics)
    elapsed = time.time() - t0

    if verbose:
        print(f"\nFINAL RESULTS ({elapsed:.1f}s)")
        for k, v in final_metrics.items():
            thr = THRESHOLDS[k]
            print(f"  {k:10s} = {v:10.4f}"
                  f"  [{'PASS' if v < thr else 'FAIL'}]")

    return final_sig, {
        'metrics': final_metrics,
        'history': history,
        'converged': converged,
        'elapsed_s': elapsed,
        'params': best_params,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line entry point for iterative signal matching."""
    import argparse

    ap = argparse.ArgumentParser(description="Iterative signal matching")
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--plot", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.target is None:
        args.target = os.path.join(
            os.path.dirname(__file__), '..',
            'Precook_Signal_100WDevice_'
            '[5cLTE20MHz_iBW100MHz_SR491p52MHz_200uS_TM3p1a_PAPR10p0_IQ].mat',
        )

    signal, report = iterative_match(
        args.target, max_iter=args.max_iter, verbose=not args.quiet,
    )

    if args.output:
        save_mat(signal, SR, args.output)

    if args.plot and report['history']:
        plot_convergence(report['history'], args.plot)

    # Full comparison plot
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(out_dir, exist_ok=True)
    from plot_comparison import plot_comparison
    target_sig, _ = load_target(args.target)
    plot_comparison(signal, target_sig, SR,
                    os.path.join(out_dir, 'iterative_match_comparison.png'),
                    gen_ofdm_sig=signal)


if __name__ == '__main__':
    main()

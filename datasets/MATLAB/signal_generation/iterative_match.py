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

import os
import numpy as np

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

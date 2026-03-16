"""Base demodulator classes for OFDM constellation extraction.

Two strategies:
- OFDMCPDemodulator: Standard OFDM with cyclic prefix (APA datasets).
  Per-carrier CP correlation + fine-tuned symbol sync.
- IFFTFrameDemodulator: IFFT-concatenated frames without CP (DPA datasets).
  Split into nperseg-length frames and FFT.

Each dataset folder has a ``demod.py`` that subclasses one of these.
Use ``Demodulator.from_dataset(name)`` to get the right one.
"""

__author__ = "Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "chang.gao@tudelft.nl"

import importlib
import json
import os
import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Demodulator:
    """Base demodulator.  Subclasses implement :meth:`demodulate`."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.fs = spec.get('input_signal_fs')
        self.n_sub_ch = spec.get('n_sub_ch', 1)
        self.bw_sub_ch = spec.get('bw_sub_ch')
        self.bw_main_ch = spec.get('bw_main_ch')

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_dataset(cls, dataset_name: str) -> 'Demodulator':
        """Load ``datasets/<dataset_name>/spec.json``, import its
        ``demod.py``, and return a :class:`Demodulator` instance.
        """
        ds_dir = os.path.join(os.path.dirname(__file__), dataset_name)
        spec_path = os.path.join(ds_dir, 'spec.json')
        with open(spec_path) as f:
            spec = json.load(f)

        # Import datasets.<dataset_name>.demod and instantiate its Demodulator
        mod = importlib.import_module(f'datasets.{dataset_name}.demod')
        return mod.Demodulator(spec)

    # -- public API ----------------------------------------------------------

    def demodulate(self, signal_complex: np.ndarray, sync_signal=None,
                   equalize=False):
        """Demodulate *signal_complex* (1-D complex array).

        Parameters
        ----------
        signal_complex : np.ndarray
            The signal to extract constellation points from.
        sync_signal : np.ndarray, optional
            A clean reference signal (e.g. PA input) used to find
            symbol boundaries.  When demodulating a PA output, pass the
            undistorted input here so that CP correlation and fine-tuning
            are performed on the clean signal rather than the distorted one.
        equalize : bool, optional
            When *True* and *sync_signal* is provided, apply per-subcarrier
            zero-forcing equalization to remove the PA's linear frequency
            response, revealing only the nonlinear distortion.

        Returns
        -------
        re, im : np.ndarray, np.ndarray
            Normalised constellation I/Q points.
        """
        raise NotImplementedError

    # -- shared utilities ----------------------------------------------------

    def _isolate_channel(self, sig, fs, f_shift, bp_half_bins):
        """Frequency-shift one carrier to DC and bandpass-filter."""
        MEM = len(sig)
        center = MEM // 2
        n = np.arange(MEM)

        if f_shift != 0:
            shifted = sig * np.exp(1j * 2 * np.pi * n / (fs / f_shift))
        else:
            shifted = sig.copy()

        fd = np.fft.fftshift(np.fft.fft(shifted))
        filt = np.zeros(MEM, dtype=complex)
        lo = max(center - bp_half_bins, 0)
        hi = min(center + bp_half_bins + 1, MEM)
        filt[lo:hi] = fd[lo:hi]
        return np.fft.ifft(np.fft.fftshift(filt))

    def _extract_subcarriers(self, ofdm_sym, nfft, n_active):
        """FFT one OFDM symbol, return ±n_active/2 bins around DC
        (excluding DC), normalised by RMS."""
        sc = self._extract_raw_subcarriers(ofdm_sym, nfft, n_active)
        rms = np.sqrt(np.mean(np.abs(sc) ** 2) + 1e-30)
        return sc / rms

    def _extract_raw_subcarriers(self, ofdm_sym, nfft, n_active):
        """FFT one OFDM symbol, return ±n_active/2 bins around DC
        (excluding DC), without normalization."""
        dc = nfft // 2
        n_half = n_active // 2
        fd = np.fft.fftshift(np.fft.fft(ofdm_sym))
        sc_neg = fd[dc - n_half:dc]
        sc_pos = fd[dc + 1:dc + n_half + 1]
        return np.concatenate([sc_neg, sc_pos])

    @staticmethod
    def _equalize(raw_pairs, n_active):
        """Per-subcarrier ZF equalization with smoothed channel estimate.

        Averages H[k] = Y[k]/X[k] across all available OFDM symbols,
        then smooths to isolate the PA's slowly-varying linear frequency
        response.  Dividing each symbol's output by H_smooth removes
        the linear response and reveals the nonlinear distortion.

        Parameters
        ----------
        raw_pairs : list of (raw_ref, raw_sig) tuples
            Raw (unnormalized) subcarrier arrays from reference and signal.
        n_active : int
            Number of active subcarriers.

        Returns
        -------
        list of np.ndarray
            Equalized, RMS-normalized subcarrier arrays.
        """
        from scipy.ndimage import uniform_filter1d

        # Mask out low-energy subcarriers (guard-band roll-off)
        ref_power = np.mean(
            np.abs(np.array([r for r, _s in raw_pairs])) ** 2, axis=0)
        mask = ref_power > 0.01 * np.median(ref_power[ref_power > 0])

        # Average channel estimate across ALL symbol pairs, then smooth
        # to isolate the slowly-varying linear PA frequency response
        H_stack = np.array([sig / (ref + 1e-30) for ref, sig in raw_pairs])
        H_avg = np.mean(H_stack, axis=0)
        window = max(1, n_active // 6)
        H_smooth = (uniform_filter1d(H_avg.real, window)
                    + 1j * uniform_filter1d(H_avg.imag, window))

        result = []
        for _raw_ref, raw_sig in raw_pairs:
            eq = (raw_sig / (H_smooth + 1e-30))[mask]
            rms = np.sqrt(np.mean(np.abs(eq) ** 2) + 1e-30)
            result.append(eq / rms)
        return result

    def _carrier_f_shifts(self):
        """Per-carrier frequency offsets (Hz)."""
        return [(i - (self.n_sub_ch - 1) / 2) * self.bw_sub_ch
                for i in range(self.n_sub_ch)]

    def _bp_half_bins(self, sig_len):
        """Bandpass filter half-width in FFT bins for one sub-channel."""
        if self.bw_sub_ch is not None:
            return int(round(self.bw_sub_ch / 2 * sig_len / self.fs))
        return int(round(sig_len * 0.02))


# ---------------------------------------------------------------------------
# OFDM with cyclic prefix  (APA datasets)
# ---------------------------------------------------------------------------

class OFDMCPDemodulator(Demodulator):
    """Standard OFDM with cyclic prefix.

    Algorithm per carrier:
    1. Frequency-shift to baseband and bandpass-filter.
    2. CP correlation to find symbol boundaries.
    3. Fine-tune FFT start offset (minimise kurtosis → best QAM clusters).
    4. Extract active subcarriers from each symbol.
    5. Mask out non-data subcarriers (pilots, guard) by power threshold.
    6. Combine all symbols across all carriers.
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.scs = spec['scs']
        self.ofdm_nfft = spec.get('ofdm_nfft', int(round(self.fs / self.scs)))
        self.n_active = spec.get('n_active', 1200)
        self.cp_first = spec.get('cp_first', 2560)
        self.cp_other = spec.get('cp_other', 2304)

    def demodulate(self, signal_complex, sync_signal=None, equalize=False):
        sig = signal_complex.ravel()
        ref = sync_signal.ravel() if sync_signal is not None else None
        bp_half = self._bp_half_bins(len(sig))
        f_shifts = self._carrier_f_shifts()
        all_sc = []

        for f_shift in f_shifts:
            clean = self._isolate_channel(sig, self.fs, f_shift, bp_half)

            # Use the clean reference for CP sync + fine-tuning when available
            if ref is not None:
                clean_ref = self._isolate_channel(ref, self.fs, f_shift, bp_half)
            else:
                clean_ref = clean

            syms = self._cp_sync(clean_ref)
            if not syms:
                continue

            # Use the first (earliest) symbol — different OFDM
            # symbols in an LTE frame carry different channels
            # (PDCCH=QPSK vs PDSCH=data QAM) which create artefacts
            # when mixed.
            cp_start = syms[0]
            off = self._fine_tune_offset(clean_ref, cp_start)
            fft_s = cp_start + self.cp_other + off
            if fft_s < 0 or fft_s + self.ofdm_nfft > len(clean):
                continue

            if equalize and ref is not None:
                raw_ref = self._extract_raw_subcarriers(
                    clean_ref[fft_s:fft_s + self.ofdm_nfft],
                    self.ofdm_nfft, self.n_active)
                raw_sig = self._extract_raw_subcarriers(
                    clean[fft_s:fft_s + self.ofdm_nfft],
                    self.ofdm_nfft, self.n_active)
                eq = self._equalize([(raw_ref, raw_sig)], self.n_active)
                all_sc.extend(eq)
            else:
                sc = self._extract_subcarriers(
                    clean[fft_s:fft_s + self.ofdm_nfft],
                    self.ofdm_nfft, self.n_active)
                all_sc.append(sc)

        if not all_sc:
            return np.array([]), np.array([])
        combined = np.concatenate(all_sc)
        return combined.real, combined.imag

    # -- internals -----------------------------------------------------------

    def _cp_sync(self, clean):
        """Find OFDM symbol boundaries via CP correlation.

        Returns list of CP-start sample indices.
        """
        if find_peaks is None:
            return self._cp_sync_manual(clean)

        nfft = self.ofdm_nfft
        cp_len = self.cp_other
        N = len(clean)
        search = N - nfft - cp_len
        if search <= 0:
            return []

        corr = np.zeros(search)
        energy = np.zeros(search)
        for n in range(search):
            s1 = clean[n:n + cp_len]
            s2 = clean[n + nfft:n + nfft + cp_len]
            corr[n] = np.abs(np.sum(s1 * np.conj(s2)))
            energy[n] = np.sqrt(
                np.sum(np.abs(s1) ** 2) * np.sum(np.abs(s2) ** 2) + 1e-30)
        corr_norm = corr / energy
        peaks, _ = find_peaks(corr_norm, height=0.5, distance=nfft // 2)
        return list(peaks)

    def _cp_sync_manual(self, clean):
        """Fallback CP sync without scipy (simple argmax)."""
        nfft = self.ofdm_nfft
        cp_len = self.cp_other
        N = len(clean)
        search = N - nfft - cp_len
        if search <= 0:
            return []

        best_corr, best_n = 0, 0
        for n in range(0, search, 100):
            s1 = clean[n:n + cp_len]
            s2 = clean[n + nfft:n + nfft + cp_len]
            c = np.abs(np.sum(s1 * np.conj(s2)))
            e = np.sqrt(
                np.sum(np.abs(s1) ** 2) * np.sum(np.abs(s2) ** 2) + 1e-30)
            if c / e > best_corr:
                best_corr = c / e
                best_n = n
        # Fine search around best
        for n in range(max(0, best_n - 200), min(search, best_n + 200)):
            s1 = clean[n:n + cp_len]
            s2 = clean[n + nfft:n + nfft + cp_len]
            c = np.abs(np.sum(s1 * np.conj(s2)))
            e = np.sqrt(
                np.sum(np.abs(s1) ** 2) * np.sum(np.abs(s2) ** 2) + 1e-30)
            if c / e > best_corr:
                best_corr = c / e
                best_n = n
        if best_corr > 0.5:
            return [best_n]
        return []

    def _fine_tune_offset(self, clean, cp_start):
        """Find the best FFT-start offset around *cp_start* by minimising
        kurtosis on *clean* (should be the undistorted reference signal)."""
        nfft = self.ofdm_nfft
        cp_len = self.cp_other
        N = len(clean)

        # Use a wider n_active for kurtosis sensitivity — outer
        # subcarriers are most affected by timing errors and make
        # the kurtosis metric sharper around the true minimum.
        n_active_tune = max(self.n_active, 1200)

        best_kurt, best_off = 999, 0
        for off in range(-30, 31):
            fft_s = cp_start + cp_len + off
            if fft_s < 0 or fft_s + nfft > N:
                continue
            sc = self._extract_subcarriers(clean[fft_s:fft_s + nfft],
                                           nfft, n_active_tune)
            kurt = self._kurtosis(sc)
            if kurt < best_kurt:
                best_kurt, best_off = kurt, off
        return best_off

    @staticmethod
    def _kurtosis(sc):
        r, im = sc.real, sc.imag
        return (np.mean(r ** 4) / (np.mean(r ** 2) ** 2 + 1e-30)
                + np.mean(im ** 4) / (np.mean(im ** 2) ** 2 + 1e-30))


# ---------------------------------------------------------------------------
# IFFT-concatenated frames without CP  (DPA datasets)
# ---------------------------------------------------------------------------

class IFFTFrameDemodulator(Demodulator):
    """IFFT-concatenated frames (no cyclic prefix).

    Since DPA signals are back-to-back IFFT frames, we chop the raw
    signal into ``nperseg``-length frames *first*, FFT each frame once,
    and read subcarrier bins directly — no bandpass filtering needed.

    Algorithm:
    1. Split the raw signal into non-overlapping frames of ``nperseg``.
    2. FFT each frame (one FFT covers all carriers).
    3. For each carrier, pick the ±n_active/2 bins around the carrier's
       centre frequency.
    4. Combine across frames and carriers.
    """

    def __init__(self, spec):
        super().__init__(spec)
        self.nperseg = spec['nperseg']
        self.n_active = spec.get('n_active') or int(round(
            self.bw_sub_ch / (self.fs / self.nperseg)))

    def demodulate(self, signal_complex, sync_signal=None, equalize=False):
        sig = signal_complex.ravel()
        ref = sync_signal.ravel() if sync_signal is not None else None
        nfft = self.nperseg
        n_active = self.n_active
        n_half = n_active // 2
        n_frames = len(sig) // nfft

        # Carrier centre bins in fftshift-ed spectrum
        bin_spacing = self.fs / nfft
        dc = nfft // 2
        carrier_centres = [dc + int(round(f / bin_spacing))
                           for f in self._carrier_f_shifts()]

        def _pick(fd, cc):
            return np.concatenate([fd[cc - n_half:cc],
                                   fd[cc + 1:cc + n_half + 1]])

        # Pre-compute per-frame FFTs
        fd_sig = [np.fft.fftshift(np.fft.fft(sig[i * nfft:(i + 1) * nfft]))
                  for i in range(n_frames)]
        fd_ref = ([np.fft.fftshift(np.fft.fft(ref[i * nfft:(i + 1) * nfft]))
                   for i in range(n_frames)]
                  if ref is not None else fd_sig)

        all_sc = []

        if equalize and ref is not None:
            for cc in carrier_centres:
                raw_pairs = [(_pick(fd_ref[i], cc), _pick(fd_sig[i], cc))
                             for i in range(n_frames)]
                if raw_pairs:
                    all_sc.extend(self._equalize(raw_pairs, n_active))
        else:
            for i in range(n_frames):
                for cc in carrier_centres:
                    sc = _pick(fd_sig[i], cc)
                    rms = np.sqrt(np.mean(np.abs(sc) ** 2) + 1e-30)
                    all_sc.append(sc / rms)

        if not all_sc:
            return np.array([]), np.array([])
        combined = np.concatenate(all_sc)
        return combined.real, combined.imag


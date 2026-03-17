#!/usr/bin/env python3
"""Tests for iterative_match.py metric functions."""

import os
import sys
import numpy as np
import pytest

# Ensure the package directory is importable
sys.path.insert(0, os.path.dirname(__file__))

import iterative_match as im

TARGET_MAT = os.path.join(
    os.path.dirname(__file__), "..",
    "Precook_Signal_100WDevice_"
    "[5cLTE20MHz_iBW100MHz_SR491p52MHz_200uS_TM3p1a_PAPR10p0_IQ].mat",
)


# -----------------------------------------------------------------------
# 1. Constants
# -----------------------------------------------------------------------

class TestConstants:
    def test_num_carriers(self):
        assert im.NUM_CARRIERS == 5

    def test_bw(self):
        assert im.BW_MHZ == 20
        assert im.BW_HZ == 20e6

    def test_sr(self):
        assert im.SR == 491.52e6

    def test_nfft(self):
        assert im.NFFT == 32768

    def test_n_active(self):
        assert im.N_ACTIVE == 1200

    def test_num_samples(self):
        assert im.NUM_SAMPLES == 98304

    def test_cp_norm(self):
        assert im.CP_NORM == int(144 * 32768 / 2048)
        assert im.CP_NORM == 2304

    def test_carrier_centers(self):
        expected = np.array([-40e6, -20e6, 0.0, 20e6, 40e6])
        np.testing.assert_allclose(im.CARRIER_CENTERS, expected)

    def test_thresholds_keys(self):
        expected_keys = {'nmse', 'psd_mae', 'evm', 'd_papr', 'ccdf_dev',
                         'ch_err'}
        assert set(im.THRESHOLDS.keys()) == expected_keys

    def test_thresholds_values(self):
        assert im.THRESHOLDS['nmse'] == -40.0
        assert im.THRESHOLDS['psd_mae'] == 0.1
        assert im.THRESHOLDS['evm'] == 1.0
        assert im.THRESHOLDS['d_papr'] == 0.1
        assert im.THRESHOLDS['ccdf_dev'] == 0.01
        assert im.THRESHOLDS['ch_err'] == 0.05


# -----------------------------------------------------------------------
# 2. load_target
# -----------------------------------------------------------------------

class TestLoadTarget:
    def test_load_target(self):
        if not os.path.isfile(TARGET_MAT):
            pytest.skip("Target .mat file not found")
        sig, sr = im.load_target(TARGET_MAT)
        assert sig.dtype == np.complex128 or np.issubdtype(sig.dtype,
                                                           np.complexfloating)
        assert sig.ndim == 1
        assert len(sig) == im.NUM_SAMPLES
        assert sr == pytest.approx(im.SR, rel=1e-3)


# -----------------------------------------------------------------------
# 3. NMSE
# -----------------------------------------------------------------------

class TestNMSE:
    def _make_signal(self, n=4096, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def test_nmse_identical_signals(self):
        sig = self._make_signal()
        nmse = im.compute_nmse(sig, sig)
        assert nmse < -100.0, f"NMSE of identical signals should be < -100 dB, got {nmse}"

    def test_nmse_known_error(self):
        sig = self._make_signal(n=8192)
        rng = np.random.default_rng(99)
        # Add noise at -20 dB relative to signal power
        noise_power = np.mean(np.abs(sig) ** 2) * 10 ** (-20.0 / 10.0)
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig))
        )
        noisy = sig + noise
        nmse = im.compute_nmse(noisy, sig)
        # Should be near -20 dB (allow 3 dB tolerance)
        assert -23.0 < nmse < -17.0, f"Expected NMSE near -20 dB, got {nmse}"

    def test_nmse_with_shift(self):
        sig = self._make_signal(n=8192)
        shifted = np.roll(sig, 5)
        nmse = im.compute_nmse(shifted, sig)
        assert nmse < -80.0, f"Alignment should recover shifted signal, got NMSE={nmse}"


# -----------------------------------------------------------------------
# 4. PSD MAE
# -----------------------------------------------------------------------

class TestPSDMAE:
    def _make_broadband(self, n=8192, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def test_psd_mae_identical(self):
        sig = self._make_broadband()
        mae = im.compute_psd_mae(sig, sig, im.SR)
        assert mae < 1e-6, f"Identical signals should have PSD MAE ~ 0, got {mae}"

    def test_psd_mae_different(self):
        sig1 = self._make_broadband(seed=0)
        sig2 = self._make_broadband(seed=1)
        mae = im.compute_psd_mae(sig1, sig2, im.SR)
        assert mae > 0, "Different signals should have PSD MAE > 0"


# -----------------------------------------------------------------------
# 5. Channel power error
# -----------------------------------------------------------------------

class TestChannelPowerError:
    def _make_broadband(self, n=8192, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def test_channel_power_error_identical(self):
        sig = self._make_broadband()
        err = im.compute_channel_power_error(sig, sig, im.SR)
        assert err < 1e-6, f"Identical signals: ch_err should be ~ 0, got {err}"


# -----------------------------------------------------------------------
# 6. PAPR
# -----------------------------------------------------------------------

class TestPAPR:
    def test_papr_constant_envelope(self):
        # Constant-envelope (CW) signal: PAPR = 0 dB
        n = 4096
        sig = np.exp(1j * np.linspace(0, 2 * np.pi * 10, n))
        papr = im.compute_papr(sig)
        assert abs(papr) < 0.01, f"CW signal PAPR should be ~0 dB, got {papr}"


# -----------------------------------------------------------------------
# 7. CCDF deviation
# -----------------------------------------------------------------------

class TestCCDFDev:
    def _make_signal(self, n=8192, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def test_ccdf_dev_identical(self):
        sig = self._make_signal()
        dev = im.compute_ccdf_dev(sig, sig)
        assert dev < 1e-10, f"Identical signals: CCDF dev should be ~ 0, got {dev}"

    def test_ccdf_dev_scaled(self):
        sig = self._make_signal()
        scaled = sig * 2.0
        # Scaling is power-normalised inside CCDF, so dev should stay ~ 0
        dev = im.compute_ccdf_dev(sig, scaled)
        assert dev < 1e-10, (
            f"Scaling should not change CCDF (power-normalised), got {dev}"
        )


# -----------------------------------------------------------------------
# 8. EVM with cache (smoke test)
# -----------------------------------------------------------------------

class TestEVMCached:
    def test_evm_cached_runs(self):
        if not os.path.isfile(TARGET_MAT):
            pytest.skip("Target .mat file not found")
        sig, sr = im.load_target(TARGET_MAT)
        evm, cache = im.compute_evm_cached(sig, sig, sr, cp_cache=None)
        assert isinstance(evm, float)
        assert evm >= 0.0
        assert cache is not None
        assert len(cache) == im.NUM_CARRIERS

        # Second call with cache should also work
        evm2, cache2 = im.compute_evm_cached(sig, sig, sr, cp_cache=cache)
        assert isinstance(evm2, float)
        assert abs(evm2 - evm) < 1e-6  # deterministic with same data


# -----------------------------------------------------------------------
# 9. compute_all_metrics (smoke test)
# -----------------------------------------------------------------------

class TestComputeAllMetrics:
    def _make_signal(self, n=8192, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n) + 1j * rng.standard_normal(n)

    def test_compute_all_metrics(self):
        sig = self._make_signal()
        metrics, cache = im.compute_all_metrics(sig, sig, im.SR)
        expected_keys = {'nmse', 'psd_mae', 'evm', 'd_papr', 'ccdf_dev',
                         'ch_err'}
        assert set(metrics.keys()) == expected_keys
        for k, v in metrics.items():
            assert isinstance(v, (int, float, np.floating)), (
                f"metrics[{k!r}] should be numeric, got {type(v)}"
            )


# -----------------------------------------------------------------------
# 10. check_convergence
# -----------------------------------------------------------------------

class TestCheckConvergence:
    def test_all_passing(self):
        metrics = {
            'nmse': -50.0,
            'psd_mae': 0.05,
            'evm': 0.5,
            'd_papr': 0.05,
            'ccdf_dev': 0.005,
            'ch_err': 0.02,
        }
        assert im.check_convergence(metrics) is True

    def test_one_failing(self):
        metrics = {
            'nmse': -30.0,   # above -40 threshold -> fail
            'psd_mae': 0.05,
            'evm': 0.5,
            'd_papr': 0.05,
            'ccdf_dev': 0.005,
            'ch_err': 0.02,
        }
        assert im.check_convergence(metrics) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

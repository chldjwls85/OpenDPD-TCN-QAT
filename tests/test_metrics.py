"""Unit tests for the signal-quality metrics in utils/metrics.py and helper
functions in utils/util.py, using synthetic signals with known properties."""

import numpy as np
import pytest
import torch
from torch import nn

from utils.metrics import ACLR, EVM, IQ_to_complex, NMSE, moving_average, power_spectrum
from utils.util import count_net_params, get_amplitude, set_target_gain

RNG = np.random.default_rng(42)
FS = 800e6
BW_MAIN = 200e6


def make_iq(shape):
    return RNG.normal(size=(*shape, 2))


def make_bandlimited_iq(n_samples, batch=2, occupied_fraction=0.9):
    """Complex noise whose spectrum is confined to the main channel."""
    freqs = np.fft.fftfreq(n_samples, d=1 / FS)
    in_band = np.abs(freqs) <= (BW_MAIN / 2) * occupied_fraction
    signals = []
    for _ in range(batch):
        spectrum = np.zeros(n_samples, dtype=complex)
        spectrum[in_band] = RNG.normal(size=in_band.sum()) + 1j * RNG.normal(
            size=in_band.sum()
        )
        signals.append(np.fft.ifft(spectrum))
    complex_signal = np.stack(signals)
    return np.stack([complex_signal.real, complex_signal.imag], axis=-1)


class TestNMSE:
    def test_error_equal_to_signal_gives_zero_db(self):
        ground_truth = make_iq((1000,))
        prediction = 2.0 * ground_truth  # error == ground truth -> NMSE = 0 dB
        assert NMSE(prediction, ground_truth) == pytest.approx(0.0, abs=1e-9)

    def test_smaller_error_gives_lower_nmse(self):
        ground_truth = make_iq((1000,))
        small = NMSE(ground_truth + 0.01 * make_iq((1000,)), ground_truth)
        large = NMSE(ground_truth + 0.5 * make_iq((1000,)), ground_truth)
        assert small < large
        assert np.isfinite(small) and np.isfinite(large)


class TestACLR:
    def test_bandlimited_signal_has_low_adjacent_leakage(self):
        iq = make_bandlimited_iq(5120)
        aclr_left, aclr_right = ACLR(iq, fs=FS, bw_main_ch=BW_MAIN, n_sub_ch=10)
        assert np.isfinite(aclr_left) and np.isfinite(aclr_right)
        assert aclr_left < -20, f"in-band signal leaked left: {aclr_left} dBc"
        assert aclr_right < -20, f"in-band signal leaked right: {aclr_right} dBc"

    def test_white_noise_has_higher_leakage_than_bandlimited(self):
        bandlimited = make_bandlimited_iq(5120)
        white = make_iq((2, 5120))
        bl_left, bl_right = ACLR(bandlimited, fs=FS, bw_main_ch=BW_MAIN, n_sub_ch=10)
        wn_left, wn_right = ACLR(white, fs=FS, bw_main_ch=BW_MAIN, n_sub_ch=10)
        assert max(bl_left, bl_right) < min(wn_left, wn_right)


class TestEVM:
    def test_noisier_prediction_has_worse_evm(self):
        ground_truth = make_bandlimited_iq(2560)
        low_noise = ground_truth + 0.001 * make_iq((2, 2560))
        high_noise = ground_truth + 0.1 * make_iq((2, 2560))
        evm_low = EVM(low_noise, ground_truth, sample_rate=int(FS), bw_main_ch=BW_MAIN)
        evm_high = EVM(high_noise, ground_truth, sample_rate=int(FS), bw_main_ch=BW_MAIN)
        assert np.isfinite(evm_low) and np.isfinite(evm_high)
        assert evm_low < evm_high


class TestHelpers:
    def test_iq_to_complex(self):
        iq = np.array([[1.0, 2.0], [3.0, -4.0]])
        result = IQ_to_complex(iq)
        np.testing.assert_allclose(result, np.array([1 + 2j, 3 - 4j]))

    def test_moving_average(self):
        result = moving_average(np.array([1.0, 2.0, 3.0, 4.0]), window_size=2)
        np.testing.assert_allclose(result, [1.5, 2.5, 3.5])

    def test_power_spectrum_shapes_match(self):
        signal = IQ_to_complex(make_iq((2, 5120)))
        freq, ps = power_spectrum(signal, fs=FS, nperseg=2560)
        assert freq.shape == ps.shape
        assert np.isfinite(ps).all()

    def test_get_amplitude(self):
        iq = np.array([[3.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose(get_amplitude(iq), [5.0, 10.0])

    def test_set_target_gain_recovers_known_gain(self):
        input_iq = make_iq((500,))
        output_iq = 3.0 * input_iq
        assert set_target_gain(input_iq, output_iq) == pytest.approx(3.0)

    def test_count_net_params(self):
        net = nn.Linear(3, 4)  # 3*4 weights + 4 biases
        assert count_net_params(net) == 16

    def test_count_net_params_matches_torch(self):
        torch.manual_seed(0)
        net = nn.GRU(input_size=2, hidden_size=8, num_layers=1)
        expected = sum(p.numel() for p in net.parameters())
        assert count_net_params(net) == expected

"""STFT waveform/window gradients and layout against independent FFT values."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
import jittor as jt


def _reference(samples, window, n_fft, hop, center=False, normalized=False,
               onesided=True):
    if center:
        samples = np.pad(samples, (n_fft // 2, n_fft // 2), mode="reflect")
    left = (n_fft - len(window)) // 2
    window = np.pad(window, (left, n_fft - len(window) - left))
    frames = np.stack([
        samples[i:i + n_fft] * window
        for i in range(0, len(samples) - n_fft + 1, hop)
    ])
    transform = np.fft.rfft if onesided else np.fft.fft
    result = transform(frames, axis=-1).T
    return result / np.sqrt(n_fft) if normalized else result


@pytest.mark.parametrize("center,normalized,onesided", [
    (False, False, True), (True, False, True),
    (True, True, True), (False, True, False),
])
def test_stft_values_and_batched_shape(center, normalized, onesided):
    with jt.runtime.scope(use_cuda=0):
        samples = np.linspace(-0.7, 1.1, 32, dtype=np.float32)
        window = np.hanning(6).astype(np.float32)
        actual = torch.stft(
            torch.tensor(np.stack([samples, samples * 2])), 8,
            hop_length=3, win_length=6, window=torch.tensor(window),
            center=center, normalized=normalized, onesided=onesided,
            return_complex=True,
        )
        expected = _reference(samples, window, 8, 3, center, normalized, onesided)
        assert str(actual.dtype) == "torch.complex64"
        np.testing.assert_allclose(
            actual.detach().cpu().numpy(), np.stack([expected, expected * 2]),
            rtol=2e-5, atol=2e-6,
        )


def test_stft_waveform_and_window_gradients():
    samples = np.linspace(-0.7, 1.1, 16, dtype=np.float32)
    window = np.linspace(0.2, 1, 6, dtype=np.float32)
    def loss(x, w):
        value = _reference(x, w, 8, 3)
        return np.sum(np.abs(value) ** 2)
    with jt.runtime.scope(use_cuda=0):
        x = torch.tensor(samples, requires_grad=True)
        w = torch.tensor(window, requires_grad=True)
        y = torch.stft(x, 8, hop_length=3, win_length=6, window=w,
                       center=False, return_complex=True)
        assert y.requires_grad, "STFT must retain the waveform/window graph"
        y.abs().square().sum().backward()
        for tensor, original, is_waveform in ((x, samples, True), (w, window, False)):
            assert tensor.grad is not None
            gradient = tensor.grad.detach().cpu().numpy()
            assert np.isfinite(gradient).all()
            expected = np.empty_like(original)
            for i in range(len(original)):
                plus, minus = original.copy(), original.copy()
                plus[i] += 1e-3
                minus[i] -= 1e-3
                upper = loss(plus, window) if is_waveform else loss(samples, plus)
                lower = loss(minus, window) if is_waveform else loss(samples, minus)
                expected[i] = (upper - lower) / 2e-3
            np.testing.assert_allclose(gradient, expected, rtol=2e-3, atol=2e-3)


def test_stft_does_not_copy_waveform_or_window_to_numpy():
    with jt.runtime.scope(use_cuda=0):
        x = torch.tensor(np.linspace(0, 1, 16, dtype=np.float32))
        window = torch.ones(8)
        with patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host read")):
            result = torch.stft(x, 8, hop_length=4, window=window,
                                center=False, return_complex=True)
            result.sync()
        assert tuple(result.shape) == (5, 3)


def test_stft_real_imag_output():
    with jt.runtime.scope(use_cuda=0):
        x = torch.tensor(np.arange(16, dtype=np.float32))
        window = torch.ones(8)
        result = torch.stft(x, 8, hop_length=4, window=window,
                            center=False, return_complex=False)
        expected = _reference(np.arange(16), np.ones(8), 8, 4)
        assert tuple(result.shape) == (5, 3, 2)
        np.testing.assert_allclose(result.numpy(), np.stack([expected.real, expected.imag], -1),
                                   rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("kwargs", [
    {"n_fft": 0}, {"n_fft": -1}, {"hop_length": 0},
    {"win_length": 9}, {"win_length": 0},
])
def test_stft_invalid_sizes_raise(kwargs):
    with jt.runtime.scope(use_cuda=0):
        options = dict(n_fft=8, hop_length=2, center=False, return_complex=True)
        options.update(kwargs)
        with pytest.raises(RuntimeError):
            torch.stft(torch.ones(16), **options)


def test_stft_unsupported_dtype_and_window_contracts_are_explicit():
    with jt.runtime.scope(use_cuda=0):
        with pytest.raises(NotImplementedError, match="float32"):
            torch.stft(torch.ones(16, dtype=torch.float64), 8, return_complex=True)
        with pytest.raises(RuntimeError, match="window"):
            torch.stft(torch.ones(16), 8, window=torch.ones(7), return_complex=True)
        with pytest.raises(RuntimeError, match="padding"):
            torch.stft(torch.ones(4), 8, return_complex=True)
        with pytest.raises(RuntimeError, match="return_complex"):
            torch.stft(torch.ones(16), 8)


def test_stft_circular_padding_rejects_multiple_wraps():
    with jt.runtime.scope(use_cuda=0):
        with pytest.raises(RuntimeError, match="wrapping around"):
            torch.stft(torch.ones(2), 8, hop_length=2, window=torch.ones(8),
                       pad_mode="circular", return_complex=True)

"""Short-time Fourier transforms built from native differentiable operations."""

import numbers

import jittor as jt
from jittor._core.dtypes import dtype_name

#: Real dtypes `stft` accepts. The shared native FFT stops at complex64, so
#: every one of them is transformed in float32 -- what the NumPy implementation
#: this replaced returned for all of them as well.
_REAL_DTYPES = ("float16", "bfloat16", "float32", "float64")


def _positive_size(name, value):
    if not isinstance(value, numbers.Integral):
        raise TypeError("stft: {} must be an integer".format(name))
    if value <= 0:
        raise RuntimeError("stft: {} must be positive".format(name))
    return int(value)


def _as_float32(value, what):
    name = dtype_name(value.dtype)
    if name not in _REAL_DTYPES:
        raise NotImplementedError("stft supports real floating {}s, got {}".format(what, name))
    return value if name == "float32" else value.float32()


def stft(
    input,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    pad_mode="reflect",
    normalized=False,
    onesided=True,
    return_complex=True,
):
    """Real STFT with complex64 output and input/window gradients.

    Frames and transforms stay in the native graph, on the input's device.
    """
    if not isinstance(input, jt.Var):
        raise TypeError("stft: input must be a tensor")
    if input.ndim not in (1, 2):
        raise RuntimeError("stft: expected a 1-D or 2-D input")
    input = _as_float32(input, "input")
    n_fft = _positive_size("n_fft", n_fft)
    hop = _positive_size("hop_length", n_fft // 4 if hop_length is None else hop_length)
    width = _positive_size("win_length", n_fft if win_length is None else win_length)
    if width > n_fft:
        raise RuntimeError("stft: win_length must not exceed n_fft")
    if window is None:
        window = jt.ones([width], dtype=input.dtype)
    elif not isinstance(window, jt.Var):
        raise TypeError("stft: window must be a tensor")
    elif window.ndim != 1 or window.shape[0] != width:
        raise RuntimeError("stft: window must be 1-D with win_length elements")
    window = _as_float32(window, "window")
    from jittor._runtime.dispatch import dispatch_context

    if dispatch_context(input)[:2] != dispatch_context(window)[:2]:
        raise RuntimeError("stft: input and window must use the same device")
    squeeze = input.ndim == 1
    samples = input.reshape((1, input.shape[0])) if squeeze else input
    if center:
        if pad_mode not in ("reflect", "constant", "replicate", "circular"):
            raise NotImplementedError("stft: unsupported pad_mode {}".format(pad_mode))
        padding = n_fft // 2
        if pad_mode == "reflect" and padding >= samples.shape[-1]:
            raise RuntimeError("stft: reflection padding must be less than input length")
        if pad_mode == "circular" and padding > samples.shape[-1]:
            raise RuntimeError("Padding value causes wrapping around more than once.")
        samples = jt.nn.pad(samples, (padding, padding), mode=pad_mode)
    if samples.shape[-1] < n_fft:
        raise RuntimeError("stft: n_fft must not exceed the padded input length")
    if width != n_fft:
        window = window.reindex([n_fft], ["i0-{}".format((n_fft - width) // 2)])
    count = 1 + (samples.shape[-1] - n_fft) // hop
    frames = samples.reindex([samples.shape[0], count, n_fft], ["i0", "i1*{}+i2".format(hop)])
    frames = frames * window.reshape((1, 1, n_fft))
    from . import fft, rfft

    transform = rfft if onesided else fft
    result = transform(frames, dim=-1, norm="ortho" if normalized else None)
    result = result.permute(0, 2, 1)
    if squeeze:
        result = result[0]
    return result if return_complex else jt.nn.view_as_real(result)

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


def istft(
    input,
    n_fft,
    hop_length=None,
    win_length=None,
    window=None,
    center=True,
    normalized=False,
    onesided=None,
    length=None,
    return_complex=False,
):
    """Inverse real STFT built from native differentiable operations.

    Stays in the native graph on the input's device with no host sync so it
    can be captured in CUDA graphs.
    """
    if not isinstance(input, jt.Var):
        raise TypeError("istft: input must be a tensor")
    name = dtype_name(input.dtype)
    is_complex = name in ("complex64", "complex128")
    if is_complex:
        if input.ndim not in (2, 3):
            raise RuntimeError("istft: expected a 2-D or 3-D complex input")
        spec = input if name == "complex64" else input.cast("complex64")
    else:
        if input.ndim not in (3, 4) or input.shape[-1] != 2:
            raise RuntimeError("istft: expected a 3-D or 4-D real input with last dimension 2")
        spec = jt.nn.view_as_complex(_as_float32(input, "input"))

    n_fft = _positive_size("n_fft", n_fft)
    hop = _positive_size("hop_length", n_fft // 4 if hop_length is None else hop_length)
    width = _positive_size("win_length", n_fft if win_length is None else win_length)
    if width > n_fft:
        raise RuntimeError("istft: win_length must not exceed n_fft")
    if length is not None:
        if not isinstance(length, numbers.Integral) or length < 0:
            raise RuntimeError("istft: length must be a non-negative integer")
        length = int(length)

    squeeze = spec.ndim == 2
    if squeeze:
        spec = spec.reshape((1, spec.shape[0], spec.shape[1]))
    batch, n_freq, n_frames = spec.shape
    if onesided is None:
        onesided = n_freq != n_fft
    expected_freq = n_fft // 2 + 1 if onesided else n_fft
    if n_freq != expected_freq:
        raise RuntimeError(
            "istft: expected {} frequency bins for n_fft={} (onesided={}), got {}".format(
                expected_freq, n_fft, onesided, n_freq
            )
        )

    if window is None:
        window = jt.ones([width], dtype="float32")
    elif not isinstance(window, jt.Var):
        raise TypeError("istft: window must be a tensor")
    elif window.ndim != 1 or window.shape[0] != width:
        raise RuntimeError("istft: window must be 1-D with win_length elements")
    window = _as_float32(window, "window")
    from jittor._runtime.dispatch import dispatch_context

    if dispatch_context(spec)[:2] != dispatch_context(window)[:2]:
        raise RuntimeError("istft: input and window must use the same device")
    if width != n_fft:
        window = window.reindex([n_fft], ["i0-{}".format((n_fft - width) // 2)])

    from . import ifft, irfft

    norm = "ortho" if normalized else None
    spec_t = spec.permute(0, 2, 1)
    if onesided:
        if return_complex:
            raise RuntimeError("istft: return_complex=True requires onesided=False")
        frames = irfft(spec_t, n=n_fft, dim=-1, norm=norm)
    else:
        frames = ifft(spec_t, n=n_fft, dim=-1, norm=norm)
        if not return_complex:
            frames = frames.real

    frames = frames * window.reshape((1, 1, n_fft))
    expected_len = n_fft + hop * (n_frames - 1)
    if return_complex:
        real_ola = frames.real.reindex_reduce(
            "add", [batch, expected_len], ["i0", "i1*{}+i2".format(hop)]
        )
        imag_ola = frames.imag.reindex_reduce(
            "add", [batch, expected_len], ["i0", "i1*{}+i2".format(hop)]
        )
        output = jt.nn.view_as_complex(jt.stack([real_ola, imag_ola], dim=-1))
    else:
        output = frames.reindex_reduce(
            "add", [batch, expected_len], ["i0", "i1*{}+i2".format(hop)]
        )

    win_sq = (window * window).reshape((1, n_fft)).broadcast([n_frames, n_fft])
    envelope = win_sq.reindex_reduce("add", [expected_len], ["i0*{}+i1".format(hop)])

    if center:
        start = n_fft // 2
        end = expected_len - n_fft // 2 if length is None else start + length
    else:
        start = 0
        end = expected_len if length is None else length

    slice_end = min(end, expected_len)
    if slice_end <= start:
        raise RuntimeError("istft: output length is non-positive")
    env_slice = envelope[start:slice_end]
    out_slice = output[:, start:slice_end] / env_slice.reshape((1, slice_end - start))
    if end > expected_len:
        pad_right = end - expected_len
        out_slice = jt.nn.pad(out_slice, (0, pad_right))
    if squeeze:
        out_slice = out_slice[0]
    return out_slice

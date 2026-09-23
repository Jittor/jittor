"""Torch numerical signal operations."""

def hann_window(window_length, periodic=True, *, dtype=None, device=None,
                requires_grad=False, **kwargs):
    """Create a Hann window through the CPU NumPy signal owner."""
    from . import (
        jt,
        np,
    )
    from ...tensor_state import compatibility_owner
    from ...types import _dtype_to_str
    owner = compatibility_owner(jt)
    selected = _dtype_to_str(dtype if dtype is not None else owner.get_default_dtype())
    if selected not in ("float16", "bfloat16", "float32", "float64"):
        raise RuntimeError("hann_window requires a floating point dtype")
    length = int(window_length)
    if length <= 1:
        window = np.ones(max(length, 0), np.float64)
    else:
        denominator = length if periodic else (length - 1)
        index = np.arange(length, dtype=np.float64)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * index / denominator)
    return owner.tensor(window, dtype=selected, device=device,
                        requires_grad=requires_grad)


def kaiser_window(window_length, periodic=True, beta=12.0, *, dtype=None,
                  device=None, requires_grad=False, **kwargs):
    """Create a Kaiser window through the CPU NumPy signal owner."""
    from . import (
        jt,
        np,
    )
    from ...tensor_state import compatibility_owner
    from ...types import _dtype_to_str
    owner = compatibility_owner(jt)
    selected = _dtype_to_str(dtype if dtype is not None else owner.get_default_dtype())
    if selected not in ("float16", "bfloat16", "float32", "float64"):
        raise RuntimeError("kaiser_window requires a floating point dtype")
    length = int(window_length)
    if length <= 1:
        window = np.ones(max(length, 0), np.float64)
    else:
        # Torch's periodic window is the symmetric window of `length + 1` points
        # with the duplicated last sample trimmed, the same convention
        # `hann_window` above follows.
        window = np.kaiser(length + 1 if periodic else length, float(beta))[:length]
    return owner.tensor(window, dtype=selected, device=device,
                        requires_grad=requires_grad)


def stft(input, n_fft, hop_length=None, win_length=None, window=None,
         center=True, pad_mode="reflect", normalized=False, onesided=None,
         return_complex=None):
    """Adapt Torch's STFT protocol to the shared native signal operation."""
    from jittor.fft import stft as native_stft
    if return_complex is None:
        raise RuntimeError(
            "stft requires the return_complex parameter for real inputs")
    return native_stft(
        input, n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=center, pad_mode=pad_mode,
        normalized=normalized, onesided=True if onesided is None else onesided,
        return_complex=return_complex,
    )

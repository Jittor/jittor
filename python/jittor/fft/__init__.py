"""Native FFT operations shared by Jittor and the Torch compatibility layer.

The implementation uses differentiable real/imag DFT matrix operations so it
works on every backend supported by the surrounding native complex bridge.  A
Torch-mode process publishes this same module object as ``torch.fft``; it does
not install a second FFT implementation.
"""

import numpy as np
from collections import OrderedDict

import jittor as jt


_ComplexNumber = jt.nn.ComplexNumber
_dft_mat_cache = OrderedDict()
_dft_mat_cache_limit = 16


def _dft_mats(size, inverse):
    key = (int(size), bool(inverse), int(jt.flags.use_acl), int(jt.flags.use_cuda))
    cached = _dft_mat_cache.get(key)
    if cached is not None:
        _dft_mat_cache.move_to_end(key)
        return cached
    indices = np.arange(size)
    angle = (2.0 * np.pi / size) * np.outer(indices, indices)
    if not inverse:
        angle = -angle
    matrices = (
        jt.array(np.cos(angle).astype("float32")),
        jt.array(np.sin(angle).astype("float32")),
    )
    _dft_mat_cache[key] = matrices
    if len(_dft_mat_cache) > _dft_mat_cache_limit:
        _dft_mat_cache.popitem(last=False)
    return matrices


def _to_last(value, dim):
    ndim = value.real.ndim if isinstance(value, _ComplexNumber) else value.ndim
    dimension = dim if dim >= 0 else dim + ndim
    if dimension == ndim - 1:
        return value, None
    permutation = [axis for axis in range(ndim) if axis != dimension] + [dimension]
    inverse = [0] * ndim
    for new_axis, old_axis in enumerate(permutation):
        inverse[old_axis] = new_axis
    return value.permute(*permutation), inverse


def _resize_last(value, size):
    if size is None:
        return value
    length = value.shape[-1]
    if length == size:
        return value
    if length > size:
        return value[..., :size]
    padding = jt.zeros(list(value.shape[:-1]) + [size - length], value.dtype)
    return jt.concat([value, padding], dim=-1)


def _make_complex(real, imag):
    return jt.nn.view_as_complex(jt.stack([real, imag], dim=-1))


def _real_imag(value):
    if isinstance(value, _ComplexNumber):
        return value.real, value.imag
    if isinstance(value, jt.Var) and "complex" in str(value.dtype):
        return value.real, value.imag
    return value, None


def _fft_core(value, size, dim, inverse, norm=None):
    value, restore = _to_last(value, dim)
    real0, imag0 = _real_imag(value)
    real = _resize_last(real0, size)
    imag = _resize_last(imag0, size) if imag0 is not None else None
    length = real.shape[-1]
    cosine, sine = _dft_mats(length, inverse)

    out_real = jt.matmul(real, cosine.transpose(1, 0))
    out_imag = jt.matmul(real, sine.transpose(1, 0))
    if imag is not None:
        out_real = out_real - jt.matmul(imag, sine.transpose(1, 0))
        out_imag = out_imag + jt.matmul(imag, cosine.transpose(1, 0))

    if norm == "ortho":
        scale = 1.0 / (length ** 0.5)
    elif norm == "forward":
        scale = (1.0 / length) if not inverse else 1.0
    else:
        scale = (1.0 / length) if inverse else 1.0
    if scale != 1.0:
        out_real = out_real * scale
        out_imag = out_imag * scale

    output = _make_complex(out_real, out_imag)
    if restore is not None:
        output = output.permute(*restore)
    return output


def fft(input, n=None, dim=-1, norm=None):
    return _fft_core(input, n, dim, False, norm)


def ifft(input, n=None, dim=-1, norm=None):
    return _fft_core(input, n, dim, True, norm)


def _fftn(input, sizes=None, dims=(-2, -1), norm=None, inverse=False):
    output = input
    dimensions = list(dims)
    lengths = list(sizes) if sizes is not None else [None] * len(dimensions)
    for dim, size in zip(dimensions, lengths):
        output = _fft_core(output, size, dim, inverse, norm)
    return output


def fft2(input, s=None, dim=(-2, -1), norm=None):
    return _fftn(input, s, dim, norm, False)


def ifft2(input, s=None, dim=(-2, -1), norm=None):
    return _fftn(input, s, dim, norm, True)


def fftn(input, s=None, dim=(-2, -1), norm=None):
    return _fftn(input, s, dim, norm, False)


def ifftn(input, s=None, dim=(-2, -1), norm=None):
    return _fftn(input, s, dim, norm, True)


def rfft(input, n=None, dim=-1, norm=None):
    full = _fft_core(input, n, dim, False, norm)
    length = input.shape[dim] if n is None else n
    keep = length // 2 + 1
    real, imag = full.real, full.imag
    slices = [slice(None)] * real.ndim
    slices[dim if dim >= 0 else dim + real.ndim] = slice(0, keep)
    selection = tuple(slices)
    return _make_complex(real[selection], imag[selection])


def _resize_at(value, dim, size):
    """Truncate or zero-pad ``value`` along ``dim`` to exactly ``size``."""
    length = value.shape[dim]
    if length == size:
        return value
    slices = [slice(None)] * value.ndim
    if length > size:
        slices[dim] = slice(0, size)
        return value[tuple(slices)]
    padding = list(value.shape)
    padding[dim] = size - length
    return jt.concat([value, jt.zeros(padding, value.dtype)], dim=dim)


def irfft(input, n=None, dim=-1, norm=None):
    # ``_real_imag`` is the accessor that tells a real input apart from a
    # complex one; ``input.real``/``input.imag`` cannot, because a real Var
    # answers both (returning itself and a zero Var).
    real, imag = _real_imag(input)
    dimension = dim if dim >= 0 else dim + real.ndim
    half = real.shape[dimension]
    length = (2 * (half - 1)) if n is None else int(n)
    if length < 1:
        raise RuntimeError(
            f"irfft: invalid number of data points ({length}) specified")
    # The half spectrum that describes a signal of `length` samples has exactly
    # length//2 + 1 entries, so resize the input to that before mirroring it.
    # Resizing the *mirrored* spectrum instead (which is what passing `length`
    # to _fft_core did) truncates or pads in the middle of the conjugate pairs.
    keep = length // 2 + 1
    real = _resize_at(real, dimension, keep)
    imag = (jt.zeros(real.shape, real.dtype) if imag is None
            else _resize_at(imag, dimension, keep))
    # Bin `length - k` is the conjugate of bin `k`; for even `length` the
    # Nyquist bin is its own mirror and must not be repeated.
    first_mirror = keep - 2 if length % 2 == 0 else keep - 1
    mirror_indices = list(range(first_mirror, 0, -1))
    if mirror_indices:
        slices = [slice(None)] * real.ndim
        slices[dimension] = mirror_indices
        mirrored = tuple(slices)
        real = jt.concat([real, real[mirrored]], dim=dimension)
        imag = jt.concat([imag, -imag[mirrored]], dim=dimension)
    return _fft_core(_make_complex(real, imag), None, dim, True, norm).real


def _shift_dims(value, dim, inverse):
    dims = list(range(value.ndim)) if dim is None else (
        [dim] if isinstance(dim, int) else list(dim)
    )
    shifts = [
        (-(int(value.shape[d]) // 2) if inverse else int(value.shape[d]) // 2)
        for d in dims
    ]
    return jt.roll(value, shifts, dims)


def fftshift(input, dim=None):
    if isinstance(input, _ComplexNumber):
        return _ComplexNumber(
            _shift_dims(input.real, dim, False),
            _shift_dims(input.imag, dim, False),
        )
    return _shift_dims(input, dim, False)


def ifftshift(input, dim=None):
    if isinstance(input, _ComplexNumber):
        return _ComplexNumber(
            _shift_dims(input.real, dim, True),
            _shift_dims(input.imag, dim, True),
        )
    return _shift_dims(input, dim, True)


def fftfreq(n, d=1.0, **kwargs):
    return jt.array(np.fft.fftfreq(n, d).astype("float32"))


def rfftfreq(n, d=1.0, **kwargs):
    return jt.array(np.fft.rfftfreq(n, d).astype("float32"))


__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "fftshift",
    "ifftshift",
    "fftfreq",
    "rfftfreq",
]

"""Native FFT operations shared by Jittor and the Torch compatibility layer.

The implementation uses differentiable real/imag DFT matrix operations so it
works on every backend supported by the surrounding native complex bridge.  A
Torch-mode process publishes this same module object as ``torch.fft``; it does
not install a second FFT implementation.
"""

import numpy as np

import jittor as jt


_ComplexNumber = jt.nn.ComplexNumber


def _dft_mats(size, inverse):
    indices = np.arange(size)
    angle = (2.0 * np.pi / size) * np.outer(indices, indices)
    if not inverse:
        angle = -angle
    return (
        jt.array(np.cos(angle).astype("float32")),
        jt.array(np.sin(angle).astype("float32")),
    )


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


def irfft(input, n=None, dim=-1, norm=None):
    dimension = dim if dim >= 0 else dim + input.real.ndim
    half = input.real.shape[dimension]
    length = (2 * (half - 1)) if n is None else n
    real, imag = input.real, input.imag
    mirror_indices = list(range(half - 2, 0, -1))
    if mirror_indices:
        slices = [slice(None)] * real.ndim
        slices[dimension] = mirror_indices
        mirrored = tuple(slices)
        real = jt.concat([real, real[mirrored]], dim=dimension)
        imag = jt.concat([imag, -imag[mirrored]], dim=dimension)
    return _fft_core(_make_complex(real, imag), length, dim, True, norm).real


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

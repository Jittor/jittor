"""Legacy real/imag-pair complex helper used by internal kernels."""

import jittor as jt


# TODO: support FFT2D only now.
def _fft2(x, inverse=False):
    assert jt.flags.use_cuda == 1
    assert len(x.shape) == 4
    assert x.shape[3] == 2
    y = jt.compile_extern.cufft_ops.cufft_fft(x, inverse)
    if inverse:
        y /= x.shape[1] * x.shape[2]
    return y


class ComplexNumber:
    """Complex number helper (real/imag float pair).

        .. deprecated::
            Prefer the native ``complex64`` dtype. ``jt.array(complex_ndarray)``,
            ``torch.complex(re, im)``, ``torch.view_as_complex``, ``torch.polar`` and the
            ``torch.fft.*`` / ``jt.linalg`` complex paths now all produce and consume the native
            complex64 Var directly. ComplexNumber is kept only as an INTERNAL bridge substrate
            for the complex linalg kernels and is no longer returned by any public jittor / torch
            API.

        It's saved as jt.stack(real, imag, dim=-1)

        You can construct ComplexNumber with real part and imaginary part like ComplexNumber(real, imag)
        or real part only with ComplexNumber(real)
        or value after jt.stack with ComplexNumber(value, is_concat_value=True)

        add, sub, mul and truediv between ComplexNumber and ComplexNumber, jt.Var, int, float are implemented

        You can use 'shape', 'reshape' etc. as jt.Var

    Example:
        >>> real = jt.array([[[1., -2., 3.]]])
        >>> imag = jt.array([[[0., 1., 6.]]])
        >>> a = ComplexNumber(real, imag)
        >>> a + a
        >>> a / a
        >>> a.norm()                # sqrt(real^2+imag^2)
        >>> a.exp()                 # e^real(cos(imag)+isin(imag))
        >>> a.conj()                # ComplexNumber(real, -imag)
        >>> a.fft2()                # cuda only now. len(real.shape) equals 3
        >>> a.ifft2()               # cuda only now. len(real.shape) equals 3

        >>> a = jt.array([[1,1],[1,-1]])
        >>> b = jt.array([[0,-1],[1,0]])
        >>> c = ComplexNumber(a,b) / jt.sqrt(3)
        >>> c @ c.transpose().conj()
        ComplexNumber(real=jt.Var([[0.99999994 0.        ]
                [0.         0.99999994]], dtype=float32), imag=jt.Var([[0. 0.]
                [0. 0.]], dtype=float32))
    """

    def __init__(self, real: jt.Var, imag: jt.Var = None, is_concat_value=False):
        if is_concat_value:
            assert real.shape[-1] == 2
            assert imag is None
            self.value = real
        elif imag is None:
            self.value = jt.stack([real, jt.zeros_like(real)], dim=-1)
        else:
            assert real.shape == imag.shape
            assert real.dtype == imag.dtype
            self.value = jt.stack([real, imag], dim=-1)

    @property
    def requires_grad(self):
        return self.value.requires_grad

    @property
    def real(self):
        return self.value[..., 0]

    @property
    def imag(self):
        return self.value[..., 1]

    @property
    def shape(self):
        return self.value.shape[:-1]

    @property
    def dtype(self):
        return "complex64"

    def norm(self):
        return jt.sqrt(jt.sqr(self.real) + jt.sqr(self.imag))

    def stop_grad(self):
        return ComplexNumber(self.value.stop_grad(), is_concat_value=True)

    def start_grad(self):
        return ComplexNumber(self.value.start_grad(), is_concat_value=True)

    def detach(self):
        return ComplexNumber(self.value.detach(), is_concat_value=True)

    def unsqueeze(self, dim=0):
        return ComplexNumber(jt.unsqueeze(self.real, dim=dim), jt.unsqueeze(self.imag, dim=dim))

    def squeeze(self, dim=0):
        return ComplexNumber(jt.squeeze(self.real, dim=dim), jt.squeeze(self.imag, dim=dim))

    def reshape(self, shape):
        return ComplexNumber(jt.reshape(self.real, shape), jt.reshape(self.imag, shape))

    def permute(self, *axes):
        return ComplexNumber(jt.permute(self.real, *axes), jt.permute(self.imag, *axes))

    def transpose(self, *axes):
        return ComplexNumber(jt.transpose(self.real, *axes), jt.transpose(self.imag, *axes))

    def broadcast(self, shape, dims):
        return ComplexNumber(self.real.broadcast(shape, dims), self.imag.broadcast(shape, dims))

    def sum(self, dims, keepdims: bool = False):
        return ComplexNumber(
            self.real.sum(dims, keepdims=keepdims), self.imag.sum(dims, keepdims=keepdims)
        )

    def exp(self):
        er = jt.exp(self.real)
        return ComplexNumber(er * jt.cos(self.imag), er * jt.sin(self.imag))

    def conj(self):
        return ComplexNumber(self.real, -self.imag)

    def abs(self):
        # magnitude |a+bi| = sqrt(a^2+b^2)  (torch.abs of a complex tensor)
        return self.norm()

    def __abs__(self):
        return self.norm()

    def angle(self):
        # phase atan2(imag, real)  (torch.angle)
        return jt.atan2(self.imag, self.real)

    def __getitem__(self, idx):
        return ComplexNumber(self.real[idx], self.imag[idx])

    def __neg__(self):
        return ComplexNumber(-self.real, -self.imag)

    def __add__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real + other.real, self.imag + other.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real + other, self.imag)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real + self.real, other.imag + self.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(other + self.real, self.imag)
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(self.real - other.real, self.imag - other.imag)
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.real - other, self.imag)
        else:
            raise NotImplementedError

    def __rsub__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(other.real - self.real, other.imag - self.imag)
        elif isinstance(other, (int, float)):
            # s - (a+bi) == (s-a) - bi: the imaginary part is negated too.
            return ComplexNumber(other - self.real, -self.imag)
        else:
            raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(
                self.real * other.real - self.imag * other.imag,
                self.real * other.imag + self.imag * other.real,
            )
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.value * other, is_concat_value=True)
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(
                other.real * self.real - other.imag * self.imag,
                other.imag * self.real + other.real * self.imag,
            )
        elif isinstance(other, (int, float)):
            return ComplexNumber(other * self.value, is_concat_value=True)
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, ComplexNumber):
            norm = jt.sqr(other.real) + jt.sqr(other.imag)
            return ComplexNumber(
                (self.real * other.real + self.imag * other.imag) / norm,
                (self.imag * other.real - self.real * other.imag) / norm,
            )
        elif isinstance(other, (int, float)):
            return ComplexNumber(self.value / other, is_concat_value=True)
        else:
            raise NotImplementedError

    def __rtruediv__(self, other):
        norm = jt.sqr(self.real) + jt.sqr(self.imag)
        if isinstance(other, ComplexNumber):
            return ComplexNumber(
                (other.real * self.real + other.imag * self.imag) / norm,
                (other.imag * self.real - other.real * self.imag) / norm,
            )
        elif isinstance(other, (int, float)):
            return ComplexNumber(other * self.real / norm, -other * self.imag / norm)
        else:
            raise NotImplementedError

    def __matmul__(self, other):
        if isinstance(other, ComplexNumber):
            return ComplexNumber(
                self.real @ other.real - self.imag @ other.imag,
                self.real @ other.imag + self.imag @ other.real,
            )
        else:
            raise NotImplementedError

    def __imatmul__(self, other):
        # `z @= w` is `z @ w`; matmul does not commute, so the operands must
        # keep the order the out-of-place operator uses.
        if isinstance(other, ComplexNumber):
            return ComplexNumber(
                self.real @ other.real - self.imag @ other.imag,
                self.real @ other.imag + self.imag @ other.real,
            )
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return f"ComplexNumber(real={self.real.__repr__()}, imag={self.imag.__repr__()})"

    def fft2(self):
        return ComplexNumber(_fft2(self.value, inverse=False), is_concat_value=True)

    def ifft2(self):
        return ComplexNumber(_fft2(self.value, inverse=True), is_concat_value=True)


__all__ = ["ComplexNumber"]

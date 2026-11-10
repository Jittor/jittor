"""Matrix functions: exponential and eigenvalues-only queries."""
import math

# exp(A) = (exp(A / 2^s))^(2^s). After scaling the 1-norm to at most 1/2, the
# Taylor series truncated after degree 18 leaves a remainder below
# 0.5^19 / 19! ~ 2e-23 relative to the leading term: under float64 epsilon.
_TAYLOR_DEGREE = 18
_SCALED_NORM = 0.5


def matrix_exp(x):
    r"""Matrix exponential of a square matrix or a batch of them (``(..., n, n)``).

    Differentiable: built from matmul, scaling and additions only, so gradients
    (including complex Wirtinger gradients) flow through the native autograd.
    The scaling exponent is chosen from the largest 1-norm in the batch and is
    a constant of the graph.
    """
    import jittor as jt
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise RuntimeError("matrix_exp: expected a tensor of shape (..., n, n), got %s" % (tuple(x.shape),))
    n = x.shape[-1]
    if n == 0:
        return x.clone()
    with jt.no_grad():
        norm = float(x.detach().abs().sum(dim=-2).max().item())
    squarings = max(0, int(math.ceil(math.log2(norm / _SCALED_NORM)))) if norm > _SCALED_NORM else 0
    scaled = x / (2.0 ** squarings) if squarings else x
    # A constant: the result must require grad only when ``x`` does.
    eye = jt.init.eye(n).cast(x.dtype).stop_grad()
    if x.ndim > 2:
        eye = eye.broadcast(x.shape)
    result = eye
    term = eye
    for k in range(1, _TAYLOR_DEGREE + 1):
        term = jt.matmul(term, scaled) / k
        result = result + term
    for _ in range(squarings):
        result = jt.matmul(result, result)
    return result


def eigvals(x):
    """Eigenvalues of a square matrix (or batch) as a native complex Var.

    A real input is cast to the native complex dtype of its precision first
    (float64 -> complex128): :func:`eig` would otherwise wrap it in a legacy
    complex64 ``ComplexNumber``. Differentiable with torch's eigenvalue
    backward ``gA = V^{-H} diag(gw) V^H`` (real part for a real input); the
    eigenvectors' own backward is not implemented by :func:`eig`. Device
    support is :func:`eig`'s: its CUDA path has no general ``eig`` yet.
    """
    import jittor as jt
    from ..nn import ComplexNumber
    from .decompositions import eig
    from ._helpers import _cn_to_native, _is_native_complex
    from .solving import inv
    if isinstance(x, ComplexNumber):
        return eig(x)[0]
    real_input = not _is_native_complex(x)
    if real_input:
        x = x.cast("complex128" if jt.core._checked_dtype_name(x.dtype) == "float64" else "complex64")

    class _Eigvals(jt.Function):
        def execute(self, a):
            w, v = eig(a)
            if isinstance(w, ComplexNumber):
                w, v = _cn_to_native(w), _cn_to_native(v)
            self.v = v.detach()
            return w

        def grad(self, gw):
            v = self.v
            vh = v.transpose(-1, -2).conj()
            ga = jt.matmul(inv(v).transpose(-1, -2).conj(), gw.unsqueeze(-1) * vh)
            return ga

    w = _Eigvals()(x)
    return w

"""Torch numerical linalg operations."""

def eye(n, m=None, dtype=None, **kwargs):
    """Create a square or rectangular identity matrix."""
    from . import (
        _dtype_to_str,
    )
    shape = (int(n), int(n)) if m is None else (int(n), int(m))
    import jittor as jt
    import jittor.init as _init
    from ...frontend import tensor_frontend
    from ...tensor_state import compatibility_owner
    target = compatibility_owner(jt)
    with tensor_frontend(target.Var):
        result = _init.eye(shape, _dtype_to_str(dtype) or "float32")
        if target is not jt:
            result.requires_grad = False
        return result


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    """Compute p-norm distances between corresponding rows of two tensors."""
    from . import (
        nn,
    )
    return nn.pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Compute cosine similarity along a tensor dimension."""
    from . import (
        nn,
    )
    return nn.cosine_similarity(x1, x2, dim=dim, eps=eps)


def svd(x, some=True, compute_uv=True, **kwargs):
    """Compute a singular value decomposition via Jittor's native linalg owner."""
    from . import (
        jt,
    )
    return jt.linalg.svd(x)


def svd_lowrank(A, q=6, niter=2, M=None):
    """Return a rank-``q`` approximation using Jittor's native SVD."""
    from . import (
        jt,
    )
    if M is not None:
        A = A - M
    u, s, v = jt.linalg.svd(A)
    q = min(q, s.shape[0])
    return u[:, :q], s[:q], v[:, :q]


def pca_lowrank(A, q=6, center=True, niter=2):
    """Compute a low-rank decomposition after optional feature centering."""
    from . import (
        svd_lowrank,
    )
    centered = A - (A.mean(0, keepdims=True) if center else 0)
    return svd_lowrank(centered, q=q, niter=niter)


def det(input):
    """Compute a matrix determinant via Jittor's native linalg owner."""
    from . import (
        jt,
    )
    return jt.linalg.det(input)


def inverse(input):
    """Compute a matrix inverse via Jittor's native linalg owner."""
    from . import (
        jt,
    )
    return jt.linalg.inv(input)


def _trace_impl(input):
    from . import (
        jt,
    )
    size = min(int(input.shape[0]), int(input.shape[1]))
    diagonal = jt.arange(size)
    return input[diagonal, diagonal].sum()


def trace(input):
    """Return the sum of a tensor's main matrix diagonal."""
    from . import (
        _trace_impl,
    )
    return _trace_impl(input)


def _diag_embed_impl(input, offset=0, dim1=-2, dim2=-1):
    from . import (
        jt,
    )
    size = int(input.shape[-1])
    return input.unsqueeze(-1) * jt.init.eye(size)


def diag_embed(input, offset=0, dim1=-2, dim2=-1):
    """Embed the final dimension of a tensor along a matrix diagonal."""
    from . import (
        _diag_embed_impl,
    )
    return _diag_embed_impl(input, offset=offset, dim1=dim1, dim2=dim2)


def _diagflat_impl(input, offset=0):
    from . import (
        _diag_embed_impl,
    )
    return _diag_embed_impl(input.reshape((-1,)), offset=offset)


def diagflat(input, offset=0):
    """Flatten an input and embed it along a matrix diagonal."""
    from . import (
        _diagflat_impl,
    )
    return _diagflat_impl(input, offset=offset)


def _mv_impl(input, vec, out=None):
    from . import (
        jt,
    )
    if input.ndim != 2 or vec.ndim != 1:
        raise RuntimeError(
            "mv: expected a 2-D matrix and a 1-D vector, got "
            f"{input.ndim}-D and {vec.ndim}-D tensors")
    if input.shape[1] != vec.shape[0]:
        raise RuntimeError(
            "mv: size mismatch, matrix has %s columns but vector has %s elements"
            % (input.shape[1], vec.shape[0]))
    result = jt.matmul(input, vec)
    if out is not None:
        out.assign(result)
        return out
    return result


def mv(input, vec, out=None):
    """Multiply a matrix by a vector, optionally writing into ``out``."""
    from . import (
        _mv_impl,
    )
    return _mv_impl(input, vec, out=out)


def _addmm_impl(input, mat1, mat2, *, beta=1, alpha=1):
    from . import (
        jt,
    )
    result = jt.matmul(mat1, mat2)
    if alpha != 1:
        result = result * alpha
    if beta == 0:
        return result
    return beta * input + result


def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    """Compute ``beta * input + alpha * (mat1 @ mat2)``."""
    from . import (
        _addmm_impl,
    )
    return _addmm_impl(input, mat1, mat2, beta=beta, alpha=alpha)


def _mm_impl(input, mat2, out=None):
    # Keep the existing compatibility boundary: ``out`` is accepted for API
    # shape compatibility but is not populated by this approximate fallback.
    from . import (
        jt,
    )
    return jt.matmul(input, mat2)


def mm(input, mat2, out=None):
    """Multiply two 2-D tensors using Jittor's matrix multiplication."""
    from . import (
        _mm_impl,
    )
    return _mm_impl(input, mat2, out=out)


def kron(a, b):
    """Compute the Kronecker product through broadcasted Jittor views."""
    nd = max(a.ndim, b.ndim)
    a2 = a.reshape((1,) * (nd - a.ndim) + tuple(a.shape))
    b2 = b.reshape((1,) * (nd - b.ndim) + tuple(b.shape))
    aex, bex, fin = [], [], []
    for i in range(nd):
        aex += [int(a2.shape[i]), 1]
        bex += [1, int(b2.shape[i])]
        fin.append(int(a2.shape[i]) * int(b2.shape[i]))
    return (a2.reshape(aex) * b2.reshape(bex)).reshape(fin)

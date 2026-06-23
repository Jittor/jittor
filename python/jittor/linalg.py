# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from functools import partial
from .nn import ComplexNumber

def complex_inv(x:ComplexNumber):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).

    TODO: Faster Implementation; Check backward.
    """
    assert isinstance(x, ComplexNumber), "complex_inv is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for complex_inv"

    def forward_code(np, data):
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)

        a = _stack_to_complex(data["inputs"][0])
        m_a = data["outputs"][0]
        t_a = np.linalg.inv(a)
        np.copyto(m_a, _complex_to_stack(t_a))


    def backward_code(np, data):
        def T(x):
            return np.conj(np.swapaxes(x, -1, -2))
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = _stack_to_complex(data["dout"])
        out = data["outputs"][0]
        mx = _stack_to_complex(data["f_outputs"][0])
        t = -_dot(_dot(T(mx), dout), T(mx))
        np.copyto(out, _complex_to_stack(t))

    lmx = jt.numpy_code(
        x.value.shape,
        x.value.dtype,
        [x.value],
        forward_code,
        [backward_code],
    )

    return ComplexNumber(lmx, is_concat_value=True)

def complex_eig(x:ComplexNumber):
    r"""
    calculate the eigenvalues and eigenvectors of x.
    :param x (...,M,M):
    :return:w, v.
    w (...,M) : the eigenvalues.
    v (...,M,M) : normalized eigenvectors.
    """
    assert isinstance(x, ComplexNumber), "complex_eig is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for complex_eig"
    def forward_code(np, data):
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        a = _stack_to_complex(data["inputs"][0])
        w, v = data["outputs"]
        tw, tv = np.linalg.eig(a)
        np.copyto(w, _complex_to_stack(tw))
        np.copyto(v, _complex_to_stack(tv))

    def backward_code(np, data):
        raise NotImplementedError

    sw = x.shape[:-2] + x.shape[-1:] + (2,)
    sv = x.value.shape
    w, v = jt.numpy_code(
        [sw, sv],
        [x.value.dtype, x.value.dtype],
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(w, is_concat_value=True), ComplexNumber(v, is_concat_value=True)

def complex_qr(x):
    r"""
    do the qr factorization of x in the below formula:
    x = QR where Q is orthogonal matrix and R is upper-triangle matrix.
    :param x (...,M,M):
    :return:q,r as the result of qr factorization.They are both in the shape of (...,M,M).
    """
    assert isinstance(x, ComplexNumber), "linalg_qr is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for linalg_qr"
    def forward_code(np, data):
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        a = _stack_to_complex(data["inputs"][0])
        qr = data["outputs"][0]
        Q, R = np.linalg.qr(a)
        QR = np.stack([Q, R], axis=0)
        np.copyto(qr, _complex_to_stack(QR))

    def backward_code(np, data):
        # reference: https://github.com/tencent-quantum-lab/tensorcircuit/blob/master/tensorcircuit/backends/pytorch_ops.py
        def H(x):
            return np.conj(np.swapaxes(x, -1, -2))
        def _TriangularSolve(x, r):
            return H(np.linalg.solve(r, H(x)))
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        _diag = partial(np.einsum, '...ii->...i')

        dout = data["dout"]
        out = data["outputs"][0]
        qr = data["f_outputs"][0]
        dout = _stack_to_complex(dout)
        dq, dr = dout[0], dout[1]
        qr = _stack_to_complex(qr)
        q, r = qr[0], qr[1]


        qdq = _dot(H(q), dq)
        qdq_ = qdq - H(qdq)
        rdr = _dot(r, H(dr))
        rdr_ = rdr - H(rdr)
        tril = np.tril(qdq_ + rdr_)

        grad_a = _dot(q, dr + _TriangularSolve(tril, r))
        grad_b = _TriangularSolve(dq - _dot(q, qdq), r)
        ret = grad_a + grad_b

        m = rdr - H(qdq)
        eyem = np.zeros_like(m)
        _diag(eyem)[:] = _diag(m)
        correction = eyem - np.real(eyem)
        ret = ret + _TriangularSolve(_dot(q, H(correction)), r)
        
        ret = _complex_to_stack(ret)
        np.copyto(out,ret)

    qr = jt.numpy_code(
        (2,) + x.value.shape,
        x.value.dtype,
        [x.value],
        forward_code,
        [backward_code],
    )
    q, r = qr[0], qr[1]
    return ComplexNumber(q, is_concat_value=True), ComplexNumber(r, is_concat_value=True)

def complex_svd(x:ComplexNumber):
    r'''
    calculate the Singular Value Decomposition of x.It follows the below fomula:
    x = usv*
    only support full matrices == False ver now, which means:
    x's shape (...,M,K)
    u's shape (...,M,K)
    s's shape (...,K)
    v's shape (...,K,N)
    where K is min(M,N).
    :param x:
    :return:u,s,v.
    '''
    def forward_code(np, data):
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        a = _stack_to_complex(data["inputs"][0])
        u, s, v = data["outputs"]
        #TODO:remove copyto
        tu, ts, tv = np.linalg.svd(a, full_matrices=0)
        np.copyto(u, _complex_to_stack(tu))
        np.copyto(s, _complex_to_stack(ts))
        np.copyto(v, _complex_to_stack(tv))

    def backward_code(np, data):
        raise NotImplementedError

    m, n = x.shape[-2:]
    k = min(m, n)
    s1 = list(x.shape)
    s1[-1] = k
    s2 = list(x.shape)
    s2[-2] = k
    s3 = list(x.shape)[:-2]
    s3.append(k)
    s1.append(2)
    s2.append(2)
    s3.append(2)
    u, s, v = jt.numpy_code(
        [s1, s3, s2],
        [x.value.dtype, x.value.dtype, x.value.dtype],
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(u, is_concat_value=True), \
            ComplexNumber(s, is_concat_value=True), \
            ComplexNumber(v, is_concat_value=True)

#TODO:full_matrices=1
def svd(x):
    r'''
    calculate the Singular Value Decomposition of x.It follows the below fomula:
    x = usv*
    only support full matrices == False ver now, which means:
    x's shape (...,M,K)
    u's shape (...,M,K)
    s's shape (...,K)
    v's shape (...,K,N)
    where K is min(M,N).
    :param x:
    :return:u,s,v.
    '''
    if isinstance(x, ComplexNumber):
        return complex_svd(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        u, s, v = data["outputs"]
        #TODO:remove copyto
        tu, ts, tv = np.linalg.svd(a, full_matrices=0)
        np.copyto(u, tu)
        np.copyto(s, ts)
        np.copyto(v, tv)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        out_index = data["out_index"]
        u, s, v = data["f_outputs"]
        v = T(v)
        m, n = inp.shape[-2:]
        k = np.min((m, n))
        i = np.reshape(np.eye(k), np.concatenate((np.ones(inp.ndim - 2, dtype=int), (k, k))))
        if out_index == 0:
            f = 1 / (s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2 + i)
            gu = dout
            utgu = _dot(T(u), gu)
            t = (f * (utgu - T(utgu))) * s[..., np.newaxis, :]
            t = _dot(_dot(u, t), T(v))
            if m > n:
                i_minus_uut = (np.reshape(np.eye(m), np.concatenate((np.ones(inp.ndim - 2, dtype=int), (m, m)))) -
                               _dot(u, np.conj(T(u))))
                t = t + T(_dot(_dot(v / s[..., np.newaxis, :], T(gu)), i_minus_uut))
            np.copyto(out, t)
        elif out_index == 1:
            gs = dout
            t = i * gs[..., :, np.newaxis]
            t = _dot(_dot(u, t), T(v))
            np.copyto(out, t)
        elif out_index == 2:
            f = 1 / (s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2 + i)
            gv = dout
            vtgv = _dot(T(v), gv)
            t = s[..., :, np.newaxis] * (f * (vtgv - T(vtgv)))
            t = _dot(_dot(u, t), T(v))
            if m < n:
                i_minus_vvt = (np.reshape(np.eye(n), np.concatenate((np.ones(inp.ndim - 2, dtype=int), (n, n)))) -
                               _dot(v, np.conj(T(v))))
                t = t + T(_dot(_dot(u / s[..., np.newaxis, :], T(gv)), i_minus_vvt))
            np.copyto(out, t)

    m, n = x.shape[-2:]
    k = min(m, n)
    s1 = list(x.shape)
    s1[-1] = k
    s2 = list(x.shape)
    s2[-2] = k
    s3 = list(x.shape)[:-2]
    s3.append(k)
    u, s, v = jt.numpy_code(
        [s1, s3, s2],
        [x.dtype, x.dtype, x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return u, s, v

def eig(x):
    r"""
    calculate the eigenvalues and eigenvectors of x.
    :param x (...,M,M):
    :return (ComplexNumber):w, v.
    w (...,M) : the eigenvalues.
    v (...,M,M) : normalized eigenvectors.
    """
    if isinstance(x, ComplexNumber):
        return complex_eig(x)
    return complex_eig(ComplexNumber(x))

def eigh(x):
    r"""
    calculate the eigenvalues and eigenvectors of x.
    :param x (...,M,M):
    :return:w, v.
    w (...,M) : the eigenvalues.
    v (...,M,M) : normalized eigenvectors.
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        w, v = data["outputs"]
        tw, tv = np.linalg.eigh(a, UPLO='L')
        np.copyto(w, tw)
        np.copyto(v, tv)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        out_index = data["out_index"]
        w, v = data["f_outputs"]
        k = int(inp.shape[-1])
        w_repeated = np.repeat(w[..., np.newaxis], k, axis=-1)
        if out_index == 0:
            t = _dot(v * dout[..., np.newaxis, :], T(v))
            np.copyto(out, t)
        elif out_index == 1:
            if np.any(dout):
                off_diag = np.ones((k, k)) - np.eye(k)
                F = off_diag / (T(w_repeated) - w_repeated + np.eye(k))
                t = _dot(_dot(v, F * _dot(T(v), dout)), T(v))
                np.copyto(out, t)

    sw = x.shape[:-2] + x.shape[-1:]
    sv = x.shape
    w, v = jt.numpy_code(
        [sw, sv],
        [x.dtype, x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return w, v


def inv(x):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).
    """
    if isinstance(x, ComplexNumber):
        return complex_inv(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.inv(a)
        np.copyto(m_a, t_a)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        lmx = data["f_outputs"]
        mx = lmx[0]
        t = -_dot(_dot(T(mx), dout), T(mx))
        np.copyto(out, t)

    lmx = jt.numpy_code(
        [x.shape],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    mx = lmx[0]
    return mx


def pinv(x):
    r"""
    calculate the pseudo-inverse of a x.
    :param x (...,M,N)
    :return: x's pinv (...N,M)
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.pinv(a)
        np.copyto(m_a, t_a)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        lmx = data["f_outputs"]
        mx = lmx[0]
        t = T(
            -_dot(_dot(mx, T(dout)), mx)
            + _dot(_dot(_dot(mx, T(mx)), dout), np.eye(inp.shape[-2]) - _dot(inp, mx))
            + _dot(_dot(_dot(np.eye(mx.shape[-2]) - _dot(mx, inp), dout), T(mx)), mx)
        )
        np.copyto(out, t)
    sw = list(x.shape[:-2]) + [x.shape[-1]] + [x.shape[-2]]
    lmx = jt.numpy_code(
        [sw],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    mx = lmx[0]
    return mx


def matrix_power(x, n):
    r"""
    Compute the ``n``-th power of a (batch of) square matrix.

    Equivalent to ``torch.linalg.matrix_power`` / ``numpy.linalg.matrix_power``.
    The power is formed entirely from existing jittor ops (``jt.matmul`` and
    :func:`inv`), so the result is differentiable on-device with no numpy
    round-trip.

    :param x (...,M,M): batch of square matrices.
    :param n (int): integer exponent. ``n == 0`` returns identity matrices,
        ``n < 0`` uses the matrix inverse ``x^{-1}`` raised to ``-n``.
    :return: ``x ** n`` (...,M,M).
    """
    if not isinstance(n, int):
        # mirror numpy/torch: only integer exponents are supported
        if hasattr(n, "__index__"):
            n = n.__index__()
        else:
            raise TypeError("matrix_power: exponent 'n' must be an integer")
    assert x.shape[-2] == x.shape[-1], \
        "matrix_power expects square matrices (last two dims equal)"

    if n == 0:
        # batched identity, broadcast to x's batch shape and dtype
        m = x.shape[-1]
        eye = jt.init.eye(m, dtype=x.dtype)
        batch = list(x.shape[:-2])
        if batch:
            eye = eye.broadcast(batch + [m, m])
        return eye
    if n < 0:
        x = inv(x)
        n = -n

    # binary exponentiation to keep the matmul count at O(log n)
    result = None
    base = x
    e = n
    while e > 0:
        if e & 1:
            result = base if result is None else jt.matmul(result, base)
        e >>= 1
        if e > 0:
            base = jt.matmul(base, base)
    return result


def matrix_rank(x, tol=None, hermitian=False):
    r"""
    Return the numerical rank of a (batch of) matrix.

    Equivalent to ``torch.linalg.matrix_rank`` / ``numpy.linalg.matrix_rank``:
    the rank is the number of singular values greater than ``tol``. When ``tol``
    is ``None`` the default threshold ``S.max(-1) * max(M, N) * eps`` is used
    (``eps`` is the machine epsilon of the input dtype), matching numpy/torch.

    The singular values are obtained from the existing differentiable
    :func:`svd`; the rank itself is an integer count and therefore has no
    gradient (matching numpy/torch).

    :param x (...,M,N): batch of matrices.
    :param tol (float, optional): explicit singular-value threshold.
    :param hermitian (bool): if ``True``, use the symmetric eigensolver
        (eigenvalue magnitudes) instead of the SVD; ``x`` must be square.
    :return: integer rank tensor with shape ``x.shape[:-2]``.
    """
    if hermitian:
        assert x.shape[-2] == x.shape[-1], \
            "matrix_rank(hermitian=True) expects square matrices"
        w, _ = eigh(x)
        s = jt.abs(w)
    else:
        _, s, _ = svd(x)

    smax = s.max(dim=-1)
    if tol is None:
        import numpy as _np
        try:
            eps = float(_np.finfo(_np.dtype(str(x.dtype))).eps)
        except Exception:
            eps = float(_np.finfo(_np.float32).eps)
        m, n = x.shape[-2], x.shape[-1]
        tol_v = smax * (max(m, n) * eps)
    else:
        tol_v = jt.array(tol).broadcast(smax.shape) if not isinstance(tol, (int, float)) \
            else smax * 0 + float(tol)
    # count singular values strictly greater than the threshold
    keep = (s > tol_v.unsqueeze(-1)).int32()
    return keep.sum(dim=-1)


def matrix_norm(x, ord='fro', dim=(-2, -1), keepdim=False):
    r"""
    Compute a matrix norm of a (batch of) matrix.

    Matches ``torch.linalg.matrix_norm`` / ``numpy.linalg.norm`` for the
    matrix-valued orders:

    - ``'fro'`` (default): Frobenius norm ``sqrt(sum(|x|**2))`` (native jittor).
    - ``'nuc'``: nuclear norm, the sum of singular values (via :func:`svd`).
    - ``2`` / ``-2``: largest / smallest singular value (via :func:`svd`).
    - ``1`` / ``-1``: max / min absolute column sum (native jittor).
    - ``inf`` / ``-inf``: max / min absolute row sum (native jittor).

    The Frobenius and the ``1/inf`` family are built from native jittor ops and
    stay differentiable on-device; the spectral (``2``/``-2``/``'nuc'``) orders
    use the existing differentiable :func:`svd`.

    :param x (...,M,N): batch of matrices.
    :param ord: matrix order, see above. Default ``'fro'``.
    :param dim (tuple[int, int]): the two dims that form each matrix.
    :param keepdim (bool): keep the reduced dims as size-1.
    :return: matrix norm with the two ``dim`` axes reduced.
    """
    import math
    d0, d1 = dim
    nd = len(x.shape)
    d0 = d0 % nd
    d1 = d1 % nd
    assert d0 != d1, "matrix_norm: dim must reference two distinct axes"

    def _restore(res, reduced_axes):
        # re-insert size-1 axes so keepdim=True lines up with the input rank
        if not keepdim:
            return res
        out = res
        for ax in sorted(reduced_axes):
            out = out.unsqueeze(ax)
        return out

    if ord == 'fro':
        res = jt.sqrt((x * x).sum(dims=[d0, d1]))
        return _restore(res, [d0, d1])
    if ord == 'nuc':
        # move matrix dims to the end, take sum of singular values
        s = _matrix_singular_values(x, d0, d1)
        res = s.sum(dim=-1)
        return _restore(res, [d0, d1])
    if ord in (2, -2):
        s = _matrix_singular_values(x, d0, d1)
        res = s.max(dim=-1) if ord == 2 else s.min(dim=-1)
        return _restore(res, [d0, d1])
    if ord in (1, -1):
        # absolute column sums (reduce the row dim d0), then max/min over cols
        col_sums = jt.abs(x).sum(dim=d0)
        # after reducing d0, the column axis d1 shifts left if d0 < d1
        col_axis = d1 - 1 if d0 < d1 else d1
        res = col_sums.max(dim=col_axis) if ord == 1 else col_sums.min(dim=col_axis)
        return _restore(res, [d0, d1])
    if ord in (math.inf, -math.inf, float('inf'), float('-inf')):
        # absolute row sums (reduce the column dim d1), then max/min over rows
        row_sums = jt.abs(x).sum(dim=d1)
        row_axis = d0 - 1 if d1 < d0 else d0
        if ord > 0:
            res = row_sums.max(dim=row_axis)
        else:
            res = row_sums.min(dim=row_axis)
        return _restore(res, [d0, d1])
    raise ValueError(f"matrix_norm: unsupported matrix order {ord!r}")


def _matrix_singular_values(x, d0, d1):
    # Bring the two matrix axes to the last two positions, run the existing
    # differentiable svd, and return the singular values (shape batch + [k]).
    nd = len(x.shape)
    others = [a for a in range(nd) if a not in (d0, d1)]
    perm = others + [d0, d1]
    xt = x.permute(perm) if perm != list(range(nd)) else x
    _, s, _ = svd(xt)
    return s


def vector_norm(x, ord=2, dim=None, keepdim=False):
    r"""
    Compute a vector norm, matching ``torch.linalg.vector_norm``.

    The whole tensor (``dim=None``) or the given ``dim`` axes are treated as a
    flat vector. Supported orders:

    - ``2`` (default): Euclidean norm ``sqrt(sum(|x|**2))``.
    - ``1``: sum of absolute values.
    - ``0``: number of non-zero entries.
    - any finite ``p``: ``(sum(|x|**p))**(1/p)``.
    - ``inf`` / ``-inf``: max / min absolute value.

    Built entirely from native jittor ops, so it is differentiable on-device.

    :param x: input tensor.
    :param ord: vector order, see above. Default ``2``.
    :param dim (int | tuple[int] | None): axis/axes to reduce; ``None`` flattens.
    :param keepdim (bool): keep reduced dims as size-1 (ignored when ``dim`` is
        ``None`` and the whole tensor is flattened).
    :return: the requested vector norm.
    """
    import math
    if dim is None:
        flat = x.reshape([-1])
        ax = 0
        out = _vector_norm_reduce(flat, ord, ax)
        return out
    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    return _vector_norm_reduce(x, ord, dims, keepdim=keepdim)


def _vector_norm_reduce(x, ord, dims, keepdim=False):
    import math
    ax = dims
    absx = jt.abs(x)
    if ord == 2:
        res = jt.sqrt((absx * absx).sum(dims=ax) if isinstance(ax, list)
                      else (absx * absx).sum(dim=ax))
    elif ord == 1:
        res = absx.sum(dims=ax) if isinstance(ax, list) else absx.sum(dim=ax)
    elif ord == 0:
        nz = (x != 0).float32()
        res = nz.sum(dims=ax) if isinstance(ax, list) else nz.sum(dim=ax)
    elif ord in (math.inf, float('inf')):
        res = _reduce_axes(absx, ax, reduce='max')
    elif ord in (-math.inf, float('-inf')):
        res = _reduce_axes(absx, ax, reduce='min')
    else:
        p = float(ord)
        powed = absx ** p
        s = powed.sum(dims=ax) if isinstance(ax, list) else powed.sum(dim=ax)
        res = s ** (1.0 / p)
    if keepdim and isinstance(ax, list):
        for a in sorted(a % len(x.shape) for a in ax):
            res = res.unsqueeze(a)
    return res


def _reduce_axes(x, ax, reduce='max'):
    # max/min over possibly several axes (jittor reduces one axis at a time
    # for the index-returning variants, but the value reductions accept lists).
    if isinstance(ax, int):
        ax = [ax]
    # reduce from the highest axis down so earlier indices stay valid
    out = x
    for a in sorted((a % len(x.shape) for a in ax), reverse=True):
        out = out.max(dim=a) if reduce == 'max' else out.min(dim=a)
    return out


def norm(x, ord=None, dim=None, keepdim=False):
    r"""
    General ``torch.linalg.norm`` / ``numpy.linalg.norm`` dispatcher.

    Behaviour (matching torch/numpy):

    - ``dim`` is a 2-tuple, or ``dim is None`` and ``x`` is 2-D: matrix norm
      (default ``ord`` is the Frobenius norm), see :func:`matrix_norm`.
    - ``dim`` is an int, or ``dim is None`` and ``x`` is 1-D, or ``ord`` is set
      with ``dim is None``: vector norm (default ``ord`` is 2), see
      :func:`vector_norm`.
    - ``dim is None`` and ``ord is None``: the 2-norm of the flattened tensor
      (Frobenius), built from native jittor ops.

    :param x: input tensor.
    :param ord: order of the norm; meaning depends on whether a matrix or a
        vector norm is selected (see :func:`matrix_norm` / :func:`vector_norm`).
    :param dim (int | tuple | None): axis/axes defining the norm.
    :param keepdim (bool): keep reduced dims as size-1.
    :return: the requested norm.
    """
    if dim is None:
        if ord is None:
            # flatten and take the 2-norm (== Frobenius for matrices)
            return vector_norm(x, ord=2, dim=None)
        if len(x.shape) == 2:
            # ord given, 2-D input, dim=None -> matrix norm (torch/numpy rule)
            return matrix_norm(x, ord=ord, dim=(-2, -1), keepdim=keepdim)
        # 1-D, or higher-rank with ord set but no dim -> vector norm over the
        # flattened input (matches numpy.linalg.norm).
        return vector_norm(x, ord=ord, dim=None)
    if isinstance(dim, (tuple, list)) and len(dim) == 2:
        m_ord = 'fro' if ord is None else ord
        return matrix_norm(x, ord=m_ord, dim=tuple(dim), keepdim=keepdim)
    v_ord = 2 if ord is None else ord
    return vector_norm(x, ord=v_ord, dim=dim, keepdim=keepdim)


def cond(x, p=None):
    r"""
    Condition number of a (batch of) matrix, matching
    ``torch.linalg.cond`` / ``numpy.linalg.cond``.

    - ``p`` in ``{None, 2}``: ratio of largest to smallest singular value
      (via the existing differentiable :func:`svd`).
    - ``p == -2``: smallest over largest singular value.
    - ``p`` in ``{1, -1, inf, -inf}``: ``norm(x, p) * norm(inv(x), p)`` using the
      matrix-norm definitions above; ``x`` must be invertible.
    - ``p == 'fro'``: ``norm(x, 'fro') * norm(inv(x), 'fro')``.

    :param x (...,M,M): batch of square matrices (square required for all ``p``
        except the ``None/2/-2`` singular-value forms, which also accept the
        general case numpy/torch allow).
    :param p: order of the norm used for the condition number.
    :return: condition number with shape ``x.shape[:-2]``.
    """
    import math
    if p is None or p == 2 or p == -2:
        _, s, _ = svd(x)
        smax = s.max(dim=-1)
        smin = s.min(dim=-1)
        if p == -2:
            return smin / smax
        return smax / smin
    # p in {1, -1, inf, -inf, 'fro'} -> norm(x, p) * norm(inv(x), p)
    assert x.shape[-2] == x.shape[-1], \
        f"cond(p={p!r}) expects square matrices"
    xi = inv(x)
    return matrix_norm(x, ord=p, dim=(-2, -1)) * matrix_norm(xi, ord=p, dim=(-2, -1))


def det(x):
    r"""
    calculate the determinant of x.
    :param x (...,M,M):
    :return:|x| (...,1)
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        L = data["outputs"][0]
        tL = np.linalg.det(a)
        np.copyto(L, tL)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        f_out = data["f_outputs"][0]
        inp = data["inputs"][0]
        n_d = np.reshape(dout, np.shape(dout) + (1, 1))
        n_o = np.reshape(f_out, np.shape(f_out) + (1, 1))
        s = n_d * n_o * T(np.linalg.inv(inp))
        np.copyto(out, s)

    s = x.shape
    x_s = s[:-2]
    if len(s) == 2:
        x_s.append(1)
    l_det = jt.numpy_code(
        [x_s],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    det = l_det[0]
    return det


def slogdet(x):
    r"""
    calculate the sign and log of the determinant of x.
    :param x (...,M,M):
    :return sign, x's logdet.
    sign array decides the sign of determinant and their values can be -1,0,1.Only Real number now.0 means det is 0 and logdet is -inf.
    logdet in shape (...,1).
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        sign, m_a = data["outputs"]
        sign_, t_a = np.linalg.slogdet(a)
        np.copyto(m_a, t_a)
        np.copyto(sign, sign_)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        out_index = data["out_index"]
        if out_index == 0:
            np.copyto(out, 0)
        if out_index == 1:
            t = np.reshape(dout, np.shape(dout) + (1, 1))
            t = t * T(np.linalg.inv(inp))
            np.copyto(out, t)

    s = x.shape
    det_s = s[:-2]
    if len(det_s) == 0:
        det_s.append(1)
    sign, mx = jt.numpy_code(
        [det_s, det_s],
        [x.dtype, x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return sign, mx


def cholesky(x):
    r"""
    do Cholesky decomposition of x in the form of below formula:
    x = LL^T
    x must be a Hermite and positive-definite matrix. L is a lower-triangular matrix.
    :param x (...,M,M):
    :return: L (...,M,M).
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        L = data["outputs"][0]
        tL = np.linalg.cholesky(a)
        np.copyto(L, tL)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        f_out = data["f_outputs"][0]
        solve_trans = lambda a, b: np.linalg.solve(T(a), b)
        phi = lambda X: np.tril(X) / (1. + np.eye(X.shape[-1]))

        def conjugate_solve(L, X):
            return solve_trans(L, T(solve_trans(L, T(X))))

        s = conjugate_solve(f_out, phi(np.einsum('...ki,...kj->...ij', f_out, dout)))
        s = (s + T(s)) / 2.
        np.copyto(out, s)

    lL = jt.numpy_code(
        [x.shape],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    L = lL[0]
    return L


def solve(a,b):
    r"""
    Solve a linear matrix equation Ax = B.This is done by calculating x = A^-1B.So A must not be singular.
    :param a:(...,M,M)
    :param b:(...,M)
    :return:solution of Ax = b formula.x in the shape of (...M)
    """
    def forward_code(np, data):
        a, b = data["inputs"]
        L = data["outputs"][0]
        ans = np.linalg.solve(a, b)
        np.copyto(L, ans)

    def backward_code1(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        f_out = data["f_outputs"][0]
        inp = data["inputs"][0]
        updim = lambda x: x if x.ndim == a.ndim else x[..., None]
        t = -_dot(updim(np.linalg.solve(T(inp), dout)), T(updim(f_out)))
        np.copyto(out, t)

    def backward_code2(np, data):
        out = data["outputs"][0]
        np.copyto(out, 0)

    l_ans = jt.numpy_code(
        [b.shape],
        [b.dtype],
        [a, b],
        forward_code,
        [backward_code1, backward_code2],
    )
    ans = l_ans[0]
    return ans


def qr(x):
    r"""
    do the qr factorization of x in the below formula:
    x = QR where Q is orthogonal matrix and R is upper-triangle matrix.
    :param x (...,M,M):
    :return:q,r as the result of qr factorization.They are both in the shape of (...,M,M).
    """
    if isinstance(x, ComplexNumber):
        return complex_qr(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        q, r = data["outputs"]
        Q, R = np.linalg.qr(a)
        np.copyto(q,Q)
        np.copyto(r,R)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        _harmard = partial(np.einsum, '...ij,...ij->...ij')
        dout = data["dout"]
        out = data["outputs"][0]
        q, r = data["f_outputs"]
        out_index = data["out_index"]
        #pl = np.tril(np.ones((inp.shape[-1],inp.shape[-1])))-diags
        if out_index == 0: # Q_TERM
            q_t = _dot(T(q),dout)
            rhs_solve = q_t - T(q_t)
            rhs_solve = T(np.tril(rhs_solve,-1))
            qsolve = np.linalg.solve(r,rhs_solve)
            qsolve = T(qsolve)
            tq = _dot(q,qsolve)
            np.copyto(out,tq)
        else: #R_TERM
            r_t = _dot(r ,T(dout))
            rhs_solve = r_t - T(r_t)
            rhs_solve = np.tril(rhs_solve,-1)
            rhs_solve = T(rhs_solve)
            r_solve = np.linalg.solve(r,rhs_solve)
            tr = _dot(q,(T(r_solve) + dout))
            np.copyto(out,tr)

    q, r = jt.numpy_code(
        [x.shape,x.shape],
        [x.dtype,x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return q, r


def einsum(equation, *operands):
    r"""Evaluate the Einstein summation convention on the operands.

    Native jittor implementation: GEMM-shaped contractions dispatch to
    ``jt.matmul`` (cublas on GPU). Other cases are expressed with the
    existing jittor tensor ops used by this implementation, including
    ``reindex`` for diagonal handling, element-wise operations, reductions
    such as ``sum``, and reshapes/transposes as needed. No numpy round-trip,
    so ``float16`` and ``bfloat16`` operands stay on-device throughout and
    autograd is provided by the underlying jittor ops.
    """
    import numpy as np_cpu
    if len(operands) == 0:
        raise ValueError("einsum requires at least one operand")
    # torch.einsum also accepts the operands packed in a single list/tuple, e.g.
    # torch.einsum("bcxd,bcyd->bcxy", (query, key)) (used by longformer's windowed
    # attention). Unpack that form so the operands are individual Vars below.
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = tuple(operands[0])
    # ``_parse_einsum_input`` calls ``asanyarray`` on the operands, so feed
    # it shape-compatible numpy stand-ins and keep the original jittor Vars.
    fake_ops = [np_cpu.empty([1] * len(o.shape), dtype=np_cpu.float32)
                for o in operands]
    in_subs_str, out_sub, _ = np_cpu.core.einsumfunc._parse_einsum_input(
        [equation, *fake_ops])
    in_subs = in_subs_str.split(",")
    operands = list(operands)

    new_subs, new_ops = [], []
    for i, (s, op) in enumerate(zip(in_subs, operands)):
        s, op = _einsum_unary_normalize(s, op, in_subs, out_sub, i)
        new_subs.append(s)
        new_ops.append(op)
    in_subs, operands = new_subs, new_ops

    while len(operands) > 1:
        i, j = _einsum_pick_pair(in_subs, out_sub)
        # Pop the higher index first so the lower-index pop stays valid.
        b = operands.pop(j); sb = in_subs.pop(j)
        a = operands.pop(i); sa = in_subs.pop(i)
        so = _einsum_intermediate_subs(sa, sb, in_subs, out_sub)
        operands.insert(0, _einsum_pair_contract(sa, sb, so, a, b))
        in_subs.insert(0, so)

    return _einsum_finalize(in_subs[0], out_sub, operands[0])


def _einsum_pick_pair(in_subs, out_sub):
    # Prefer a pair that shares at least one label, to avoid unnecessary
    # outer-products. Among shared-label pairs, prefer ones whose contraction
    # result is non-empty when there are still operands queued — collapsing
    # to a scalar mid-stream forces the leftover operands through a degenerate
    # ``scalar * tensor`` path.
    n = len(in_subs)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            sa, sb = in_subs[i], in_subs[j]
            shared = set(sa) & set(sb)
            if not shared:
                continue
            remaining = [in_subs[k] for k in range(n) if k != i and k != j]
            keep = set(out_sub)
            for s in remaining:
                keep |= set(s)
            result_chars = (set(sa) | set(sb)) & keep
            score = (1 if result_chars else 0, len(shared))
            if best is None or score > best[0]:
                best = (score, i, j)
    if best is not None:
        return best[1], best[2]
    return 0, 1


def _einsum_dedup(s):
    seen, out = set(), []
    for c in s:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _einsum_permute_to(s_from, s_to, op):
    if s_from == s_to:
        return op
    perm = [s_from.index(c) for c in s_to]
    if perm == list(range(len(perm))):
        return op
    return op.permute(perm)


def _einsum_unary_normalize(sub, op, all_subs, out_sub, my_index):
    # Diagonal extraction: collapse repeated indices via a reindex.
    if len(set(sub)) != len(sub):
        unique = _einsum_dedup(sub)
        out_shape = []
        for c in unique:
            sizes = [op.shape[k] for k, ch in enumerate(sub) if ch == c]
            for sz in sizes[1:]:
                if sz != sizes[0]:
                    raise ValueError(
                        f"einsum: repeated index '{c}' has mismatched dim sizes {sizes}")
            out_shape.append(sizes[0])
        formulas = [f"i{unique.index(c)}" for c in sub]
        op = op.reindex(out_shape, formulas)
        sub = "".join(unique)

    # Sum out indices that don't appear in any other operand or in the output.
    other_chars = set(out_sub)
    for k, s in enumerate(all_subs):
        if k != my_index:
            other_chars |= set(s)
    sum_dims = [i for i, c in enumerate(sub) if c not in other_chars]
    if sum_dims:
        op = op.sum(sum_dims)
        drop = set(sum_dims)
        sub = "".join(c for i, c in enumerate(sub) if i not in drop)
    return sub, op


def _einsum_intermediate_subs(sa, sb, remaining_subs, out_sub):
    keep = set(out_sub)
    for s in remaining_subs:
        keep |= set(s)
    return "".join(c for c in _einsum_dedup(sa + sb) if c in keep)


def _einsum_resolve_size(label, size_a, size_b):
    # numpy-style broadcasting: 1 ↔ N → N; equal → that size; mismatch → error.
    if size_a is None: return size_b
    if size_b is None: return size_a
    if size_a == size_b: return size_a
    if size_a == 1: return size_b
    if size_b == 1: return size_a
    raise ValueError(
        f"einsum: shape mismatch for index '{label}': {size_a} vs {size_b}")


def _einsum_broadcast_axes(op, target_shape):
    # Expand size-1 axes of ``op`` to match ``target_shape``. Lengths must match.
    cur_shape = list(op.shape)
    if len(cur_shape) != len(target_shape):
        raise ValueError(
            f"einsum: cannot broadcast shape {tuple(cur_shape)} to "
            f"{tuple(target_shape)}: rank mismatch")
    needs = False
    for axis, (cs, ts) in enumerate(zip(cur_shape, target_shape)):
        if cs != ts:
            if cs != 1:
                raise ValueError(
                    f"einsum: cannot broadcast shape {tuple(cur_shape)} to "
                    f"{tuple(target_shape)}: axis {axis} has size {cs}, "
                    f"expected {ts} or 1")
            needs = True
    if not needs:
        return op
    return op.broadcast(target_shape)


def _einsum_pair_contract(sa, sb, so, a, b):
    # If either operand is the result of an earlier full contraction (sub == ""),
    # it has no labelled axes; fall back to a plain element-wise multiply, which
    # will broadcast a 0-D / shape-(1,) scalar against the remaining operand.
    if not sa and not sb:
        out = a * b
        return out
    if not sa:
        return _einsum_finalize_scalar_pair(a, sb, so, b)
    if not sb:
        return _einsum_finalize_scalar_pair(b, sa, so, a)

    a_set, b_set, o_set = set(sa), set(sb), set(so)

    drop_a = [i for i, c in enumerate(sa) if c not in b_set and c not in o_set]
    if drop_a:
        a = a.sum(drop_a)
        ds = set(drop_a)
        sa = "".join(c for i, c in enumerate(sa) if i not in ds)
        a_set = set(sa)
    drop_b = [i for i, c in enumerate(sb) if c not in a_set and c not in o_set]
    if drop_b:
        b = b.sum(drop_b)
        ds = set(drop_b)
        sb = "".join(c for i, c in enumerate(sb) if i not in ds)
        b_set = set(sb)

    shared = a_set & b_set
    batch = [c for c in so if c in shared]
    contract = [c for c in sa if c in shared and c not in o_set]
    free_a = [c for c in sa if c not in shared]
    free_b = [c for c in sb if c not in shared]

    # Resolve broadcasted size per label for shared (batch + contract) axes.
    sizes_a = {c: a.shape[i] for i, c in enumerate(sa)}
    sizes_b = {c: b.shape[i] for i, c in enumerate(sb)}
    resolved = {}
    for c in shared:
        resolved[c] = _einsum_resolve_size(c, sizes_a[c], sizes_b[c])
    for c in free_a:
        resolved[c] = sizes_a[c]
    for c in free_b:
        resolved[c] = sizes_b[c]

    if not contract:
        return _einsum_outer(sa, sb, so, a, b, batch, free_a, free_b, resolved)

    return _einsum_matmul(sa, sb, so, a, b, batch, free_a, free_b, contract, resolved)


def _einsum_finalize_scalar_pair(scalar, sb, so, b):
    # ``scalar`` is the result of a prior full contraction (sub == "").
    # In jittor that's a shape-(1,) Var; multiply broadcasts it against ``b``.
    if scalar.ndim > 1:
        scalar = scalar.reshape([1])
    out = b * scalar
    drop = [i for i, c in enumerate(sb) if c not in so]
    if drop:
        out = out.sum(drop)
        ds = set(drop)
        sb = "".join(c for i, c in enumerate(sb) if i not in ds)
    return _einsum_permute_to(sb, so, out)


def _einsum_outer(sa, sb, so, a, b, batch, free_a, free_b, resolved):
    a_target = "".join(batch + free_a)
    a_perm = _einsum_permute_to(sa, a_target, a)
    # Broadcast batch axes of A to resolved sizes (free_a comes from A only).
    a_target_shape = [resolved[c] for c in batch] + [a_perm.shape[len(batch) + i]
                                                     for i in range(len(free_a))]
    a_perm = _einsum_broadcast_axes(a_perm, a_target_shape)
    for _ in free_b:
        a_perm = a_perm.unsqueeze(-1)

    b_target = "".join(batch + free_b)
    b_perm = _einsum_permute_to(sb, b_target, b)
    b_target_shape = [resolved[c] for c in batch] + [b_perm.shape[len(batch) + i]
                                                     for i in range(len(free_b))]
    b_perm = _einsum_broadcast_axes(b_perm, b_target_shape)
    insert_at = len(batch)
    for _ in free_a:
        b_perm = b_perm.unsqueeze(insert_at)

    out = a_perm * b_perm
    intermediate = "".join(batch + free_a + free_b)
    return _einsum_permute_to(intermediate, so, out)


def _einsum_matmul(sa, sb, so, a, b, batch, free_a, free_b, contract, resolved):
    a_target = "".join(batch + free_a + contract)
    b_target = "".join(batch + contract + free_b)
    a_p = _einsum_permute_to(sa, a_target, a)
    b_p = _einsum_permute_to(sb, b_target, b)

    nb = len(batch)
    nfa = len(free_a)
    nc = len(contract)

    # Broadcast each operand's batch and contract axes to the resolved sizes;
    # free axes come from a single operand and stay as-is.
    batch_shape = [resolved[c] for c in batch]
    fa_shape = [a_p.shape[nb + i] for i in range(nfa)]
    fb_shape = [b_p.shape[nb + nc + i] for i in range(len(free_b))]
    contract_shape = [resolved[c] for c in contract]

    a_p = _einsum_broadcast_axes(a_p, batch_shape + fa_shape + contract_shape)
    b_p = _einsum_broadcast_axes(b_p, batch_shape + contract_shape + fb_shape)

    fa_size = 1
    for s in fa_shape: fa_size *= s
    c_size = 1
    for s in contract_shape: c_size *= s
    fb_size = 1
    for s in fb_shape: fb_size *= s

    a_flat = a_p.reshape(batch_shape + [fa_size, c_size])
    b_flat = b_p.reshape(batch_shape + [c_size, fb_size])

    out_flat = jt.matmul(a_flat, b_flat)

    target_shape = batch_shape + fa_shape + fb_shape
    if target_shape:
        out = out_flat.reshape(target_shape)
    else:
        # Pure dot product: collapse the [1,1] result of the GEMM to a 0-D scalar.
        out = out_flat.sum()

    intermediate = "".join(batch + free_a + free_b)
    return _einsum_permute_to(intermediate, so, out)


def _einsum_finalize(s_from, out_sub, op):
    drop = [i for i, c in enumerate(s_from) if c not in out_sub]
    if drop:
        op = op.sum(drop)
        ds = set(drop)
        s_from = "".join(c for i, c in enumerate(s_from) if i not in ds)
    return _einsum_permute_to(s_from, out_sub, op)
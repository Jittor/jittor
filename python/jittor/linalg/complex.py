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
"""Legacy ComplexNumber matrix decompositions and inversion."""
from functools import partial
from ..nn import ComplexNumber
from ._helpers import (
    _complex_to_stack, _conj_transpose, _matmul, _stack_to_complex,
)


def complex_inv(x:ComplexNumber):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).

    TODO: Faster Implementation; Check backward.
    """
    import jittor as jt
    assert isinstance(x, ComplexNumber), "complex_inv is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for complex_inv"

    def forward_code(np, data):

        a = _stack_to_complex(data["inputs"][0])
        m_a = data["outputs"][0]
        t_a = np.linalg.inv(a)
        np.copyto(m_a, _complex_to_stack(t_a))


    def backward_code(np, data):
        T = _conj_transpose
        _dot = _matmul
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
    import jittor as jt
    assert isinstance(x, ComplexNumber), "complex_eig is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for complex_eig"
    def forward_code(np, data):
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


def complex_eigh(x:ComplexNumber):
    r"""
    Hermitian eigendecomposition of a complex matrix (counterpart of the real
    :func:`eigh`). ``x`` is assumed Hermitian; only the lower triangle is read
    (``UPLO='L'``), matching the real ``eigh``. Returns ``(w, v)`` as
    ``ComplexNumber``\ s for type-consistency with :func:`complex_eig`; the
    eigenvalues ``w`` are mathematically real (carried with a zero imaginary
    part). Forward-only (numpy), like ``complex_eig``/``complex_svd``.

    :param x (...,M,M):
    :return: w (...,M) eigenvalues, v (...,M,M) eigenvectors.
    """
    import jittor as jt
    assert isinstance(x, ComplexNumber), "complex_eigh is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for complex_eigh"
    def forward_code(np, data):
        a = _stack_to_complex(data["inputs"][0])
        w, v = data["outputs"]
        # np.linalg.eigh handles complex Hermitian natively: w real, v complex.
        tw, tv = np.linalg.eigh(a, UPLO='L')
        # carry the (real) eigenvalues as a complex stack (imag = 0) so the
        # ComplexNumber wrapper round-trips cleanly through the P1 bridge.
        np.copyto(w, _complex_to_stack(tw.astype(a.dtype)))
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
    import jittor as jt
    assert isinstance(x, ComplexNumber), "linalg_qr is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for linalg_qr"
    def forward_code(np, data):
        a = _stack_to_complex(data["inputs"][0])
        qr = data["outputs"][0]
        Q, R = np.linalg.qr(a)
        QR = np.stack([Q, R], axis=0)
        np.copyto(qr, _complex_to_stack(QR))

    def backward_code(np, data):
        # reference: https://github.com/tencent-quantum-lab/tensorcircuit/blob/master/tensorcircuit/backends/pytorch_ops.py
        H = _conj_transpose
        def _TriangularSolve(x, r):
            return H(np.linalg.solve(r, H(x)))
        _dot = _matmul
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
    import jittor as jt
    def forward_code(np, data):
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


def complex_pinv(x:ComplexNumber):
    r"""
    Moore-Penrose pseudo-inverse of a complex matrix (counterpart of the real
    :func:`pinv`). For ``x`` of shape ``(...,M,N)`` returns ``(...,N,M)``.
    Forward-only (numpy ``np.linalg.pinv`` handles complex natively), wired
    through the ComplexNumber machinery like ``complex_svd``/``complex_eig``.

    :param x (...,M,N):
    :return: x's pinv (...,N,M).
    """
    import jittor as jt
    assert isinstance(x, ComplexNumber), "complex_pinv is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    def forward_code(np, data):
        a = _stack_to_complex(data["inputs"][0])
        m_a = data["outputs"][0]
        t_a = np.linalg.pinv(a)
        np.copyto(m_a, _complex_to_stack(t_a))

    def backward_code(np, data):
        raise NotImplementedError

    # pinv transposes the last two dims (M,N) -> (N,M); the trailing 2 (re/im) stays.
    sw = list(x.shape[:-2]) + [x.shape[-1], x.shape[-2]] + [2]
    lmx = jt.numpy_code(
        sw,
        x.value.dtype,
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(lmx, is_concat_value=True)

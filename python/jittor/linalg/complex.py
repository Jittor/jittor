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
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from functools import partial
from ..nn import ComplexNumber
from ._helpers import (
    _cn_to_native, _complex_to_stack, _native_to_cn, _stack_to_complex,
)


def complex_inv(x:ComplexNumber):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).

    Computed by the native complex :func:`~jittor.linalg.inv` through the differentiable
    ComplexNumber <-> native complex bridge, so its gradient differentiates
    again (a numpy callback's could not: a second derivative raised).
    """
    import jittor as jt
    if not isinstance(x, ComplexNumber):
        raise TypeError("complex_inv is implemented for nn.ComplexNumber")
    _rn, _in = _jittor_dtype_name(x.real.dtype), _jittor_dtype_name(x.imag.dtype)
    if not (_rn == _in and _rn in ("float32", "float64")):
        raise TypeError("real and imag in ComplexNumber should both be jt.float32 "
                         "(complex64) or both jt.float64 (complex128)")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("only square matrix is supported for complex_inv")
    return _native_to_cn(jt.linalg.inv(_cn_to_native(x)))


def complex_eig(x:ComplexNumber):
    r"""
    calculate the eigenvalues and eigenvectors of x.
    :param x (...,M,M):
    :return:w, v.
    w (...,M) : the eigenvalues.
    v (...,M,M) : normalized eigenvectors.
    """
    import jittor as jt
    if not isinstance(x, ComplexNumber):
        raise TypeError("complex_eig is implemented for nn.ComplexNumber")
    _rn, _in = _jittor_dtype_name(x.real.dtype), _jittor_dtype_name(x.imag.dtype)
    if not (_rn == _in and _rn in ("float32", "float64")):
        raise TypeError("real and imag in ComplexNumber should both be jt.float32 "
                         "(complex64) or both jt.float64 (complex128)")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("only square matrix is supported for complex_eig")
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
    part). The gradient is the native one: losses invariant to the
    per-eigenvector complex phase.

    Computed by the native complex :func:`~jittor.linalg.eigh` through the differentiable
    ComplexNumber <-> native complex bridge, so its gradient differentiates
    again (a numpy callback's could not: a second derivative raised).

    :param x (...,M,M):
    :return: w (...,M) eigenvalues, v (...,M,M) eigenvectors.
    """
    import jittor as jt
    if not isinstance(x, ComplexNumber):
        raise TypeError("complex_eigh is implemented for nn.ComplexNumber")
    _rn, _in = _jittor_dtype_name(x.real.dtype), _jittor_dtype_name(x.imag.dtype)
    if not (_rn == _in and _rn in ("float32", "float64")):
        raise TypeError("real and imag in ComplexNumber should both be jt.float32 "
                         "(complex64) or both jt.float64 (complex128)")
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("only square matrix is supported for complex_eigh")
    w, v = jt.linalg.eigh(_cn_to_native(x))
    return _native_to_cn(w), _native_to_cn(v)


def complex_qr(x):
    r"""
    do the qr factorization of x in the below formula:
    x = QR where Q has orthonormal columns and R is upper-triangular.
    :param x (...,M,N): forward and backward both work for any M, N
        (tall/square M>=N and wide M<N).
    :return: q (...,M,K), r (...,K,N), K=min(M,N).

    Computed by the native complex :func:`~jittor.linalg.qr` through the differentiable
    ComplexNumber <-> native complex bridge, so its gradient differentiates
    again (a numpy callback's could not: a second derivative raised).
    """
    import jittor as jt
    if not isinstance(x, ComplexNumber):
        raise TypeError("linalg_qr is implemented for nn.ComplexNumber")
    _rn, _in = _jittor_dtype_name(x.real.dtype), _jittor_dtype_name(x.imag.dtype)
    if not (_rn == _in and _rn in ("float32", "float64")):
        raise TypeError("real and imag in ComplexNumber should both be jt.float32 "
                         "(complex64) or both jt.float64 (complex128)")
    q, r = jt.linalg.qr(_cn_to_native(x))
    return _native_to_cn(q), _native_to_cn(r)


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
    The gradient is the native reduced SVD's: phase-invariant losses,
    including joint U/V reconstruction losses.

    Computed by the native complex reduced SVD through the differentiable
    ComplexNumber <-> native complex bridge, so its gradient differentiates
    again (a numpy callback's could not: a second derivative raised).
    :param x:
    :return:u,s,v.
    '''
    import jittor as jt
    # A native complex input takes the reduced form, whatever full_matrices.
    u, s, v = jt.linalg.svd(_cn_to_native(x))
    return _native_to_cn(u), _native_to_cn(s), _native_to_cn(v)


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
    if not isinstance(x, ComplexNumber):
        raise TypeError("complex_pinv is implemented for nn.ComplexNumber")
    _rn, _in = _jittor_dtype_name(x.real.dtype), _jittor_dtype_name(x.imag.dtype)
    if not (_rn == _in and _rn in ("float32", "float64")):
        raise TypeError("real and imag in ComplexNumber should both be jt.float32 "
                         "(complex64) or both jt.float64 (complex128)")
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

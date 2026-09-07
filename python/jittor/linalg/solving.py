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
"""Linear solves, inverses, determinants and matrix powers."""
from ._helpers import (
    _cn_to_native, _is_native_complex, _matmul, _native_to_cn, _transpose,
)
from .results import INVEX


def inv(x):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).
    """
    import jittor as jt
    from ..nn import ComplexNumber
    from .complex import complex_inv
    if _is_native_complex(x):
        # native complex64 -> bridge to the ComplexNumber path, return native.
        return _cn_to_native(complex_inv(_native_to_cn(x)))
    if isinstance(x, ComplexNumber):
        return complex_inv(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.inv(a)
        np.copyto(m_a, t_a)

    def backward_code(np, data):
        T = _transpose
        _dot = _matmul
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


def inv_ex(x, *, check_errors=False, out=None):
    r"""
    Compute a matrix inverse and return ``(inverse, info)`` like
    ``torch.linalg.inv_ex``.

    .. warning::
        ``info`` is **always zero**. torch reports a singular input by returning
        ``info > 0`` for the offending matrix and leaving ``check_errors=False``
        callers to build a validity mask from ``info == 0``; Jittor's :func:`inv`
        raises instead, so a singular input never reaches the ``info`` tensor and
        a mask built from it marks every matrix valid. Detect singular inputs by
        catching the exception until non-raising reporting is implemented.
    """
    import jittor as jt
    from .. import _arg_policy
    if not check_errors:
        # check_errors=True happens to be honoured -- jt.linalg.inv raises on a
        # singular input, which is what torch does for that flag.  It is the
        # *default* that is broken: torch promises the caller can keep going and
        # read the failure out of `info`, and here `info` never reports anything.
        _arg_policy.ignored(
            "jittor.linalg.inv_ex", "check_errors", check_errors,
            "info is always 0 -- a singular input raises out of jt.linalg.inv "
            "instead of being reported through info, so an `info == 0` validity "
            "mask is unconditionally all-true (check_errors=True *is* honoured)")
    inverse = inv(x)
    info_shape = tuple(int(s) for s in x.shape[:-2])
    info = jt.zeros(info_shape, dtype="int32")
    if out is not None:
        out_inverse, out_info = out
        out_inverse.assign(inverse)
        out_info.assign(info)
        inverse, info = out_inverse, out_info
    return INVEX(inverse, info)


def pinv(x):
    r"""
    calculate the pseudo-inverse of a x.
    :param x (...,M,N)
    :return: x's pinv (...N,M)
    """
    import jittor as jt
    from ..nn import ComplexNumber
    from .complex import complex_pinv
    if _is_native_complex(x):
        # native complex64 -> bridge to the ComplexNumber path, return native.
        return _cn_to_native(complex_pinv(_native_to_cn(x)))
    if isinstance(x, ComplexNumber):
        # complex pseudo-inverse on the legacy ComplexNumber type. (The real
        # path below cannot take a ComplexNumber — previously this raised.)
        return complex_pinv(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.pinv(a)
        np.copyto(m_a, t_a)

    def backward_code(np, data):
        T = _transpose
        _dot = _matmul
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
    import jittor as jt
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


def det(x):
    r"""
    calculate the determinant of x.
    :param x (...,M,M):
    :return:|x| (...,1)
    """
    import jittor as jt
    def forward_code(np, data):
        a = data["inputs"][0]
        L = data["outputs"][0]
        tL = np.linalg.det(a)
        np.copyto(L, tL)

    def backward_code(np, data):
        T = _transpose
        _dot = _matmul
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
    import jittor as jt
    def forward_code(np, data):
        a = data["inputs"][0]
        sign, m_a = data["outputs"]
        sign_, t_a = np.linalg.slogdet(a)
        np.copyto(m_a, t_a)
        np.copyto(sign, sign_)

    def backward_code(np, data):
        T = _transpose
        _dot = _matmul
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


def solve(a,b):
    r"""
    Solve a linear matrix equation Ax = B.This is done by calculating x = A^-1B.So A must not be singular.
    :param a:(...,M,M)
    :param b:(...,M)
    :return:solution of Ax = b formula.x in the shape of (...M)
    """
    import jittor as jt
    def forward_code(np, data):
        a, b = data["inputs"]
        L = data["outputs"][0]
        ans = np.linalg.solve(a, b)
        np.copyto(L, ans)

    def backward_code1(np, data):
        T = _transpose
        _dot = _matmul
        dout = data["dout"]
        out = data["outputs"][0]
        f_out = data["f_outputs"][0]
        inp = data["inputs"][0]
        updim = lambda x: x if x.ndim == a.ndim else x[..., None]
        t = -_dot(updim(np.linalg.solve(T(inp), dout)), T(updim(f_out)))
        np.copyto(out, t)

    def backward_code2(np, data):
        # gradient wrt b: solve(A,b)=A^-1 b  =>  dL/db = A^-T @ dout.
        # (was a stub writing 0 -> silently zero grad through the RHS, breaking
        #  any training that backprops into b, e.g. differentiable solves / GP.)
        T = _transpose
        dout = data["dout"]
        out = data["outputs"][0]
        a = data["inputs"][0]
        np.copyto(out, np.linalg.solve(T(a), dout))

    l_ans = jt.numpy_code(
        [b.shape],
        [b.dtype],
        [a, b],
        forward_code,
        [backward_code1, backward_code2],
    )
    ans = l_ans[0]
    return ans

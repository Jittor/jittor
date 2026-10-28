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
import numpy as np

from ._helpers import (
    _cuda_gesv, _cuda_trsm, _host_lapack, _host_trsm, _is_batched, _is_cupy,
    _batching_aware, _linalg_errors_raise,
    _cn_to_native, _is_native_complex, _matmul, _native_to_cn, _reconnect,
    _transpose,
)
from .results import INVEX
from jittor._runtime.dispatch import select_kernel


@_batching_aware("single")
def inv(x):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).
    """
    import jittor as jt
    from ..nn import ComplexNumber
    from .complex import complex_inv
    if isinstance(x, ComplexNumber):
        return complex_inv(x)
    # Native complex takes the same path as real: numpy_code carries complex
    # dtypes, and the backward below is written with conjugate transposes.
    # Bridging to the ComplexNumber callback chain instead left its backward
    # an opaque callback, so a second derivative raised.
    @_host_lapack
    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        if _is_cupy(np) and a.ndim == 2 and a.size and a.dtype == m_a.dtype:
            _cuda_gesv(np, a, np.eye(a.shape[0], dtype=a.dtype), m_a)
            return
        with _linalg_errors_raise(np):
            t_a = np.linalg.inv(a)
        np.copyto(m_a, t_a)

    class _Inv(jt.Function):
        # A plain numpy_code op's analytic backward is built as a SECOND,
        # opaque numpy_code call with no backward of its own -- so a second
        # derivative through `inv` raised instead of computing a real answer.
        # Real higher-order AD instead: express the backward *formula* with
        # ordinary differentiable jittor ops (matmul/transpose) and recompute
        # `inv(x)` via a live, non-taped recursive call to this same function.
        # `jt.Function` tapes and stop-grads its own `execute()` inputs/
        # outputs (so a value cached as `self.mx` is permanently detached
        # from x's autograd graph) -- but a value computed by calling
        # `inv(x)` again, closing over the *outer*, still-live `x`, is not,
        # so `_reconnect` reconnects the backward expression to x's real
        # graph. `jt.grad(jt.grad(loss, x), x)` then differentiates straight
        # through it like any composed op. The one-time cost is recomputing
        # the inversion during backward (the cached forward value can't be
        # reused for this) -- the standard trade-off for a recompute-based
        # higher-order-differentiable op.
        def execute(self, xt):
            kernel = select_kernel("linalg_inv", xt)
            if kernel is not None:
                self.mx = kernel(xt)
            else:
                self.mx = jt.numpy_code([xt.shape], [xt.dtype], [xt], forward_code)[0]
            return self.mx

        def grad(self, dout):
            # -A^-H dout A^-H: the conjugate-Wirtinger gradient of A^-1
            # (.conj() is the identity for real dtypes).
            mx = _reconnect(self.mx, inv(x))
            mxH = mx.transpose(-1, -2).conj()
            return -jt.matmul(jt.matmul(mxH, dout), mxH)

    return _Inv()(x)


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
    @_host_lapack
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
    if x.shape[-2] != x.shape[-1]:
        raise ValueError("matrix_power expects square matrices (last two dims equal)")

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
    @_host_lapack
    def forward_code(np, data):
        a = data["inputs"][0]
        L = data["outputs"][0]
        tL = np.linalg.det(a)
        np.copyto(L, tL)

    s = x.shape
    x_s = s[:-2]
    if len(s) == 2:
        x_s.append(1)

    class _Det(jt.Function):
        # See `inv`/`_Inv` above for the general real-higher-order-AD
        # pattern (live recursive call instead of an opaque numpy_code
        # backward-of-backward). `d(det)/dx = det(x) * inv(x)^T` is
        # holomorphic, so the conjugate-Wirtinger gradient is
        # `dout * conj(det(x)) * inv(x)^H`; without the conjugates a complex
        # determinant's gradient was silently wrong (identical for real).
        # Both `det(x)` and `inv(x)` recurse into their own higher-order-
        # differentiable public functions, closing over the live `x`, so this
        # composes to any order.
        def execute(self, xt):
            self.d = jt.numpy_code([x_s], [xt.dtype], [xt], forward_code)[0]
            return self.d

        def grad(self, dout):
            d = _reconnect(self.d, det(x))
            n_d = dout.reshape(list(dout.shape) + [1, 1])
            n_o = d.reshape(list(d.shape) + [1, 1])
            # inv(x) was never cached during forward (det's forward never
            # computes it) -- this recompute is not new, the original numpy
            # backward_code called np.linalg.inv(inp) fresh every time too.
            xinvH = inv(x).transpose(-1, -2).conj()
            return (n_d * n_o.conj() * xinvH).reshape(x.shape)

    return _Det()(x)


def slogdet(x):
    r"""
    calculate the sign and log of the determinant of x.
    :param x (...,M,M):
    :return sign, x's logdet.
    sign array decides the sign of determinant and their values can be -1,0,1.Only Real number now.0 means det is 0 and logdet is -inf.
    logdet in shape (...,1).
    """
    import jittor as jt
    @_host_lapack
    def forward_code(np, data):
        a = data["inputs"][0]
        sign, m_a = data["outputs"]
        sign_, t_a = np.linalg.slogdet(a)
        np.copyto(m_a, t_a)
        np.copyto(sign, sign_)

    s = x.shape
    det_s = s[:-2]
    if len(det_s) == 0:
        det_s.append(1)
    complex_input = _is_native_complex(x)

    class _Slogdet(jt.Function):
        # With l = log det A (holomorphic, dl = tr(A^-1 dA)): logabs = Re l
        # and sign = exp(i Im l). The conjugate-Wirtinger gradient of l is
        #   g_l = g_logabs - i Im(conj(g_sign) sign)
        # and the input gradient g_l * A^-H. For real input sign is +-1 and
        # the second term vanishes, giving the previous g_logabs * A^-T; the
        # previous backward also left out both the conjugate and the sign
        # term, so a complex input's gradient was silently wrong. Written
        # with differentiable ops around live recursive calls (see `inv`).
        def execute(self, xt):
            self.sign, self.logabs = jt.numpy_code(
                [det_s, det_s], [xt.dtype, xt.dtype], [xt], forward_code)
            return self.sign, self.logabs

        def grad(self, gsign, glogabs):
            g = None
            if glogabs is not None:
                g = glogabs.real * (1 + 0j) if complex_input else glogabs
            if complex_input and gsign is not None:
                sign = _reconnect(self.sign, slogdet(x)[0])
                term = -(gsign.conj() * sign).imag * 1j
                g = term if g is None else g + term
            if g is None:
                return jt.zeros_like(x)
            if len(s) == 2:
                g = g.reshape([])
            inv_h = inv(x).transpose(-1, -2).conj()
            return g.unsqueeze(-1).unsqueeze(-1) * inv_h

    sign, mx = _Slogdet()(x)
    return sign, mx


def _solve_rhs_is_vector(a_shape, b_shape):
    """Whether `b` is a (batch of) vector right-hand side(s) for `a`.

    PyTorch's rule: `b` is 1-D, or `b.shape == a.shape[:-1]` exactly. It does
    not depend on which NumPy is installed (1.x reads any `b.ndim ==
    a.ndim - 1` as vectors, 2.x only a 1-D `b`), so a mapped (2,2,2) `a` with
    a shared (2,1) `b` is a batch of 2x1 matrices under every NumPy.
    """
    return len(b_shape) == 1 or (
        len(b_shape) == len(a_shape) - 1 and list(b_shape) == list(a_shape[:-1]))


@_batching_aware("solve")
def solve(a,b):
    r"""
    Solve a linear matrix equation Ax = B.This is done by calculating x = A^-1B.So A must not be singular.
    :param a:(...,M,M)
    :param b:(...,M) or (...,M,K); leading batch dimensions broadcast
    :return:solution of Ax = b formula, in the broadcast shape of b
    """
    import jittor as jt
    vector = _solve_rhs_is_vector(a.shape, b.shape)
    b_mat = b.unsqueeze(-1) if vector else b
    # Broadcast the batch dimensions explicitly, so the core below only ever
    # sees matching (...,M,M) and (...,M,K) operands: the output is allocated
    # at the broadcast shape, and the broadcast op's own backward sums a
    # gradient back over every dimension an operand was broadcast along.
    batch = list(np.broadcast_shapes(tuple(a.shape[:-2]), tuple(b_mat.shape[:-2])))
    a_full = batch + list(a.shape[-2:])
    b_full = batch + list(b_mat.shape[-2:])
    if list(a.shape) != a_full:
        a = a.broadcast(a_full)
    if list(b_mat.shape) != b_full:
        b_mat = b_mat.broadcast(b_full)
    x = _solve_matrix(a, b_mat)
    return x.squeeze(-1) if vector else x


def _solve_matrix(a, b):
    """solve() on (...,M,M) and (...,M,K) operands with identical batch dims."""
    import jittor as jt
    @_host_lapack
    def forward_code(np, data):
        a, b = data["inputs"]
        L = data["outputs"][0]
        if _is_cupy(np) and a.ndim == 2 and b.size and a.dtype == b.dtype == L.dtype:
            _cuda_gesv(np, a, b, L)
            return
        with _linalg_errors_raise(np):
            ans = np.linalg.solve(a, b)
        np.copyto(L, ans)

    def _T(v):
        # conjugate (Hermitian) transpose; .conj() is a no-op for real
        # dtypes, so this also covers the real case unchanged, matching the
        # Wirtinger-conjugate convention used throughout this module.
        return v.transpose(-1, -2).conj()

    class _Solve(jt.Function):
        # See `inv`/`_Inv` above for the general real-higher-order-AD
        # pattern. dL/db = A^-H @ dout = solve(A^H, dout); dL/dA = -db @ x^H
        # (x = solve(A,b)). The operands already share their batch shape and
        # `b` is always a matrix, so both gradients have exactly the input
        # shapes. Both recursive calls close over the live `a`/`b`, so this
        # composes to any order.
        def execute(self, at, bt):
            kernel = select_kernel("linalg_solve", at, bt)
            if kernel is not None:
                self.x = kernel(at, bt)
            else:
                self.x = jt.numpy_code([bt.shape], [bt.dtype], [at, bt], forward_code)[0]
            return self.x

        def grad(self, dout):
            db = _solve_matrix(_T(a), dout)
            x = _reconnect(self.x, _solve_matrix(a, b))
            dA = -jt.matmul(db, _T(x))
            return dA, db

    return _Solve()(a, b)


def _solve_triangular(a, b, upper):
    """``solve(a, b)`` for an ``a`` that is triangular -- ``upper`` or lower,
    the other triangle zero -- as the backward of QR and Cholesky has it.

    A 2-D solve is one ``trsm`` rather than an LU and two triangular solves:
    QR's backward solves against R once per call, 1,368 times in one gradient
    of a 40-site MPS, and on CUDA each of those went through a general solve
    and its status reads. The gradient is ``_solve_matrix``'s, unprojected, so
    every order of derivative is what a general solve gives. A batch, or a
    host without a CBLAS ``trsm``, takes the general solve.
    """
    import jittor as jt
    if _is_batched(a) or _is_batched(b) or a.ndim != 2 or b.ndim != 2:
        return jt.linalg.solve(a, b)

    @_host_lapack
    def forward_code(np, data):
        a, b = data["inputs"]
        x = data["outputs"][0]
        if _is_cupy(np):
            if a.dtype == b.dtype == x.dtype:
                _cuda_trsm(np, a, b, x, upper)
                return
        elif _host_trsm(a, b, x, upper):
            return
        with _linalg_errors_raise(np):
            np.copyto(x, np.linalg.solve(a, b))

    def _T(v):
        return v.transpose(-1, -2).conj()

    class _SolveTriangular(jt.Function):
        # `_Solve`'s backward; A^H is triangular the other way up.
        def execute(self, at, bt):
            kernel = select_kernel("linalg_solve_triangular", at, bt, upper)
            if kernel is not None:
                self.x = kernel(at, bt, upper)
            else:
                self.x = jt.numpy_code([bt.shape], [bt.dtype], [at, bt], forward_code)[0]
            return self.x

        def grad(self, dout):
            db = _solve_triangular(_T(a), dout, not upper)
            x = _reconnect(self.x, _solve_triangular(a, b, upper))
            dA = -jt.matmul(db, _T(x))
            return dA, db

    return _SolveTriangular()(a, b)

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
import numpy as np
from functools import partial
from . import _arg_policy
from .nn import ComplexNumber


# ---------------------------------------------------------------------------
# Helpers shared by the numpy_code bodies below.
#
# Each of these used to be re-declared inside every forward_code/backward_code
# in this file -- the batched transpose 12 times, the einsum matmul 11 times,
# and the complex<->stack pair 8 times each. They were byte-identical copies,
# so a fix to one of them reached only that one op.
#
# The bodies take ``np`` as a parameter (numpy is injected by numpy_code),
# which is why these lived inside; defining them here needs numpy at module
# scope, which is what the import above adds. The injected ``np`` is the same
# numpy module, so behaviour is unchanged.
# ---------------------------------------------------------------------------
def _transpose(x):
    """Batched transpose: swap the last two axes."""
    return np.swapaxes(x, -1, -2)


def _conj_transpose(x):
    """Batched conjugate transpose (the Hermitian adjoint)."""
    return np.conj(np.swapaxes(x, -1, -2))


def _matmul(a, b):
    """Batched matrix product over the last two axes."""
    return np.einsum('...ij,...jk->...ik', a, b)


def _stack_to_complex(x):
    """float ``[..., 2]`` (real, imag) stack -> complex array."""
    return x[..., 0] + 1j * x[..., 1]


def _complex_to_stack(x):
    """complex array -> float ``[..., 2]`` (real, imag) stack."""
    return np.stack([np.real(x), np.imag(x)], axis=-1)


# ---------------------------------------------------------------------------
# Native complex64 <-> nn.ComplexNumber bridge for the complex linalg ops
# (Phase 6 "P4" of the ComplexNumber deprecation). The complex math itself
# still lives in the ComplexNumber code paths below (complex_inv / complex_eig
# / complex_qr / complex_svd / complex_eigh / complex_pinv); these helpers only
# move a *native* complex64 Var across the P1 bridge (jt.nn.view_as_real /
# jt.nn._real2_to_complex64) so every public entry point can ALSO take a native
# complex64 input and return native complex64 output(s). No linear-algebra math
# is reimplemented here.
# ---------------------------------------------------------------------------
def _is_native_complex(x):
    # A native complex64 Var (NOT an nn.ComplexNumber, which is a plain object).
    return isinstance(x, jt.Var) and "complex" in str(x.dtype)


def _native_to_cn(z):
    # native complex64 [...]  ->  nn.ComplexNumber (differentiable, via P1 bridge)
    return ComplexNumber(jt.nn.view_as_real(z), is_concat_value=True)


def _cn_to_native(cn):
    # nn.ComplexNumber  ->  native complex64 [...]  (differentiable, via P1 bridge)
    # cn.value is the float32 [..., 2] stack; _real2_to_complex64 rebuilds complex64.
    return jt.nn._real2_to_complex64(cn.value)


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

import collections as _collections
# torch.linalg.svd / torch.svd both return a named (U, S, Vh) result. We make
# jittor's svd return one too: it still unpacks as a plain 3-tuple `u, s, v` (so
# every existing `u, s, v = svd(x)` / `_, s, _ = svd(x)` caller is untouched), but
# it also exposes `.U`, `.S`, `.Vh` for torch-grade attribute access.
SVD = _collections.namedtuple("svd", ["U", "S", "Vh"])
INVEX = _collections.namedtuple("inv_ex", ["inverse", "info"])


def _svd_reduced(x):
    r'''
    Reduced (a.k.a. "thin"/"economy") SVD: A = U @ diag(S) @ Vh with
    U:(...,M,K), S:(...,K), Vh:(...,K,N), K=min(M,N). This is torch's
    ``full_matrices=False`` form. Differentiable (numpy forward + analytic
    backward); returns the raw ``(u, s, v)`` tuple.
    '''
    def forward_code(np, data):
        a = data["inputs"][0]
        u, s, v = data["outputs"]
        #TODO:remove copyto
        tu, ts, tv = np.linalg.svd(a, full_matrices=0)
        np.copyto(u, tu)
        np.copyto(s, ts)
        np.copyto(v, tv)

    def backward_code(np, data):
        T = _transpose
        _dot = _matmul
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        out_index = data["out_index"]
        u, s, v = data["f_outputs"]
        v = T(v)
        m, n = inp.shape[-2:]
        k = min(m, n)
        i = np.reshape(np.eye(k), (1,) * (inp.ndim - 2) + (k, k))
        if out_index == 0:
            f = 1 / (s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2 + i)
            gu = dout
            utgu = _dot(T(u), gu)
            t = (f * (utgu - T(utgu))) * s[..., np.newaxis, :]
            t = _dot(_dot(u, t), T(v))
            if m > n:
                i_minus_uut = (np.reshape(np.eye(m), (1,) * (inp.ndim - 2) + (m, m)) -
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
            # `v` is the (...,n,k) form (transposed above); the upstream grad
            # `gv` is wrt the (...,k,n) output, i.e. the (n,k)-form grad is T(gv).
            # The antisymmetric inner term must contract the n (range) axis:
            #   V^T (gV) = T(v) @ T(gv)   -- mirrors the U branch's T(u) @ gu.
            # The old `_dot(T(v), gv)` contracted the wrong axis (only shape-
            # conformable for square v, where it was silently wrong, not a crash).
            vtgv = _dot(T(v), T(gv))
            t = s[..., :, np.newaxis] * (f * (vtgv - T(vtgv)))
            t = _dot(_dot(u, t), T(v))
            if m < n:
                i_minus_vvt = (np.reshape(np.eye(n), (1,) * (inp.ndim - 2) + (n, n)) -
                               _dot(v, np.conj(T(v))))
                # extra (range-complement) term, mirror of the m>n U branch:
                #   U S^-1 (gV)^T (I - V V^T) = (u/s) @ gv @ (I - v v^T)
                # old code used T(gv) and an outer T(), giving a (m,k)·(n,k)
                # einsum that crashed for m<n.
                t = t + _dot(_dot(u / s[..., np.newaxis, :], gv), i_minus_vvt)
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


def _svd_full(x):
    r'''
    Full SVD: A = U @ diag(S) @ Vh with U:(...,M,M), S:(...,K), Vh:(...,N,N),
    K=min(M,N). This is torch's ``full_matrices=True`` form for non-square A
    (for square A the reduced form already has these shapes, so the caller uses
    the differentiable reduced path instead). The extra (range-complement)
    columns of U / rows of Vh have no well-defined gradient, so this path is a
    numpy forward only (no backward) — matching the project's torch_shim, which
    likewise falls back to numpy for full non-square SVD. Use ``full_matrices=
    False`` (or :func:`svdvals`) when you need gradients.
    '''
    def forward_code(np, data):
        a = data["inputs"][0]
        u, s, v = data["outputs"]
        tu, ts, tv = np.linalg.svd(a, full_matrices=1)
        np.copyto(u, tu)
        np.copyto(s, ts)
        np.copyto(v, tv)

    m, n = x.shape[-2:]
    k = min(m, n)
    su = list(x.shape[:-2]) + [m, m]
    sv = list(x.shape[:-2]) + [n, n]
    ss = list(x.shape[:-2]) + [k]
    u, s, v = jt.numpy_code(
        [su, ss, sv],
        [x.dtype, x.dtype, x.dtype],
        [x],
        forward_code,
    )
    return u, s, v


def svd(x, full_matrices=False, *, compute_uv=True, driver=None):
    r'''
    Singular Value Decomposition: ``A = U @ diag(S) @ Vh``. Returns the same
    named ``(U, S, Vh)`` result as ``torch.linalg.svd`` (and it also unpacks as
    a plain 3-tuple ``u, s, v``, preserving every existing jittor caller).

    For ``A`` of shape ``(...,M,N)`` with ``K = min(M, N)``:

    - ``full_matrices=False`` (default, reduced / "thin"): ``U`` is ``(...,M,K)``,
      ``Vh`` is ``(...,K,N)``, ``S`` is ``(...,K)``.
    - ``full_matrices=True``: ``U`` is ``(...,M,M)``, ``Vh`` is ``(...,N,N)``,
      ``S`` is ``(...,K)``.

    .. note::
        ``torch.linalg.svd`` defaults to ``full_matrices=True``; this jittor-
        native entry point keeps the historical reduced default so that the
        differentiable path and all jittor callers (``matrix_rank``/``cond``/
        ``matrix_norm``/the native ``test_linalg`` suite) are unchanged. Pass
        ``full_matrices=True`` explicitly for torch's full shapes. (The torch-
        facing ``torch.linalg.svd`` default is meant to be supplied at the
        torch-compat boundary.)

    ``S`` is sorted in descending order. The reduced form (and the square case,
    where reduced == full) is differentiable; the full form on a *non-square*
    matrix is computed via numpy without a gradient on ``U``/``Vh`` (the extra
    orthogonal-complement columns/rows have no unique gradient) — use
    ``full_matrices=False`` or :func:`svdvals` when gradients are needed.

    :param x: ``(...,M,N)`` real matrix (or ``nn.ComplexNumber``).
    :param full_matrices (bool): see above. Default ``False`` (reduced).
    :param compute_uv (bool): if ``False``, only ``S`` is meaningful (``U`` and
        ``Vh`` are still returned for shape compatibility but may be skipped).
    :param driver: accepted for torch signature compatibility (ignored).
    :return: named tuple ``SVD(U, S, Vh)``.
    '''
    if not compute_uv:
        _arg_policy.ignored(
            "jittor.linalg.svd", "compute_uv", compute_uv,
            "U and Vh are computed and returned anyway, so none of the work the "
            "flag asks to skip is skipped (S is correct either way; use "
            "jt.linalg.svdvals to actually skip it)")
    if driver is not None:
        _arg_policy.ignored(
            "jittor.linalg.svd", "driver", driver,
            "the decomposition always goes through numpy/cupy's default driver")
    if _is_native_complex(x):
        # native complex64 -> bridge to the ComplexNumber path, return native.
        u, s, v = complex_svd(_native_to_cn(x))
        # s is real (singular values) but complex_svd carries it as a
        # ComplexNumber (imag=0); _cn_to_native keeps it complex64 for a
        # uniform native-complex return (callers reconstruct via u@diag(s)@v).
        return SVD(_cn_to_native(u), _cn_to_native(s), _cn_to_native(v))
    if isinstance(x, ComplexNumber):
        # complex_svd is the reduced form; full_matrices for complex is not
        # supported (would need a complex orthogonal completion).
        u, s, v = complex_svd(x)
        return SVD(u, s, v)
    m, n = x.shape[-2:]
    if (not full_matrices) or m == n:
        u, s, v = _svd_reduced(x)
    else:
        u, s, v = _svd_full(x)
    return SVD(u, s, v)


def svdvals(x, *, driver=None):
    r'''
    Singular values only, matching ``torch.linalg.svdvals``. Returns the
    ``(...,K)`` tensor ``S`` (``K = min(M, N)``) in descending order. This uses
    the reduced differentiable path, so ``S`` carries a gradient.

    :param x: ``(...,M,N)`` real matrix.
    :param driver: accepted for torch signature compatibility (ignored).
    :return: singular values ``S`` ``(...,K)``.
    '''
    if driver is not None:
        _arg_policy.ignored(
            "jittor.linalg.svdvals", "driver", driver,
            "the decomposition always goes through numpy/cupy's default driver")
    if _is_native_complex(x):
        return _cn_to_native(complex_svd(_native_to_cn(x))[1])
    if isinstance(x, ComplexNumber):
        return complex_svd(x)[1]
    return _svd_reduced(x)[1]

def eig(x):
    r"""
    calculate the eigenvalues and eigenvectors of x.
    :param x (...,M,M):
    :return (ComplexNumber):w, v.
    w (...,M) : the eigenvalues.
    v (...,M,M) : normalized eigenvectors.
    """
    if _is_native_complex(x):
        # native complex64 -> bridge to the ComplexNumber path, return native.
        w, v = complex_eig(_native_to_cn(x))
        return _cn_to_native(w), _cn_to_native(v)
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

    .. note::
        Eigenvectors are only defined up to a per-column sign (and, for repeated
        eigenvalues, up to a rotation within the eigenspace), and this function
        does **not** normalize that choice. It is computed by LAPACK on the host
        and by cuSOLVER under ``jt.flags.use_cuda`` -- ``jt.numpy_code`` hands
        its callback ``cupy`` instead of ``numpy`` when CUDA is on -- and the two
        do not agree on the signs. ``w``, ``v @ diag(w) @ v.T`` and ``v.T @ v``
        are the same on both; individual columns of ``v`` may differ in sign.

        The gradient follows the same rule: it is the correct gradient of the
        ``v`` that *this* device returned, so a loss that is not invariant to the
        sign convention (``(v * seed).sum()``, say) has a device-dependent
        gradient. Prefer a sign-invariant formulation. Same caveat as
        ``torch.linalg.eigh``.
    """
    if _is_native_complex(x):
        # native complex64 Hermitian -> bridge to the ComplexNumber path. The
        # eigenvalues are real (returned as complex64 with imag~0 for a uniform
        # native-complex return); eigenvectors are native complex64.
        w, v = complex_eigh(_native_to_cn(x))
        return _cn_to_native(w), _cn_to_native(v)
    if isinstance(x, ComplexNumber):
        # Hermitian eigendecomposition on the legacy ComplexNumber type. (The
        # real path below cannot take a ComplexNumber — previously this raised.)
        return complex_eigh(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        w, v = data["outputs"]
        tw, tv = np.linalg.eigh(a, UPLO='L')
        np.copyto(w, tw)
        np.copyto(v, tv)

    def backward_code(np, data):
        T = _transpose
        _dot = _matmul
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
            else:
                # ``out`` is a freshly allocated, *uninitialized* buffer: a
                # zero eigenvector gradient still has to be written, otherwise
                # recycled memory is returned as the gradient.  Same reason
                # slogdet's out_index == 0 branch does an explicit copyto(0).
                np.copyto(out, 0)

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


def eigvalsh(x, UPLO='L'):
    r"""
    Eigenvalues of a symmetric / Hermitian matrix, matching
    ``torch.linalg.eigvalsh``. Returns only the eigenvalues ``w`` of shape
    ``(...,M)`` in **ascending** order (the eigenvectors are discarded).

    This reuses the differentiable :func:`eigh`, so ``w`` carries a gradient.
    Like ``torch.linalg.eigvalsh`` / ``numpy.linalg.eigvalsh`` the matrix is
    assumed symmetric/Hermitian and only one triangle is referenced; jittor's
    eigensolver reads the lower (``UPLO='L'``) triangle. For a genuinely
    symmetric input ``UPLO='U'`` yields the same eigenvalues; when ``'U'`` is
    requested the upper triangle is mirrored down so the contract still holds.

    :param x: ``(...,M,M)`` symmetric/Hermitian real matrix.
    :param UPLO ({'L','U'}): which triangle defines the matrix. Default ``'L'``.
    :return: ascending eigenvalues ``w`` ``(...,M)``.
    """
    if UPLO not in ('L', 'U'):
        raise ValueError(f"eigvalsh: UPLO must be 'L' or 'U', got {UPLO!r}")
    if UPLO == 'U':
        # jittor's eigh references the LOWER triangle. To honour UPLO='U', build
        # the full symmetric matrix from x's upper triangle: the upper part
        # (incl. diagonal) plus the strict-upper part reflected below the
        # diagonal. For an already-symmetric input this is a no-op; it only
        # matters when the two triangles disagree.
        up = jt.triu(x, 0)                        # upper triangle incl. diagonal
        x = up + jt.triu(x, 1).transpose(-1, -2)  # mirror strict-upper -> lower
    w, _ = eigh(x)
    return w


def inv(x):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).
    """
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
        # reduced svd keeps S differentiable and is shape-agnostic; the rank
        # itself is non-differentiable (an integer count), but other callers of
        # the returned S (matrix_norm/cond) rely on the gradient.
        _, s, _ = svd(x, full_matrices=False)

    smax = s.max(dim=-1)
    # torch_compat gives max(dim=...) the torch return type (values, indices),
    # while this native linalg helper needs only the values tensor.
    if hasattr(smax, "values"):
        smax = smax.values
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
    # reduced form -> differentiable singular values (nuc/2/-2 matrix norms)
    _, s, _ = svd(xt, full_matrices=False)
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
        # reduced form -> differentiable singular values
        _, s, _ = svd(x, full_matrices=False)
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
        T = _transpose
        _dot = _matmul
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


def qr(x):
    r"""
    do the qr factorization of x in the below formula:
    x = QR where Q is orthogonal matrix and R is upper-triangle matrix.
    :param x (...,M,M):
    :return:q,r as the result of qr factorization.They are both in the shape of (...,M,M).
    """
    if _is_native_complex(x):
        # native complex64 -> bridge to the ComplexNumber path, return native.
        q, r = complex_qr(_native_to_cn(x))
        return _cn_to_native(q), _cn_to_native(r)
    if isinstance(x, ComplexNumber):
        return complex_qr(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        q, r = data["outputs"]
        Q, R = np.linalg.qr(a)
        np.copyto(q,Q)
        np.copyto(r,R)

    def backward_code(np, data):
        # Reduced-QR backward (m>=n). A=QR, Q:(...,m,k), R:(...,k,n), k=min(m,n).
        # Standard form (mirrors torch): with M = R gR^T - gQ^T Q,
        #   gA = (gQ + Q copyltu(M)) R^{-T},  copyltu(X)=tril(X)+tril(X,-1)^T.
        # jittor calls this once per output, so out_index selects the gQ-only /
        # gR-only contribution (the total is linear in (gQ,gR), summed by autodiff).
        # The OLD code assumed square R (output shapes were both x.shape) and the
        # Q term lived entirely in span(Q) — wrong/crash for tall m>n. R was even
        # allocated (m,n) instead of (k,n).
        T = _transpose
        _dot = _matmul
        dout = data["dout"]
        out = data["outputs"][0]
        q, r = data["f_outputs"]
        out_index = data["out_index"]
        m = q.shape[-2]; n = r.shape[-1]
        if m < n:
            raise NotImplementedError(
                "qr backward is only implemented for tall/square inputs (m>=n); "
                f"got m={m} < n={n}. Forward works for all shapes.")
        def copyltu(X):
            return np.tril(X) + T(np.tril(X, -1))
        def rinvT(X):           # X @ R^{-T}
            return T(np.linalg.solve(r, T(X)))
        if out_index == 0:      # contribution from gQ (gR=0)
            gQ = dout
            M = -_dot(T(gQ), q)
            np.copyto(out, rinvT(gQ + _dot(q, copyltu(M))))
        else:                   # contribution from gR (gQ=0)
            gR = dout
            M = _dot(r, T(gR))
            np.copyto(out, rinvT(_dot(q, copyltu(M))))

    m, n = x.shape[-2:]
    k = min(m, n)
    sq = list(x.shape[:-2]) + [m, k]
    sr = list(x.shape[:-2]) + [k, n]
    q, r = jt.numpy_code(
        [sq, sr],
        [x.dtype, x.dtype],
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

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
"""Real and native-complex matrix factorizations."""
from ._helpers import (
    _cn_to_native, _is_native_complex, _matmul, _native_to_cn, _transpose,
)
from .results import SVD


def _svd_reduced(x):
    r'''
    Reduced (a.k.a. "thin"/"economy") SVD: A = U @ diag(S) @ Vh with
    U:(...,M,K), S:(...,K), Vh:(...,K,N), K=min(M,N). This is torch's
    ``full_matrices=False`` form. Differentiable (numpy forward + analytic
    backward); returns the raw ``(u, s, v)`` tuple.
    '''
    import jittor as jt
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
    import jittor as jt
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
    from .. import _arg_policy
    from ..nn import ComplexNumber
    from .complex import complex_svd
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
    from .. import _arg_policy
    from ..nn import ComplexNumber
    from .complex import complex_svd
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
    from ..nn import ComplexNumber
    from .complex import complex_eig
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
    import jittor as jt
    from ..nn import ComplexNumber
    from .complex import complex_eigh
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
    import jittor as jt
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


def cholesky(x):
    r"""
    do Cholesky decomposition of x in the form of below formula:
    x = LL^T
    x must be a Hermite and positive-definite matrix. L is a lower-triangular matrix.
    :param x (...,M,M):
    :return: L (...,M,M).
    """
    import jittor as jt
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


def qr(x):
    r"""
    do the qr factorization of x in the below formula:
    x = QR where Q is orthogonal matrix and R is upper-triangle matrix.
    :param x (...,M,M):
    :return:q,r as the result of qr factorization.They are both in the shape of (...,M,M).
    """
    import jittor as jt
    from ..nn import ComplexNumber
    from .complex import complex_qr
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

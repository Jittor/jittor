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
    _batching_aware, _conj_T, _diag_embed, _diag_of, _eye_like, _linalg_errors_raise,
    _cn_to_native, _is_native_complex, _matmul, _native_to_cn, _reconnect,
    _transpose,
)
from .results import SVD


def _svd_reduced(x):
    r'''
    Reduced (a.k.a. "thin"/"economy") SVD: A = U @ diag(S) @ Vh with
    U:(...,M,K), S:(...,K), Vh:(...,K,N), K=min(M,N). This is torch's
    ``full_matrices=False`` form. Real or native complex (S is then carried
    as complex with a zero imaginary part). Differentiable to any order:
    numpy forward, analytic backward written with differentiable ops and a
    live recursive call (see `inv`/`_Inv` in solving.py). Returns the raw
    ``(u, s, v)`` tuple.
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

    complex_input = _is_native_complex(x)
    m, n = x.shape[-2:]
    k = min(m, n)
    s1 = list(x.shape)
    s1[-1] = k
    s2 = list(x.shape)
    s2[-2] = k
    s3 = list(x.shape)[:-2]
    s3.append(k)

    class _SVD(jt.Function):
        # The adjoint the numpy callbacks computed (real: `_svd_reduced`'s,
        # complex: `complex_svd`'s), unchanged. With f_ij = 1/(s_j^2 - s_i^2)
        # off the diagonal:
        #   t  = (f * (U^H gU - h.c.)) S + S (f * (V^H gV - h.c.)) + diag(gS)
        #   gA = U t Vh + (I - U U^H) gU S^-1 Vh [m > k]
        #                + U S^-1 gVh (I - V V^H) [n > k]
        # Complex input adds i*(Im diag(U^H gU) - Im diag(V^H gV))/(2S) to
        # the diagonal of t: U^H dU is only skew-Hermitian, and the per-
        # column phase leaves just that difference determined (derivation in
        # complex.py::complex_svd).
        def execute(self, xt):
            self.u, self.s, self.v = jt.numpy_code(
                [s1, s3, s2], [xt.dtype, xt.dtype, xt.dtype], [xt], forward_code)
            return self.u, self.s, self.v

        def grad(self, gu, gs, gvh):
            u_live, s_live, vh_live = _svd_reduced(x)
            u = _reconnect(self.u, u_live)
            s = _reconnect(self.s, s_live)
            vh = _reconnect(self.v, vh_live)
            s = s.real if complex_input else s
            v = _conj_T(vh)
            eye = _eye_like(s, k)
            s_i = s.unsqueeze(-1)
            s_j = s.unsqueeze(-2)
            f = (1 - eye) / (s_j * s_j - s_i * s_i + eye)
            t = None
            def add(term):
                return term if t is None else t + term
            if gu is not None:
                utgu = jt.matmul(_conj_T(u), gu)
                t = add((f * (utgu - _conj_T(utgu))) * s_j)
                if complex_input:
                    t = t + _diag_embed(_diag_of(utgu).imag / (2 * s), k) * 1j
            if gs is not None:
                t = add(_diag_embed(gs.real if complex_input else gs, k)
                        * ((1 + 0j) if complex_input else 1))
            if gvh is not None:
                vtgv = jt.matmul(_conj_T(v), _conj_T(gvh))
                t = add(s_i * (f * (vtgv - _conj_T(vtgv))))
                if complex_input:
                    t = t - _diag_embed(_diag_of(vtgv).imag / (2 * s), k) * 1j
            if t is None:
                return jt.zeros_like(x)
            dA = jt.matmul(jt.matmul(u, t), vh)
            if gu is not None and m > k:
                projected = gu - jt.matmul(u, jt.matmul(_conj_T(u), gu))
                dA = dA + jt.matmul(projected / s_j, vh)
            if gvh is not None and n > k:
                projected = gvh - jt.matmul(jt.matmul(gvh, v), vh)
                dA = dA + jt.matmul(u / s_j, projected)
            return dA

    u, s, v = _SVD()(x)
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


@_batching_aware("single")
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
        # Reduced form only, as before (a complex full_matrices completion is
        # not implemented). S is real but carried as the input's complex
        # dtype for a uniform native-complex return.
        u, s, v = _svd_reduced(x)
        return SVD(u, s, v)
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


@_batching_aware("single")
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
        return _svd_reduced(x)[1]
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


@_batching_aware("single")
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
    if isinstance(x, ComplexNumber):
        # Hermitian eigendecomposition on the legacy ComplexNumber type. (The
        # real path below cannot take a ComplexNumber — previously this raised.)
        return complex_eigh(x)
    complex_input = _is_native_complex(x)

    def forward_code(np, data):
        a = data["inputs"][0]
        w, v = data["outputs"]
        tw, tv = np.linalg.eigh(a, UPLO='L')
        np.copyto(w, tw)
        np.copyto(v, tv)

    sw = x.shape[:-2] + x.shape[-1:]
    k = x.shape[-1]

    class _Eigh(jt.Function):
        # First-order adjoints unchanged; written with differentiable ops and
        # a live recursive eigh(x) so they differentiate again (see `inv`/
        # `_Inv` in solving.py). With F_ij = 1/(w_j - w_i) off the diagonal:
        #   t = V diag(gw) V^H + V (F * (V^H gV)) V^H.
        # Real input returns t (it assumes a symmetric perturbation). Native
        # complex input -- eigenvalues carried as complex with a zero
        # imaginary part, only their real part is differentiated -- folds t
        # onto the lower triangle UPLO='L' actually reads: each strict-lower
        # entry also contributes through its conjugate mirror, the real
        # diagonal once, the upper never (the ComplexNumber path's adjoint).
        def execute(self, xt):
            self.w, self.v = jt.numpy_code([sw, x.shape], [xt.dtype, xt.dtype], [xt], forward_code)
            return self.w, self.v

        def grad(self, gw, gv):
            w_live, v_live = eigh(x)
            w = _reconnect(self.w, w_live)
            v = _reconnect(self.v, v_live)
            vH = _conj_T(v)
            w_r = w.real if complex_input else w
            t = None
            if gw is not None:
                gw = gw.real if complex_input else gw
                t = jt.matmul(v * gw.unsqueeze(-2), vH)
            if gv is not None:
                eye = _eye_like(w_r, k)
                factors = (1 - eye) / (w_r.unsqueeze(-2) - w_r.unsqueeze(-1) + eye)
                term = jt.matmul(jt.matmul(v, factors * jt.matmul(vH, gv)), vH)
                t = term if t is None else t + term
            if t is None:
                return jt.zeros_like(x)
            if not complex_input:
                return t
            lower = jt.tril(t + _conj_T(t), -1)
            return lower + _diag_embed(_diag_of(t).real, k) * (1 + 0j)

    # jt.Function returns a list for a multi-output execute(); keep the
    # established (w, v) tuple contract.
    w, v = _Eigh()(x)
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


@_batching_aware("single")
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
        with _linalg_errors_raise(np):
            tL = np.linalg.cholesky(a)
        np.copyto(L, tL)

    k = x.shape[-1]

    class _Cholesky(jt.Function):
        # gA = sym(L^-H phi(L^H gL) L^-1), phi = lower triangle with a halved
        # diagonal, sym(X) = (X + X^H)/2 -- the Hermitian form of the adjoint
        # (identical to the previous real formula; the real-only transposes it
        # used gave native complex input a wrong gradient). Written with
        # differentiable ops (triangular inverses via solve) and a live
        # recursive cholesky(x), so it differentiates again (see `inv`).
        def execute(self, xt):
            self.L = jt.numpy_code([xt.shape], [xt.dtype], [xt], forward_code)[0]
            return self.L

        def grad(self, gL):
            L = _reconnect(self.L, cholesky(x))
            LH = _conj_T(L)
            eye = _eye_like(L.real if _is_native_complex(L) else L, k)
            phi = jt.tril(jt.matmul(LH, gL)) * (1 - 0.5 * eye)
            left = jt.linalg.solve(LH, phi)                          # L^-H phi
            gA = _conj_T(jt.linalg.solve(LH, _conj_T(left)))         # ... L^-1
            return (gA + _conj_T(gA)) * 0.5

    return _Cholesky()(x)


@_batching_aware("single")
def qr(x):
    r"""
    do the qr factorization of x in the below formula:
    x = QR where Q has orthonormal columns and R is upper-triangular.
    :param x (...,M,N): forward and backward both work for any M, N (tall/square M>=N and wide M<N).
    :return: q (...,M,K), r (...,K,N), K=min(M,N).
    """
    import jittor as jt
    from ..nn import ComplexNumber
    from .complex import complex_qr
    if isinstance(x, ComplexNumber):
        return complex_qr(x)
    # Native complex takes the real path below with conjugate transposes:
    # its gradient then differentiates again, which the ComplexNumber callback
    # bridge's could not.
    def forward_code(np, data):
        a = data["inputs"][0]
        q, r = data["outputs"]
        Q, R = np.linalg.qr(a)
        np.copyto(q,Q)
        np.copyto(r,R)

    m, n = x.shape[-2:]
    k = min(m, n)
    sq = list(x.shape[:-2]) + [m, k]
    sr = list(x.shape[:-2]) + [k, n]

    def _copyltu(X):
        # tril(X) + tril(X,-1)^H with the real part of the diagonal: LAPACK's
        # R has a real diagonal, so only Re(diag) is determined. For real X
        # this is tril(X) + tril(X,-1)^T.
        lower = jt.tril(X, -1)
        diag = _diag_embed(_diag_of(X).real if _is_native_complex(X) else _diag_of(X), X.shape[-1])
        return lower + _conj_T(lower) + (diag * (1 + 0j) if _is_native_complex(X) else diag)

    def _rinvT(X, r_):
        # X @ r_^{-H} (r_^{-T} for real), expressed via solve so it stays
        # differentiable (this is what makes the whole backward below a real,
        # further-differentiable op graph instead of an opaque numpy_code
        # callback -- see `inv`/`_Inv` in solving.py for the general pattern
        # this and det/solve/qr all share).
        return _conj_T(jt.linalg.solve(r_, _conj_T(X)))

    class _QR(jt.Function):
        # Reduced-QR backward. A=QR, Q:(...,m,k), R:(...,k,n), k=min(m,n).
        #
        # Tall/square (m>=n, k=n, R square mxn->nxn): standard form (mirrors
        # torch), with M = R gR^T - gQ^T Q:
        #   gA = (gQ + Q copyltu(M)) R^{-T},  copyltu(X)=tril(X)+tril(X,-1)^T.
        #
        # Wide (m<n, k=m, Q is square mxm, R=[R1|R2] with R1 mxm upper
        # triangular and R2 mx(n-m) the rest): A=[A1|A2], A1=Q@R1 is itself a
        # square QR pair, A2=Q@R2. Deriving the adjoint from
        # dR1 = Q^T dA1 - Q^T dQ R1, dR2 = Q^T dA2 - Q^T dQ R2 (and matching
        # coefficients of dA1/dA2 in
        #   dL = tr(gQ^T dQ) + tr(gR1^T dR1) + tr(gR2^T dR2)
        #      = tr(gA1^T dA1) + tr(gA2^T dA2))
        # gives gA2 = Q@gR2 directly, and gA1 = the SAME square-QR formula
        # above applied to the pair (Q, R1) with an effective gQ of
        #   G = gQ - Q @ gR2 @ R2^T
        # (the gR1-only part of the coupling stays inside the square formula
        # unchanged; only the R2/gR2 cross term needs folding into G). This
        # reduces to the m>=n formula exactly when n==m (R2, gR2 are empty).
        #
        # Unlike the original numpy_code `backward_code` (which built a
        # SEPARATE opaque, non-further-differentiable op per output,
        # dispatched on `out_index`), a `jt.Function`'s `grad()` receives
        # BOTH outputs' cotangents (gq, gr) at once, either of which may be
        # None if that output wasn't used downstream -- so the two
        # `out_index` branches collapse into ONE combined formula by
        # linearity of the QR vjp (gQ-only and gR-only are just the special
        # cases gr=0 / gq=0). Expressed with ordinary differentiable jittor
        # ops (solve/matmul/transpose/tril) and a live recursive call to
        # `qr(x)`, so `jt.grad(jt.grad(loss, x), x)` differentiates straight
        # through it -- see `inv`/`_Inv` in solving.py for why the live
        # recursive call (rather than the taped/stop-grad'd `self.q`/
        # `self.r`) is what makes this work.
        def execute(self, xt):
            self.q, self.r = jt.numpy_code([sq, sr], [xt.dtype, xt.dtype], [xt], forward_code)
            return self.q, self.r

        def grad(self, gq, gr):
            q_live, r_live = qr(x)
            q = _reconnect(self.q, q_live)
            r = _reconnect(self.r, r_live)
            if gq is None:
                gq = jt.zeros_like(q)
            if gr is None:
                gr = jt.zeros_like(r)
            if m >= n:
                M = jt.matmul(r, _conj_T(gr)) - jt.matmul(_conj_T(gq), q)
                dA = _rinvT(gq + jt.matmul(q, _copyltu(M)), r)
            else:
                r1 = r[..., :, :m]
                r2 = r[..., :, m:]
                gr1 = gr[..., :, :m]
                gr2 = gr[..., :, m:]
                G = gq - jt.matmul(jt.matmul(q, gr2), _conj_T(r2))
                M = jt.matmul(r1, _conj_T(gr1)) - jt.matmul(_conj_T(G), q)
                a1 = _rinvT(G + jt.matmul(q, _copyltu(M)), r1)
                a2 = jt.matmul(q, gr2)
                dA = jt.concat([a1, a2], dim=-1)
            return dA.reshape(x.shape)

    # jt.Function returns a plain list (not a tuple) for a multi-output
    # execute(); convert back to a tuple to preserve qr()'s established
    # (q, r) return contract (e.g. `isinstance(..., tuple)` checks and
    # code doing `q, r = qr(x)` both keep working identically).
    q, r = _QR()(x)
    return q, r

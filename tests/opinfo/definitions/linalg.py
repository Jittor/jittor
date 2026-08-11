# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Linear-algebra OpInfos: vector/matrix products, traces/diagonals, and the
matrix factorizations (inv/det/slogdet/solve/cholesky/qr/svd).

Two flavors of forward oracle live here:

  * Value-pinned ops (dot, outer, mv, inner, trace, diag, diagonal, inv, det,
    slogdet, solve): the numpy ``ref`` returns the same value jittor computes, so
    ``test_ops.py`` compares element-wise. The matrix ops use *well-conditioned*
    inputs (``A = random + k*I``) so the inverse/solve/determinant are stable.

  * Gauge-invariant ops (cholesky, qr, svd): the factorization is unique only up
    to a sign/phase/ordering gauge, so an element-wise compare against numpy's
    factors is meaningless. Following the validated pattern in
    ``test_linalg.py`` / ``test_complex64_linalg.py``, the ``op`` wrapper
    RECONSTRUCTS the original matrix (``L@Lᵀ``, ``Q@R``, ``U@diag(S)@Vh``) and the
    ``ref`` returns the input matrix ``A`` itself — so the forward check is the
    reconstruction identity, and the SAME differentiable wrapper drives gradcheck
    (exercising the factorization backward through the reconstruction).

The factorizations all route through ``jt.numpy_code`` (numpy forward + a single
analytic backward that is not itself differentiable), so they are first-order
differentiable only: ``supports_gradgrad=False``. ``dot/outer/mv/inner/trace/
diag/diagonal`` are pure native-op compositions and keep second derivatives.
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo, skip


# =============================================================================
# Small native-op compositions for the products / trace / diagonal that jittor
# does not expose as a single `jt.*` symbol (they live only in the torch shim).
# Built from native jittor ops so they stay fully differentiable on-device.
# =============================================================================

def _dot(a, b):
    # 1-D · 1-D inner product; jt.matmul's len_b==1 path returns a (1,)-Var.
    return jt.matmul(a, b)


def _mv(a, v):
    # (M,N) @ (N,) -> (M,)
    return jt.matmul(a, v)


def _inner(a, b):
    # numpy np.inner: contract the LAST axis of each. For 2-D operands
    # A:(n,k), B:(m,k) -> (n,m), i.e. A @ Bᵀ.
    return jt.matmul(a, b.transpose(-1, -2))


def _trace(x):
    # sum of the main diagonal over the last two axes (batched-friendly).
    return jt.diagonal(x, 0, -2, -1).sum(-1)


# =============================================================================
# numpy references
# =============================================================================

def dot_ref(a, b):
    return np.atleast_1d(np.dot(a, b))            # jittor has no 0-d scalar


def outer_ref(a, b):
    return np.outer(a, b)


def mv_ref(a, v):
    return np.matmul(a, v)


def inner_ref(a, b):
    return np.inner(a, b)


def trace_ref(x):
    return np.atleast_1d(np.trace(x, axis1=-2, axis2=-1))   # jittor has no 0-d scalar


def diag_ref(x, diagonal=0):
    # jittor's kwarg is `diagonal`; numpy's is `k`.
    return np.diag(x, k=diagonal)


def diagonal_ref(x, offset=0, dim1=0, dim2=1):
    # jittor places the diagonal axis LAST, exactly like np.diagonal.
    return np.diagonal(x, offset=offset, axis1=dim1, axis2=dim2)


def inv_ref(a):
    return np.linalg.inv(a)


def det_ref(a):
    return np.linalg.det(a)


def solve_ref(a, b):
    return np.linalg.solve(a, b)


def slogdet_ref(a):
    sign, logabsdet = np.linalg.slogdet(a)
    return sign, logabsdet


# ---- gauge-invariant: op reconstructs, ref returns the input matrix ----------

def cholesky_recon(x):
    L = jt.linalg.cholesky(x)
    return jt.matmul(L, L.transpose(-1, -2))      # L @ Lᵀ == A


def qr_recon(x):
    q, r = jt.linalg.qr(x)
    return jt.matmul(q, r)                          # Q @ R == A


def svd_recon(x):
    u, s, v = jt.linalg.svd(x)                      # reduced (thin) form
    # rebuild diag(S) on the last two axes, then U @ diag(S) @ Vh == A.
    # s: (...,K) -> (...,K,K) without a numpy round-trip (keeps it differentiable).
    eye = jt.init.eye(s.shape[-1], dtype=s.dtype)
    diag_s = s.unsqueeze(-1) * eye
    return jt.matmul(jt.matmul(u, diag_s), v)


def recon_ref(x, *args, **kwargs):
    # the reconstruction target is the input matrix itself
    return x


# =============================================================================
# sample-input builders  (kept tiny: gradcheck is O(numel) forward passes)
# =============================================================================

def _well_conditioned(*batch, n, dtype, requires_grad, seed):
    """A = random + k*I  -> diagonally dominant, safely invertible/positive det."""
    a = make_tensor(*batch, n, n, dtype=dtype, requires_grad=requires_grad,
                    seed=seed, low=-1.0, high=1.0)
    eye = jt.init.eye(n, dtype=a.dtype)
    if batch:
        eye = eye.broadcast(list(batch) + [n, n])
    return a + (n * 1.0) * eye                       # shift onto the diagonal


def _spd(*batch, n, dtype, requires_grad, seed):
    """SPD matrix A = MᵀM + k·I (well-conditioned, symmetric positive-definite).

    NB: the differentiated leaf is the SYMMETRIC matrix A (so cholesky's gradient
    is checked on a genuinely symmetric input, matching how cholesky is used);
    M is consumed eagerly into the constant A, then A is made a fresh leaf.
    """
    m = make_tensor(*batch, n, n, dtype=dtype, seed=seed, low=-1.0, high=1.0)
    a = jt.matmul(m, m.transpose(-1, -2))
    eye = jt.init.eye(n, dtype=a.dtype)
    if batch:
        eye = eye.broadcast(list(batch) + [n, n])
    a = a + (n * 1.0) * eye
    # detach to a fresh leaf carrying the requires_grad tag for the diff plan
    a = jt.array(a.numpy(), dtype=str(a.dtype))
    if requires_grad:
        try:
            a.requires_grad = True
        except Exception:
            pass
    return a


def sample_dot(op_info, device, dtype, requires_grad):
    return [SampleInput(make_tensor(4, dtype=dtype, requires_grad=requires_grad, seed=600),
                        make_tensor(4, dtype=dtype, requires_grad=requires_grad, seed=601))]


def sample_outer(op_info, device, dtype, requires_grad):
    return [SampleInput(make_tensor(4, dtype=dtype, requires_grad=requires_grad, seed=602),
                        make_tensor(5, dtype=dtype, requires_grad=requires_grad, seed=603))]


def sample_mv(op_info, device, dtype, requires_grad):
    return [SampleInput(make_tensor(4, 5, dtype=dtype, requires_grad=requires_grad, seed=604),
                        make_tensor(5, dtype=dtype, requires_grad=requires_grad, seed=605))]


def sample_inner(op_info, device, dtype, requires_grad):
    # 2-D · 2-D: A:(3,4), B:(2,4) -> (3,2)  (contract the last axis)
    return [SampleInput(make_tensor(3, 4, dtype=dtype, requires_grad=requires_grad, seed=606),
                        make_tensor(2, 4, dtype=dtype, requires_grad=requires_grad, seed=607))]


def sample_trace(op_info, device, dtype, requires_grad):
    return [SampleInput(make_tensor(4, 4, dtype=dtype, requires_grad=requires_grad, seed=608)),
            SampleInput(make_tensor(2, 3, 3, dtype=dtype, requires_grad=requires_grad, seed=609))]


def sample_diag(op_info, device, dtype, requires_grad):
    out = []
    # 1-D -> matrix (offsets 0 / +1 / -1)
    for i, k in enumerate((0, 1, -1)):
        out.append(SampleInput(
            make_tensor(4, dtype=dtype, requires_grad=requires_grad, seed=610 + i),
            diagonal=k))
    # square matrix -> extracted diagonal
    out.append(SampleInput(
        make_tensor(4, 4, dtype=dtype, requires_grad=requires_grad, seed=620),
        diagonal=0))
    return out


def sample_diagonal(op_info, device, dtype, requires_grad):
    out = []
    base = make_tensor(4, 5, dtype=dtype, requires_grad=requires_grad, seed=630)
    for i, off in enumerate((0, 1, -1)):
        out.append(SampleInput(
            make_tensor(4, 5, dtype=dtype, requires_grad=requires_grad, seed=631 + i),
            offset=off, dim1=0, dim2=1))
    # 3-D, explicit (non-default) axis pair
    out.append(SampleInput(
        make_tensor(2, 4, 4, dtype=dtype, requires_grad=requires_grad, seed=640),
        offset=0, dim1=-2, dim2=-1))
    return out


def sample_inv(op_info, device, dtype, requires_grad):
    return [SampleInput(_well_conditioned(n=3, dtype=dtype, requires_grad=requires_grad, seed=650)),
            SampleInput(_well_conditioned(2, n=3, dtype=dtype, requires_grad=requires_grad, seed=651))]


def sample_det(op_info, device, dtype, requires_grad):
    # batched only: jittor det/slogdet emit a (1,)-Var for an UNbatched (M,M)
    # input, which would not line up with numpy's 0-d scalar; a batch axis keeps
    # both shapes equal to the batch shape.
    return [SampleInput(_well_conditioned(2, n=3, dtype=dtype, requires_grad=requires_grad, seed=660)),
            SampleInput(_well_conditioned(2, 2, n=3, dtype=dtype, requires_grad=requires_grad, seed=661))]


def sample_slogdet(op_info, device, dtype, requires_grad):
    return [SampleInput(_well_conditioned(2, n=3, dtype=dtype, requires_grad=requires_grad, seed=670))]


def sample_solve(op_info, device, dtype, requires_grad):
    # A:(...,M,M) well-conditioned, b:(...,M).  Both float -> both differentiated.
    # A:(M,M) well-conditioned, b:(M,K).  Both float -> both differentiated.
    a0 = _well_conditioned(n=3, dtype=dtype, requires_grad=requires_grad, seed=680)
    b0 = make_tensor(3, 1, dtype=dtype, requires_grad=requires_grad, seed=681)
    return [SampleInput(a0, b0)]


def sample_cholesky(op_info, device, dtype, requires_grad):
    return [SampleInput(_spd(n=3, dtype=dtype, requires_grad=requires_grad, seed=690)),
            SampleInput(_spd(2, n=3, dtype=dtype, requires_grad=requires_grad, seed=691))]


def sample_qr(op_info, device, dtype, requires_grad):
    # square / tall (m>=n): the analytic qr backward requires m>=n.
    return [SampleInput(_well_conditioned(n=3, dtype=dtype, requires_grad=requires_grad, seed=700)),
            SampleInput(make_tensor(5, 3, dtype=dtype, requires_grad=requires_grad, seed=701))]


def sample_svd(op_info, device, dtype, requires_grad):
    # square + non-square (reduced/thin form is differentiable for both).
    return [SampleInput(make_tensor(4, 4, dtype=dtype, requires_grad=requires_grad, seed=710)),
            SampleInput(make_tensor(5, 3, dtype=dtype, requires_grad=requires_grad, seed=711)),
            SampleInput(make_tensor(3, 5, dtype=dtype, requires_grad=requires_grad, seed=712))]


# =============================================================================
# op_db
# =============================================================================

op_db = [
    # ---- vector / matrix products (native-op compositions; fully diff'able) ----
    OpInfo("dot",   op=_dot,        ref=dot_ref,   sample_inputs_func=sample_dot),
    OpInfo("outer", op=jt.outer,    ref=outer_ref, sample_inputs_func=sample_outer),
    OpInfo("mv",    op=_mv,         ref=mv_ref,    sample_inputs_func=sample_mv),
    OpInfo("inner", op=_inner,      ref=inner_ref, sample_inputs_func=sample_inner),

    # ---- trace / diagonal extraction ----
    OpInfo("trace",    op=_trace,       ref=trace_ref,     sample_inputs_func=sample_trace),
    OpInfo("diag",     op=jt.diag,      ref=diag_ref,      sample_inputs_func=sample_diag),
    OpInfo("diagonal", op=jt.diagonal,  ref=diagonal_ref,  sample_inputs_func=sample_diagonal),

    # ---- well-conditioned matrix ops (value-pinned vs numpy) ----
    # numpy_code backward is not itself differentiable -> no 2nd derivative.
    OpInfo("inv", op=jt.linalg.inv, ref=inv_ref,
           sample_inputs_func=sample_inv, supports_gradgrad=False),
    OpInfo("det", op=jt.linalg.det, ref=det_ref,
           sample_inputs_func=sample_det, supports_gradgrad=False),
    # slogdet returns (sign, logabsdet); sign is locally constant on a det>0
    # well-conditioned input, so its (zero) gradient gradchecks cleanly.
    OpInfo("slogdet", op=jt.linalg.slogdet, ref=slogdet_ref,
           sample_inputs_func=sample_slogdet, supports_gradgrad=False),
    OpInfo("solve", op=jt.linalg.solve, ref=solve_ref,
           sample_inputs_func=sample_solve, supports_gradgrad=False),

    # ---- gauge-invariant factorizations: op reconstructs A, ref returns A ----
    # forward check is the reconstruction identity; gradcheck drives the
    # factorization backward through the (differentiable) reconstruction.
    # cholesky's input is SYMMETRIC, so the generic gradcheck (which perturbs each
    # element independently) sees a factor-of-2 gauge mismatch vs the symmetrized
    # analytic gradient -- a well-known gradcheck limitation torch handles specially.
    # The cholesky/qr/svd backward is properly verified by the preserved
    # gauge-invariant FD checks in test_linalg.py; skip only the generic gradcheck.
    OpInfo("cholesky", op=cholesky_recon, ref=recon_ref,
           sample_inputs_func=sample_cholesky, supports_gradgrad=False,
           skips=(skip("test_gradcheck", reason="symmetric-input gauge; backward "
                       "covered by test_linalg gauge-invariant FD checks"),)),
    OpInfo("qr", op=qr_recon, ref=recon_ref,
           sample_inputs_func=sample_qr, supports_gradgrad=False),
    OpInfo("svd", op=svd_recon, ref=recon_ref,
           sample_inputs_func=sample_svd, supports_gradgrad=False),
]

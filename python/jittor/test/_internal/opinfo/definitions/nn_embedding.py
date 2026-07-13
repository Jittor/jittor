# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Embedding-lookup OpInfos: ``F.embedding`` (+ its ``padding_idx`` gradient contract).

``F.embedding(input, weight, padding_idx=None, ...)`` is a *gather*: ``out =
weight[input]``. The addressing operand ``input`` is an int64 index tensor and is
**not** differentiable; the gradient flows only into the float ``weight`` matrix
(scatter-add of the cotangent back into the looked-up rows). So, exactly as the
indexing OpInfos do, ``input`` is passed as the (int64) primary ``input`` of the
SampleInput -- the gradcheck driver's ``_diff_plan`` differentiates only the
*floating* Vars in ``[input, *args]``, so it skips the indices and differentiates the
``weight`` Var that is passed positionally as the first arg. The numpy reference is the
plain fancy-index ``weight[input]`` (an INDEPENDENT oracle for the forward).

padding_idx contract (regression-lock for §4 bug 311eedf6)
----------------------------------------------------------
``torch``'s ``nn.Embedding(padding_idx=p)`` / ``F.embedding(.., padding_idx=p)``
*freezes* the padding row: its gradient is zeroed so the pad row never trains. jittor
used to do a bare ``weight[x]`` and let the pad row receive a normal gradient
(silent-wrong vs torch for every NLP model with a pad token). The fix
(``res = res*keep + (res*(1-keep)).stop_grad()``, ``keep = (x != padding_idx)``)
keeps the *forward values* identical while zeroing the backward on padding positions.

That makes the ``padding_idx`` variant **harness-awkward for gradcheck**: the forward
output still depends on the pad row's value, so a finite-difference Jacobian is
*non-zero* on the pad row, whereas jittor's analytical gradient is (correctly) zero
there -- gradcheck would "fail" by construction. The forward, however, is unaffected by
``padding_idx`` (the mask only splits the value across a stop_grad branch), so we still
pin it to the same numpy ref and run ``test_reference``; the backward is declared
``supports_autograd=False`` (skips gradcheck) and the actual gradient contract is
asserted by the standalone :func:`check_padding_idx_grad` below -- a direct, robust
check of "pad row grad == 0, other rows' grad unchanged" that ``test_ops`` cannot
express. (torch likewise excludes ``padding_idx`` from its embedding gradcheck.)
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# ------------------------------------------------------------------- numpy refs

def embedding_ref(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
                  scale_grad_by_freq=False, sparse=False):
    """out = weight[input]. ``padding_idx`` does not change forward VALUES (it only
    zeroes the pad row's gradient), so the forward oracle ignores it. ``max_norm``
    renormalizes rows whose p-norm exceeds the bound before the lookup."""
    w = np.asarray(weight)
    idx = np.asarray(input).astype("int64")
    if max_norm is not None:
        pn = (np.abs(w) ** norm_type).sum(axis=-1, keepdims=True) ** (1.0 / norm_type)
        w = w * (np.minimum(pn, max_norm) / (pn + 1e-12))
    return w[idx]


# --------------------------------------------------------------- sample builders
# The int64 index tensor is the (non-differentiated) ``input``; the float ``weight``
# matrix is passed positionally so the gradcheck driver differentiates ONLY it. weight
# is kept small (<= ~24-32 elements: num_embeddings*embedding_dim) -- gradcheck is
# O(numel) finite-difference forward passes. Indices are in-bounds and deterministic.

def _idx(*shape, num_embeddings, seed):
    """Deterministic int64 index Var with values in [0, num_embeddings)."""
    return make_tensor(*shape, dtype="int64", low=0, high=num_embeddings, seed=seed)


def sample_embedding(op_info, device, dtype, requires_grad):
    """int64 indices (input) + float weight (differentiated positional arg).

    Several index *shapes* (1-D token row, 2-D batch x seq) over a small vocab so the
    scatter-back-into-rows backward is exercised, including REPEATED indices (a row
    looked up twice accumulates two cotangents -- the historically fragile path)."""
    out = []
    # (num_embeddings, embedding_dim, index_shape, seed_idx, seed_w)
    cfgs = [
        (5, 4, (6,),    810, 800),     # 1-D lookup, 5x4=20 weight elems, repeats likely
        (6, 3, (2, 4),  811, 801),     # 2-D (batch, seq), 6x3=18 weight elems
        (4, 4, (3, 2),  812, 802),     # 4x4=16 weight elems
    ]
    for num, dim, ishape, si, sw in cfgs:
        out.append(SampleInput(
            _idx(*ishape, num_embeddings=num, seed=si),
            make_tensor(num, dim, dtype=dtype, requires_grad=requires_grad, seed=sw)))
    return out


def sample_embedding_max_norm(op_info, device, dtype, requires_grad):
    """max_norm path: rows whose L2 (norm_type=2) norm exceeds the bound are scaled
    down before the lookup. Differentiates weight through that renormalization.

    The renorm uses ``minimum(pn, max_norm)``, a kink at ``pn == max_norm``: we pick a
    generous ``max_norm`` (4.0) and a low weight range so every row's norm stays
    strictly under the bound -> the active branch is the smooth ``pn`` side and
    gradcheck never straddles the corner. (It then reduces to a plain lookup, but it
    pins that the max_norm *plumbing* is differentiable and forward-correct.)"""
    num, dim = 5, 4
    return [SampleInput(
        _idx(6, num_embeddings=num, seed=820),
        make_tensor(num, dim, dtype=dtype, low=-1.0, high=1.0,
                    requires_grad=requires_grad, seed=821),
        max_norm=4.0)]


# ---------------------------------------------- standalone padding_idx grad contract
# Not an OpInfo: a direct assertion of the §4 311eedf6 contract that gradcheck cannot
# express (the forward depends on the pad row but its analytical grad is zero -- a
# finite-difference Jacobian disagrees by construction). Call from a regression test.

def check_padding_idx_grad(num_embeddings=5, embedding_dim=3, padding_idx=0,
                           seed=830, atol=1e-6):
    """Assert F.embedding(.., padding_idx=p) zeroes ONLY the pad row's gradient.

    Returns the (pad_row_grad_max_abs, other_rows_grad_match) pair and raises
    AssertionError on violation. Construction:
      * weight: (num_embeddings, embedding_dim) float32 leaf.
      * input: a 1-D int64 index sequence that DOES look up the pad row at least once
        (so a buggy bare-gather backward would put a non-zero grad on it).
      * grad of sum(embedding(input, weight, padding_idx=p)) w.r.t. weight must be 0 on
        row p and equal to the no-padding gather-backward (a plain bincount of lookups)
        on every other row.
    """
    rng = np.random.RandomState(seed)
    w_np = rng.uniform(-1.0, 1.0, size=(num_embeddings, embedding_dim)).astype("float32")
    # ensure the pad row is actually indexed (the only way the bug can manifest)
    ids_np = rng.randint(0, num_embeddings, size=(8,)).astype("int64")
    ids_np[0] = padding_idx
    ids_np[3] = padding_idx
    weight = jt.array(w_np)
    ids = jt.array(ids_np)

    out = F.embedding(ids, weight, padding_idx=padding_idx)
    gw = jt.grad(out.sum(), weight)
    gw_np = gw.numpy()

    # expected gradient WITHOUT the freeze: each row's grad = (count of lookups) * 1
    # (d sum(weight[ids]) / d weight[r] = number of times r appears in ids, per column).
    counts = np.bincount(ids_np, minlength=num_embeddings).astype("float32")
    expected = np.tile(counts.reshape(-1, 1), (1, embedding_dim))
    expected[padding_idx] = 0.0   # the freeze zeroes the pad row

    pad_max = float(np.abs(gw_np[padding_idx]).max())
    assert pad_max <= atol, \
        f"padding_idx row {padding_idx} grad not zeroed (max |g|={pad_max:.3e})"
    other = [r for r in range(num_embeddings) if r != padding_idx]
    assert np.allclose(gw_np[other], expected[other], atol=1e-4), \
        f"non-pad rows' grad changed by the freeze:\n got {gw_np[other]}\n want {expected[other]}"
    return pad_max, True


# --------------------------------------------------------------------- op_db

op_db = [
    # ---- F.embedding: plain lookup (weight differentiated; int64 indices fixed) ----
    # out = weight[input]. The lookup is LINEAR in weight (a gather), so the backward is
    # a scatter-add of the cotangent into the looked-up rows and the 2nd derivative is
    # the trivial zero -> gradgrad supported.
    OpInfo("embedding", op=F.embedding, ref=embedding_ref,
           sample_inputs_func=sample_embedding),

    # ---- F.embedding with max_norm renormalization (still differentiable in weight) ----
    OpInfo("embedding", variant_test_name="max_norm",
           op=F.embedding, ref=embedding_ref,
           sample_inputs_func=sample_embedding_max_norm),

    # ---- F.embedding with padding_idx: FORWARD-ONLY here (gradcheck is not meaningful) ----
    # padding_idx leaves forward values untouched (the mask only routes the pad row's
    # value through a stop_grad branch) so the forward is pinned to the SAME numpy ref
    # and checked. Its backward zeroes the pad row's gradient while a finite-difference
    # Jacobian -- which still sees the pad row in the forward -- would not, so gradcheck
    # would disagree by construction: supports_autograd=False skips it. The actual
    # gradient contract (pad row grad == 0) is asserted by check_padding_idx_grad above.
    OpInfo("embedding", variant_test_name="padding_idx",
           op=F.embedding, ref=embedding_ref,
           sample_inputs_func=lambda oi, dev, dt, rg: [SampleInput(
               _idx(8, num_embeddings=5, seed=840),
               make_tensor(5, 3, dtype=dt, requires_grad=rg, seed=841),
               padding_idx=0)],
           supports_autograd=False),
]

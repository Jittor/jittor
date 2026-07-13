# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Loss-function OpInfos: mse / l1 / smooth_l1 / huber / cross_entropy / nll /
binary_cross_entropy(+logits) / kl_div.

Losses are the highest-value *backward* targets: every training loop differentiates
one, and the audit found most of them forward-only. Each OpInfo here pins the forward
to an INDEPENDENT numpy reference (adapted from the validated ``test_torch_compat_loss``
references) and -- via the generic ``test_ops`` driver -- gradchecks the backward in
float64.

Differentiation contract (see ``test_ops._diff_plan``): the primary ``input`` and any
*floating* positional ``args`` are the differentiated leaves; everything else (int64
class-index / target tensors, python scalars, ``reduction``/``ignore_index``/
``label_smoothing``/``delta`` flags passed as kwargs) is held fixed. So:

  * regression-style losses (mse/l1/smooth_l1/huber/bce/bce_logits/kl_div) take the
    float ``target`` as a positional Var -- it is differentiated too, which is correct
    (the loss is genuinely differentiable w.r.t. the target) and adds free coverage.
  * classification losses (cross_entropy/nll) take an **int64** class-index ``target``
    positionally -- non-floating, so it is held fixed; only ``input`` is differentiated.

Kink note: l1/smooth_l1/huber and bce_with_logits have non-smooth breakpoints (at
|input-target| == 0 / delta / beta, and at logit == 0). Samples keep every element a
comfortable margin away from those breakpoints so the float64 finite-difference step
(eps=1e-6) never straddles a corner. Those ops also carry ``supports_gradgrad=False``:
their second derivative is the derivative of a sign/step mask -- 0 a.e. but undefined at
the kink -- so a 2nd-order gradcheck is not meaningful (torch declares the same gap).
"""
from ._refs import *  # noqa: F401,F403  (make_tensor, SampleInput, refs, np, jt, nn, F)
from ..core import OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo


# ------------------------------------------------------------------- numpy refs

def _reduce_np(loss, reduction):
    """jittor losses return a (1,)-shaped Var for mean/sum (no 0-d scalar); match it."""
    if reduction == "none":
        return loss
    if reduction == "sum":
        return np.atleast_1d(loss.sum())
    return np.atleast_1d(loss.mean())   # "mean"


def mse_loss_ref(input, target, reduction="mean"):
    return _reduce_np((input - target) ** 2, reduction)


def l1_loss_ref(input, target, reduction="mean"):
    return _reduce_np(np.abs(input - target), reduction)


def smooth_l1_loss_ref(input, target, reduction="mean"):
    # torch beta=1 (jittor's smooth_l1_loss hard-codes beta=1).
    d = np.abs(input - target)
    per = np.where(d < 1.0, 0.5 * d * d, d - 0.5)
    return _reduce_np(per, reduction)


def huber_loss_ref(input, target, reduction="mean", delta=1.0):
    d = np.abs(input - target)
    per = np.where(d < delta, 0.5 * d * d, delta * (d - 0.5 * delta))
    return _reduce_np(per, reduction)


def _np_log_softmax(x, axis):
    m = x.max(axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis, keepdims=True))


def cross_entropy_ref(input, target, reduction="mean", ignore_index=-100,
                      label_smoothing=0.0):
    # integer class-index target; (N, C) logits (the only shape we sample here).
    C = input.shape[1]
    logp = _np_log_softmax(input, 1)
    tgt = np.asarray(target).reshape(-1).astype("int64")
    rows = np.arange(input.shape[0])
    nll = -logp[rows, tgt]
    if label_smoothing:
        smooth = -logp.sum(axis=1)
        per = (1.0 - label_smoothing) * nll + (label_smoothing / C) * smooth
    else:
        per = nll
    keep = (tgt != ignore_index).astype(input.dtype)
    per = per * keep
    if reduction == "sum":
        return np.atleast_1d(per.sum())
    if reduction == "none":
        return per
    return np.atleast_1d(per.sum() / max(keep.sum(), 1e-8))


def nll_loss_ref(input, target, ignore_index=-100, reduction="mean"):
    # input are (treated as) log-probabilities, (N, C); target (N,) int64 class index.
    tgt = np.asarray(target).reshape(-1).astype("int64")
    rows = np.arange(input.shape[0])
    per = -input[rows, tgt]
    keep = (tgt != ignore_index).astype(input.dtype)
    per = per * keep
    if reduction == "sum":
        return np.atleast_1d(per.sum())
    if reduction == "none":
        return per
    return np.atleast_1d(per.sum() / max(keep.sum(), 1e-8))


def bce_ref(input, target, reduction="mean"):
    per = -(target * np.log(input) + (1.0 - target) * np.log(1.0 - input))
    return _reduce_np(per, reduction)


def bce_with_logits_ref(input, target, reduction="mean"):
    # numerically stable: max(x,0) - x*y + log(1 + exp(-|x|))
    per = np.maximum(input, 0.0) - input * target + np.log1p(np.exp(-np.abs(input)))
    return _reduce_np(per, reduction)


def kl_div_ref(input, target, reduction="mean", log_target=False):
    # input are log-probabilities; target probabilities (or log-probs if log_target).
    if log_target:
        per = np.exp(target) * (target - input)
    else:
        per = target * (np.log(target) - input)
    if reduction == "batchmean":
        return np.atleast_1d(per.sum() / input.shape[0])
    return _reduce_np(per, reduction)


# --------------------------------------------------------------- sample builders
# Tensors are kept tiny (<= ~24 differentiated elements) -- gradcheck is O(numel)
# forward passes. Targets are built so kinked losses stay clear of their breakpoints.

def _t_i64(values):
    return jt.array(np.asarray(values, dtype="int64"))


def sample_regression(op_info, device, dtype, requires_grad):
    """input & float target for mse/l1 (smooth everywhere these are sampled)."""
    out = []
    shapes = [(6,), (3, 4)]
    for i, s in enumerate(shapes):
        a = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=600 + i)
        b = make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=610 + i)
        out.append(SampleInput(a, b, reduction="none"))
        out.append(SampleInput(
            make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=620 + i),
            make_tensor(*s, dtype=dtype, requires_grad=requires_grad, seed=630 + i),
            reduction="mean"))
    return out


def _offset_pair(shape, base_seed, dtype, requires_grad, offset):
    """input and target = input + offset (offset keeps |input-target| pinned)."""
    a = make_tensor(*shape, dtype=dtype, low=-2.0, high=2.0,
                    requires_grad=requires_grad, seed=base_seed)
    off = np.asarray(offset, dtype="float64").reshape(shape)
    b = a + jt.array(off.astype("float32")).cast(dtype)
    if requires_grad:
        try:
            b.requires_grad = True
        except Exception:
            pass
    return a, b


def sample_kinked(op_info, device, dtype, requires_grad):
    """For l1/smooth_l1/huber: every element a safe margin from the kink(s).

    For 6 elements use offsets {0.3, -0.4, 0.5, -0.6, 0.35, -0.45}: all |d| in
    (0.25, 0.65) -> inside the smooth_l1/huber(delta=1) quadratic region and far
    from 0, and (for the linear-region sweep) we add a second sample with |d| ~ 2.x,
    clear of delta in [0.5, 2.0].
    """
    quad = [0.30, -0.40, 0.50, -0.60, 0.35, -0.45]      # |d| in (0.25, 0.65)
    lin = [2.30, -2.40, 2.50, -2.60, 2.35, -2.45]       # |d| in (2.2, 2.7)
    out = []
    for j, off in enumerate((quad, lin)):
        a, b = _offset_pair((6,), 640 + j, dtype, requires_grad, off)
        out.append(SampleInput(a, b, reduction="none"))
        a2, b2 = _offset_pair((6,), 650 + j, dtype, requires_grad, off)
        out.append(SampleInput(a2, b2, reduction="mean"))
    return out


def sample_huber(op_info, device, dtype, requires_grad):
    """Same as kinked but sweeps delta; offsets chosen clear of each delta."""
    out = []
    # (offset magnitudes, delta) pairs: |d| stays strictly inside one regime of delta.
    cfgs = [
        ([0.30, -0.40, 0.50, -0.60, 0.35, -0.45], 1.0),   # quadratic side, delta=1
        ([2.30, -2.40, 2.50, -2.60, 2.35, -2.45], 1.0),   # linear side, delta=1
        ([0.30, -0.40, 0.50, -0.60, 0.35, -0.45], 2.0),   # quadratic side, delta=2
        ([1.30, -1.40, 1.50, -1.60, 1.35, -1.45], 0.5),   # linear side, delta=0.5
    ]
    for j, (off, delta) in enumerate(cfgs):
        a, b = _offset_pair((6,), 660 + j, dtype, requires_grad, off)
        out.append(SampleInput(a, b, reduction="none", delta=delta))
        a2, b2 = _offset_pair((6,), 670 + j, dtype, requires_grad, off)
        out.append(SampleInput(a2, b2, reduction="mean", delta=delta))
    return out


def sample_cross_entropy(op_info, device, dtype, requires_grad):
    """(N, C) logits + int64 class-index target (non-differentiated)."""
    N, C = 5, 4
    tgt = _t_i64([0, 1, 2, 3, 1])
    out = []
    for red in ("none", "mean", "sum"):
        out.append(SampleInput(
            make_tensor(N, C, dtype=dtype, requires_grad=requires_grad, seed=680),
            tgt, reduction=red))
    # ignore_index (held fixed; index 0 is the historically-mishandled case)
    out.append(SampleInput(
        make_tensor(N, C, dtype=dtype, requires_grad=requires_grad, seed=681),
        tgt, reduction="mean", ignore_index=0))
    return out


def sample_cross_entropy_smoothing(op_info, device, dtype, requires_grad):
    N, C = 5, 4
    tgt = _t_i64([0, 1, 2, 3, 1])
    return [SampleInput(
        make_tensor(N, C, dtype=dtype, requires_grad=requires_grad, seed=682),
        tgt, reduction="mean", label_smoothing=0.1)]


def sample_nll_loss(op_info, device, dtype, requires_grad):
    """(N, C) log-probs (consumed linearly) + int64 target."""
    N, C = 5, 4
    tgt = _t_i64([0, 1, 2, 3, 1])
    out = []
    for red in ("none", "mean", "sum"):
        out.append(SampleInput(
            make_tensor(N, C, dtype=dtype, requires_grad=requires_grad, seed=690),
            tgt, reduction=red))
    out.append(SampleInput(
        make_tensor(N, C, dtype=dtype, requires_grad=requires_grad, seed=691),
        tgt, reduction="mean", ignore_index=0))
    return out


def sample_bce(op_info, device, dtype, requires_grad):
    """input probabilities in (eps, 1-eps); float target in (eps, 1-eps).

    Both are differentiated. Keep them off 0/1 so the log refs (and jittor's
    log-clamps) stay in their smooth interior.
    """
    out = []
    for i, s in enumerate([(6,), (3, 4)]):
        p = make_tensor(*s, dtype=dtype, low=0.1, high=0.9,
                        requires_grad=requires_grad, seed=700 + i)
        y = make_tensor(*s, dtype=dtype, low=0.15, high=0.85,
                        requires_grad=requires_grad, seed=710 + i)
        out.append(SampleInput(p, y, reduction="none"))
        out.append(SampleInput(
            make_tensor(*s, dtype=dtype, low=0.1, high=0.9,
                        requires_grad=requires_grad, seed=720 + i),
            make_tensor(*s, dtype=dtype, low=0.15, high=0.85,
                        requires_grad=requires_grad, seed=730 + i),
            reduction="mean"))
    return out


def sample_bce_with_logits(op_info, device, dtype, requires_grad):
    """Logits kept off 0 (|x| in ~[0.5, 4]) to stay clear of the max(x,0) kink."""
    out = []
    for i, s in enumerate([(6,), (3, 4)]):
        # bias away from 0: shift positive, then targets in [0.15, 0.85].
        x = make_tensor(*s, dtype=dtype, low=0.6, high=4.0,
                        requires_grad=requires_grad, seed=740 + i)
        y = make_tensor(*s, dtype=dtype, low=0.15, high=0.85,
                        requires_grad=requires_grad, seed=750 + i)
        out.append(SampleInput(x, y, reduction="none"))
        x2 = make_tensor(*s, dtype=dtype, low=-4.0, high=-0.6,
                         requires_grad=requires_grad, seed=760 + i)
        y2 = make_tensor(*s, dtype=dtype, low=0.15, high=0.85,
                         requires_grad=requires_grad, seed=770 + i)
        out.append(SampleInput(x2, y2, reduction="mean"))
    return out


def sample_kl_div(op_info, device, dtype, requires_grad):
    """input log-probs (linear); target probabilities in (eps, 1) off 0."""
    out = []
    for i, s in enumerate([(6,), (3, 4)]):
        inp = make_tensor(*s, dtype=dtype, low=-3.0, high=-0.5,
                          requires_grad=requires_grad, seed=780 + i)   # log-probs < 0
        tgt = make_tensor(*s, dtype=dtype, low=0.1, high=0.9,
                          requires_grad=requires_grad, seed=790 + i)
        out.append(SampleInput(inp, tgt, reduction="none"))
        out.append(SampleInput(
            make_tensor(*s, dtype=dtype, low=-3.0, high=-0.5,
                        requires_grad=requires_grad, seed=795 + i),
            make_tensor(*s, dtype=dtype, low=0.1, high=0.9,
                        requires_grad=requires_grad, seed=797 + i),
            reduction="sum"))
    # batchmean (knowledge-distillation default) on the 2-D sample
    out.append(SampleInput(
        make_tensor(3, 4, dtype=dtype, low=-3.0, high=-0.5,
                    requires_grad=requires_grad, seed=798),
        make_tensor(3, 4, dtype=dtype, low=0.1, high=0.9,
                    requires_grad=requires_grad, seed=799),
        reduction="batchmean"))
    return out


# --------------------------------------------------------------------- op_db

op_db = [
    # ---- regression losses (smooth: full backward + gradgrad) ----
    OpInfo("mse_loss", op=F.mse_loss, ref=mse_loss_ref,
           sample_inputs_func=sample_regression),

    # l1 / smooth_l1 / huber: abs / ternary kink -> 2nd derivative is a step (0 a.e.,
    # undefined at the corner). gradcheck (1st order) is meaningful and passes because
    # samples avoid the kink; gradgrad is declared unsupported (torch does the same).
    OpInfo("l1_loss", op=F.l1_loss, ref=l1_loss_ref,
           sample_inputs_func=sample_kinked, supports_gradgrad=False),
    OpInfo("smooth_l1_loss", op=F.smooth_l1_loss, ref=smooth_l1_loss_ref,
           sample_inputs_func=sample_kinked, supports_gradgrad=False),
    OpInfo("huber_loss", op=F.huber_loss, ref=huber_loss_ref,
           sample_inputs_func=sample_huber, supports_gradgrad=False),

    # ---- classification losses (int64 target held fixed; only logits differentiated) ----
    OpInfo("cross_entropy", op=F.cross_entropy, ref=cross_entropy_ref,
           sample_inputs_func=sample_cross_entropy),
    # label_smoothing path goes through gather backward; gradgrad not guaranteed there.
    OpInfo("cross_entropy", variant_test_name="label_smoothing",
           op=F.cross_entropy, ref=cross_entropy_ref,
           sample_inputs_func=sample_cross_entropy_smoothing, supports_gradgrad=False),
    # nll consumes log-probs linearly (2nd deriv 0); fancy-index backward -> no gradgrad.
    OpInfo("nll_loss", op=F.nll_loss, ref=nll_loss_ref,
           sample_inputs_func=sample_nll_loss, supports_gradgrad=False),

    # ---- binary / divergence losses ----
    OpInfo("binary_cross_entropy", op=F.binary_cross_entropy, ref=bce_ref,
           sample_inputs_func=sample_bce),
    # max(x,0) kink in the stable formula -> 1st-order only (samples avoid logit==0).
    OpInfo("binary_cross_entropy_with_logits", op=F.binary_cross_entropy_with_logits,
           ref=bce_with_logits_ref, sample_inputs_func=sample_bce_with_logits,
           supports_gradgrad=False),
    OpInfo("kl_div", op=F.kl_div, ref=kl_div_ref, sample_inputs_func=sample_kl_div),
]

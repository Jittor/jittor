"""Functional loss implementations exposed through :mod:`jittor.nn`."""

import jittor as jt

from .vector import cosine_similarity

#: the reductions every torch loss accepts
_REDUCTIONS = ("none", "mean", "sum")


def _check_reduction(reduction, allowed=_REDUCTIONS):
    """Reject an unrecognised ``reduction``.

    Each loss used to spell its own reduction dispatch, and they disagreed
    about what an unknown value means: ``cross_entropy_loss`` fell through to
    the per-element branch (silently ``'none'``), ``l1_loss`` fell through to
    ``mean``, and ``mse_loss`` handed the string to ``Var.reduce`` and got
    jittor's internal "no such reduce". Two of the three returned a number.
    """
    if reduction not in allowed:
        raise ValueError(
            "reduction must be one of %s, got %r"
            % (list(allowed), reduction))


def _reduce(loss, reduction, allowed=_REDUCTIONS):
    """Apply a torch-style ``reduction`` to a per-element loss."""
    _check_reduction(reduction, allowed)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def _legacy_reduction(reduction, size_average, reduce):
    """Translate torch's deprecated ``size_average``/``reduce`` pair.

    Follows ``torch.nn._reduction.legacy_get_string``: both default to *True*
    when left as ``None``. The copies of this translation in this module read
    ``size_average=None`` as false and answered ``'sum'`` where torch says
    ``'mean'``.
    """
    if size_average is None and reduce is None:
        return reduction
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if not reduce:
        return "none"
    return "mean" if size_average else "sum"


def cross_entropy_loss(output, target, weight=None, ignore_index=None,reduction='mean'):
    target_shape = target.shape
    if len(output.shape) == 4:
        c_dim = output.shape[1]
        output = output.transpose((0, 2, 3, 1))
        output = output.reshape((-1, c_dim))

    target = target.reshape((-1, ))
    target_weight = ((target >= 0) & (target < output.shape[1])).float32()
    if weight is not None:
        target_weight = target_weight * weight[target]
    if ignore_index is not None:
        target_weight = jt.ternary(
            target==ignore_index,
            jt.array(0).broadcast(target_weight).type_as(target_weight),
            target_weight
        )

    target = target.broadcast(output, [1])
    target = target.index(1) == target

    output = output - output.max([1], keepdims=True)
    logsum = output.exp().sum(1).log()
    loss = (logsum - (output*target).sum(1)) * target_weight
    _check_reduction(reduction)
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'mean':
        # Torch divides by the sum of selected class weights. That denominator
        # may legitimately be negative; clamping it to a positive epsilon
        # changes both the sign and scale of the result.
        return loss.sum() / target_weight.sum()
    return loss.reshape(target_shape)


# PyTorch and older Jittor applications use this shorter spelling.  The native
# implementation remains the owner; Torch mode may wrap this object for extra
# keyword features such as label smoothing.
cross_entropy = cross_entropy_loss


def mse_loss(output, target, reduction="mean"):
    loss = (output-target).sqr()
    return _reduce(loss, reduction)


def bce_loss(output, target, weight=None, size_average=True):
    loss = - (target * jt.log(jt.maximum(output, 1e-20)) + (1 - target) * jt.log(jt.maximum(1 - output, 1e-20)))

    if weight is not None:
        loss *= weight

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def l1_loss(output, target, reduction="mean"):
    loss = (output-target).abs()
    return _reduce(loss, reduction)


def smooth_l1_loss(y_true, y_pred,reduction="mean"):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.

    Args:
         y_true - ground truth
         y_pred - predictions
         reduction - the mode of cal loss which must be in ['mean','sum','none']
    """
    diff = jt.abs(y_true - y_pred)
    less_than_one = (diff<1.0).float32()
    loss = (less_than_one * 0.5 * diff.sqr()) + (1 - less_than_one) * (diff - 0.5)
    return _reduce(loss, reduction)


def huber_loss(input, target, reduction="mean", delta=1.0):
    """Huber loss with the PyTorch reduction and ``delta`` contract."""
    if delta <= 0:
        raise ValueError("delta must be greater than zero")
    distance = (input - target).abs()
    loss = jt.ternary(
        distance < delta,
        0.5 * distance * distance,
        delta * (distance - 0.5 * delta),
    )
    return _reduce(loss, reduction)


def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None,
                        reduction="mean"):
    """Binary cross entropy for probabilities in ``[0, 1]``."""
    log_input = jt.maximum(jt.log(jt.maximum(input, 1e-44)), -100.0)
    log_inverse = jt.maximum(jt.log(jt.maximum(1.0 - input, 1e-44)), -100.0)
    loss = -(target * log_input + (1.0 - target) * log_inverse)
    if weight is not None:
        loss = loss * weight
    reduction = _legacy_reduction(reduction, size_average, reduce)
    return _reduce(loss, reduction)


def kl_div(input, target, size_average=None, reduce=None, reduction="mean",
           log_target=False):
    """Kullback-Leibler divergence with Torch-compatible reductions."""
    if log_target:
        loss = jt.exp(target) * (target - input)
    else:
        safe_target = jt.maximum(target, 1e-12)
        loss = target * (jt.log(safe_target) - input)
    allowed = _REDUCTIONS + ("batchmean",)
    _check_reduction(reduction, allowed)
    if reduction == "batchmean":
        return loss.sum() / input.shape[0]
    return _reduce(loss, reduction, allowed)


def margin_ranking_loss(input1, input2, target, margin=0.0,
                        size_average=None, reduce=None, reduction="mean"):
    loss = jt.maximum(-target * (input1 - input2) + margin, 0.0)
    reduction = _legacy_reduction(reduction, size_average, reduce)
    return _reduce(loss, reduction)


def cosine_embedding_loss(input1, input2, target, margin=0.0,
                          size_average=None, reduce=None, reduction="mean"):
    cosine = cosine_similarity(input1, input2)
    loss = jt.ternary(target == 1, 1.0 - cosine, jt.maximum(cosine - margin, 0.0))
    reduction = _legacy_reduction(reduction, size_average, reduce)
    return _reduce(loss, reduction)


def gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction="mean"):
    variance = jt.maximum(var, eps)
    loss = 0.5 * (jt.log(variance) + (input - target) ** 2 / variance)
    if full:
        loss = loss + 0.5 * 1.8378770664093453
    return _reduce(loss, reduction)


def nll_loss(output,target,weight=None,ignore_index=-100,reduction='mean'):
    assert output.ndim<=2 and output.ndim>0 and target.ndim==1
    n_classes = output.shape[-1]
    assert weight is None or weight.numel()==n_classes
    assert ignore_index<0 or ignore_index<n_classes
    if weight is None:
        weight = jt.ones((n_classes,))
    # torch ignores the class `ignore_index` (any value >=0 is a valid class id, incl.
    # 0); the default -100 is a sentinel for "ignore nothing". The old `>0` test silently
    # let ignore_index=0 through, so class 0 was still counted. Clone before zeroing so a
    # user-supplied weight Var isn't mutated in place.
    if ignore_index>=0:
        weight = weight.clone()
        weight[ignore_index]=0
    if output.ndim==2:
        index = jt.index((output.shape[0],),dim=0)
        loss = -output[index,target]*weight[target]
    else:
        loss = -output[target[0]]*weight[target[0]]
    _check_reduction(reduction)
    if reduction=="mean":
        total_weight  = weight[target].sum() if output.ndim==2 else weight[target[0]].sum()
        return loss.sum()/total_weight
    if reduction=="sum":
        return loss.sum()
    return loss


def binary_cross_entropy_with_logits(output, target, weight=None, pos_weight=None, size_average=True, reduction=None):
    if not (target.shape == output.shape):
        raise ValueError(f"Target size ({target.shape}) must be the same as output size ({output.shape})")

    # The stable formula below is exact for any FINITE logit, but literal +/-inf
    # (e.g. Grounding-DINO's ContrastiveEmbed masks padding logits with -inf) makes
    # (1-target)*(-inf) and inf-inf produce NaN. Clamp to +/-50 — sigmoid is fully
    # saturated there, so loss/grad for finite logits are unchanged; only inf is removed.
    output = jt.clamp(output, -50.0, 50.0)
    max_val = jt.clamp(-output,min_v=0)
    if pos_weight is not None:
        log_weight = (pos_weight-1)*target + 1
        loss = (1-target)*output+(log_weight*(((-max_val).exp()+(-output - max_val).exp()).log()+max_val))
    else:
        loss = (1-target)*output+max_val+((-max_val).exp()+(-output -max_val).exp()).log()
    if weight is not None:
        loss *=weight

    # torch supports reduction='none'/'sum'/'mean'; the original only had the
    # size_average bool (no per-element 'none'). When reduction is given it wins.
    if reduction is not None:
        return _reduce(loss, reduction)
    return loss.mean() if size_average else loss.sum()

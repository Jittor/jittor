"""Functional loss implementations exposed through :mod:`jittor.nn`."""

from .runtime import jt, preserve_facade_origins


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
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.sum() / jt.maximum(target_weight.sum(), 1e-8)
    else:
        return loss.reshape(target_shape)


def mse_loss(output, target, reduction="mean"):
    loss = (output-target).sqr()
    # reduction='none' returns the per-element loss (torch semantics); Var.reduce only
    # knows mean/sum, so don't forward 'none' to it (it raised "no such reduce").
    return loss if reduction == "none" else loss.reduce(reduction)


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
    if reduction == "none": return loss
    if reduction == "sum": return loss.sum()
    return loss.mean()


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
    if reduction=="mean":
        return loss.mean()
    elif reduction=="sum":
        return loss.sum()
    elif reduction=="none":
        return loss
    else:
        raise ValueError(f'not support {reduction}')


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
    if reduction=="mean":
        total_weight  = weight[target].sum() if output.ndim==2 else weight[target[0]].sum()
        return loss.sum()/total_weight
    elif reduction=="sum":
        return loss.sum()
    elif reduction=="none":
        return loss
    else:
        raise ValueError(f'not support {reduction}')


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
        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f'not support {reduction}')
    if size_average:
        return loss.mean()
    else:
        return loss.sum()


_FACADE_SYMBOLS = (
    cross_entropy_loss, mse_loss, bce_loss, l1_loss, smooth_l1_loss,
    nll_loss, binary_cross_entropy_with_logits,
)
preserve_facade_origins(_FACADE_SYMBOLS)

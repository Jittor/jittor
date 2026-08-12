"""Stateful loss modules exposed through :mod:`jittor.nn`."""

import jittor as jt


class CrossEntropyLoss(jt.Module):
    def __init__(self, weight=None, ignore_index=None, reduction="mean"):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def execute(self, output, target):
        return jt.nn.cross_entropy_loss(
            output,
            target,
            self.weight,
            self.ignore_index,
            reduction=self.reduction,
        )


class MSELoss(jt.Module):
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def execute(self, output, target):
        return jt.nn.mse_loss(output, target, self.reduction)


class BCELoss(jt.Module):
    def __init__(self, weight=None, size_average=True):
        self.weight = weight
        self.size_average = size_average

    def execute(self, output, target):
        return jt.nn.bce_loss(output, target, self.weight, self.size_average)


class L1Loss(jt.Module):
    def __init__(self):
        pass

    def execute(self, output, target):
        return jt.nn.l1_loss(output, target)


class BCEWithLogitsLoss(jt.Module):
    def __init__(
        self,
        weight=None,
        pos_weight=None,
        size_average=True,
        reduction=None,
    ):
        self.pos_weight = pos_weight
        self.weight = weight
        self.size_average = size_average
        self.reduction = reduction

    def execute(self, output, target):
        return jt.nn.binary_cross_entropy_with_logits(
            output,
            target,
            self.weight,
            self.pos_weight,
            self.size_average,
            self.reduction,
        )


class KLDivLoss(jt.Module):
    def __init__(self, reduction: str = "mean", log_target: bool = False):
        self.reduction = reduction
        self.log_target = log_target

    def execute(self, input: jt.Var, target: jt.Var) -> jt.Var:
        if not self.log_target:
            loss_pointwise = target * (target.log() - input)
        else:
            loss_pointwise = target.exp() * (target - input)

        if self.reduction == "mean":
            return loss_pointwise.mean()
        if self.reduction == "batchmean":
            return loss_pointwise.sum() / input.size(0)
        if self.reduction == "sum":
            return loss_pointwise.sum()
        return loss_pointwise


__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CrossEntropyLoss",
    "KLDivLoss",
    "L1Loss",
    "MSELoss",
]

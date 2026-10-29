"""Torch LambdaLR initial learning-rate initialization and resume checks."""
import jittor as jt


def lambda_base_lrs(optimizer, last_epoch):
    groups = optimizer.param_groups
    if last_epoch == -1:
        for group in groups:
            initial_lr = group["lr"]
            if isinstance(initial_lr, jt.Var):
                initial_lr = initial_lr.clone()
            group.setdefault("initial_lr", initial_lr)
    else:
        for index, group in enumerate(groups):
            if "initial_lr" not in group:
                raise KeyError(
                    f"param 'initial_lr' is not specified in param_groups[{index}] "
                    "when resuming an optimizer"
                )
    return [group["initial_lr"] for group in groups]

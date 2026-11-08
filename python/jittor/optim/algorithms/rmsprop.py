"""RMSprop optimizer."""

import jittor as jt

from ..base import (
    Optimizer, _grad_matches_param, _normalize_param_group, _param_requires_grad,
    _adaptive_state_buffer, _update_preserve_dtype,
)
from ..._core.dtypes import dtype_name

class RMSprop(Optimizer):
    """ RMSprop Optimizer.
    Args:
        params(list): parameters of model.
        lr(float): learning rate.
        eps(float): term added to the denominator to avoid division by zero, default 1e-8.
        alpha(float): smoothing constant, default 0.99.
        weight_decay(float): coupled L2 penalty, default 0.

    Example:
        optimizer = nn.RMSprop(model.parameters(), lr)
        optimizer.step(loss)
    """
    def __init__(self, params, lr=1e-2, eps=1e-8, alpha=0.99, weight_decay=0):
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        super().__init__(params, lr)
        self.eps = eps
        self.alpha = alpha
        self.weight_decay = weight_decay

        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            for p in pg["params"]:
                values.append(_adaptive_state_buffer(p))

    def add_param_group(self, group):
        _normalize_param_group(group)
        values = group["values"] = []
        for p in group["params"]:
            values.append(_adaptive_state_buffer(p))
        super().add_param_group(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            alpha = pg.get("alpha", self.alpha)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            for p, g, v in zip(pg["params"], pg["grads"], pg["values"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                if dtype_name(p.dtype) in ("float16", "bfloat16"):
                    with jt.flag_scope(amp_reg=0):
                        if dtype_name(v.dtype) != "float32":
                            v.update(v.float32())
                        grad = g.float32()
                        _update_preserve_dtype(
                            v, alpha * v + (1-alpha) * grad * grad)
                        _update_preserve_dtype(
                            p, p.float32() - lr * grad / (jt.sqrt(v) + eps))
                else:
                    _update_preserve_dtype(v, alpha * v + (1-alpha) * g * g)
                    _update_preserve_dtype(
                        p, p - lr * g / (jt.sqrt(v) + eps))
        self.post_step()

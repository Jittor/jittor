"""Adamax optimizer (dense tensors, Torch-compatible defaults)."""
from numbers import Real
import jittor as jt

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _state_buffer, _update_preserve_dtype,
)


def validate_adamax_options(options):
    betas = options["betas"]
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("Invalid Adamax betas: %r" % (betas,))
    for name in ("lr", "eps", "weight_decay"):
        value = options[name]
        if isinstance(value, jt.Var) or not isinstance(value, Real):
            raise TypeError("Adamax %s must be a Python number" % name)
    if not 0 <= options["lr"] or not 0 <= options["eps"] or not 0 <= options["weight_decay"]:
        raise ValueError("Invalid Adamax hyperparameters")
    for name in ("foreach", "differentiable", "fused"):
        if options.get(name):
            raise NotImplementedError("Adamax %s=True is not supported" % name)


class Adamax(Optimizer):
    """Torch-compatible Adamax: an Adam-style first moment with an infinity norm."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, foreach=None, *, maximize=False,
                 differentiable=False, fused=None):
        self._adamax_defaults = dict(
            lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
            foreach=foreach, maximize=maximize, differentiable=differentiable,
            fused=fused)
        validate_adamax_options(self._adamax_defaults)
        super().__init__(params, lr)
        for key, value in self._adamax_defaults.items():
            setattr(self, key, value)
        for group in self.param_groups:
            self._initialize_group(group)

    def _initialize_group(self, group):
        for key, value in self._adamax_defaults.items():
            group.setdefault(key, value)
        validate_adamax_options(group)
        group["params"] = list(group["params"])
        group["values"] = [_state_buffer(p) for p in group["params"]]   # exp_avg
        group["inf_norm"] = [_state_buffer(p) for p in group["params"]]
        group["_adamax_steps"] = [0 for _ in group["params"]]

    def add_param_group(self, group):
        group = dict(group)
        params = group["params"]
        group["params"] = [params] if isinstance(params, jt.Var) else list(params)
        existing = {id(p) for pg in self.param_groups for p in pg["params"]}
        if any(id(p) in existing for p in group["params"]):
            raise ValueError("some parameters appear in more than one parameter group")
        self._initialize_group(group)
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        for pg in self.param_groups:
            validate_adamax_options(pg)
            lr = pg["lr"]
            beta1, beta2 = pg["betas"]
            eps = pg["eps"]
            weight_decay = pg["weight_decay"]
            maximize = pg["maximize"]
            for p, g, exp_avg, inf_norm, index in zip(
                    pg["params"], pg.get("grads", ()),
                    pg["values"], pg["inf_norm"],
                    range(len(pg["params"]))):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g):
                    continue
                step = pg["_adamax_steps"][index] + 1
                pg["_adamax_steps"][index] = step
                if maximize:
                    g = -g
                if weight_decay != 0:
                    g = g + p * weight_decay
                _update_preserve_dtype(
                    exp_avg, beta1 * exp_avg + (1 - beta1) * g)
                _update_preserve_dtype(
                    inf_norm, jt.maximum(beta2 * inf_norm, jt.abs(g) + eps))
                denom = inf_norm / (1 - beta1 ** step)
                _update_preserve_dtype(p, p - (lr / (denom + eps)) * exp_avg)
        self.post_step()

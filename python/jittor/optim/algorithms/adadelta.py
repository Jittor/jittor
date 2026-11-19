"""Adadelta optimizer (dense tensors, Torch-compatible defaults)."""
from numbers import Real
import jittor as jt

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _state_buffer, _update_preserve_dtype,
)


def validate_adadelta_options(options):
    for name in ("lr", "rho", "eps", "weight_decay"):
        value = options[name]
        if isinstance(value, jt.Var) or not isinstance(value, Real):
            raise TypeError("Adadelta %s must be a Python number" % name)
    if not 0 <= options["lr"]:
        raise ValueError("Invalid Adadelta lr: %r" % options["lr"])
    if not 0 <= options["rho"] <= 1:
        raise ValueError("Invalid Adadelta rho: %r" % options["rho"])
    if not 0 <= options["eps"]:
        raise ValueError("Invalid Adadelta eps: %r" % options["eps"])
    if not 0 <= options["weight_decay"]:
        raise ValueError("Invalid Adadelta weight_decay: %r" % options["weight_decay"])
    for name in ("foreach", "differentiable", "fused"):
        if options.get(name):
            raise NotImplementedError("Adadelta %s=True is not supported" % name)


class Adadelta(Optimizer):
    """Torch-compatible Adadelta with per-parameter square averages."""

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0,
                 foreach=None, *, maximize=False, differentiable=False,
                 fused=None):
        self._adadelta_defaults = dict(
            lr=lr, rho=rho, eps=eps, weight_decay=weight_decay,
            foreach=foreach, maximize=maximize, differentiable=differentiable,
            fused=fused)
        validate_adadelta_options(self._adadelta_defaults)
        super().__init__(params, lr)
        for key, value in self._adadelta_defaults.items():
            setattr(self, key, value)
        for group in self.param_groups:
            self._initialize_group(group)

    def _initialize_group(self, group):
        for key, value in self._adadelta_defaults.items():
            group.setdefault(key, value)
        validate_adadelta_options(group)
        group["params"] = list(group["params"])
        group["values"] = [_state_buffer(p) for p in group["params"]]
        group["acc_delta"] = [_state_buffer(p) for p in group["params"]]

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
            validate_adadelta_options(pg)
            lr = pg["lr"]
            rho = pg["rho"]
            eps = pg["eps"]
            weight_decay = pg["weight_decay"]
            maximize = pg["maximize"]
            for p, g, square_avg, acc_delta in zip(
                    pg["params"], pg.get("grads", ()),
                    pg["values"], pg["acc_delta"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g):
                    continue
                if maximize:
                    g = -g
                if weight_decay != 0:
                    g = g + p * weight_decay
                _update_preserve_dtype(
                    square_avg, rho * square_avg + (1 - rho) * g * g)
                delta = (jt.sqrt(acc_delta + eps) / jt.sqrt(square_avg + eps)) * g
                _update_preserve_dtype(p, p - lr * delta)
                _update_preserve_dtype(
                    acc_delta, rho * acc_delta + (1 - rho) * delta * delta)
        self.post_step()

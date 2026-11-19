"""RAdam optimizer (rectified Adam), Torch-compatible dense implementation."""
from numbers import Real
import jittor as jt

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _state_buffer, _update_preserve_dtype,
)


def validate_radam_options(options):
    betas = options["betas"]
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("Invalid RAdam betas: %r" % (betas,))
    for name in ("lr", "eps", "weight_decay"):
        value = options[name]
        if isinstance(value, jt.Var) or not isinstance(value, Real):
            raise TypeError("RAdam %s must be a Python number" % name)
    if not 0 <= options["lr"] or not 0 <= options["eps"] or not 0 <= options["weight_decay"]:
        raise ValueError("Invalid RAdam hyperparameters")
    for name in ("foreach", "differentiable", "fused", "capturable"):
        if options.get(name):
            raise NotImplementedError("RAdam %s=True is not supported" % name)


class RAdam(Optimizer):
    """Torch-compatible RAdam; supports the decoupled weight-decay spelling."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, *, foreach=None, capturable=False,
                 differentiable=False, decoupled=False, fused=None):
        self._radam_defaults = dict(
            lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
            foreach=foreach, capturable=capturable,
            differentiable=differentiable, decoupled=decoupled, fused=fused)
        validate_radam_options(self._radam_defaults)
        super().__init__(params, lr)
        for key, value in self._radam_defaults.items():
            setattr(self, key, value)
        for group in self.param_groups:
            self._initialize_group(group)

    def _initialize_group(self, group):
        for key, value in self._radam_defaults.items():
            group.setdefault(key, value)
        validate_radam_options(group)
        group["params"] = list(group["params"])
        group["values"] = [_state_buffer(p) for p in group["params"]]   # exp_avg
        group["sq_avg"] = [_state_buffer(p) for p in group["params"]]
        group["_radam_steps"] = [0 for _ in group["params"]]

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
            validate_radam_options(pg)
            lr = pg["lr"]
            beta1, beta2 = pg["betas"]
            eps = pg["eps"]
            weight_decay = pg["weight_decay"]
            decoupled = pg["decoupled"]
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            for index, (p, g, exp_avg, exp_avg_sq) in enumerate(zip(
                    pg["params"], pg.get("grads", ()),
                    pg["values"], pg["sq_avg"])):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g):
                    continue
                step = pg["_radam_steps"][index] + 1
                pg["_radam_steps"][index] = step
                if weight_decay != 0:
                    if decoupled:
                        _update_preserve_dtype(p, p * (1 - lr * weight_decay))
                    else:
                        g = g + p * weight_decay
                _update_preserve_dtype(
                    exp_avg, beta1 * exp_avg + (1 - beta1) * g)
                _update_preserve_dtype(
                    exp_avg_sq, beta2 * exp_avg_sq + (1 - beta2) * g * g)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                beta2_t = beta2 ** step
                rho_t = rho_inf - 2.0 * step * beta2_t / bias_correction2
                if rho_t > 5.0:
                    rect = ((rho_t - 4) * (rho_t - 2) * rho_inf /
                            ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                    step_size = lr * rect / bias_correction1
                    denom = jt.sqrt(exp_avg_sq) / (bias_correction2 ** 0.5) + eps
                    _update_preserve_dtype(p, p - step_size * exp_avg / denom)
                else:
                    step_size = lr / bias_correction1
                    _update_preserve_dtype(p, p - step_size * exp_avg)
        self.post_step()

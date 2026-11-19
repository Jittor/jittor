"""NAdam optimizer (Adam with Nesterov momentum), Torch-compatible."""
from numbers import Real
import jittor as jt

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _state_buffer, _update_preserve_dtype,
)


def validate_nadam_options(options):
    betas = options["betas"]
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("Invalid NAdam betas: %r" % (betas,))
    for name in ("lr", "eps", "weight_decay", "momentum_decay"):
        value = options[name]
        if isinstance(value, jt.Var) or not isinstance(value, Real):
            raise TypeError("NAdam %s must be a Python number" % name)
    if not 0 <= options["lr"] or not 0 <= options["eps"] or not 0 <= options["weight_decay"]:
        raise ValueError("Invalid NAdam hyperparameters")
    for name in ("foreach", "differentiable", "fused", "capturable"):
        if options.get(name):
            raise NotImplementedError("NAdam %s=True is not supported" % name)


class NAdam(Optimizer):
    """Torch-compatible NAdam with the momentum schedule and decoupled decay."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, momentum_decay=4e-3, *, foreach=None,
                 capturable=False, differentiable=False,
                 decoupled_weight_decay=False, fused=None):
        self._nadam_defaults = dict(
            lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
            momentum_decay=momentum_decay, foreach=foreach,
            capturable=capturable, differentiable=differentiable,
            decoupled_weight_decay=decoupled_weight_decay, fused=fused)
        validate_nadam_options(self._nadam_defaults)
        super().__init__(params, lr)
        for key, value in self._nadam_defaults.items():
            setattr(self, key, value)
        for group in self.param_groups:
            self._initialize_group(group)

    def _initialize_group(self, group):
        for key, value in self._nadam_defaults.items():
            group.setdefault(key, value)
        validate_nadam_options(group)
        group["params"] = list(group["params"])
        group["values"] = [_state_buffer(p) for p in group["params"]]   # exp_avg
        group["sq_avg"] = [_state_buffer(p) for p in group["params"]]
        group["_nadam_steps"] = [0 for _ in group["params"]]
        group["_nadam_mu_product"] = [1.0 for _ in group["params"]]

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
            validate_nadam_options(pg)
            lr = pg["lr"]
            beta1, beta2 = pg["betas"]
            eps = pg["eps"]
            weight_decay = pg["weight_decay"]
            md = pg["momentum_decay"]
            decoupled = pg["decoupled_weight_decay"]
            for index, (p, g, exp_avg, exp_avg_sq) in enumerate(zip(
                    pg["params"], pg.get("grads", ()),
                    pg["values"], pg["sq_avg"])):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g):
                    continue
                step = pg["_nadam_steps"][index] + 1
                pg["_nadam_steps"][index] = step
                if weight_decay != 0:
                    if decoupled:
                        _update_preserve_dtype(p, p * (1 - lr * weight_decay))
                    else:
                        g = g + p * weight_decay
                mu = beta1 * (1.0 - 0.5 * (0.96 ** (step * md)))
                mu_next = beta1 * (1.0 - 0.5 * (0.96 ** ((step + 1) * md)))
                mu_product = pg["_nadam_mu_product"][index] * mu
                pg["_nadam_mu_product"][index] = mu_product
                mu_product_next = mu_product * mu_next
                _update_preserve_dtype(
                    exp_avg, beta1 * exp_avg + (1 - beta1) * g)
                _update_preserve_dtype(
                    exp_avg_sq, beta2 * exp_avg_sq + (1 - beta2) * g * g)
                denom = jt.sqrt(exp_avg_sq) / ((1 - beta2 ** step) ** 0.5) + eps
                step_size = lr / (1 - beta1 ** step)
                update = (step_size * (1 - mu) / (1 - mu_product) * exp_avg / denom
                          + step_size * mu_next / (1 - mu_product_next) * g / denom)
                _update_preserve_dtype(p, p - update)
        self.post_step()

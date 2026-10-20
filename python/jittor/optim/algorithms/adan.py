"""Adan optimizer."""

import jittor as jt
from ..._core.dtypes import dtype_name

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _adaptive_state_buffer, _update_preserve_dtype,
)


def _adan_update(param, param_value, grad, m, v, d, pre_grad, *,
                 first_step, lr, betas, eps, weight_decay, step):
    beta1, beta2, beta3 = betas
    if first_step:
        _update_preserve_dtype(pre_grad, jt.zeros_like(pre_grad))
    else:
        _update_preserve_dtype(pre_grad, grad - pre_grad)

    _update_preserve_dtype(m, beta1 * m + (1 - beta1) * grad)
    _update_preserve_dtype(d, beta2 * d + (1 - beta2) * pre_grad)
    _update_preserve_dtype(pre_grad, jt.multiply(pre_grad, beta2) + grad)
    _update_preserve_dtype(
        v, beta3 * v + (1 - beta3) * pre_grad * pre_grad)

    bias_correction1 = 1 - beta1 ** float(step)
    bias_correction2 = 1 - beta2 ** float(step)
    bias_correction3_sqrt = jt.sqrt(1 - beta3 ** float(step))
    step_size_diff = lr * beta2 * bias_correction3_sqrt / bias_correction2
    step_size = lr * bias_correction3_sqrt / bias_correction1
    eps_bias_sqrt = eps * bias_correction3_sqrt

    updated_param = param_value - (step_size * m + step_size_diff * d) / (
        jt.sqrt(v) + eps_bias_sqrt)
    _update_preserve_dtype(param, updated_param)
    _update_preserve_dtype(param, updated_param / (1 + lr * weight_decay))
    _update_preserve_dtype(pre_grad, grad)

class Adan(Optimizer):
    """ Adan Optimizer.
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for
        Faster Optimizing Deep Models[J].arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Adan is an efficient optimizer for most DNN frameworks:
    - About 2x fewer computational load than SOTAs
    - Robust to training setting and batch size
    - Easy to Plug-and-play

    Arguments:
        params (iterable): iterable of parameters to optimize or
            dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for
            first- and second-order moments. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay
            (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip
            global grad norm (default: 0.0 no clip)
    """
    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99),
                eps=1e-8, weight_decay=0.0, max_grad_norm=0.0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        for pg in self.param_groups:
            pg["m"] = []
            pg["v"] = []
            pg["d"] = []
            pg["pre_grad"] = []
            for p in pg["params"]:
                pg["m"].append(_adaptive_state_buffer(p))
                pg["v"].append(_adaptive_state_buffer(p))
                pg["d"].append(_adaptive_state_buffer(p))
                pg["pre_grad"].append(_adaptive_state_buffer(p))


    def add_param_group(self, group):
        group["m"] = []
        group["v"] = []
        group["d"] = []
        group["pre_grad"] = []
        for p in group["params"]:
            group["m"].append(_adaptive_state_buffer(p))
            group["v"].append(_adaptive_state_buffer(p))
            group["d"].append(_adaptive_state_buffer(p))
            group["pre_grad"].append(_adaptive_state_buffer(p))
        super().add_param_group(group)

    def _global_max_grad_norm(self):
        """The single gradient-norm bound to enforce this step.

        ``Optimizer.clip_grad_norm`` renormalises the gradients of *every*
        param group at once, so the bound cannot be applied group by group.
        When groups disagree the tightest positive bound wins: it is the only
        value that violates nobody's request.
        """
        bounds = [self.max_grad_norm]
        bounds += [pg["max_grad_norm"] for pg in self.param_groups
                   if "max_grad_norm" in pg]
        positive = [b for b in bounds if b > 0]
        return min(positive) if positive else 0.0

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        # Global clipping happens once, before any group is updated. Doing it
        # inside the loop below applied the clip once per param group (leaving
        # the gradients well under the requested norm) and let the groups
        # visited first take their step on gradients that were not clipped yet.
        max_grad_norm = self._global_max_grad_norm()
        if max_grad_norm > 0: self.clip_grad_norm(max_grad_norm)
        for pg in self.param_groups:
            # Per param group, not the optimizer-wide self.n_step: n_step counts
            # backward() calls, and the gradient-accumulation loop documented on
            # Optimizer.backward calls it once per micro-batch, so a correction
            # keyed on it runs ahead by the accumulation factor. Same root cause
            # as 6.P14 fixed in Adam/AdamW.
            n = float(self._advance_step_count(pg))
            first_step = n == 1
            lr = pg.get("lr", self.lr)
            betas = pg.get("betas", self.betas)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            for p, g, m, v, d, pre_g in zip(pg["params"],
                                            pg["grads"],
                                            pg["m"],
                                            pg["v"],
                                            pg["d"],
                                            pg["pre_grad"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                if dtype_name(p.dtype) in ("float16", "bfloat16"):
                    with jt.flag_scope(amp_reg=0):
                        for state in (m, v, d, pre_g):
                            if dtype_name(state.dtype) != "float32":
                                state.update(state.float32())
                        _adan_update(
                            p, p.float32(), g.float32(), m, v, d, pre_g,
                            first_step=first_step, lr=lr, betas=betas,
                            eps=eps, weight_decay=weight_decay, step=n)
                else:
                    _adan_update(
                        p, p, g, m, v, d, pre_g, first_step=first_step,
                        lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, step=n)
        self.post_step()

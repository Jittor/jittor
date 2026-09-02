"""Adan optimizer."""

import jittor as jt

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _update_preserve_dtype,
)

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
                pg["m"].append(jt.zeros(p.shape, p.dtype).stop_grad())
                pg["v"].append(jt.zeros(p.shape, p.dtype).stop_grad())
                pg["d"].append(jt.zeros(p.shape, p.dtype).stop_grad())
                pg["pre_grad"].append(jt.zeros(p.shape, p.dtype).stop_grad())


    def add_param_group(self, group):
        group["m"] = []
        group["v"] = []
        group["d"] = []
        group["pre_grad"] = []
        for p in group["params"]:
            group["m"].append(jt.zeros(p.shape, p.dtype).stop_grad())
            group["v"].append(jt.zeros(p.shape, p.dtype).stop_grad())
            group["d"].append(jt.zeros(p.shape, p.dtype).stop_grad())
            group["pre_grad"].append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

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
            beta1, beta2, beta3 = betas

            bias_correction1 = 1 - beta1 ** n
            bias_correction2 = 1 - beta2 ** n
            bias_correction3_sqrt = jt.sqrt(1 - beta3 ** n)


            step_size_diff = lr * beta2 * bias_correction3_sqrt / bias_correction2
            step_size = lr * bias_correction3_sqrt / bias_correction1
            eps_bias_sqrt = eps * bias_correction3_sqrt

            for p, g, m, v, d, pre_g in zip(pg["params"],
                                            pg["grads"],
                                            pg["m"],
                                            pg["v"],
                                            pg["d"],
                                            pg["pre_grad"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue

                if first_step:
                    # Official Adan seeds pre_grad with the FIRST gradient, so
                    # grad_diff is 0 on step 1. jittor seeded pre_grad with
                    # zeros and still took the difference, making grad_diff = g
                    # and giving a first update no other Adan produces. The old
                    # guard here (`if self.n_step > 0`) never excluded anything:
                    # pre_step() runs backward(), so n_step is already 1 by the
                    # time the first step reaches this line.
                    _update_preserve_dtype(pre_g, jt.zeros_like(pre_g))
                else:
                    _update_preserve_dtype(
                        pre_g, g - pre_g)  # Update pre_g as grad_diff


                _update_preserve_dtype(m, beta1 * m + (1 - beta1) * g)
                _update_preserve_dtype(
                    d, beta2 * d + (1 - beta2) * pre_g)

                _update_preserve_dtype(
                    pre_g, jt.multiply(pre_g, beta2) + g)

                _update_preserve_dtype(
                    v, beta3 * v + (1 - beta3) * pre_g * pre_g)

                _update_preserve_dtype(
                    p,
                    p - (step_size * m + step_size_diff * d)
                    / (jt.sqrt(v) + eps_bias_sqrt),
                )
                _update_preserve_dtype(p, p / (1 + lr * weight_decay))

                _update_preserve_dtype(
                    pre_g, g)  # Update pre_g for the next iteration
        self.post_step()

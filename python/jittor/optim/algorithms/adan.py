"""Adan optimizer."""

import jittor as jt

from ..base import Optimizer, _grad_matches_param, _param_requires_grad

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

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        n = float(self.n_step)
        for pg in self.param_groups:
            lr = pg.get("lr", self.lr)
            betas = pg.get("betas", self.betas)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            max_grad_norm = pg.get("max_grad_norm", self.max_grad_norm)
            if max_grad_norm>0: self.clip_grad_norm(max_grad_norm)
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

                if self.n_step>0:
                    pre_g.update(g - pre_g)  # Update pre_g as grad_diff


                m.update(beta1 * m + (1 - beta1) * g)
                d.update(beta2 * d + (1 - beta2) * pre_g)  # Use pre_g as grad_diff

                pre_g.update(jt.multiply(pre_g, beta2) + g)  # Update pre_g as update (g + beta2 * grad_diff)

                v.update(beta3 * v + (1 - beta3) * pre_g * pre_g)  # Use pre_g as update

                p.update(p - (step_size * m + step_size_diff * d) / (jt.sqrt(v) + eps_bias_sqrt))
                p.update(p / (1 + lr * weight_decay))

                pre_g.update(g)  # Update pre_g for the next iteration
        self.post_step()

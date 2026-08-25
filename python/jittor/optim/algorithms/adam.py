"""Adam-family optimizers."""

import jittor as jt

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _update_preserve_dtype,
)

class Adam(Optimizer):
    """ Adam Optimizer.

    Example::

        optimizer = nn.Adam(model.parameters(), lr, eps=1e-8, betas=(0.9, 0.999))
        optimizer.step(loss)
    """
    def __init__(self, params, lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0):
        super().__init__(params, lr)
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay
        # assert weight_decay==0, "weight_decay is not supported yet"

        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            m = pg["m"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())
                m.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        m = group["m"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
            m.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        n = float(self.n_step)
        jt.flags.node_order = 1
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            b0, b1 = pg.get("betas", self.betas)
            for p, g, v, m in zip(pg["params"], pg["grads"], pg["values"], pg["m"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                g = p * weight_decay + g
                _update_preserve_dtype(m, b0 * m + (1-b0) * g)
                _update_preserve_dtype(v, b1 * v + (1-b1) * g * g)
                step_size = lr * jt.sqrt(1-b1**n) / (1-b0 ** n)
                _update_preserve_dtype(
                    p, p - m * step_size / (jt.sqrt(v) + eps))
        self.post_step()


class AdamW(Optimizer):
    """ AdamW Optimizer.

    Example::

        optimizer = nn.AdamW(model.parameters(), lr, eps=1e-8, betas=(0.9, 0.999))
        optimizer.step(loss)
    """
    def __init__(self, params, lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0):
        super().__init__(params, lr)
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay
        # assert weight_decay==0, "weight_decay is not supported yet"

        # initialize required arguments for each param_groups
        for pg in self.param_groups:
            values = pg["values"] = []
            m = pg["m"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())
                m.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        m = group["m"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
            m.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        n = float(self.n_step)
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            b0, b1 = pg.get("betas", self.betas)
            for p, g, v, m in zip(pg["params"], pg["grads"], pg["values"], pg["m"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                _update_preserve_dtype(p, p * (1 - lr * weight_decay))
                bias_correction1 = 1 - b0 ** n
                bias_correction2 = 1 - b1 ** n
                _update_preserve_dtype(m, b0 * m + (1-b0) * g) #exp_avg
                _update_preserve_dtype(v, b1 * v + (1-b1) * g * g) #exp_avg_sq
                denom = jt.sqrt(v) / jt.sqrt(bias_correction2) + eps
                step_size = lr / bias_correction1
                _update_preserve_dtype(p, p - step_size * m / denom)
        self.post_step()

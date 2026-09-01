"""Adam-family optimizers."""

import jittor as jt

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _update_preserve_dtype,
)


def _acl_fused_adamw_updates(entries, lr, beta1, beta2, weight_decay, eps):
    from jittor.extern.acl.aclops.adamw_op import fused_adamw_acl

    results = [None] * len(entries)
    buckets = {}
    for index, entry in enumerate(entries):
        buckets.setdefault(int(entry[4]), []).append((index,) + entry[:4])
    for step_value, bucket in buckets.items():
        step = jt.array(float(step_value), dtype="float32").stop_grad()
        updated = fused_adamw_acl(
            [item[1] for item in bucket], [item[2] for item in bucket],
            [item[3] for item in bucket], [item[4] for item in bucket],
            step, lr, beta1, beta2, weight_decay, eps)
        for output_index, item in enumerate(bucket):
            results[item[0]] = tuple(
                values[output_index] for values in updated)
    return results

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
    def __init__(self, params, lr, eps=1e-8, betas=(0.9, 0.999),
                 weight_decay=0, fused=None):
        super().__init__(params, lr)
        self.eps = eps
        self.betas = betas
        self.weight_decay = weight_decay
        self.fused = fused
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
            fused = pg.get("fused", self.fused) is True and jt.flags.use_acl
            if fused:
                active = [(p, m, v, g, n - 1) for p, g, v, m in zip(
                    pg["params"], pg["grads"], pg["values"], pg["m"])
                    if _param_requires_grad(p) and _grad_matches_param(p, g)]
                updates = _acl_fused_adamw_updates(
                    active, lr, b0, b1, weight_decay, eps)
                for (p, m, v, _, _), (new_p, new_m, new_v) in zip(
                        active, updates):
                    _update_preserve_dtype(p, new_p)
                    _update_preserve_dtype(m, new_m)
                    _update_preserve_dtype(v, new_v)
                    if p.is_stop_grad():
                        p.start_grad()
                continue
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

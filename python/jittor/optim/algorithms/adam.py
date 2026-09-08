"""Adam-family optimizers."""

import jittor as jt
from ..._core.dtypes import dtype_name
from ..._runtime.dispatch import register_kernel, select_kernel

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _update_preserve_dtype,
)


def _acl_fused_adamw_updates(entries, lr, beta1, beta2, weight_decay, eps):
    from jittor.backends.acl.kernels.ops.adamw_op import fused_adamw_acl

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


register_kernel("optim.adamw_fused", "acl", _acl_fused_adamw_updates)


def adam_update(param, grad, value, momentum, *, lr, eps, weight_decay,
                betas, step, decoupled_weight_decay=False, torch_math=False):
    """Shared Adam arithmetic; callers own gradient sourcing and step counters.

    Native Adam historically puts epsilon before bias scaling. Torch and
    AdamW put it after scaling; keep that policy explicit at the call site.
    """
    b0, b1 = betas
    if weight_decay != 0 and decoupled_weight_decay:
        param = (param * (1 - lr * weight_decay)).cast(param.dtype)
    elif weight_decay != 0 or not torch_math and not decoupled_weight_decay:
        grad = grad + param * weight_decay
    _update_preserve_dtype(momentum, b0 * momentum + (1 - b0) * grad)
    _update_preserve_dtype(value, b1 * value + (1 - b1) * grad * grad)
    if torch_math or decoupled_weight_decay:
        correction = (1 - b1 ** float(step)) ** 0.5
        scalar = (jt.array(correction, dtype="float32" if dtype_name(value.dtype) == "bfloat16"
                           else value.dtype).cast(value.dtype).stop_grad()
                  if torch_math else jt.sqrt(1 - b1 ** float(step)))
        denom = jt.sqrt(value) / scalar + eps
        return param - momentum * (lr / (1 - b0 ** float(step))) / denom
    step_size = lr * jt.sqrt(1 - b1 ** float(step)) / (1 - b0 ** float(step))
    return param - momentum * step_size / (jt.sqrt(value) + eps)


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
        jt.flags.node_order = 1
        for pg in self.param_groups:
            # bias correction counts optimizer steps, not backward calls
            n = float(self._advance_step_count(pg))
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            b0, b1 = pg.get("betas", self.betas)
            for p, g, v, m in zip(pg["params"], pg["grads"], pg["values"], pg["m"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                _update_preserve_dtype(p, adam_update(
                    p, g, v, m, lr=lr, eps=eps, weight_decay=weight_decay,
                    betas=(b0, b1), step=n))
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
        for pg in self.param_groups:
            # bias correction counts optimizer steps, not backward calls
            n = float(self._advance_step_count(pg))
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            eps = pg.get("eps", self.eps)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            b0, b1 = pg.get("betas", self.betas)
            fused = None
            if pg.get("fused", self.fused) is True:
                active = [(p, m, v, g, n - 1) for p, g, v, m in zip(
                    pg["params"], pg["grads"], pg["values"], pg["m"])
                    if _param_requires_grad(p) and _grad_matches_param(p, g)]
                if active:
                    fused = select_kernel("optim.adamw_fused", active)
            if fused is not None:
                updates = fused(
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
                _update_preserve_dtype(p, adam_update(
                    p, g, v, m, lr=lr, eps=eps, weight_decay=weight_decay,
                    betas=(b0, b1), step=n, decoupled_weight_decay=True))
        self.post_step()

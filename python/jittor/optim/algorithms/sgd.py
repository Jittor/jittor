"""Stochastic gradient descent optimizer."""

import jittor as jt

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _update_preserve_dtype,
)

class SGD(Optimizer):
    """ SGD Optimizer.

    Example::

        optimizer = nn.SGD(model.parameters(), lr, momentum=0.9)
        optimizer.step(loss)
    """
    def __init__(self, params, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

        # initialize required arguments
        for pg in self.param_groups:
            values = pg["values"] = []
            for p in pg["params"]:
                values.append(jt.zeros(p.shape, p.dtype).stop_grad())

    def add_param_group(self, group):
        values = group["values"] = []
        for p in group["params"]:
            values.append(jt.zeros(p.shape, p.dtype).stop_grad())
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph=retain_graph)
        jt.flags.node_order = 1
        for pg in self.param_groups:
            # get arguments from each param_groups
            lr = pg.get("lr", self.lr)
            momentum = pg.get("momentum", self.momentum)
            weight_decay = pg.get("weight_decay", self.weight_decay)
            dampening = pg.get("dampening", self.dampening)
            nesterov = pg.get("nesterov", self.nesterov)

            # optimize main body
            # Without momentum the velocity buffer holds nothing the step needs:
            # `v` comes out equal to `dp` and is read back only to be scaled by
            # lr. Keeping it costs a full write and read of every parameter, and
            # this is the default configuration -- on a ViT training step the
            # fused update kernel was 17% of the whole step, against 6% for the
            # same update in PyTorch. `dampening` still scales the update here
            # even at momentum 0 (unlike torch, where it only applies inside the
            # momentum branch), so the shortcut is limited to dampening 0 rather
            # than quietly changing that. `v` is then left at whatever it held;
            # turning momentum on later resumes from zeros, which is what this
            # optimizer has always started from.
            plain = momentum == 0 and dampening == 0 and not nesterov
            for p, g, v in zip(pg["params"], pg["grads"], pg["values"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                # `p * 0 + g` is a whole extra pass over the parameter.
                dp = g if weight_decay == 0 else p * weight_decay + g
                if plain:
                    _update_preserve_dtype(p, p - dp * lr)
                    continue
                _update_preserve_dtype(
                    v, momentum * v + dp * (1 - dampening))
                if nesterov:
                    _update_preserve_dtype(
                        p, p - (dp + momentum * v) * lr)
                else:
                    _update_preserve_dtype(p, p - v * lr)
        self.post_step()

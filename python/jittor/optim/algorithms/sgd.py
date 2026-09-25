"""Stochastic gradient descent optimizer."""

import jittor as jt

from ..._runtime.dispatch import register_kernel, select_kernel

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _state_buffer, _update_preserve_dtype,
)

def sgd_update(param, grad, velocity, *, lr, momentum=0, weight_decay=0,
               dampening=0, nesterov=False):
    """Native SGD arithmetic shared by full parameters and FSDP shards."""
    dp = grad if weight_decay == 0 else param * weight_decay + grad
    if momentum == 0 and dampening == 0 and not nesterov:
        return param - dp * lr
    _update_preserve_dtype(velocity, momentum * velocity + dp * (1 - dampening))
    return param - (dp + momentum * velocity if nesterov else velocity) * lr


def _momentum_buffer(param):
    """A real, contiguous zero buffer.

    `jt.zeros` is a broadcast of a scalar and carries no storage of its own,
    which the fused optimizer kernels reject: they write their inputs in place
    and so require contiguous storage. Paying one materialisation at
    construction keeps the hot path free of the copy.

    It is also built on the *parameter's* device rather than the ambient one
    (`_state_buffer` -> `zeros_like`): an optimizer created for a model that
    is already on cuda:1 used to put its velocity on cuda:0 and fail in the
    fused kernel on the first step.
    """
    return _state_buffer(param).contiguous().stop_grad()


def _acl_fused_sgd_updates(entries, lr, momentum, weight_decay, dampening, nesterov):
    """One op for the whole parameter list.

    The portable update is five elementwise passes per parameter, and on ACL
    every pass is its own graph node and its own launch, so the optimizer --
    not the model -- was the bulk of a training step.
    """
    from jittor.backends.acl.kernels.ops.fused_sgd_op import fused_sgd_acl

    parameters = [entry[0] for entry in entries]
    gradients = [entry[1] for entry in entries]
    velocities = [entry[2] for entry in entries]
    new_parameters, new_velocities = fused_sgd_acl(
        parameters, velocities, gradients, lr, momentum, weight_decay,
        dampening, nesterov)
    return list(zip(new_parameters, new_velocities))


# float32 only: the ACL runner hands the CANN foreach operators their
# coefficients as float32 device scalars, which is the pairing those kernels
# accept for float32 and bfloat16 parameters but not for float16. Restricting
# the registration is what makes an unsupported dtype fall back to the portable
# update instead of failing, and bfloat16 stays out until its parity with the
# portable update is measured rather than assumed.
register_kernel("optim.sgd_fused", "acl", _acl_fused_sgd_updates,
                dtypes=("float32",))

# Registers the CUDA entry for the same operator. Imported for its side effect
# and last, so that a build without the CUDA kernels still gets the ACL one.
from jittor.backends.cuda.kernels.optim import fused_sgd_cuda as _fused_sgd_cuda  # noqa: E402,F401


class SGD(Optimizer):
    """ SGD Optimizer.

    Example::

        optimizer = nn.SGD(model.parameters(), lr, momentum=0.9)
        optimizer.step(loss)
    """
    def __init__(self, params, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False,
                 fused=None):
        super().__init__(params, lr)
        # None means "wherever a backend publishes a fused update"; only ACL
        # does, and only for the dtypes its kernels cover, so everything else
        # keeps the portable path without asking for it. False turns it off.
        self.fused = fused
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

        # initialize required arguments
        for pg in self.param_groups:
            values = pg["values"] = []
            for p in pg["params"]:
                values.append(_momentum_buffer(p))

    def add_param_group(self, group):
        values = group["values"] = []
        for p in group["params"]:
            values.append(_momentum_buffer(p))
        self.param_groups.append(group)

    def step(self, loss=None, retain_graph=False):
        from jittor._runtime import step_capture
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
            active = [(p, g, v) for p, g, v in zip(pg["params"], pg["grads"], pg["values"])
                      if _param_requires_grad(p) and _grad_matches_param(p, g)]
            if not active:
                continue
            fused = None
            if pg.get("fused", getattr(self, "fused", None)) is not False:
                # Momentum-free is considered too. That shortcut is a single
                # pass, but it is a single pass *per parameter*: two elementwise
                # ops and a holder rebind, 96 times for an 8-layer transformer,
                # which measured 1.11 ms of a 7.41 ms training step. A fused
                # kernel does the whole list in one launch, which is what
                # PyTorch's `foreach` SGD does.
                # Every Var the kernel will dereference, not just the
                # parameters. `_fused_sgd_cuda` declares `float* param[]`,
                # `float* grad[]` and `float* vel[]` and is registered
                # `dtypes=("float32",)`, and the dispatcher filters on the
                # dtypes of the Vars it is *shown*. Shown the parameters alone,
                # it selected the float32 kernel under
                # `auto_mixed_precision_level` 4, 5 and 6 -- where the
                # parameters stay float32 and dtype inference lowers the
                # *gradients* to float16, which is the entire point of those
                # levels -- and handed it a `__half*`. nvcc refused at the first
                # optimizer step with "a value of type \"jittor::float16 *\"
                # cannot be assigned to an entity of type \"float *\"" pointed
                # at `src/ops/composite/code_op.cc`, so native mixed-precision
                # training on CUDA did not run at all. On CPU there is no fused
                # kernel to select and the same script trained.
                fused = select_kernel(
                    "optim.sgd_fused",
                    [var for item in active for var in item
                     if isinstance(var, jt.Var)])
            if fused is not None:
                rate = lr
                if step_capture.active():
                    rate = step_capture.live_rate(
                        fused, lambda pg=pg: pg.get("lr", self.lr),
                        lambda pg=pg: (pg.get("momentum", self.momentum),
                                       pg.get("weight_decay", self.weight_decay),
                                       pg.get("dampening", self.dampening),
                                       pg.get("nesterov", self.nesterov)))
                updates = fused(active, rate, momentum, weight_decay, dampening, nesterov)
                for (p, _, v), (new_p, new_v) in zip(active, updates):
                    # Without momentum the velocity buffer holds nothing the
                    # step needs, and a kernel that keeps it updates it in
                    # place, so it is handed back as the same Var: rebinding it
                    # would be a holder write per parameter for no reason.
                    if new_v is not v:
                        _update_preserve_dtype(v, new_v)
                    _update_preserve_dtype(p, new_p)
                continue
            # Baked into the graph as numbers; a captured step re-captures
            # when they change.
            step_capture.guard(lambda pg=pg: (pg.get("lr", self.lr), pg.get("momentum", self.momentum),
                                              pg.get("weight_decay", self.weight_decay),
                                              pg.get("dampening", self.dampening),
                                              pg.get("nesterov", self.nesterov)))
            for p, g, v in active:
                # `p * 0 + g` is a whole extra pass over the parameter.
                _update_preserve_dtype(p, sgd_update(
                    p, g, v, lr=lr, momentum=momentum, weight_decay=weight_decay,
                    dampening=dampening, nesterov=nesterov))
        self.post_step()

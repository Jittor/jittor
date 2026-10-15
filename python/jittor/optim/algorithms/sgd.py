"""Stochastic gradient descent optimizer."""

import jittor as jt

from ..._runtime.dispatch import register_kernel, select_kernel

from ..base import (
    Optimizer, _grad_matches_param, _param_requires_grad,
    _state_buffer, _update_preserve_dtype,
)

#: Per-parameter list, parallel to ``params``: whether that parameter's
#: momentum buffer has had its first real update. It rides along in the
#: param-group dict, so native ``state_dict``/``load_state_dict`` carry it.
MOMENTUM_INITIALIZED = "momentum_initialized"


def momentum_initialized_flags(pg):
    """The group's first-touch flags, created (all False) and kept aligned.

    The first momentum update of a buffer is a property of that buffer, not
    of the optimizer or the param group: a parameter unfrozen late, one that
    had no gradient in earlier steps, or any parameter when momentum is
    switched on mid-training all reach their first momentum update after the
    group has already stepped. Keying the seeding on the group's step count
    sent those buffers through the dampened recurrence from zero.
    """
    params = pg.get("params", [])
    flags = pg.get(MOMENTUM_INITIALIZED)
    if not isinstance(flags, list):
        flags = pg[MOMENTUM_INITIALIZED] = [False] * len(params)
    while len(flags) < len(params):
        flags.append(False)
    del flags[len(params):]
    return flags


def sgd_update(param, grad, velocity, *, lr, momentum=0, weight_decay=0,
               dampening=0, nesterov=False, momentum_initialized=False):
    """Native SGD arithmetic shared by full parameters and FSDP shards.

    ``momentum_initialized`` says whether ``velocity`` already holds a
    momentum buffer (see ``momentum_initialized_flags``). Dampening is a
    PyTorch "dampening for momentum" knob -- meaningless without momentum, so
    it must not scale the update when momentum is off. And momentum's buffer
    is *seeded* from the raw gradient on its first real touch, not folded
    into the zero-initialized buffer via the dampened recurrence: that
    recurrence is only correct from the second update onward. The caller
    marks the buffer initialized after any update that used momentum.
    """
    dp = grad if weight_decay == 0 else param * weight_decay + grad
    if momentum == 0 and not nesterov:
        return param - dp * lr
    if not momentum_initialized:
        _update_preserve_dtype(velocity, dp)
    else:
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


def _acl_fused_sgd_updates(entries, lr, momentum, weight_decay, dampening, nesterov, step=1):
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
        dampening, nesterov, step)
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
            pg[MOMENTUM_INITIALIZED] = [False] * len(pg["params"])

    def add_param_group(self, group):
        values = group["values"] = []
        for p in group["params"]:
            values.append(_momentum_buffer(p))
        group[MOMENTUM_INITIALIZED] = [False] * len(group["params"])
        self.param_groups.append(group)

    def load_state_dict(self, state):
        super().load_state_dict(state)
        defaults = state.get("defaults") if isinstance(state, dict) else None
        if not isinstance(defaults, dict):
            # Not the native layout (e.g. a torch-format dict under the
            # compatibility layer, which sets the flags from its own state).
            return
        saved_groups = defaults.get("param_groups") or []
        for pg, saved in zip(self.param_groups, saved_groups):
            if MOMENTUM_INITIALIZED in saved:
                momentum_initialized_flags(pg)
                continue
            # A state written before the per-buffer flags existed. It cannot
            # say which buffers were ever touched (a zero buffer may be
            # untouched or a legitimate result), so resume exactly as the
            # writing version would have: it seeded only when the group's
            # own step counter was about to reach 1 (``n_step`` saved as 0),
            # and before ``n_step`` existed it never seeded at all.
            legacy = int(saved.get("n_step", 1)) >= 1
            pg[MOMENTUM_INITIALIZED] = [legacy] * len(pg.get("params", []))

    def step(self, loss=None, retain_graph=False):
        from jittor._runtime import step_capture
        self.pre_step(loss, retain_graph=retain_graph)
        jt.flags.node_order = 1
        for pg in self.param_groups:
            # Counts optimizer steps (not backward calls), like Adam's own
            # bias-correction counter (Optimizer._advance_step_count). Kept for
            # state compatibility; whether a buffer's update is its first is
            # decided per parameter by `initialized` below, not by this count.
            # Nothing reads it into the update, so unlike
            # `_advance_step_count` it does not refuse step capture.
            pg["n_step"] = int(pg.get("n_step", 0)) + 1
            initialized = momentum_initialized_flags(pg)
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
            # same update in PyTorch. `v` is then left at whatever it held;
            # turning momentum on later resumes from zeros, which is what this
            # optimizer has always started from.
            # A parameter the loss never reached is skipped outright, as
            # torch.optim.SGD skips a None .grad: no weight decay, and no
            # momentum buffer seeded from a zero that was never a gradient.
            absent = getattr(self, "_absent_grads", ())
            active_index = [i for i, (p, g) in enumerate(zip(pg["params"], pg["grads"]))
                            if _param_requires_grad(p) and _grad_matches_param(p, g)
                            and id(p) not in absent]
            if not active_index:
                continue
            active = [(pg["params"][i], pg["grads"][i], pg["values"][i]) for i in active_index]
            uses_momentum = not (momentum == 0 and not nesterov)
            if not uses_momentum:
                # Dampening only scales momentum's recurrence. The fused ACL
                # kernel applied it to the plain update as well.
                dampening = 0
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
                # The fused kernels take one first-step switch per launch, so
                # buffers seeding now and buffers continuing their recurrence
                # go in separate launches (`step` 1 and 2). Without momentum
                # the switch is unused and the list stays one launch.
                batches = [active_index]
                if uses_momentum:
                    batches = [[i for i in active_index if not initialized[i]],
                               [i for i in active_index if initialized[i]]]
                for step, batch in zip((1, 2), batches):
                    if not batch:
                        continue
                    entries = [(pg["params"][i], pg["grads"][i], pg["values"][i]) for i in batch]
                    updates = fused(entries, rate, momentum, weight_decay, dampening, nesterov, step)
                    for (p, _, v), (new_p, new_v) in zip(entries, updates):
                        # Without momentum the velocity buffer holds nothing
                        # the step needs, and a kernel that keeps it updates it
                        # in place, so it is handed back as the same Var:
                        # rebinding it would be a holder write per parameter
                        # for no reason.
                        if new_v is not v:
                            _update_preserve_dtype(v, new_v)
                        _update_preserve_dtype(p, new_p)
            else:
                # Baked into the graph as numbers; a captured step re-captures
                # when they change.
                step_capture.guard(lambda pg=pg: (pg.get("lr", self.lr), pg.get("momentum", self.momentum),
                                                  pg.get("weight_decay", self.weight_decay),
                                                  pg.get("dampening", self.dampening),
                                                  pg.get("nesterov", self.nesterov)))
                for i, (p, g, v) in zip(active_index, active):
                    # `p * 0 + g` is a whole extra pass over the parameter.
                    _update_preserve_dtype(p, sgd_update(
                        p, g, v, lr=lr, momentum=momentum, weight_decay=weight_decay,
                        dampening=dampening, nesterov=nesterov,
                        momentum_initialized=initialized[i]))
            if uses_momentum:
                # Whether a buffer seeds is baked into the step too (which
                # launch it is in, or which formula): a captured seeding step
                # must not be replayed once the buffers are initialized. Read
                # before the flags flip, so the next step re-captures.
                step_capture.guard(lambda pg=pg: tuple(momentum_initialized_flags(pg)))
                for i in active_index:
                    initialized[i] = True
        self.post_step()

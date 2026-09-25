# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Dynamic loss scaling for float16 training.

float16 has about five decimal digits and its smallest normal value is 6e-5.
A gradient that is perfectly representable as a float32 number -- 1e-7, say --
is zero in float16, so the backward pass of an fp16 model silently loses the
small gradients and the model stops learning. Multiplying the loss by a large
constant before the backward moves the whole gradient distribution up into
float16's range; dividing the gradients by the same constant afterwards leaves
the update unchanged. That is all loss scaling is.

The constant cannot be fixed, because too large a scale overflows to inf and
too small a one still underflows, and which is which depends on the model and
on how far into training you are. So it is chosen dynamically: start high, halve
it whenever a gradient comes back non-finite (and skip that step, because those
gradients are garbage), and double it after a long enough run of clean steps.

    scaler = jt.amp.GradScaler()
    for x, y in data:
        loss = loss_fn(model(x), y)
        optimizer.backward(scaler.scale(loss))
        scaler.step(optimizer)      # unscales, skips the step on inf/nan
        scaler.update()             # grows or backs off the scale

bfloat16 has float32's exponent range and does not need any of this; it is
float16, and float16 only, that this exists for.

This is the implementation ``torch.amp.GradScaler`` is built on in the
compatibility frontend (``compat/torch/grad_scaler.py``), which adds torch's
device-first signature and its legacy per-device subclasses on top. One
algorithm, two front doors.
"""
import collections.abc as _collections_abc
import warnings

import numpy as np

from jittor._core.dtypes import dtype_name as _jittor_dtype_name

__all__ = ["GradScaler"]


def _jittor():
    """The package, at call time.

    ``jittor/__init__.py`` imports this module, so importing it back at module
    scope would put ``jittor.amp`` on the import-time cycle that
    ``tests/structure/test_import_layering.py`` ratchets against. Nothing here
    needs the package until a method runs, by which point the import is a
    ``sys.modules`` lookup.
    """
    import jittor
    return jittor


def _device_scalar(like, value):
    """A finished one-element float32 Var holding `value`, placed as `like` is.

    Placed, not derived: ``like * 0 + value`` would put the loss's graph under
    it, and executing it -- as a Var that is state must be -- would execute
    part of a step being captured.
    """
    jt = _jittor()
    core = jt.core
    token = None
    backend = like.placement_backend
    if backend >= 0:
        token = core._set_tensor_placement(backend, max(int(like.device_id), 0))
    try:
        var = jt.array(np.array([value], np.float32))
    finally:
        if token is not None:
            core._reset_tensor_placement(token)
    before = jt.flags.keep_graph
    jt.flags.keep_graph = 0
    try:
        var.sync(False, False)
    finally:
        jt.flags.keep_graph = before
    return var.stop_grad()


class GradScaler:
    """Scale a loss for the backward pass, and unscale its gradients after.

    ``enabled=False`` makes every method a no-op that returns its input, so a
    training loop can be written once and run at any precision without
    branching on the dtype.
    """

    def __init__(self, init_scale=2.0 ** 16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000, enabled=True):
        self._enabled = bool(enabled)
        self._init_scale = float(init_scale)
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._growth_tracker = 0
        self._found_inf = False
        self._unscaled = False
        # [scale, growth tracker] on the device, once a captured step used
        # the scaler; see `_device_state`. From then on they are the state.
        self._device = None
        self._found_inf_var = None

    def is_enabled(self):
        return self._enabled

    # -- device state, for a captured step -------------------------------
    def _device_state(self, like):
        """The scale and growth tracker as device Vars, made on first use.

        A captured step replays without running this class's Python, so what
        it decides each step -- skip or not, the next scale -- has to be
        computed on the device from state the step updates (and the capture
        tracks, as it tracks parameters). Made next to `like`, the loss, so
        they share its placement.
        """
        if self._device is None:
            self._device = [_device_scalar(like, self._scale),
                            _device_scalar(like, float(self._growth_tracker))]
        return self._device

    def _pull(self):
        """Bring the device state back into the Python fields (a read)."""
        if self._device is not None:
            scale, tracker = self._device
            self._scale = float(scale.numpy().reshape(-1)[0])
            self._growth_tracker = int(tracker.numpy().reshape(-1)[0])

    def _on_device(self):
        from jittor._runtime import step_capture
        return self._device is not None or step_capture.active()

    def get_scale(self):
        if not self._enabled:
            return 1.0
        self._pull()
        return self._scale

    def get_growth_factor(self):
        return self._growth_factor

    def set_growth_factor(self, new_factor):
        self._growth_factor = float(new_factor)

    def get_backoff_factor(self):
        return self._backoff_factor

    def set_backoff_factor(self, new_factor):
        self._backoff_factor = float(new_factor)

    def get_growth_interval(self):
        return self._growth_interval

    def set_growth_interval(self, new_interval):
        self._growth_interval = int(new_interval)

    #: float16's largest finite value. A scale above it cannot be applied to a
    #: float16 loss of magnitude 1 or more without overflowing.
    _FLOAT16_MAX = 65504.0

    def scale(self, outputs):
        """Multiply a Var, or every Var in a container, by the scale.

        The product keeps the input's dtype, which is what torch does -- its
        ``_scale`` is a float32 *tensor*, and a half tensor times a float32
        tensor promotes, but a half tensor times a python float does not. So
        scaling a float16 loss directly overflows in both frameworks; torch
        users rarely meet it because autocast keeps loss functions in float32,
        and a native jittor model cast wholesale to float16 has a float16 loss.
        Scale a float32 copy -- ``scaler.scale(loss.float32())`` -- and the
        backward still runs in half from the first op.
        """
        if not self._enabled:
            return outputs
        jt = _jittor()
        if isinstance(outputs, jt.Var) and self._on_device():
            scale = self._device_state(outputs)[0]
            return outputs * scale.reshape(()).cast(outputs.dtype)
        if isinstance(outputs, jt.Var):
            if (self._scale > self._FLOAT16_MAX
                    and _jittor_dtype_name(outputs.dtype) == "float16"):
                # Guaranteed inf for any |loss| >= 1, and the failure is silent:
                # every step then finds a non-finite gradient, skips, and backs
                # the scale off until it reaches 1, so training simply does not
                # happen and no exception is raised.
                warnings.warn(
                    "GradScaler.scale() on a float16 tensor with scale %g "
                    "overflows float16 (max %g). Scale a float32 copy of the "
                    "loss -- scaler.scale(loss.float32()) -- or lower "
                    "init_scale; otherwise every step is skipped and the "
                    "scale collapses to 1."
                    % (self._scale, self._FLOAT16_MAX),
                    RuntimeWarning, stacklevel=2)
            return outputs * self._scale
        if isinstance(outputs, (list, tuple)):
            return type(outputs)(self.scale(item) for item in outputs)
        if isinstance(outputs, _collections_abc.Iterable):
            return (self.scale(item) for item in outputs)
        raise ValueError("outputs must be a Var or an iterable of Vars")

    def _grads(self, optimizer):
        gs = []
        for pg in getattr(optimizer, "param_groups", []):
            for g in (pg.get("grads", []) or []):
                if g is not None:
                    gs.append(g)
        return gs

    def unscale_(self, optimizer):
        """Divide the optimizer's gradients by the scale, in place.

        Call this directly only when something has to happen between the
        unscale and the step -- gradient clipping is the usual reason, and it
        has to see true gradients. :meth:`step` calls it otherwise.
        """
        if not self._enabled:
            return
        jt = _jittor()
        if self._on_device():
            return self._unscale_on_device(optimizer)
        inv = np.float32(1.0 / self._scale)
        flattened = []
        for g in self._grads(optimizer):
            if not g.numel():
                continue
            unscaled = g * inv
            if _jittor_dtype_name(unscaled.dtype) != _jittor_dtype_name(g.dtype):
                unscaled = unscaled.cast(_jittor_dtype_name(g.dtype))
            g.update(unscaled)
            flattened.append(unscaled.cast("float32").reshape((-1,)))
        # The step has to be skipped or not, on the host, so this reduction has
        # to come back to Python -- but once, over everything, rather than once
        # per gradient. It reads the values actually assigned back to the
        # gradients, so an overflow introduced by the unscale itself is caught
        # along with one that arrived from the backward.
        self._found_inf = (
            not bool(jt.isfinite(jt.concat(flattened)).all().item())
            if flattened else False
        )
        self._unscaled = True

    def _unscale_on_device(self, optimizer):
        """`unscale_` without reading anything back: found-inf stays a Var."""
        jt = _jittor()
        grads = [g for g in self._grads(optimizer) if g.numel()]
        flattened = []
        if grads:
            inv = 1.0 / self._device_state(grads[0])[0]
            for g in grads:
                unscaled = g * inv.reshape(()).cast(g.dtype)
                g.update(unscaled)
                flattened.append(unscaled.cast("float32").reshape((-1,)))
        self._found_inf_var = (
            jt.logical_not(jt.isfinite(jt.concat(flattened)).all())
            if flattened else None)
        self._unscaled = True

    def step(self, optimizer, *args, **kwargs):
        """Unscale if needed, then step -- unless a gradient was not finite.

        Returns the optimizer's own return value, or None when the step was
        skipped. A skipped step is not an error: it is the scale being too
        large, which :meth:`update` is about to correct.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)
        if not self._unscaled:
            self.unscale_(optimizer)
        self._unscaled = False
        if self._on_device() and self._found_inf_var is not None:
            from jittor._runtime import step_capture
            skips = getattr(optimizer, "_skips_step_on_device", None)
            if step_capture.active() and skips is not None and skips():
                # The optimizer's update reads the flag and skips on the device.
                optimizer.__dict__["_amp_found_inf"] = self._found_inf_var
                try:
                    return optimizer.step(*args, **kwargs)
                finally:
                    optimizer.__dict__.pop("_amp_found_inf", None)
            # Decided here, which reads the flag back: a captured step then
            # runs as written from this point, and is not replayed.
            self._found_inf = bool(self._found_inf_var.numpy().reshape(-1)[0])
        if self._found_inf:
            # Clear the gradients the skipped step would have consumed.
            # `Optimizer.backward` *accumulates* into `pg["grads"]` and
            # `post_step` is what normally empties them, so a skip that left
            # them in place would add the next iteration's gradients on top of
            # these -- and these contain the inf that caused the skip. Measured
            # without this line: one overflow poisoned every later step, the
            # scale backed off to 1 and stayed there, and training silently did
            # not happen. torch does not need it because its callers zero the
            # gradients themselves each iteration.
            # `getattr`, because the only thing torch's GradScaler requires of
            # an optimizer is `step`, and the compatibility suite steps objects
            # that implement exactly that. An optimizer without `zero_grad` is
            # one whose gradients the caller manages.
            zero_grad = getattr(optimizer, "zero_grad", None)
            if callable(zero_grad):
                zero_grad()
            return None
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        """Back off after an overflow, or grow after a clean run."""
        if not self._enabled:
            return
        if self._on_device() and new_scale is None and self._found_inf_var is not None:
            return self._update_on_device()
        if new_scale is not None and self._device is not None:
            self._pull()
            self._device = None     # a new scale from outside: start over
        if new_scale is not None:
            self._scale = float(new_scale.item()
                                if isinstance(new_scale, _jittor().Var)
                                else new_scale)
            self._found_inf = False
            return
        if self._found_inf:
            # Never below 1: a scale under 1 shrinks the gradients, which is
            # the opposite of the point.
            self._scale = max(1.0, self._scale * self._backoff_factor)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        self._found_inf = False

    def _update_on_device(self):
        """`update` as arithmetic on the device state, for a captured step."""
        jt = _jittor()
        from jittor._runtime import step_capture
        step_capture.guard(lambda: (self._growth_factor, self._backoff_factor,
                                    self._growth_interval))
        scale, tracker = self._device
        found = self._found_inf_var.float32().reshape((1,))
        grown = tracker + 1
        grow = (grown >= float(self._growth_interval)).float32()
        backed_off = jt.maximum(scale * self._backoff_factor, 1.0)
        kept = grow * (scale * self._growth_factor) + (1 - grow) * scale
        scale.update((found * backed_off + (1 - found) * kept).stop_grad())
        tracker.update(((1 - found) * (1 - grow) * grown).stop_grad())
        self._found_inf_var = None
        self._found_inf = False

    def state_dict(self):
        """The scale and the policy that moves it.

        These are the five keys ``torch.amp.GradScaler`` writes, and the names
        are torch's, so a checkpoint written here loads there and the other way
        round. A disabled scaler has no state, and returns ``{}``, also like
        torch.
        """
        if not self._enabled:
            return {}
        self._pull()
        return {"scale": self._scale,
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._growth_tracker}

    def load_state_dict(self, state_dict):
        if not self._enabled:
            return
        if not state_dict:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler.")
        self._scale = float(state_dict["scale"])
        self._growth_factor = float(state_dict["growth_factor"])
        self._backoff_factor = float(state_dict["backoff_factor"])
        self._growth_interval = int(state_dict["growth_interval"])
        self._growth_tracker = int(state_dict["_growth_tracker"])
        self._device = None

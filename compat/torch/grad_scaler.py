"""``torch.amp.GradScaler`` and its legacy per-device subclasses.

Split out of :mod:`jittor.compat.torch.amp` so each domain module stays inside
the compatibility layer's per-file line budget, and named after the module torch
itself puts the class in (``torch.amp.grad_scaler``).
"""

import enum as _enum
import math as _math
import warnings as _warnings

import numpy as _np

from jittor.amp import GradScaler as _NativeGradScaler
from jittor._core.dtypes import dtype_name as _dtype_name

from .fidelity import Fidelity, register_fidelity


class OptState(_enum.Enum):
    """torch.amp.grad_scaler.OptState: where one optimizer is in the step."""
    READY = 0
    UNSCALED = 1
    STEPPED = 2


class GradScaler(_NativeGradScaler):
    """torch.amp.GradScaler: native scaling with Torch's eager lifecycle.

    Gradient arithmetic is delegated to :class:`jittor.amp.GradScaler`; this
    class owns Torch's eager per-optimizer state and device-first signature.
    During step capture, it keeps the native single-optimizer device path.

    A bare number in the ``device`` slot can only be the older call, so it is
    shifted rather than taken for a device.
    """

    _DEFAULT_GROWTH_INTERVAL = 2000

    def __init__(self, device="cuda", init_scale=2.0 ** 16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000, enabled=True):
        if not isinstance(device, str) and getattr(device, "type", None) is None:
            if growth_interval != self._DEFAULT_GROWTH_INTERVAL:
                raise TypeError(
                    "GradScaler(init_scale, ...) is the pre-2.3 positional "
                    "order and has no slot left for `enabled`; pass the "
                    "remaining arguments by keyword, or use "
                    "torch.cuda.amp.GradScaler")
            device, init_scale, growth_factor, backoff_factor, growth_interval = (
                "cuda", device, init_scale, growth_factor, backoff_factor)
        self._torch_device = (getattr(device, "type", None) or str(device)).split(":", 1)[0]
        super().__init__(init_scale=init_scale, growth_factor=growth_factor,
                         backoff_factor=backoff_factor,
                         growth_interval=growth_interval, enabled=enabled)
        self._scale_initialized = False
        self._per_optimizer_states = {}

    @staticmethod
    def _capture_active():
        from jittor._runtime import step_capture
        return step_capture.active()

    def _check_scale(self, operation):
        if not self._scale_initialized:
            raise AssertionError(
                "Attempted %s but _scale is None.  This may indicate your "
                "script did not use scaler.scale(loss or outputs) earlier "
                "in the iteration." % operation)

    def _optimizer_state(self, optimizer):
        key = id(optimizer)
        state = self._per_optimizer_states.get(key)
        if state is None or state["optimizer"] is not optimizer:
            state = {"optimizer": optimizer, "stage": OptState.READY,
                     "found_inf": None}
            self._per_optimizer_states[key] = state
        return state

    @staticmethod
    def _discard_skipped_grads(optimizer):
        groups = getattr(optimizer, "param_groups", ())
        has_jittor_slots = False
        for group in groups:
            if "grads" in group:
                group.pop("grads", None)
                has_jittor_slots = True
        if has_jittor_slots:
            state = getattr(optimizer, "__dict__", {})
            if "_Optimizer__zero_grad" in state:
                object.__setattr__(optimizer, "_Optimizer__zero_grad", True)
            if "_grad_map" in state:
                object.__setattr__(optimizer, "_grad_map", {})
            if "_torch_backward_advanced_n_step" in state:
                object.__setattr__(optimizer, "_torch_backward_advanced_n_step", False)
            return
        zero_grad = getattr(optimizer, "zero_grad", None)
        if callable(zero_grad):
            zero_grad()

    def scale(self, outputs):
        output_device = getattr(getattr(outputs, "device", None), "type", None)
        if (self._enabled and self._torch_device == "cuda"
                and output_device == "cuda"
                and _dtype_name(getattr(outputs, "dtype", None)) == "float16"
                and not self._capture_active()):
            # Torch CUDA GradScaler multiplies by its typed float32 scale
            # tensor. CUDA's weak-scalar and 0-d tensor protocols differ for
            # fp16 at the overflow boundary, so keep the scale typed here.
            from jittor.amp.grad_scaler import _device_scalar

            scale = _device_scalar(outputs, self._scale).cast("float16")
            result = outputs * scale
            if self._scale > self._FLOAT16_MAX:
                _warnings.warn(
                    "GradScaler.scale() on a float16 tensor with scale %g "
                    "overflows float16 (max %g). Scale a float32 copy of the "
                    "loss -- scaler.scale(loss.float32()) -- or lower "
                    "init_scale; otherwise every step is skipped and the "
                    "scale collapses to 1."
                    % (self._scale, self._FLOAT16_MAX),
                    RuntimeWarning, stacklevel=2)
        else:
            result = super().scale(outputs)
        if self._enabled:
            self._scale_initialized = True
        return result

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        self._scale_initialized = False
        self._per_optimizer_states.clear()
        self._found_inf = False
        self._found_inf_var = None
        self._unscaled = False
        return result

    def unscale_(self, optimizer):
        if not self._enabled or self._capture_active():
            return super().unscale_(optimizer)
        self._check_scale("unscale_")
        state = self._optimizer_state(optimizer)
        if state["stage"] is OptState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since "
                "the last update().")
        if state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        grads = self._grads(optimizer)
        if any(_dtype_name(grad.dtype) == "float16" for grad in grads):
            raise ValueError(
                "Attempting to unscale FP16 gradients.")

        if not grads:
            state["stage"] = OptState.UNSCALED
            state["found_inf"] = None
            self._unscaled = True
            return

        if self._scale == 0.0:
            import jittor as jt

            nonempty = [grad for grad in grads if grad.numel()]
            if nonempty:
                finite = jt.isfinite(jt.concat([
                    jt.isfinite(grad).reshape((-1,)) for grad in nonempty
                ])).all().item()
                found_inf = not bool(finite)
            else:
                found_inf = False
            inf = _np.float32(_np.inf)
            for grad in nonempty:
                unscaled = grad * inf
                if _dtype_name(unscaled.dtype) != _dtype_name(grad.dtype):
                    unscaled = unscaled.cast(_dtype_name(grad.dtype))
                grad.update(unscaled)
        else:
            super().unscale_(optimizer)
            if self._found_inf_var is not None:
                found_inf = bool(self._found_inf_var.numpy().reshape(-1)[0])
                self._found_inf_var = None
                self._found_inf = found_inf
            else:
                found_inf = bool(self._found_inf)
            from .types import _var_is_cpu_resident
            if (any(_dtype_name(grad.dtype) == "float64" for grad in grads)
                    and all(_var_is_cpu_resident(grad) for grad in grads)):
                import jittor as jt
                found_inf = any(
                    not bool(jt.isfinite(grad).all().item())
                    for grad in grads if grad.numel())
                self._found_inf = found_inf

        state["stage"] = OptState.UNSCALED
        state["found_inf"] = found_inf
        self._found_inf = bool(self._found_inf or found_inf)
        self._unscaled = True

    def step(self, optimizer, *args, **kwargs):
        if not self._enabled:
            return super().step(optimizer, *args, **kwargs)
        closure = kwargs.get("closure")
        if args and closure is None:
            closure = args[0]
        if closure is not None:
            raise RuntimeError(
                "Closure use is not currently supported if GradScaler is enabled.")
        if self._capture_active():
            return super().step(optimizer, *args, **kwargs)

        self._check_scale("step")
        state = self._optimizer_state(optimizer)
        if state["stage"] is OptState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update() for "
                "this optimizer.")
        if state["stage"] is OptState.READY:
            self.unscale_(optimizer)
        if state["found_inf"] is None:
            raise AssertionError(
                "No inf checks were recorded for this optimizer.")

        state["stage"] = OptState.STEPPED
        self._unscaled = False
        if state["found_inf"]:
            self._discard_skipped_grads(optimizer)
            return None
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        if not self._enabled or self._capture_active():
            return super().update(new_scale)
        self._check_scale("update")
        if new_scale is not None:
            if self._device is not None:
                self._pull()
                self._device = None
            result = super().update(new_scale)
            self._per_optimizer_states.clear()
            self._unscaled = False
            self._found_inf_var = None
            return result

        recorded = [state["found_inf"] for state in
                    self._per_optimizer_states.values()
                    if state["found_inf"] is not None]
        if not recorded:
            raise AssertionError("No inf checks were recorded prior to update.")

        if self._device is not None:
            self._pull()
            self._device = None
        found_inf = any(recorded)
        scale = float(_np.float32(self._scale))
        if found_inf:
            with _np.errstate(over="ignore", invalid="ignore"):
                self._scale = float(_np.float32(
                    _np.float32(scale) * _np.float32(self._backoff_factor)))
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                with _np.errstate(over="ignore", invalid="ignore"):
                    candidate = _np.float32(
                        _np.float32(scale) * _np.float32(self._growth_factor))
                if _math.isfinite(float(candidate)):
                    self._scale = float(candidate)
                self._growth_tracker = 0

        self._found_inf = False
        self._found_inf_var = None
        self._unscaled = False
        self._per_optimizer_states.clear()


class _LegacyDeviceGradScaler(GradScaler):
    """``torch.cuda.amp.GradScaler`` / ``torch.cpu.amp.GradScaler``.

    torch keeps these as subclasses with the device baked in and the pre-2.3
    argument order, and code in the wild still writes
    ``GradScaler(2.0 ** 10, enabled=use_amp)``.
    """

    _device_name = "cuda"

    def __init__(self, init_scale=2.0 ** 16, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000, enabled=True):
        super().__init__(self._device_name, init_scale=init_scale,
                         growth_factor=growth_factor,
                         backoff_factor=backoff_factor,
                         growth_interval=growth_interval, enabled=enabled)


class _CudaGradScaler(_LegacyDeviceGradScaler):
    _device_name = "cuda"


class _CpuGradScaler(_LegacyDeviceGradScaler):
    _device_name = "cpu"


#: Historical private name; ``jittor.torch_compat._GradScaler`` is pickled by
#: checked-in payloads and re-exported from :mod:`jittor.compat.torch`.
_GradScaler = GradScaler


_GRAD_SCALER_FIDELITY = (
    (("torch.GradScaler", "torch.amp.GradScaler",
      "torch.amp.grad_scaler.GradScaler"),
     GradScaler, Fidelity.APPROXIMATE,
     "Dynamic loss scaling over the jittor optimizer bridge: scale/unscale_/"
     "step/update and the five state_dict keys match torch, the inf check is "
     "one flat host-visible reduction per optimizer rather than torch's "
     "per-device _found_inf tensors; step capture retains the native "
     "single-optimizer path and multi-device replication is not implemented"),
    (("torch.cuda.amp.GradScaler",), _CudaGradScaler, Fidelity.APPROXIMATE,
     "Legacy cuda-pinned subclass of torch.amp.GradScaler (init_scale first)"),
    (("torch.cpu.amp.GradScaler",), _CpuGradScaler, Fidelity.APPROXIMATE,
     "Legacy cpu-pinned subclass of torch.amp.GradScaler (init_scale first)"),
)

for _apis, _implementation, _level, _detail in _GRAD_SCALER_FIDELITY:
    for _api in _apis:
        register_fidelity(_api, _implementation, _level, _detail)
del _apis, _implementation, _level, _detail, _api

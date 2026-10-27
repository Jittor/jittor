"""``torch.amp.GradScaler`` and its legacy per-device subclasses.

Split out of :mod:`jittor.compat.torch.amp` so each domain module stays inside
the compatibility layer's per-file line budget, and named after the module torch
itself puts the class in (``torch.amp.grad_scaler``).
"""

import enum as _enum

import numpy as np

import jittor as jt
from jittor._core.dtypes import dtype_name

from jittor.amp import GradScaler as _NativeGradScaler

from .fidelity import Fidelity, register_fidelity


class OptState(_enum.Enum):
    """torch.amp.grad_scaler.OptState: where one optimizer is in the step."""
    READY = 0
    UNSCALED = 1
    STEPPED = 2


class GradScaler(_NativeGradScaler):
    """torch.amp.GradScaler: the native scaler with torch's signature.

    The algorithm -- scale the loss, unscale the gradients, skip the step on a
    non-finite one, grow or back off -- is :class:`jittor.amp.GradScaler`, and
    nothing about it is torch-specific. What this class adds is the shape of
    torch's API: the device-first argument order of torch >= 2.3, the pre-2.3
    order that ``torch.cuda.amp.GradScaler`` still uses, per-optimizer stage
    checks, and the fidelity records below. Keeping one implementation means
    a fix to the scaling
    policy cannot land in one mode and miss the other.

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
            if (init_scale, growth_factor, backoff_factor) == (2.0 ** 16, 2.0, 0.5):
                # A lone legacy scale has no supplied tail to shift. Shifting
                # defaults would turn backoff_factor into the default 2.0.
                device, init_scale = "cuda", device
            else:
                device, init_scale, growth_factor, backoff_factor, growth_interval = (
                    "cuda", device, init_scale, growth_factor, backoff_factor)
        self._device_type = getattr(device, "type", None) or str(device)
        if enabled:
            if growth_factor <= 1.0:
                raise AssertionError("The growth factor must be > 1.0.")
            if backoff_factor >= 1.0:
                raise AssertionError("The backoff factor must be < 1.0.")
        super().__init__(init_scale=init_scale, growth_factor=growth_factor,
                         backoff_factor=backoff_factor,
                         growth_interval=growth_interval, enabled=enabled)
        self._scale_initialized = False
        self._optimizer_states = {}

    # Native optimizers clear consumed gradients on a skipped step. Torch's
    # caller owns zero_grad(), and may inspect non-finite gradients afterwards.
    _clear_grads_on_skip = False

    def scale(self, outputs):
        if self._enabled and isinstance(outputs, jt.Var) and not self._scale_initialized:
            with np.errstate(over="ignore", invalid="ignore"):
                initial = np.float32(self._scale)
            if np.isfinite(self._scale) and not np.isfinite(initial):
                raise RuntimeError("value cannot be converted to type float without overflow")
            self._scale = float(initial)
            self._scale_initialized = True
        return super().scale(outputs)

    def _check_initialized(self, operation):
        assert self._scale_initialized, (
            "Attempted %s but _scale is None. This may indicate your script did "
            "not use scaler.scale(loss or outputs) earlier in the iteration."
            % operation)

    def _optimizer_state(self, optimizer):
        return self._optimizer_states.setdefault(id(optimizer), {
            "stage": OptState.READY, "checked": False,
            "found_inf": False, "found_inf_var": None})

    def unscale_(self, optimizer):
        if not self._enabled:
            return
        self._check_initialized("unscale_")
        state = self._optimizer_state(optimizer)
        if state["stage"] is OptState.UNSCALED:
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        if state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")
        grads = self._grads(optimizer)
        if any(dtype_name(grad.dtype) == "float16" for grad in grads):
            raise ValueError("Attempting to unscale FP16 gradients.")
        super().unscale_(optimizer)
        state.update(stage=OptState.UNSCALED, checked=bool(grads),
                     found_inf=self._found_inf, found_inf_var=self._found_inf_var)

    def step(self, optimizer, *args, **kwargs):
        if not self._enabled:
            return optimizer.step(*args, **kwargs)
        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled.")
        self._check_initialized("step")
        state = self._optimizer_state(optimizer)
        if state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")
        if state["stage"] is OptState.READY:
            self.unscale_(optimizer)
        assert state["checked"], "No inf checks were recorded for this optimizer."
        self._unscaled = True
        self._found_inf = state["found_inf"]
        self._found_inf_var = state["found_inf_var"]
        result = super().step(optimizer, *args, **kwargs)
        state["stage"] = OptState.STEPPED
        return result

    def update(self, new_scale=None):
        if not self._enabled:
            return
        self._check_initialized("update")
        if new_scale is None:
            states = [state for state in self._optimizer_states.values() if state["checked"]]
            assert states, "No inf checks were recorded prior to update."
            self._found_inf = any(state["found_inf"] for state in states)
            flags = [state["found_inf_var"] for state in states
                     if state["found_inf_var"] is not None]
            self._found_inf_var = jt.stack(flags).any() if flags else None
            if self._found_inf and self._found_inf_var is not None:
                self._found_inf_var = jt.ones_like(self._found_inf_var)
        elif isinstance(new_scale, jt.Var):
            reason = ("new_scale should be a float or a 1-element tensor "
                      "with requires_grad=False on the scaler device.")
            assert new_scale.numel() == 1, reason
            assert not new_scale.requires_grad, reason
            assert new_scale.device.type == self._device_type, reason
            if new_scale.ndim != 0:
                raise RuntimeError(
                    "output with shape [] does not match the broadcast shape %s"
                    % list(new_scale.shape))
        elif not isinstance(new_scale, float):
            raise TypeError("new_scale should be a float or a 1-element tensor")
        if isinstance(new_scale, float):
            with np.errstate(over="ignore", invalid="ignore"):
                converted = np.float32(new_scale)
            if np.isfinite(new_scale) and not np.isfinite(converted):
                raise RuntimeError("value cannot be converted to type float without overflow")
        super().update(new_scale)
        if new_scale is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                self._scale = float(np.float32(self._scale))
        self._optimizer_states.clear()
        self._unscaled = False


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
     "one host-visible reduction per optimizer rather than torch's per-device "
     "_found_inf tensors. OptState guards each optimizer independently; "
     "_check_inf_per_device and multi-device replication are not implemented"),
    (("torch.cuda.amp.GradScaler",), _CudaGradScaler, Fidelity.APPROXIMATE,
     "Legacy cuda-pinned subclass of torch.amp.GradScaler (init_scale first)"),
    (("torch.cpu.amp.GradScaler",), _CpuGradScaler, Fidelity.APPROXIMATE,
     "Legacy cpu-pinned subclass of torch.amp.GradScaler (init_scale first)"),
)

for _apis, _implementation, _level, _detail in _GRAD_SCALER_FIDELITY:
    for _api in _apis:
        register_fidelity(_api, _implementation, _level, _detail)
del _apis, _implementation, _level, _detail, _api

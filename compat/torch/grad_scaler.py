"""``torch.amp.GradScaler`` and its legacy per-device subclasses.

Split out of :mod:`jittor.compat.torch.amp` so each domain module stays inside
the compatibility layer's per-file line budget, and named after the module torch
itself puts the class in (``torch.amp.grad_scaler``).
"""

from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import collections.abc as _collections_abc
import enum as _enum

import numpy as np
import jittor as jt

from .fidelity import Fidelity, register_fidelity


class OptState(_enum.Enum):
    """torch.amp.grad_scaler.OptState: where one optimizer is in the step."""
    READY = 0
    UNSCALED = 1
    STEPPED = 2


class GradScaler:
    """Functional fp16 dynamic loss scaler (torch.amp.GradScaler).

    Works with the jittor optimizer bridge: scale(loss).backward() routes scaled
    grads into the optimizer; step() unscales, SKIPS the step on inf/nan, and
    update() grows/backs off the scale. bf16 doesn't need scaling but this is
    correct (and required) for fp16 mixed-precision training.

    The signature is torch >= 2.3's, device first.  The pre-2.3 order --
    ``GradScaler(init_scale, ...)``, which ``torch.cuda.amp.GradScaler`` still
    uses -- lives on :class:`_LegacyDeviceGradScaler`; a bare number in the
    ``device`` slot here can only be that older call, so it is shifted rather
    than taken for a device.
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
        self._device = getattr(device, "type", None) or str(device)
        self._enabled = bool(enabled)
        self._init_scale = float(init_scale)
        self._scale = float(init_scale)
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._growth_tracker = 0
        self._found_inf = False
        self._unscaled = False

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        return self._scale if self._enabled else 1.0

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

    def scale(self, outputs):
        """Multiply a tensor, or every tensor in a container, by the scale."""
        if not self._enabled:
            return outputs
        if isinstance(outputs, jt.Var):
            return outputs * self._scale
        if isinstance(outputs, (list, tuple)):
            return type(outputs)(self.scale(item) for item in outputs)
        if isinstance(outputs, _collections_abc.Iterable):
            return (self.scale(item) for item in outputs)
        raise ValueError("outputs must be a Tensor or an iterable of Tensors")

    def _grads(self, opt):
        gs = []
        for pg in getattr(opt, "param_groups", []):
            for g in (pg.get("grads", []) or []):
                if g is not None:
                    gs.append(g)
        return gs

    def unscale_(self, opt):
        if not self._enabled:
            return
        inv = np.float32(1.0 / self._scale)
        flattened = []
        for g in self._grads(opt):
            if not g.numel():
                continue
            unscaled = g * inv
            if _jittor_dtype_name(unscaled.dtype) != _jittor_dtype_name(g.dtype):
                unscaled = unscaled.cast(_jittor_dtype_name(g.dtype))
            g.update(unscaled)
            flattened.append(unscaled.cast("float32").reshape((-1,)))
        # Optimizer.step still needs a host decision to skip state updates, but
        # one flat reduction avoids both per-gradient reductions and per-gradient
        # D2H syncs. The finite check consumes the values actually assigned back
        # to the gradients, including any low-precision overflow from unscaling.
        self._found_inf = (
            not bool(jt.isfinite(jt.concat(flattened)).all().item())
            if flattened else False
        )
        self._unscaled = True

    def step(self, opt, *a, **k):
        if not self._enabled:
            return opt.step(*a, **k)
        if not self._unscaled:
            self.unscale_(opt)
        self._unscaled = False
        if self._found_inf:
            return None  # skip optimizer step on overflow
        return opt.step(*a, **k)

    def update(self, new_scale=None):
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale.item()
                                if isinstance(new_scale, jt.Var) else new_scale)
            self._found_inf = False
            return
        if self._found_inf:
            self._scale = max(1.0, self._scale * self._backoff_factor)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        self._found_inf = False

    def state_dict(self):
        """torch's five keys, so a torch checkpoint round-trips.

        This used to return ``{"scale", "growth_tracker"}``: loading a real
        torch scaler checkpoint silently restored nothing but the scale, and a
        checkpoint written here silently reset a torch scaler's growth
        parameters to their defaults.  A disabled scaler has no state, and torch
        returns ``{}`` for it.
        """
        if not self._enabled:
            return {}
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
     "one flat host-visible reduction over all gradients rather than torch's "
     "per-device _found_inf tensors, so per-optimizer state (OptState, "
     "_check_inf_per_device) and multi-device replication are not implemented"),
    (("torch.cuda.amp.GradScaler",), _CudaGradScaler, Fidelity.APPROXIMATE,
     "Legacy cuda-pinned subclass of torch.amp.GradScaler (init_scale first)"),
    (("torch.cpu.amp.GradScaler",), _CpuGradScaler, Fidelity.APPROXIMATE,
     "Legacy cpu-pinned subclass of torch.amp.GradScaler (init_scale first)"),
)

for _apis, _implementation, _level, _detail in _GRAD_SCALER_FIDELITY:
    for _api in _apis:
        register_fidelity(_api, _implementation, _level, _detail)
del _apis, _implementation, _level, _detail, _api

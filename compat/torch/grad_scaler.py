"""``torch.amp.GradScaler`` and its legacy per-device subclasses.

Split out of :mod:`jittor.compat.torch.amp` so each domain module stays inside
the compatibility layer's per-file line budget, and named after the module torch
itself puts the class in (``torch.amp.grad_scaler``).
"""

import enum as _enum

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
    order that ``torch.cuda.amp.GradScaler`` still uses, and the fidelity
    records below. Keeping one implementation means a fix to the scaling
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
            device, init_scale, growth_factor, backoff_factor, growth_interval = (
                "cuda", device, init_scale, growth_factor, backoff_factor)
        self._device = getattr(device, "type", None) or str(device)
        super().__init__(init_scale=init_scale, growth_factor=growth_factor,
                         backoff_factor=backoff_factor,
                         growth_interval=growth_interval, enabled=enabled)
















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

"""Query surface of the emulated XLA runtime.

Answers describe the *emulated* launch: processes map one-to-one onto visible
CUDA devices (or one CPU process), there is a single host, and the device type
is ``"GPU"`` or ``"CPU"``.  ``"TPU"`` is never returned; asking for TPU or SPMD
raises instead of pretending.
"""

from __future__ import absolute_import

import torch

from . import _emulation


def device_type():
    """``"GPU"`` when CUDA is visible, else ``"CPU"`` (never ``"TPU"``)."""
    return _emulation.device_type()


def is_bf16_supported():
    """Best-effort bf16 capability of the emulated device."""
    if device_type() != "GPU":
        return False
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(probe):
        try:
            return bool(probe())
        except Exception:  # pragma: no cover - partial shims
            return True
    return True


def world_size():
    """Number of processes taking part in the emulated launch."""
    return _emulation.world_size()


def global_ordinal():
    """Global rank of this process."""
    return _emulation.ordinal()


def local_ordinal():
    """Local rank of this process (one host, so it equals the global rank)."""
    return _emulation.local_ordinal()


def process_index():
    """Alias of :func:`global_ordinal` used by newer torch_xla releases."""
    return global_ordinal()


def process_count():
    """Alias of :func:`world_size` used by newer torch_xla releases."""
    return world_size()


def host_index():
    """Index of this host; the emulation launches one host, so always 0."""
    return 0


def global_device_count():
    """Number of devices in the launch (ranks outside a spawn: visible GPUs)."""
    if _emulation.session():
        return world_size()
    return max(_emulation.visible_device_count(), 1)


def local_device_count():
    """Number of devices visible to this process (1 per spawned rank)."""
    return max(_emulation.visible_device_count(), 1)


def addressable_device_count():
    """Devices this process can address; equals :func:`local_device_count`."""
    return local_device_count()


def global_runtime_device_count():
    """Alias of :func:`global_device_count`."""
    return global_device_count()


def addressable_runtime_device_count():
    """Alias of :func:`local_device_count`."""
    return local_device_count()


def runtime_device_attributes(device):
    """Minimal attribute record for one emulated device."""
    return {
        "device_type": device_type(),
        "device": str(device),
        "name": str(device),
    }


def global_runtime_device_attributes():
    """Attribute records for every device in the launch."""
    count = global_device_count()
    return [runtime_device_attributes("xla:%d" % index) for index in range(count)]


def get_master_ip():
    """The emulation is single-host; the master is always local."""
    return "127.0.0.1"


def is_spmd():
    """SPMD sharding is not emulated."""
    return False


def use_spmd(auto=False):
    """SPMD sharding is not emulated; fail instead of ignoring the request."""
    raise NotImplementedError(
        "torch_xla emulation: SPMD sharding is not implemented (Jittor device "
        "placement is process-local)"
    )


def set_device_type(pjrt_device):
    """Validate the requested device kind; the emulation has no PJRT switch.

    ``GPU``/``CUDA`` and ``CPU`` are accepted because they describe what the
    emulation already does.  ``TPU`` is rejected: no TPU is present.
    """
    requested = str(pjrt_device).strip().upper()
    if requested in ("GPU", "CUDA", "CPU"):
        return None
    if requested == "TPU":
        raise NotImplementedError(
            "torch_xla emulation: TPU device type requested, but this layer "
            "maps XLA client programs onto Jittor CUDA/CPU devices"
        )
    raise ValueError("unknown PJRT device type: %r" % (pjrt_device,))

"""Pass-through emulation of ``torch_xla.distributed.parallel_loader``.

Real ``MpDeviceLoader`` starts XLA preloading threads that enqueue batches on
the device ahead of the training loop.  Jittor executes eagerly, so the
emulation simply iterates the wrapped loader and moves each batch to the
emulated device on the main thread.  The observable contract (an iterator of
batches already placed on the device) is preserved; the prefetch overlap and
the XLA graph barriers are not.
"""

from __future__ import absolute_import

from .. import _emulation
from ..core.xla_model import mark_step


class MpDeviceLoader(object):
    """Iterate ``loader`` and place each batch on ``device``.

    Attributes the loader owns (``dataset``, ``batch_sampler``, ``rng_types``,
    ...) are forwarded lazily instead of copied: Accelerate subclasses this
    class and may define some of them as read-only properties, which an eager
    ``setattr`` in ``__init__`` would break.
    """

    def __init__(self, loader, device, **kwargs):
        self._loader = loader
        self.device = device
        self._kwargs = dict(kwargs)

    def __iter__(self):
        for batch in self._loader:
            # Preserve the batch structure: a bare tensor batch stays a
            # tensor, a dict stays a dict.  ``send_cpu_data_to_device`` wraps a
            # bare tensor in a list to describe its return shape; iteration
            # should hand back exactly what the loader produced.
            yield _emulation.to_device(batch, self.device)
            mark_step()

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._loader, name)


class ParallelLoader(object):
    """Minimal legacy wrapper: ``per_device_loader(device)`` per device."""

    def __init__(self, loader, devices, batchdim=0, loader_kwargs=None):
        self._loader = loader
        self._devices = list(devices)
        self._batchdim = batchdim
        self._loader_kwargs = dict(loader_kwargs or {})

    def per_device_loader(self, device):
        return MpDeviceLoader(self._loader, device, **self._loader_kwargs)


__all__ = ["MpDeviceLoader", "ParallelLoader"]

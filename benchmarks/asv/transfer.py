"""Host/device transfer costs, and what a read does to where a tensor lives.

The operator benchmarks time arithmetic on tensors that are already resident.
Nothing here was measured, and two real defects lived in that gap:

* ``x.cpu()`` allocated its destination on the *device*, so moving a tensor off
  the card transiently needed twice its size there and a tensor over half the
  card could not be moved at all -- the operation whose purpose is to free
  device memory failing exactly when that is what you need. Invisible to
  ``track_working_set_bytes``, which reads what is still allocated after the
  call, not the spike during it.
* Reading a device tensor relocates it. ``numpy()``, ``repr()`` and an element
  read each move the storage to the host; the next device operation migrates it
  back, measured at 215x the cost of the same operation without the read
  (KI-MEM-002). A benchmark that only times resident tensors never sees it.

So this measures the transfer itself, the peak it costs, and the read-then-use
pattern -- with the reduction spelling alongside as the control, because
``loss.item()`` does *not* relocate its source and a benchmark that lumped them
together would report a regression on the one spelling that behaves.
"""

from __future__ import annotations

import numpy as np

from ._shared import (
    as_numpy,
    backend_tensor,
    cleanup_backend,
    load_backend,
    peak_device_bytes,
    reset_peak_device_bytes,
    synchronize,
)


#: Large enough that a transfer dominates launch overhead, small enough to fit
#: beside a second copy on any supported card -- the doubling defect this
#: guards would otherwise fail to allocate instead of showing up as a number.
ELEMENTS = 8 * 1024 * 1024  # 32 MiB of float32


class TransferBenchmarks:
    params = (["jittor", "torch"], ["cuda"])
    param_names = ["backend", "device"]
    number = 1
    repeat = (3, 7, 30.0)
    rounds = 1
    timeout = 300

    def setup(self, backend_name, device):
        self.backend_name = backend_name
        self.device = device
        self.backend = load_backend(backend_name, device)
        rng = np.random.default_rng(20260909)
        self.host = rng.standard_normal(ELEMENTS).astype("float32")
        self.resident = backend_tensor(backend_name, self.backend, self.host, device)
        synchronize(backend_name, self.backend, device)
        # A transfer benchmark that silently measured a wrong-sized or empty
        # tensor would still produce a tidy curve.
        check = as_numpy(backend_name, self.resident)
        if check.shape != self.host.shape or not np.isfinite(check).all():
            raise RuntimeError("the resident tensor is not the array that was uploaded")

    def _to_host(self, tensor):
        if self.backend_name == "torch":
            return tensor.cpu()
        return tensor.cpu()

    def _to_device(self, tensor):
        if self.backend_name == "torch":
            return tensor.cuda()
        return tensor.cuda()

    # -- the transfers themselves --------------------------------------------

    def time_host_to_device(self, backend_name, device):
        tensor = backend_tensor(backend_name, self.backend, self.host, device)
        synchronize(backend_name, self.backend, device)
        self._keep = tensor

    def time_device_to_host(self, backend_name, device):
        self._keep = self._to_host(self.resident)
        synchronize(backend_name, self.backend, device)

    def track_device_to_host_peak_bytes(self, backend_name, device):
        """Peak device memory while moving a tensor *off* the device.

        Should be about one tensor. Two means the destination was allocated on
        the card, which is the defect this exists to keep fixed.
        """
        # The pool-growth proxy is only a peak if the pool starts empty, and
        # anything this object still holds keeps it from emptying. The first
        # draft left the previous measurement's tensors on `self` and read
        # 2.00x on a build where the defect is fixed -- a benchmark reporting a
        # defect that is not there, from its own leftovers.
        self._release()
        reset_peak_device_bytes(backend_name, self.backend, device)
        resident = backend_tensor(backend_name, self.backend, self.host, device)
        synchronize(backend_name, self.backend, device)
        host_copy = self._to_host(resident)
        synchronize(backend_name, self.backend, device)
        peak = peak_device_bytes(backend_name, self.backend, device)
        del resident, host_copy
        return peak

    track_device_to_host_peak_bytes.unit = "bytes"

    def track_device_to_host_peak_ratio(self, backend_name, device):
        """The same measurement as a multiple of the tensor, which is the claim.

        A byte count moves with the machine and the tensor size; the ratio is
        what a reader can judge. One is correct, two is the defect.
        """
        peak = self.track_device_to_host_peak_bytes(backend_name, device)
        return float(peak) / float(self.host.nbytes)

    def _release(self):
        """Drop what an earlier timing kept alive, so the pool can empty."""
        if hasattr(self, "_keep"):
            del self._keep

    track_device_to_host_peak_ratio.unit = "x tensor"

    # -- what a read costs the next device operation -------------------------

    def time_device_op_after_a_host_read(self, backend_name, device):
        """Read the tensor, then use it on the device again.

        This is the shape of KI-MEM-002: the read moves the storage to the host
        and the device operation has to bring it back. Paired with the control
        below, which is the identical operation without the read -- the gap
        between the two *is* the finding, and neither number means much alone.
        """
        tensor = backend_tensor(backend_name, self.backend, self.host, device)
        synchronize(backend_name, self.backend, device)
        as_numpy(backend_name, tensor)
        self._keep = self._sum(tensor)
        synchronize(backend_name, self.backend, device)

    def time_device_op_without_a_read(self, backend_name, device):
        tensor = backend_tensor(backend_name, self.backend, self.host, device)
        synchronize(backend_name, self.backend, device)
        self._keep = self._sum(tensor)
        synchronize(backend_name, self.backend, device)

    def time_reduction_read_then_device_op(self, backend_name, device):
        """The control that must stay fast: a reduction read does not relocate.

        ``u.sum().item()`` leaves ``u`` on the device because the scalar is a
        new value rather than a view of it, so this is the spelling every
        training loop uses. Measured separately so a fix for the two above can
        be seen not to have cost anything here.
        """
        tensor = backend_tensor(backend_name, self.backend, self.host, device)
        synchronize(backend_name, self.backend, device)
        float(self._sum(tensor).item())
        self._keep = self._sum(tensor)
        synchronize(backend_name, self.backend, device)

    def _sum(self, tensor):
        if self.backend_name == "torch":
            return tensor.sum()
        return self.backend.sum(tensor)

    def teardown(self, backend_name, device):
        backend = getattr(self, "backend", None)
        for name in ("host", "resident", "_keep"):
            if hasattr(self, name):
                delattr(self, name)
        if backend is not None:
            cleanup_backend(backend_name, backend)

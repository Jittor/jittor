"""Real CUDA placement, transfer and synchronization through BackendOps."""

import numpy as np
import pytest

from _helpers.backend_probe import _backend_probe


def test_cuda_execution_uses_backend_allocation_copy_and_sync_callbacks():
    import jittor as jt

    if not jt.has_cuda:
        pytest.skip("CUDA runtime required")
    assert "cuda" in jt.core.registered_backends()
    assert jt.core.backend_device_count("cuda") >= 1
    probe = _backend_probe(jt)
    data = np.arange(67, dtype=np.float32)
    with jt.flag_scope(use_cuda=1, use_cuda_managed_allocator=0):
        jt.sync_all(True)
        probe.native_backend_probe_install(1)
        try:
            value = jt.array(data) * 2 + 5
            value.sync()
            assert value.device_id >= 0
            assert value.location() == "device"
            jt.sync_all(True)
            np.testing.assert_array_equal(value.numpy(), data * 2 + 5)
            allocations, copies, async_copies, syncs, uploads, downloads = (
                probe.native_backend_probe_counts()
            )
            assert allocations > 0
            assert copies + async_copies > 0
            assert syncs > 0
            assert uploads > 0
            assert downloads > 0
        finally:
            probe.native_backend_probe_restore()
        np.testing.assert_array_equal((value + 1).numpy(), data * 2 + 6)


def test_registered_managed_pools_preserve_device_placement():
    import jittor as jt

    if not jt.has_cuda:
        pytest.skip("CUDA runtime required")
    data = np.arange(17, dtype=np.float32)
    for device in range(min(2, jt.get_device_count())):
        with jt.runtime.scope(use_cuda=1, use_cuda_managed_allocator=1, device_id=device):
            value = jt.array(data) * 3
            jt.sync_all(True)
            assert value.device_id == device
            assert value.location() == "device"
            np.testing.assert_array_equal(value.numpy(), data * 3)
            del value

"""``Var.device_ptr_ready``: a device pointer without the per-read device sync.

The Triton bridge materialises a launch's operands once (``jt.sync_all``) and
then reads one device pointer per argument, so the accessor it uses must not
synchronise. ``device_raw_ptr`` calls ``sync(true, false)``; on MiniMax-H3's
video VAE that per-argument device sync made an 80 s decode 3.4x slower -- 4
reads x 9,072 launches, 41.5 s of a 131 s bridge total by the bridge's own
``JITTOR_TRITON_STATS``, and 63,816 ``cudaDeviceSynchronize`` against fp32's 312.

A holder that has never been allocated is the case this accessor cannot simply
skip: ``Var::allocator`` is null until ``alloc`` (see ``var.h``) and the
residency test dereferences it. Writing this accessor by copying
``device_raw_ptr``'s body *without* its ``sync`` -- which is how it was first
written -- segfaults the interpreter the moment anything reads the pointer of a
not-yet-materialised holder, and nothing here covered that.
"""

from _helpers import capability as _test_capability

import ctypes
import unittest

import numpy as np

import jittor as jt


_CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9


def _pointer_device(pointer):
    """Which CUDA device owns ``pointer``, or None if the driver rejects it."""
    library = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int(-1)
    result = library.cuPointerGetAttribute(
        ctypes.byref(device), _CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
        ctypes.c_void_p(int(pointer)))
    return device.value if result == 0 else None


class TestDevicePtrReady(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not _test_capability.check_accelerator('cuda', backend=jt).enabled \
                or _test_capability.device_count('cuda', backend=jt) < 1:
            raise unittest.SkipTest("a real CUDA device is required")

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=1, device_id=0)
        self._scope.__enter__()

    def tearDown(self):
        jt.sync_all(True)
        self._scope.__exit__(None, None, None)

    def test_a_holder_with_no_allocation_still_yields_a_pointer(self):
        """The null-allocator case: nothing has called ``Var::alloc`` yet."""
        a = jt.random((8,))
        jt.sync_all(True)
        pending = a + 1                      # producer has not run: no allocation
        self.assertEqual(pending.location(), "none")

        pointer = pending.device_ptr_ready

        self.assertNotEqual(pointer, 0)
        self.assertEqual(pending.device_ptr_ready, pending.device_raw_ptr)
        np.testing.assert_allclose(pending.numpy(), np.asarray(a.numpy()) + 1)

    def test_the_pointer_is_a_device_pointer(self):
        x = jt.random((16,))
        jt.sync_all(True)
        self.assertEqual(_pointer_device(x.device_ptr_ready), 0)
        self.assertEqual(x.device_ptr_ready, x.device_raw_ptr)

    def test_a_host_resident_operand_is_migrated_before_it_is_reported(self):
        """A host Var must never be handed to a device kernel as its own address."""
        x = jt.random((8,))
        jt.sync_all(True)
        host = x.cpu()

        pointer = host.device_ptr_ready

        self.assertNotEqual(pointer, 0)
        self.assertEqual(host.location(), "device")
        self.assertEqual(_pointer_device(pointer), 0)
        np.testing.assert_allclose(host.numpy(), x.numpy())


if __name__ == "__main__":
    unittest.main()

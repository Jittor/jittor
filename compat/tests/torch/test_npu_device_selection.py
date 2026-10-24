"""NPU selection must reach the real driver without changing CPU factories."""
import ctypes
import unittest

import numpy as np
import jittor as jt
import torch

from jittor._runtime.fallback import forbid_backend_fallbacks


class TestNPUDeviceSelection(unittest.TestCase):
    def test_selection_driver_and_implicit_index(self):
        if not jt.compiler.has_acl:
            self.skipTest("requires ACL")
        count = jt.core.backend_device_count("acl")
        self.assertGreaterEqual(count, 2, "two visible NPUs required")
        driver = ctypes.CDLL("libascendcl.so")
        driver.aclrtGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        driver.aclrtGetDevice.restype = ctypes.c_int
        original = jt.core.current_device()
        try:
            with jt.runtime.scope(use_cuda=0), forbid_backend_fallbacks():
                before = jt.core.backend_fallback_count()
                self.assertTrue(torch.npu.is_available())
                self.assertEqual(torch.npu.device_count(), count)
                for target in (0, 1, 0):
                    torch.npu.set_device(target)
                    actual = ctypes.c_int32(-1)
                    self.assertEqual(driver.aclrtGetDevice(ctypes.byref(actual)), 0)
                    self.assertEqual(actual.value, target)
                    self.assertEqual(jt.core.current_device(), target)
                    self.assertEqual(torch.npu.current_device(), target)
                    # Bare NPU means the selected device, not hard-coded zero.
                    x = torch.tensor([1., 2., 3.], device="npu", requires_grad=True)
                    loss = (x * x).sum()
                    loss.backward()
                    for value in (x, loss, x.grad):
                        self.assertIsNotNone(value)
                        value.sync()
                        self.assertEqual(value.location(), "device")
                        self.assertEqual(value.device_id, target)
                    np.testing.assert_array_equal(x.grad.detach().cpu().numpy(), [2., 4., 6.])
                    for value in (torch.zeros(1), jt.zeros(1)):
                        value.sync()
                        self.assertEqual(value.location(), "cpu")
                    self.assertEqual(jt.runtime.use_cuda, 0)
                    self.assertEqual(torch.npu.current_device(), target)
                self.assertEqual(jt.core.backend_fallback_count() - before, 0)
        finally:
            # Native owner restores state even while the facade is under test.
            jt.core.set_device(original)


if __name__ == "__main__":
    unittest.main()

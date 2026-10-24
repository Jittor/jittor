"""Real NPU names must match an independent direct ACL driver query."""
import ctypes
import unittest

import jittor as jt
import torch
from _helpers import capability
from jittor._runtime.fallback import forbid_backend_fallbacks


class TestNPUDeviceName(unittest.TestCase):
    def setUp(self):
        if not capability.check_accelerator("acl", backend=jt).enabled:
            self.skipTest("requires a real ACL NPU")
        self.driver = ctypes.CDLL("libascendcl.so")
        self.driver.aclrtGetSocName.argtypes = []
        self.driver.aclrtGetSocName.restype = ctypes.c_char_p
        self.driver.aclrtGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.driver.aclrtGetDevice.restype = ctypes.c_int
        self.driver.aclrtSetDevice.argtypes = [ctypes.c_int32]
        self.driver.aclrtSetDevice.restype = ctypes.c_int
        # Initialize the normal provider before the independent driver read.
        self.original = jt.core.current_device()

    def driver_device(self):
        current = ctypes.c_int32()
        self.assertEqual(self.driver.aclrtGetDevice(ctypes.byref(current)), 0)
        return current.value

    def driver_name(self, index):
        previous = self.driver_device()
        self.assertEqual(self.driver.aclrtSetDevice(index), 0)
        try:
            result = self.driver.aclrtGetSocName()
            self.assertTrue(result, "ACL driver did not supply a real SoC name")
            return result.decode("utf-8")
        finally:
            self.assertEqual(self.driver.aclrtSetDevice(previous), 0)

    def test_names_match_driver_and_preserve_current_device_and_cpu_factory(self):
        from jittor.backends.acl import get_device_name as native_name
        with jt.runtime.scope(use_cuda=0), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            for index in range(jt.core.get_device_count()):
                expected = self.driver_name(index)
                self.assertEqual(native_name(index), expected)
                for argument in (index, "npu:%d" % index, torch.device("npu", index)):
                    self.assertEqual(torch.npu.get_device_name(argument), expected)
                    self.assertEqual(jt.core.current_device(), self.original)
                    self.assertEqual(self.driver_device(), self.original)
                    self.assertEqual(jt.runtime.use_cuda, 0)
            self.assertEqual(torch.npu.get_device_name(), self.driver_name(self.original))
            self.assertEqual(torch.npu.get_device_name("npu"), self.driver_name(self.original))
            for cpu in (torch.empty(2), torch.tensor([1., 2.])):
                cpu.sync()
                self.assertEqual(cpu.location(), "cpu")
            self.assertEqual(jt.core.backend_fallback_count(), before)

    def test_invalid_devices_fail_without_switching(self):
        with jt.runtime.scope(use_cuda=0):
            for argument in (True, False, -1, jt.core.get_device_count(), "cpu", "cuda:0", 0.5):
                with self.subTest(argument=argument), self.assertRaises((ValueError, TypeError)):
                    torch.npu.get_device_name(argument)
                self.assertEqual(jt.core.current_device(), self.original)
                self.assertEqual(self.driver_device(), self.original)
                self.assertEqual(jt.runtime.use_cuda, 0)


if __name__ == "__main__":
    unittest.main()

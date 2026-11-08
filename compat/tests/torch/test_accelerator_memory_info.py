"""The device-agnostic memory query must retain the driver's memory semantics."""
import unittest
from unittest import mock

import jittor
import torch
from jittor.compat.torch.installers.cuda import api


class TestAcceleratorMemoryInfo(unittest.TestCase):
    def test_driver_values_are_returned(self):
        with mock.patch.object(api, "_cuda_mem_get_info_fn", return_value=lambda: (4096, 8192)), \
                mock.patch.object(api, "current_device", return_value=0):
            self.assertEqual(torch.accelerator.get_memory_info(0), (4096, 8192))

    def test_driver_failure_is_not_hidden(self):
        def fail():
            raise RuntimeError("memory query failed")
        with mock.patch.object(api, "_cuda_mem_get_info_fn", return_value=fail), \
                mock.patch.object(api, "current_device", return_value=0):
            with self.assertRaisesRegex(RuntimeError, "memory query failed"):
                torch.accelerator.get_memory_info(0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a real CUDA device")
    def test_real_device_total_and_current_device_are_preserved(self):
        index = torch.cuda.current_device()
        free, total = torch.accelerator.get_memory_info(torch.device("cuda", index))
        self.assertGreater(free, 0)
        self.assertLessEqual(free, total)
        self.assertEqual(total, torch.cuda.get_device_properties(index).total_memory)
        self.assertEqual(torch.cuda.current_device(), index)


if __name__ == "__main__":
    unittest.main()

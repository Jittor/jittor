"""A refused device allocation raises ``torch.OutOfMemoryError`` with PyTorch's
message.

``torch.cuda.OutOfMemoryError`` existed, but nothing raised it: the refusal was
a plain jittor error worded "out of memory on the accelerator", so handlers
that recognise an OOM by its class or by PyTorch's "CUDA out of memory." text
to retry with a smaller batch did not see one.
"""

import traceback
import unittest

import torch

from _helpers import capability as _test_capability

import jittor as jt

HAS_CUDA = bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)


@unittest.skipIf(not HAS_CUDA, "needs a CUDA device: the refusal under test is the device's")
class TestTorchOutOfMemory(unittest.TestCase):
    def test_the_refusal_is_torchs_error_with_torchs_message(self):
        ooms = torch.cuda.memory_stats()["num_ooms"]
        with self.assertRaises(torch.cuda.OutOfMemoryError) as caught:
            torch.empty(1 << 40, device="cuda").sum().item()
        error = caught.exception
        self.assertIs(torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)
        self.assertIsInstance(error, RuntimeError)
        self.assertIsInstance(error, jt.JittorOutOfMemoryError)
        self.assertTrue(traceback.format_exception_only(type(error), error)[-1].startswith(
            "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4096.00 GiB. GPU "))
        self.assertIn("is allocated by PyTorch, and ", str(error))
        self.assertIn("is reserved by PyTorch but unallocated.", str(error))
        self.assertEqual(torch.cuda.memory_stats()["num_ooms"], ooms + 1)
        # The device still works afterwards.
        self.assertEqual(torch.ones(4, device="cuda").sum().item(), 4.0)


if __name__ == "__main__":
    unittest.main()

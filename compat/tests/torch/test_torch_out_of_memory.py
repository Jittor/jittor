"""A refused device allocation is caught as ``torch.OutOfMemoryError`` and
opens with "CUDA out of memory.".

``torch.cuda.OutOfMemoryError`` existed, but nothing raised it: the refusal was
a plain jittor error worded "out of memory on the accelerator", so handlers
that recognise an OOM by its class or by the "CUDA out of memory." text to
retry with a smaller batch did not see one. The error is still jittor's: its
class and wording name Jittor, not PyTorch.
"""

import traceback
import unittest

import torch

from _helpers import capability as _test_capability

import jittor as jt

HAS_CUDA = bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)


@unittest.skipIf(not HAS_CUDA, "needs a CUDA device: the refusal under test is the device's")
class TestTorchOutOfMemory(unittest.TestCase):
    def test_the_refusal_is_caught_as_torchs_error_and_names_jittor(self):
        ooms = torch.cuda.memory_stats()["num_ooms"]
        with self.assertRaises(torch.cuda.OutOfMemoryError) as caught:
            torch.empty(1 << 40, device="cuda").sum().item()
        error = caught.exception
        self.assertIs(torch.OutOfMemoryError, torch.cuda.OutOfMemoryError)
        self.assertIs(torch.OutOfMemoryError, jt.JittorOutOfMemoryError)
        self.assertIsInstance(error, RuntimeError)
        self.assertTrue(traceback.format_exception_only(type(error), error)[-1].startswith(
            "jittor.JittorOutOfMemoryError: CUDA out of memory. Tried to allocate 4096.00 GiB. GPU "))
        self.assertIn("is allocated by Jittor, and ", str(error))
        self.assertIn("is reserved by Jittor but unallocated.", str(error))
        self.assertNotIn("PyTorch", str(error))
        self.assertEqual(torch.cuda.memory_stats()["num_ooms"], ooms + 1)
        # The device still works afterwards.
        self.assertEqual(torch.ones(4, device="cuda").sum().item(), 4.0)


if __name__ == "__main__":
    unittest.main()

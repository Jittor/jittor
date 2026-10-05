"""Kernel selection follows the frontend's placement when no input is placed.

A factory with no ``device=`` builds on torch's default device, the CPU. Its
operators often have no tensor inputs (``torch.arange`` is an ``index``), so
the only placement there is the one the frontend asked for -- which the native
kernel selector, a binding that takes no Var, did not see: it picked the
accelerator's kernel, and the op then ran on the host. On ACL that kernel is a
code op with no host source, and ``torch.arange`` raised.
"""
import unittest

import torch
from jittor._runtime import dispatch


@unittest.skipUnless(torch.cuda.is_available(), "No CUDA found")
class TestKernelSelectFrontendPlacement(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def accelerator_index(inshape, dim=None, dtype="int32"):
            self.calls.append(tuple(inshape))
            return None  # decline, so the native index still builds the result

        dispatch.register_kernel("tensor.index", "cuda", accelerator_index)
        self.addCleanup(dispatch.unregister_kernel, "tensor.index", "cuda", accelerator_index)

    def test_default_device_selects_no_accelerator_kernel(self):
        x = torch.arange(96)
        self.assertEqual(x.device.type, "cpu")
        self.assertEqual(x.numpy()[:3].tolist(), [0, 1, 2])
        self.assertEqual(self.calls, [])

    def test_accelerator_device_still_selects_it(self):
        torch.arange(96, device="cuda").numpy()
        self.assertEqual(self.calls, [(96,)])


if __name__ == "__main__":
    unittest.main()

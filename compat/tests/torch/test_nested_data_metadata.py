"""Public data-only jagged metadata, independently on CPU and real CUDA.

Trainable jagged autograd-node metadata is a separate unsupported contract;
never substitute packed-values metadata for that node. Arrays stay in RAM.
"""
from contextlib import ExitStack
import os
import sys
import unittest

import numpy as np


class TestNestedDataMetadata(unittest.TestCase):
    def _run(self, device):
        if os.environ.get("JITTOR_TEST_DEVICES", "cpu") != device:
            self.skipTest("node not selected by JITTOR_TEST_DEVICES")
        import torch
        jt = sys.modules.get("jittor")
        if jt is None:
            self.assertTrue(hasattr(torch, "_C"))
            self.assertFalse(hasattr(torch, "_torch_compat_install_context"))
        else:
            from jittor.compat.torch.publication import namespace_owner
            self.assertIs(namespace_owner(torch), jt)
        with ExitStack() as stack:
            if jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(jt.runtime.scope(
                    use_cuda=int(device == "cuda"), device_id=0 if device == "cuda" else -1,
                    backend_fallback="error"))
                stack.enter_context(forbid_backend_fallbacks())
                self.assertEqual(jt.core.backend_fallback_count(), 0)
            elif device == "cpu":
                self.assertFalse(torch.cuda.is_initialized())
            if device == "cuda":
                self.assertTrue(torch.cuda.is_available())
                self.assertEqual(torch.cuda.current_device(), 0)
            cases = 0
            for dtype in (torch.int64, torch.float32):
                for disabled in (False, True):
                    for holes in (False, True):
                        with self.subTest(dtype=str(dtype), no_grad=disabled, holes=holes):
                            values = torch.arange(12, dtype=dtype, device=device).reshape(6, 2)
                            offsets = torch.tensor([0, 2, 6], dtype=torch.int64, device=device)
                            lengths = (torch.tensor([1, 3], dtype=torch.int64, device=device)
                                       if holes else None)
                            before = (values.requires_grad, values.is_leaf, values.grad_fn is None,
                                      values.grad is None, values.retains_grad)
                            with torch.set_grad_enabled(not disabled):
                                nested = torch.nested.nested_tensor_from_jagged(
                                    values, offsets, lengths=lengths, jagged_dim=1)
                            for _ in range(2):
                                self.assertFalse(nested.requires_grad)
                                self.assertIsNone(nested.grad)
                                self.assertIsNone(nested.grad_fn)
                                self.assertIs(nested.is_leaf, True)
                                self.assertIs(nested.retains_grad, False)
                            self.assertEqual(before, (values.requires_grad, values.is_leaf,
                                values.grad_fn is None, values.grad is None, values.retains_grad))
                            packed = nested.values()
                            self.assertEqual(packed.device.type, device)
                            if device == "cuda":
                                self.assertEqual(packed.device.index, 0)
                            if jt is not None:
                                native = getattr(packed, "_var", packed)
                                native.sync()
                                self.assertEqual(jt.core.dispatch_context([native]),
                                                 (device, 0 if device == "cuda" else -1))
                                self.assertEqual(native.location(), "device" if device == "cuda" else "cpu")
                            with torch.no_grad():
                                actual = np.array(packed.clone().cpu().numpy(), copy=True)
                            self.assertEqual(actual.shape, (6, 2))
                            self.assertTrue(bool(np.array_equal(actual, np.arange(12).reshape(6, 2))),
                                            "complete packed values changed during metadata reads")
                            cases += 1
            self.assertEqual(cases, 8)
            if jt is not None:
                self.assertEqual(jt.core.backend_fallback_count(), 0)
            elif device == "cpu":
                self.assertFalse(torch.cuda.is_initialized())

    def test_cpu(self):
        self._run("cpu")

    def test_cuda(self):
        self._run("cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)

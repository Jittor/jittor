import unittest

import numpy as np
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks


@unittest.skipIf(not _test_capability.check_accelerator("acl", backend=jt).enabled, "No ACL found")
class TestACLTorchNumPyIndex(unittest.TestCase):
    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_numpy_integer_array_index_values_and_gradient(self):
        import torch
        self.assertIsNot(torch, jt)
        self.assertIs(torch.Tensor._frontend_backend, jt)
        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            for dtype in (np.int32, np.int64):
                with self.subTest(dtype=dtype):
                    source = torch.tensor([10.0, 20.0, 30.0], device="npu", requires_grad=True)
                    indices = np.array([2, 0], dtype=dtype)
                    selected = source[indices]
                    selected.sync()
                    self.assertEqual(selected.location(), "device")
                    np.testing.assert_array_equal(selected.detach().cpu().numpy(), [30.0, 10.0])
                    selected.sum().backward()
                    source.grad.sync()
                    self.assertEqual(source.grad.location(), "device")
                    np.testing.assert_array_equal(source.grad.detach().cpu().numpy(), [1.0, 0.0, 1.0])
        self.assertEqual(jt.core.backend_fallback_count(), before)

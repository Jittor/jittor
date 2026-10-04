"""Uniform repeat_interleave must stay on ACL and preserve gradients."""
import unittest

import jittor as jt
import numpy as np
import torch
from _helpers import capability


@unittest.skipUnless(capability.check_accelerator("acl", backend=jt).enabled, "No ACL found")
class TestRepeatInterleaveACL(unittest.TestCase):
    def test_uniform_repeat_values_and_gradients(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="npu:0", requires_grad=True)
            y = x.repeat_interleave(2, dim=1)
            weights = torch.tensor([[1., 2., 3., 4., 5., 6.],
                                    [7., 8., 9., 10., 11., 12.]], device="npu:0")
            (y * weights).sum().backward()
            np.testing.assert_array_equal(y.detach().cpu().numpy(),
                                          [[1., 1., 2., 2., 3., 3.],
                                           [4., 4., 5., 5., 6., 6.]])
            np.testing.assert_array_equal(x.grad.detach().cpu().numpy(),
                                          [[3., 7., 11.], [15., 19., 23.]])

            grouped = torch.tensor([0.5], device="npu:0", requires_grad=True)
            repeated = grouped.repeat_interleave(2)
            (repeated * torch.tensor([1., 3.], device="npu:0")).sum().backward()
            np.testing.assert_array_equal(repeated.detach().cpu().numpy(), [0.5, 0.5])
            np.testing.assert_array_equal(grouped.grad.detach().cpu().numpy(), [4.])
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)


if __name__ == "__main__":
    unittest.main()

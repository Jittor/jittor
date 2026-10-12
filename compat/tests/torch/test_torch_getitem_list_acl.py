"""ACL advanced indexing with a Python list and trailing Ellipsis."""
import unittest

import numpy as np
import jittor as jt
import torch
from _helpers import capability


@unittest.skipUnless(capability.check_accelerator("acl", backend=jt).enabled, "No ACL found")
class TestListIndexACL(unittest.TestCase):
    def test_list_ellipsis_values_and_accumulated_gradient(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]],
                             device="npu:0", requires_grad=True)
            selected = x[[2, 0, 2], ...]
            np.testing.assert_array_equal(selected.detach().cpu().numpy(),
                                          [[5., 6.], [1., 2.], [5., 6.]])
            selected.sum().backward()
            np.testing.assert_array_equal(x.grad.detach().cpu().numpy(),
                                          [[1., 1.], [0., 0.], [2., 2.]])
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)


if __name__ == "__main__":
    unittest.main()

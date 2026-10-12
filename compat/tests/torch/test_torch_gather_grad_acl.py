"""ACL gather backward must scatter into the original input shape."""
import unittest

import numpy as np
import jittor as jt
import torch
from _helpers import capability


@unittest.skipUnless(capability.check_accelerator("acl", backend=jt).enabled, "No ACL found")
class TestGatherGradientACL(unittest.TestCase):
    def test_sparse_indices_restore_full_input_gradient(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            x = torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]],
                             device="npu:0", requires_grad=True)
            index = torch.tensor([[2, 2], [3, 0]], dtype=torch.long, device="npu:0")
            y = torch.gather(x, 1, index)
            y.sum().backward()
            np.testing.assert_array_equal(y.detach().cpu().numpy(), [[3., 3.], [8., 5.]])
            np.testing.assert_array_equal(
                x.grad.detach().cpu().numpy(),
                np.array([[0., 0., 2., 0.], [1., 0., 0., 1.]], dtype=np.float32))
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)


if __name__ == "__main__":
    unittest.main()

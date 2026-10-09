"""DPO starts at zero policy-reference log-ratio; its gradient must survive ACL."""
import unittest

import numpy as np
import jittor as jt
import torch
import torch.nn.functional as F
from _helpers import capability


@unittest.skipUnless(capability.check_accelerator("acl", backend=jt).enabled, "No ACL found")
class TestLogSigmoidACL(unittest.TestCase):
    def test_value_and_derivative_at_zero_and_extremes(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks

        x_np = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
        expected = -np.logaddexp(0.0, -x_np)
        grad_expected = 1.0 / (1.0 + np.exp(x_np.astype(np.float64)))
        with jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            x = torch.tensor(x_np, device="npu:0", requires_grad=True)
            y = F.logsigmoid(x)
            y.sum().backward()
            np.testing.assert_allclose(y.detach().cpu().numpy(), expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(x.grad.detach().cpu().numpy(), grad_expected, atol=1e-5, rtol=1e-5)
            self.assertAlmostEqual(float(x.grad[2].item()), 0.5, places=5)
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)


if __name__ == "__main__":
    unittest.main()

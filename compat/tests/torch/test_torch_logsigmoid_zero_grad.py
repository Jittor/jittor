"""``F.logsigmoid`` is ``-softplus(-x)``: stable at both ends, and with the
derivative 1/2 at zero that preference losses (DPO) start from. The piecewise
``min(x, 0) - log1p(exp(-|x|))`` form it replaced had a zero subgradient there.
"""
import unittest

import numpy as np
import torch
import torch.nn.functional as F


class TestLogSigmoid(unittest.TestCase):
    def _check(self, device):
        x_np = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
        x = torch.tensor(x_np, device=device, requires_grad=True)
        y = F.logsigmoid(x)
        y.sum().backward()
        np.testing.assert_allclose(y.detach().cpu().numpy(), -np.logaddexp(0.0, -x_np),
                                   atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(x.grad.detach().cpu().numpy(),
                                   1.0 / (1.0 + np.exp(x_np.astype(np.float64))),
                                   atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(float(x.grad[2].item()), 0.5, places=5)

    def test_cpu(self):
        self._check("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA found")
    def test_cuda(self):
        self._check("cuda")


if __name__ == "__main__":
    unittest.main()

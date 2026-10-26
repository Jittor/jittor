"""Public zero_grad preserves missing slots over repeated AdamW updates."""
from _helpers import capability as _test_capability

import unittest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
import numpy as np
import torch


class TestAdamWUnusedGradient(unittest.TestCase):
    def test_repeated_zero_grad_cpu(self):
        self._run_device("cpu", 0)

    def test_repeated_zero_grad_cuda(self):
        _test_capability.require_accelerator("cuda")
        self._run_device("cuda", 1)

    def _run_device(self, device, use_cuda):
        fallbacks_before = int(jt.core.backend_fallback_count())
        with jt.runtime.scope(use_cuda=use_cuda, backend_fallback="error"), \
                forbid_backend_fallbacks():
            for set_to_none in (False, True):
                for unused_first in (False, True):
                    with self.subTest(device=device, set_to_none=set_to_none,
                                      unused_first=unused_first):
                        self._run(device, set_to_none, unused_first)
            jt.sync_all(True)
            self.assertEqual(int(jt.core.backend_fallback_count()), fallbacks_before)

    def _run(self, device, set_to_none, unused_first):
        used = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=device))
        late = torch.nn.Parameter(torch.tensor([3.0, -4.0], device=device))
        parameters = [late, used] if unused_first else [used, late]
        optimizer = torch.optim.AdamW(
            parameters, lr=0.05, weight_decay=0.2, foreach=False, fused=False)
        used_reference = np.array([1.0, -2.0], dtype="float32")
        moment = np.zeros(2, dtype="float32")
        variance = np.zeros(2, dtype="float32")
        for step in range(5):
            published = used.grad
            optimizer.zero_grad(set_to_none=set_to_none)
            if published is not None and not set_to_none:
                self.assertIs(used.grad, published)
                np.testing.assert_array_equal(
                    published.detach().cpu().numpy(), np.zeros(2, dtype="float32"))
            if step < 3:
                self.assertIsNone(late.grad)
                self.assertNotIn(late, optimizer.state)
            loss = used.square().sum()
            if step >= 3:
                loss = loss + late.square().sum()
            loss.backward()
            expected_device = "cuda:0" if device == "cuda" else "cpu"
            self.assertEqual(str(used.device), expected_device)
            self.assertEqual(str(used.grad.device), expected_device)
            if step < 3:
                self.assertIsNone(late.grad)
            else:
                self.assertEqual(str(late.grad.device), expected_device)
            gradient = 2.0 * used_reference
            moment = 0.9 * moment + 0.1 * gradient
            variance = 0.999 * variance + 0.001 * gradient * gradient
            used_reference = used_reference * (1.0 - 0.05 * 0.2) - 0.05 * (
                moment / (1.0 - 0.9 ** (step + 1))) / (
                    np.sqrt(variance / (1.0 - 0.999 ** (step + 1))) + 1e-8)
            optimizer.step()
            np.testing.assert_allclose(
                used.detach().cpu().numpy(), used_reference, atol=2e-6, rtol=2e-6)
            self.assertEqual(int(optimizer.state[used]["step"]), step + 1)
            if step < 3:
                self.assertIsNone(late.grad)
                self.assertNotIn(late, optimizer.state)
                np.testing.assert_array_equal(late.detach().cpu().numpy(), [3.0, -4.0])
            else:
                self.assertEqual(int(optimizer.state[late]["step"]), step - 2)


if __name__ == "__main__":
    unittest.main()

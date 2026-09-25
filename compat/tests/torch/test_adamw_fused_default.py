"""torch.optim.AdamW takes the fused list update on CUDA by default.

torch's default AdamW (fused=None, foreach=None) is a multi-tensor kernel on
CUDA. The per-parameter update here built about ten graph nodes per tensor,
and on a 450-tensor diffusers UNet that step alone outweighed the forward:
ddpm training went 210.7 -> 105.6 ms a step with the fused update. What the
fused path must preserve is the answer, and the caller's choice to opt out.
"""

import unittest

import numpy as np
import torch

import jittor as jt
from _helpers import capability as _test_capability


def _train(fused=None, steps=4):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 33), torch.nn.GELU(),
                                torch.nn.Linear(33, 4)).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1, fused=fused)
    rng = np.random.RandomState(1)
    for _ in range(steps):
        x = torch.tensor(rng.randn(8, 16).astype("float32"), device="cuda")
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return [p.detach().cpu().numpy() for p in model.parameters()]


@unittest.skipIf(not _test_capability.check_accelerator("cuda", backend=jt).enabled,
                 "no usable CUDA in this build")
class TestAdamwFusedByDefault(unittest.TestCase):
    def setUp(self):
        from jittor.backends.cuda.kernels.optim import fused_adamw_cuda
        self.module = fused_adamw_cuda
        self.calls = []
        original = fused_adamw_cuda._cuda_fused_adamw_updates

        def counted(*args):
            self.calls.append(1)
            return original(*args)

        from jittor._runtime.dispatch import override_kernel
        scope = override_kernel("optim.adamw_fused", "cuda", counted,
                                supports=fused_adamw_cuda._supports)
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_the_default_is_fused_and_matches_the_per_parameter_update(self):
        with jt.flag_scope(use_cuda=1):
            reference = _train(fused=False)
            self.assertEqual(self.calls, [])
            fused = _train()
        self.assertEqual(len(self.calls), 4)
        for want, got in zip(reference, fused):
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

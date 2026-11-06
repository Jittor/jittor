"""A captured optimizer step must apply its update exactly once.

The capture's post-trace sync used to be forced whenever the traced step
published gradients, even when the call had already materialized every root.
That re-ran the kept graph and applied the optimizer update a second time:
AdamW's fused update materializes its state roots inside the traced call, so
a four-step loss trajectory skipped a step. These tests compare the captured
trajectory against an eager twin on the same initial weights.
"""

import numpy as np
import pytest
import torch

from jittor._runtime import step_capture


def _inputs(device):
    generator = np.random.RandomState(0)
    x = torch.from_numpy(generator.randn(16, 4).astype("float32")).to(device)
    y = torch.from_numpy(generator.randn(16, 1).astype("float32")).to(device)
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1).to(device)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(
            (0.1 * generator.randn(1, 4)).astype("float32")))
        model.bias.copy_(torch.from_numpy(np.zeros(1, "float32")))
    return x, y, model


def _trajectory(kind, device, captured, steps=4):
    x, y, model = _inputs(device)
    if kind == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    def step():
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        return loss

    runner = step_capture.StepCapture(step) if captured else step
    losses, params = [], []
    for _ in range(steps):
        losses.append(float(runner().numpy()))
        params.append(model.weight.detach().cpu().numpy().copy())
    return losses, params, getattr(runner, "refused", None), getattr(runner, "stats", None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", ["adamw", "sgd"])
def test_captured_optimizer_step_matches_eager(kind):
    eager_losses, eager_params, _, _ = _trajectory(kind, "cuda", False)
    losses, params, refused, stats = _trajectory(kind, "cuda", True)
    assert refused is None
    assert stats["replayed"] >= 2
    np.testing.assert_allclose(losses, eager_losses, rtol=1e-6, atol=1e-7)
    for captured, eager in zip(params, eager_params):
        np.testing.assert_allclose(captured, eager, rtol=1e-6, atol=1e-7)

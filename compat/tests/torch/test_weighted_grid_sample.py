import numpy as np
import pytest
import torch

import jittor as jt
from _helpers.capability import require_accelerator


@pytest.mark.cuda
@pytest.mark.parametrize("levels, points, channels, queries",
                         [(3, 3, 5, 47), (4, 4, 5, 47), (3, 5, 19, 145)])
def test_stacked_bilinear_samples_reduce_with_shared_weights(levels, points, channels, queries):
    require_accelerator("cuda")
    rng = np.random.default_rng(1889)
    batches = 2
    images = [rng.normal(size=(batches, 5 + level, 6 + level, channels)).astype("float32")
              for level in range(levels)]
    grids = [rng.uniform(-1.5, 1.5, size=(batches, queries, points, 2)).astype("float32")
             for _ in range(levels)]
    weights = rng.normal(size=(batches, queries, levels * points)).astype("float32")

    def calculate(device, separate=False):
        samples = [
            torch.nn.functional.grid_sample(
                torch.tensor(image, device=device).permute(0, 3, 1, 2),
                torch.tensor(grid, device=device), align_corners=bool(level % 2))
            for level, (image, grid) in enumerate(zip(images, grids))
        ]
        packed = torch.stack(samples, dim=-2).flatten(-2)
        if separate:
            packed.sync()
        return (packed * torch.tensor(weights, device=device)[:, None]).sum(-1)

    with torch.no_grad(), jt.flag_scope(backend_fallback="error"):
        expected = calculate("cpu")
        separate = calculate("cuda", separate=True)
        result = calculate("cuda")
        assert result._producer_name() == "code"
        assert result.device.type == "cuda"
        np.testing.assert_allclose(result.cpu().numpy(), separate.cpu().numpy(), rtol=2e-5, atol=3e-6)
        np.testing.assert_allclose(result.cpu().numpy(), expected.numpy(), rtol=2e-5, atol=3e-6)

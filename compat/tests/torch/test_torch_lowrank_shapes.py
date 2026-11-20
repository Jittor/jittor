"""Low-rank SVD and centered PCA keep matrix axes apart from batch axes."""

import numpy as np
import pytest
import torch

from _helpers.capability import require_accelerator


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("shape", ((4, 7), (7, 4), (2, 4, 7)))
def test_lowrank_svd_returns_right_vector_columns(device, shape):
    if device == "cuda":
        require_accelerator("cuda")
    values = np.random.default_rng(617).standard_normal(shape).astype(np.float32)
    source = torch.tensor(values, device=device)
    u, s, v = torch.svd_lowrank(source, q=3)
    batch = shape[:-2]
    assert tuple(u.shape) == batch + (shape[-2], 3)
    assert tuple(s.shape) == batch + (3,)
    assert tuple(v.shape) == batch + (shape[-1], 3)
    assert u.device == s.device == v.device == source.device

    actual = ((u * s.unsqueeze(-2)) @ v.transpose(-2, -1)).detach().cpu().numpy()
    independent_u, independent_s, independent_vh = np.linalg.svd(values, full_matrices=False)
    expected = (independent_u[..., :3] * independent_s[..., None, :3]) @ independent_vh[..., :3, :]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_pca_lowrank_centers_each_batch(device):
    if device == "cuda":
        require_accelerator("cuda")
    values = np.random.default_rng(41).standard_normal((2, 4, 7)).astype(np.float32)
    values += np.array([3.0, -5.0], dtype=np.float32)[:, None, None]
    source = torch.tensor(values, device=device)
    u, s, v = torch.pca_lowrank(source, q=4, center=True)
    actual = ((u * s.unsqueeze(-2)) @ v.transpose(-2, -1)).detach().cpu().numpy()
    expected = values - values.mean(-2, keepdims=True)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

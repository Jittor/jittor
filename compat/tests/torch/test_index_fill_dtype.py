"""index_fill_ preserves values as well as dtype, on CPU or CUDA.

Select JITTOR_TEST_DEVICE=cpu or cuda in separate test processes.
"""
import os

import numpy as np
import torch
import pytest


@pytest.fixture
def device():
    return os.environ.get("JITTOR_TEST_DEVICE", "cuda")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_index_fill_preserves_dtype(dtype, device):
    x = torch.zeros((1, 8), dtype=dtype, device=device)
    idx = torch.tensor([0], dtype=torch.int64, device=device)
    out = x.index_fill_(0, idx, 0)
    assert out is x
    assert x.dtype is dtype
    assert out.dtype is dtype


@pytest.mark.parametrize("dtype,large", [
    (torch.int32, 2 ** 24 + 1),
    (torch.int64, 2 ** 60 + 7),
])
@pytest.mark.parametrize("fill_large", [False, True])
def test_index_fill_preserves_exact_large_integer_values(device, dtype, large, fill_large):
    numpy_dtype = np.int32 if dtype == torch.int32 else np.int64
    original = np.array([[large, large + 2, -large],
                         [large + 4, -large - 2, large + 6]], dtype=numpy_dtype)
    x = torch.tensor(original, dtype=dtype, device=device)
    indices = torch.tensor([1], dtype=torch.int64, device=device)
    fill = large + 10 if fill_large else 0
    expected = original.copy()
    expected[:, 1] = fill
    assert x.index_fill_(1, indices, fill) is x
    assert x.dtype == dtype
    np.testing.assert_array_equal(x.cpu().numpy(), expected)


def test_index_fill_overwrites_selected_nan_and_infinity(device):
    x = torch.tensor([float("nan"), 3.0, float("inf"), float("nan")],
                     dtype=torch.float32, device=device)
    indices = torch.tensor([0, 2], dtype=torch.int64, device=device)
    x.index_fill_(0, indices, 7.0)
    np.testing.assert_equal(x.cpu().numpy(), [7.0, 3.0, 7.0, float("nan")])


def test_index_fill_nan_value_does_not_poison_unselected_elements(device):
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    indices = torch.tensor([1], dtype=torch.int64, device=device)
    x.index_fill_(0, indices, float("nan"))
    np.testing.assert_equal(x.cpu().numpy(), [1.0, float("nan"), 3.0])

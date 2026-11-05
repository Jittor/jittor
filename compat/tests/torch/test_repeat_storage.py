"""Repeat produces independent, writable storage even for unit repetitions."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("repetitions", [(2, 1, 1), (1, 1)])
def test_repeat_has_writable_independent_storage(device, repetitions):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.tensor([[1., 2.], [3., 4.]], device=device, requires_grad=True)
    repeated = x.repeat(*repetitions)
    expected = np.tile([[1., 2.], [3., 4.]], repetitions)
    assert repeated.is_contiguous()
    assert repeated.data_ptr() != x.data_ptr()
    (repeated * 2).sum().backward()
    np.testing.assert_array_equal(x.grad.cpu().numpy(), np.full((2, 2), 2 * np.prod(repetitions)))
    with torch.no_grad():
        repeated.add_(3.)
    np.testing.assert_array_equal(repeated.detach().cpu().numpy(), expected + 3.)
    np.testing.assert_array_equal(x.detach().cpu().numpy(), [[1., 2.], [3., 4.]])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("repeats", [1, 2])
def test_scalar_repeat_interleave_has_independent_writable_storage(device, repeats):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    x = torch.tensor([[1.], [2.], [3.]], device=device, requires_grad=True)
    output = torch.repeat_interleave(x, repeats, dim=1)
    assert output.is_contiguous() and output.data_ptr() != x.data_ptr()
    output.sum().backward()
    np.testing.assert_array_equal(x.grad.cpu().numpy(), np.full((3, 1), repeats))
    with torch.no_grad():
        output.add_(4.)
    np.testing.assert_array_equal(output.detach().cpu().numpy(), np.repeat([[1.], [2.], [3.]], repeats, axis=1) + 4.)
    np.testing.assert_array_equal(x.detach().cpu().numpy(), [[1.], [2.], [3.]])

"""Tuple mask assignment preserves the selected region and its gradient."""
import numpy as np
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    return request.param


def test_column_mask_assignment_preserves_source_gradient(device):
    destination = torch.zeros((3, 4), device=device)
    mask = torch.tensor([True, False, True, False], device=device)
    data = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
    source = torch.tensor(data, device=device, requires_grad=True)
    expected = np.zeros((3, 4), dtype=np.float32)
    expected[:, [0, 2]] = data

    destination[:, mask] = source

    assert destination.device.type == device
    np.testing.assert_array_equal(destination.detach().cpu().numpy(), expected)
    destination.sum().backward()
    np.testing.assert_array_equal(source.grad.cpu().numpy(), np.ones_like(data))


def test_multidimensional_mask_assignment_with_ellipsis(device):
    destination = torch.zeros((2, 2, 3), device=device)
    selected = np.array([[True, False, False], [False, True, True]])
    mask = torch.tensor(selected, device=device)
    data = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    expected = np.zeros((2, 2, 3), dtype=np.float32)
    expected[..., selected] = data

    with torch.no_grad():
        destination[..., mask] = torch.tensor(data, device=device)

    assert destination.device.type == device
    np.testing.assert_array_equal(destination.cpu().numpy(), expected)
    np.testing.assert_array_equal(destination[..., mask].cpu().numpy(), data)


def test_column_mask_assignment_rejects_wrong_shape(device):
    destination = torch.zeros((3, 4), device=device)
    mask = torch.tensor([True, False], device=device)
    with pytest.raises(IndexError):
        destination[:, mask] = 1

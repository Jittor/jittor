import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestTensorMode:
    def test_batched_integral_modes_and_indices(self, device):
        sample = torch.tensor([[2, 1, 2, 1, 0], [5, 3, 3, 2, 3]], dtype=torch.int64, device=device)
        result = sample.mode(-1)
        expected = ([2, 3], [2, 4]) if device == "cuda" else ([1, 3], [3, 4])
        assert result.values.tolist() == expected[0]
        assert result.indices.tolist() == expected[1]
        assert result.values.device.type == result.indices.device.type == device
        values, indices = torch.mode(sample.T, 0, keepdim=True)
        assert tuple(values.shape) == tuple(indices.shape) == (1, 2)
        assert torch.equal(values[0], result.values) and torch.equal(indices[0], result.indices)

    def test_mode_gradient_follows_selected_input(self, device):
        sample = torch.tensor([3., 3., 1., 4.], device=device, requires_grad=True)
        result = sample.mode()
        assert result.values.item() == 3.
        assert result.indices.item() == 1
        result.values.backward()
        assert sample.grad.tolist() == [0., 1., 0., 0.]

    def test_scalar_empty_and_out(self, device):
        value = torch.tensor(7., device=device, requires_grad=True)
        result = torch.mode(value, keepdim=True)
        assert result.values.ndim == 0 and result.indices.item() == 0
        result.values.backward()
        assert value.grad.item() == 1.
        empty = torch.empty((0, 5), device=device)
        assert tuple(torch.mode(empty, dim=1, keepdim=True).indices.shape) == (0, 1)
        with pytest.raises(IndexError, match="empty"):
            torch.mode(empty, dim=0)
        sample = torch.tensor([False, True, True], device=device)
        tmp = (torch.empty((), dtype=torch.bool, device=device),
               torch.empty((), dtype=torch.int64, device=device))
        result = torch.mode(sample, out=tmp)
        assert result.values is tmp[0] and result.indices is tmp[1]
        assert result.values.item() is True and result.indices.item() == 2

    def test_large_noncontiguous_slices_resize_out(self, device):
        indices = torch.arange(4096, device=device)
        values = torch.stack((indices % 97, indices % 91))
        values[:, 700:1100] = 11
        values = values.T
        out = (torch.empty(0, dtype=torch.int64, device=device),
               torch.empty(0, dtype=torch.int64, device=device))
        modes = torch.mode(values, dim=0, keepdim=True, out=out)
        assert modes.values is out[0] and modes.indices is out[1]
        assert modes.values.tolist() == [[11, 11]] and tuple(modes.indices.shape) == (1, 2)
        assert torch.equal(values.gather(0, modes.indices), modes.values)


instantiate_device_type_tests(TestTensorMode, globals(), only_for=("cpu", "cuda"))

"""Copying checkpoint loads preserve parameter/buffer allocations and aliases."""
import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestModuleLoadStorage:
    def test_pending_checkpoint_snapshot_keeps_preload_values(self, device):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
        snapshot = module.weight.detach().clone()
        module.load_state_dict({"weight": torch.tensor([3., 4.], device=device)})
        np.testing.assert_array_equal(snapshot.cpu().numpy(), [1., 2.])
        np.testing.assert_array_equal(module.weight.detach().cpu().numpy(), [3., 4.])

    def test_parameter_and_buffer_remain_aliases_of_flat_storage(self, device):
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.zeros(4, dtype=dtype, device=device))
        module.register_buffer("count", torch.zeros(2, dtype=torch.int64, device=device))
        flat = torch.zeros(4, dtype=dtype, device=device)
        buffer_alias = module.count.detach()
        module.weight.data = flat.data
        parameter = module.weight
        pointers = (parameter.data_ptr(), flat.data_ptr(), module.count.data_ptr())
        state = {"weight": torch.tensor([1., 2., 3., 4.], dtype=torch.float64, device="cpu"),
                 "count": torch.tensor([5, 6], dtype=torch.int32, device="cpu")}
        result = module.load_state_dict(state)
        assert not result.missing_keys and not result.unexpected_keys
        assert module.weight is parameter and parameter.requires_grad and parameter.is_leaf
        assert parameter.dtype == dtype and module.count.dtype == torch.int64
        assert pointers == (parameter.data_ptr(), flat.data_ptr(), module.count.data_ptr())
        assert parameter.device.type == device and module.count.device.type == device
        np.testing.assert_array_equal(flat.float().cpu().numpy(), [1., 2., 3., 4.])
        np.testing.assert_array_equal(buffer_alias.cpu().numpy(), [5, 6])
        parameter.square().sum().backward()
        np.testing.assert_array_equal(parameter.grad.float().cpu().numpy(), [2., 4., 6., 8.])

    def test_strict_and_shape_failures_do_not_write_flat_storage(self, device):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
        module.register_buffer("count", torch.tensor([3, 4], device=device))
        alias = module.weight.detach()
        pointer = module.weight.data_ptr()
        for state, message in (
                ({"weight": torch.tensor([9., 9.], device=device)}, "Missing key"),
                ({"weight": torch.tensor([9., 9.], device=device),
                  "count": torch.tensor([7], device=device)}, "size mismatch")):
            with pytest.raises(RuntimeError, match=message):
                module.load_state_dict(state)
            assert module.weight.data_ptr() == pointer
            np.testing.assert_array_equal(alias.cpu().numpy(), [1., 2.])
            np.testing.assert_array_equal(module.count.cpu().numpy(), [3, 4])


instantiate_device_type_tests(TestModuleLoadStorage, globals())

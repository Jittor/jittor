"""Storage addresses must describe native allocations, including aliases."""

import numpy as np
import tempfile
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestStoragePointerIdentity:
    def test_safetensors_distinguishes_overlapping_and_disjoint_views(self, device):
        safetensors = pytest.importorskip("safetensors.torch")
        root = torch.arange(8, dtype=torch.float32, device=device)
        data = {"left": root[:4], "right": root[4:],
                "empty_a": torch.empty(0, device=device),
                "empty_b": torch.empty(0, device=device)}
        restored = safetensors.load(safetensors.save(data))
        for name, tensor in data.items():
            np.testing.assert_array_equal(restored[name].cpu().numpy(), tensor.cpu().numpy())
        with pytest.raises(RuntimeError, match="share memory"):
            safetensors.save({"left": root[:5], "right": root[4:]})

    def test_views_report_the_allocation_origin(self, device):
        root = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        pointer = root.data_ptr()
        for value in (root, root.detach(), root[1:], root.reshape(-1)[-1:]):
            assert value.device.type == device
            assert value.untyped_storage().data_ptr() == pointer
            assert value.storage().data_ptr() == pointer
            assert value.data_ptr() == pointer + value.storage_offset() * value.element_size()
        copied = root.clone()
        assert copied.untyped_storage().data_ptr() != pointer

    def test_safetensors_model_roundtrip_keeps_tied_parameters(self, device):
        safetensors = pytest.importorskip("safetensors.torch")

        def make_model():
            model = torch.nn.Module()
            model.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32,
                                                         device=device).reshape(3, 4))
            model.alias = model.weight
            return model

        model = make_model()
        state = model.state_dict()
        assert set(state) == {"weight", "alias"}
        assert len(list(model.named_parameters())) == 1
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/tied.safetensors"
            safetensors.save_model(model, path)
            restored = make_model()
            with torch.no_grad():
                restored.weight.zero_()
            safetensors.load_model(restored, path)
        assert restored.weight is restored.alias
        assert restored.weight.device.type == device
        np.testing.assert_array_equal(restored.weight.detach().cpu().numpy(),
                                      np.arange(12).reshape(3, 4))


instantiate_device_type_tests(TestStoragePointerIdentity, globals())

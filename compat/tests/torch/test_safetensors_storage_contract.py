"""Storage identity and safetensors shared-weight save/load contracts."""
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
from safetensors import safe_open

from _helpers.device_types import instantiate_device_type_tests


class TestStorageIdentity:
    def test_storage_pointer_matches_tensor_pointer(self, device):
        # ROCm uses Torch's public cuda namespace, while the native test label
        # and flag scope continue to identify the actual ROCm backend.
        device = "cuda" if device == "rocm" else device
        whole = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        assert whole.storage().data_ptr() == whole.data_ptr()
        assert whole.untyped_storage().data_ptr() == whole.data_ptr()
        assert whole.untyped_storage().nbytes() == whole.numel() * whole.element_size()

    def test_view_reports_allocation_base(self, device):
        device = "cuda" if device == "rocm" else device
        whole = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
        view = whole[:, 1:]
        base = whole.storage().data_ptr()
        assert view.storage().data_ptr() == base
        assert view.data_ptr() == base + view.storage_offset() * view.element_size()
        assert view.untyped_storage().nbytes() == whole.numel() * whole.element_size()


class TestModelSharedWeights:
    def test_tied_linear_saves_one_name_and_restores_the_alias(self, device):
        device = "cuda" if device == "rocm" else device
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(4, 4)
                self.b = self.a

            def forward(self, x):
                return self.b(self.a(x))

        model = Model().to(device)
        with torch.no_grad():
            model.a.weight.copy_(torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4))
            model.a.bias.copy_(torch.arange(4, dtype=torch.float32, device=device))
        path = os.path.join(directory.name, "model.safetensors")
        safetensors_torch.save_model(model, path)

        assert sorted(safetensors_torch.load_file(path)) == ["a.bias", "a.weight"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            assert handle.metadata() == {"b.bias": "a.bias", "b.weight": "a.weight"}

        fresh = Model().to(device)
        missing, unexpected = safetensors_torch.load_model(fresh, path, device=device)
        # load_state_dict reports incompatible keys as an empty set plus list;
        # the contract is that nothing is missing and nothing is unexpected.
        assert not missing and not unexpected
        assert fresh.b is fresh.a
        assert torch.equal(fresh.a.weight, model.a.weight)
        assert torch.equal(fresh.a.bias, model.a.bias)
        x = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
        assert torch.equal(fresh(x), model(x))


instantiate_device_type_tests(TestStorageIdentity, globals())
instantiate_device_type_tests(TestModelSharedWeights, globals())

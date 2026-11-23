"""Public InstanceNorm module paths preserve class identity."""

import torch
import torch.nn.modules.instancenorm as instancenorm

from _helpers.device_types import instantiate_device_type_tests


def test_instancenorm_submodule_reexports_public_classes():
    for name in ("InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"):
        assert getattr(instancenorm, name) is getattr(torch.nn, name)


class TestInstanceNormCheckpoint:
    def test_default_instance_norm_has_no_affine_parameters(self, device):
        for dimensions in (1, 2, 3):
            factory = getattr(torch.nn, f"InstanceNorm{dimensions}d")
            layer = torch.nn.Sequential(factory(3)).to(device)
            assert layer[0].affine is False
            assert layer[0].weight is None and layer[0].bias is None
            assert not list(layer.parameters())
            assert not layer.state_dict()
            assert not layer.load_state_dict({}, strict=True).missing_keys

            values = torch.linspace(-1, 1, steps=6 * 4**dimensions, device=device)
            values = values.reshape(2, 3, *((4,) * dimensions)).requires_grad_()
            normalized = layer(values)
            normalized.square().sum().backward()
            assert normalized.device.type == values.grad.device.type == device
            assert torch.isfinite(normalized).all().item()
            assert torch.isfinite(values.grad).all().item()

            affine = factory(3, affine=True).to(device)
            assert set(affine.state_dict()) == {"weight", "bias"}


instantiate_device_type_tests(TestInstanceNormCheckpoint, globals(), only_for=("cpu", "cuda"))

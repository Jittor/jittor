"""Torch layer constructors own dtype selection without mutating native aliases."""

import numpy as np
import pytest
import torch
import jittor as jt


@pytest.fixture(autouse=True)
def restore_default_dtype():
    previous = torch.get_default_dtype()
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize("factory,args", [
    (torch.nn.Conv2d, (3, 4, 3)),
    (torch.nn.Conv3d, (3, 4, 3)),
    (torch.nn.LayerNorm, (4,)),
    (torch.nn.GroupNorm, (2, 4)),
    (torch.nn.BatchNorm2d, (4,)),
    (torch.nn.MultiheadAttention, (8, 2)),
])
@pytest.mark.parametrize("target", [torch.float16, torch.float64])
def test_default_dtype_applies_to_owned_parameters_and_float_buffers(factory, args, target):
    torch.set_default_dtype(target)
    layer = factory(*args)
    assert list(layer.parameters())
    for parameter in layer.parameters():
        assert parameter.dtype == target
        assert parameter.requires_grad
    for name, buffer in layer.named_buffers():
        assert buffer.dtype == (torch.int64 if name == "num_batches_tracked" else target)


@pytest.mark.parametrize("factory,args", [
    (torch.nn.Conv2d, (3, 4, 3)),
    (torch.nn.LayerNorm, (4,)),
    (torch.nn.BatchNorm2d, (4,)),
    (torch.nn.MultiheadAttention, (8, 2)),
])
def test_explicit_dtype_overrides_default(factory, args):
    torch.set_default_dtype(torch.float16)
    layer = factory(*args, dtype=torch.float32)
    assert all(value.dtype == torch.float32 for value in layer.parameters())


def test_container_constructor_keeps_supplied_modules_and_parameter_identity():
    child = torch.nn.Linear(4, 3, dtype=torch.float32)
    weight = child.weight
    torch.set_default_dtype(torch.float16)
    parent = torch.nn.Sequential(child)
    assert parent[0] is child
    assert child.weight is weight
    assert child.weight.dtype == torch.float32


def test_torch_default_dtype_does_not_change_native_constructors():
    torch.set_default_dtype(torch.float16)
    for layer in (jt.nn.Conv2d(3, 4, 3), jt.nn.Conv3d(3, 4, 3), jt.nn.LayerNorm(4)):
        assert all(str(value.dtype).replace("torch.", "") == "float32" for value in layer.parameters())


def test_constructor_cast_preserves_external_parameters_shared_aliases_and_buffer_roles():
    class NativeLayer(jt.Module):
        def __init__(self, external):
            self.external = external
            self.weight = self.weight_alias = jt.ones((2,), dtype="float32")
            state = jt.ones((2,), dtype="float32").stop_grad()
            self.register_buffer("state", state)
            self.register_buffer("state_alias", state)
            legacy = jt.zeros((2,), dtype="float32").stop_grad()
            legacy.is_buffer = True
            legacy.persistent = False
            self.register_buffer("legacy", legacy, persistent=False)

    external = torch.nn.Parameter(torch.ones((2,), dtype=torch.float32))
    owner = torch.nn.LayerNorm._nn_frontend_owner
    layer_type = owner.adapt_class(NativeLayer)
    torch.set_default_dtype(torch.float16)
    layer = layer_type(external)
    assert layer.external is external
    assert external.dtype == torch.float32
    assert layer.weight is layer.weight_alias
    assert layer.weight.dtype == torch.float16
    assert layer.weight.requires_grad
    assert layer.state is layer.state_alias
    assert layer.state.dtype == layer.legacy.dtype == torch.float16
    assert layer.legacy.is_buffer and not layer.legacy.persistent
    assert "legacy" in dict(layer.named_buffers())
    assert "legacy" not in layer.state_dict()


@torch.no_grad()
def test_half_default_layer_norm_linear_chain_matches_numpy():
    torch.set_default_dtype(torch.float16)
    norm, linear = torch.nn.LayerNorm(4), torch.nn.Linear(4, 3)
    weights = np.array([[0.1, -0.2, 0.3, 0.2], [0.2, 0.0, -0.1, 0.1],
                        [-0.1, 0.1, 0.1, 0.2]], dtype=np.float16)
    linear.weight.copy_(torch.tensor(weights))
    linear.bias.copy_(torch.zeros(3, dtype=torch.float16))
    values = np.array([[0.2, 0.5, -0.7, 1.0], [1.2, -0.4, 0.8, -0.1]], dtype=np.float16)
    source = values.astype(np.float64)
    normalized = ((source - source.mean(-1, keepdims=True)) /
                  np.sqrt(source.var(-1, keepdims=True) + 1e-5)).astype(np.float16)
    expected = (normalized.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float16)
    output = linear(norm(torch.tensor(values)))
    assert output.dtype == torch.float16
    np.testing.assert_allclose(output.detach().numpy(), expected, rtol=3e-3, atol=5e-4)

"""Explicit packed FP16 snapshots, exercised on a real ACL device."""

import numpy as np
import pytest
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.inference import (
    FrozenLinear, freeze_linear_layers_, pack_linear_weight, packed_linear,
)


@pytest.fixture(autouse=True)
def acl_inference():
    if not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required for packed FP16 linear snapshots")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_acl=1, use_cuda=1), jt.no_grad(), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


@pytest.mark.parametrize("prefix", [(), (0,), (1,), (5,), (2, 3), (2, 0, 3), (1, 2, 3, 4)])
@pytest.mark.parametrize("in_features,out_features", [(16, 32), (17, 31), (33, 19)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_packed_linear_matches_independent_reference(prefix, in_features, out_features, with_bias):
    rng = np.random.RandomState(4901)
    weight_np = rng.uniform(-0.3, 0.3, (out_features, in_features)).astype(np.float16)
    input_np = rng.uniform(-0.3, 0.3, prefix + (in_features,)).astype(np.float16)
    bias_np = rng.uniform(-0.2, 0.2, (out_features,)).astype(np.float16) if with_bias else None
    frozen = FrozenLinear(jt.array(weight_np), None if bias_np is None else jt.array(bias_np))
    output = frozen(jt.array(input_np))
    output.sync()
    assert tuple(output.shape) == prefix + (out_features,)
    assert str(output.dtype) == "float16"
    assert frozen.packed_weight.location() == "device"
    if input_np.size:
        assert output.location() == "device"
    expected = (input_np.astype(np.float64) @ weight_np.astype(np.float64).T).astype(np.float16)
    if bias_np is not None:
        expected = (expected.astype(np.float32) + bias_np).astype(np.float16)
    np.testing.assert_allclose(output.numpy(), expected, rtol=3e-3, atol=5e-4)


def test_packed_snapshot_is_materialized_and_independent_of_original_updates():
    weight_np = (np.arange(19 * 17).reshape(19, 17) / 1024).astype(np.float16)
    bias_np = (np.arange(19) / 64).astype(np.float16)
    input_np = np.full((2, 17), 0.125, dtype=np.float16)
    weight, bias = jt.array(weight_np), jt.array(bias_np)
    frozen = FrozenLinear(weight, bias)
    # Mutate both source tensors before the snapshot's first forward.
    weight.assign(jt.zeros(weight.shape, dtype="float16"))
    bias.assign(jt.ones(bias.shape, dtype="float16"))
    jt.sync([weight, bias])
    assert frozen.packed_weight._storage_address != weight._storage_address
    assert frozen.bias._storage_address != bias._storage_address
    actual = frozen(jt.array(input_np))
    actual.sync()
    expected = (input_np.astype(np.float64) @ weight_np.astype(np.float64).T).astype(np.float16)
    expected = (expected.astype(np.float32) + bias_np).astype(np.float16)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=3e-3, atol=5e-4)
    assert list(frozen.named_parameters()) == []
    assert set(frozen.state_dict()) == {"packed_weight", "bias"}
    assert not hasattr(frozen, "weight")


@pytest.mark.parametrize("prefix", [(), (0,), (1,), (2, 3)])
def test_packed_weight_layout_and_low_level_forward(prefix):
    weight_np = (np.arange(19 * 17).reshape(19, 17) / 1024).astype(np.float16)
    packed = pack_linear_weight(jt.array(weight_np))
    padded = np.pad(weight_np, ((0, 13), (0, 15)))
    expected_layout = padded.reshape(2, 16, 2, 16).transpose(2, 0, 1, 3).copy()
    x_np = np.full(prefix + (17,), 0.25, dtype=np.float16)
    x = jt.array(x_np)
    result = packed_linear(x, packed, 17, 19)
    native = jt.core.ops.mapped_packed_matmul(x, packed, 17, 19)
    jt.sync([result, native])
    if x_np.size:
        assert result.location() == native.location() == "device"
    assert tuple(result.shape) == tuple(native.shape) == prefix + (19,)
    np.testing.assert_array_equal(packed.numpy(), expected_layout)
    np.testing.assert_allclose(result.numpy(), x_np.astype(np.float64) @ weight_np.astype(np.float64).T,
                               rtol=3e-3, atol=5e-4)
    np.testing.assert_array_equal(native.numpy(), result.numpy())


@pytest.mark.parametrize("axes", [(1, 0, 2), (0, 2, 1)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_packed_linear_materializes_strided_activations(axes, with_bias):
    rng = np.random.RandomState(4902)
    base_np = rng.uniform(-0.3, 0.3, (2, 3, 17)).astype(np.float16)
    input_np = base_np.transpose(axes)
    weight_np = rng.uniform(-0.3, 0.3, (19, input_np.shape[-1])).astype(np.float16)
    bias_np = rng.uniform(-0.2, 0.2, (19,)).astype(np.float16) if with_bias else None
    frozen = FrozenLinear(jt.array(weight_np), None if bias_np is None else jt.array(bias_np))
    base = jt.array(base_np)
    base.sync()
    with jt.flag_scope(transpose_storage_view=1):
        x = jt.core.ops.transpose(base, axes)
    assert not x._storage_is_contiguous()
    result = frozen(x)
    result.sync()
    assert result.location() == "device"
    assert tuple(result.shape) == input_np.shape[:-1] + (19,)
    expected = (input_np.astype(np.float64) @ weight_np.astype(np.float64).T).astype(np.float16)
    if bias_np is not None:
        expected = (expected.astype(np.float32) + bias_np).astype(np.float16)
    np.testing.assert_allclose(result.numpy(), expected, rtol=3e-3, atol=5e-4)


def test_packed_linear_rejects_strided_packed_storage_before_materialization():
    packed = pack_linear_weight(jt.ones((16, 16), dtype="float16"))
    with jt.flag_scope(transpose_storage_view=1):
        strided = jt.core.ops.transpose(packed, (0, 1, 3, 2))
    assert tuple(strided.shape) == tuple(packed.shape)
    assert not strided._storage_is_contiguous()
    with pytest.raises(ValueError, match="packed weight shape or storage"):
        packed_linear(jt.ones((1, 16), dtype="float16"), strided, 16, 16)


@pytest.mark.parametrize("features", [0, -1, True, 16.0, "16"])
@pytest.mark.parametrize("which", ["in_features", "out_features"])
def test_packed_linear_rejects_invalid_logical_features(features, which):
    packed = pack_linear_weight(jt.ones((16, 16), dtype="float16"))
    dimensions = {"in_features": 16, "out_features": 16}
    dimensions[which] = features
    with pytest.raises(ValueError, match=which):
        packed_linear(jt.ones((1, 16), dtype="float16"), packed, **dimensions)


@pytest.mark.parametrize("invalid", ["dtype", "last_dim", "packed_shape", "features", "rank"])
def test_native_packed_matmul_rejects_invalid_inputs(invalid):
    x = jt.ones((2, 3, 16), dtype="float16")
    packed = jt.ones((1, 1, 16, 16), dtype="float16")
    in_features, out_features = 16, 16
    if invalid == "dtype":
        x = x.float32()
        message = "float16"
    elif invalid == "last_dim":
        in_features = 15
        message = "in_features"
    elif invalid == "packed_shape":
        packed = jt.ones((1, 1, 16, 15), dtype="float16")
        message = "packed weight shape"
    elif invalid == "features":
        out_features = 0
        message = "positive"
    else:
        x = jt.core.ops.reshape(jt.ones((1,), dtype="float16"), ())
        assert x.ndim == 0
        message = "in_features"
    with pytest.raises(RuntimeError, match=message):
        jt.core.ops.mapped_packed_matmul(x, packed, in_features, out_features)


def test_packed_linear_preserves_public_dtype_shape_and_tensor_errors():
    x = jt.ones((1, 16), dtype="float16")
    packed = pack_linear_weight(jt.ones((16, 16), dtype="float16"))
    with pytest.raises(ValueError, match="packed weight shape"):
        packed_linear(x, jt.ones((1, 1, 16, 15), dtype="float16"), 16, 16)
    with pytest.raises(TypeError, match="float16"):
        packed_linear(x, packed.float32(), 16, 16)
    with pytest.raises(TypeError, match="Jittor tensors"):
        packed_linear(np.ones((1, 16), dtype=np.float16), packed, 16, 16)
    with pytest.raises(TypeError, match="Jittor tensors"):
        packed_linear(x, np.ones((1, 1, 16, 16), dtype=np.float16), 16, 16)
    with pytest.raises(TypeError, match="float16"):
        packed_linear(x, packed, 16, 16, jt.ones((16,), dtype="float32"))
    with pytest.raises(ValueError, match="bias shape"):
        packed_linear(x, packed, 16, 16, jt.ones((17,), dtype="float16"))


def test_packed_linear_and_native_entry_reject_cpu_and_training():
    frozen = FrozenLinear(jt.ones((16, 16), dtype="float16"))
    x = jt.ones((1, 16), dtype="float16")
    with jt.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        jt.core.ops.mapped_packed_matmul(x, frozen.packed_weight, 16, 16)
    with jt.flag_scope(use_acl=0, use_cuda=0):
        cpu_x = jt.ones((1, 16), dtype="float16")
        cpu_packed = jt.ones((1, 1, 16, 16), dtype="float16")
        jt.sync([cpu_x, cpu_packed], True, False)
        assert cpu_x.location() == cpu_packed.location() == "cpu"
        with pytest.raises(RuntimeError, match="ACL device backend"):
            packed_linear(cpu_x, cpu_packed, 16, 16)
        with pytest.raises(RuntimeError, match="ACL device backend"):
            jt.core.ops.mapped_packed_matmul(cpu_x, cpu_packed, 16, 16)
        with pytest.raises(RuntimeError, match="ACL device backend"):
            frozen(cpu_x)
        with pytest.raises(RuntimeError, match="ACL device backend"):
            FrozenLinear(jt.ones((16, 16), dtype="float16"))


def test_packed_linear_does_not_translate_unclassified_runtime_errors(monkeypatch):
    x = jt.ones((1, 16), dtype="float16")
    packed = pack_linear_weight(jt.ones((16, 16), dtype="float16"))
    failure = RuntimeError("unexpected native construction failure")

    def fail(*args):
        raise failure

    monkeypatch.setattr(jt.core.ops, "mapped_packed_matmul", fail)
    with pytest.raises(RuntimeError) as caught:
        packed_linear(x, packed, 16, 16)
    assert caught.value is failure


def test_conversion_preserves_containers_shared_modules_and_exact_type_scope():
    class CustomLinear(jt.nn.Linear):
        pass

    class Network(jt.Module):
        def __init__(self):
            shared = jt.nn.Linear(17, 19)
            self.first = shared
            self.again = shared
            self.sequence = jt.nn.Sequential(shared, jt.nn.relu)
            self.layers = jt.nn.ModuleList([jt.nn.Linear(19, 16)])
            self.custom = CustomLinear(17, 19)

    network = Network().float16().eval()
    source, custom = network.first, network.custom
    returned = freeze_linear_layers_(network)
    assert returned is network
    assert isinstance(network.first, FrozenLinear)
    assert network.first is network.again is network.sequence[0]
    assert isinstance(network.layers[0], FrozenLinear)
    assert network.custom is custom
    x_np = np.full((1, 17), 0.125, dtype=np.float16)
    x = jt.array(x_np)
    expected = source(x)
    actual = network.first(x)
    jt.sync([actual, expected])
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=3e-3, atol=5e-4)
    # An explicitly supplied frontend class is matched exactly, too.
    assert isinstance(freeze_linear_layers_(custom, linear_type=CustomLinear), FrozenLinear)


def test_frozen_linear_rejects_training_wrong_dtype_and_shape():
    frozen = FrozenLinear(jt.ones((16, 16), dtype="float16"))
    with pytest.raises(RuntimeError, match="training"):
        frozen.train()
    with jt.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
        frozen(jt.ones((1, 16), dtype="float16"))
    with pytest.raises(TypeError, match="float16"):
        FrozenLinear(jt.ones((16, 16), dtype="float32"))
    with pytest.raises(TypeError, match="float16"):
        frozen(jt.ones((1, 16), dtype="float32"))
    with pytest.raises(ValueError, match="in_features"):
        frozen(jt.ones((1, 17), dtype="float16"))
    with pytest.raises(ValueError, match="bias shape"):
        FrozenLinear(jt.ones((16, 16), dtype="float16"), jt.ones((17,), dtype="float16"))
    training = jt.nn.Linear(16, 16).float16()
    with pytest.raises(RuntimeError, match="eval"):
        FrozenLinear.from_linear(training)
    frozen.is_train = True
    with pytest.raises(RuntimeError, match="training"):
        frozen(jt.ones((1, 16), dtype="float16"))


def test_rejected_conversion_leaves_original_module_tree_intact():
    first = jt.nn.Linear(16, 16).float16().eval()
    second = jt.nn.Linear(16, 16).float32().eval()
    network = jt.nn.Sequential(first, second).eval()
    with pytest.raises(TypeError, match="float16"):
        freeze_linear_layers_(network)
    assert network[0] is first
    assert network[1] is second


@pytest.mark.parametrize("customization", [
    "forward_hook", "forward_pre_hook", "forward", "execute",
])
def test_conversion_rejects_custom_forward_without_replacing_modules(customization):
    first = jt.nn.Linear(16, 16).float16().eval()
    second = jt.nn.Linear(16, 16).float16().eval()
    network = jt.nn.Sequential(first, second).eval()
    if customization == "forward_hook":
        second.register_forward_hook(lambda module, args, output: output + 1)
        message = "forward hooks"
    elif customization == "forward_pre_hook":
        second.register_forward_pre_hook(lambda module, args: (args[0] + 1,))
        message = "forward hooks"
    else:
        setattr(second, customization, lambda value: value + 1)
        message = "instance forward/execute overrides"

    with pytest.raises(RuntimeError, match=message):
        FrozenLinear.from_linear(second)
    # The first layer is eligible and has already been prepared when traversal
    # rejects the second one. Neither edge may be changed by that failed pass.
    with pytest.raises(RuntimeError, match=message):
        freeze_linear_layers_(network)
    assert network[0] is first
    assert network[1] is second
    assert not any(isinstance(layer, FrozenLinear) for layer in network.modules())

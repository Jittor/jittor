"""FSDP2 keeps fp32 master shards while gathering bf16 compute weights."""

from types import SimpleNamespace
import os
from unittest import mock

import numpy as np
import pytest
import torch
import jittor as jt

from jittor.compat.fsdp2 import shard
from jittor.compat.fsdp2.config import MixedPrecisionPolicy


@pytest.mark.parametrize("flat", [False, True])
def test_unshard_casts_collective_input_without_changing_master_shard(flat):
    master = torch.tensor([1.25, 2.5], dtype=torch.float32, requires_grad=True)
    owner = SimpleNamespace(weight=master)
    entry = SimpleNamespace(
        owner=owner, attr="weight", shard=master, full_param=None,
        shape=(2,), dtype=master.dtype, numel=2, padded_numel=2,
        flat_offset=0, requires_grad=True)
    state = SimpleNamespace(
        true_fsdp_initialized=True, true_fsdp_unsharded=False,
        true_fsdp_flat=flat, true_fsdp_params=(entry,),
        true_fsdp_flat_shard=master, shard_group=None,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        frontend_type=torch.Tensor)
    module = SimpleNamespace(_fsdp_state=state)
    seen = []

    def all_gather(value, group=None):
        seen.append(value.dtype)
        return value

    with mock.patch.object(shard.common, "_all_gather_shards", side_effect=all_gather):
        shard._unshard_module_params(module)

    assert seen == [torch.bfloat16]
    assert master.dtype == torch.float32
    assert entry.shard is master
    assert entry.full_param.dtype == torch.bfloat16
    assert owner.weight is entry.full_param
    np.testing.assert_array_equal(owner.weight.float().numpy(), [1.25, 2.5])
    shard._reshard_module_params(module)
    assert owner.weight is master


@pytest.mark.parametrize("flat", [False, True])
def test_unshard_without_policy_keeps_master_dtype(flat):
    master = torch.tensor([1.25, 2.5], dtype=torch.float32)
    owner = SimpleNamespace(weight=master)
    entry = SimpleNamespace(
        owner=owner, attr="weight", shard=master, full_param=None,
        shape=(2,), dtype=master.dtype, numel=2, padded_numel=2,
        flat_offset=0, requires_grad=False)
    state = SimpleNamespace(
        true_fsdp_initialized=True, true_fsdp_unsharded=False,
        true_fsdp_flat=flat, true_fsdp_params=(entry,),
        true_fsdp_flat_shard=master, shard_group=None,
        mp_policy=MixedPrecisionPolicy(), frontend_type=torch.Tensor)
    module = SimpleNamespace(_fsdp_state=state)
    with mock.patch.object(shard.common, "_all_gather_shards", return_value=master):
        shard._unshard_module_params(module)
    assert entry.full_param.dtype == torch.float32
    assert entry.shard is master


@pytest.mark.parametrize("cast_inputs", [False, True])
def test_forward_policy_casts_float_trees_and_preserves_integer_inputs(cast_inputs):
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, output_dtype=torch.float32,
        cast_forward_inputs=cast_inputs)
    state = SimpleNamespace(
        true_fsdp_initialized=True, true_fsdp_params=(),
        reshard_after_forward=True, mp_policy=policy)
    module = SimpleNamespace(_fsdp_state=state)
    values = torch.tensor([1.25, 2.5], dtype=torch.float32)
    indices = torch.tensor([1, 2], dtype=torch.int64)
    observed = []

    def forward(payload):
        observed.append((payload["values"].dtype, payload["indices"].dtype))
        return (payload["values"] * 2, {"indices": payload["indices"]})

    with mock.patch.object(shard, "_unshard_module_params"), mock.patch.object(
            shard, "_reshard_module_params"):
        output = shard._execute_with_true_fsdp(
            module, forward, {"values": values, "indices": indices})

    assert observed == [(
        torch.bfloat16 if cast_inputs else torch.float32, torch.int64)]
    assert output[0].dtype == torch.float32
    assert output[1]["indices"].dtype == torch.int64



@pytest.mark.parametrize(
    ("policy_dtype", "expected_amp_bit"),
    [
        (torch.float16, jt.amp_flags.prefer16),
        (torch.bfloat16, jt.amp_flags.prefer_bfloat16),
    ],
)
def test_forward_policy_sets_and_restores_compute_amp_register(
        policy_dtype, expected_amp_bit):
    policy = MixedPrecisionPolicy(param_dtype=policy_dtype)
    state = SimpleNamespace(
        true_fsdp_initialized=True, true_fsdp_params=(),
        reshard_after_forward=True, mp_policy=policy)
    module = SimpleNamespace(_fsdp_state=state)
    original_amp = jt.amp_flags.prefer32 | jt.amp_flags.keep_reduce
    observed = []

    def forward():
        observed.append(int(jt.flags.amp_reg))
        return 1

    with jt.flag_scope(amp_reg=original_amp), mock.patch.object(
            shard, "_unshard_module_params"), mock.patch.object(
                shard, "_reshard_module_params"):
        result = shard._execute_with_true_fsdp(module, forward)
        assert int(jt.flags.amp_reg) == original_amp

    assert result == 1
    assert len(observed) == 1
    assert observed[0] & expected_amp_bit
    assert not observed[0] & (
        jt.amp_flags.prefer32 | jt.amp_flags.prefer16
        if expected_amp_bit == jt.amp_flags.prefer_bfloat16
        else jt.amp_flags.prefer32 | jt.amp_flags.prefer_bfloat16
    )
    assert observed[0] & jt.amp_flags.keep_reduce


def test_forward_policy_respects_disabled_input_cast_amp():
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, cast_forward_inputs=False)
    state = SimpleNamespace(
        true_fsdp_initialized=True, true_fsdp_params=(),
        reshard_after_forward=True, mp_policy=policy)
    module = SimpleNamespace(_fsdp_state=state)
    observed = []

    def forward():
        observed.append(int(jt.flags.amp_reg))
        return 1

    with jt.flag_scope(amp_reg=jt.amp_flags.prefer32), mock.patch.object(
            shard, "_unshard_module_params"), mock.patch.object(
                shard, "_reshard_module_params"):
        shard._execute_with_true_fsdp(module, forward)
        assert int(jt.flags.amp_reg) == jt.amp_flags.prefer32

    assert observed == [jt.amp_flags.prefer32]


def test_cast_frozen_forward_input_keeps_gradient_graph():
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    state = SimpleNamespace(
        true_fsdp_initialized=True,
        true_fsdp_params=(SimpleNamespace(requires_grad=False),),
        reshard_after_forward=True, mp_policy=policy)
    module = SimpleNamespace(_fsdp_state=state)
    values = torch.tensor([1.25, 2.5], dtype=torch.float32,
                          requires_grad=True)
    frozen = torch.tensor([2.0, 4.0], dtype=torch.bfloat16,
                          requires_grad=False)

    def forward(tensor):
        return (tensor * frozen).sum()

    with mock.patch.object(shard, "_unshard_module_params"), mock.patch.object(
            shard, "_reshard_module_params"):
        output = shard._execute_with_true_fsdp(module, forward, values)
    gradient = jt.grad(output, values)
    np.testing.assert_allclose(gradient.numpy(), [2.0, 4.0])


@pytest.mark.parametrize("world,group_size,distributed,expected", [
    (1, 1, False, True), (1, 2, False, False),
    (2, 1, False, False), (2, 2, False, False), (2, 2, True, True),
])
def test_initialization_gate_accepts_only_real_distributed_or_singleton_mesh(
        world, group_size, distributed, expected):
    group = SimpleNamespace(size=lambda: group_size)
    with mock.patch.object(shard.common, "_world_size", return_value=world), mock.patch.object(
            shard.common, "_in_true_distributed", return_value=distributed):
        assert shard.common._can_initialize_fsdp(group) is expected
        if not distributed:
            assert not shard.common._can_initialize_fsdp(None)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("call_execute", [False, True])
def test_public_singleton_policy_unshards_compute_weights_before_sequential_linear(
        flat, nested, input_dtype, call_execute):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, fully_shard
    from jittor._runtime.fallback import forbid_backend_fallbacks

    device = os.environ.get("FSDP_SINGLETON_TEST_DEVICE", "cpu")
    mesh = init_device_mesh(device, (1,))
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3),
    ).to(device)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = np.arange(parameter.numel(), dtype=np.float32).reshape(tuple(parameter.shape))
            parameter.copy_(torch.tensor((values % 7 - 3) / 32 + (index + 1) / 64, device=device))
    initial = {name: parameter.detach().cpu().numpy().copy()
               for name, parameter in model.named_parameters()}
    events = []
    for index in (0, 2):
        layer = model[index]
        original = layer.execute

        def observed(value, layer=layer, index=index, original=original):
            events.append(("linear", index, value.dtype, layer.weight.dtype, layer.bias.dtype))
            assert value.dtype == layer.weight.dtype == layer.bias.dtype == torch.bfloat16
            assert int(jt.flags.amp_reg) & jt.amp_flags.prefer_bfloat16
            return original(value)

        object.__setattr__(layer, "execute", observed)
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
        output_dtype=torch.float32, cast_forward_inputs=True)
    amp_before = int(jt.flags.amp_reg)
    fallback_before = jt.core.backend_fallback_count()
    with mock.patch.dict(os.environ, {"JITTOR_FSDP2_FLAT": "1" if flat else "0"}), forbid_backend_fallbacks():
        if nested:
            fully_shard(model[0], mesh=mesh, mp_policy=policy)
            fully_shard(model[2], mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        wrapped = (model, model[0], model[2]) if nested else (model,)
        for module in wrapped:
            assert isinstance(module, FSDPModule)
            assert module._fsdp_state.true_fsdp_initialized
            assert module._fsdp_state.true_fsdp_world_size == 1
        parameters = dict(model.named_parameters())
        assert len(parameters) == 4
        for name, parameter in parameters.items():
            assert parameter.dtype == torch.float32
            np.testing.assert_array_equal(parameter.full_tensor().detach().cpu().numpy(), initial[name])
        optimizer = torch.optim.SGD(parameters.values(), lr=.01)
        leaf = torch.tensor(np.arange(32, dtype=np.float32).reshape(4, 8) / 64,
                            dtype=input_dtype, device=device, requires_grad=True)
        original_unshard = shard._unshard_module_params
        original_reshard = shard._reshard_module_params

        def unshard(module):
            events.append(("unshard", id(module)))
            return original_unshard(module)

        def reshard(module):
            events.append(("reshard", id(module)))
            return original_reshard(module)

        with mock.patch.object(shard, "_unshard_module_params", side_effect=unshard), mock.patch.object(
                shard, "_reshard_module_params", side_effect=reshard):
            output = model.execute(leaf[:, ::2]) if call_execute else model(leaf[:, ::2])
        assert output.dtype == torch.float32 and output.device.type == device
        assert int(jt.flags.amp_reg) == amp_before
        linear_events = [event for event in events if event[0] == "linear"]
        assert [event[1] for event in linear_events] == [0, 2]
        for module in wrapped:
            assert events.count(("unshard", id(module))) == 1
            assert events.count(("reshard", id(module))) == 1
        if nested:
            for index in (0, 2):
                entered = events.index(("unshard", id(model[index])))
                executed = next(i for i, event in enumerate(events)
                                if event[0] == "linear" and event[1] == index)
                exited = events.index(("reshard", id(model[index])))
                assert entered < executed < exited
        loss = output.square().mean()
        loss.backward()
        assert leaf.grad is not None and leaf.grad.dtype == input_dtype
        assert leaf.grad.device.type == device
        np.testing.assert_array_equal(leaf.grad.detach().float().cpu().numpy()[:, 1::2], 0)
        assert np.isfinite(leaf.grad.detach().float().cpu().numpy()).all()
        for parameter in model.parameters():
            assert parameter.dtype == torch.float32
            assert parameter.grad is not None and parameter.grad.dtype == torch.float32
            assert parameter.grad.device.type == device
        optimizer.step()
        changed = False
        for name, parameter in model.named_parameters():
            assert parameter.dtype == torch.float32
            updated = parameter.full_tensor().detach().cpu().numpy()
            assert np.isfinite(updated).all()
            changed |= not np.array_equal(updated, initial[name])
        assert changed
        jt.sync_all(True)
    assert int(jt.flags.amp_reg) == amp_before
    assert jt.core.backend_fallback_count() == fallback_before



class _FSDPPointwiseFailure(RuntimeError):
    pass


def _assert_fsdp_public_autocast_disabled():
    assert not torch.is_autocast_enabled()
    assert not torch.is_autocast_enabled("cpu")


def _public_cpu_pointwise_fsdp_model(policy_dtype):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, fully_shard

    child_dtype = (torch.float16 if policy_dtype == torch.bfloat16
                   else torch.bfloat16)
    events = []

    class Child(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False).to("cpu")
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor(
                    np.arange(16, dtype=np.float32).reshape(4, 4) / 16 - .5,
                    dtype=torch.float32, device="cpu"))

        def forward(self, value, fail=False):
            _assert_fsdp_public_autocast_disabled()
            assert value.dtype == self.linear.weight.dtype == child_dtype
            result = self.linear(value)
            assert result.dtype == child_dtype
            events.append("child_gemm")
            if fail:
                events.append("child_failure")
                raise _FSDPPointwiseFailure("child")
            return result

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child()
            self.weight = torch.nn.Parameter(torch.eye(
                4, dtype=torch.float32, device="cpu"))
            self.tanh = torch.nn.Tanh()
            self.silu = torch.nn.SiLU()

        def forward(self, value, failure=None):
            _assert_fsdp_public_autocast_disabled()
            try:
                result = self.child(
                    value, fail=failure in ("child", "catch_child"))
            except _FSDPPointwiseFailure:
                if failure != "catch_child":
                    raise
                events.append("caught_child")
                result = self.child(value)
            # This assertion is inside the parent FSDP region: an output cast
            # after its forward must not conceal a low-precision intermediate.
            assert result.dtype == torch.float32
            input_mean = value.mean()
            top_input_mean = torch.mean(value)
            assert input_mean.dtype == value.dtype
            assert top_input_mean.dtype == value.dtype
            outputs = {
                "operator_add": result + result,
                "torch_add": torch.add(result, result),
                "tensor_add": result.add(result),
                "tensor_add_alpha": result.add(result, alpha=2),
                "tensor_mean": result.mean(),
                "torch_mean": torch.mean(result),
                "torch_tanh": torch.tanh(result),
                "tensor_tanh": result.tanh(),
                "module_tanh": self.tanh(result),
                "module_silu": self.silu(result),
                "functional_silu": torch.nn.functional.silu(result),
                "add_tanh_mean": (result + result).tanh().mean(),
            }
            for name, output in outputs.items():
                assert output.dtype == torch.float32, name
                assert output.device.type == "cpu", name
            _assert_fsdp_public_autocast_disabled()
            # Keeping pointwise FP32 must not disable low-precision GEMM.
            assert self.weight.dtype == policy_dtype
            gemm = torch.matmul(result, self.weight)
            assert gemm.dtype == policy_dtype
            events.append("root_gemm")
            if failure == "parent":
                raise _FSDPPointwiseFailure("parent")
            return dict(outputs, child=result, gemm=gemm,
                        input_mean=input_mean, top_input_mean=top_input_mean)

    mesh = init_device_mesh("cpu", (1,))
    model = Root().to("cpu")
    fully_shard(model.child, mesh=mesh, mp_policy=MixedPrecisionPolicy(
        param_dtype=child_dtype, reduce_dtype=torch.float32,
        output_dtype=torch.float32))
    fully_shard(model, mesh=mesh, mp_policy=MixedPrecisionPolicy(
        param_dtype=policy_dtype, reduce_dtype=torch.float32))
    assert isinstance(model, FSDPModule)
    assert isinstance(model.child, FSDPModule)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    return model, events


def _assert_pointwise_fsdp_output_materializes(output, policy_dtype):
    assert output["child"].dtype == torch.float32
    assert output["gemm"].dtype == policy_dtype
    arrays = {}
    for name, value in output.items():
        assert value.device.type == "cpu"
        arrays[name] = value.detach().float().cpu().numpy().copy()
        assert np.isfinite(arrays[name]).all(), name
    np.testing.assert_array_equal(
        arrays["operator_add"], arrays["child"] + arrays["child"])
    np.testing.assert_array_equal(
        arrays["tensor_add"], arrays["child"] + arrays["child"])
    np.testing.assert_array_equal(
        arrays["tensor_add_alpha"], arrays["child"] * 3)
    for name in ("input_mean", "top_input_mean"):
        np.testing.assert_array_equal(arrays[name], np.array(.109375))


def _assert_legacy_amp_outside_fsdp():
    _assert_fsdp_public_autocast_disabled()
    value = torch.tensor([.25, -.125], dtype=torch.float32, device="cpu")
    saved_amp = int(jt.flags.amp_reg)
    with jt.flag_scope(amp_reg=2):
        result = value * value
        assert result.dtype == torch.float16
        _assert_fsdp_public_autocast_disabled()
    assert int(jt.flags.amp_reg) == saved_amp
    np.testing.assert_array_equal(
        result.float().numpy(), np.array([.0625, .015625], dtype=np.float32))


@pytest.mark.parametrize("policy_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("catch_child_failure", [False, True])
def test_public_singleton_fsdp_pointwise_preserves_fp32_without_autocast(
        policy_dtype, flat, catch_child_failure):
    from jittor._runtime.fallback import forbid_backend_fallbacks

    saved_amp = int(jt.flags.amp_reg)
    with jt.flag_scope(use_cuda=0, amp_reg=0), mock.patch.dict(
            os.environ, {"JITTOR_FSDP2_FLAT": "1" if flat else "0"}), forbid_backend_fallbacks():
        _assert_fsdp_public_autocast_disabled()
        fallback_before = jt.core.backend_fallback_count()
        _assert_legacy_amp_outside_fsdp()
        model, events = _public_cpu_pointwise_fsdp_model(policy_dtype)
        value = torch.tensor(
            np.arange(8, dtype=np.float32).reshape(2, 4) / 32,
            dtype=torch.float32, device="cpu")
        failure = "catch_child" if catch_child_failure else None
        output = model(value, failure=failure)
        _assert_pointwise_fsdp_output_materializes(output, policy_dtype)
        assert events.count("child_gemm") == (2 if catch_child_failure else 1)
        assert events.count("caught_child") == int(catch_child_failure)
        assert events.count("root_gemm") == 1
        assert int(jt.flags.amp_reg) == 0
        _assert_legacy_amp_outside_fsdp()
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        assert jt.core.backend_fallback_count() == fallback_before
    assert int(jt.flags.amp_reg) == saved_amp
    _assert_fsdp_public_autocast_disabled()


@pytest.mark.parametrize("policy_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("failure", ["child", "parent"])
def test_public_singleton_fsdp_pointwise_restores_after_exception(
        policy_dtype, failure):
    from jittor._runtime.fallback import forbid_backend_fallbacks

    saved_amp = int(jt.flags.amp_reg)
    outer_amp = jt.amp_flags.prefer32 | jt.amp_flags.keep_reduce
    with jt.flag_scope(use_cuda=0, amp_reg=outer_amp), mock.patch.dict(
            os.environ, {"JITTOR_FSDP2_FLAT": "0"}), forbid_backend_fallbacks():
        _assert_fsdp_public_autocast_disabled()
        fallback_before = jt.core.backend_fallback_count()
        model, events = _public_cpu_pointwise_fsdp_model(policy_dtype)
        value = torch.tensor(
            np.arange(8, dtype=np.float32).reshape(2, 4) / 32,
            dtype=torch.float32, device="cpu")
        with pytest.raises(_FSDPPointwiseFailure, match=failure):
            model(value, failure=failure)
        assert int(jt.flags.amp_reg) == outer_amp
        _assert_fsdp_public_autocast_disabled()
        _assert_legacy_amp_outside_fsdp()
        # Retry the same genuinely sharded instances after their failed call.
        output = model(value)
        _assert_pointwise_fsdp_output_materializes(output, policy_dtype)
        assert events.count("child_gemm") == 2
        assert events.count("root_gemm") == (2 if failure == "parent" else 1)
        assert int(jt.flags.amp_reg) == outer_amp
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        _assert_legacy_amp_outside_fsdp()
        assert jt.core.backend_fallback_count() == fallback_before
    assert int(jt.flags.amp_reg) == saved_amp
    _assert_fsdp_public_autocast_disabled()


def test_public_singleton_fsdp_linear_keeps_native_fp32_accumulation():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from jittor._runtime.fallback import forbid_backend_fallbacks

    saved_amp = int(jt.flags.amp_reg)
    with jt.flag_scope(use_cuda=0, amp_reg=0), mock.patch.dict(
            os.environ, {"JT_BUILD_USE_MKL": "0", "JITTOR_FSDP2_FLAT": "0"}), forbid_backend_fallbacks():
        fallback_before = jt.core.backend_fallback_count()
        layer = torch.nn.Linear(1, 1, device="cpu")
        with torch.no_grad():
            layer.weight.fill_(1.0078125)
            layer.bias.fill_(.00390625)
        fully_shard(layer, mesh=init_device_mesh("cpu", (1,)),
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                        output_dtype=torch.float32))
        value = torch.tensor([[1.0078125]], dtype=torch.float32,
                             device="cpu", requires_grad=True)
        output = layer(value)
        assert output.dtype == torch.float32 and output.device.type == "cpu"
        _assert_fsdp_public_autocast_disabled()
        # Native Torch 2.7.1 CPU rounds once after adding bias. Rounding the
        # product first would land on the midpoint and produce 1.015625.
        np.testing.assert_array_equal(output.detach().numpy(), [[1.0234375]])
        output.square().sum().backward()
        for parameter, expected in (
                (value, [[2.0625]]), (layer.weight, [[2.0625]]),
                (layer.bias, [2.046875])):
            gradient = parameter.grad
            assert gradient is not None
            if callable(getattr(gradient, "full_tensor", None)):
                gradient = gradient.full_tensor()
            assert gradient.dtype == torch.float32
            assert gradient.device.type == "cpu"
            np.testing.assert_array_equal(gradient.detach().numpy(), expected)
        assert layer.weight.dtype == layer.bias.dtype == torch.float32
        assert int(jt.flags.amp_reg) == 0
        assert jt.core.backend_fallback_count() == fallback_before
    assert int(jt.flags.amp_reg) == saved_amp


def test_fsdp2_adamw_first_moment_matches_foreach_lerp_rounding():
    from jittor.optim.algorithms.adam import adam_update

    old_avg_np = np.array([1.2345, -2.3456], dtype=np.float32)
    grad_np = np.array([-3.14159, 2.71828], dtype=np.float32)
    expected_np = old_avg_np + np.float32(0.1) * (grad_np - old_avg_np)
    weighted_sum_np = (np.float32(0.9) * old_avg_np
                       + np.float32(0.1) * grad_np)
    assert not np.array_equal(expected_np, weighted_sum_np)

    with jt.flag_scope(use_cuda=0, amp_reg=0):
        param = jt.array([1.0, -2.0], dtype="float32").stop_grad()
        grad = jt.array(grad_np).stop_grad()
        value = jt.zeros_like(param).stop_grad()
        momentum = jt.array(old_avg_np).stop_grad()
        adam_update(
            param, grad, value, momentum, lr=0.01, eps=1e-8,
            weight_decay=0.0, betas=(0.9, 0.999), step=2,
            decoupled_weight_decay=True, torch_math=True)
        np.testing.assert_array_equal(momentum.numpy(), expected_np)

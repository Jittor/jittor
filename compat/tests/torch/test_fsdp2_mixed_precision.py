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

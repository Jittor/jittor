"""FSDP2 keeps fp32 master shards while gathering bf16 compute weights."""

from types import SimpleNamespace
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

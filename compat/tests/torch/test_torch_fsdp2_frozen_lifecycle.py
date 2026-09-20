"""Regression for mixed frozen/trainable FSDP2 backward ownership."""

from types import SimpleNamespace
from unittest import mock

import jittor as jt
import numpy as np
import pytest

from jittor.compat.fsdp2 import grad_sync


@pytest.fixture(autouse=True)
def cpu_only():
    with jt.flag_scope(use_cuda=0):
        yield


def test_backward_keeps_forward_trainable_var_without_regathering_frozen():
    frozen = SimpleNamespace(requires_grad=False, full_param=None)
    trainable = SimpleNamespace(requires_grad=True, full_param=None)
    state = SimpleNamespace(true_fsdp_params=(frozen, trainable), true_fsdp_module=object())
    optimizer = object()
    value = jt.array([2.0])
    losses = []
    with mock.patch.object(grad_sync, "_fsdp_states_from_optimizers", return_value=[state]), \
            mock.patch.object(grad_sync.shard, "_unshard_module_params") as regather:
        for _ in range(2):
            forward_lora = value * 1.0
            trainable.full_param = forward_lora
            loss = (forward_lora * forward_lora).sum()
            targets = grad_sync.collect_fsdp_full_params_for_backward([optimizer])
            assert targets[0] is forward_lora
            grad = jt.grad(loss, targets)[0]
            losses.append(float(loss.numpy()))
            value = (value - 0.1 * grad).stop_grad()
    assert losses == pytest.approx([4.0, 2.56], abs=1e-5)
    assert float(value.numpy().item()) == pytest.approx(1.28, abs=1e-5)
    regather.assert_not_called()


def test_nonflat_frozen_entry_never_reduce_scatters():
    frozen = SimpleNamespace(requires_grad=False, shard=jt.array([3.0]),
                             full_param=None, shape=(1,), dtype="float32",
                             padded_numel=1)
    trainable = SimpleNamespace(requires_grad=True, shard=jt.array([2.0]),
                                full_param=jt.array([2.0]), shape=(1,),
                                dtype="float32", padded_numel=1)
    state = SimpleNamespace(true_fsdp_params=(frozen, trainable),
                            true_fsdp_flat=False, true_fsdp_world_size=2)
    with mock.patch.object(grad_sync.common, "_reduce_scatter_padded",
                           side_effect=lambda value, group=None: value) as scatter:
        grads = grad_sync._sync_sharded_grads_from_full_grads(
            state, [None, jt.array([1.0])])
    assert grads[0] is None
    assert grads[1] is not None
    assert scatter.call_count == 1


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("explicit_reduce", [False, True])
def test_mixed_precision_collective_dtype_and_master_shards(flat, explicit_reduce):
    frozen = SimpleNamespace(requires_grad=False, shard=jt.array([7., 8.]),
                             shape=(2,), dtype="float32", numel=2,
                             padded_numel=2, flat_offset=0)
    trainable = SimpleNamespace(requires_grad=True, shard=jt.array([3., 4.]),
                                shape=(2,), dtype="float32", numel=2,
                                padded_numel=2, flat_offset=2)
    state = SimpleNamespace(
        true_fsdp_params=(frozen, trainable), true_fsdp_flat=flat,
        true_fsdp_world_size=1, true_fsdp_rank=0,
        true_fsdp_flat_padded_numel=4, true_fsdp_flat_shard_numel=4,
        true_fsdp_flat_shard=jt.array([7., 8., 3., 4.]),
        mp_policy=SimpleNamespace(param_dtype="bfloat16",
                                  reduce_dtype="float32" if explicit_reduce else None),
    )
    received = []

    def scatter(value, group=None):
        received.append((str(value.dtype), value.numpy().copy()))
        return value

    with mock.patch.object(grad_sync.common, "_reduce_scatter_padded",
                           side_effect=scatter), mock.patch.object(
                               grad_sync.shard, "_mark_fsdp_param_var",
                               side_effect=lambda value, *args: value):
        grads = grad_sync._sync_sharded_grads_from_full_grads(
            state, [None, jt.array([1., 2.])])
    expected_dtype = "float32" if explicit_reduce else "bfloat16"
    assert len(received) == 1
    assert received[0][0] == expected_dtype
    np.testing.assert_array_equal(received[0][1], [0., 0., 1., 2.] if flat else [1., 2.])
    assert grads[1].dtype == trainable.shard.dtype
    if flat:
        assert grads[0].dtype == frozen.shard.dtype
        np.testing.assert_array_equal(grads[0].numpy(), [0., 0.])
        np.testing.assert_array_equal(grads[1].numpy(), [1., 2.])
    else:
        assert grads[0] is None

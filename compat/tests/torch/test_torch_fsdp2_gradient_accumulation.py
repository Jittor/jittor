"""Observable FSDP2 accumulation and reduction contracts."""

from types import SimpleNamespace
from unittest import mock

import jittor as jt
import numpy as np
import pytest

from jittor.compat.fsdp2 import grad_sync
from jittor.compat.fsdp2 import shard
from jittor.compat.fsdp2.api import FSDPModule


@pytest.fixture(autouse=True)
def cpu_only():
    with jt.flag_scope(use_cuda=0):
        yield


@pytest.mark.parametrize("flat", [False, True])
def test_disabled_gradient_sync_does_not_reduce_scatter(flat):
    frozen = SimpleNamespace(requires_grad=False, shard=jt.array([9.]),
                             shape=(1,), dtype="float32", padded_numel=1,
                             flat_offset=0, numel=1)
    trainable = SimpleNamespace(requires_grad=True, shard=jt.array([2.]),
                                shape=(1,), dtype="float32", padded_numel=1,
                                flat_offset=1, numel=1)
    state = SimpleNamespace(true_fsdp_params=[frozen, trainable],
                            true_fsdp_flat=flat, true_fsdp_world_size=2,
                            true_fsdp_flat_padded_numel=2,
                            true_fsdp_flat_shard=jt.array([9., 2.]),
                            requires_gradient_sync=False)
    with mock.patch.object(grad_sync.common, "_reduce_scatter_padded",
                           side_effect=AssertionError("unexpected reduce-scatter")) as scatter:
        grad_sync._sync_sharded_grads_from_full_grads(
            state, [None, jt.array([3.])])
    scatter.assert_not_called()


def test_sync_disabled_retains_full_gradient_across_microbatches():
    entry = SimpleNamespace(requires_grad=True, shard=jt.array([2.]),
                            full_param=jt.array([2.]), shape=(1,),
                            dtype="float32", padded_numel=1)
    state = SimpleNamespace(true_fsdp_params=[entry], true_fsdp_flat=False,
                            true_fsdp_world_size=2, requires_gradient_sync=False)
    with mock.patch.object(grad_sync.common, "_reduce_scatter_padded",
                           side_effect=AssertionError("unexpected reduce-scatter")):
        grad_sync._sync_sharded_grads_from_full_grads(state, [jt.array([2.])])
        grad_sync._sync_sharded_grads_from_full_grads(state, [jt.array([5.])])
    np.testing.assert_allclose(state.true_fsdp_pending_full_grads[0].numpy(), [7.])


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("microbatches", [2, 4])
def test_setter_accumulates_full_grads_and_scatter_once(flat, microbatches):
    frozen = SimpleNamespace(requires_grad=False, shard=jt.array([9., 8.]),
                             full_param=None, shape=(2,), dtype="float32",
                             numel=2, padded_numel=2, flat_offset=0)
    trainable = SimpleNamespace(requires_grad=True, shard=jt.array([2., 3.]),
                                full_param=None, shape=(2,), dtype="float32",
                                numel=2, padded_numel=2, flat_offset=2)
    state = SimpleNamespace(true_fsdp_params=[frozen, trainable],
                            true_fsdp_flat=flat, true_fsdp_world_size=2,
                            true_fsdp_rank=0, true_fsdp_flat_padded_numel=4,
                            true_fsdp_flat_shard_numel=2,
                            true_fsdp_flat_shard=jt.array([9., 8.]))
    module = SimpleNamespace(_is_fsdp_module=True, _fsdp_state=state)
    optimizer = SimpleNamespace(param_groups=[{"params": [frozen.shard,
                                                         trainable.shard]}],
                                _build_grad_map=lambda: None,
                                _Optimizer__zero_grad=True)
    entries = {id(frozen.shard): frozen, id(trainable.shard): trainable}
    captured = []

    def scatter(value, group=None):
        captured.append(value.numpy().copy())
        return value[:2] if flat else value[:2]

    with mock.patch.object(grad_sync, "_fsdp_states_from_optimizers",
                           return_value=[state]), mock.patch.object(
                           grad_sync.shard, "_fsdp_param_entry",
                           side_effect=lambda param: (state, entries[id(param)])), mock.patch.object(
                           grad_sync.shard, "_mark_fsdp_param_var",
                           side_effect=lambda value, *args: value), mock.patch.object(
                           grad_sync.common, "_reduce_scatter_padded",
                           side_effect=scatter), mock.patch.object(
                           grad_sync, "refresh_visible_full_grads"):
        for window in range(2):
            FSDPModule.set_requires_gradient_sync(module, False)
            for index in range(microbatches - 1):
                trainable.full_param = jt.array([2., 3.])
                grad = jt.array([float(index + 1), float(2 * (index + 1))])
                grad_sync.fill_fsdp_optimizer_grads_from_grad_map(
                    [optimizer], {id(trainable.full_param): grad})
                assert len(captured) == window
            FSDPModule.set_requires_gradient_sync(module, True)
            trainable.full_param = jt.array([2., 3.])
            grad_sync.fill_fsdp_optimizer_grads_from_grad_map(
                [optimizer], {id(trainable.full_param): jt.array([1., 2.])})
            assert len(captured) == window + 1
            scale = (1 + microbatches * (microbatches - 1) / 2)
            expected = np.asarray([scale, 2 * scale], dtype=np.float32)
            np.testing.assert_allclose(captured[-1][-2:], expected)
            assert state.true_fsdp_pending_full_grads is None
            if not flat:
                np.testing.assert_allclose(optimizer.param_groups[0]["grads"][1].numpy(),
                                           expected / 2)
            object.__setattr__(optimizer, "_Optimizer__zero_grad", True)
            optimizer.param_groups[0]["grads"] = [None, None]
            object.__setattr__(trainable.shard, "_torch_grad", None)


def test_disabling_all_reduce_keeps_scatter_on_pure_shard_mesh():
    entry = SimpleNamespace(requires_grad=True, shard=jt.array([2.]),
                            shape=(1,), dtype="float32", padded_numel=1)
    state = SimpleNamespace(true_fsdp_params=[entry], true_fsdp_flat=False,
                            true_fsdp_world_size=2)
    module = SimpleNamespace(_is_fsdp_module=True, _fsdp_state=state)
    FSDPModule.set_requires_all_reduce(module, False)
    with mock.patch.object(grad_sync.common, "_reduce_scatter_padded",
                           side_effect=lambda value, group=None: value) as scatter:
        grad_sync._sync_sharded_grads_from_full_grads(state, [jt.array([4.])])
    scatter.assert_called_once()


def test_hsdp_all_reduce_disable_refuses_without_losing_pending_grad():
    entry = SimpleNamespace(requires_grad=True, shard=jt.array([2.]),
                            shape=(1,), dtype="float32", padded_numel=1)
    group = SimpleNamespace(size=lambda: 2)
    state = SimpleNamespace(true_fsdp_params=[entry], true_fsdp_flat=False,
                            true_fsdp_world_size=2, replicate_group=group,
                            true_fsdp_pending_full_grads=[jt.array([7.])])
    module = SimpleNamespace(_is_fsdp_module=True, _fsdp_state=state)
    with pytest.raises(NotImplementedError, match="partial-gradient accumulation"):
        FSDPModule.set_requires_all_reduce(module, False)
    assert not hasattr(state, "requires_all_reduce")
    np.testing.assert_allclose(state.true_fsdp_pending_full_grads[0].numpy(), [7.])


def test_explicit_zero_grad_releases_pending_full_gradient_immediately():
    entry = SimpleNamespace(requires_grad=True, shard=jt.array([2.]),
                            full_param=None)
    state = SimpleNamespace(true_fsdp_params=[entry],
                            true_fsdp_pending_full_grads=[jt.array([7.])])
    opt = SimpleNamespace(_Optimizer__zero_grad=True,
                          param_groups=[{"params": [entry.shard]}])
    with mock.patch.object(grad_sync, "_fsdp_states_from_optimizers",
                           return_value=[state]), mock.patch.object(
                           grad_sync.shard, "_fsdp_param_entry",
                           return_value=(state, entry)):
        grad_sync.refresh_visible_full_grads(opt)
    assert state.true_fsdp_pending_full_grads is None


def test_unused_microbatch_preserves_earlier_full_gradient():
    entry = SimpleNamespace(requires_grad=True, shard=jt.array([2.]),
                            shape=(1,), dtype="float32", padded_numel=1)
    state = SimpleNamespace(true_fsdp_params=[entry], true_fsdp_flat=False,
                            true_fsdp_world_size=2, requires_gradient_sync=False)
    with mock.patch.object(grad_sync.common, "_reduce_scatter_padded",
                           side_effect=AssertionError("unsynced collective")):
        grad_sync._sync_sharded_grads_from_full_grads(state, [jt.array([7.])])
        grad_sync._sync_sharded_grads_from_full_grads(state, [None])
    state.requires_gradient_sync = True
    received = []

    def scatter(value, group=None):
        received.append(value.numpy().copy())
        return value

    with mock.patch.object(grad_sync.common, "_reduce_scatter_padded",
                           side_effect=scatter):
        grad_sync._sync_sharded_grads_from_full_grads(state, [None])
    assert len(received) == 1
    np.testing.assert_allclose(received[0], [7.])


@pytest.mark.parametrize("set_to_none", [False, True])
def test_real_optimizer_zero_grad_clears_pending_before_next_backward(set_to_none):
    import torch

    param = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float32))
    opt = torch.optim.AdamW([param], lr=.01)
    entry = SimpleNamespace(requires_grad=True, shard=param, full_param=None)
    state = SimpleNamespace(true_fsdp_initialized=True, true_fsdp_params=[entry],
                            true_fsdp_pending_full_grads=[jt.array([7.])])
    shard._mark_fsdp_param_var(param, state, entry, "shard")
    object.__setattr__(opt, "_Optimizer__zero_grad", False)
    opt.zero_grad(set_to_none=set_to_none)
    assert state.true_fsdp_pending_full_grads is None


def test_one_optimizer_zero_grad_preserves_other_parameters_pending_grad():
    first = SimpleNamespace(shard=jt.array([1.]), full_param=None)
    second = SimpleNamespace(shard=jt.array([2.]), full_param=None)
    state = SimpleNamespace(true_fsdp_params=[first, second],
                            true_fsdp_pending_full_grads=[jt.array([3.]),
                                                          jt.array([7.])])
    first_opt = SimpleNamespace(param_groups=[{"params": [first.shard]}],
                                _Optimizer__zero_grad=True)
    second_opt = SimpleNamespace(param_groups=[{"params": [second.shard]}],
                                 _Optimizer__zero_grad=True)
    entries = {id(first.shard): first, id(second.shard): second}
    with mock.patch.object(grad_sync, "_fsdp_states_from_optimizers",
                           return_value=[state]), mock.patch.object(
                           grad_sync.shard, "_fsdp_param_entry",
                           side_effect=lambda param: (state, entries[id(param)])):
        grad_sync.refresh_visible_full_grads(first_opt)
        assert state.true_fsdp_pending_full_grads[0] is None
        np.testing.assert_allclose(state.true_fsdp_pending_full_grads[1].numpy(), [7.])
        grad_sync.refresh_visible_full_grads(second_opt)
    assert state.true_fsdp_pending_full_grads is None


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("read_first", [False, True])
def test_public_accumulated_gradient_views_keep_optimizer_buffer(
        flat, read_first, monkeypatch, tmp_path):
    """Full/local views expose the same accumulated buffer used by step()."""
    import torch
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    monkeypatch.setenv("JITTOR_FSDP2_FLAT", "1" if flat else "0")
    weight = np.asarray([[0.25, -0.125], [0.5, 0.75]], dtype=np.float32)
    inputs = [
        np.asarray([[1., 2.], [-1., 3.]], dtype=np.float32),
        np.asarray([[0., 4.], [2., -1.]], dtype=np.float32),
    ]
    targets = [
        np.asarray([[0.5, -0.25], [1., 0.125]], dtype=np.float32),
        np.asarray([[0.25, 0.5], [-0.5, 1.]], dtype=np.float32),
    ]
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(weight))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, weight_decay=0.01, foreach=False)
    fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
    optimizer.zero_grad(set_to_none=True)
    expected = np.zeros_like(weight)
    first_gradient = None
    for micro, (x, target) in enumerate(zip(inputs, targets)):
        prediction = model(torch.tensor(x))
        loss = (prediction - torch.tensor(target)).square().mean() / 2
        loss.backward()
        # Independent derivative of mean(square(linear(x)-target))/2.
        expected += ((x @ weight.T - target).T @ x) / target.size
        gradient = model.weight.grad
        assert gradient is not None
        raw = gradient.detach().cpu().numpy().copy()
        observe = read_first or micro == 1
        full = (gradient.full_tensor().detach().cpu().numpy().copy()
                if observe else np.empty((0,), dtype=np.float32))
        local = (gradient.to_local().detach().cpu().numpy().copy()
                 if observe else np.empty((0,), dtype=np.float32))
        # Preserve the complete failed observation before any value assertion.
        np.savez(tmp_path / ("micro-%d.npz" % micro),
                 prediction=prediction.detach().cpu().numpy(),
                 loss=loss.detach().cpu().numpy(),
                 expected_gradient=expected.copy(), raw_gradient=raw,
                 public_full_gradient=full, public_local_gradient=local)
        np.testing.assert_array_equal(raw.reshape(weight.shape), expected)
        if observe:
            np.testing.assert_array_equal(full, expected)
            np.testing.assert_array_equal(local.reshape(weight.shape), expected)
        if first_gradient is None:
            first_gradient = gradient
        else:
            assert gradient is first_gradient
        entry = model.weight._jittor_fsdp2_entry
        state = model._fsdp_state
        index = next(i for i, item in enumerate(state.true_fsdp_params)
                     if item is entry)
        assert state.true_fsdp_last_grads[index] is gradient
        assert entry.last_grad is gradient

    expected_norm = float(np.linalg.norm(expected))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scale = min(1.0, 0.5 / (expected_norm + 1e-6))
    clipped = expected * scale
    full = model.weight.grad.full_tensor().detach().cpu().numpy().copy()
    raw = model.weight.grad.detach().cpu().numpy().copy()
    np.savez(tmp_path / "clipped.npz", expected_gradient=clipped,
             public_full_gradient=full, raw_gradient=raw,
             expected_norm=np.asarray(expected_norm),
             actual_norm=norm.detach().cpu().numpy())
    np.testing.assert_allclose(float(norm), expected_norm, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(full, clipped, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(raw.reshape(weight.shape), clipped,
                               rtol=1e-6, atol=1e-7)

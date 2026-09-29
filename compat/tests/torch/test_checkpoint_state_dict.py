"""Named state-dict transforms reuse the existing optimizer load/state owner."""

import os

import numpy as np
import pytest
import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    get_state_dict, set_model_state_dict, set_optimizer_state_dict, set_state_dict,
)


DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().cpu()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    return value


def array(value):
    return value.detach().clone().cpu().numpy()


def model_and_optimizer():
    model = torch.nn.Linear(4, 3).to(DEVICE)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(np.arange(12, dtype=np.float32).reshape(3, 4) / 50, device=DEVICE))
        model.bias.copy_(torch.tensor([.03, -.02, .01], device=DEVICE))
    optimizer = torch.optim.AdamW([{"params": [model.weight], "lr": .01},
                                 {"params": [model.bias], "lr": .015}],
                                betas=(.7, .93), eps=1e-4, weight_decay=.02)
    return model, optimizer


def update(model, optimizer):
    optimizer.zero_grad()
    model.weight.grad = torch.full_like(model.weight, .25)
    model.bias.grad = torch.full_like(model.bias, -.125)
    optimizer.step()


def test_named_groups_model_optimizer_restore_and_resume():
    model, optimizer = model_and_optimizer()
    update(model, optimizer)
    saved_model, saved_optimizer = clone_tree(get_state_dict(model, optimizer))
    assert list(saved_optimizer["state"]) == ["weight", "bias"]
    assert [group["params"] for group in saved_optimizer["param_groups"]] == [["weight"], ["bias"]]
    update(model, optimizer)
    expected_model = clone_tree(get_model_state_dict(model))
    expected_optimizer = clone_tree(get_optimizer_state_dict(model, optimizer))
    optimizer.param_groups[0]["lr"] = .9
    set_state_dict(model, optimizer, model_state_dict=saved_model, optim_state_dict=saved_optimizer)
    assert [group["lr"] for group in optimizer.param_groups] == [.01, .015]
    for name, param in model.named_parameters():
        assert param.device.type == DEVICE
        np.testing.assert_array_equal(array(param), array(saved_model[name]))
    update(model, optimizer)
    actual_model, actual_optimizer = get_state_dict(model, optimizer)
    for name in expected_model:
        np.testing.assert_allclose(array(actual_model[name]), array(expected_model[name]), atol=2e-7, rtol=2e-6)
        for field in ("step", "exp_avg", "exp_avg_sq"):
            np.testing.assert_allclose(array(actual_optimizer["state"][name][field]),
                                       array(expected_optimizer["state"][name][field]), atol=2e-8, rtol=2e-6)
        for field in ("exp_avg", "exp_avg_sq"):
            assert actual_optimizer["state"][name][field].device.type == DEVICE


def test_multiple_optimizers_and_missing_state():
    model, unused = model_and_optimizer()
    del unused
    optimizers = [torch.optim.SGD([model.weight], lr=.03, momentum=.9),
                  torch.optim.AdamW([model.bias], lr=.07)]
    model.weight.grad = torch.ones_like(model.weight)
    optimizers[0].step()
    saved = clone_tree(get_optimizer_state_dict(model, optimizers))
    assert set(saved["state"]) == {"weight", "bias"}
    assert [group["params"] for group in saved["param_groups"]] == [["weight"], ["bias"]]
    optimizers[0].param_groups[0]["lr"] = .8
    optimizers[1].param_groups[0]["lr"] = .9
    set_optimizer_state_dict(model, optimizers, saved)
    assert [opt.param_groups[0]["lr"] for opt in optimizers] == [.03, .07]
    actual = get_optimizer_state_dict(model, optimizers)
    assert set(actual["state"]) == {"weight", "bias"}
    np.testing.assert_array_equal(array(actual["state"]["weight"]["momentum_buffer"]),
                                  array(saved["state"]["weight"]["momentum_buffer"]))


def test_adamw_initialized_optimizer_keeps_missing_parameter_state():
    model, optimizer = model_and_optimizer()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    saved = clone_tree(get_optimizer_state_dict(model, optimizer))
    assert set(saved["state"]) == {"weight"}
    if not hasattr(torch, "_torch_compat_install_context"):
        return  # Torch 2.6 DCP setter raises KeyError for its own missing state.
    set_optimizer_state_dict(model, optimizer, saved)
    assert set(get_optimizer_state_dict(model, optimizer)["state"]) == {"weight"}


def test_tied_and_frozen_parameter_names():
    model, optimizer = model_and_optimizer()
    model.alias = model.weight
    model.bias.requires_grad_(False)
    model.weight.grad = torch.full_like(model.weight, .25)
    optimizer.step()
    state = get_optimizer_state_dict(model, optimizer)
    assert state["param_groups"][0]["params"] == ["weight"]
    assert set(state["state"]) == {"weight"}
    model_state = get_model_state_dict(model, options=StateDictOptions(ignore_frozen_params=True))
    assert set(model_state) == {"weight", "alias"}
    np.testing.assert_array_equal(array(model_state["weight"]), array(model_state["alias"]))


def test_non_strict_model_restore_reports_real_missing_and_unexpected_keys():
    model, optimizer = model_and_optimizer()
    original_bias = array(model.bias).copy()
    result = set_model_state_dict(
        model, {"weight": torch.ones_like(model.weight), "unexpected": torch.zeros(1, device=DEVICE)},
        options=StateDictOptions(strict=False))
    assert list(result.missing_keys) == ["bias"]
    assert list(result.unexpected_keys) == ["unexpected"]
    np.testing.assert_array_equal(array(model.bias), original_bias)
    np.testing.assert_array_equal(array(model.weight), np.ones((3, 4), dtype=np.float32))


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor unsupported-boundary policy")
def test_unimplemented_state_dict_options_are_not_silently_accepted():
    model, optimizer = model_and_optimizer()
    with pytest.raises(NotImplementedError, match="flatten_optimizer_state_dict"):
        get_optimizer_state_dict(model, optimizer, options=StateDictOptions(flatten_optimizer_state_dict=True))
    with pytest.raises(NotImplementedError, match="prefix removal"):
        get_model_state_dict(model, options=StateDictOptions(keep_submodule_prefixes=False))
    with pytest.raises(ValueError, match="requires full_state_dict"):
        get_model_state_dict(model, options=StateDictOptions(broadcast_from_rank0=True))


def test_mismatched_parameter_groups_refused_before_optimizer_changes():
    if not hasattr(torch, "_torch_compat_install_context"):
        pytest.skip("Jittor's stricter malformed-checkpoint validation")
    model, optimizer = model_and_optimizer()
    update(model, optimizer)
    state = clone_tree(get_optimizer_state_dict(model, optimizer))
    state["param_groups"][1]["params"] = ["weight"]
    state["param_groups"][0]["lr"] = .9
    with pytest.raises(ValueError):
        set_optimizer_state_dict(model, optimizer, state)
    assert optimizer.param_groups[0]["lr"] == .01


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor unsupported-boundary policy")
def test_join_uneven_inputs_refuses_before_training_but_balanced_contexts_work(monkeypatch):
    from jittor.compat import stub_policy
    from jittor.compat.torch.installers import distributed as owner

    previous = stub_policy.set_allow_stub(False)
    try:
        monkeypatch.setattr(owner, "_distributed_world_size", lambda: 2)
        with pytest.raises(NotImplementedError, match="notify/shadow-collective"):
            with owner.Join([object()], enable=True):
                pytest.fail("unsupported Join entered the training block")
        with owner.Join([object()], enable=False) as context:
            assert context.enable is False
        monkeypatch.setattr(owner, "_distributed_world_size", lambda: 1)
        with owner.Join([object()], enable=True):
            pass
    finally:
        stub_policy.set_allow_stub(previous)


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor unsupported-boundary policy")
def test_sharded_dcp_storage_remains_explicitly_unsupported(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
    from torch.distributed._shard.sharded_tensor import ShardedTensor, empty, init_from_local_shards
    from jittor.compat import stub_policy

    previous = stub_policy.set_allow_stub(False)
    try:
        for planner in (DefaultSavePlanner, DefaultLoadPlanner):
            with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
                planner()
        for reader_or_writer in (dcp.FileSystemReader, dcp.FileSystemWriter):
            with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
                reader_or_writer(tmp_path / "sharded")
        with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
            ShardedTensor()
        with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
            init_from_local_shards([torch.ones(2, device=DEVICE)])
        with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
            empty((2,), dtype=torch.float32)
        with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
            dcp.save({"value": torch.ones(2, device=DEVICE)}, checkpoint_id=tmp_path / "sharded")
        assert not (tmp_path / "sharded").exists()
    finally:
        stub_policy.set_allow_stub(previous)


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor unsupported-boundary policy")
def test_sharded_tensor_rejection_does_not_construct_stub_storage(monkeypatch):
    from jittor.compat import stub_policy
    from jittor.compat.torch.installers import distributed as owner

    class UninspectableShards:
        def __bool__(self):
            raise AssertionError("rejection path inspected shards")

        def __getitem__(self, key):
            raise AssertionError("rejection path accessed shards")

    previous = stub_policy.set_allow_stub(False)
    try:
        with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
            owner._api_sharded_tensor_init_from_local_shards(UninspectableShards())

        def unexpected_empty(*args, **kwargs):
            raise AssertionError("rejection path allocated jt.empty storage")

        monkeypatch.setattr(owner.jt, "empty", unexpected_empty)
        with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
            owner._api_sharded_tensor_empty((2,), dtype=np.float32)
    finally:
        stub_policy.set_allow_stub(previous)


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor unsupported-boundary policy")
def test_sharded_tensor_stub_opt_in_keeps_legacy_local_results(monkeypatch):
    from jittor.compat import stub_policy
    from jittor.compat.torch.installers import distributed as owner

    sentinel = object()
    previous = stub_policy.set_allow_stub(True)
    try:
        assert owner._api_sharded_tensor_init_from_local_shards([sentinel]) is sentinel
        monkeypatch.setattr(owner.jt, "empty", lambda *args, **kwargs: sentinel)
        assert owner._api_sharded_tensor_empty((2,), dtype=np.float32) is sentinel
    finally:
        stub_policy.set_allow_stub(previous)

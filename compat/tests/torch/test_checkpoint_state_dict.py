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


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor DCP compatibility")
def test_default_dcp_planners_roundtrip_rank_local_tensor_state(tmp_path):
    import json
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner

    model_weight = torch.tensor(np.arange(6, dtype=np.float32).reshape(2, 3), device=DEVICE)
    optimizer_step = torch.tensor(7, dtype=torch.int64, device=DEVICE)
    source = {
        "model": {"weight": model_weight},
        "optimizer": {"step": optimizer_step, "lr": 0.025},
        "counter": 4,
    }
    checkpoint = tmp_path / "rank-local"
    save_planner = DefaultSavePlanner()
    load_planner = DefaultLoadPlanner()
    dcp.save(source, storage_writer=dcp.FileSystemWriter(checkpoint), planner=save_planner)

    target_weight = torch.zeros_like(model_weight)
    target_step = torch.zeros_like(optimizer_step)
    target = {
        "model": {"weight": target_weight},
        "optimizer": {"step": target_step, "lr": 0.0},
        "counter": 0,
    }
    dcp.load(target, storage_reader=dcp.FileSystemReader(checkpoint), planner=load_planner)

    assert target["model"]["weight"] is target_weight
    assert target["optimizer"]["step"] is target_step
    np.testing.assert_array_equal(array(target_weight), array(model_weight))
    np.testing.assert_array_equal(array(target_step), array(optimizer_step))
    assert target["optimizer"]["lr"] == 0.025
    assert target["counter"] == 4
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["world_size"] == 1
    assert metadata["files"] == ["rank-00000.pt"]
    assert (checkpoint / "generations" / metadata["generation"] / "rank-00000.pt").is_file()


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor DCP compatibility")
def test_dcp_stage_propagates_remote_rank_failure(monkeypatch):
    from jittor.compat.torch.installers import distributed as owner

    monkeypatch.setattr(owner, "_distributed_rank", lambda: 0)
    monkeypatch.setattr(owner, "_distributed_world_size", lambda: 2)
    gathered = []

    def gather(output, local_status, group=None):
        gathered.append(local_status)
        output[:] = [
            local_status,
            {
                "rank": 1,
                "ok": False,
                "error_type": "builtins.OSError",
                "error": "rank-local file write failed",
            },
        ]

    monkeypatch.setattr(owner, "_native_all_gather_object", gather)
    with pytest.raises(RuntimeError) as caught:
        owner._dcp_run_stage(True, "rank file write", lambda: "local success")

    assert gathered == [{
        "rank": 0, "ok": True, "error_type": None, "error": None,
    }]
    assert str(caught.value) == (
        "distributed DCP rank file write failed; rank 1: "
        "builtins.OSError: rank-local file write failed"
    )


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor DCP compatibility")
def test_dcp_stage_reports_local_failure_to_all_ranks(monkeypatch):
    from jittor.compat.torch.installers import distributed as owner

    monkeypatch.setattr(owner, "_distributed_rank", lambda: 1)
    monkeypatch.setattr(owner, "_distributed_world_size", lambda: 2)
    gathered = []

    def gather(output, local_status, group=None):
        gathered.append(local_status)
        output[:] = [{
            "rank": 0, "ok": True, "error_type": None, "error": None,
        }, local_status]

    monkeypatch.setattr(owner, "_native_all_gather_object", gather)

    def fail_write():
        raise OSError("disk quota exceeded")

    with pytest.raises(RuntimeError, match="rank 1: builtins.OSError: disk quota exceeded"):
        owner._dcp_run_stage(True, "rank file write", fail_write)
    assert gathered[0]["ok"] is False
    assert gathered[0]["error_type"] == "builtins.OSError"
    assert gathered[0]["error"] == "disk quota exceeded"


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor DCP compatibility")
def test_distributed_dcp_save_propagates_rank_file_failure(tmp_path, monkeypatch):
    from jittor.compat.torch.installers import distributed as owner

    monkeypatch.setattr(owner, "_dcp_rank_info", lambda *args, **kwargs: (0, 2, True))
    monkeypatch.setattr(owner, "_distributed_rank", lambda: 0)
    monkeypatch.setattr(owner, "_distributed_world_size", lambda: 2)
    gathered = []

    def gather(output, local_status, group=None):
        gathered.append(local_status)
        remote = ({"rank": 1, "ok": True, "error_type": None, "error": None}
                  if len(gathered) == 1 else
                  {"rank": 1, "ok": False, "error_type": "builtins.OSError",
                   "error": "rank 1 disk failure"})
        output[:] = [local_status, remote]

    monkeypatch.setattr(owner, "_native_all_gather_object", gather)
    monkeypatch.setattr(
        owner, "_broadcast_object_list",
        lambda values, src=0, group=None, device=None: values.__setitem__(
            0, "test-generation"),
    )

    class Writer:
        @staticmethod
        def save(state, filename):
            with open(filename, "wb") as stream:
                stream.write(b"rank-zero-state")

    save = owner._make_dcp_save(Writer())
    with pytest.raises(RuntimeError, match="rank 1: builtins.OSError: rank 1 disk failure"):
        save({"step": 1}, checkpoint_id=tmp_path)

    assert len(gathered) == 2
    assert (tmp_path / "generations" / "test-generation" /
            "rank-00000.pt").is_file()
    assert not (tmp_path / "metadata.json").exists()


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor DCP compatibility")
def test_distributed_dcp_load_propagates_rank_file_read_failure(tmp_path, monkeypatch):
    import json
    from jittor.compat.torch.installers import distributed as owner

    generation = "test-generation"
    generation_root = tmp_path / "generations" / generation
    generation_root.mkdir(parents=True)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "format": owner._DCP_FORMAT,
        "world_size": 2,
        "generation": generation,
        "files": ["rank-00000.pt", "rank-00001.pt"],
    }))
    monkeypatch.setattr(owner, "_dcp_rank_info", lambda *args, **kwargs: (0, 2, True))
    monkeypatch.setattr(owner, "_distributed_rank", lambda: 0)
    monkeypatch.setattr(owner, "_distributed_world_size", lambda: 2)
    gathered = []

    def gather(output, local_status, group=None):
        gathered.append(local_status)
        remote = ({"rank": 1, "ok": True, "error_type": None, "error": None}
                  if len(gathered) == 1 else
                  {"rank": 1, "ok": False,
                   "error_type": "builtins.FileNotFoundError",
                   "error": "rank 1 checkpoint shard missing"})
        output[:] = [local_status, remote]

    monkeypatch.setattr(owner, "_native_all_gather_object", gather)

    class Reader:
        @staticmethod
        def load(filename, map_location=None):
            return {"value": 9}

    target = {"value": 0}
    load = owner._make_dcp_load(Reader())
    with pytest.raises(RuntimeError, match="rank 1 checkpoint shard missing"):
        load(target, checkpoint_id=tmp_path)

    assert len(gathered) == 2
    assert target["value"] == 0


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor DCP compatibility")
def test_dcp_single_rank_stage_preserves_local_exception():
    from jittor.compat.torch.installers import distributed as owner

    def fail_write():
        raise OSError("single-rank disk failure")

    with pytest.raises(OSError, match="single-rank disk failure"):
        owner._dcp_run_stage(False, "rank file write", fail_write)


@pytest.mark.skipif(not hasattr(torch, "_torch_compat_install_context"), reason="Jittor unsupported-boundary policy")
def test_sharded_dcp_values_remain_explicitly_unsupported(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner, DefaultSavePlanner
    from torch.distributed._shard.sharded_tensor import ShardedTensor, empty, init_from_local_shards
    from jittor.compat import stub_policy

    previous = stub_policy.set_allow_stub(False)
    try:
        assert isinstance(DefaultSavePlanner(), DefaultSavePlanner)
        assert isinstance(DefaultLoadPlanner(), DefaultLoadPlanner)
        assert dcp.FileSystemReader(tmp_path / "reader").path == tmp_path / "reader"
        assert dcp.FileSystemWriter(tmp_path / "writer").path == tmp_path / "writer"
        with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
            ShardedTensor()
        with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
            init_from_local_shards([torch.ones(2, device=DEVICE)])
        with pytest.raises(NotImplementedError, match="DTensor chunk metadata"):
            empty((2,), dtype=torch.float32)

        class DTensor:
            pass

        with pytest.raises(NotImplementedError, match="chunk metadata and resharding"):
            dcp.save({"value": DTensor()}, checkpoint_id=tmp_path / "sharded")
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

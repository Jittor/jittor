"""Native ZeRO stage-one arithmetic, ownership and checkpoint preflight."""

import sys

import jittor as jt
import numpy as np
import pytest
from jittor.distributed import ZeroRedundancyOptimizer, checkpoint
from jittor.serialization.sharded import RankLocal
from _helpers.native_zero import (model_optimizer, step, numpy_trajectory,
                                  assert_reference, assert_exact, check_bfloat16_resume)


@pytest.mark.parametrize("algorithm", ["Adam", "AdamW", "SGD"])
@pytest.mark.parametrize("frozen", [False, True])
def test_owned_math_accumulation_and_fresh_checkpoint(algorithm, frozen, tmp_path):
    assert "torch" not in sys.modules
    model, optimizer = model_optimizer(algorithm, frozen=frozen)
    baseline, full = model_optimizer(algorithm, zero=False, frozen=frozen)
    references = numpy_trajectory(algorithm, frozen=frozen)
    for index in range(3):
        actual = step(model, optimizer, accumulate=2)
        assert_exact(actual, step(baseline, full, accumulate=2))
        assert_reference(actual, references[index], optimizer)
    assert all("m" not in g and "values" not in g for g in optimizer.param_groups)
    directory = tmp_path / "zero"
    checkpoint.save_training_state(model, optimizer, directory)
    expected = []
    for index in range(3, 5):
        actual = step(model, optimizer, accumulate=2)
        assert_reference(actual, references[index], optimizer)
        expected.append(actual)
    fresh, resumed = model_optimizer(algorithm, frozen=not frozen)
    resumed.lr = 0.9
    checkpoint.load_training_state(fresh, resumed, directory)
    assert fresh.bias.is_stop_grad() == frozen
    for want in expected:
        assert_exact(step(fresh, resumed, accumulate=2), want)
    checkpoint.load_training_state(model, optimizer, directory)
    for want in expected:
        assert_exact(step(model, optimizer, accumulate=2), want)


@pytest.mark.parametrize("field", ["algorithm", "owners", "rank", "world_size", "moment", "flags",
                                   "local_gradient", "global_moment"])
def test_bad_owned_state_refused_before_model_mutation(field, tmp_path):
    model, optimizer = model_optimizer()
    step(model, optimizer)
    directory = tmp_path / field
    checkpoint.save_training_state(model, optimizer, directory)
    state = checkpoint.load(directory)
    zero = state["optimizer"]["state"]["defaults"]["zero_state"]
    if field in ("algorithm", "owners", "rank", "world_size"):
        zero[field] = {"algorithm": "other", "owners": [[1], [0]], "rank": 4, "world_size": 9}[field]
    elif field == "moment":
        zero["state"]["defaults"]["param_groups"][0]["m"][0] = jt.ones((9,), dtype="float32")
    elif field == "flags":
        zero["state"]["defaults"]["param_groups"][0]["n_step"] = -1
    elif field == "local_gradient":
        zero["state"]["defaults"]["param_groups"][0]["grads"][0] = jt.ones((9,))
    else:
        state["optimizer"]["state"]["defaults"]["param_groups"][0]["m"] = [jt.zeros((2, 3))]
    # Reconstituted rank-local values are unwrapped by the reader.
    state["optimizer"]["state"]["defaults"]["zero_state"] = RankLocal(zero)
    state["rng"] = RankLocal(state["rng"])
    broken = tmp_path / (field + "-broken")
    checkpoint.save(state, broken)
    fresh, resumed = model_optimizer()
    before = {name: value.numpy().copy() for name, value in fresh.state_dict().items()}
    before_flags = [bool(p.requires_grad) for p in fresh.parameters()]
    with pytest.raises(ValueError, match="ZeRO checkpoint"):
        checkpoint.load_training_state(fresh, resumed, broken)
    for name, value in fresh.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), before[name])
    assert [bool(p.requires_grad) for p in fresh.parameters()] == before_flags


def test_aliases_are_deduplicated_by_module_and_fixed_group_contract():
    class Tied(jt.Module):
        def __init__(self):
            self.weight = jt.array([0.25]).start_grad()
            self.tied = self.weight

        def execute(self):
            return (self.weight + self.tied).sum()

    model = Tied()
    assert len(model.parameters()) == 1
    optimizer = ZeroRedundancyOptimizer(model.parameters(), jt.optim.SGD, lr=0.125)
    optimizer.step(model())
    np.testing.assert_array_equal(model.weight.numpy(), [0.0])
    assert model.tied is model.weight
    with pytest.raises(ValueError, match="aliases"):
        ZeroRedundancyOptimizer([model.weight, model.tied], lr=0.1)
    with pytest.raises(NotImplementedError, match="fixed"):
        optimizer.add_param_group({"params": [jt.ones(2)]})
    optimizer.param_groups[0]["params"] = [jt.ones(1)]
    with pytest.raises(ValueError, match="binding"):
        optimizer.step()


def test_checkpoint_refuses_pending_accumulation(tmp_path):
    model, optimizer = model_optimizer()
    optimizer.backward(model(jt.ones((4, 3))).sum())
    with pytest.raises(NotImplementedError, match="completed"):
        checkpoint.save_training_state(model, optimizer, tmp_path / "pending")
    assert not (tmp_path / "pending" / "manifest.json").exists()


@pytest.mark.parametrize("algorithm", ["Adam", "AdamW", "SGD"])
def test_late_unfreeze_keeps_native_algorithm_first_touch(algorithm):
    model, optimizer = model_optimizer(algorithm, frozen=True)
    baseline, full = model_optimizer(algorithm, zero=False, frozen=True)
    references = numpy_trajectory(algorithm, frozen_until=2)
    for index in range(5):
        if index == 2:
            model.bias.start_grad()
            baseline.bias.start_grad()
        actual = step(model, optimizer, accumulate=2)
        assert_exact(actual, step(baseline, full, accumulate=2))
        assert_reference(actual, references[index], optimizer)


def test_bfloat16_owned_state_and_raw_checkpoint(tmp_path):
    result = check_bfloat16_resume(tmp_path / "bfloat16-zero")
    assert result["owned_bfloat16_state_numel"] == 2


def test_state_metadata_edit_does_not_change_live_ownership():
    _, optimizer = model_optimizer()
    state = optimizer.state_dict()
    state["defaults"]["zero_state"].value["owners"][0][0] = 1
    assert optimizer.state_partition()["owners"] == [[0], [0]]
    with pytest.raises(ValueError, match="partition"):
        optimizer.validate_state_dict(state)


@pytest.mark.parametrize("algorithm", ["Adam", "AdamW", "SGD"])
def test_removed_group_override_uses_current_default(algorithm, tmp_path):
    model, optimizer = model_optimizer(algorithm)
    baseline, full = model_optimizer(algorithm, zero=False)
    references = numpy_trajectory(algorithm, group_lrs=[[0.01, 0.02]] + [[0.005, 0.02]] * 4)
    expected = []
    for index in range(5):
        if index == 1:
            del optimizer.param_groups[0]["lr"]
            del full.param_groups[0]["lr"]
            optimizer.lr = full.lr = 0.005
        actual = step(model, optimizer)
        assert_exact(actual, step(baseline, full))
        assert_reference(actual, references[index], optimizer)
        if index == 2:
            checkpoint.save_training_state(model, optimizer, tmp_path / "deleted-override")
        if index > 2:
            expected.append(actual)
    fresh, resumed = model_optimizer(algorithm)
    checkpoint.load_training_state(fresh, resumed, tmp_path / "deleted-override")
    assert "lr" not in resumed.param_groups[0]
    assert "lr" not in resumed.local_optimizer.param_groups[0]
    for want in expected:
        assert_exact(step(fresh, resumed), want)

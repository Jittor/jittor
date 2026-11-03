"""Offline numerical and reward-integrity controls; no backend coverage claim."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


SCRIPTS = (Path(__file__).resolve().parents[2]
           / "agent/skills/torch-compat-repo-runbook/scripts")


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    result = {}
    for name in ("trl_rl_probe", "trl_dpo_case", "trl_grpo_case"):
        spec = importlib.util.spec_from_file_location(name + "_offline_test", SCRIPTS / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result[name] = module
    return SimpleNamespace(probe=result["trl_rl_probe"], dpo=result["trl_dpo_case"],
                           grpo=result["trl_grpo_case"])


@pytest.mark.parametrize("candidate", [np.ones((2, 1), dtype=np.float32),
                                      np.ones(2, dtype=np.float64)])
def test_shape_and_dtype_mismatches_are_rejected(modules, candidate):
    with pytest.raises(AssertionError, match="shape/dtype"):
        modules.probe.compare_arrays(np.ones(2, dtype=np.float32), candidate)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_nonfinite_observations_fail_closed(modules, bad, side):
    arrays = {name: np.array([0., 1.]) for name in ("reference", "candidate")}
    arrays[side][1] = bad
    with pytest.raises(AssertionError, match="Non-finite"):
        modules.probe.compare_arrays(**arrays)


def test_exact_comparison_rejects_small_float_difference(modules):
    reference = np.array([1.], dtype=np.float32)
    candidate = np.nextafter(reference, np.float32(2.))
    assert modules.probe.compare_arrays(reference, candidate)["passed"]
    result = modules.probe.compare_arrays(reference, candidate, exact=True)
    assert not result["passed"] and result["mismatched_elements"] == 1


def test_exact_integer_comparison_preserves_all_int64_bits(modules):
    reference = np.array([2 ** 53, 2 ** 63 - 2], dtype=np.int64)
    candidate = reference + 1
    result = modules.probe.compare_arrays(reference, candidate, exact=True)
    assert not result["passed"]
    assert result["mismatched_elements"] == 2
    assert result["max_abs"] == 1.


def test_exact_bool_change_is_visible(modules):
    result = modules.probe.compare_arrays(np.array([True, False]),
                                         np.array([False, False]), exact=True)
    assert not result["passed"] and result["mismatched_elements"] == 1


def test_zero_and_near_zero_norms_are_reported_without_nan(modules):
    zeros = np.zeros(2, dtype=np.float64)
    identical = modules.probe.compare_arrays(zeros, zeros)
    assert identical["passed"] and identical["relative_l2"] == 0.
    assert identical["cosine"] is None
    differing = modules.probe.compare_arrays(zeros, np.array([1e-12, 0.]))
    assert differing["relative_l2"] is None
    assert differing["difference_l2"] == pytest.approx(1e-12)
    near_zero = modules.probe.compare_arrays(np.array([1e-12]), np.array([2e-12]))
    assert near_zero["relative_l2"] == pytest.approx(1.)
    assert near_zero["cosine"] == pytest.approx(1.)


def test_large_parameter_norm_cannot_hide_error_relative_to_update(modules):
    initial = np.array([1000., 1000.], dtype=np.float64)
    reference = initial + np.array([.01, 0.])
    candidate = initial + np.array([.02, 0.])
    result = modules.probe.compare_arrays(reference, candidate, initial=initial)
    assert result["relative_l2"] < 1e-5
    assert result["reference_update_l2"] == pytest.approx(.01)
    assert result["relative_to_update_l2"] == pytest.approx(1.)


def test_no_reference_update_has_no_defined_update_relative_error(modules):
    initial = np.array([1., 2.])
    result = modules.probe.compare_arrays(initial, initial + 1e-6, initial=initial)
    assert result["reference_update_l2"] == 0.
    assert result["relative_to_update_l2"] is None


@pytest.mark.parametrize("initial", [np.zeros(1), np.zeros(2, dtype=np.float32),
                                    np.array([0., np.nan]), np.array([0., np.inf])])
def test_invalid_initial_baseline_cannot_produce_update_statistics(modules, initial):
    reference = np.array([1., 2.], dtype=np.float64)
    with pytest.raises(AssertionError):
        modules.probe.compare_arrays(reference, reference, initial=initial)


@pytest.mark.parametrize("algorithm,columns", [("dpo", {"prompt", "chosen", "rejected"}),
                                               ("grpo", {"prompt", "target"})])
def test_ten_distinct_examples_are_deterministic_and_independent(modules, algorithm, columns):
    case = getattr(modules, algorithm)
    first, second = case.records(), case.records()
    assert first == second and len(first) == 10
    assert len({row["prompt"] for row in first}) == 10
    assert all(set(row) == columns and all(row.values()) for row in first)
    if algorithm == "dpo":
        assert all(row["chosen"] != row["rejected"] for row in first)
    first[0]["prompt"] = "changed"
    assert second == case.records()


def test_reward_depends_on_actual_answer_and_target(modules):
    reward = modules.grpo.answer_reward
    assert reward(" Paris ", "PARIS") == pytest.approx(1.)
    assert reward("Paris", "Paris") > reward("Berlin", "Paris")
    assert reward("Paris", "Paris") > reward("Paris", "Tokyo")
    assert reward("5", "5") > reward("5", "6")
    for completion in ("", "5", "6", "long wrong answer", "5" * 100):
        value = reward(completion, "5")
        assert 0. <= value <= 1.
        assert value == reward(completion, "5")


def test_identical_completions_do_not_get_artificial_reward_variance(modules, monkeypatch, tmp_path):
    """A stub checks reward bookkeeping only; no model or Trainer math runs."""
    class StubTrainer:
        def __init__(self, **kwargs):
            self.reward_function = kwargs["reward_funcs"]
            self.model = SimpleNamespace(config=SimpleNamespace(), generate=None)
            self.ref_model = SimpleNamespace(config=SimpleNamespace())

    trl = ModuleType("trl")
    trl.GRPOTrainer = StubTrainer
    trl.GRPOConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    datasets = ModuleType("datasets")
    datasets.Dataset = SimpleNamespace(from_list=lambda values: values)
    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    args = SimpleNamespace(model=tmp_path / "unused", output=tmp_path,
                           runtime="torch", reference=None, steps=10)
    trainer, metadata = modules.grpo.build(args, None, None, {}, lambda *a, **k: None)
    rewards = trainer.reward_function(prompts=["same", "same"],
                                      completions=["5", "5"], target=["5", "5"])
    assert rewards == [1., 1.]
    assert metadata["nonzero_reward_groups"] == 0
    assert metadata["reward_groups"][0]["nonzero_variance"] is False
    assert metadata["rollout_calls"] == 0  # No fake generation/coverage event.
    assert metadata["training_groups"] == []


def test_evaluation_mean_excludes_padding(modules):
    result = modules.grpo._masked_means([[-2., -4., 1000.]], [[1, 1, 0]])
    assert result == [-3.]


@pytest.mark.parametrize("logps,masks", [([[-1.]], []), ([[-1.]], [[1, 0]]),
                                        ([[-1.]], [[0]])])
def test_incomplete_or_empty_scored_completion_is_rejected(modules, logps, masks):
    with pytest.raises(AssertionError):
        modules.grpo._masked_means(logps, masks)

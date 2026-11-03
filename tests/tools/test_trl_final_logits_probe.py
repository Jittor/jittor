"""Offline controls for using trained weights and comparing full-vocab logits."""

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = (Path(__file__).resolve().parents[2]
           / "agent/skills/torch-compat-repo-runbook/scripts")


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    path = SCRIPTS / "trl_final_logits_probe.py"
    spec = importlib.util.spec_from_file_location("final_logits_offline_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact(probe, path, values):
    np.save(path, values, allow_pickle=False)
    return {"path": path.name, "shape": list(values.shape), "dtype": str(values.dtype),
            "sha256": probe.array_hash(values)}


def test_identical_logits_have_zero_error_and_full_agreement(probe):
    values = np.array([[1000., 1001., -1000.], [3., 1., 2.]], dtype=np.float32)
    result = probe.logit_metrics(values, values)
    assert result["max_abs"] == result["rmse"] == result["relative_l2"] == 0.
    assert result["top1_matches"] == result["positions"] == 2
    assert result["mean_softmax_kl_torch_to_jittor"] == pytest.approx(0.)


def test_logit_constant_shift_changes_raw_values_but_not_distribution(probe):
    values = np.array([[1., 2., 3.]], dtype=np.float32)
    result = probe.logit_metrics(values, values + 1000.)
    assert result["max_abs"] == result["rmse"] == 1000.
    assert result["top1_agreement"] == 1.
    assert result["mean_softmax_kl_torch_to_jittor"] == pytest.approx(0., abs=1e-14)


def test_changed_winning_token_and_kl_are_detected(probe):
    reference = np.array([[1., 2.], [4., 1.]], dtype=np.float32)
    candidate = np.array([[3., 2.], [4., 1.]], dtype=np.float32)
    result = probe.logit_metrics(reference, candidate)
    assert result["top1_matches"] == 1 and result["top1_agreement"] == .5
    assert result["mean_softmax_kl_torch_to_jittor"] > 0.


@pytest.mark.parametrize("values", [np.zeros((0, 2)), np.zeros((1, 1)), np.array(1.)])
def test_empty_or_missing_vocabulary_axis_rejected(probe, values):
    with pytest.raises(AssertionError):
        probe.logit_metrics(values, values)


def test_global_metrics_weight_elements_and_positions_correctly(probe):
    a = np.array([[1., 2.]], dtype=np.float32)
    b = np.array([[3., 4.], [4., 3.]], dtype=np.float32)
    combined = probe.aggregate_logits([probe.logit_metrics(a, a + 1.), probe.logit_metrics(b, b)])
    direct = probe.logit_metrics(np.concatenate([a, b]), np.concatenate([a + 1., b]))
    assert combined["global_relative_l2"] == pytest.approx(direct["relative_l2"])
    assert combined["rmse"] == pytest.approx(direct["rmse"])
    assert combined["top1_agreement"] == direct["top1_agreement"]


def test_altered_final_parameter_artifact_is_rejected(probe, tmp_path):
    path = tmp_path / "weight.npy"
    info = artifact(probe, path, np.array([1., 2.], dtype=np.float32))
    np.save(path, np.array([1., 3.], dtype=np.float32))
    with pytest.raises(AssertionError, match="hash mismatch"):
        probe.checked_array(tmp_path, info)


@pytest.mark.parametrize("change", [{"shape": [1]}, {"dtype": "float64"},
                                   {"path": "../escape.npy"}])
def test_parameter_artifact_metadata_and_directory_are_checked(probe, tmp_path, change):
    info = artifact(probe, tmp_path / "weight.npy", np.ones(2, dtype=np.float32))
    info.update(change)
    with pytest.raises(AssertionError):
        probe.checked_array(tmp_path, info)


@pytest.fixture
def fake_model():
    class Tensor:
        device = "cpu"

        def __init__(self, data):
            self.data = np.array(data, copy=True)
            self.shape, self.dtype = self.data.shape, self.data.dtype

        def numel(self):
            return self.data.size

        def copy_(self, other):
            self.data[...] = other.data

        def to(self, device):
            assert device == self.device
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    parameters = {"a": Tensor(np.array([0., 0.], dtype=np.float32)),
                  "b": Tensor(np.array([0.], dtype=np.float32))}
    model = SimpleNamespace(named_parameters=lambda: iter(parameters.items()))
    torch = SimpleNamespace(no_grad=nullcontext, from_numpy=Tensor)
    return torch, model, parameters


def test_all_trained_parameters_replace_pretrained_values_and_are_verified(probe, tmp_path, fake_model):
    torch, model, parameters = fake_model
    values = {"a": np.array([1., 2.], dtype=np.float32), "b": np.array([3.], dtype=np.float32)}
    trace = {"parameter_names": list(values), "parameter_count": 3,
             "arrays": {"final_parameter/" + name: artifact(probe, tmp_path / (name + ".npy"), array)
                        for name, array in values.items()}}
    checked = []
    hashes = probe.load_trained_parameters(torch, model, trace, tmp_path,
                                           lambda value, name: checked.append(name))
    assert checked == list(values) and set(hashes) == set(values)
    for name, expected in values.items():
        np.testing.assert_array_equal(parameters[name].data, expected)
        assert hashes[name] == probe.array_hash(expected)


def test_copy_noop_cannot_claim_trained_weight_loading(probe, tmp_path, fake_model):
    torch, model, parameters = fake_model
    parameters["a"].copy_ = lambda other: None
    trace = {"parameter_names": ["a", "b"], "parameter_count": 3,
             "arrays": {"final_parameter/a": artifact(probe, tmp_path / "a.npy", np.ones(2, dtype=np.float32))}}
    with pytest.raises(AssertionError, match="Loaded trained weight bytes differ"):
        probe.load_trained_parameters(torch, model, trace, tmp_path, lambda *args: None)


def minimal_trace():
    return {"status": "complete", "runtime": "torch", "device": "cpu", "fallback_count": 0,
            "updates": 10, "protocol": {"steps": 10, "records": [{"prompt": str(i)} for i in range(10)]},
            "parameter_names": ["weight"], "arrays": {"final_parameter/weight": {}}}


@pytest.mark.parametrize("change", [{"status": "running"}, {"runtime": "jittor"},
                                   {"device": "cuda"}, {"fallback_count": 1}, {"updates": 9},
                                   {"arrays": {}}])
def test_wrong_or_incomplete_training_trace_is_rejected(probe, tmp_path, change):
    trace = minimal_trace()
    trace.update(change)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(trace))
    with pytest.raises(AssertionError):
        probe.read_training(path, "torch", "cpu")


def test_cross_runtime_pair_checks_training_initial_state_and_protocol(probe):
    reference = {key: {"fixed": True} for key in probe.TRAINING_SHARED}
    reference["runtime"] = "torch"
    candidate = dict(reference, runtime="jittor")
    probe.validate_training_pair(candidate, reference)
    candidate["initial"] = {"changed": True}
    with pytest.raises(AssertionError, match="mismatch: initial"):
        probe.validate_training_pair(candidate, reference)

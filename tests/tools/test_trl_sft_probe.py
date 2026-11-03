"""Small offline negative controls for the streamed TRL training comparator."""

import copy
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def probe():
    path = ROOT / "agent/skills/torch-compat-repo-runbook/scripts/trl_sft_probe.py"
    spec = importlib.util.spec_from_file_location("trl_sft_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def experiment(probe, tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    base = {"status": "complete", "fallback_count": 0, "device": "cpu", "versions": {},
            "trl_source": {"source_verified": True}, "protocol": {"steps": 1}, "precision": {},
            "model_artifacts": {}, "initial": {}, "parameter_names": ["weight"],
            "parameter_count": 2, "probe_sha256": "probe", "runner_sha256": "runner", "arrays": {}}
    for directory, runtime in ((reference, "torch"), (candidate, "jittor")):
        directory.mkdir()
        manifest = copy.deepcopy(base)
        manifest["runtime"] = runtime
        for index, (name, data) in enumerate((
                ("step/0/logits", np.array([1., 2.], dtype=np.float32)),
                ("step/0/loss", np.array(.5, dtype=np.float32)),
                ("step/0/intermediate_grad::embedding_output", np.array([.2, .3], dtype=np.float32)),
                ("step/0/gradient::weight", np.array([.1, .2], dtype=np.float32)),
                ("step/0/parameter::weight", np.array([.9, 1.8], dtype=np.float32)),
                ("step/0/input::input_ids", np.array([1, 2], dtype=np.int64)))):
            relative = "%d.npy" % index
            np.save(directory / relative, data)
            manifest["arrays"][name] = {"path": relative, "sha256": probe.digest(directory / relative),
                "shape": list(data.shape), "dtype": str(data.dtype), "device": "cpu"}
        probe.write_json(directory / "manifest.json", manifest)
    return SimpleNamespace(reference=reference, candidate=candidate,
                           output=tmp_path / "comparison.json", atol=1e-5, rtol=1e-4)


def manifest(experiment):
    return json.loads((experiment.candidate / "manifest.json").read_text())


def test_identical_observations_pass(probe, experiment):
    probe.compare(experiment)
    result = json.loads(experiment.output.read_text())
    assert result["passed"] and result["tensor_count"] == 6


def test_changed_parameter_gradient_fails_even_when_loss_and_parameters_match(probe, experiment):
    changed = manifest(experiment)
    info = changed["arrays"]["step/0/gradient::weight"]
    target = experiment.candidate / info["path"]
    np.save(target, np.array([.1, .3], dtype=np.float32))
    info["sha256"] = probe.digest(target)
    probe.write_json(experiment.candidate / "manifest.json", changed)
    with pytest.raises(SystemExit) as error:
        probe.compare(experiment)
    assert error.value.code == 1
    result = json.loads(experiment.output.read_text())
    assert not result["results"]["step/0/gradient::weight"]["passed"]


def test_missing_parameter_gradient_fails(probe, experiment):
    changed = manifest(experiment)
    del changed["arrays"]["step/0/gradient::weight"]
    probe.write_json(experiment.candidate / "manifest.json", changed)
    with pytest.raises(AssertionError, match="missing gradients"):
        probe.compare(experiment)


def test_gradient_missing_on_both_sides_is_not_a_matching_observation(probe, experiment):
    for directory in (experiment.reference, experiment.candidate):
        changed = json.loads((directory / "manifest.json").read_text())
        del changed["arrays"]["step/0/gradient::weight"]
        probe.write_json(directory / "manifest.json", changed)
    with pytest.raises(AssertionError, match="missing gradients"):
        probe.compare(experiment)


def test_tampered_array_fails_before_numeric_comparison(probe, experiment):
    info = manifest(experiment)["arrays"]["step/0/gradient::weight"]
    np.save(experiment.candidate / info["path"], np.array([.1, .3], dtype=np.float32))
    with pytest.raises(AssertionError, match="changed after measurement"):
        probe.compare(experiment)


@pytest.mark.parametrize("key,value", [
    ("protocol", {"steps": 2}), ("versions", {"trl": "unexpected"}),
    ("initial", {"parameter::weight": {"sha256": "different"}}),
    ("trl_source", {"source_verified": False}),
])
def test_protocol_or_provenance_mismatch_fails(probe, experiment, key, value):
    changed = manifest(experiment)
    changed[key] = value
    probe.write_json(experiment.candidate / "manifest.json", changed)
    with pytest.raises(AssertionError, match="mismatch: " + key):
        probe.compare(experiment)


@pytest.mark.parametrize("key,value", [("status", "running"), ("fallback_count", 1)])
def test_partial_run_and_fallback_cannot_pass(probe, experiment, key, value):
    changed = manifest(experiment)
    changed[key] = value
    probe.write_json(experiment.candidate / "manifest.json", changed)
    with pytest.raises(AssertionError, match="complete with zero fallback"):
        probe.compare(experiment)


@pytest.fixture
def buffer_runtime():
    """Only the public tensor surface needed for buffer artifact validation."""
    class Buffer:
        device = "cpu"

        def __init__(self, data):
            self.data = np.array(data, copy=True)
            self.shape, self.dtype = self.data.shape, self.data.dtype
            self.copies = 0

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

        def to(self, device):
            assert device == self.device
            return self

        def copy_(self, other):
            self.data[...] = other.data
            self.copies += 1
            return self

    buffers = {"rotary.inv_freq": Buffer(np.array([1., .1], dtype=np.float32))}
    model = SimpleNamespace(named_buffers=lambda: iter(buffers.items()))
    torch = SimpleNamespace(no_grad=nullcontext, from_numpy=Buffer)
    return torch, model, buffers


def test_oracle_creates_buffer_artifact_and_both_runtimes_load_it(probe, tmp_path, buffer_runtime):
    torch, model, buffers = buffer_runtime
    target = tmp_path / "buffers.npz"
    expected = buffers["rotary.inv_freq"].data.copy()
    oracle_hash = probe.synchronize_initial_buffers(torch, model, target, "torch")
    assert buffers["rotary.inv_freq"].copies == 1  # The writer also loads the artifact.
    buffers["rotary.inv_freq"].data[:] = 42.
    candidate_hash = probe.synchronize_initial_buffers(torch, model, target, "jittor")
    assert candidate_hash == oracle_hash == probe.digest(target)
    np.testing.assert_array_equal(buffers["rotary.inv_freq"].data, expected)


def test_candidate_cannot_create_its_own_buffer_reference(probe, tmp_path, buffer_runtime):
    torch, model, _ = buffer_runtime
    target = tmp_path / "missing.npz"
    with pytest.raises(ValueError, match="existing oracle"):
        probe.synchronize_initial_buffers(torch, model, target, "jittor")
    assert not target.exists()


@pytest.mark.parametrize("arrays,message", [
    ({"wrong_name": np.ones(2, dtype=np.float32)}, "names differ"),
    ({"rotary.inv_freq": np.ones(3, dtype=np.float32)}, "shape mismatch"),
    ({"rotary.inv_freq": np.ones(2, dtype=np.float64)}, "dtype mismatch"),
    ({"rotary.inv_freq": np.array([1., np.nan], dtype=np.float32)}, "Non-finite"),
])
def test_bad_buffer_artifact_fails_before_any_copy(probe, tmp_path, buffer_runtime, arrays, message):
    torch, model, buffers = buffer_runtime
    target = tmp_path / "buffers.npz"
    np.savez(target, **arrays)
    with pytest.raises(ValueError, match=message):
        probe.synchronize_initial_buffers(torch, model, target, "jittor")
    assert buffers["rotary.inv_freq"].copies == 0

"""Offline negative contracts for complete ecosystem state and gradient transfer."""

import ast
from contextlib import nullcontext
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
ECOSYSTEM = ROOT / "compat/tests/torch"


def _functions(filename, names, **globals_):
    tree = ast.parse((ECOSYSTEM / filename).read_text(encoding="utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    namespace = {"np": np, "hashlib": hashlib, **globals_}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, filename, "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def runner():
    return _functions("_ecosystem_runner.py", {
        "_tensor_manifest", "_snapshot_state", "_restore_state",
        "_state_fingerprints", "_numpy_snapshot", "_collect_arrays",
    })


class Tensor:
    """Only the public serialization/gradient protocol; no framework import."""

    def __init__(self, array, *, sparse=False, requires_grad=False):
        self.array = np.array(array, copy=True)
        self.layout = "torch.sparse_coo" if sparse else "torch.strided"
        self.shape = self.array.shape
        self.dtype = "torch." + str(self.array.dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self.copies = 0

    def sparse_dim(self):
        return len(self.shape)

    def dense_dim(self):
        return 0

    def is_coalesced(self):
        return True

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        if self.layout == "torch.sparse_coo":
            raise TypeError("sparse tensors require explicit to_dense")
        return self.array

    def to_dense(self):
        return Tensor(self.array)

    def to_sparse(self):
        return Tensor(self.array, sparse=True)

    def copy_(self, source):
        assert self.layout == source.layout
        self.array[...] = source.array
        self.copies += 1
        return self


TORCH = SimpleNamespace(from_numpy=Tensor, no_grad=nullcontext)


def test_sparse_buffer_round_trip_preserves_layout_and_bool_values(runner):
    saved = Tensor([[False, True], [True, False]], sparse=True)
    arrays, manifest = runner._snapshot_state([("alignment_heads", saved)])
    assert arrays["alignment_heads"].dtype == np.bool_
    assert manifest["alignment_heads"] == {
        "layout": "sparse_coo", "shape": [2, 2], "dtype": "bool",
        "sparse_dim": 2, "dense_dim": 0,
    }
    target = Tensor(np.zeros((2, 2), dtype=bool), sparse=True)
    runner._restore_state(TORCH, [("alignment_heads", target)], arrays, manifest, lambda x: x)
    restored, restored_manifest = runner._snapshot_state([("alignment_heads", target)])
    assert target.copies == 1
    assert restored_manifest == manifest
    np.testing.assert_array_equal(restored["alignment_heads"], arrays["alignment_heads"])
    assert runner._state_fingerprints(restored) == runner._state_fingerprints(arrays)


@pytest.mark.parametrize("difference", ["layout", "shape", "dtype", "missing"])
def test_state_metadata_failure_precedes_any_parameter_mutation(runner, difference):
    entries = [("parameter", Tensor([1.0])), ("buffer", Tensor([True], sparse=True))]
    arrays, manifest = runner._snapshot_state(entries)
    if difference == "missing":
        del manifest["buffer"]
    elif difference == "layout":
        manifest["buffer"]["layout"] = "strided"
    elif difference == "shape":
        arrays["buffer"] = np.ones((1, 1), dtype=bool)
    else:
        arrays["buffer"] = np.ones((1,), dtype=np.float32)
    with pytest.raises(ValueError):
        runner._restore_state(TORCH, entries, arrays, manifest, lambda x: x)
    assert all(value.copies == 0 for _, value in entries)


def test_legacy_npz_is_accepted_only_for_dense_state(runner):
    dense = Tensor(np.zeros((2,), dtype=np.float32))
    arrays = {"value": np.ones((2,), dtype=np.float32)}
    runner._restore_state(TORCH, [("value", dense)], arrays, None, lambda x: x)
    np.testing.assert_array_equal(dense.array, arrays["value"])
    sparse = Tensor([False, True], sparse=True)
    with pytest.raises(ValueError, match="layout manifest"):
        runner._restore_state(TORCH, [("value", sparse)], arrays, None, lambda x: x)


def _gradient_fixture():
    parameter = Tensor([0.3], requires_grad=True)
    parameter.grad = Tensor([0.8])
    mel = Tensor([[0.5]], requires_grad=True)
    mel.grad = Tensor([[0.2]])
    tokens = Tensor([1])
    model = SimpleNamespace(named_parameters=lambda: iter([("weight", parameter)]))
    return model, {"mel": mel, "tokens": tokens}, Tensor([[0.1]]), parameter


def test_all_trainable_and_floating_input_gradients_are_captured(runner):
    model, inputs, output, _ = _gradient_fixture()
    arrays = runner._collect_arrays(model, inputs, output, required_parameters={"weight"})
    assert set(arrays) == {"__output__", "grad::weight", "ingrad::mel"}


@pytest.mark.parametrize("failure", ["parameter", "input", "shape", "nan", "inf"])
def test_strict_gradient_capture_rejects_broken_training_paths(runner, failure):
    model, inputs, output, parameter = _gradient_fixture()
    if failure == "parameter":
        parameter.grad = None
    elif failure == "input":
        inputs["mel"].grad = None
    elif failure == "shape":
        parameter.grad = Tensor([[0.8]])
    elif failure == "nan":
        parameter.grad = Tensor([np.nan])
    else:
        output = Tensor([[np.inf]])
    with pytest.raises(AssertionError):
        runner._collect_arrays(model, inputs, output, required_parameters={"weight"})


def test_legacy_case_does_not_acquire_new_gradient_requirements(runner):
    model, inputs, output, parameter = _gradient_fixture()
    parameter.grad = None
    arrays = runner._collect_arrays(model, inputs, output)
    assert "grad::weight" not in arrays


@pytest.mark.parametrize("required", [False, True])
def test_missing_whisper_is_visible_and_required_mode_fails(required):
    harness = _functions("_ecosystem_harness.py", {"_require_case_dependencies"},
                         _missing_distributions=lambda names: ["whisper"],
                         _enabled=lambda flag: required)
    expected = AssertionError if required else unittest.SkipTest
    with pytest.raises(expected, match="missing OpenAI Whisper dependency: whisper"):
        harness._require_case_dependencies(unittest.TestCase(), "openai_whisper", ("whisper",))


@pytest.mark.parametrize("difference", ["nan", "shape", "dtype", "missing", "extra", "layout"])
def test_comparison_rejects_false_green_arrays_and_metadata(difference):
    harness = _functions("_ecosystem_harness.py", {"_assert_strict_results"})
    reference = {"__output__": np.ones((1,)), "grad::weight": np.ones((1,)),
                 "ingrad::mel": np.ones((1,))}
    candidate = {key: value.copy() for key, value in reference.items()}
    report = {"state_manifest": {"buffer": "sparse_coo"},
              "state_fingerprints": {"buffer": "fingerprint"},
              "required_parameter_gradients": ["weight"],
              "required_input_gradients": ["mel"]}
    candidate_report = dict(report)
    if difference == "nan":
        candidate["grad::weight"][:] = np.nan
    elif difference == "shape":
        candidate["grad::weight"] = np.ones((1, 1))
    elif difference == "dtype":
        candidate["grad::weight"] = candidate["grad::weight"].astype("float32")
    elif difference == "missing":
        del candidate["ingrad::mel"]
    elif difference == "extra":
        candidate["grad::unexpected"] = np.ones((1,))
    else:
        candidate_report["state_manifest"] = {"buffer": "strided"}
    with pytest.raises(AssertionError):
        harness._assert_strict_results(unittest.TestCase(), reference, candidate,
                                       report, candidate_report)


def test_strict_capture_preserves_dtype_for_comparison(runner):
    model, inputs, output, parameter = _gradient_fixture()
    parameter.grad = Tensor(np.array([0.8], dtype="float64"))
    output = Tensor(np.array([[0.1]], dtype="float16"))
    arrays = runner._collect_arrays(model, inputs, output, required_parameters={"weight"})
    assert arrays["__output__"].dtype == np.dtype("float16")
    assert arrays["grad::weight"].dtype == np.dtype("float64")

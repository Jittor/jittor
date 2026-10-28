"""Synthetic rejection tests for the FlashAttention artifact comparator."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from _helpers.child_process import run_python_child

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "agent/skills/flash-attention-torch-compat/scripts/compare_training.py"
)


@pytest.fixture(scope="module")
def comparison():
    spec = importlib.util.spec_from_file_location("flash_training_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, arrays, metadata):
    np.savez(path, **arrays)
    metadata = dict(metadata, artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(metadata))


@pytest.fixture(params=("float16", "bfloat16"))
def artifacts(tmp_path, request):
    arrays = {f"initial_{index}": np.array([index + 1.0], dtype="float32") for index in range(3)}
    for step in range(3):
        for field in ("output", "loss", "input_grad"):
            arrays[f"step{step}_{field}"] = np.array([1e-5], dtype="float32")
        for index in range(3):
            arrays[f"step{step}_grad{index}"] = np.array([1e-5], dtype="float32")
            arrays[f"step{step}_param{index}"] = arrays[f"initial_{index}"] - (step + 1) * 1e-7
    metadata = {
        "flash_attn": "2.7.4.post1",
        "api": "flash_attn_func",
        "dtype": request.param,
        "head_dim": 32,
        "causal": False,
        "perturbation": 0.0,
        "device": "synthetic",
        "tf32": False,
        "precision": {"matmul": False, "cudnn": False, "cudnn_benchmark": False,
                      "matmul_precision": "highest"},
        "matmul_precision": "highest",
        "arrays": len(arrays),
        "baseline": "a" * 40,
        "upstream": {"commit": "b" * 40},
        "runtime_threads": 2,
        "script_sha256": "c" * 64,
        "source_hashes": {name: "c" * 64 for name in
                          ("flash_attn/__init__.py", "flash_attn/flash_attn_interface.py")},
        "input_hashes": {name: "d" * 64 for name in ("q", "k", "v", "noise", "target")},
        "jittor_source_clean": True,
        "jittor_source_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "inputs": {
            "steps": 3,
            "optimizer": "SGD",
            "learning_rate": 0.01,
            "dropout": 0.0,
            "parameter_dtype": "float32",
            "gradient_dtype": "float32",
        },
    }
    reference, candidate = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    _write(reference, arrays, dict(metadata, runtime="torch", fallback_count=None))
    candidate_metadata = dict(metadata, runtime="jittor", fallback_count=0)
    _write(candidate, arrays, candidate_metadata)
    return reference, candidate, arrays, candidate_metadata


def test_identical_trajectories_pass(artifacts, comparison):
    reference, candidate, _, _ = artifacts
    result = comparison.compare(reference, candidate)
    assert result["status"] == "passed"
    assert len(result["tensors"]) == 39


def test_one_small_gradient_cannot_disappear(artifacts, comparison):
    reference, candidate, arrays, metadata = artifacts
    arrays["step1_grad1"] = np.zeros_like(arrays["step1_grad1"])
    _write(candidate, arrays, metadata)
    result = comparison.compare(reference, candidate)
    assert result["status"] == "failed"
    assert any(row["tensor"] == "step1_grad1" and not row["passed"] for row in result["tensors"])


def test_small_update_cannot_hide_behind_large_parameter(artifacts, comparison):
    reference, candidate, arrays, metadata = artifacts
    arrays["step0_param0"] = arrays["initial_0"].copy()
    _write(candidate, arrays, metadata)
    result = comparison.compare(reference, candidate)
    rows = {row["tensor"]: row for row in result["tensors"]}
    assert rows["step0_param0"]["passed"]
    assert not rows["step0_param0_update"]["passed"]
    assert result["status"] == "failed"


@pytest.mark.parametrize(
    "key,value",
    [
        ("jittor_source_clean", False),
        ("fallback_count", 1),
        ("script_sha256", "d" * 64),
        ("tf32", True),
        ("input_hashes", {name: "e" * 64 for name in ("q", "k", "v", "noise", "target")}),
    ],
)
def test_invalid_provenance_or_protocol_is_rejected(artifacts, key, value, comparison):
    reference, candidate, arrays, metadata = artifacts
    metadata[key] = value
    _write(candidate, arrays, metadata)
    with pytest.raises(AssertionError):
        comparison.compare(reference, candidate)


@pytest.mark.parametrize("defect", ("missing", "nonfinite", "stale_hash"))
def test_incomplete_or_corrupted_arrays_are_rejected(artifacts, defect, comparison):
    reference, candidate, arrays, metadata = artifacts
    if defect == "missing":
        del arrays["step2_input_grad"]
    else:
        arrays["step1_input_grad"] = np.array([np.nan], dtype="float32")
    if defect == "stale_hash":
        np.savez(candidate, **arrays)
    else:
        _write(candidate, arrays, metadata)
    with pytest.raises(AssertionError):
        comparison.compare(reference, candidate)


def test_matched_but_invalid_precision_is_rejected(artifacts, comparison):
    reference, candidate, arrays, _metadata = artifacts
    for path in (reference, candidate):
        metadata = json.loads(path.with_suffix(".json").read_text())
        metadata["precision"]["matmul"] = True
        _write(path, arrays, metadata)
    with pytest.raises(AssertionError):
        comparison.compare(reference, candidate)


def test_both_sides_omitting_public_source_are_rejected(artifacts, comparison):
    reference, candidate, arrays, _metadata = artifacts
    for path in (reference, candidate):
        metadata = json.loads(path.with_suffix(".json").read_text())
        del metadata["source_hashes"]["flash_attn/__init__.py"]
        _write(path, arrays, metadata)
    with pytest.raises(AssertionError):
        comparison.compare(reference, candidate)


def test_optimized_python_cannot_disable_validation(tmp_path):
    result = run_python_child(
        [
            "-O",
            str(SCRIPT),
            str(tmp_path / "missing-reference.npz"),
            str(tmp_path / "missing-candidate.npz"),
            "--output",
            str(tmp_path / "result.json"),
        ],
        timeout=30,
        merge_stderr=True,
    )
    assert result.returncode != 0
    assert "Comparison checks require Python without -O" in result.stdout
    assert not (tmp_path / "result.json").exists()

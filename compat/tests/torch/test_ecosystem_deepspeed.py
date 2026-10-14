"""Dedicated real DeepSpeed Engine gate, separate from ordinary model cases."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pytest

from _helpers.child_process import PYTHON, run_python_child
from _ecosystem_harness import (
    REAL_TORCH_PYTHON, _comparison_floor, _distributions_available,
    _divergence, _enabled, _runner_package_site, _versions,
    EcosystemComparison,
)

_RUNNER = Path(__file__).with_name("_ecosystem_deepspeed_runner.py")
_DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
if _DEVICE not in ("cpu", "cuda"):
    raise ValueError("DeepSpeed Engine gate requires exactly cpu or cuda")


def _run_engine(python, runtime, output, optimizer, zero_stage, perturbation):
    arguments = [str(_RUNNER), "--runtime", runtime, "--device", _DEVICE,
                 "--optimizer", optimizer, "--zero-stage", str(zero_stage),
                 "--perturbation", str(perturbation), "--output", str(output)]
    environment = os.environ.copy()
    site = _runner_package_site(python)
    if site:
        environment["JITTOR_ECOSYSTEM_PACKAGE_SITE"] = site
    else:
        environment.pop("JITTOR_ECOSYSTEM_PACKAGE_SITE", None)
    if runtime == "jittor":
        completed = run_python_child(arguments, env=environment, inherit=False,
                                     merge_stderr=True, timeout=1800)
    else:
        environment["PYTHONPATH"] = ""
        environment.pop("LD_LIBRARY_PATH", None)
        for key in ("JITTOR_SOURCE_ROOT", "JITTOR_HOME", "JITTOR_TORCH_CACHE_ROOT",
                    "JITTOR_TORCH_SHIM", "JITTOR_TORCH_KEEP_HOME", "JT_BACKEND",
                    "JT_USE_CUDA", "JT_BUILD_NVCC_PATH", "use_cuda", "nvcc_path",
                    "REAL_TORCH_SITE"):
            environment.pop(key, None)
        if _DEVICE == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        completed = subprocess.run([python, *arguments], env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf8", errors="replace", timeout=1800)
    assert completed.returncode == 0, completed.stdout[-8000:]
    assert output.is_file(), "DeepSpeed Engine runner did not write its observations"


def _compare_engine_outputs(actual, reference):
    assert set(actual.files) == set(reference.files), "DeepSpeed observation schema differs"
    ours = json.loads(str(actual["metadata"]))
    theirs = json.loads(str(reference["metadata"]))
    assert ours["runtime"] == "jittor" and theirs["runtime"] == "torch"
    assert theirs["torch_binary"].endswith((".so", ".pyd"))
    assert _versions(ours) == _versions(theirs)
    for key in ("device", "world_size", "optimizer", "zero_stage", "perturbation",
                "engine_class", "parameter_gradients_per_microstep", "input_gradients_per_microstep",
                "microsteps", "global_steps", "runtime_conditions", "optimizer_groups"):
        assert ours[key] == theirs[key], "DeepSpeed protocol differs: " + key
    assert ours["device"] == _DEVICE and ours["world_size"] == 1
    assert ours["fallback_policy"] == "error" and ours["fallback_count"] == 0
    assert bool(ours["backend"]["use_cuda"]) == (_DEVICE == "cuda")
    keys = [key for key in actual.files if key != "metadata"]
    gradient_keys = [key for key in keys if "_grad/" in key]
    forward_keys = [key for key in keys if key not in gradient_keys]
    errors = {}
    for category, group, tolerance in (
            ("forward_and_state", forward_keys, EcosystemComparison.forward_tolerance),
            ("gradients", gradient_keys, EcosystemComparison.backward_tolerance)):
        assert group, "DeepSpeed runner omitted " + category
        floor = _comparison_floor(reference, group)
        for key in group:
            assert actual[key].shape == reference[key].shape, key
            assert np.isfinite(actual[key]).all() and np.isfinite(reference[key]).all(), key
            difference = _divergence(actual[key], reference[key], floor)
            assert difference <= tolerance, (key, difference, tolerance)
            errors[category] = max(errors.get(category, 0), difference)
    return errors


@pytest.mark.parametrize("zero_stage", (0, 1, 2))
@pytest.mark.parametrize("optimizer", ("Adam", "AdamW"))
@pytest.mark.parametrize("perturbation", (0., .125))
def test_real_deepspeed_engine_training_and_resume(zero_stage, optimizer, perturbation):
    if not REAL_TORCH_PYTHON:
        if _enabled("JITTOR_REQUIRE_REAL_TORCH"):
            pytest.fail("DeepSpeed gate requires configured REAL_TORCH_PYTHON")
        pytest.skip("independent Torch interpreter is not configured")
    if not _distributions_available(("deepspeed",)):
        if _enabled("JITTOR_REQUIRE_DEEPSPEED"):
            pytest.fail("DeepSpeed gate requires the installed pinned deepspeed package")
        pytest.skip("deepspeed is not installed")
    import jittor as jt
    import torch
    assert issubclass(torch.Tensor, jt.Var), "DeepSpeed gate must run in the Jittor Torch frontend"
    if _DEVICE == "cuda":
        from _helpers import capability
        status = capability.check_accelerator("cuda", backend=jt)
        assert status.enabled, status.reason
    with tempfile.TemporaryDirectory(prefix="deepspeed-engine-") as directory:
        actual_path = Path(directory) / "jittor.npz"
        reference_path = Path(directory) / "torch.npz"
        _run_engine(REAL_TORCH_PYTHON, "torch", reference_path, optimizer, zero_stage, perturbation)
        _run_engine(PYTHON, "jittor", actual_path, optimizer, zero_stage, perturbation)
        with np.load(actual_path) as actual, np.load(reference_path) as reference:
            errors = _compare_engine_outputs(actual, reference)
        print("DeepSpeed", _DEVICE, optimizer, zero_stage, perturbation, errors)

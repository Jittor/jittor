"""Artifact isolation and failure cleanup need no framework or GPUs."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import pytest

from _helpers.child_process import PYTHON, child_env


def _launcher():
    path = Path(__file__).resolve().parents[2] / "compat/tests/torch/_ecosystem_deepspeed_multirank.py"
    spec = importlib.util.spec_from_file_location("deepspeed_multirank_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_identity_and_previous_evidence_cannot_be_overwritten(tmp_path):
    launch = _launcher()
    first = launch.reserve_case(tmp_path, 1, "AdamW", "bfloat16", 2, 0.)
    evidence = first / "rank0.npz"
    evidence.write_bytes(b"previous numerical evidence")
    with pytest.raises(FileExistsError):
        launch.reserve_case(tmp_path, 1, "AdamW", "bfloat16", 2, 0.)
    assert evidence.read_bytes() == b"previous numerical evidence"
    second = launch.reserve_case(tmp_path, 2, "AdamW", "bfloat16", 2, 0.)
    perturbed = launch.reserve_case(tmp_path, 2, "AdamW", "bfloat16", 2, .125)
    assert len({first, second, perturbed}) == 3
    assert json.loads((second / "protocol.json").read_text())["zero_stage"] == 2
    assert all(part in second.parts for part in ("zero2", "AdamW", "bfloat16", "world2"))
    with pytest.raises(ValueError, match="Unverified"):
        launch.reserve_case(tmp_path, 2, "AdamW", "float32", 1, 0.)


def test_artifacts_inside_checkout_are_rejected():
    root = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="outside"):
        _launcher().reserve_case(root / "forbidden-output", 2, "AdamW", "bfloat16", 2, 0.)
    with pytest.raises(ValueError, match="outside"):
        _launcher()._rank_environment("jittor", 0, "0", root, "")


def test_rank_caches_and_oracle_framework_environment_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JITTOR_HOME", "/foreign/cache")
    monkeypatch.setenv("JITTOR_TORCH_SHIM", "1")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/foreign/cuda/lib")
    launch = _launcher()
    first = launch._rank_environment("jittor", 0, "0", tmp_path, "")
    second = launch._rank_environment("jittor", 1, "1", tmp_path, "")
    oracle = launch._rank_environment("torch", 0, "0", tmp_path, "")
    for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR"):
        assert first[name] != second[name] and first[name] != oracle[name]
        assert Path(first[name]).is_dir()
    assert first["JITTOR_HOME"] != second["JITTOR_HOME"]
    assert "JITTOR_HOME" not in oracle and "JITTOR_TORCH_SHIM" not in oracle
    assert "LD_LIBRARY_PATH" not in oracle


def test_same_case_has_shared_control_plane_and_private_rank_caches(tmp_path, monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "foreign-launcher")
    monkeypatch.setenv("MASTER_PORT", "29500")
    first, second = _launcher().pair_environments(
        "jittor", ("0", "1"), tmp_path / "caches", "", tmp_path / "case")
    assert first["JITTOR_DIST_RENDEZVOUS_DIR"] == second["JITTOR_DIST_RENDEZVOUS_DIR"]
    assert Path(first["JITTOR_DIST_RENDEZVOUS_DIR"]).is_dir()
    for name in ("HOME", "JITTOR_HOME", "TMPDIR"):
        assert first[name] != second[name]
    assert all(name not in environment for environment in (first, second)
               for name in ("MASTER_ADDR", "MASTER_PORT"))


def test_file_rendezvous_is_per_runtime_and_rejects_stale_stores(tmp_path):
    launch = _launcher()
    native = tmp_path / "jittor"
    oracle = tmp_path / "torch"
    native.mkdir()
    oracle.mkdir()
    native_url = launch.file_rendezvous(native)
    oracle_url = launch.file_rendezvous(oracle)
    assert native_url != oracle_url
    assert launch.file_rendezvous(native) == native_url
    parsed = urlparse(native_url)
    assert parsed.scheme == "file" and not parsed.netloc
    store = Path(parsed.path)
    assert store.parent == native and not store.exists()
    store.write_bytes(b"old process-group state")
    with pytest.raises(FileExistsError, match="already exists"):
        launch.file_rendezvous(native)
    assert store.read_bytes() == b"old process-group state"


@pytest.mark.parametrize("failure", ("rank", "timeout"))
def test_failed_or_timed_out_rank_terminates_its_live_peer(tmp_path, failure):
    paths = [tmp_path / ("rank%d.log" % i) for i in range(2)]
    handles = [path.open("w") for path in paths]
    programs = ["import time; time.sleep(60)",
                "raise SystemExit(7)" if failure == "rank" else "import time; time.sleep(60)"]
    processes = []
    try:
        for program, handle in zip(programs, handles):
            processes.append(subprocess.Popen(
                [PYTHON, "-c", program], env=child_env(without_torch_mode=True),
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        expected = RuntimeError if failure == "rank" else TimeoutError
        with pytest.raises(expected):
            _launcher().wait_for_ranks(processes, paths, 10 if failure == "rank" else .05)
        assert all(process.poll() is not None for process in processes)
        if failure == "rank":
            assert processes[1].returncode == 7
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        for handle in handles:
            handle.close()


def _runner_definition(name):
    path = Path(__file__).resolve().parents[2] / "compat/tests/torch/_ecosystem_deepspeed_runner.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function.name]


def test_full_gradient_dtype_preserves_the_upstream_stage1_shortcut_and_stage2_partition():
    dtype = _runner_definition("_full_gradient_dtype")
    torch = SimpleNamespace(float32="float32")
    stage1 = SimpleNamespace(grad=SimpleNamespace(dtype="bfloat16"))
    stage2 = SimpleNamespace(grad=None)
    assert dtype(torch, stage1) == "bfloat16"
    assert dtype(torch, stage2) == "float32"


@pytest.mark.parametrize("tamper", ("gradient", "moment", "missing", "dtype"))
def test_strict_resume_rejects_hidden_gradient_or_optimizer_divergence(tamper):
    arrays = {}
    for before, after, field in (("saved_optimizer", "restored_optimizer", "exp_avg"),
                                 ("saved_master_partition", "restored_master_partition", "0/0"),
                                 ("final_optimizer", "resumed_optimizer", "exp_avg"),
                                 ("final_master_partition", "resumed_master_partition", "0/0"),
                                 ("step4", "resumed4", "input_grad/inputs"),
                                 ("step5", "resumed5", "incoming_parameter_grad/weight")):
        value = np.array([2.] if before.startswith("final") else [1.], dtype="float32")
        arrays[before + "/" + field] = value
        arrays[after + "/" + field] = value.copy()
    strict = _runner_definition("_assert_strict_resume")
    dtypes = {key: "bfloat16" for key in arrays if "_grad/" in key}
    assert strict(arrays, dtypes)[1] == 1
    if tamper == "gradient":
        arrays["resumed5/incoming_parameter_grad/weight"][0] += .01
    elif tamper == "moment":
        arrays["resumed_optimizer/exp_avg"][0] += .01
    elif tamper == "missing":
        arrays.pop("resumed4/input_grad/inputs")
    else:
        dtypes["resumed5/incoming_parameter_grad/weight"] = "float32"
    with pytest.raises(AssertionError):
        strict(arrays, dtypes)

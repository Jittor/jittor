"""DeepSpeed L0 against an unmodified, independent binary PyTorch process."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from _helpers.child_process import PYTHON, REPO_ROOT, child_env, default_timeout


def _required(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _dependency(package, reason):
    if importlib.util.find_spec(package) is None:
        if _required("JITTOR_REQUIRE_DEEPSPEED"):
            pytest.fail(reason)
        pytest.skip(reason)


def test_deepspeed_l0_against_unmodified_binary_torch(tmp_path):
    _dependency("deepspeed", "deepspeed is not installed; L0 unverified")
    _dependency("jittor_adapters", "deepspeed adapter is not installed; L0 unverified")
    oracle = os.environ.get("REAL_TORCH_PYTHON", "").strip()
    if not oracle:
        reason = "REAL_TORCH_PYTHON is not configured; DeepSpeed L0 unverified"
        if _required("JITTOR_REQUIRE_REAL_TORCH") or _required("JITTOR_REQUIRE_DEEPSPEED"):
            pytest.fail(reason)
        pytest.skip(reason)
    device = os.environ.get("JITTOR_TEST_DEVICES", "cpu").strip().lower()
    if device not in ("cpu", "npu"):
        pytest.skip("DeepSpeed L0 backend %s is unverified; scope is CPU/NPU" % device)
    output = Path(os.environ.get("JITTOR_DEEPSPEED_L0_OUT", str(tmp_path))) / device
    output.mkdir(parents=True, exist_ok=True)
    probe = REPO_ROOT / "agent/skills/deepspeed-torch-compat/scripts/l0_probe.py"
    reports = {}
    for runtime, python in (("oracle", oracle), ("shim", PYTHON)):
        side = output / runtime
        side.mkdir(exist_ok=True)
        environment = child_env(without_torch_mode=runtime == "oracle", repo_paths=runtime != "oracle")
        # Never reuse an earlier patched DeepSpeed source or provider experiment.
        for name in ("DS_EXPERIMENTAL_SOURCE", "DS_ACCELERATOR_PROVIDER"):
            environment.pop(name, None)
        for name in tuple(environment):
            if name.startswith(("JT_HCCL_", "JT_NCCL_")):
                environment.pop(name)
        environment.update(JT_BACKEND="acl" if device == "npu" else "cpu",
                           JITTOR_TORCH_SHIM="1" if runtime == "shim" else "0",
                           JT_BACKEND_FALLBACK="error", HF_HUB_OFFLINE="1",
                           TRANSFORMERS_OFFLINE="1")
        for key, leaf in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache")):
            (side / leaf).mkdir(exist_ok=True)
            environment[key] = str(side / leaf)
        if runtime == "oracle":
            (side / "jittor-home").mkdir(exist_ok=True)
            environment["JITTOR_HOME"] = str(side / "jittor-home")
            environment["DS_ACCELERATOR"] = device
        elif device == "npu":
            environment.pop("DS_ACCELERATOR", None)
            environment.update(JT_HCCL_WORLD_SIZE="1", JT_HCCL_RANK="0",
                               JT_HCCL_LOCAL_RANK="0",
                               JT_HCCL_ROOTINFO_FILE=str(side / "hccl-world.bin"))
        target = side / "result.json"
        command = [python, str(probe), "run", "--runtime", runtime,
                   "--device", device, "--out", str(target)]
        result = subprocess.run(command, env=environment, cwd=str(side), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=default_timeout())
        (side / "run.log").write_text(result.stdout, encoding="utf-8")
        assert result.returncode == 0, runtime + "\n" + result.stdout[-16000:]
        reports[runtime] = json.loads(target.read_text(encoding="utf-8"))
    expected, actual = reports["oracle"], reports["shim"]
    assert expected["runtime"] == "oracle" and actual["runtime"] == "shim"
    assert expected["torch_origin"], "Binary oracle must report its module origin"
    for field in ("device", "deepspeed_version", "dependencies", "installed_engine_sha256", "snapshots", "invalid_config_errors"):
        assert actual[field] == expected[field], field
    assert actual["fallback_delta"] == 0
    assert actual["rejected"], "Out-of-scope calls must be tested and rejected"
    (output / "comparison.json").write_text(json.dumps({
        "status": "passed", "device": device, "cases": 2,
        "parameter_count": 4, "buffer_count": 3,
        "scope": "import/config/model" + (", plus single-NPU FP32 Stage0 Engine" if device == "npu" else ""),
        "fallback_delta": 0}, indent=2), encoding="utf-8")

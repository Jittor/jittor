"""Host-only negative controls; never import a real accelerator runtime."""
import os
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest


BOOTSTRAP = Path(__file__).resolve().parents[1] / "runtime/oracle/sitecustomize.py"
CANDIDATE_BOOTSTRAP = BOOTSTRAP.parent.parent / "sitecustomize.py"


def _fault_observer_fixture(tmp_path, body="return True"):
    directory = (tmp_path / "observer").resolve()
    directory.mkdir()
    source = directory / "verl_agent_fault_diagnostic.py"
    source.write_text("print('FAULT_MODULE_IMPORTED', flush=True)\n"
                      "def install():\n"
                      "    print('FAULT_INSTALL_CALLED', flush=True)\n"
                      "    " + body + "\n")
    return directory, hashlib.sha256(source.read_bytes()).hexdigest()


def _fault_observer_child(module_directory, digest, mode="1", prelude="", search_directory=None,
                          startup="1", context="0", **variables):
    """Run the actual candidate bootstrap under Python site's exception policy.

    -S plus an explicit source-tree bootstrap path prevents loading either real
    framework. The observer is only a task-local source fixture, never a model.
    """
    program = (
        "import os, sys, types, importlib.util, importlib.machinery\n"
        "torch = types.ModuleType('torch')\n"
        "torch.__torch_version__ = 'test-stub'\n"
        "def reporter(enabled):\n"
        "    print('API_REPORTER_COMPLETED', flush=True)\n"
        "    return 'test-stub'\n"
        "torch.compat_report_torch_api_version = reporter\n"
        "torch.cuda = types.SimpleNamespace(is_available=lambda: True, "
        "set_device=lambda index: print('CUDA_CONTEXT_BOUND', flush=True))\n"
        "sys.modules['torch'] = torch\n"
        + "sys.path.insert(0, " + repr(str(search_directory or module_directory)) + ")\n"
        + "sys.path.insert(0, " + repr(str(CANDIDATE_BOOTSTRAP.parent)) + ")\n"
        + prelude + "\n"
        "import site\nsite.execsitecustomize()\nprint('WORKER_PAYLOAD_EXECUTED', flush=True)\n"
    )
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("JITTOR_", "VERL_"))}
    environment.update(JITTOR_TORCH_SHIM="1", JITTOR_VERL_HF_ASYNC_ROLLOUT=startup,
                       VERL_AGENT_FAULT_MODULE_DIR=str(module_directory),
                       VERL_AGENT_FAULT_MODULE_SHA256=digest, JITTOR_VERL_RAY_CUDA_CONTEXT=context,
                       JITTOR_VERL_RAY_NPU_CONTEXT="0", CUDA_VISIBLE_DEVICES="0")
    if mode is not None:
        environment["VERL_AGENT_FAULT_DIAGNOSTIC"] = mode
    environment.update(variables)
    return subprocess.run([sys.executable, "-B", "-S", "-c", program], env=environment,
                          text=True, capture_output=True, timeout=15)


@pytest.mark.parametrize("mode", ("0", None))
def test_fault_observer_default_off_never_imports_module_or_checks_path(tmp_path, mode):
    prelude = ("class RejectFaultImport:\n"
               "    def find_spec(self, fullname, path=None, target=None):\n"
               "        if fullname == 'verl_agent_fault_diagnostic':\n"
               "            raise RuntimeError('disabled observer import attempted')\n"
               "sys.meta_path.insert(0, RejectFaultImport())\n")
    result = _fault_observer_child(tmp_path / "missing", "invalid", mode=mode, prelude=prelude)
    assert result.returncode == 0, result.stderr
    assert "WORKER_PAYLOAD_EXECUTED" in result.stdout
    assert "FAULT_MODULE_IMPORTED" not in result.stdout


@pytest.mark.parametrize("mode", ("", "2", "true", "yes", "-1"))
def test_fault_observer_invalid_opt_in_blocks_worker_payload(tmp_path, mode):
    result = _fault_observer_child(tmp_path, "0" * 64, mode=mode, startup="0")
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "absent, 0 or 1" in result.stderr


@pytest.mark.parametrize("kind", ("empty", "relative", "missing", "symlink", "not_directory"))
def test_fault_observer_invalid_module_directory_blocks_payload(tmp_path, kind):
    directory, digest = _fault_observer_fixture(tmp_path)
    if kind == "empty":
        directory = ""
    elif kind == "relative":
        directory = "relative-observer"
    elif kind == "missing":
        directory = tmp_path / "missing"
    elif kind == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        directory = alias
    else:
        directory = directory / "verl_agent_fault_diagnostic.py"
    result = _fault_observer_child(directory, digest)
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "FAULT_MODULE_IMPORTED" not in result.stdout


@pytest.mark.parametrize("digest", ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "0" * 64))
def test_fault_observer_invalid_or_wrong_digest_blocks_import_and_payload(tmp_path, digest):
    directory, _ = _fault_observer_fixture(tmp_path)
    result = _fault_observer_child(directory, digest)
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "FAULT_MODULE_IMPORTED" not in result.stdout


def test_fault_observer_shadowed_module_origin_blocks_import(tmp_path):
    directory, digest = _fault_observer_fixture(tmp_path)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "verl_agent_fault_diagnostic.py").write_text("print('SHADOW_EXECUTED')\n")
    result = _fault_observer_child(directory, digest, search_directory=shadow)
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "SHADOW_EXECUTED" not in result.stdout and "FAULT_MODULE_IMPORTED" not in result.stdout


@pytest.mark.parametrize("stage", ("missing_module", "missing_origin", "source_symlink"))
def test_fault_observer_missing_or_noncanonical_source_origin_blocks_payload(tmp_path, stage):
    directory, digest = _fault_observer_fixture(tmp_path)
    source = directory / "verl_agent_fault_diagnostic.py"
    prelude = ""
    if stage == "missing_module":
        source.unlink()
    elif stage == "source_symlink":
        destination = tmp_path / "relocated.py"
        source.rename(destination)
        source.symlink_to(destination)
    else:
        prelude = ("class NoOriginFinder:\n"
                   "    def find_spec(self, fullname, path=None, target=None):\n"
                   "        if fullname == 'verl_agent_fault_diagnostic':\n"
                   "            return importlib.machinery.ModuleSpec(fullname, loader=None, origin=None)\n"
                   "sys.meta_path.insert(0, NoOriginFinder())\n")
    result = _fault_observer_child(directory, digest, prelude=prelude)
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "FAULT_MODULE_IMPORTED" not in result.stdout


@pytest.mark.parametrize("body", ("return True\ndel install", "return True\ninstall = None"))
def test_fault_observer_missing_or_noncallable_installer_blocks_payload(tmp_path, body):
    directory, digest = _fault_observer_fixture(tmp_path, body)
    result = _fault_observer_child(directory, digest)
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "FAULT_MODULE_IMPORTED" in result.stdout and "FAULT_INSTALL_CALLED" not in result.stdout


@pytest.mark.parametrize("shim", ("0", ""))
def test_fault_observer_requested_without_candidate_bootstrap_blocks_payload(tmp_path, shim):
    directory, digest = _fault_observer_fixture(tmp_path)
    result = _fault_observer_child(directory, digest, JITTOR_TORCH_SHIM=shim)
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "API_REPORTER_COMPLETED" not in result.stdout
    assert "FAULT_MODULE_IMPORTED" not in result.stdout


def test_fault_observer_cannot_register_before_failed_reporter(tmp_path):
    directory, digest = _fault_observer_fixture(tmp_path)
    result = _fault_observer_child(directory, digest, prelude="torch.compat_report_torch_api_version = lambda enabled: 'wrong-version'\n")
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "FAULT_MODULE_IMPORTED" not in result.stdout


@pytest.mark.parametrize("body", ("return False", "return None", "return 1", "raise RuntimeError('registration failed')",
                                  "raise ValueError('registration value failed')", "raise OSError('registration filesystem failed')",
                                  "raise ModuleNotFoundError('registration dependency missing')"))
def test_fault_observer_noop_or_registration_error_cannot_execute_payload(tmp_path, body):
    directory, digest = _fault_observer_fixture(tmp_path, body)
    result = _fault_observer_child(directory, digest)
    assert result.returncode != 0, "Python continued after invalid observer registration"
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "FAULT_INSTALL_CALLED" in result.stdout


def test_fault_observer_missing_source_file_cannot_be_swallowed_by_python_site(tmp_path):
    directory = (tmp_path / "observer").resolve()
    directory.mkdir()
    source = directory / "verl_agent_fault_diagnostic.py"
    prelude = (
        "class MissingSourceFinder:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'verl_agent_fault_diagnostic':\n"
        + "            return importlib.machinery.ModuleSpec(fullname, loader=None, origin=" + repr(str(source)) + ")\n"
        "sys.meta_path.insert(0, MissingSourceFinder())\n"
    )
    result = _fault_observer_child(directory, "0" * 64, prelude=prelude)
    assert result.returncode != 0, "sitecustomize swallowed unavailable observer source"
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout


@pytest.mark.parametrize("context,startup", (("0", "1"), ("1", "1"), ("0", "0")))
def test_fault_observer_valid_real_source_hash_registers_after_api_reporter_and_context(tmp_path, context, startup):
    directory, digest = _fault_observer_fixture(tmp_path)
    result = _fault_observer_child(directory, digest, context=context, startup=startup)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines.index("API_REPORTER_COMPLETED") < lines.index("FAULT_MODULE_IMPORTED")
    assert lines.index("FAULT_MODULE_IMPORTED") < lines.index("FAULT_INSTALL_CALLED") < lines.index("WORKER_PAYLOAD_EXECUTED")
    if context == "1":
        assert lines.index("CUDA_CONTEXT_BOUND") < lines.index("FAULT_MODULE_IMPORTED")


def test_fault_observer_preloaded_same_origin_module_does_not_bypass_source_validation(tmp_path):
    directory, digest = _fault_observer_fixture(tmp_path)
    prelude = (
        "stale = types.ModuleType('verl_agent_fault_diagnostic')\n"
        + "stale.__spec__ = importlib.machinery.ModuleSpec(stale.__name__, loader=None, origin="
        + repr(str(directory / "verl_agent_fault_diagnostic.py")) + ")\n"
        "stale.install = lambda: print('STALE_MODULE_INSTALLED', flush=True) or True\n"
        "sys.modules[stale.__name__] = stale\n"
    )
    result = _fault_observer_child(directory, digest, prelude=prelude)
    assert result.returncode != 0, "source SHA cannot approve a different preloaded module object"
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "STALE_MODULE_INSTALLED" not in result.stdout


def _precision_child(framework, stage="ok", profile="1", device="cuda", dtype="fp32", startup="1"):
    bootstrap = CANDIDATE_BOOTSTRAP if framework == "jittor" else BOOTSTRAP
    program = """
import os, sys, types
class Flags:
    def __init__(self):
        object.__setattr__(self, 'allow_tf32', True)
        object.__setattr__(self, 'benchmark', True)
    def __setattr__(self, name, value):
        if os.environ['FAIL_STAGE'] == 'noop_flags':
            return
        object.__setattr__(self, name, value)
torch = types.ModuleType('torch')
torch._C = object()
torch.__torch_version__ = 'test-stub'
torch.compat_report_torch_api_version = lambda enabled: 'test-stub'
torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=Flags()), cudnn=Flags())
precision = 'high'
def set_precision(value):
    global precision
    if os.environ['FAIL_STAGE'] == 'setter_error':
        raise RuntimeError('precision setter failed')
    if os.environ['FAIL_STAGE'] == 'setter_value_error':
        raise ValueError('precision value failed')
    if os.environ['FAIL_STAGE'] != 'noop_precision':
        precision = value
def get_precision():
    if os.environ['FAIL_STAGE'] == 'getter_error':
        raise RuntimeError('precision getter failed')
    return precision
torch.set_float32_matmul_precision = set_precision
torch.get_float32_matmul_precision = get_precision
sys.modules['torch'] = torch
print('EXPECTED_PID=' + str(os.getpid()), flush=True)
"""
    program += "sys.path.insert(0, " + repr(str(bootstrap.parent)) + ")\n"
    program += "import site\nsite.execsitecustomize()\nprint('WORKER_PAYLOAD_EXECUTED')\n"
    program += "print('FINAL_POLICY=' + repr((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, precision)))\n"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("JITTOR_", "VERL_"))}
    environment.update(JITTOR_VERL_HF_ASYNC_ROLLOUT=startup, VERL_DEVICE=device,
                       VERL_TRAIN_DTYPE=dtype, VERL_CUDA_STRICT_FP32=profile, FAIL_STAGE=stage)
    if framework == "jittor":
        environment["JITTOR_TORCH_SHIM"] = "1"
    return subprocess.run([sys.executable, "-B", "-S", "-c", program], env=environment,
                          text=True, capture_output=True, timeout=15)


@pytest.mark.parametrize("framework", ("jittor", "torch"))
@pytest.mark.parametrize("stage", ("setter_error", "setter_value_error", "getter_error", "noop_flags", "noop_precision"))
def test_strict_fp32_failure_cannot_execute_payload(framework, stage):
    result = _precision_child(framework, stage)
    assert result.returncode != 0, "Python continued after an invalid precision policy"
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "VERL_CUDA_STRICT_FP32_READY " not in result.stdout


@pytest.mark.parametrize("framework", ("jittor", "torch"))
@pytest.mark.parametrize("startup", ("0", "1"))
def test_strict_fp32_each_process_records_actual_public_readback(framework, startup):
    result = _precision_child(framework, startup=startup)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    markers = [line.removeprefix("VERL_CUDA_STRICT_FP32_READY ") for line in lines
               if line.startswith("VERL_CUDA_STRICT_FP32_READY ")]
    assert len(markers) == 1
    record = json.loads(markers[0])
    assert record["pid"] == int(next(line.split("=", 1)[1] for line in lines if line.startswith("EXPECTED_PID=")))
    assert record["framework"] == framework
    assert record["runtime_conditions"] == {
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "cudnn_benchmark": False, "float32_matmul_precision": "highest"}
    assert record["scope"] == "public_precision_readback_only"
    assert "FINAL_POLICY=(False, False, False, 'highest')" in lines


@pytest.mark.parametrize("framework", ("jittor", "torch"))
def test_nonprofile_startup_does_not_change_precision(framework):
    result = _precision_child(framework, profile="0")
    assert result.returncode == 0, result.stderr
    assert "VERL_CUDA_STRICT_FP32_READY " not in result.stdout
    assert "FINAL_POLICY=(True, True, True, 'high')" in result.stdout


@pytest.mark.parametrize("framework", ("jittor", "torch"))
@pytest.mark.parametrize("profile,device,dtype", (("1", "npu", "fp32"), ("1", "cuda", "bf16"), ("2", "cuda", "fp32"), ("", "cuda", "fp32")))
def test_bootstrap_rejects_misconfigured_strict_fp32(framework, profile, device, dtype):
    result = _precision_child(framework, profile=profile, device=device, dtype=dtype)
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout


@pytest.mark.parametrize("failure", ("cuda", "npu", "reporter"))
def test_candidate_startup_failure_cannot_execute_worker_payload(failure):
    program = (
        "import os, sys, types\n"
        "def set_device(index):\n"
        "    raise RuntimeError('device binding failed')\n"
        "torch = types.ModuleType('torch')\n"
        "torch.__torch_version__ = 'test-stub'\n"
        "torch.compat_report_torch_api_version = lambda enabled: "
        "('wrong-version' if os.environ['FAIL_STAGE'] == 'reporter' else 'test-stub')\n"
        "torch.cuda = types.SimpleNamespace(is_available=lambda: True, set_device=set_device)\n"
        "torch.npu = types.SimpleNamespace(set_device=set_device)\n"
        "sys.modules['torch'] = torch\n"
        + "sys.path.insert(0, " + repr(str(CANDIDATE_BOOTSTRAP.parent)) + ")\n"
        "import site\nsite.execsitecustomize()\nprint('WORKER_PAYLOAD_EXECUTED')\n"
    )
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("JITTOR_")}
    environment.update(JITTOR_TORCH_SHIM="1", JITTOR_VERL_HF_ASYNC_ROLLOUT="1",
                       CUDA_VISIBLE_DEVICES="0", FAIL_STAGE=failure)
    if failure in ("cuda", "npu"):
        environment["JITTOR_VERL_RAY_" + failure.upper() + "_CONTEXT"] = "1"
    result = subprocess.run([sys.executable, "-B", "-S", "-c", program], env=environment,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode != 0, "Python swallowed candidate startup failure"
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert ("inconsistent value" if failure == "reporter" else "device binding failed") in result.stderr


def test_candidate_rejects_conflicting_contexts_before_binding_or_payload():
    program = (
        "import sys, types\n"
        "torch = types.ModuleType('torch')\n"
        "torch.__torch_version__ = 'test-stub'\n"
        "torch.compat_report_torch_api_version = lambda enabled: 'test-stub'\n"
        "torch.cuda = types.SimpleNamespace(is_available=lambda: True, "
        "set_device=lambda index: print('CUDA_BOUND'))\n"
        "torch.npu = types.SimpleNamespace(set_device=lambda index: print('NPU_BOUND'))\n"
        "sys.modules['torch'] = torch\n"
        + "sys.path.insert(0, " + repr(str(CANDIDATE_BOOTSTRAP.parent)) + ")\n"
        "import site\nsite.execsitecustomize()\nprint('WORKER_PAYLOAD_EXECUTED')\n"
    )
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("JITTOR_")}
    environment.update(JITTOR_TORCH_SHIM="1", JITTOR_VERL_HF_ASYNC_ROLLOUT="1",
                       CUDA_VISIBLE_DEVICES="0", JITTOR_VERL_RAY_NPU_CONTEXT="1",
                       JITTOR_VERL_RAY_CUDA_CONTEXT="1")
    result = subprocess.run([sys.executable, "-B", "-S", "-c", program], env=environment,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert "conflicting" in result.stderr.lower()
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "NPU_BOUND" not in result.stdout and "CUDA_BOUND" not in result.stdout


@pytest.mark.parametrize("device", ("cuda", "npu", "driver"))
def test_candidate_single_context_binds_only_selected_backend(device):
    expected = [] if device == "driver" else [device]
    program = (
        "import sys, types\ncalls = []\n"
        "torch = types.ModuleType('torch')\n"
        "torch.__torch_version__ = 'test-stub'\n"
        "torch.compat_report_torch_api_version = lambda enabled: 'test-stub'\n"
        "torch.cuda = types.SimpleNamespace(is_available=lambda: True, "
        "set_device=lambda index: calls.append('cuda'))\n"
        "torch.npu = types.SimpleNamespace(set_device=lambda index: calls.append('npu'))\n"
        "sys.modules['torch'] = torch\n"
        + "sys.path.insert(0, " + repr(str(CANDIDATE_BOOTSTRAP.parent)) + ")\n"
        "import site\nsite.execsitecustomize()\n"
        + "assert calls == " + repr(expected) + ", calls\nprint('CONTEXT_OK')\n"
    )
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("JITTOR_")}
    environment.update(JITTOR_TORCH_SHIM="1", JITTOR_VERL_HF_ASYNC_ROLLOUT="1",
                       CUDA_VISIBLE_DEVICES="0", JITTOR_VERL_RAY_NPU_CONTEXT="0",
                       JITTOR_VERL_RAY_CUDA_CONTEXT="0")
    if device != "driver":
        environment["JITTOR_VERL_RAY_" + device.upper() + "_CONTEXT"] = "1"
    result = subprocess.run([sys.executable, "-B", "-S", "-c", program], env=environment,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "CONTEXT_OK" in result.stdout


def _child(prelude="", suffix="", **variables):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("JITTOR_")}
    environment["JITTOR_VERL_HF_ASYNC_ROLLOUT"] = "1"
    environment.update(variables)
    program = ("import runpy, sys, types\n" + prelude + "\n"
               + "bootstrap = runpy.run_path(" + repr(str(BOOTSTRAP)) + ")\n"
               + suffix + "\nprint('BOOTSTRAP_COMPLETED')\n")
    return subprocess.run([sys.executable, "-B", "-S", "-c", program],
                          env=environment, text=True, capture_output=True, timeout=15)


def test_rejects_jittor_already_loaded_before_sitecustomize():
    result = _child("sys.modules['jittor'] = types.ModuleType('jittor')")
    assert result.returncode != 0
    assert "BOOTSTRAP_COMPLETED" not in result.stdout
    assert "already loaded" in result.stderr


def test_rejects_inherited_candidate_shim_switch():
    result = _child(JITTOR_TORCH_SHIM="1")
    assert result.returncode != 0
    assert "JITTOR_TORCH_SHIM" in result.stderr


def test_blocker_is_installed_before_rollout_torch_context():
    result = _child(
        "def set_device(index):\n"
        "    assert index == 0\n"
        "    assert any(type(f).__name__ == '_BlockJittorFinder' for f in sys.meta_path)\n"
        "torch = types.ModuleType('torch')\n"
        "torch._C = object()\n"
        "torch.npu = types.SimpleNamespace(set_device=set_device)\n"
        "sys.modules['torch'] = torch\n",
        JITTOR_VERL_RAY_NPU_CONTEXT="1", ASCEND_RT_VISIBLE_DEVICES="2",
    )
    assert result.returncode == 0, result.stderr


def test_fresh_jittor_import_is_rejected():
    result = _child(suffix="import jittor")
    assert result.returncode != 0
    assert "refuses to import" in result.stderr


def test_cuda_rollout_context_is_bound_after_oracle_guards():
    result = _child(
        "def set_device(index):\n"
        "    assert index == 0\n"
        "    assert any(type(f).__name__ == '_BlockJittorFinder' for f in sys.meta_path)\n"
        "torch = types.ModuleType('torch')\n"
        "torch._C = object()\n"
        "torch.cuda = types.SimpleNamespace(is_available=lambda: True, set_device=set_device)\n"
        "sys.modules['torch'] = torch\n",
        JITTOR_VERL_RAY_CUDA_CONTEXT="1", CUDA_VISIBLE_DEVICES="1",
    )
    assert result.returncode == 0, result.stderr


def test_cuda_rollout_refuses_multiple_devices_before_import():
    result = _child(JITTOR_VERL_RAY_CUDA_CONTEXT="1", CUDA_VISIBLE_DEVICES="0,1")
    assert result.returncode != 0
    assert "exactly one pre-bound CUDA device" in result.stderr


def test_cuda_rollout_unavailable_aborts_instead_of_falling_back():
    result = _child(
        "torch = types.ModuleType('torch')\n"
        "torch._C = object()\n"
        "torch.cuda = types.SimpleNamespace(is_available=lambda: False)\n"
        "sys.modules['torch'] = torch\n",
        JITTOR_VERL_RAY_CUDA_CONTEXT="1", CUDA_VISIBLE_DEVICES="0",
    )
    assert result.returncode != 0
    assert "CUDA unavailable" in result.stderr


def test_context_failure_cannot_be_swallowed_by_python_site():
    # Python normally logs Exception from sitecustomize and continues. This
    # bootstrap must turn a known initialization failure into SystemExit.
    program = (
        "import sys, types\n"
        "def set_device(index):\n"
        "    raise RuntimeError('context unavailable')\n"
        "torch = types.ModuleType('torch')\n"
        "torch._C = object()\n"
        "torch.npu = types.SimpleNamespace(set_device=set_device)\n"
        "sys.modules['torch'] = torch\n"
        + "sys.path.insert(0, " + repr(str(BOOTSTRAP.parent)) + ")\n"
        "import site\nsite.execsitecustomize()\nprint('WORKER_PAYLOAD_EXECUTED')\n"
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("JITTOR_")}
    environment.update(JITTOR_VERL_HF_ASYNC_ROLLOUT="1", JITTOR_VERL_RAY_NPU_CONTEXT="1",
                       ASCEND_RT_VISIBLE_DEVICES="2")
    result = subprocess.run([sys.executable, "-B", "-S", "-c", program], env=environment,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode != 0
    assert "WORKER_PAYLOAD_EXECUTED" not in result.stdout
    assert "context unavailable" in result.stderr


def test_rejects_shim_even_when_it_exposes_a_c_extension_stub():
    result = _child(
        "torch = types.ModuleType('torch')\n"
        "torch._C = object()\n"
        "torch._torch_compat_install_context = object()\n"
        "sys.modules['torch'] = torch\n",
        suffix="bootstrap['require_official_torch'](sys.modules['torch'])",
    )
    assert result.returncode != 0
    assert "resolved a shim" in result.stderr


def test_bootstrap_disabled_does_not_patch_import_machinery():
    result = _child(
        "import importlib.util\noriginal = importlib.util.find_spec",
        suffix="assert importlib.util.find_spec is original",
        JITTOR_VERL_HF_ASYNC_ROLLOUT="0",
    )
    assert result.returncode == 0, result.stderr


def test_wrapper_drops_inherited_framework_paths_and_switches(tmp_path):
    # Stub only the binary-framework identity surface. This is a control-plane
    # unit test, NOT evidence that official Torch or any NPU operation ran.
    (tmp_path / "torch.py").write_text("_C = object()\n__version__ = 'test-stub'\n")
    root = BOOTSTRAP.parents[5]
    wrapper = BOOTSTRAP.parent.parent / "oracle_worker_python.sh"
    environment = os.environ.copy()
    environment.update(
        JITTOR_SOURCE_ROOT=str(root), VERL_SOURCE_ROOT=str(tmp_path),
        VERL_FULL_PPO_ROOT=str(tmp_path), VERL_ORACLE_PYTHON=sys.executable,
        PYTHONPATH="/untrusted/shared/site-packages", JITTOR_TORCH_SHIM="1",
        JITTOR_VERL_NATIVE_NPU_KERNELS="1",
    )
    result = subprocess.run(
        ["bash", str(wrapper), "-c",
         "import os,sys; assert '/untrusted/shared/site-packages' not in sys.path; "
         "assert os.getenv('JITTOR_TORCH_SHIM') is None; "
         "assert os.getenv('JITTOR_VERL_NATIVE_NPU_KERNELS') is None; "
         "assert sys.flags.no_user_site; print('CONTROL_PLANE_OK')"],
        env=environment, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "CONTROL_PLANE_OK" in result.stdout


def test_oracle_rejects_dual_context_markers_before_torch_import():
    result = _child(
        JITTOR_VERL_RAY_NPU_CONTEXT="1", JITTOR_VERL_RAY_CUDA_CONTEXT="1",
        ASCEND_RT_VISIBLE_DEVICES="2", CUDA_VISIBLE_DEVICES="5",
    )
    assert result.returncode != 0
    assert "simultaneous NPU and CUDA" in result.stderr
    assert "BOOTSTRAP_COMPLETED" not in result.stdout


def _valuehead_source():
    # Reuse the exact complete upstream fixture, without importing pytest or
    # any framework in the child interpreter started with -S.
    import ast

    root = BOOTSTRAP.parents[5]
    fixture = root / "adapters/jittor_adapters/verl/tests/test_valuehead_attention.py"
    tree = ast.parse(fixture.read_text())
    return next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "SOURCE" for target in node.targets))


def _valuehead_prelude(filename, *, preload):
    adapters = BOOTSTRAP.parents[5] / "adapters"
    program = (
        "import importlib, importlib.util\n"
        + "sys.path.insert(0, " + repr(str(adapters)) + ")\n"
        "root = types.ModuleType('verl')\nroot.__version__ = '0.9.1'\nroot.__path__ = []\n"
        "sys.modules['verl'] = root\n"
        "utils = types.ModuleType('verl.utils')\n"
        + "utils.__path__ = [" + repr(str(filename.parent)) + "]\n"
        "sys.modules['verl.utils'] = utils\n"
        "transformers = types.ModuleType('transformers')\n"
        "def load(**kwargs):\n"
        "    assert kwargs['attn_implementation'] == 'eager'\n"
        "    return 'HOST_VALUEHEAD_RESULT'\n"
        "transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=load)\n"
        "transformers.AutoModelForTokenClassification = types.SimpleNamespace(from_pretrained=load)\n"
        "sys.modules['transformers'] = transformers\n"
    )
    if preload:
        program += (
            "spec = importlib.util.spec_from_file_location('verl.utils.model', " + repr(str(filename)) + ")\n"
            "module = importlib.util.module_from_spec(spec)\nsys.modules[spec.name] = module\n"
            "spec.loader.exec_module(module)\n"
            "alias = types.ModuleType('verl.workers.valuehead_alias')\n"
            "alias.load_valuehead_model = module.load_valuehead_model\n"
            "sys.modules[alias.__name__] = alias\n"
        )
    return program


@pytest.mark.parametrize("preload", (False, True))
def test_oracle_valuehead_uses_shared_pinned_policy_without_jittor(tmp_path, preload):
    filename = tmp_path / "model.py"
    filename.write_text(_valuehead_source())
    suffix = (
        "module = importlib.import_module('verl.utils.model')\n"
        "assert module.load_valuehead_model('offline-model', 'fp32', "
        "types.SimpleNamespace(_attn_implementation='eager'), False) == 'HOST_VALUEHEAD_RESULT'\n"
        "assert module.load_valuehead_model._jittor_verl_valuehead_attention\n"
        "bootstrap['_register_valuehead_attention'](module)\n"
        "assert not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules)\n"
        "assert 'jittor.compat.module_patcher' not in sys.modules\n"
    )
    if preload:
        suffix += "assert alias.load_valuehead_model is module.load_valuehead_model\n"
    result = _child(_valuehead_prelude(filename, preload=preload), suffix,
                    VERL_DEVICE="cuda", JITTOR_VERL_HF_ASYNC_ROLLOUT="1")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("device,flag", (("", "1"), ("cpu", "1"), ("npu", "1"), ("cuda", "0")))
def test_oracle_valuehead_default_and_other_devices_leave_original(tmp_path, device, flag):
    filename = tmp_path / "model.py"
    filename.write_text(_valuehead_source())
    result = _child(
        _valuehead_prelude(filename, preload=True),
        "assert not hasattr(module.load_valuehead_model, '_jittor_verl_valuehead_attention')\n"
        "assert alias.load_valuehead_model is module.load_valuehead_model\n"
        "assert not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules)",
        VERL_DEVICE=device, JITTOR_VERL_HF_ASYNC_ROLLOUT=flag,
    )
    assert result.returncode == 0, result.stderr


def test_oracle_valuehead_changed_source_is_rejected(tmp_path):
    filename = tmp_path / "model.py"
    filename.write_text(_valuehead_source().replace('attn_implementation="flash_attention_2"',
                                                   'attn_implementation="eager"', 1))
    result = _child(_valuehead_prelude(filename, preload=True),
                    VERL_DEVICE="cuda", JITTOR_VERL_HF_ASYNC_ROLLOUT="1")
    assert result.returncode != 0
    assert "source SHA256 changed" in result.stderr
    assert "BOOTSTRAP_COMPLETED" not in result.stdout

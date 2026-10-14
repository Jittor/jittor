"""The dedicated Engine session must fail closed and isolate its oracle."""
import pytest

from _helpers.nox_probe import _FakeSession, _load_noxfile


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_engine_session_isolates_oracle_and_installs_adapter_before_execution(
        monkeypatch, tmp_path, device):
    monkeypatch.setenv("REAL_TORCH_PYTHON", "/opt/oracle/bin/python")
    monkeypatch.setenv("JITTOR_CI_PYTHON", "/opt/native/bin/python")
    monkeypatch.setenv("JITTOR_TEST_DEVICES", device)
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("JT_BUILD_NVCC_PATH", "/opt/cuda/bin/nvcc")
    monkeypatch.setenv("JT_BUILD_MKL_INCLUDE_PATH", "/opt/onednn/include")
    monkeypatch.setenv("JT_BUILD_MKL_LIB_PATH", "/opt/onednn/lib/libdnnl.so")
    monkeypatch.setenv("PYTHONPATH", "/foreign/source")
    monkeypatch.setenv("JITTOR_TORCH_SHIM", "1")
    monkeypatch.setenv("JITTOR_SOURCE_ROOT", "/foreign/core")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/opt/native/bin/python3-config")
    session.python = False
    module["ecosystem_deepspeed"](session)
    oracle = [(args, kw) for args, kw in session.calls if args[0] == "/opt/oracle/bin/python"]
    assert len(oracle) == 1
    env = oracle[0][1]["env"]
    assert {env[name] for name in module["THREAD_POOL_ENV_NAMES"]} == {"6"}
    assert not env["PYTHONPATH"] and not env["JITTOR_SOURCE_ROOT"]
    assert env["JITTOR_TORCH_SHIM"] == "0"
    adapter = [i for i, (args, _) in enumerate(session.calls)
               if "install" in args and any(str(arg).endswith("/adapters") for arg in args)]
    executions = [(i, args, kw) for i, (args, kw) in enumerate(session.calls)
                  if "pytest" in args]
    assert len(adapter) == len(executions) == 1
    assert adapter[0] < executions[0][0]
    args, env = executions[0][1], executions[0][2]["env"]
    assert {env[name] for name in module["THREAD_POOL_ENV_NAMES"]} == {"6"}
    assert args[0] == "/opt/native/bin/python"
    assert "compat/tests/torch/test_ecosystem_deepspeed.py" in args
    assert env["JITTOR_REQUIRE_DEEPSPEED"] == env["JITTOR_REQUIRE_REAL_TORCH"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ("2" if device == "cuda" else "")
    assert env["JT_BUILD_MKL_INCLUDE_PATH"] == "/opt/onednn/include"
    assert env["JT_BUILD_MKL_LIB_PATH"] == "/opt/onednn/lib/libdnnl.so"
    assert env["JITTOR_HOME"].endswith("/" + device + "/jittor-home")


@pytest.mark.parametrize("missing", ("oracle", "device", "allocation", "nvcc"))
def test_engine_session_rejects_incomplete_hardware_configuration(monkeypatch, tmp_path, missing):
    monkeypatch.setenv("REAL_TORCH_PYTHON", "/opt/oracle/bin/python")
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("JT_BUILD_NVCC_PATH", "/opt/cuda/bin/nvcc")
    if missing == "oracle":
        monkeypatch.delenv("REAL_TORCH_PYTHON")
    elif missing == "device":
        monkeypatch.setenv("JITTOR_TEST_DEVICES", "cpu,cuda")
    elif missing == "allocation":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    else:
        monkeypatch.delenv("JT_BUILD_NVCC_PATH")
        monkeypatch.delenv("nvcc_path", raising=False)
        monkeypatch.setenv("PATH", "")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/opt/native/bin/python3-config")
    session.python = False
    with pytest.raises(AssertionError, match="DeepSpeed"):
        module["ecosystem_deepspeed"](session)
    assert not session.calls, "Invalid configuration must fail before any interpreter or installer runs"

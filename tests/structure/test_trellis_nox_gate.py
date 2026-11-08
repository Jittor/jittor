"""TRELLIS sessions require explicit devices and isolate the binary Torch oracle."""

import pytest

from _helpers.nox_probe import _FakeSession, _load_noxfile
from _helpers.pytest_policy import _accepted_skip_patterns


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_trellis_session_isolates_devices_and_oracle(monkeypatch, tmp_path, device):
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    monkeypatch.setenv("JITTOR_TRELLIS_CHECKOUT", str(checkout))
    monkeypatch.setenv("REAL_TORCH_PYTHON", "/opt/oracle/bin/python")
    monkeypatch.setenv("JITTOR_CI_PYTHON", "/opt/native/bin/python")
    monkeypatch.setenv("JITTOR_TEST_DEVICES", device)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("JT_BUILD_NVCC_PATH", "/opt/cuda/bin/nvcc")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("JITTOR_SOURCE_ROOT", "/foreign/source")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/opt/native/bin/python3-config")
    session.python = False
    module["ecosystem_trellis"](session)
    oracle = [(args, kw) for args, kw in session.calls if args[0] == "/opt/oracle/bin/python"]
    assert len(oracle) == 1
    oracle_env = oracle[0][1]["env"]
    assert not oracle_env["PYTHONPATH"] and not oracle_env["JITTOR_SOURCE_ROOT"]
    assert oracle_env["JITTOR_TORCH_SHIM"] == "0"
    executions = [(args, kw) for args, kw in session.calls if "pytest" in args]
    assert len(executions) == 1
    arguments, env = executions[0][0], executions[0][1]["env"]
    assert "compat/tests/torch/test_ecosystem_trellis.py" in arguments
    assert env["JITTOR_REQUIRE_TRELLIS"] == env["JITTOR_REQUIRE_REAL_TORCH"] == "1"
    assert env["JITTOR_TRELLIS_CHECKOUT"] == str(checkout)
    assert {env[name] for name in module["THREAD_POOL_ENV_NAMES"]} == {"3"}
    assert env["CUDA_VISIBLE_DEVICES"] == ("2" if device == "cuda" else "")
    assert env["JITTOR_HOME"].endswith("/" + device + "/jittor-home")
    assert env["JT_BUILD_NVCC_PATH"] == ("/opt/cuda/bin/nvcc" if device == "cuda" else "")


@pytest.mark.parametrize("missing", ("oracle", "checkout", "device", "allocation", "nvcc"))
def test_trellis_session_rejects_incomplete_configuration(monkeypatch, tmp_path, missing):
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    monkeypatch.setenv("JITTOR_TRELLIS_CHECKOUT", str(checkout))
    monkeypatch.setenv("REAL_TORCH_PYTHON", "/opt/oracle/bin/python")
    monkeypatch.setenv("JITTOR_TEST_DEVICES", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("JT_BUILD_NVCC_PATH", "/opt/cuda/bin/nvcc")
    if missing == "oracle":
        monkeypatch.delenv("REAL_TORCH_PYTHON")
    elif missing == "checkout":
        monkeypatch.setenv("JITTOR_TRELLIS_CHECKOUT", str(checkout / "absent"))
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
    with pytest.raises(AssertionError, match="TRELLIS"):
        module["ecosystem_trellis"](session)
    assert not session.calls


def test_required_trellis_cannot_be_an_environment_skip(monkeypatch):
    monkeypatch.delenv("JITTOR_REQUIRE_TRELLIS", raising=False)
    assert "trellis" in _accepted_skip_patterns()
    monkeypatch.setenv("JITTOR_REQUIRE_TRELLIS", "1")
    assert "trellis" not in _accepted_skip_patterns()

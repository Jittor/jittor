"""The canonical TRL session cannot shrink the maintained Trainer matrix."""
import pytest

from _helpers.nox_probe import _FakeSession, _load_noxfile


def configured(monkeypatch, tmp_path, device):
    oracle = tmp_path / "oracle/bin/python"
    oracle.parent.mkdir(parents=True)
    oracle.touch()
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    monkeypatch.setenv("REAL_TORCH_PYTHON", str(oracle))
    monkeypatch.setenv("JITTOR_CI_PYTHON", "/opt/candidate/bin/python")
    monkeypatch.setenv("JITTOR_TRL_CHECKOUT", str(upstream))
    monkeypatch.setenv("JITTOR_TEST_DEVICES", device)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("JT_BUILD_NVCC_PATH", "/opt/cuda/bin/nvcc")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/opt/candidate/bin/python3-config")
    session.python = False
    return module, session, oracle, upstream


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_trl_session_uses_complete_offline_protocol(monkeypatch, tmp_path, device):
    module, session, oracle, upstream = configured(monkeypatch, tmp_path, device)
    session.posargs = ("--output", str(tmp_path / "measurements"),
                       "--jittor-home", str(tmp_path / "cache"))
    module["ecosystem_trl"](session)
    args, kwargs = next((args, kwargs) for args, kwargs in session.calls
                        if any(str(value).endswith("tools/run_trl_gate.py") for value in args))
    assert args[0] == "/opt/candidate/bin/python"
    assert args[args.index("--oracle-python") + 1] == str(oracle)
    assert args[args.index("--jittor-python") + 1] == args[0]
    assert args[args.index("--upstream") + 1] == str(upstream)
    assert args[args.index("--steps") + 1] == "3"
    assert args[args.index("--device") + 1] == device
    assert args.count("--output") == args.count("--jittor-home") == 1
    if device == "cuda":
        assert args[args.index("--cuda-device") + 1] == "2"
        assert kwargs["env"]["JT_BUILD_NVCC_PATH"] == "/opt/cuda/bin/nvcc"
    else:
        assert "--cuda-device" not in args
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["JT_BUILD_NVCC_PATH"] == ""


@pytest.mark.parametrize("invalid", ("oracle", "checkout", "device", "allocation",
                                     "steps", "selector", "duplicate", "missing-value"))
def test_trl_invalid_configuration_fails_before_execution(monkeypatch, tmp_path, invalid):
    module, session, _, _ = configured(monkeypatch, tmp_path, "cuda")
    if invalid == "oracle":
        monkeypatch.delenv("REAL_TORCH_PYTHON")
    elif invalid == "checkout":
        monkeypatch.setenv("JITTOR_TRL_CHECKOUT", str(tmp_path / "missing"))
    elif invalid == "device":
        monkeypatch.setenv("JITTOR_TEST_DEVICES", "cpu,cuda")
    elif invalid == "allocation":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    else:
        session.posargs = {"steps": ("--steps", "1"), "selector": ("--algorithm", "sft"),
                           "duplicate": ("--output", "a", "--output", "b"),
                           "missing-value": ("--output",)}[invalid]
    with pytest.raises(AssertionError, match="TRL"):
        module["ecosystem_trl"](session)
    assert not session.calls

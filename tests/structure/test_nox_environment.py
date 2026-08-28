import runpy
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeOptions:
    pass


class _FakeSession:
    python = "3.11"

    def __init__(self, root, python_config):
        self.root = root
        self.python_config = python_config
        self.calls = []

    def create_tmp(self):
        return str(self.root / "session-tmp")

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.python_config + "\n"

    def error(self, message):
        raise AssertionError(message)


def _load_noxfile(monkeypatch, tmp_path):
    fake_nox = types.ModuleType("nox")
    fake_nox.options = _FakeOptions()
    fake_nox.session = lambda *args, **kwargs: lambda function: function
    monkeypatch.setitem(sys.modules, "nox", fake_nox)
    monkeypatch.setenv("JITTOR_LAB_ROOT", str(tmp_path / "lab"))
    return runpy.run_path(str(REPO_ROOT / "noxfile.py"), run_name="jittor_noxfile")


def test_session_env_uses_the_session_interpreters_python_config(monkeypatch, tmp_path):
    expected = "/opt/python-3.11/bin/python3.11-config"
    monkeypatch.setenv("python_config_path", "/opt/python-3.12/bin/python3.12-config")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, expected)

    _root, env = module["_session_env"](session, "structure")

    assert env["python_config_path"] == expected
    assert session.calls[0][0][:2] == ("python", "-c")


def test_session_env_clears_an_inherited_config_when_the_helper_is_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("python_config_path", "/opt/python-3.12/bin/python3.12-config")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "")

    _root, env = module["_session_env"](session, "py37")

    assert "python_config_path" not in env


def test_hardware_session_uses_the_external_interpreters_python_config(monkeypatch, tmp_path):
    expected = "/opt/hardware-python/bin/python3.10-config"
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, expected)
    env = {"python_config_path": "/opt/host-python/bin/python3.12-config"}

    module["_set_hardware_python_config"](session, "/opt/hardware-python/bin/python", env)

    assert env["python_config_path"] == expected
    args, kwargs = session.calls[0]
    assert args[:2] == ("/opt/hardware-python/bin/python", "-c")
    assert kwargs["external"] is True

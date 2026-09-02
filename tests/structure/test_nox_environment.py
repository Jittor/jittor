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


def test_sessions_share_one_jittor_cache(monkeypatch, tmp_path):
    """A session must not start from an empty Jittor cache.

    Every session used to get its own JITTOR_HOME under the scratch directory
    it wipes on entry, so each one rebuilt the C++ core and every operator and
    re-downloaded MKL, cub, cutt and NCCL first. Nothing about that was under
    test: the build is a function of the build configuration, which the cache
    path already partitions by.
    """
    monkeypatch.delenv("JITTOR_NOX_SHARED_CACHE", raising=False)
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")

    root, first = module["_session_env"](session, "structure")
    _root, second = module["_session_env"](session, "py312")

    assert first["JITTOR_HOME"] == second["JITTOR_HOME"]
    assert Path(first["JITTOR_HOME"]) == module["NOX_JITTOR_CACHE"]
    assert root not in Path(first["JITTOR_HOME"]).parents
    # The scratch directories stay under the session root; only the cache
    # moves out of it.
    for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "JITTOR_TEST_STATE_ROOT"):
        assert Path(first[name]).is_relative_to(root), name


def test_the_shared_cache_can_be_turned_off(monkeypatch, tmp_path):
    monkeypatch.setenv("JITTOR_NOX_SHARED_CACHE", "0")
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")

    root, env = module["_session_env"](session, "structure")

    assert Path(env["JITTOR_HOME"]).is_relative_to(root)


def test_a_populated_mirror_is_offered_to_the_session(monkeypatch, tmp_path):
    module = _load_noxfile(monkeypatch, tmp_path)
    session = _FakeSession(tmp_path, "/usr/bin/python3-config")

    _root, env = module["_session_env"](session, "structure")
    assert "JITTOR_OFFLINE_PATH" not in env

    module["NOX_JITTOR_ASSETS"].mkdir(parents=True, exist_ok=True)
    _root, env = module["_session_env"](session, "structure")
    assert Path(env["JITTOR_OFFLINE_PATH"]) == module["NOX_JITTOR_ASSETS"]


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

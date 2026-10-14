"""Host-only Nox session recorder shared by gate contract tests."""
import runpy
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeOptions:
    pass


class _FakeSession:
    python = "3.11"
    posargs = ()

    def __init__(self, root, python_config):
        self.root = root
        self.python_config = python_config
        self.calls = []
        self.logs = []

    def create_tmp(self):
        return str(self.root / "session-tmp")

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.python_config + "\n"

    def error(self, message):
        raise AssertionError(message)

    def log(self, message):
        self.logs.append(message)


def _load_noxfile(monkeypatch, tmp_path):
    fake_nox = types.ModuleType("nox")
    fake_nox.options = _FakeOptions()
    fake_nox.session = lambda *args, **kwargs: lambda function: function
    monkeypatch.setitem(sys.modules, "nox", fake_nox)
    monkeypatch.setenv("JITTOR_LAB_ROOT", str(tmp_path / "lab"))
    return runpy.run_path(str(REPO_ROOT / "noxfile.py"), run_name="jittor_noxfile")

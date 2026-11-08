"""Actual dependency bytes accompany every new shared-runner measurement."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    root = Path(__file__).resolve().parents[2]
    path = root / "compat/tests/torch/_ecosystem_runner.py"
    spec = importlib.util.spec_from_file_location("provenance_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_binds_actual_sources_data_and_binary_and_detects_changed_bytes(tmp_path, monkeypatch, runner):
    package = tmp_path / "dependency"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = '1.2.3'\n")
    (package / "kernel.so").write_bytes(b"compiled package bytes")
    (package / "config.json").write_text('{"dimension": 4}\n')
    nested = package / "nested"
    nested.mkdir()
    (nested / "operation.py").write_text("value = 1\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"ephemeral")
    module = SimpleNamespace(__file__=str(package / "__init__.py"),
                             __path__=[str(package)], __version__="1.2.3")
    monkeypatch.setitem(sys.modules, "provenance_dependency", module)
    before = runner._dependency_report(["provenance_dependency"])["provenance_dependency"]
    files = before["installed_files_sha256"]
    assert before["version"] == "1.2.3"
    assert before["origin"] == str(package / "__init__.py")
    assert set(files) == {"__init__.py", "kernel.so", "config.json", "nested/operation.py"}
    assert all(files[name] == hashlib.sha256((package / name).read_bytes()).hexdigest() for name in files)
    (package / "kernel.so").write_bytes(b"changed binary")
    after = runner._dependency_report(["provenance_dependency"])["provenance_dependency"]
    assert after["installed_files_sha256"]["kernel.so"] != files["kernel.so"]
    assert after["installed_files_sha256"]["__init__.py"] == files["__init__.py"]


def test_single_file_module_does_not_hash_neighboring_distributions(tmp_path, runner):
    source = tmp_path / "dependency.py"
    source.write_text("value = 1\n")
    (tmp_path / "unrelated-secret.txt").write_text("not part of this distribution")
    assert runner._installed_package_files(SimpleNamespace(__file__=str(source))) == {
        source.name: hashlib.sha256(source.read_bytes()).hexdigest()}


@pytest.mark.parametrize("kind", ["missing", "namespace", "extra-root", "linked-directory"])
def test_incomplete_package_identity_is_rejected(tmp_path, kind, runner):
    package = tmp_path / "dependency"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("")
    module = SimpleNamespace(__file__=str(source), __path__=[str(package)])
    if kind == "missing":
        source.unlink()
    elif kind == "namespace":
        module.__file__ = None
    elif kind == "extra-root":
        module.__path__.append(str(tmp_path / "other"))
    else:
        target = tmp_path / "outside"
        target.mkdir()
        (package / "linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        runner._installed_package_files(module)

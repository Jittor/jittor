"""The optional launcher must ship, build independently, and expose failures."""
import importlib.util
from pathlib import Path
import sys

import pytest


def _loader():
    source = Path(__file__).resolve().parents[2] / "triton" / "fastlaunch.py"
    spec = importlib.util.spec_from_file_location("fastlaunch_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="backend: Linux CUDA launcher build")
def test_fastlaunch_build_cache_and_opt_out(tmp_path, monkeypatch):
    module = _loader()
    monkeypatch.setenv("JITTOR_HOME", str(tmp_path))
    monkeypatch.setenv("JITTOR_TRITON_FASTLAUNCH", "1")
    monkeypatch.setenv("CC", "cc -DJITTOR_LAUNCHER_TEST=1")
    first = module._build()
    stamp = Path(first).stat().st_mtime_ns
    assert module._build() == first
    assert Path(first).stat().st_mtime_ns == stamp
    platform = module.sysconfig.get_platform()
    monkeypatch.setattr(module.sysconfig, "get_platform", lambda: platform + "-abi-probe")
    assert module._build() != first
    assert callable(module.load().launch)
    monkeypatch.setenv("JITTOR_TRITON_FASTLAUNCH", "0")
    assert module.load() is None
    assert not list(tmp_path.rglob("build-*.so"))


def test_fastlaunch_missing_compiler_is_diagnosed(tmp_path, monkeypatch):
    module = _loader()
    monkeypatch.setenv("JITTOR_HOME", str(tmp_path))
    monkeypatch.setenv("JITTOR_TRITON_FASTLAUNCH", "1")
    monkeypatch.setenv("CC", str(tmp_path / "missing-compiler"))
    with pytest.warns(RuntimeWarning, match="using ctypes"):
        assert module.load() is None
    assert not list(tmp_path.rglob("build-*.so"))


def test_fastlaunch_unexpected_build_failure_propagates(monkeypatch):
    module = _loader()
    monkeypatch.setenv("JITTOR_TRITON_FASTLAUNCH", "1")
    def fail():
        raise ValueError("implementation error")
    monkeypatch.setattr(module, "_build", fail)
    with pytest.raises(ValueError, match="implementation error"):
        module.load()
    assert not module._tried

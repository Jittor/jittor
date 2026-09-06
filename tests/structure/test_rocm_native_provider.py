"""Contracts for the source-owned ROCm provider."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from jittor_utils.build_config import BuildConfig, BuildContext, BuildSource


ROOT = Path(__file__).resolve().parents[2]
PROVIDER = ROOT / "backends" / "rocm" / "__init__.py"


def test_legacy_entrypoint_is_only_a_compatibility_forwarder():
    source = (ROOT / "python/jittor/extern/rocm/rocm_compiler.py").read_text()
    tree = ast.parse(source)
    assert "rocm_cache" not in source
    assert "transform_sources" not in source
    assert "process_rocm" not in source
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                   any((alias.name or "").startswith("jittor_utils")
                       for alias in node.names) for node in ast.walk(tree))


def test_native_provider_returns_build_config_without_transform(tmp_path, monkeypatch):
    sdk = tmp_path / "rocm"
    (sdk / "include").mkdir(parents=True)
    hipcc = sdk / "bin" / "hipcc"
    hipcc.parent.mkdir()
    hipcc.write_text("#!/bin/sh\nexit 0\n")
    hipcc.chmod(0o755)
    monkeypatch.setenv("ROCM_HOME", str(sdk))
    monkeypatch.setenv("hipcc_path", str(hipcc))

    spec = importlib.util.spec_from_file_location("rocm_native_provider", PROVIDER)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)

    config = BuildConfig(cc_flags=" -std=c++14 ", jittor_path=str(ROOT / "python/jittor"))
    context = BuildContext(config=config, compile_module=lambda *args: None)
    result = provider.configure(context)
    assert result.backend == "rocm"
    assert result.has_rocm and result.has_accelerator
    assert result.nvcc_path == result.hipcc_path == str(hipcc)
    assert result.kernel_language == "hip"
    assert len(result.backend_sources) == 1
    assert isinstance(result.backend_sources[0], BuildSource)
    assert result.backend_sources[0].language == "hip"
    assert result.backend_sources[0].path.endswith("backends/rocm/runtime/driver.cc")
    assert "rocm_converter" not in result.resources


def test_native_library_provider_rejects_unimplemented_miopen_and_rccl(tmp_path):
    path = ROOT / "backends/rocm/build.py"
    spec = importlib.util.spec_from_file_location("rocm_native_build", path)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)

    for name in ("MIOpen", "rccl"):
        with pytest.raises(NotImplementedError, match="no native ROCm library provider"):
            build.library_build(tmp_path, ROOT / "backends/rocm", name)

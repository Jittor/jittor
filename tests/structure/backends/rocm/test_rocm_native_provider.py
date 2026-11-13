"""Contracts for the source-owned ROCm provider."""

import ast
import importlib.util
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from jittor_utils.build_config import BuildConfig, BuildContext, BuildSource


ROOT = Path(__file__).resolve().parents[4]
PROVIDER = ROOT / "backends" / "rocm" / "__init__.py"


def test_legacy_entrypoint_is_only_a_compatibility_forwarder():
    source = (ROOT / "backends/rocm/__init__.py").read_text()
    tree = ast.parse(source)
    assert "rocm_cache" not in source
    assert "transform_sources" not in source
    assert "process_rocm" not in source
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                   any((alias.name or "").startswith("jittor_utils")
                       for alias in node.names) for node in ast.walk(tree))


@pytest.mark.parametrize("library_dir", ["lib", "lib64", "hip/lib", "hip/lib64"])
def test_native_provider_returns_build_config_without_transform(tmp_path, monkeypatch, library_dir):
    sdk = tmp_path / "rocm"
    (sdk / "include/hip").mkdir(parents=True)
    (sdk / "include/hip/hip_version.h").write_text(
        "#define HIP_VERSION_MAJOR 6\n#define HIP_VERSION_MINOR 3\n"
        "#define HIP_VERSION_PATCH 12345\n")
    hipcc = sdk / "bin" / "hipcc"
    hipcc.parent.mkdir()
    hipcc.write_text("#!/bin/sh\nexit 0\n")
    hipcc.chmod(0o755)
    (sdk / library_dir).mkdir(parents=True)
    (sdk / library_dir / "libamdhip64.so").touch()
    monkeypatch.setenv("ROCM_HOME", str(sdk))
    monkeypatch.setenv("hipcc_path", str(hipcc))

    spec = importlib.util.spec_from_file_location("rocm_native_provider", PROVIDER)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)

    config = BuildConfig(cc_flags=" -std=c++14 ", jittor_path=str(ROOT / "python/jittor"),
                         backend_link_flags=" -lretained ")
    context = BuildContext(config=config, compile_module=lambda *args: None)
    result = provider.configure(context)
    assert result.backend == "rocm"
    assert result.has_rocm and result.has_accelerator
    assert result.resources["hip_version"] == "6.3.12345"
    assert result.nvcc_path == result.hipcc_path == str(hipcc)
    assert result.kernel_language == "hip"
    assert "-D__HIP_PLATFORM_AMD__" in shlex.split(result.cc_flags)
    assert "-D__HIP_PLATFORM_AMD__" in shlex.split(result.kernel_compile_flags)
    assert "--gpu-max-threads-per-block=1024" in shlex.split(result.kernel_compile_flags)
    assert len(result.backend_sources) == 1
    assert isinstance(result.backend_sources[0], BuildSource)
    assert result.backend_sources[0].language == "hip"
    assert result.backend_sources[0].path.endswith("backends/rocm/runtime/driver.cc")
    assert "-lamdhip64" in shlex.split(result.backend_link_flags)
    assert "-lretained" in shlex.split(result.backend_link_flags)
    assert "-Wl,-rpath," + str(sdk / library_dir) in shlex.split(result.backend_link_flags)
    assert "rocm_converter" not in result.resources


def test_runtime_dependency_is_required_and_paths_are_quoted(tmp_path):
    spec = importlib.util.spec_from_file_location("rocm_runtime_dependency", PROVIDER)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    sdk = tmp_path / "ROCm SDK"
    with pytest.raises(RuntimeError, match="runtime library libamdhip64.so"):
        provider._runtime_link_flags(str(sdk))
    (sdk / "lib64").mkdir(parents=True)
    (sdk / "lib64/libamdhip64.so").touch()
    flags = shlex.split(provider._runtime_link_flags(str(sdk)))
    assert "-L" + str(sdk / "lib64") in flags
    assert "-Wl,-rpath," + str(sdk / "lib64") in flags
    assert "-lamdhip64" in flags


def test_hip_version_requires_complete_unambiguous_sdk_metadata(tmp_path):
    spec = importlib.util.spec_from_file_location("rocm_sdk_version", PROVIDER)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    with pytest.raises(RuntimeError, match="HIP version header is missing"):
        provider._hip_sdk_version(tmp_path)
    header = tmp_path / "include/hip/hip_version.h"
    header.parent.mkdir(parents=True)
    header.write_text("#define HIP_VERSION_MAJOR 7\n#define HIP_VERSION_MINOR 1\n")
    with pytest.raises(RuntimeError, match="numeric HIP_VERSION_PATCH"):
        provider._hip_sdk_version(tmp_path)
    header.write_text(header.read_text() + "#define HIP_VERSION_PATCH unknown\n")
    with pytest.raises(RuntimeError, match="numeric HIP_VERSION_PATCH"):
        provider._hip_sdk_version(tmp_path)
    header.write_text("#define HIP_VERSION_MAJOR 7 // major\n"
                      "#define HIP_VERSION_MINOR 1\n#define HIP_VERSION_PATCH 12\n")
    assert provider._hip_sdk_version(tmp_path) == "7.1.12"
    header.write_text(header.read_text() + "#define HIP_VERSION_PATCH 13\n")
    with pytest.raises(RuntimeError, match="one numeric HIP_VERSION_PATCH"):
        provider._hip_sdk_version(tmp_path)


def test_native_library_provider_rejects_unimplemented_miopen_and_rccl(tmp_path):
    path = ROOT / "backends/rocm/build.py"
    spec = importlib.util.spec_from_file_location("rocm_native_build", path)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)

    for name in ("MIOpen", "rccl"):
        with pytest.raises(NotImplementedError, match="no native ROCm library provider"):
            build.library_build(tmp_path, ROOT / "backends/rocm", name)

"""Offline provider contracts; these do not import Jittor or compile kernels."""

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"


def _load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def providers(monkeypatch):
    utils = ModuleType("jittor_utils")
    utils.LOG = Mock()
    utils.env_or_try_find = Mock(return_value="/rocm/bin/hipcc")
    utils.run_cmd = Mock(side_effect=lambda command: {
        "hipconfig -R": "/rocm",
        "hipconfig -v": "6.0",
        "hipconfig -C": "-I/rocm/include",
        "gcc -v": "--with-default-libstdcxx-abi=new",
    }[command])
    misc = ModuleType("jittor_utils.misc")
    misc.safe_tar_extractall = Mock()
    monkeypatch.setitem(sys.modules, "jittor_utils", utils)
    monkeypatch.setitem(sys.modules, "jittor_utils.misc", misc)
    _load(monkeypatch, "jittor_utils.compiler_flags", PYTHON / "jittor_utils/compiler_flags.py")
    _load(monkeypatch, "jittor_utils.backend_resources", PYTHON / "jittor_utils/backend_resources.py")
    config = _load(monkeypatch, "jittor_utils.build_config",
                   PYTHON / "jittor_utils/build_config.py")
    corex = _load(monkeypatch, "legacy_corex_contract",
                  PYTHON / "jittor/extern/corex/corex_compiler.py")
    return SimpleNamespace(utils=utils, misc=misc, config=config, corex=corex)


def _context(providers, tmp_path, **changes):
    config = providers.config.BuildConfig(
        cc_path="/usr/bin/g++", cc_type="g++",
        cc_flags="-std=c++14 -fopenmp -I/original", cache_path=str(tmp_path),
        jittor_path=str(PYTHON / "jittor"), resources={"retained": object()},
        environment={"retained": "1"}, extra_core_files=("existing.cc",),
    ).evolve(**changes)
    converter = SimpleNamespace(process=lambda source, name, args: source)
    return providers.config.BuildContext(
        config=config, compile_module=Mock(return_value=converter),
        compile=Mock(), compile_custom_ops=Mock(side_effect=lambda *args, **kw: object()),
        publish_library=Mock(), make_cache_dir=Mock(), load_library=Mock(),
        mpi_compile_flags=" -DMPI_ENABLED", so=".test.so",
    )


def test_import_does_not_probe_or_own_backend_state(providers):
    providers.utils.env_or_try_find.assert_not_called()
    providers.utils.run_cmd.assert_not_called()
    for module in (providers.corex,):
        assert not {"cc_flags", "has_rocm", "has_corex", "hipcc_path", "rocm_home"}.intersection(vars(module))
        tree = ast.parse(Path(module.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("jittor.") for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("jittor.")
            assert not isinstance(node, ast.Global)


@pytest.mark.parametrize("initial_cuda", [False, True])
def test_corex_configuration_is_detached_and_openmp_removed_before_publication(providers, tmp_path, initial_cuda):
    context = _context(providers, tmp_path, has_cuda=initial_cuda, is_cuda=initial_cuda,
                       cc_flags=" -std=c++14 -fopenmp -DIS_CUDA -I/original ",
                       kernel_flags=" -fopenmp -O2 ")
    home = tmp_path / "corex"
    (home / "bin").mkdir(parents=True)
    (home / "bin/clang++").touch()
    before_environment = dict(os.environ)
    result = providers.corex.configure(context, str(home))
    assert result.backend == "corex" and result.has_corex and not result.is_cuda
    assert result.has_accelerator and not result.has_cuda
    assert "-DHAS_ACCELERATOR" in result.cc_flags and "-DIS_COREX" in result.cc_flags
    assert "-DHAS_CUDA" not in result.cc_flags and "-DIS_CUDA" not in result.cc_flags
    assert "-DJT_DEFAULT_PARA_OPT_LEVEL=4" in result.cc_flags
    assert result.cc_path == result.nvcc_path == str(home / "bin/clang++")
    assert result.cc_type == "clang"
    assert "-fopenmp" not in result.cc_flags + result.nvcc_flags
    assert "-fopenmp" not in result.kernel_flags
    assert "-O2" in result.kernel_flags
    assert "-fopenmp" in context.config.kernel_flags
    context.compile_module.assert_not_called()
    assert "-x cu" in result.nvcc_flags and "-DNO_ATOMIC64" in result.nvcc_flags
    assert "-DHAS_CUDA" in result.nvcc_flags and "-DIS_CUDA" in result.nvcc_flags
    assert result.environment == {"retained": "1", "use_cutt": "0"}
    assert dict(os.environ) == before_environment
    assert "-fopenmp" in context.config.cc_flags
    assert not context.config.has_corex and context.config.backend == "cpu"
    assert result.extra_core_files == ("existing.cc",)
    assert result.resources["retained"] is context.config.resources["retained"]
    assert "corex_converter" not in result.resources
    assert result.kernel_compiler == str(home / "bin/clang++")
    assert result.kernel_language == "cuda" and result.kernel_source_suffix == ".cc"
    assert result.kernel_source_roots == (
        str((ROOT / "backends/cuda/kernels/core").resolve()),
    )
    assert result.kernel_flag_filter == ("--extended-lambda", "--expt-relaxed-constexpr")
    assert result.kernel_device_link is False
    assert len(result.backend_sources) == 4
    assert all(isinstance(unit, providers.config.BuildSource) for unit in result.backend_sources)
    assert result.backend_sources[0].path.endswith("cuda/runtime/driver.cc")
    assert result.backend_sources[-1].path.endswith("corex/runtime/corex_backend.cc")
    assert result.backend_sources[1].path.endswith("cuda/runtime/nan_checker.cc")
    assert result.backend_sources[2].path.endswith("cuda/kernels/debug/nan_checker.cu")
    assert result.backend_sources[2].language == "cuda"
    assert "-DHAS_CUDA" in result.backend_sources[0].flags
    assert result.resources["cuda_include"] == str(home / "include")
    assert result.resources["cuda_lib"] == str(home / "lib64")
    assert "-lcudart" in result.backend_link_flags
    assert result.convert_nvcc_flags("local") == "local"
    post_context = context.with_config(result)
    assert providers.corex.post_process(post_context) is result
    assert providers.corex.install_extern(post_context) is False


def test_corex_unavailable_is_an_error_without_compilation(providers, tmp_path):
    context = _context(providers, tmp_path)
    with pytest.raises(RuntimeError, match="compiler is missing"):
        providers.corex.configure(context, str(tmp_path))
    context.compile_module.assert_not_called()


def test_corex_flag_conversion_preserves_link_intent_and_quoted_values(providers):
    import shlex

    flags = ('--extended-lambda --expt-relaxed-constexpr -dc '
             '-DKEEP=--extended-lambda -I"/sdk path/include" '
             '"/tool path/dlink_compiler.py"')
    assert shlex.split(providers.corex.convert_nvcc_flags(flags)) == [
        "-dc", "-DKEEP=--extended-lambda", "-I/sdk path/include",
        "/tool path/dlink_compiler.py",
    ]


def test_corex_configuration_does_not_rewrite_core_source(providers):
    source = Path(providers.corex.__file__).read_text()
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Attribute)
                   and node.attr in {"compile_module", "transform_sources"}
                   for node in ast.walk(tree))
    assert "process_acl" not in source and "token_replace_all" not in source
    provider = (ROOT / "backends/corex/runtime/corex_backend.cc").read_text()
    assert "auto ops = make_cuda_backend();" in provider
    assert 'ops.name = "corex";' in provider
    assert "ops.execution.prefer_compaction_kernel = true;" in provider
    assert "ops.execution.warp_shuffle_width = 64;" in provider
    assert "ops.execution.supports_generated_device_kernels = true;" in provider
    assert "ops.execution.ordered_float_atomics = true;" in provider


def test_corex_factory_reuses_sdk_callbacks_and_declares_64_lane_policy(tmp_path):
    import shutil
    import subprocess

    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("host C++ compiler is unavailable")
    probe = tmp_path / "corex_factory.cc"
    probe.write_text('''
#include "runtime/backend.h"
#include <cassert>
#include <cstring>
namespace jittor {
int constructions = 0;
int sdk_device_count() { return 2; }
BackendOps make_cuda_backend() {
    ++constructions;
    BackendOps ops;
    ops.id = BackendId::Cuda;
    ops.name = "cuda";
    ops.device_count = sdk_device_count;
    ops.execution.supports_parallel_compile = false;
    return ops;
}
}
int main() {
    assert(jittor::constructions == 0);
    auto ops = jittor::make_corex_backend();
    assert(jittor::constructions == 1);
    assert(ops.id == jittor::BackendId::Corex);
    assert(std::strcmp(ops.name, "corex") == 0);
    assert(ops.device_count == jittor::sdk_device_count);
    assert(ops.device_count() == 2);
    assert(!ops.execution.supports_parallel_compile);
    assert(ops.execution.prefer_compaction_kernel);
    assert(ops.execution.supports_generated_device_kernels);
    assert(ops.execution.warp_shuffle_width == 64);
    assert(ops.execution.ordered_float_atomics);
}
''')
    binary = tmp_path / "corex_factory"
    subprocess.run([
        compiler, "-std=c++14", "-I" + str(PYTHON / "jittor/src"), str(probe),
        str(ROOT / "backends/corex/runtime/corex_backend.cc"), "-o", str(binary),
    ], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
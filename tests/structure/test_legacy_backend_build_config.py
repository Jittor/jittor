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
    config = _load(monkeypatch, "legacy_build_config_contract",
                   PYTHON / "jittor_utils/build_config.py")
    rocm = _load(monkeypatch, "legacy_rocm_contract",
                 PYTHON / "jittor/extern/rocm/rocm_compiler.py")
    corex = _load(monkeypatch, "legacy_corex_contract",
                  PYTHON / "jittor/extern/corex/corex_compiler.py")
    return SimpleNamespace(utils=utils, misc=misc, config=config, rocm=rocm, corex=corex)


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
        transform_sources=Mock(side_effect=lambda config, name, callback:
            config.evolve(cc_flags=config.cc_flags + " -DTRANSFORMED")),
        compile=Mock(), compile_custom_ops=Mock(side_effect=lambda *args, **kw: object()),
        publish_library=Mock(), make_cache_dir=Mock(), load_library=Mock(),
        mpi_compile_flags=" -DMPI_ENABLED", so=".test.so",
    )


def test_import_does_not_probe_or_own_backend_state(providers):
    providers.utils.env_or_try_find.assert_not_called()
    providers.utils.run_cmd.assert_not_called()
    for module in (providers.rocm, providers.corex):
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
    assert result.has_cuda
    assert "-DHAS_CUDA" in result.cc_flags and "-DIS_CUDA" not in result.cc_flags
    assert result.cc_path == result.nvcc_path == str(home / "bin/clang++")
    assert result.cc_type == "clang"
    assert "-fopenmp" not in result.cc_flags + result.nvcc_flags
    assert "-fopenmp" not in result.kernel_flags
    assert "-O2" in result.kernel_flags
    assert "-fopenmp" in context.config.kernel_flags
    assert "-DTRANSFORMED" in result.cc_flags
    assert "-x cu" in result.nvcc_flags and "-DNO_ATOMIC64" in result.nvcc_flags
    assert result.environment == {"retained": "1", "use_cutt": "0"}
    assert dict(os.environ) == before_environment
    assert "-fopenmp" in context.config.cc_flags
    assert not context.config.has_corex and context.config.backend == "cpu"
    assert result.extra_core_files == ("existing.cc",)
    assert result.resources["retained"] is context.config.resources["retained"]
    assert result.resources["corex_converter"] is context.compile_module.return_value
    assert result.convert_nvcc_flags("local") == "local"
    post_context = context.with_config(result)
    assert providers.corex.post_process(post_context) is result
    assert providers.corex.install_extern(post_context) is False


def test_corex_unavailable_is_an_error_without_compilation(providers, tmp_path):
    context = _context(providers, tmp_path)
    with pytest.raises(RuntimeError, match="compiler is missing"):
        providers.corex.configure(context, str(tmp_path))
    context.compile_module.assert_not_called()


@pytest.mark.parametrize("new_abi,member", [
    (True, "rocm_cache_cxx11.o"), (False, "rocm_cache.o"),
])
def test_rocm_configuration_preserves_abi_archive_and_returns_all_inputs(
        providers, tmp_path, monkeypatch, new_abi, member):
    context = _context(providers, tmp_path)
    monkeypatch.setattr(providers.rocm, "check_gcc_use_cxx11_abi", lambda: new_abi)
    driver = SimpleNamespace(hipDeviceSynchronize=Mock(return_value=0))
    monkeypatch.setattr(providers.rocm.ctypes, "CDLL", Mock(return_value=driver))
    result = providers.rocm.configure(context)
    assert result.backend == "rocm" and result.has_rocm
    assert not context.config.has_cuda
    assert result.has_cuda and not result.is_cuda
    assert result.nvcc_path == result.hipcc_path == "/rocm/bin/hipcc"
    assert "-DHAS_CUDA" in result.cc_flags and "-DIS_ROCM" in result.cc_flags
    assert "-DTRANSFORMED" in result.cc_flags and "-lamdhip64" in result.cc_flags
    assert "-std=c++17" in result.nvcc_flags and "-std=c++14" in result.cc_flags
    assert result.resources["rocm_home"] == "/rocm"
    assert result.resources["rocm_driver"] is driver
    assert result.resources["rocm_converter"] is context.compile_module.return_value
    assert result.resources["retained"] is context.config.resources["retained"]
    assert result.extra_core_files == ("existing.cc",)
    assert not context.config.has_rocm and "-DIS_ROCM" not in context.config.cc_flags
    extraction = providers.misc.safe_tar_extractall.call_args
    assert extraction.args[1] == str(tmp_path / "rocm")
    assert [entry.name for entry in extraction.kwargs["members"]] == [member]
    assert str(tmp_path / "rocm" / member) in context.compile_module.call_args.args[1]
    assert providers.rocm.post_process(context.with_config(result)) is result
    with pytest.raises(TypeError):
        result.resources["rocm_home"] = "other"


def test_rocm_missing_compiler_does_not_start_build(providers, tmp_path):
    context = _context(providers, tmp_path)
    providers.utils.env_or_try_find.return_value = ""
    with pytest.raises(RuntimeError, match="hipcc is unavailable"):
        providers.rocm.configure(context)
    providers.utils.run_cmd.assert_not_called()
    context.compile_module.assert_not_called()


def test_rocm_failed_driver_initialization_is_not_advertised(providers, tmp_path, monkeypatch):
    context = _context(providers, tmp_path)
    driver = SimpleNamespace(hipDeviceSynchronize=Mock(return_value=17))
    monkeypatch.setattr(providers.rocm.ctypes, "CDLL", Mock(return_value=driver))
    with pytest.raises(RuntimeError, match="hipDeviceSynchronize=17"):
        providers.rocm.configure(context)
    assert not context.config.has_rocm


def test_rocm_extern_uses_injected_build_and_publication_services(providers, tmp_path):
    context = _context(providers, tmp_path, backend="rocm", has_rocm=True,
                       resources={"rocm_home": "/selected/rocm"})
    assert providers.rocm.install_extern(context)
    args = context.compile.call_args.args
    assert args[0] == context.config.cc_path
    assert context.config.cc_flags in args[1]
    cuda_root = providers.rocm.backend_root(context.config.jittor_path, "cuda")
    assert args[2] and all(path.startswith(cuda_root) for path in args[2])
    assert args[3] == str(tmp_path / "cuda/libcuda_extern.test.so")
    context.load_library.assert_called_once_with(args[3], os.RTLD_NOW | os.RTLD_GLOBAL)
    published = context.publish_library.call_args_list
    assert [call.args[0] for call in published] == ["cuda", "cudnn", "cublas", "cub", "nccl"]
    assert published[0].args[1] is context.load_library.return_value
    builds = context.compile_custom_ops.call_args_list
    assert len(builds) == 4
    for call in builds:
        assert call.kwargs["return_module"] and call.kwargs["backend"] == "accelerator"
        assert "/selected/rocm" in call.kwargs["extra_flags"]
        assert str(tmp_path / "cuda") in call.kwargs["extra_flags"]
    assert "-DMPI_ENABLED" in builds[-1].kwargs["extra_flags"]
    assert "-lrocprim" not in builds[2].kwargs["extra_flags"]


def test_rocm_unselected_extern_does_nothing(providers, tmp_path):
    context = _context(providers, tmp_path)
    assert providers.rocm.install_extern(context) is False
    context.compile.assert_not_called()
    context.compile_custom_ops.assert_not_called()
    context.publish_library.assert_not_called()


def test_rocm_extern_compile_failure_propagates(providers, tmp_path):
    context = _context(providers, tmp_path, has_rocm=True, resources={"rocm_home": "/rocm"})
    context.compile.side_effect = RuntimeError("HIP build failed")
    with pytest.raises(RuntimeError, match="HIP build failed"):
        providers.rocm.install_extern(context)
    context.publish_library.assert_not_called()

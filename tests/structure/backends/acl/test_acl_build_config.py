"""ACL configuration ownership without importing Jittor or loading CANN."""

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "backends/acl/__init__.py"


def _load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def acl(monkeypatch):
    def unexpected_probe(*args, **kwargs):
        raise AssertionError("module import probed the toolchain")
    with monkeypatch.context() as guard:
        guard.setattr(shutil, "which", unexpected_probe)
        return _load_module(monkeypatch, "acl_build_config_test", SOURCE)


@pytest.fixture
def setup(acl, monkeypatch, tmp_path):
    # The provider now returns BuildSource values, and BuildConfig validates
    # them with isinstance. A second copy of build_config.py loaded by path
    # would define a different BuildSource class than the provider imports, so
    # take the canonical module -- the same one the ROCm provider test uses.
    import jittor_utils.build_config as api
    toolkit = tmp_path / "toolkit"
    toolkit.mkdir()
    monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(toolkit))
    monkeypatch.setenv("tikcc_path", "selected-ccec")
    monkeypatch.setenv("use_mkl", "1")
    monkeypatch.setattr(acl.shutil, "which", lambda name: "/cann/bin/" + name)
    calls = []
    library = object()
    converter = SimpleNamespace(process=lambda *args: "converted",
                                init_acl_ops=lambda: calls.append(("init",)))
    base = api.BuildConfig(
        cc_flags="-std=c++14 -I/source/src", jittor_path="/source",
        cache_path="/cache", extra_core_files=("existing.cc",),
        environment={"existing_env": "yes"}, resources={"existing_resource": 3},
    )
    def compile_module(source, flags):
        calls.append(("compile", source, flags))
        return converter
    def load_library(name, mode):
        calls.append(("load", name, mode))
        return library
    context = api.BuildContext(base, compile_module, load_library=load_library)
    return SimpleNamespace(api=api, context=context, base=base, calls=calls,
                           library=library, converter=converter)


def test_configure_returns_complete_value_without_global_writes(acl, setup):
    before_environment = dict(os.environ)
    config = acl.configure(setup.context)
    assert config.backend == "acl"
    assert config.has_acl and config.has_cuda and not config.is_cuda
    assert not config.has_rocm and not config.has_corex
    assert config.nvcc_path == config.tikcc_path == "/cann/bin/selected-ccec"
    # No fake CUDA libraries: that path compiles backends/cuda/kernels/<lib>,
    # which are CUDA/cuDNN translation units. They only built under ACL while
    # the 1.x provider rewrote every jittor source through process_acl(); this
    # provider exposes no converter.
    assert not config.setup_fake_cuda_lib
    assert config.has_accelerator
    # The provider runtime is compiled by BuildConfig, not folded into the
    # registration module.
    assert [os.path.basename(source.path) for source in config.backend_sources] == [
        "backend.cc", "workspace.cc"]
    assert all(isinstance(source, setup.api.BuildSource)
               for source in config.backend_sources)
    # A generated ACL operator is host C++ calling aclnn, not ccec device source.
    assert config.kernel_language == "cxx"
    assert config.kernel_compiler == setup.base.cc_path
    assert config.kernel_compile_flags == config.cc_flags
    assert config.kernel_source_suffix == ".cc"
    assert not config.kernel_device_link
    assert "-I/source/src" in config.cc_flags
    assert "-DIS_ACL" in config.cc_flags
    assert config.nvcc_flags == config.cc_flags.replace("-std=c++14", "")
    assert config.environment == {"existing_env": "yes", "use_mkl": "0"}
    assert config.resources["acl_initializer"] is setup.converter
    assert config.resources["acl_library"] is setup.library
    assert config.resources["existing_resource"] == 3
    assert config.extra_core_files[0] == "existing.cc"
    expected_extra = [str(SOURCE.parent / "src/acl_op_exec.cc")]
    expected_extra.extend(str(path) for path in sorted(
        (SOURCE.parent / "kernels/native").glob("*.cc")))
    converter_sources = [str(SOURCE.parent / "src" / name) for name in (
        "acl_error_code.cc", "acl_jittor.cc", "aclnn.cc")]
    assert len(expected_extra) == 42
    assert len(converter_sources) == 3
    assert config.extra_core_files == ("existing.cc", *expected_extra)
    assert [call[0] for call in setup.calls] == ["load", "compile"]
    assert setup.calls[0][1:] == ("libascendcl.so", os.RTLD_NOW | os.RTLD_GLOBAL)
    converter_flags = setup.calls[1][2]
    assert "-I/source/src" in converter_flags
    assert all(name in converter_flags for name in converter_sources)
    assert all(name not in converter_flags for name in expected_extra)
    for name in ("backend.cc", "workspace.cc"):
        assert str(SOURCE.parent / "src" / name) not in converter_flags
        assert str(SOURCE.parent / "src" / name) not in config.extra_core_files
    for directory in ("include", "include/aclnn", "include/aclops"):
        assert "-I" + str(SOURCE.parent / directory) in config.cc_flags
    assert setup.base.extra_core_files == ("existing.cc",)
    assert setup.base.cc_flags == "-std=c++14 -I/source/src"
    assert setup.base.environment == {"existing_env": "yes"}
    assert dict(os.environ) == before_environment
    assert acl.configure(setup.context) == config
    with pytest.raises(TypeError):
        config.environment["use_mkl"] = "1"


@pytest.mark.parametrize("initial_cuda", [False, True])
def test_configuration_declares_accelerator_independently_of_nvcc(acl, setup, initial_cuda):
    from dataclasses import replace
    base = setup.base.evolve(has_cuda=initial_cuda)
    context = replace(setup.context, config=base)
    assert acl.configure(context).has_cuda
    assert acl.install_extern(context) is False


@pytest.mark.parametrize("compiler", ["", "missing-ccec"])
def test_selected_acl_missing_compiler_fails_before_work(acl, setup, monkeypatch, compiler):
    monkeypatch.setenv("tikcc_path", compiler)
    monkeypatch.setattr(acl.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="compiler was not found"):
        acl.configure(setup.context)
    assert not setup.calls


def test_configure_finds_ccec_only_when_selected(acl, setup, monkeypatch):
    monkeypatch.delenv("tikcc_path")
    assert acl.configure(setup.context).tikcc_path == "/cann/bin/ccec"


@pytest.mark.parametrize("toolkit", [None, "/nonexistent/cann/toolkit"])
def test_missing_toolkit_fails_before_work(acl, setup, monkeypatch, toolkit):
    if toolkit is None:
        monkeypatch.delenv("ASCEND_TOOLKIT_HOME")
    else:
        monkeypatch.setenv("ASCEND_TOOLKIT_HOME", toolkit)
    with pytest.raises(RuntimeError, match="ASCEND_TOOLKIT_HOME"):
        acl.configure(setup.context)
    assert not setup.calls


@pytest.mark.parametrize("service", ["load_library", "compile_module"])
def test_configuration_failures_propagate(acl, setup, service):
    from dataclasses import replace
    def failure(*args, **kwargs):
        raise RuntimeError(service + " failed")
    context = replace(setup.context, **{service: failure})
    with pytest.raises(RuntimeError, match=service + " failed"):
        acl.configure(context)
    assert setup.base.environment == {"existing_env": "yes"}
    assert os.environ["use_mkl"] == "1"


def test_post_process_uses_returned_converter(acl, setup, monkeypatch):
    config = acl.configure(setup.context)
    flags = SimpleNamespace(use_cuda_host_allocator=0, use_parallel_op_compiler=1,
                            amp_reg=8)
    pool = SimpleNamespace(pool_use_code_op=True)
    fake_jittor = SimpleNamespace(
        pool=pool, flags=flags,
        amp_flags=SimpleNamespace(reduce16_no_fp32_acc=2, keep_reduce=4),
    )
    monkeypatch.setitem(sys.modules, "jittor", fake_jittor)
    acl.post_process(setup.context.with_config(config))
    # These defaults are native BackendOps.execution policies now, not
    # mutations of CPU/Torch state during ACL bootstrap.
    assert pool.pool_use_code_op is True
    assert flags.use_cuda_host_allocator == 0
    assert flags.use_parallel_op_compiler == 1
    assert flags.amp_reg == 8
    assert setup.calls[-1] == ("init",)
    before = list(setup.calls)
    acl.post_process(setup.context)
    assert setup.calls == before


def test_backend_has_no_mutable_configuration_or_compiler_imports(acl):
    assert not any(name in vars(acl) for name in
                   ("has_acl", "cc_flags", "tikcc_path", "mod", "compiler", "jt", "check"))
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "jittor.compiler" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "jittor.compiler"
    assert not any(isinstance(node, ast.Global) for node in ast.walk(tree))


def test_provider_source_inventory_is_explicit_and_complete(acl):
    sources = acl.REGISTRATION_SOURCES + acl.CORE_SOURCES
    assert len(sources) == len(set(sources)) == 45
    assert all((SOURCE.parent / name).is_file() for name in sources)
    actual = {str(path.relative_to(SOURCE.parent))
              for path in SOURCE.parent.rglob("*.cc")}
    assert actual == set(sources) | {"src/backend.cc", "src/workspace.cc"}
    assert "glob" not in vars(acl)

"""ACL configuration ownership without importing Jittor or loading CANN."""

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "python/jittor/extern/acl/acl_compiler.py"


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
    api = _load_module(
        monkeypatch, "acl_build_config_values_test",
        ROOT / "python/jittor_utils/build_config.py",
    )
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
    def transform_sources(config, backend, callback):
        assert config is base
        assert callback is converter.process
        calls.append(("transform", backend))
        return config.evolve(jittor_path="/converted",
                             cc_flags=config.cc_flags.replace("/source", "/converted"))
    def load_library(name, mode):
        calls.append(("load", name, mode))
        return library
    context = api.BuildContext(base, compile_module, transform_sources,
                               load_library=load_library)
    return SimpleNamespace(api=api, context=context, base=base, calls=calls,
                           library=library, converter=converter)


def test_configure_returns_complete_value_without_global_writes(acl, setup):
    before_environment = dict(os.environ)
    config = acl.configure(setup.context)
    assert config.backend == "acl"
    assert config.has_acl and config.has_cuda and not config.is_cuda
    assert not config.has_rocm and not config.has_corex
    assert config.nvcc_path == config.tikcc_path == "/cann/bin/selected-ccec"
    assert config.setup_fake_cuda_lib
    assert "-I/source/src" in config.cc_flags
    assert "-DIS_ACL" in config.cc_flags
    assert config.nvcc_flags == config.cc_flags.replace("-std=c++14", "")
    assert config.environment == {"existing_env": "yes", "use_mkl": "0"}
    assert config.resources["acl_initializer"] is setup.converter
    assert config.resources["acl_library"] is setup.library
    assert config.resources["existing_resource"] == 3
    assert config.extra_core_files[0] == "existing.cc"
    expected_extra = []
    converter_sources = []
    for path in sorted(SOURCE.parent.rglob("*.cc")):
        name = str(path)
        if "hccl" in name:
            continue
        if "acl_op_exec" in name or "_op_acl.cc" in name or "utils.cc" in name:
            expected_extra.append(name)
        else:
            converter_sources.append(name)
    assert config.extra_core_files == ("existing.cc", *expected_extra)
    assert [call[0] for call in setup.calls] == ["load", "compile"]
    assert setup.calls[0][1:] == ("libascendcl.so", os.RTLD_NOW | os.RTLD_GLOBAL)
    converter_flags = setup.calls[1][2]
    assert "-I/source/src" in converter_flags
    assert all(name in converter_flags for name in converter_sources)
    assert all(name not in converter_flags for name in expected_extra)
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
    context = replace(setup.context, config=base,
                      transform_sources=lambda config, backend, callback: config)
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

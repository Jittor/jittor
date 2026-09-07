"""Build-provider value semantics and bootstrap direction without a JIT build."""

import ast
import builtins
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from jittor_utils.env_config import build_env, build_flag


ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "python/jittor/build/utils"


def _load(name):
    spec = importlib.util.spec_from_file_location("build_boundary_" + name, UTILS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_config_copies_mutable_inputs_and_is_immutable():
    api = _load("build_config")
    sources, environment, resources = ["one.cc"], {"use_mkl": "0"}, {"home": "/sdk"}
    config = api.BuildConfig(extra_core_files=sources, environment=environment, resources=resources)
    sources.append("two.cc")
    environment["use_mkl"] = "1"
    resources["home"] = "/other"
    assert config.extra_core_files == ("one.cc",)
    assert config.environment["use_mkl"] == "0"
    assert config.resources["home"] == "/sdk"
    with pytest.raises(FrozenInstanceError):
        config.cc_flags = "changed"
    with pytest.raises(TypeError):
        config.environment["use_mkl"] = "1"
    changed = config.evolve(cc_flags="new")
    assert changed.cc_flags == "new" and config.cc_flags == ""


def test_native_selection_does_not_load_optional_entry_points():
    discovery = _load("backend_discovery")
    class Entry:
        name, group = "corex", discovery.ENTRY_POINT_GROUP
        def load(self):
            pytest.fail("unselected provider was imported")
    assert discovery.requested_backend({}, is_file=lambda path: False) is None
    for name in (None, "cpu", "cuda"):
        assert discovery.load_backend_provider(name, entries=[Entry()]) is None


def test_entrypoint_selection_is_explicit_and_order_independent():
    discovery = _load("backend_discovery")
    calls = []
    class Entry:
        group = discovery.ENTRY_POINT_GROUP
        def __init__(self, name):
            self.name = name
        def load(self):
            calls.append(self.name)
            return SimpleNamespace(configure=lambda context: context.config)
    entries = [Entry("acl"), Entry("rocm"), Entry("corex")]
    assert discovery.requested_backend({"JT_BACKEND": "npu", "COREX_HOME": "/sdk"}) == "acl"
    discovery.load_backend_provider("acl", entries=entries)
    discovery.load_backend_provider("acl", entries=list(reversed(entries)))
    assert calls == ["acl", "acl"]
    with pytest.raises(RuntimeError, match="multiple backend SDKs"):
        discovery.requested_backend({"ROCM_HOME": "/rocm", "COREX_HOME": "/corex"})
    with pytest.raises(RuntimeError, match="not installed"):
        discovery.load_backend_provider("unavailable_plugin", entries=[])


def test_available_sdk_selection_checks_paths_without_imports():
    discovery = _load("backend_discovery")
    checked = []
    def exists(path):
        checked.append(path)
        return path == "/opt/rocm/bin/hipcc"
    assert discovery.requested_backend({}, is_file=exists) == "rocm"
    assert len(checked) == 3
    assert discovery.requested_backend({"hipcc_path": "/custom/hipcc"}, is_file=exists) == "rocm"
    assert len(checked) == 3


def test_utils_do_not_import_the_runtime():
    violations = []
    for path in UTILS.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(name == "jittor" or name.startswith("jittor.") for name in names):
                violations.append((str(path.relative_to(UTILS)), node.lineno))
    assert not violations, violations


def test_compile_module_requires_injected_services_before_writing(tmp_path):
    path = UTILS / "__init__.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "compile_module")
    function.decorator_list = []
    namespace = {"_module_build_services": None}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(RuntimeError, match="explicit ModuleBuildServices"):
        namespace["compile_module"]("source", "flags")
    assert list(tmp_path.iterdir()) == []


def test_compile_module_consumes_injected_generator_and_formatter(tmp_path):
    path = UTILS / "__init__.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "compile_module")
    function.decorator_list = []
    commands, generated = [], []
    module = object()
    def generate(header, source):
        generated.append(Path(header).read_text())
        Path(source).write_text("// generated binding\n")
        return True
    services = SimpleNamespace(
        compile_single=generate, fix_flags=lambda command: "formatted " + command,
        cc_path="/chosen/compiler", cache_path=str(tmp_path), jittor_path="/chosen/source",
    )
    namespace = {
        "os": os, "_module_build_services": None,
        "get_str_hash": lambda text: "probe", "get_py3_extension_suffix": lambda: ".so",
        "do_compile": commands.append, "lock": SimpleNamespace(unlock_scope=nullcontext),
        "import_scope": lambda mode: nullcontext(),
        "__builtins__": dict(vars(builtins), __import__=lambda name: module),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["compile_module"]("binding source", " -DLOCAL=1 ", services=services) is module
    assert generated == ["binding source"]
    assert len(commands) == 1
    command, cache, source = commands[0]
    assert command.startswith('formatted "/chosen/compiler" ')
    assert " -DLOCAL=1 " in command
    assert cache == str(tmp_path) and source == "/chosen/source"
    assert '#include "bindings/pyjt/py_arg_printer.cc"' in (tmp_path / "tmp/hash_probe.cc").read_text()


def test_default_cache_config_stays_unchanged_and_explicit_backend_isolated(monkeypatch):
    path = UTILS / "__init__.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("get_build_config", "save_mem_build_flags")]
    names = ("cc_flags", "nvcc_flags", "kernel_flags", "cuda_archs", "enable_lto", "nvcc_path")
    # The real resolver: since 2.22 every build variable is read through it,
    # so a stub here would test the stub rather than the fingerprint.
    namespace = {"os": os, "BUILD_CONFIG_VARS": names, "build_env": build_env,
                 "build_flag": build_flag}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    for name in names + ("disable_lock", "JT_SAVE_MEM", "JT_BACKEND", "ASCEND_TOOLKIT_HOME",
                         "ASCEND_HOME_PATH", "tikcc_path", "ROCM_HOME", "ROCM_PATH", "HIP_PATH",
                         "hipcc_path", "COREX_HOME"):
        monkeypatch.delenv(name, raising=False)
    assert namespace["get_build_config"]() == dict.fromkeys(names)
    monkeypatch.setenv("JT_BACKEND", "corex")
    assert namespace["get_build_config"]() == dict(dict.fromkeys(names), JT_BACKEND="corex")


def test_explicit_cpu_skips_every_cuda_discovery_and_installation_service():
    path = ROOT / "python/jittor/build/compiler.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_discover_cuda_compiler")
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("CPU selection touched CUDA discovery")
    namespace = {
        "install_cuda": SimpleNamespace(has_installation=forbidden, install_cuda=forbidden),
        "env_or_try_find": forbidden, "try_find_exe": forbidden,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["_discover_cuda_compiler"]("cpu") == ""
    assert not calls


@pytest.mark.parametrize("name", ["setup_cuda_extern", "setup_cuda_lib", "setup_cub",
                                  "setup_cutt", "_load_cuda_library"])
def test_compatible_library_loaders_keep_corex_but_not_acl_or_rocm(name):
    path = ROOT / "python/jittor/build/compile_extern.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    class EnteredLoader(Exception):
        pass
    def entered(*args, **kwargs):
        raise EnteredLoader()
    namespace = {
        "has_cuda": True, "is_cuda": False, "has_corex": False,
        "cuda_wheel_stack": None, "build_env": build_env, "build_flag": build_flag,
        "setup_fake_cuda_lib": False, "os": SimpleNamespace(environ={}, path=os.path),
        "platform": SimpleNamespace(machine=lambda: "x86_64"),
        "LOG": SimpleNamespace(v=entered, vv=entered),
        "jit_utils": SimpleNamespace(home=entered), "library_resource": entered,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    args = ("cufft",) if name in ("setup_cuda_lib", "_load_cuda_library") else ()
    assert namespace[name](*args) is None
    namespace["has_corex"] = True
    with pytest.raises(EnteredLoader):
        namespace[name](*args)


def test_nccl_loader_does_not_probe_nvidia_for_non_nvidia_build():
    path = ROOT / "python/jittor/build/compile_extern.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "setup_nccl")
    namespace = {"os": SimpleNamespace(environ={"JT_NCCL_WORLD_SIZE": "2"}),
                 "has_cuda": True, "is_cuda": False, "has_mpi": True,
                 "build_env": build_env, "build_flag": build_flag}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["setup_nccl"]() is None
    assert namespace["use_nccl"] is False

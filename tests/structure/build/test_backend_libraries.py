"""Library publication and lazy-load contracts without native initialization."""

import ast
import importlib.util
import os
from pathlib import Path
import types

import pytest

from jittor_utils.env_config import build_flag


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/jittor/_runtime/backend_libraries.py"
EXTERN = ROOT / "python/jittor/build/compile_extern.py"


@pytest.fixture
def api():
    spec = importlib.util.spec_from_file_location("library_registry_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_library_queries_do_not_load_or_cache_misses(api):
    calls = []
    library = types.SimpleNamespace(ops=object())
    def load():
        calls.append("load")
        api.register_library("cub", library)
    api.register_library_loader("cub", load)
    assert api.get_library("cub") is None
    assert api.get_library_ops("cub") is None
    assert api.get_library("unknown", load=True) is None
    assert not calls
    assert api.get_library_ops("cub", load=True) is library.ops
    assert api.get_library("cub", load=True) is library
    assert calls == ["load"]
    assert api.library_attribute("cub") is library
    assert api.library_attribute("cub_ops") is library.ops


def test_missing_and_failed_loaders_remain_retryable(api):
    calls = []
    library = types.SimpleNamespace(ops=object())
    def load():
        calls.append("load")
        if len(calls) == 1:
            return
        api.register_library("mkl", library)
        if len(calls) == 2:
            raise RuntimeError("compile failed")
    api.register_library_loader("mkl", load)
    assert api.get_library("mkl", load=True) is None
    with pytest.raises(RuntimeError, match="compile failed"):
        api.get_library("mkl", load=True)
    assert api.get_library("mkl") is None
    assert api.get_library("mkl", load=True) is library
    assert calls == ["load"] * 3


def test_ops_follow_the_module_and_resources_have_one_owner(api):
    first = types.SimpleNamespace(ops=object())
    second = types.SimpleNamespace(ops=object())
    api.register_library_resources("cub", home="")
    assert api.library_attribute("cub_home") == ""
    assert api.library_resource("cub", "missing") is None
    api.register_library("cub", first)
    api.register_library("cub", second)
    assert api.get_library_ops("cub") is second.ops
    replacement = object()
    second.ops = replacement
    assert api.library_attribute("cub_ops") is replacement
    api.register_library_resources("cub", home="/toolkit/cub/")
    assert api.library_attribute("cub_home") == "/toolkit/cub/"


def test_recursive_load_is_rejected_and_loading_state_is_released(api):
    api.register_library_loader("mkl", lambda: api.get_library("mkl", load=True))
    with pytest.raises(RuntimeError, match="recursive backend library load"):
        api.get_library("mkl", load=True)
    library = types.SimpleNamespace(ops=object())
    api.register_library_loader("mkl", lambda: api.register_library("mkl", library))
    assert api.get_library("mkl", load=True) is library


def test_legacy_attributes_are_live_read_only_queries(api):
    module = types.ModuleType("compile_extern_test")
    module.__getattr__ = api.library_attribute
    api.protect_library_attributes(module)
    assert module.mkl_ops is None
    library = types.SimpleNamespace(ops=object())
    api.register_library("mkl", library)
    assert module.mkl_ops is library.ops
    assert "mkl_ops" not in module.__dict__
    assert "mkl_ops" in dir(module)
    with pytest.raises(AttributeError, match="read-only"):
        module.mkl_ops = object()
    with pytest.raises(AttributeError, match="read-only"):
        del module.mkl_ops
    module.setup_mkl = object()
    with pytest.raises(AttributeError):
        module.not_a_library


def _mkl_bootstrap(api):
    tree = ast.parse(EXTERN.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("setup_mkl", "_mkl_library_enabled")]
    registration = next(node for node in tree.body
                        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "register_library_loader"
                        and ast.literal_eval(node.value.args[0]) == "mkl")
    # The real resolver, not a stub: `use_mkl` in the environment is what this
    # test varies, and since 2.22 that lookup is `env_config`'s job (it also
    # accepts JT_BUILD_USE_MKL). Substituting a fake here would test the fake.
    namespace = {"os": os, "use_mkl": True, "build_flag": build_flag,
                 "register_library_loader": api.register_library_loader}
    exec(compile(ast.Module(body=functions + [registration], type_ignores=[]),
                 str(EXTERN), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("flag,environment", [(False, "1"), (True, "0")])
def test_disabled_mkl_loader_does_not_download_or_publish(api, monkeypatch, flag, environment):
    monkeypatch.setenv("use_mkl", environment)
    namespace = _mkl_bootstrap(api)
    namespace["use_mkl"] = flag
    # A direct call cannot silently reset an explicit False to the env default.
    namespace["setup_mkl"]()
    assert namespace["use_mkl"] is flag
    namespace["setup_mkl"] = lambda: pytest.fail("disabled loader was called")
    assert api.get_library("mkl", load=True) is None
    assert api.get_library_ops("mkl", load=True) is None


@pytest.mark.parametrize("loaded", [False, True])
def test_mkl_enabled_policy_applies_before_cache_or_load(api, monkeypatch, loaded):
    monkeypatch.setenv("use_mkl", "1")
    namespace = _mkl_bootstrap(api)
    library = types.SimpleNamespace(ops=object())
    calls = []
    def load():
        calls.append("load")
        api.register_library("mkl", library)
    namespace["setup_mkl"] = load
    if loaded:
        api.register_library("mkl", library)
    namespace["use_mkl"] = False
    assert api.get_library("mkl") is None
    assert api.get_library("mkl", load=True) is None
    assert api.get_library_ops("mkl", load=True) is None
    assert api.library_attribute("mkl_ops") is None
    assert not calls

    namespace["use_mkl"] = True
    monkeypatch.setenv("use_mkl", "0")
    assert api.get_library("mkl", load=True) is None
    assert not calls
    monkeypatch.setenv("use_mkl", "1")
    assert api.get_library("mkl", load=True) is library
    assert api.get_library_ops("mkl") is library.ops
    assert calls == ([] if loaded else ["load"])


def test_bootstrap_publishes_libraries_instead_of_module_globals(api):
    tree = ast.parse(EXTERN.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                assert not isinstance(target, ast.Name) or target.id not in api.LEGACY_LIBRARY_NAMES
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            assert set(node.names).isdisjoint(api.LEGACY_LIBRARY_NAMES)
    assert 'globals()[lib_name' not in EXTERN.read_text()
    assert 'root_module.mkl_ops' not in EXTERN.read_text()

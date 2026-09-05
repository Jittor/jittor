"""The fallback gate controls the native policy, not a separate Python model."""

import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_fallback():
    source = ROOT / "python/jittor/_runtime/fallback.py"
    spec = importlib.util.spec_from_file_location("_fallback_scope_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native(monkeypatch):
    state = SimpleNamespace(policy="warn", attempts=7, queries=0, scopes=[])

    def count():
        state.queries += 1
        return state.attempts

    @contextmanager
    def scope(**changes):
        assert changes == {"backend_fallback": "error"}
        previous = state.policy
        state.scopes.append(changes)
        state.policy = changes["backend_fallback"]
        try:
            yield
        finally:
            state.policy = previous

    def attempt():
        state.attempts += 1
        if state.policy == "error":
            raise RuntimeError("native backend fallback rejected")

    state.attempt = attempt
    fake = SimpleNamespace(core=SimpleNamespace(backend_fallback_count=count),
                           runtime=SimpleNamespace(scope=scope))
    monkeypatch.setitem(sys.modules, "jittor", fake)
    return state


def test_scope_changes_the_native_policy_and_restores_it(native):
    with _load_fallback().forbid_backend_fallbacks():
        assert native.policy == "error"
        assert native.queries == 1
    assert native.policy == "warn"
    assert native.queries == 2
    assert native.scopes == [{"backend_fallback": "error"}]


def test_scope_rejects_a_swallowed_native_fallback_exception(native):
    with pytest.raises(RuntimeError, match="attempted 1 time"):
        with _load_fallback().forbid_backend_fallbacks():
            try:
                native.attempt()
            except RuntimeError:
                pass
    assert native.attempts == 8
    assert native.policy == "warn"


def test_scope_rejects_an_allowed_attempt_after_inner_policy_override(native):
    with pytest.raises(RuntimeError, match="attempted 2 time"):
        with _load_fallback().forbid_backend_fallbacks():
            native.policy = "allow"
            native.attempt()
            native.attempt()
    assert native.policy == "warn"


@pytest.mark.parametrize("exception_type", [ValueError, RuntimeError, KeyboardInterrupt])
def test_primary_exception_propagates_without_a_counter_error(native, exception_type):
    primary = exception_type("primary failure")
    with pytest.raises(exception_type) as caught:
        with _load_fallback().forbid_backend_fallbacks():
            native.attempts += 1
            raise primary
    assert caught.value is primary
    assert native.queries == 1
    assert native.policy == "warn"


def test_nested_scopes_restore_the_outer_policy(native):
    scope = _load_fallback().forbid_backend_fallbacks
    with scope():
        with scope():
            assert native.policy == "error"
        assert native.policy == "error"
    assert native.policy == "warn"
    assert native.queries == 4


def test_loading_the_scope_does_not_import_jittor(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "jittor" or name.startswith("jittor."):
            raise AssertionError("fallback module imported native runtime")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert callable(_load_fallback().forbid_backend_fallbacks)


def test_hardware_sessions_force_native_error_policy():
    tree = ast.parse((ROOT / "noxfile.py").read_text())
    configure = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_set_hardware_python_config")
    assignments = [node for node in ast.walk(configure) if isinstance(node, ast.Assign)]
    def sets_error_policy(node):
        target = node.targets[0]
        if not (isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name) and target.value.id == "env"):
            return False
        key = target.slice.value if isinstance(target.slice, ast.Index) else target.slice
        return (ast.literal_eval(key) == "backend_fallback"
                and ast.literal_eval(node.value) == "error")

    assert any(sets_error_policy(node) for node in assignments)
    for session in ("cuda", "npu", "rocm", "nccl", "benchmark_cuda"):
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == session)
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "_set_hardware_python_config"
                   for node in ast.walk(function)), session


@pytest.mark.parametrize("already_failed", [False, True])
def test_npu_fixture_detects_swallowed_attempts_without_masking_primary_failure(
        native, monkeypatch, already_failed):
    module = _load_fallback()
    monkeypatch.setitem(sys.modules, "jittor._runtime.fallback", module)
    path = ROOT / "tests/backends/npu/conftest.py"
    spec = importlib.util.spec_from_file_location("_npu_fallback_fixture_test", path)
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    node = SimpleNamespace()
    fixture = conftest._forbid_backend_fallbacks.__wrapped__(SimpleNamespace(node=node))
    next(fixture)
    assert native.policy == "error"
    native.attempts += 1
    if already_failed:
        primary = ValueError("primary test failure")
        node._npu_fallback_excinfo = (ValueError, primary, None)
        with pytest.raises(StopIteration):
            next(fixture)
        assert not hasattr(node, "_npu_fallback_excinfo")
    else:
        with pytest.raises(RuntimeError, match="attempted 1 time"):
            next(fixture)
    assert native.policy == "warn"


@pytest.mark.parametrize("bootstrap_fails", [False, True])
def test_npu_fixture_does_not_bypass_policy_before_native_import(
        native, monkeypatch, bootstrap_fails):
    import builtins

    module = _load_fallback()
    fake_native = sys.modules["jittor"]
    monkeypatch.delitem(sys.modules, "jittor")
    path = ROOT / "tests/backends/npu/conftest.py"
    spec = importlib.util.spec_from_file_location("_npu_lazy_fixture_test", path)
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    original_import = builtins.__import__
    imports = []

    def import_fake_native(name, *args, **kwargs):
        if name == "jittor._runtime.fallback":
            imports.append(name)
            return module
        if name == "jittor":
            imports.append(name)
            if bootstrap_fails:
                raise RuntimeError("native bootstrap failed")
            monkeypatch.setitem(sys.modules, "jittor", fake_native)
            return fake_native
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_fake_native)
    request = SimpleNamespace(node=SimpleNamespace())
    fixture = conftest._forbid_backend_fallbacks.__wrapped__(request)
    assert "jittor" not in sys.modules
    if bootstrap_fails:
        with pytest.raises(RuntimeError, match="native bootstrap failed"):
            next(fixture)
        assert native.scopes == []
    else:
        next(fixture)
        assert native.policy == "error"
        assert native.scopes == [{"backend_fallback": "error"}]
        native.attempts += 1
        with pytest.raises(RuntimeError, match="attempted 1 time"):
            next(fixture)
        assert native.policy == "warn"
    assert imports == ["jittor._runtime.fallback", "jittor"]


def test_standalone_runners_use_native_scope_instead_of_log_wording():
    paths = (
        "tests/compat/torch/_ecosystem_runner.py",
        "tests/backends/npu/manual/run_qwen3_transformers.py",
        "agent/skills/jittor-transformers-perf/scripts/benchmark_qwen3_ascend.py",
    )
    for path in paths:
        tree = ast.parse((ROOT / path).read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                   and node.module == "jittor._runtime.fallback"]
        assert any(alias.name == "forbid_backend_fallbacks"
                   for node in imports for alias in node.names), path
        assert not any(isinstance(node, ast.Str)
                       and any(word in node.s for word in ("fallback cpu", "compile cpu"))
                       for node in ast.walk(tree)), path
        scopes = [node for node in ast.walk(tree) if isinstance(node, ast.With)
                  and any((isinstance(item.context_expr, ast.Name)
                           and item.context_expr.id == "fallback_scope")
                          or (isinstance(item.context_expr, ast.Call)
                              and isinstance(item.context_expr.func, ast.Name)
                              and item.context_expr.func.id in {
                                  "fallback_scope", "forbid_backend_fallbacks"})
                          for item in node.items)]
        assert scopes, path
        for scope in scopes:
            assert any(isinstance(node, ast.Call)
                       and ((isinstance(node.func, ast.Name)
                             and node.func.id in {"synchronize", "_synchronize"})
                            or (isinstance(node.func, ast.Attribute)
                                and node.func.attr in {"sync", "sync_all"}))
                       for node in ast.walk(scope)), path

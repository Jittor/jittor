"""Execute actual entry predicates/callbacks without importing native Jittor."""
import ast
from pathlib import Path
import sys
import types
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def _functions(path, names, namespace):
    source = ast.parse((ROOT / path).read_text())
    selected = [node for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def test_native_compat_request_uses_explicit_environment_only():
    placeholder = types.SimpleNamespace(_jittor_torch_shim_placeholder=True)
    namespace = {"sys": types.SimpleNamespace(modules={"torch": placeholder})}
    _functions("python/jittor/_runtime/compat_bootstrap.py", {"is_truthy", "_requested"}, namespace)
    requested = namespace["_requested"]
    assert not requested({})
    assert not requested({"JITTOR_TORCH_SHIM": "0"})
    assert requested({"JITTOR_TORCH_SHIM": "1"})
    assert requested({"JITTOR_TORCH_PROJECT_ROOT": "/explicit/project"})
    assert requested({"JITTOR_TORCH_RUNTIME_ROOT": "/explicit/runtime"})


def test_historical_alias_requests_independent_activation_without_native_install():
    namespace = {"sys": sys, "__package__": "jittor.compat"}
    _functions("compat/_aliases.py", {"_activate_torch_alias"}, namespace)
    native = types.ModuleType("jittor")
    compatibility = types.SimpleNamespace(install=mock.Mock())
    runtime = types.ModuleType("jittor.compat.shim.runtime")
    runtime.activate = mock.Mock()
    with mock.patch.dict(sys.modules, {"jittor": native, runtime.__name__: runtime}):
        namespace["_activate_torch_alias"](compatibility)
    compatibility.install.assert_not_called()
    runtime.activate.assert_called_once_with(_root_module=native, _composition=True, verbose=False)

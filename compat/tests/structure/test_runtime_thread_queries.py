"""Frontend thread observation delegates to the native runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def test_get_num_threads_observes_current_native_limit():
    path = ROOT / "torch/installers/utilities.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    getter = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_api_g_get_num_threads")
    query = mock.Mock(side_effect=[2, 5])
    namespace = {"jt": SimpleNamespace(core=SimpleNamespace(runtime_openmp_max_threads=query))}
    exec(compile(ast.Module(body=[getter], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace[getter.name]() == 2
    assert namespace[getter.name]() == 5
    assert query.call_count == 2

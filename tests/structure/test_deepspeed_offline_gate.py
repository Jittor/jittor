"""Execute the maintained structure session with a recording session."""
import ast
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def structure_calls(repo, posargs=()):
    tree = ast.parse((ROOT / "noxfile.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "structure"
    )
    function.decorator_list = []
    session = mock.Mock(posargs=posargs)
    namespace = {
        "nox": mock.Mock(), "REPO_ROOT": repo,
        "_session_env": lambda *args: (None, {}),
        "_install_compat_source": lambda *args: None,
        "_mode_env": lambda env, paths: env,
        "STRUCTURE_TESTS": ("tests/structure",),
        "NATIVE_MODE_PATHS": (),
    }
    for name in ("PYTEST", "PYTEST_TIMEOUT", "SETUPTOOLS", "SCIPY", "JUPYTEXT", "NBFORMAT"):
        namespace[name] = name
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<structure>", "exec"), namespace)
    namespace["structure"](session)
    return [call.args for call in session.run.call_args_list]


def test_default_structure_executes_adapter_contract_entry(tmp_path):
    entry = tmp_path / "adapters/tests/test_deepspeed_offline.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("", encoding="utf-8")
    calls = structure_calls(tmp_path)
    assert ("python", str(entry), "-v") in calls


def test_structure_without_adapter_tree_has_no_adapter_dependency(tmp_path):
    calls = structure_calls(tmp_path)
    assert calls
    assert all("deepspeed_offline" not in str(call) for call in calls)


def test_targeted_structure_selection_does_not_force_adapter_suite(tmp_path):
    entry = tmp_path / "adapters/tests/test_deepspeed_offline.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("", encoding="utf-8")
    calls = structure_calls(tmp_path, ("tests/structure/test_example.py",))
    assert all("deepspeed_offline" not in str(call) for call in calls)

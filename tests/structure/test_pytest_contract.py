"""Filesystem-level tests for the repository pytest contract."""

from __future__ import print_function

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock


try:
    import tomllib
except ImportError:
    from setuptools._vendor import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "tests"

_ALLOWED_COLLECTION_GENERATORS = {
    ("backends/parity/test_device_parity.py", "_install"),
    ("compiler/test_jit_tests.py", "_install_jit_tests"),
}
_PURE_COLLECTION_QUERIES = {
    "bool",
    "getattr",
    "hasattr",
    "importlib.util.find_spec",
    "modules_available",
}
_PROHIBITED_COLLECTION_IMPORT_PREFIXES = (
    "jittor.compat.triton",
    "jittor.triton_shim",
    "triton",
)
_PROHIBITED_COLLECTION_CALLS = {
    "eval",
    "exec",
    "jittor.dirty_fix_pytorch_runtime_error",
    "jittor.compat.triton.install",
    "jittor.triton_shim.install",
    "jt.dirty_fix_pytorch_runtime_error",
    "jt.compiler.run_cmd",
    "jittor.compiler.run_cmd",
    "open",
    "os.makedirs",
    "os.mkdir",
    "os.popen",
    "os.system",
    "pytest.skip",
    "triton_shim.install",
}
_PROHIBITED_COLLECTION_CALL_SUFFIXES = (".exec_module", ".mkdir")


def _test_files():
    return sorted(TEST_ROOT.rglob("test_*.py"))


def _dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return (prefix + "." if prefix else "") + node.attr
    if isinstance(node, ast.Subscript):
        return _dotted_name(node.value) + "[]"
    return ""


def _is_main_guard(node):
    return (
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "__name__"
    )


def _runtime_nodes(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    if isinstance(node, ast.If) and _is_main_guard(node.test):
        for child in node.orelse:
            yield from _runtime_nodes(child)
        return
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _runtime_nodes(child)


def _assignment_targets(node):
    if isinstance(node, ast.Assign):
        return node.targets
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return [node.target]
    return []


def _assignment_value(node):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        return node.value
    return None


def _pytest_config():
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["tool"]["pytest"]["ini_options"]


def _load_test_conftest():
    path = TEST_ROOT / "conftest.py"
    spec = importlib.util.spec_from_file_location("jittor_test_conftest_contract", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _automatic_markers(relative, device=None, selected=None):
    module = _load_test_conftest()

    class Item:
        fspath = TEST_ROOT / relative
        cls = type("GeneratedTest", (), {"device_type": device}) if device else None

        def __init__(self):
            self.markers = []

        def add_marker(self, marker):
            self.markers.append(marker.name)

        @staticmethod
        def get_closest_marker(_name):
            return None

    class Config:
        """Minimal stand-in for the pytest config the hook now reads."""

        @staticmethod
        def getoption(name):
            assert name == "--network", name
            return False

    previous = os.environ.pop("JITTOR_TEST_DEVICES", None)
    if selected is not None:
        os.environ["JITTOR_TEST_DEVICES"] = selected
    try:
        item = Item()
        module.pytest_collection_modifyitems(Config(), [item])
        return set(item.markers)
    finally:
        os.environ.pop("JITTOR_TEST_DEVICES", None)
        if previous is not None:
            os.environ["JITTOR_TEST_DEVICES"] = previous


def test_pytest_owns_collection_and_strict_xfail_policy():
    config = _pytest_config()
    assert config["testpaths"] == ["tests"]
    assert config["xfail_strict"] is True
    markers = {entry.split(":", 1)[0] for entry in config["markers"]}
    assert markers == {
        "structure",
        "cpu",
        "cuda",
        "rocm",
        "npu",
        "mpi",
        "slow",
        "network",
        "manual",
    }


def test_automatic_backend_markers_follow_device_sessions():
    dynamic = "ops/test_ops.py"
    assert _automatic_markers(dynamic, "cpu", "cpu") == {"cpu"}
    assert _automatic_markers(dynamic, "cuda", "cuda") == {"cuda"}
    assert _automatic_markers(dynamic, "npu", "npu") == {"npu"}
    assert _automatic_markers(dynamic, "cuda", "cpu,cuda") == {"cuda"}

    parity = _automatic_markers("backends/parity/test_device_parity.py")
    triton = _automatic_markers("backends/triton/test_triton_backend.py")
    assert parity == {"cuda", "npu"}
    assert triton == {"cuda"}
    assert "cpu" not in parity | triton

    external = REPO_ROOT / "agent" / "scripts" / "test_check_wheel_contents.py"
    assert _automatic_markers(external) == set()


def test_torch_semantic_core_suites_run_in_the_torch_process():
    module = _load_test_conftest()
    required = {
        "tests/core/test_regression.py",
        "tests/core/test_type_system.py",
    }
    assert required <= set(module.TORCH_MODE_PATHS)


def test_network_access_is_explicitly_marked():
    path = TEST_ROOT / "compiler" / "test_trace_var.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {
        node.name: {_dotted_name(decorator) for decorator in node.decorator_list}
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "pytest.mark.network" in functions["test_resnet_infer_with_feature"]


def test_optional_dependency_probe_only_checks_top_level_packages(monkeypatch):
    from _helpers import torch_runtime

    monkeypatch.delenv("REAL_TORCH_SITE", raising=False)
    checked = []
    independent = SimpleNamespace(__name__="torch")

    def fake_find_spec(module_name):
        checked.append(module_name)
        return object()

    with mock.patch.dict(torch_runtime.sys.modules, {"torch": independent}):
        with mock.patch.object(torch_runtime.importlib.util, "find_spec", fake_find_spec):
            assert torch_runtime.modules_available(
                "torch.nn.functional", "torchvision.models", "torch.autograd"
            )
    assert checked == ["torchvision"]


def test_optional_dependency_probe_accepts_preloaded_independent_torch(monkeypatch):
    from _helpers import torch_runtime

    monkeypatch.delenv("REAL_TORCH_SITE", raising=False)
    independent = SimpleNamespace(__name__="torch")
    with mock.patch.dict(torch_runtime.sys.modules, {"torch": independent}):
        assert torch_runtime.modules_available("torch.nn")


def test_optional_dependency_probe_rejects_loaded_jittor_torch_alias(monkeypatch):
    from _helpers import torch_runtime

    monkeypatch.delenv("REAL_TORCH_SITE", raising=False)
    alias = SimpleNamespace(__name__="jittor")
    with mock.patch.dict(torch_runtime.sys.modules, {"torch": alias}):
        assert not torch_runtime.modules_available("torch.nn")


def test_optional_dependency_probe_rejects_discoverable_jittor_torch_stub(monkeypatch):
    from _helpers import torch_runtime

    monkeypatch.delenv("REAL_TORCH_SITE", raising=False)
    monkeypatch.delitem(torch_runtime.sys.modules, "torch", raising=False)
    source = """
import sys as _sys
_sys.modules[__name__]._jittor_torch_shim_placeholder = True
"""
    loader = SimpleNamespace(get_source=lambda _name: source)
    spec = SimpleNamespace(loader=loader)
    with mock.patch.object(torch_runtime.importlib.util, "find_spec", return_value=spec):
        assert not torch_runtime.modules_available("torch.autograd")


def test_optional_dependency_probe_rejects_a_deployed_shim_as_real_torch(monkeypatch, tmp_path):
    from _helpers import torch_runtime

    torch_init = tmp_path / "torch" / "__init__.py"
    torch_init.parent.mkdir()
    torch_init.write_text("", encoding="utf-8")
    monkeypatch.setenv("REAL_TORCH_SITE", str(tmp_path))
    shim = SimpleNamespace(
        __name__="torch",
        __file__=str(torch_init),
        _C=SimpleNamespace(),
    )
    with mock.patch.dict(torch_runtime.sys.modules, {"torch": shim}):
        assert not torch_runtime.modules_available("torch.autograd")


def test_optimizer_roundtrip_helper_is_not_collected_as_a_test():
    path = TEST_ROOT / "optim" / "test_optimizer_save_load.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_tests = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "test_optim" not in module_tests
    assert "_run_optimizer_roundtrip" in module_tests


def test_private_test_method_holders_are_plain_mixins():
    """A ``_Base`` with test methods must not inherit from a ``TestCase``.

    pytest collects every ``unittest.TestCase`` subclass regardless of a
    leading underscore, so an "abstract" ``_Base(SomeTestCase)`` runs its test
    methods once more, without whatever gates or ``device`` attributes the
    concrete subclasses add. The safe shape is a plain mixin: the private class
    holds only the methods, and each concrete class inherits from both the
    mixin and the ``TestCase``.
    """
    offenders = []
    for path in _test_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not node.name.startswith("_"):
                continue
            has_tests = any(
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name.startswith("test")
                for member in node.body
            )
            public_bases = [
                name
                for name in (_dotted_name(base) for base in node.bases)
                if name.split(".")[-1] not in ("object",)
                and not name.split(".")[-1].startswith("_")
            ]
            if has_tests and public_bases:
                offenders.append(
                    "{}::{} inherits {}".format(
                        path.relative_to(REPO_ROOT).as_posix(), node.name, public_bases
                    )
                )
    assert offenders == []


#: Module-level ``test_*`` functions whose parameters are genuine pytest
#: fixtures rather than caller-supplied arguments.
_PYTEST_FIXTURE_PARAMETERS = frozenset(
    ("tmp_path", "tmp_path_factory", "monkeypatch", "capsys", "capfd", "caplog", "request")
)


def test_module_level_helpers_are_not_named_like_tests():
    """A ``test_*`` helper that takes arguments is collected and then errors.

    pytest collects every module-level ``test_*`` function. When such a function
    is really a helper called from a TestCase method, pytest still collects it,
    fails to find fixtures for its parameters, and reports an error that looks
    like a broken test. Helpers belong under a ``check_*`` name.
    """
    offenders = []
    for path in sorted(TEST_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test"):
                continue
            arguments = node.args
            required = arguments.args[: len(arguments.args) - len(arguments.defaults)]
            unsatisfied = [
                argument.arg
                for argument in required
                if argument.arg not in _PYTEST_FIXTURE_PARAMETERS
            ]
            if unsatisfied:
                offenders.append(
                    "{}::{} requires {}".format(
                        path.relative_to(REPO_ROOT).as_posix(), node.name, unsatisfied
                    )
                )
    assert offenders == []


def test_legacy_packaged_runner_is_absent():
    legacy = REPO_ROOT / "python" / "jittor" / "test"
    assert not (legacy / "__main__.py").exists()
    assert not (legacy / "_runner.py").exists()


def test_legacy_numeric_selection_fails_loudly():
    env = os.environ.copy()
    env["test_skip_l"] = "10"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "tests/structure/test_pytest_contract.py",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    assert result.returncode != 0
    assert "legacy jittor.test selection variables are unsupported" in result.stdout


def test_test_modules_do_not_import_other_test_modules():
    test_files = _test_files()
    test_module_stems = {path.stem for path in test_files}
    violations = []
    for path in test_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [("." * node.level) + (node.module or "")]
            for module in imported:
                stripped = module.lstrip(".")
                parts = stripped.split(".") if stripped else []
                is_legacy = stripped == "jittor.test" or stripped.startswith("jittor.test.")
                targets_test = any(part in test_module_stems for part in parts)
                if is_legacy or targets_test:
                    relative = path.relative_to(TEST_ROOT)
                    violations.append("{}:{} imports {}".format(relative, node.lineno, module))
    assert not violations, "test modules must depend on _helpers/opinfo, not tests:\n" + "\n".join(
        violations
    )


def test_test_modules_avoid_collection_time_backend_side_effects():
    violations = []
    for path in _test_files():
        relative = path.relative_to(TEST_ROOT)
        relative_text = relative.as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        local_functions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for statement in tree.body:
            for node in _runtime_nodes(statement):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported = [node.module or ""]
                for module in imported:
                    if module.startswith(_PROHIBITED_COLLECTION_IMPORT_PREFIXES):
                        violations.append(
                            "{}:{} imports {} during collection".format(
                                relative_text, node.lineno, module
                            )
                        )
                for target in _assignment_targets(node):
                    name = _dotted_name(target)
                    if name.startswith(("jt.flags.", "jittor.flags.")):
                        violations.append(
                            "{}:{} writes {} during collection".format(
                                relative_text, node.lineno, name
                            )
                        )
                value = _assignment_value(node)
                if value is not None:
                    for call in ast.walk(value):
                        if not isinstance(call, ast.Call):
                            continue
                        name = _dotted_name(call.func)
                        allowed = (
                            name in _PURE_COLLECTION_QUERIES
                            or (
                                relative_text,
                                name,
                            )
                            in _ALLOWED_COLLECTION_GENERATORS
                        )
                        if name in local_functions and not allowed:
                            violations.append(
                                "{}:{} assigns result of local helper {} during collection".format(
                                    relative_text, call.lineno, name
                                )
                            )
                if isinstance(node, ast.Call):
                    name = _dotted_name(node.func)
                    prohibited = (
                        name in _PROHIBITED_COLLECTION_CALLS
                        or name.startswith("subprocess.")
                        or name.endswith(_PROHIBITED_COLLECTION_CALL_SUFFIXES)
                    )
                    if prohibited:
                        violations.append(
                            "{}:{} calls {} during collection".format(
                                relative_text, node.lineno, name
                            )
                        )
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    name = _dotted_name(node.value.func)
                    allowed = (relative_text, name) in _ALLOWED_COLLECTION_GENERATORS
                    if name in local_functions and not allowed:
                        violations.append(
                            "{}:{} invokes local helper {} during collection".format(
                                relative_text, node.lineno, name
                            )
                        )
    assert not violations, "collection must not execute tests or mutate backends:\n" + "\n".join(
        violations
    )

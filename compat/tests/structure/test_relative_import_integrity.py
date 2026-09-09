"""Every name a compat module imports from a sibling must actually exist there.

This is the shape of a defect that reached the branch twice in one day. A
cleanup pass reads a module, sees a name it does not use locally, and deletes
it -- but another module *re-exports* that name, so the import chain breaks at
a file the cleanup never opened. `6f9d7e7bc` ("polish installer imports")
removed `InstallContext` and `registry_for` from
`torch/installers/cuda/api.py`; `bindings.py` imports both from there and uses
`registry_for` three times, so every Torch-mode CUDA import died from then on.

Importing the tree would also catch it, but only where the whole toolchain is
present: the CUDA installer needs CUDA, and a test that skips on a CPU box is
exactly the gate entry that looks identical to a passing one. This check is
static, so it runs everywhere and needs no accelerator.

Deliberately conservative. A module whose namespace is built by a star import
or by writing into `globals()`/`vars()` cannot be resolved by reading its
syntax tree, so it is skipped rather than guessed at -- a false red here would
be paid for by the next person deleting the rule.
"""

import ast
import pathlib
import unittest

import pytest

pytestmark = pytest.mark.structure

_COMPAT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _provided_names(tree):
    """Names a module binds, plus the flags that make the answer unreliable."""
    names, star, dynamic = set(), False, False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    star = True
                else:
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("globals", "vars", "setattr"):
                dynamic = True
            elif isinstance(func, ast.Attribute) and func.attr in ("update", "setdefault"):
                dynamic = True
    return names, star, dynamic


def _resolve(source, level, module):
    base = source.parent
    for _ in range(level - 1):
        base = base.parent
    if module:
        base = base.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def unresolved_relative_imports(root):
    """``(source, line, target, name)`` for every relative import that cannot resolve."""
    broken = []
    for source in sorted(pathlib.Path(root).rglob("*.py")):
        parts = source.parts
        if "__pycache__" in parts or "tests" in parts:
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            target = _resolve(source, node.level, node.module or "")
            if target is None:
                continue
            try:
                target_tree = ast.parse(target.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            provided, star, dynamic = _provided_names(target_tree)
            if star or dynamic:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if target.name == "__init__.py":
                    package = target.parent
                    if (package / (alias.name + ".py")).is_file():
                        continue
                    if (package / alias.name / "__init__.py").is_file():
                        continue
                if alias.name not in provided:
                    broken.append((source, node.lineno, target, alias.name))
    return broken


class TestRelativeImportIntegrity(unittest.TestCase):
    def test_every_relative_import_resolves(self):
        broken = unresolved_relative_imports(_COMPAT_ROOT)
        report = [
            "{0}:{1} imports {2!r} from {3}, which does not define it".format(
                source.relative_to(_COMPAT_ROOT).as_posix(), line, name,
                target.relative_to(_COMPAT_ROOT).as_posix(),
            )
            for source, line, target, name in broken
        ]
        self.assertEqual(report, [], "\n".join(report))

    def test_the_rule_catches_a_deleted_re_export(self):
        """The historical shape, rebuilt: a name deleted from the module re-exporting it."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            package = pathlib.Path(directory) / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "api.py").write_text("def kept():\n    return 1\n", encoding="utf-8")
            (package / "bindings.py").write_text(
                "from .api import kept, registry_for\n", encoding="utf-8")

            broken = unresolved_relative_imports(package)
            self.assertEqual(
                [(source.name, name) for source, _line, _target, name in broken],
                [("bindings.py", "registry_for")],
            )

    def test_the_rule_does_not_fire_on_a_star_import(self):
        """A namespace built by ``import *`` is unreadable statically, so it is skipped."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            package = pathlib.Path(directory) / "pkg"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "api.py").write_text("from math import *\n", encoding="utf-8")
            (package / "bindings.py").write_text("from .api import sqrt\n", encoding="utf-8")

            self.assertEqual(unresolved_relative_imports(package), [])


if __name__ == "__main__":
    unittest.main()

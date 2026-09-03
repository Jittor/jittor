# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``python/jittor/utils`` is being emptied; what leaves must not come back.

Task 5.25. ``utils/`` held eleven files with no shared responsibility --
compiler resources, repository tooling, a 718-line PyTorch-source translator, a
Flask app whose launcher lived in ``tools/``, and a benchmark that ran its
measurement at import. It is not even a package: there is no ``__init__.py``, so
it is an implicit namespace directory shipped by one ``MANIFEST.in`` line, and
the repository-layout gate waves the whole directory through as a single entry.
Junk accumulated on the inside of the gate.

What is left has one responsibility, and this file states it as a rule rather
than a list: **a file may stay in ``utils/`` only while something outside
Python reaches it by a hard-coded path or module name.** Three do. Each such
test asserts both halves -- the file is there *and* the reference that pins it
is there, so removing a caller makes this go red and says the file may now move.

Static only.
"""

import ast
import unittest
from pathlib import Path

import jittor


PACKAGE = Path(jittor.__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DRAWER = PACKAGE / "utils"


def _text(path):
    return path.read_text(encoding="utf-8", errors="replace")


class TestWhatLeftTheDrawer(unittest.TestCase):
    """Each move, and the caller that had to move with it."""

    #: source name -> (new location, the caller that names it)
    _MOVED = {
        # The PyTorch-source translator is compatibility code, not a utility.
        "pytorch_converter.py": PACKAGE / "compat" / "pytorch_converter.py",
        # Its HTTP front end belongs next to the translator it exposes.
        "converter_server.py": PACKAGE / "compat" / "converter_server.py",
        # Things a user runs against their own model.
        "nvtx.py": PACKAGE / "tools" / "nvtx.py",
        "jtune.py": PACKAGE / "tools" / "jtune.py",
        # Things a maintainer runs against the checkout. These regenerate or
        # post-process repository content and must not ship in the wheel.
        "gen_pyi.py": REPO_ROOT / "tools" / "build" / "gen_pyi.py",
        "local_doc_builder.py": REPO_ROOT / "tools" / "docs" / "local_doc_builder.py",
        "bench_klo.py": REPO_ROOT / "tools" / "benchmarks" / "legacy" / "bench_klo.py",
    }

    def test_each_moved_file_is_at_its_new_address_and_not_the_old_one(self):
        for name, destination in sorted(self._MOVED.items()):
            with self.subTest(name=name):
                self.assertTrue(destination.is_file(), destination)
                self.assertFalse((DRAWER / name).exists())

    def test_nothing_imports_the_old_paths(self):
        # This file names the retired paths in order to forbid them, so it is
        # the one file the scan has to skip -- otherwise the rule reports
        # itself and can never go green.
        self_path = Path(__file__).resolve()
        stale = tuple("jittor.utils." + name[:-len(".py")]
                      for name in self._MOVED)
        offenders = []
        for base in (PACKAGE, REPO_ROOT / "tests", REPO_ROOT / "tools",
                     REPO_ROOT / "docs", REPO_ROOT / "examples"):
            for path in sorted(base.rglob("*")):
                if path.suffix not in (".py", ".sh", ".md", ".pyi"):
                    continue
                if "__pycache__" in path.parts or path.resolve() == self_path:
                    continue
                text = _text(path)
                for name in stale:
                    if name in text:
                        offenders.append("%s -> %s" % (path, name))
        self.assertEqual(offenders, [])

    def test_the_converter_launcher_names_the_module_it_runs(self):
        # The container installs the published wheel and names the module in
        # FLASK_APP, so the script and the package have to agree.
        script = _text(REPO_ROOT / "tools" / "services" / "legacy"
                       / "converter_server.sh")
        self.assertIn("FLASK_APP=jittor.compat.converter_server", script)

    def test_the_user_tools_package_costs_nothing_to_have(self):
        # jittor.tools.nvtx dlopens the NVTX library at import; importing
        # jittor must not pay for that, so the package body stays empty.
        tree = ast.parse(_text(PACKAGE / "tools" / "__init__.py"))
        self.assertEqual(
            [node for node in tree.body
             if not isinstance(node, ast.Expr)], [],
            "jittor/tools/__init__.py must stay a docstring and nothing else")

    def test_the_repository_tools_are_not_shipped_in_the_wheel(self):
        # They regenerate committed files; a user's site-packages is the wrong
        # place to run them from, and gen_pyi.py in particular writes back into
        # the checkout.
        from setuptools import find_packages
        if not (REPO_ROOT / "pyproject.toml").is_file():
            self.skipTest("packaging metadata is only available in a checkout")
        packages = find_packages(where=str(REPO_ROOT / "python"))
        self.assertIn("jittor.tools", packages)
        self.assertNotIn("tools", packages)
        self.assertNotIn("tools.build", packages)

    def test_the_repository_tools_only_run_from_main(self):
        # bench_klo used to run a CUDA measurement at import, while the local
        # documentation helper changed into one developer's home directory.
        # Standalone repository tools may define helpers at import, but their
        # work belongs behind an explicit __main__ guard.
        scripts = (
            REPO_ROOT / "tools" / "benchmarks" / "legacy" / "bench_klo.py",
            REPO_ROOT / "tools" / "build" / "gen_pyi.py",
            REPO_ROOT / "tools" / "docs" / "local_doc_builder.py",
        )
        for path in scripts:
            tree = ast.parse(_text(path))
            with self.subTest(path=path):
                side_effects = []
                for node in tree.body:
                    if isinstance(node, (ast.Import, ast.ImportFrom,
                                         ast.FunctionDef, ast.If)):
                        continue
                    if isinstance(node, ast.Expr) \
                            and isinstance(node.value, (ast.Str, ast.Constant)):
                        continue
                    if isinstance(node, ast.Assign) \
                            and isinstance(node.value, (ast.Str, ast.Constant)):
                        continue
                    side_effects.append(type(node).__name__)
                self.assertEqual(side_effects, [])
                guards = [node for node in tree.body if isinstance(node, ast.If)]
                self.assertEqual(len(guards), 1)
                guard = guards[0].test
                self.assertIsInstance(guard, ast.Compare)
                self.assertIsInstance(guard.left, ast.Name)
                self.assertEqual(guard.left.id, "__name__")
                self.assertEqual(guard.comparators[0].s, "__main__")


class TestWhatStaysAndTheReferenceThatPinsIt(unittest.TestCase):
    """A file may stay only while something outside Python names its path."""

    def _pinned_by(self, name, *references):
        """``name`` stays in the drawer, and every reference still names it.

        ``assertIn`` on a whole C++ file dumps the file into the failure, so
        the containment check is done separately and reported by location.
        """
        self.assertTrue((DRAWER / name).is_file(), name)
        for relative, needle in references:
            path = (PACKAGE / relative) if relative != "compiler.py" \
                else (PACKAGE / "compiler.py")
            with self.subTest(reference=relative):
                self.assertTrue(
                    needle in _text(path),
                    "%s no longer contains %r: %s may now be free to move "
                    "(task 3.18), and this rule needs rewriting"
                    % (relative, needle, name))

    def test_dlink_compiler_is_pinned_by_the_jit_compiler_command_line(self):
        self._pinned_by(
            "dlink_compiler.py",
            ("src/jit_compiler.cc", 'jittor_path+"/utils/dlink_compiler.py'))

    def test_dumpdef_is_reached_from_the_installed_package_not_the_checkout(self):
        # `agent/design/target-layout.md` lists dumpdef.py under the repository
        # `tools/`. It cannot go there: compiler.py builds the .def file during
        # an *extension build on Windows*, from `jittor_path` -- the installed
        # package. The repository tools/ tree is sdist-only and absent from a
        # wheel, so that move would break Windows builds and nothing else.
        self._pinned_by(
            "dumpdef.py",
            ("compiler.py",
             'os.path.join(jittor_path, "utils", "dumpdef.py")'))

    def test_tracer_is_pinned_by_a_module_name_baked_into_the_core(self):
        # Not a path but an import: the core calls my_import with this dotted
        # name, so moving tracer.py is a C++ change like the other three.
        self._pinned_by(
            "tracer.py",
            ("src/pybind/py_var_tracer.cc",
             'my_import("jittor.utils.tracer", "fill_module_name")'))

    def test_the_drawer_holds_nothing_else(self):
        # Whatever is still in there, nothing NEW may be added: a file with no
        # home belongs in a package that states one.
        self.assertEqual(
            sorted(path.name for path in DRAWER.glob("*.py")),
            ["dlink_compiler.py", "dumpdef.py", "tracer.py"],
            "python/jittor/utils holds only files something outside Python "
            "reaches by hard-coded path (task 5.25). Put new code in a package "
            "whose name says what it is for.")

    def test_the_layout_document_still_pins_the_compiler_resources(self):
        doc = _text(REPO_ROOT / "docs" / "architecture" / "repository-layout.md")
        self.assertIn(
            "python/jittor/utils/{dlink_compiler.py,dumpdef.py}",
            doc,
            "if the contract moved, this test and 5.25's remaining half both "
            "need rewriting")


if __name__ == "__main__":
    unittest.main()

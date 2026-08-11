"""Architecture contracts for the canonical Triton compatibility domain."""

from __future__ import print_function

import ast
import importlib
from pathlib import Path
import sys
import unittest

import jittor


_CHILD_MODULES = ("backend", "deploy", "language", "launch")


class TestTritonStructure(unittest.TestCase):
    def test_canonical_package_owns_the_physical_implementation(self):
        canonical = importlib.import_module("jittor.compat.triton")
        package_path = Path(canonical.__file__).resolve().parent
        repo_root = package_path.parents[3]
        self.assertEqual(package_path.name, "triton")
        self.assertEqual(package_path.parent.name, "compat")
        self.assertEqual(
            {path.name for path in package_path.glob("*.py")},
            {"__init__.py", "backend.py", "deploy.py", "language.py", "launch.py"},
        )
        self.assertFalse((repo_root / "python" / "jittor" / "triton_shim").exists())

    def test_legacy_package_and_children_are_same_object_aliases(self):
        canonical = importlib.import_module("jittor.compat.triton")
        legacy = importlib.import_module("jittor.triton_shim")
        self.assertEqual(canonical.__name__, "jittor.compat.triton")
        self.assertEqual(canonical.__package__, "jittor.compat.triton")
        self.assertIs(legacy, canonical)
        self.assertIs(jittor.triton_shim, canonical)
        self.assertIs(sys.modules["jittor.triton_shim"], canonical)
        for child in _CHILD_MODULES:
            legacy_child = importlib.import_module("jittor.triton_shim." + child)
            canonical_child = importlib.import_module("jittor.compat.triton." + child)
            with self.subTest(child=child):
                self.assertEqual(
                    canonical_child.__name__, "jittor.compat.triton." + child
                )
                self.assertEqual(canonical_child.__package__, "jittor.compat.triton")
                self.assertIs(legacy_child, canonical_child)
                self.assertIs(
                    sys.modules["jittor.triton_shim." + child], canonical_child
                )

    def test_console_script_targets_canonical_deploy_module(self):
        canonical = importlib.import_module("jittor.compat.triton")
        repo_root = Path(canonical.__file__).resolve().parents[4]
        pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            'jittor-triton-shim = "jittor.compat.triton.deploy:main"',
            pyproject,
        )
        self.assertNotIn("jittor.triton_shim.deploy:main", pyproject)

    def test_repeated_install_preserves_shim_object_identity(self):
        canonical = importlib.import_module("jittor.compat.triton")
        names = (
            "triton", "triton.language", "triton.runtime",
            "triton.runtime.jit", "triton.runtime.autotuner",
        )
        missing = object()
        previous = {name: sys.modules.get(name, missing) for name in names}
        try:
            first = canonical.install(force=True)
            first_modules = {name: sys.modules[name] for name in names}
            second = canonical.install(force=True)
            self.assertIs(first, canonical)
            self.assertIs(second, canonical)
            for name, module in first_modules.items():
                with self.subTest(name=name):
                    self.assertIs(sys.modules[name], module)
        finally:
            for name, module in previous.items():
                if module is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_deploy_redirects_and_cli_target_are_canonical(self):
        deploy = importlib.import_module("jittor.compat.triton.deploy")
        self.assertIn("from jittor.compat.triton import *", deploy._INIT_BODY)
        self.assertIn(
            "from jittor.compat.triton.language import *", deploy._LANG_BODY
        )
        self.assertNotIn("jittor.triton_shim", deploy._INIT_BODY)
        self.assertNotIn("jittor.triton_shim", deploy._LANG_BODY)
        self.assertTrue(callable(deploy.main))

    def test_all_moved_sources_parse_as_python37(self):
        canonical = importlib.import_module("jittor.compat.triton")
        package_path = Path(canonical.__file__).resolve().parent
        for path in sorted(package_path.glob("*.py")):
            with self.subTest(path=path.name):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 7),
                )

    def test_package_discovery_contains_only_the_canonical_package(self):
        canonical = importlib.import_module("jittor.compat.triton")
        repo_root = Path(canonical.__file__).resolve().parents[4]
        if not (repo_root / "pyproject.toml").is_file():
            self.skipTest("package discovery requires a source checkout")
        from setuptools import find_packages

        packages = find_packages(where=str(repo_root / "python"))
        self.assertIn("jittor.compat.triton", packages)
        self.assertNotIn("jittor.triton_shim", packages)


if __name__ == "__main__":
    unittest.main(verbosity=2)

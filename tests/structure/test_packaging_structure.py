"""Source-checkout contracts for package discovery and runtime resources."""

import unittest
from pathlib import Path


class TestPackagingStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.python_root = cls.repo_root / "python"
        cls.pyproject_path = cls.repo_root / "pyproject.toml"
        if not cls.pyproject_path.is_file():
            raise unittest.SkipTest("packaging metadata requires a source checkout")

    def test_find_packages_matches_every_regular_package(self):
        from setuptools import find_packages

        expected = {
            path.parent.relative_to(self.python_root).as_posix().replace("/", ".")
            for path in self.python_root.rglob("__init__.py")
        }
        discovered = set(find_packages(where=str(self.python_root)))
        self.assertEqual(discovered, expected)

    def test_pyproject_uses_regular_package_discovery(self):
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                from setuptools._vendor import tomli as tomllib

        with self.pyproject_path.open("rb") as stream:
            config = tomllib.load(stream)
        discovery = config["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(discovery["where"], ["python"])
        self.assertIs(discovery["namespaces"], False)
        self.assertTrue(config["tool"]["setuptools"]["include-package-data"])

    def test_manifest_covers_runtime_trees_without_cache_payloads(self):
        manifest = (self.repo_root / "MANIFEST.in").read_text(encoding="utf-8")
        directives = {
            line.strip()
            for line in manifest.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("recursive-include python/jittor *", directives)
        self.assertIn("recursive-include python/jittor_utils *", directives)
        self.assertIn("recursive-include examples *", directives)
        self.assertIn("recursive-include tools *", directives)
        self.assertIn("global-exclude *.py[cod]", directives)
        self.assertIn("global-exclude __pycache__", directives)

    def test_root_development_trees_do_not_become_runtime_packages(self):
        for relative in ("examples", "tools"):
            root = self.repo_root / relative
            self.assertTrue(root.is_dir(), relative)
            self.assertFalse((root / "__init__.py").exists(), relative)

    def test_built_sdist_has_an_executable_contents_gate(self):
        checker = self.repo_root / "agent" / "scripts" / "check_sdist_contents.py"
        self.assertTrue(checker.is_file())

    def test_required_deep_runtime_resources_exist(self):
        required = (
            "python/jittor/torch_shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh",
            "python/jittor/torch_shim/stubs/flash_attn/flash_attn_interface.py",
            "python/jittor/utils/data.gz",
            "python/jittor/other/code_softmax.py",
        )
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((self.repo_root / relative).is_file())


if __name__ == "__main__":
    unittest.main()

"""Source-checkout contracts for package discovery and runtime resources."""

import unittest
import ast
import importlib.util
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
        backend_root = self.repo_root / "backends"
        backend_expected = {
            "jittor.backends." + path.parent.relative_to(backend_root).as_posix().replace("/", ".")
            for path in backend_root.rglob("__init__.py")
        }
        backend_discovered = {"jittor.backends." + name
                              for name in find_packages(where=str(backend_root))}
        self.assertEqual(backend_discovered, backend_expected)

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
        package_dirs = config["tool"]["setuptools"]["package-dir"]
        self.assertEqual(package_dirs[""], "python")
        for backend in ("cuda", "acl", "comm"):
            self.assertEqual(package_dirs["jittor.backends." + backend], "backends/" + backend)
        setup_tree = ast.parse((self.repo_root / "setup.py").read_text())
        discovery_roots = {ast.literal_eval(node.args[0]) for node in ast.walk(setup_tree)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                           and node.func.id == "find_packages"}
        self.assertEqual(discovery_roots, {"python", "backends"})
        self.assertTrue(config["tool"]["setuptools"]["include-package-data"])
        with (self.repo_root / "compat/pyproject.toml").open("rb") as stream:
            compat_config = tomllib.load(stream)
        self.assertEqual(compat_config["project"]["name"], "jittor-torch")
        self.assertNotIn("jittor-torch-shim", config["project"].get("scripts", {}))
        self.assertEqual(
            compat_config["project"]["scripts"]["jittor-torch-shim"],
            "jittor.compat.shim.deploy:main",
        )

    def test_manifest_covers_runtime_trees_without_cache_payloads(self):
        from importlib.util import module_from_spec, spec_from_file_location
        path = self.repo_root / "tools/build/generate_manifest.py"
        spec = spec_from_file_location("manifest_contract", path)
        generator = module_from_spec(spec)
        spec.loader.exec_module(generator)
        for project in (self.repo_root, self.repo_root / "compat"):
            self.assertEqual((project / "MANIFEST.in").read_text(),
                             generator.manifest_text(project))
        resources = generator.runtime_resources(self.repo_root)
        self.assertEqual(resources["src/core/common.h"], "jittor/src/core/common.h")
        self.assertEqual(resources["backends/cuda/include/helper_cuda.h"],
                         "jittor/backends/cuda/include/helper_cuda.h")
        self.assertIn("python/jittor/contrib/math_util/src/igamma.h", resources)
        self.assertFalse(any(path.startswith("compat/") for path in resources))
        compat_resources = generator.runtime_resources(self.repo_root / "compat")
        self.assertEqual(compat_resources["shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh"],
                         "jittor/compat/shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh")
        self.assertTrue(set(resources.values()).isdisjoint(compat_resources.values()))
        for source in (self.repo_root / "backends").rglob("*"):
            if source.is_file() and source.suffix in {".h", ".cc", ".cpp", ".cu", ".cuh"}:
                self.assertIn(source.relative_to(self.repo_root).as_posix(), resources)

    def test_generated_manifest_handles_spaces_and_dirty_source_caches(self):
        from importlib.util import module_from_spec, spec_from_file_location
        from tempfile import TemporaryDirectory
        from setuptools._distutils.filelist import FileList

        spec = spec_from_file_location("manifest_fixture", self.repo_root / "tools/build/generate_manifest.py")
        generator = module_from_spec(spec)
        spec.loader.exec_module(generator)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                '[tool.setuptools]\npackage-dir={demo="pkg"}\n'
                '[tool.setuptools.package-data]\ndemo=["assets/**/*"]\n'
                '[tool.jittor.sdist]\ninclude=["examples/**"]\nexclude=[]\n')
            files = ["setup.py", "MANIFEST.in", "pkg/assets/value.dat",
                     "pkg/assets/__pycache__/leak.dat", "examples/tutorial 1.md",
                     "examples/.pytest_cache/v/cache/data"]
            for relative in files:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            resources = generator.runtime_resources(root)
            self.assertEqual(resources, {"pkg/assets/value.dat": "demo/assets/value.dat"})
            manifest = generator.manifest_text(root)
            selected = FileList()
            selected.allfiles = files + ["pyproject.toml"]
            for line in manifest.splitlines():
                if line.startswith("include "):
                    selected.process_template_line(line)
            self.assertIn("examples/tutorial 1.md", selected.files)
            self.assertFalse(any("cache" in name for name in selected.files))
            (root / "pkg/assets/new.dat").write_text("new resource")
            self.assertIn("pkg/assets/new.dat", generator.runtime_resources(root))
            self.assertNotEqual(manifest, generator.manifest_text(root))

    def test_root_development_trees_do_not_become_runtime_packages(self):
        for relative in ("examples", "tools"):
            root = self.repo_root / relative
            self.assertTrue(root.is_dir(), relative)
            self.assertFalse((root / "__init__.py").exists(), relative)

    def test_built_sdist_has_an_executable_contents_gate(self):
        checker = self.repo_root / "tools" / "release" / "check_sdist_contents.py"
        self.assertTrue(checker.is_file())

    def test_required_deep_runtime_resources_exist(self):
        required = (
            "compat/shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh",
            "compat/shim/resources/stubs/flash_attn/flash_attn_interface.py",
            "compat/shim/resources/torch/__init__.py",
            "backends/cuda/kernels/nn/softmax_cuda.py",
            "backends/cuda/kernels/nn/group_norm_cuda.py",
            "backends/cuda/include/helper_cuda.h",
            "backends/cuda/libraries/cutt/include/cutt_wrapper.h",
            "python/jittor/tools/tracer.py",
        )
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((self.repo_root / relative).is_file())

    def test_wheel_audit_distinguishes_runtime_helpers_from_build_artifacts(self):
        path = self.repo_root / "tools/release/check_wheel_contents.py"
        spec = importlib.util.spec_from_file_location("wheel_layout_contract", path)
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        for name in ("__init__.py", "dlink_compiler.py", "dumpdef.py"):
            self.assertIsNone(checker._pollution_reason("jittor/build/" + name))
        self.assertIsNotNone(checker._pollution_reason("jittor/build/temp.o"))
        self.assertIsNotNone(checker._pollution_reason("build/jittor/core.py"))


if __name__ == "__main__":
    unittest.main()

"""Architecture contracts for the canonical Triton compatibility domain."""

from __future__ import print_function

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

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
            for name in names:
                sys.modules.pop(name, None)
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

    def test_force_install_rejects_foreign_tree_without_partial_writes(self):
        canonical = importlib.import_module("jittor.compat.triton")
        names = (
            "triton", "triton.language", "triton.runtime",
            "triton.runtime.jit", "triton.runtime.autotuner",
        )
        with mock.patch.dict(sys.modules, {}, clear=False):
            for name in names:
                sys.modules.pop(name, None)
            foreign_root = types.ModuleType("triton")
            foreign_runtime = types.ModuleType("triton.runtime")
            sys.modules["triton"] = foreign_root
            sys.modules["triton.runtime"] = foreign_runtime
            before = {
                name: sys.modules[name] for name in names if name in sys.modules
            }
            with self.assertRaisesRegex(RuntimeError, "preloaded Triton"):
                canonical.install(force=True)
            after = {
                name: sys.modules[name] for name in names if name in sys.modules
            }
            self.assertEqual(after, before)

    def test_force_install_rejects_forged_deploy_marker(self):
        canonical = importlib.import_module("jittor.compat.triton")
        names = (
            "triton", "triton.language", "triton.runtime",
            "triton.runtime.jit", "triton.runtime.autotuner",
        )
        with tempfile.TemporaryDirectory() as target:
            package = Path(target) / "triton"
            package.mkdir()
            source = package / "__init__.py"
            source.write_text(
                "__jittor_triton_shim__ = True\n# forged redirect\n",
                encoding="utf-8",
            )
            forged = types.ModuleType("triton")
            forged.__jittor_triton_shim__ = True
            forged.__file__ = os.fspath(source)
            with mock.patch.dict(sys.modules, {}, clear=False):
                for name in names:
                    sys.modules.pop(name, None)
                sys.modules["triton"] = forged
                before = {"triton": forged}
                with self.assertRaisesRegex(RuntimeError, "preloaded Triton"):
                    canonical.install(force=True)
                after = {
                    name: sys.modules[name]
                    for name in names
                    if name in sys.modules
                }
                self.assertEqual(after, before)

    def test_force_install_rejects_marker_without_source_file(self):
        canonical = importlib.import_module("jittor.compat.triton")
        names = (
            "triton", "triton.language", "triton.runtime",
            "triton.runtime.jit", "triton.runtime.autotuner",
        )
        with mock.patch.dict(sys.modules, {}, clear=False):
            for name in names:
                sys.modules.pop(name, None)
            foreign = types.ModuleType("triton")
            foreign.__jittor_triton_shim__ = True
            foreign.__file__ = None
            sys.modules["triton"] = foreign
            with self.assertRaisesRegex(RuntimeError, "preloaded Triton"):
                canonical.install(force=True)
            self.assertEqual(
                {name: sys.modules[name] for name in names if name in sys.modules},
                {"triton": foreign},
            )

    def test_nonforce_real_bridge_preserves_foreign_tree(self):
        canonical = importlib.import_module("jittor.compat.triton")
        real = types.ModuleType("triton")
        child = types.ModuleType("triton.runtime")
        with mock.patch.dict(
            sys.modules, {"triton": real, "triton.runtime": child}, clear=False
        ), mock.patch.object(
            canonical, "_detect_real_triton", return_value=real
        ), mock.patch.object(canonical, "activate_bridge") as bridge:
            self.assertIs(canonical.install(force=False), real)
            self.assertIs(sys.modules["triton"], real)
            self.assertIs(sys.modules["triton.runtime"], child)
        bridge.assert_called_once_with(real)

    def test_deploy_redirects_and_cli_target_are_canonical(self):
        deploy = importlib.import_module("jittor.compat.triton.deploy")
        self.assertIn("from jittor.compat.triton import *", deploy._INIT_BODY)
        self.assertIn(
            "from jittor.compat.triton.language import *", deploy._LANG_BODY
        )
        self.assertNotIn("jittor.triton_shim", deploy._INIT_BODY)
        self.assertNotIn("jittor.triton_shim", deploy._LANG_BODY)
        self.assertTrue(callable(deploy.main))

    def test_deployed_redirect_cold_import_converges_to_canonical_modules(self):
        canonical = importlib.import_module("jittor.compat.triton")
        deploy = importlib.import_module("jittor.compat.triton.deploy")
        python_root = Path(canonical.__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as target:
            deploy.deploy(target=target)
            env = os.environ.copy()
            pythonpath = [target, os.fspath(python_root)]
            if env.get("PYTHONPATH"):
                pythonpath.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["nvcc_path"] = ""
            env["JITTOR_TORCH_KEEP_CUDA"] = "1"
            code = (
                "import sys\n"
                "import triton\n"
                "import triton.language as tl\n"
                "import jittor\n"
                "from jittor.compat import triton as canonical\n"
                "assert triton is canonical\n"
                "assert tl is canonical.language\n"
                "assert sys.modules['triton'] is canonical\n"
                "assert sys.modules['triton.language'] is canonical.language\n"
                "assert jittor.triton_shim is canonical\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertEqual(result.returncode, 0, result.stdout)

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

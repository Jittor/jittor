"""Filesystem-only tests for the torch-shim deployment helper."""

from __future__ import print_function

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


# Loading the leaf file avoids importing jittor and starting its JIT runtime.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "python"
    / "jittor"
    / "compat"
    / "shim"
    / "deploy.py"
)
deploy_module = None


def setUpModule():
    global deploy_module
    spec = importlib.util.spec_from_file_location(
        "_jittor_torch_shim_deploy_test", _MODULE_PATH
    )
    deploy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy_module)


class TestTorchShimDeploy(unittest.TestCase):
    def _deployed_target(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        target = Path(temporary_directory.name) / "site packages"
        deploy_module.deploy(target)
        return target

    def test_deploys_every_stub_python_file_recursively(self):
        target = self._deployed_target()
        stub_root = Path(deploy_module._RESOURCES) / "stubs"
        expected = sorted(
            path.relative_to(stub_root)
            for path in stub_root.rglob("*.py")
            if path.parent != stub_root and path.is_file() and not path.is_symlink()
        )

        self.assertIn(Path("flash_attn/flash_attn_interface.py"), expected)
        self.assertFalse((target / "__init__.py").exists())
        for relative in expected:
            with self.subTest(relative=str(relative)):
                deployed = target / relative
                self.assertTrue(deployed.is_file())
                self.assertEqual(deployed.read_bytes(), (stub_root / relative).read_bytes())

        checked_target, problems = deploy_module.check(target)
        self.assertEqual(Path(checked_target), target.resolve())
        self.assertEqual(problems, [])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = deploy_module.main(["--check", "--target", str(target)])
        self.assertEqual(status, 0)
        self.assertIn("all files present and match source", output.getvalue())

    def test_recursive_plan_handles_nested_stub_packages(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        base = Path(temporary_directory.name)
        source = base / "source"
        target = base / "target"
        (source / "stubs" / "example" / "nested").mkdir(parents=True)
        (source / "torch_dist_info").mkdir()
        (source / "torch_init.py").write_text("shim = True\n", encoding="utf-8")
        (source / "torch_dist_info" / "METADATA").write_text(
            "Name: torch\nVersion: 9.9.0\n", encoding="utf-8"
        )
        for relative in (
            "example/__init__.py",
            "example/helpers.py",
            "example/nested/__init__.py",
            "example/nested/api.py",
        ):
            path = source / "stubs" / relative
            path.write_text("SOURCE = %r\n" % relative, encoding="utf-8")
        (source / "stubs" / "example" / "ignored.txt").write_text(
            "not Python\n", encoding="utf-8"
        )

        with mock.patch.object(deploy_module, "_RESOURCES", str(source)):
            deploy_module.deploy(target)
            planned = {
                Path(destination).relative_to(target.resolve())
                for _source, destination in deploy_module._plan(target)
            }
            self.assertIn(Path("example/nested/api.py"), planned)
            self.assertNotIn(Path("example/ignored.txt"), planned)
            self.assertEqual(deploy_module.check(target)[1], [])

    def test_check_detects_modified_and_missing_files_but_allows_extras(self):
        target = self._deployed_target()
        modified = target / "flash_attn" / "flash_attn_interface.py"
        missing = target / "torchaudio" / "__init__.py"
        extra = target / "torchvision" / "unexpected.py"
        modified.write_text("tampered = True\n", encoding="utf-8")
        missing.unlink()
        extra.write_text("unexpected = True\n", encoding="utf-8")
        unrelated = target / "unrelated" / "keep.py"
        unrelated.parent.mkdir()
        unrelated.write_text("keep = True\n", encoding="utf-8")

        checked_target, problems = deploy_module.check_details(target)
        problem_map = {Path(path): kind for kind, path in problems}
        self.assertEqual(Path(checked_target), target.resolve())
        self.assertEqual(problem_map[modified], "modified")
        self.assertEqual(problem_map[missing], "missing")
        self.assertNotIn(extra, problem_map)
        self.assertNotIn(unrelated, problem_map)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = deploy_module.main(["--check", "--target", str(target)])
        self.assertEqual(status, 1)
        self.assertIn("modified: flash_attn/flash_attn_interface.py", output.getvalue())
        self.assertIn("missing: torchaudio/__init__.py", output.getvalue())

    def test_repeated_deploy_is_byte_identical(self):
        target = self._deployed_target()
        first = {
            Path(destination).relative_to(target.resolve()): Path(destination).read_bytes()
            for _source, destination in deploy_module._plan(target)
        }

        deploy_module.deploy(target)

        second = {
            Path(destination).relative_to(target.resolve()): Path(destination).read_bytes()
            for _source, destination in deploy_module._plan(target)
        }
        self.assertEqual(second, first)
        self.assertEqual(deploy_module.check_details(target)[1], [])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_deploy_rejects_symlinked_managed_package(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        base = Path(temporary_directory.name)
        target = base / "target"
        outside = base / "outside"
        target.mkdir()
        outside.mkdir()
        sentinel = outside / "__init__.py"
        sentinel.write_text("outside = True\n", encoding="utf-8")
        try:
            os.symlink(str(outside), str(target / "flash_attn"))
        except OSError as error:
            self.skipTest("cannot create symlink: %s" % error)

        _checked_target, problems = deploy_module.check_details(target)
        self.assertIn("unsafe", {kind for kind, _path in problems})
        with self.assertRaisesRegex(RuntimeError, r"unsafe.*\(symlink\)"):
            deploy_module.deploy(target)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "outside = True\n")
        self.assertFalse((target / "torch" / "__init__.py").exists())

    def test_deploy_preflight_rejects_regular_file_parent_without_partial_write(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        target = Path(temporary_directory.name) / "target"
        target.mkdir()
        (target / "torchvision").write_text("not a directory\n", encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, r"unsafe.*\(non-directory\)"):
            deploy_module.deploy(target)
        self.assertFalse((target / "torch" / "__init__.py").exists())

    def test_plan_requires_stub_packages_and_metadata(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        source = Path(temporary_directory.name) / "source"
        source.mkdir()
        (source / "torch_init.py").write_text("shim = True\n", encoding="utf-8")

        with mock.patch.object(deploy_module, "_RESOURCES", str(source)):
            with self.assertRaisesRegex(RuntimeError, "stubs directory"):
                deploy_module._plan(source / "target")

            package = source / "stubs" / "example"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("stub = True\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "torch metadata"):
                deploy_module._plan(source / "target")

    def test_plan_rejects_any_first_level_stub_directory_without_initializer(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        source = Path(temporary_directory.name) / "source"
        complete = source / "stubs" / "complete"
        incomplete = source / "stubs" / "incomplete"
        metadata = source / "torch_dist_info" / "METADATA"
        complete.mkdir(parents=True)
        incomplete.mkdir()
        metadata.parent.mkdir()
        (source / "torch_init.py").write_text("shim = True\n", encoding="utf-8")
        (complete / "__init__.py").write_text("stub = True\n", encoding="utf-8")
        (incomplete / "api.py").write_text("stub = False\n", encoding="utf-8")
        metadata.write_text("Name: torch\nVersion: 9.9.0\n", encoding="utf-8")

        with mock.patch.object(deploy_module, "_RESOURCES", str(source)):
            with self.assertRaisesRegex(RuntimeError, r"missing __init__\.py"):
                deploy_module._plan(source / "target")

    def test_plan_ignores_install_generated_stub_bytecode_cache(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        source = Path(temporary_directory.name) / "source"
        package = source / "stubs" / "example"
        bytecode_cache = source / "stubs" / "__pycache__"
        metadata = source / "torch_dist_info" / "METADATA"
        package.mkdir(parents=True)
        bytecode_cache.mkdir()
        metadata.parent.mkdir()
        (source / "torch_init.py").write_text("shim = True\n", encoding="utf-8")
        (package / "__init__.py").write_text("stub = True\n", encoding="utf-8")
        (bytecode_cache / "__init__.cpython-311.pyc").write_bytes(b"bytecode")
        metadata.write_text("Name: torch\nVersion: 9.9.0\n", encoding="utf-8")

        with mock.patch.object(deploy_module, "_RESOURCES", str(source)):
            planned = deploy_module._plan(source / "target")

        destinations = [destination for _source, destination in planned]
        self.assertTrue(any("example" in destination for destination in destinations))
        self.assertFalse(any("__pycache__" in destination for destination in destinations))

    def test_deploy_rejects_source_destination_alias_without_partial_write(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        source = Path(temporary_directory.name) / "source"
        package = source / "stubs" / "example"
        metadata = source / "torch_dist_info" / "METADATA"
        package.mkdir(parents=True)
        metadata.parent.mkdir()
        (source / "torch_init.py").write_text("shim = True\n", encoding="utf-8")
        package_init = package / "__init__.py"
        package_init.write_text("stub = True\n", encoding="utf-8")
        metadata.write_text("Name: torch\nVersion: 9.9.0\n", encoding="utf-8")

        with mock.patch.object(deploy_module, "_RESOURCES", str(source)):
            _checked_target, problems = deploy_module.check_details(source / "stubs")
            problem_map = {Path(path): kind for kind, path in problems}
            self.assertEqual(problem_map[package_init], "unsafe")
            with self.assertRaisesRegex(RuntimeError, r"unsafe.*\(same-file\)"):
                deploy_module.deploy(source / "stubs")
        self.assertEqual(package_init.read_text(encoding="utf-8"), "stub = True\n")
        self.assertFalse((source / "stubs" / "torch" / "__init__.py").exists())

    def test_explicit_relative_target_preserves_public_return_shape(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        previous_directory = os.getcwd()
        os.chdir(temporary_directory.name)
        try:
            requested_target = Path("relative site")
            returned_target, deployed = deploy_module.deploy(requested_target)
            self.assertIs(returned_target, requested_target)
            self.assertTrue(deployed)
            self.assertTrue(all(not os.path.isabs(path) for path in deployed))
            self.assertTrue(
                all(Path(path).parts[0] == str(requested_target) for path in deployed)
            )

            missing = requested_target / "torchaudio" / "__init__.py"
            missing.unlink()
            checked_target, problems = deploy_module.check(requested_target)
            self.assertIs(checked_target, requested_target)
            self.assertIn(str(missing), problems)
            self.assertTrue(all(not os.path.isabs(path) for path in problems))
        finally:
            os.chdir(previous_directory)

    def test_cli_rejects_missing_target_value(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            at_end = deploy_module.main(["--target"])
            before_check = deploy_module.main(["--target", "--check"])
        self.assertEqual(at_end, 2)
        self.assertEqual(before_check, 2)
        self.assertIn("--target requires a path", output.getvalue())

    def test_destination_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as target:
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                deploy_module._destination(target, "..", "escape.py")


if __name__ == "__main__":
    unittest.main()

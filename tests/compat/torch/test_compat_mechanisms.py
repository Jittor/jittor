"""Focused contracts for compatibility mechanisms shared with optional packages."""

import importlib
import json
import os
from pathlib import Path
import pkgutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
import uuid
from unittest import mock

from jittor.compat import external_backend
from jittor.compat import module_patcher


class _EntryPoint:
    def __init__(self, name, value, loaded):
        self.name = name
        self.value = value
        self._loaded = loaded

    def load(self):
        if isinstance(self._loaded, BaseException):
            raise self._loaded
        return self._loaded


class TestModulePatcher(unittest.TestCase):
    def _module_name(self, label):
        return "_jittor_%s_%s" % (label, uuid.uuid4().hex)

    def test_patches_modules_loaded_before_and_after_install(self):
        loaded_name = self._module_name("loaded_patch")
        future_name = self._module_name("future_patch")
        loaded = ModuleType(loaded_name)
        sys.modules[loaded_name] = loaded
        calls = []

        def apply(module):
            calls.append(module.__name__)
            module.patched = True
            return True

        module_patcher.register_module_patch(loaded_name, apply)
        report = module_patcher.install_module_patches(load_entry_points=False)
        self.assertTrue(report.ok)
        self.assertTrue(loaded.patched)

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, future_name + ".py").write_text("value = 7\n", encoding="utf-8")
            sys.path.insert(0, directory)
            try:
                module_patcher.register_module_patch(future_name, apply)
                module_patcher.install_module_patches(load_entry_points=False)
                future = importlib.import_module(future_name)
            finally:
                sys.path.remove(directory)
                sys.modules.pop(future_name, None)
        self.assertTrue(future.patched)
        self.assertEqual(calls.count(loaded_name), 2)
        self.assertEqual(calls.count(future_name), 1)
        sys.modules.pop(loaded_name, None)

    def test_patch_failures_are_reported_without_aborting_import(self):
        name = self._module_name("isolated_failure")

        def fail(module):
            raise RuntimeError("expected patch failure")

        def succeed(module):
            module.succeeded = True
            return True

        module_patcher.register_module_patch(name, fail)
        module_patcher.register_module_patch(name, succeed)
        module_patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, name + ".py").write_text("imported = True\n", encoding="utf-8")
            sys.path.insert(0, directory)
            try:
                module = importlib.import_module(name)
            finally:
                sys.path.remove(directory)
                sys.modules.pop(name, None)
        self.assertTrue(module.imported)
        self.assertTrue(module.succeeded)
        report = module_patcher.last_module_patch_report()
        self.assertEqual(len(report.failures), 1)
        self.assertIn("expected patch failure", report.failures[0].detail)

    def test_method_restore_never_overwrites_a_later_patch(self):
        owner = SimpleNamespace(method=object())
        original = owner.method
        replacement = object()
        handle = module_patcher.patch_method(owner, "method", replacement, expected=original)
        self.assertIs(owner.method, replacement)
        self.assertTrue(module_patcher.restore_method(handle))
        self.assertIs(owner.method, original)

        handle = module_patcher.patch_method(owner, "method", replacement)
        later = object()
        owner.method = later
        self.assertFalse(module_patcher.restore_method(handle))
        self.assertIs(owner.method, later)

    def test_entry_points_are_isolated_and_idempotent(self):
        module_name = self._module_name("entry_point")

        def register(register_patch):
            register_patch(module_name, lambda module: True)

        good = _EntryPoint("good-" + module_name, "provider:register", register)
        bad = _EntryPoint("bad-" + module_name, "provider:broken", RuntimeError("broken"))
        with mock.patch.object(module_patcher, "_entry_points", return_value=[bad, good]):
            first = module_patcher.install_module_patches()
            second = module_patcher.install_module_patches()
        statuses = {(item.name, item.status) for item in first.results if item.kind == "entry_point"}
        self.assertIn((bad.name, "failed"), statuses)
        self.assertIn((good.name, "loaded"), statuses)
        self.assertIn("already_loaded", [item.status for item in second.results])

    def test_install_uses_one_process_wide_finder(self):
        module_patcher.install_module_patches(load_entry_points=False)
        module_patcher.install_module_patches(load_entry_points=False)
        finders = [
            finder for finder in sys.meta_path
            if type(finder).__module__ == module_patcher.__name__
        ]
        self.assertEqual(len(finders), 1)

    def test_patched_package_preserves_loader_resource_interfaces(self):
        name = self._module_name("package_resources")

        def apply(module):
            module.patched = True
            return True

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory, name)
            package.mkdir()
            Path(package, "__init__.py").write_text("value = 7\n", encoding="utf-8")
            Path(package, "payload.txt").write_text("resource-data\n", encoding="utf-8")
            sys.path.insert(0, directory)
            try:
                module_patcher.register_module_patch(name, apply)
                module_patcher.install_module_patches(load_entry_points=False)
                package_module = importlib.import_module(name)
                self.assertTrue(package_module.patched)
                self.assertEqual(pkgutil.get_data(name, "payload.txt"), b"resource-data\n")
            finally:
                sys.path.remove(directory)
                sys.modules.pop(name, None)


class TestExternalBackend(unittest.TestCase):
    def _name(self, label):
        return "%s-%s" % (label, uuid.uuid4().hex)

    def test_manifest_discovery_build_and_capability_cache_are_generic(self):
        name = self._name("manifest")
        env_name = "JITTOR_TEST_BACKEND_" + uuid.uuid4().hex.upper()
        calls = []
        prepared = []
        backend_module = ModuleType(name.replace("-", "_"))
        backend_module.execute = lambda: "ok"

        def load_extension(**kwargs):
            calls.append(kwargs)
            return backend_module

        spec = external_backend.ExternalBackendSpec(
            name=name,
            public_functions=("execute",),
            source_envs=(env_name,),
            manifest_names=("backend.json",),
            default_module_name="test_backend",
            environment_names=(env_name,),
        )
        resolver = external_backend.ExternalBackend(
            spec,
            extension_loader=load_extension,
            prepare_capability=prepared.append,
            capability_miss=lambda module, key: None,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Path(root, "kernel.cpp").write_text("// test\n", encoding="utf-8")
            Path(root, "backend.json").write_text(
                json.dumps(
                    {
                        "module": "test_backend",
                        "sources": ["kernel.cpp"],
                        "build_dir": "build",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {env_name: directory}, clear=False):
                first = resolver.load(capability_key=("float16", 64))
                second = resolver.load(capability_key=("float16", 64))
        self.assertIs(first, backend_module)
        self.assertIs(second, backend_module)
        self.assertEqual(len(calls), 1)
        self.assertEqual(prepared, [("float16", 64), ("float16", 64)])
        self.assertEqual(resolver.generation, 1)
        self.assertEqual(resolver.last_report.attempts[-1].status, "loaded")

    def test_source_capability_miss_restores_installed_module_and_path(self):
        backend_name = self._name("source-rollback")
        module_name = "_jittor_backend_" + uuid.uuid4().hex
        env_name = "JITTOR_TEST_SOURCE_" + uuid.uuid4().hex.upper()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed_root = root / "installed"
            source_root = root / "source"
            installed_package = installed_root / module_name
            source_package = source_root / module_name
            installed_package.mkdir(parents=True)
            source_package.mkdir(parents=True)
            Path(installed_package, "__init__.py").write_text(
                "origin = 'installed'\n"
                "def execute(): return origin\n",
                encoding="utf-8",
            )
            Path(installed_package, "shared.py").write_text(
                "origin = 'installed-submodule'\n", encoding="utf-8"
            )
            Path(source_package, "__init__.py").write_text(
                "from . import local_only\n"
                "origin = 'source'\n"
                "def execute(): return origin\n",
                encoding="utf-8",
            )
            Path(source_package, "local_only.py").write_text(
                "origin = 'source-submodule'\n", encoding="utf-8"
            )

            sys.path.insert(0, os.fspath(installed_root))
            try:
                installed = importlib.import_module(module_name)
                installed_shared = importlib.import_module(module_name + ".shared")
                path_before = tuple(sys.path)
                resolver = external_backend.ExternalBackend(
                    external_backend.ExternalBackendSpec(
                        name=backend_name,
                        public_functions=("execute",),
                        source_envs=(env_name,),
                        module_names=(module_name,),
                    ),
                    capability_miss=lambda module, key: (
                        "source capability mismatch"
                        if module.origin == "source"
                        else None
                    ),
                )
                with mock.patch.dict(
                    os.environ, {env_name: os.fspath(source_root)}, clear=False
                ):
                    selected = resolver.load(capability_key="required")

                self.assertIs(selected, installed)
                self.assertEqual(tuple(sys.path), path_before)
                self.assertIs(sys.modules[module_name], installed)
                self.assertIs(sys.modules[module_name + ".shared"], installed_shared)
                self.assertNotIn(module_name + ".local_only", sys.modules)
                self.assertEqual(
                    [attempt.status for attempt in resolver.last_report.attempts],
                    ["capability_miss", "loaded"],
                )
            finally:
                sys.path.remove(os.fspath(installed_root))
                for key in list(sys.modules):
                    if key == module_name or key.startswith(module_name + "."):
                        sys.modules.pop(key, None)

    def test_source_exception_restores_state_before_installed_fallback(self):
        backend_name = self._name("source-failure")
        module_name = "_jittor_backend_" + uuid.uuid4().hex
        env_name = "JITTOR_TEST_SOURCE_" + uuid.uuid4().hex.upper()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installed_root = root / "installed"
            source_root = root / "source"
            installed_package = installed_root / module_name
            installed_package.mkdir(parents=True)
            source_root.mkdir()
            Path(installed_package, "__init__.py").write_text(
                "origin = 'installed'\n"
                "def execute(): return origin\n",
                encoding="utf-8",
            )

            def fail_source(path):
                partial = ModuleType(module_name)
                partial.__file__ = os.fspath(path / "partial.py")
                sys.modules[module_name] = partial
                raise RuntimeError("source load failed")

            sys.path.insert(0, os.fspath(installed_root))
            try:
                installed = importlib.import_module(module_name)
                path_before = tuple(sys.path)
                resolver = external_backend.ExternalBackend(
                    external_backend.ExternalBackendSpec(
                        name=backend_name,
                        public_functions=("execute",),
                        source_envs=(env_name,),
                        module_names=(module_name,),
                        source_predicates=(lambda path: path == source_root,),
                    ),
                    special_source_loader=fail_source,
                )
                with mock.patch.dict(
                    os.environ, {env_name: os.fspath(source_root)}, clear=False
                ):
                    selected = resolver.load()

                self.assertIs(selected, installed)
                self.assertEqual(tuple(sys.path), path_before)
                self.assertIs(sys.modules[module_name], installed)
                self.assertEqual(
                    [attempt.status for attempt in resolver.last_report.attempts],
                    ["failed", "loaded"],
                )
            finally:
                sys.path.remove(os.fspath(installed_root))
                for key in list(sys.modules):
                    if key == module_name or key.startswith(module_name + "."):
                        sys.modules.pop(key, None)

    def test_project_provider_can_extend_discovery_before_registration(self):
        name = self._name("hint")
        project_env = "JITTOR_TEST_PROJECT_" + uuid.uuid4().hex.upper()
        external_backend.register_external_backend_hint(
            name,
            project_root_envs=(project_env,),
            relative_source_dirs=("vendor/backend",),
        )
        spec = external_backend.ExternalBackendSpec(
            name=name,
            public_functions=("execute",),
            manifest_names=("backend.json",),
        )
        resolver = external_backend.register_external_backend(
            external_backend.ExternalBackend(spec)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "vendor", "backend")
            root.mkdir(parents=True)
            Path(root, "backend.json").write_text("{}", encoding="utf-8")
            with mock.patch.dict(os.environ, {project_env: directory}, clear=False):
                self.assertEqual(resolver.source_roots(), [os.fspath(root.resolve())])
                self.assertIn((project_env, directory), resolver.configuration_key()[0])

    def test_external_backend_entry_points_allow_hint_only_providers(self):
        name = self._name("entry-hint")

        def register_hint():
            external_backend.register_external_backend_hint(
                name, project_root_envs=("JITTOR_TEST_HINT_ROOT",)
            )

        good = _EntryPoint("good-" + name, "provider:hint", register_hint)
        bad = _EntryPoint("bad-" + name, "provider:broken", RuntimeError("broken"))
        with mock.patch.object(external_backend, "_entry_points", return_value=[bad, good]):
            first = external_backend.load_external_backend_entry_points()
            second = external_backend.load_external_backend_entry_points()
        self.assertEqual([item.status for item in first], ["failed", "loaded"])
        self.assertEqual([item.status for item in second], ["failed", "already_loaded"])

    def test_registered_resolver_accepts_late_discovery_hint(self):
        name = self._name("late-hint")
        project_env = "JITTOR_TEST_LATE_PROJECT_" + uuid.uuid4().hex.upper()
        resolver = external_backend.register_external_backend(
            external_backend.ExternalBackend(
                external_backend.ExternalBackendSpec(
                    name=name,
                    public_functions=("execute",),
                    manifest_names=("backend.json",),
                )
            )
        )
        external_backend.register_external_backend_hint(
            name,
            project_root_envs=(project_env,),
            relative_source_dirs=("optional/backend",),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "optional", "backend")
            root.mkdir(parents=True)
            Path(root, "backend.json").write_text("{}", encoding="utf-8")
            with mock.patch.dict(os.environ, {project_env: directory}, clear=False):
                self.assertEqual(resolver.source_roots(), [os.fspath(root.resolve())])

    def test_backend_claim_covers_nested_build_directories(self):
        name = self._name("nested-claim")
        resolver = external_backend.register_external_backend(
            external_backend.ExternalBackend(
                external_backend.ExternalBackendSpec(
                    name=name,
                    public_functions=("execute",),
                    manifest_names=("backend.json",),
                )
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested" / "extension"
            nested.mkdir(parents=True)
            Path(root, "backend.json").write_text("{}", encoding="utf-8")
            self.assertIs(
                external_backend.external_backend_for_source_root(nested), resolver
            )

    def test_project_hint_extends_the_canonical_flash_resolver(self):
        from jittor.compat.shim.backends import flash_attention as flashattn_jittor

        project_env = "JITTOR_TEST_FLASH_PROJECT_" + uuid.uuid4().hex.upper()
        relative = "optional-" + uuid.uuid4().hex
        external_backend.register_external_backend_hint(
            "flash-attn",
            project_root_envs=(project_env,),
            relative_source_dirs=(relative,),
        )
        self.assertIs(
            external_backend.registered_external_backends()["flash-attn"],
            flashattn_jittor._EXTERNAL_BACKEND,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, relative)
            root.mkdir()
            Path(root, "flashattn_jittor.json").write_text("{}", encoding="utf-8")
            with mock.patch.dict(os.environ, {project_env: directory}, clear=False):
                self.assertIn(os.fspath(root.resolve()), flashattn_jittor.candidate_source_roots())
                self.assertIn(
                    (project_env, directory),
                    flashattn_jittor._backend_environment_key(),
                )

    def test_runtime_modules_share_the_common_import_hook(self):
        jittor_root = Path(module_patcher.__file__).resolve().parents[1]
        for relative in (
            "compat/shim/extensions/readonly.py",
        ):
            source = Path(jittor_root, relative).read_text(encoding="utf-8")
            with self.subTest(relative=relative):
                self.assertNotIn("MetaPathFinder", source)
                self.assertNotIn("PathFinder.find_spec", source)
                self.assertIn("register_module_patch", source)
    def test_project_runtime_glue_is_not_shipped(self):
        jittor_root = Path(module_patcher.__file__).resolve().parents[1]
        for relative in (
            "monkeypatch_ops.py",
            "torch_shim",
        ):
            with self.subTest(relative=relative):
                self.assertFalse(Path(jittor_root, relative).exists())

    def test_bootstrap_has_no_backend_specific_source_scanner(self):
        jittor_root = Path(module_patcher.__file__).resolve().parents[1]
        runtime = Path(jittor_root, "compat", "shim", "runtime.py").read_text(
            encoding="utf-8"
        )
        discovery = Path(
            jittor_root, "compat", "shim", "discovery.py"
        ).read_text(encoding="utf-8")
        flash = Path(
            jittor_root, "compat", "shim", "backends", "flash_attention.py"
        ).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_is_official_flash_attention_root", runtime)
        self.assertNotIn("_is_official_flash_attention_root", discovery)
        self.assertNotIn("TRELLIS_ROOT", flash)
        self.assertNotIn("TRELLIS2_ROOT", flash)
        self.assertIn("external_backend_for_source_root", discovery)
        self.assertIn("ExternalBackendSpec", flash)

    def test_flash_child_environment_uses_the_canonical_python_root(self):
        import jittor as jt
        from jittor.compat.shim.backends import flash_attention

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "flash-attention")
            root.mkdir()
            with mock.patch.dict(os.environ, {"PYTHONPATH": "sentinel"}, clear=False):
                child = flash_attention._setup_child_env(root)
        paths = child["PYTHONPATH"].split(os.pathsep)
        self.assertIn(os.fspath(Path(jt.__file__).resolve().parents[1]), paths)
        self.assertNotIn(os.fspath(Path(flash_attention.__file__).resolve().parents[2]), paths)


if __name__ == "__main__":
    unittest.main()

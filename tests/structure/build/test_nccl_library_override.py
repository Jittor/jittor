"""NCCL library selection contracts without importing or initializing CUDA."""

import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
EXTERN = ROOT / "python/jittor/build/compile_extern.py"
COMPILER = ROOT / "python/jittor/build/compiler.py"


def _leaf(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function(path, name, namespace):
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class TestNcclLibraryOverride(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        utils = ROOT / "python/jittor/build/utils"
        cls.env_config = _leaf("nccl_test_env", utils / "env_config.py")
        cls.wheel = _leaf("nccl_test_wheel", utils / "cuda_wheel.py")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.wheel_root = self._component("wheel")
        self.override_root = self._component("override")
        versions = {name: self.wheel.reference_version(version)
                    for name, _, version, _ in self.wheel.CUDA12_COMPONENTS}
        self.stack = self.wheel.CudaWheelStack({"nccl": str(self.wheel_root)}, versions)
        self.calls = []

    def _component(self, name):
        root = self.base / name
        (root / "include").mkdir(parents=True)
        (root / "lib").mkdir()
        (root / "include/nccl.h").write_text("/* fixture, never compiled */\n")
        (root / "lib/libnccl.so").write_bytes(b"fixture, never loaded")
        return root

    def _run(self, explicit=None, wheel=True, load_error=None, extra_environment=None):
        def cdll(path, flags):
            self.calls.append(("cdll", path))
            if load_error:
                raise load_error
            return object()
        def compile_ops(sources, **kwargs):
            self.calls.append(("compile", kwargs["extra_flags"]))
            return types.SimpleNamespace(ops=object())
        def reject_fallback(*args, **kwargs):
            raise AssertionError("Explicit NCCL selection unexpectedly fell back")
        namespace = {
            "os": os, "has_cuda": True, "is_cuda": True, "has_mpi": False,
            "build_flag": self.env_config.build_flag,
            "build_env": self.env_config.build_env,
            "cuda_wheel_stack": self.stack if wheel else None,
            "inside_mpi": lambda: False, "_skip_nccl_p2p_without_peer_access": lambda: None,
            "search_file": lambda paths, name: str(Path(paths[0]) / name),
            "ctypes": types.SimpleNamespace(CDLL=cdll), "dlopen_flags": 123,
            "preload_cuda_library": lambda name, required: self.calls.append(("preload", name)),
            "LOG": types.SimpleNamespace(v=lambda *args: None, vv=lambda *args: None),
            "jittor_path": str(ROOT / "python/jittor"),
            "backend_root": lambda path, backend: str(ROOT / "backends" / backend),
            "cuda_sdk_flags": "", "cuda_link_flags": "", "mpi_compile_flags": "",
            "compile_custom_ops": compile_ops,
            "lock": types.SimpleNamespace(unlock_scope=nullcontext),
            "_init_nccl_from_store": lambda nccl, store: self.calls.append(("init", store)),
            "register_library": lambda name, module: self.calls.append(("publish", name)),
            "find_system_nccl": reject_fallback, "install_nccl": reject_fallback,
        }
        namespace["cuda_library_link_flags"] = _function(
            COMPILER, "cuda_library_link_flags", {"os": os})
        setup = _function(EXTERN, "setup_nccl", namespace)
        environment = {"JT_NCCL_WORLD_SIZE": "2", "JT_BUILD_USE_NCCL": "1"}
        if explicit is not None:
            environment.update(JT_BUILD_NCCL_INCLUDE_PATH=str(explicit / "include"),
                               JT_BUILD_NCCL_LIB_PATH=str(explicit / "lib"))
        environment.update(extra_environment or {})
        with mock.patch.dict(os.environ, environment, clear=True):
            setup(store="fixture-store")

    def _assert_linked(self, root):
        flags = next(value for kind, value in self.calls if kind == "compile")
        self.assertIn('-I"%s"' % (root / "include"), flags)
        self.assertIn('-L"%s" -l:libnccl.so' % (root / "lib"), flags)
        self.assertEqual(self.calls[-2:], [("init", "fixture-store"), ("publish", "nccl")])

    def test_explicit_library_wins_over_installed_wheel(self):
        self._run(explicit=self.override_root)
        self.assertEqual(self.calls[0], ("cdll", str(self.override_root / "lib/libnccl.so")))
        self.assertFalse(any(kind == "preload" for kind, _ in self.calls))
        self._assert_linked(self.override_root)

    def test_default_wheel_keeps_dependency_preload(self):
        self._run()
        self.assertEqual(self.calls[0], ("preload", "nccl"))
        self.assertFalse(any(kind == "cdll" for kind, _ in self.calls))
        self._assert_linked(self.wheel_root)

    def test_explicit_wheel_path_remains_the_selected_library(self):
        self._run(explicit=self.wheel_root)
        self.assertEqual(self.calls[0], ("cdll", str(self.wheel_root / "lib/libnccl.so")))
        self._assert_linked(self.wheel_root)

    def test_explicit_library_without_wheels(self):
        self._run(explicit=self.override_root, wheel=False)
        self.assertEqual(self.calls[0], ("cdll", str(self.override_root / "lib/libnccl.so")))
        self._assert_linked(self.override_root)

    def test_canonical_explicit_paths_take_precedence_over_legacy_paths(self):
        self._run(explicit=self.override_root, extra_environment={
            "nccl_include_path": str(self.wheel_root / "include"),
            "nccl_lib_path": str(self.wheel_root / "lib"),
        })
        self.assertEqual(self.calls[0], ("cdll", str(self.override_root / "lib/libnccl.so")))
        self._assert_linked(self.override_root)

    def test_missing_explicit_library_does_not_load_wheel(self):
        (self.override_root / "lib/libnccl.so").unlink()
        with self.assertRaises(AssertionError):
            self._run(explicit=self.override_root)
        self.assertEqual(self.calls, [])

    def test_explicit_load_failure_does_not_fall_back_to_wheel(self):
        with self.assertRaisesRegex(OSError, "selected NCCL failed"):
            self._run(explicit=self.override_root, load_error=OSError("selected NCCL failed"))
        self.assertEqual(self.calls, [("cdll", str(self.override_root / "lib/libnccl.so"))])

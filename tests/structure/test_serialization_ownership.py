"""Serialization compatibility contracts without importing a real runtime."""

import ast
import importlib
import io
import pickle
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np


PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
MODULES = ("load_pytorch", "load_pytorch_old", "save_pytorch")


class TestSerializationOwnership(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        for name in tuple(sys.modules):
            if name in ("jittor", "jittor_utils") or name.startswith(
                    ("jittor.", "jittor_utils.")):
                del sys.modules[name]
        utils = ModuleType("jittor_utils")
        utils.__path__ = [str(PYTHON_ROOT / "jittor_utils")]
        sys.modules[utils.__name__] = utils
        self.services = importlib.import_module("jittor_utils.runtime_services")

    def tearDown(self):
        self.modules.stop()

    def bootstrap_fake_runtime(self):
        runtime = ModuleType("jittor")
        runtime.__path__ = [str(PYTHON_ROOT / "jittor")]
        runtime.Var = type("Var", (), {})
        runtime.array = np.array
        runtime.nn = ModuleType("jittor.nn")
        sys.modules[runtime.__name__] = runtime
        package = importlib.import_module("jittor.serialization")
        package.register_compatibility_services()
        return package

    def test_old_modules_do_not_bootstrap_runtime(self):
        for name in MODULES:
            legacy = importlib.import_module("jittor_utils." + name)
            with self.assertRaisesRegex(RuntimeError, "import jittor before"):
                getattr(legacy, "save_pytorch" if name == "save_pytorch" else "load_pytorch")
        self.assertNotIn("jittor", sys.modules)

    def test_registration_is_lazy_and_idempotent(self):
        package = self.bootstrap_fake_runtime()
        package.register_compatibility_services()
        for name in MODULES:
            self.assertNotIn("jittor.serialization." + name, sys.modules)

    def test_legacy_attributes_and_pickle_resolve_to_owner(self):
        self.bootstrap_fake_runtime()
        # The writer imports Torch only at its own operation boundary. A fake
        # module is enough here because these tests never execute its writer.
        sys.modules["torch"] = ModuleType("torch")
        for name in MODULES:
            canonical = importlib.import_module("jittor.serialization." + name)
            legacy = importlib.import_module("jittor_utils." + name)
            entry = "save_pytorch" if name == "save_pytorch" else "load_pytorch"
            function = getattr(canonical, entry)
            self.assertIs(getattr(legacy, entry), function)
            self.assertEqual(function.__module__, canonical.__name__)
            self.assertIs(pickle.loads(pickle.dumps(function)), function)
            old_pickle = ("cjittor_utils.%s\n%s\n." % (name, entry)).encode("ascii")
            self.assertIs(pickle.loads(old_pickle), function)
            self.assertIn(entry, dir(legacy))
            self.assertIn(entry, legacy.__all__)

    def test_historical_rebuild_pickle_executes_canonical_algorithm(self):
        self.bootstrap_fake_runtime()
        args = (np.arange(8, dtype=np.int32), 2, (2, 2), (2, 1), False, {})
        payload = b"cjittor_utils.load_pytorch\njittor_rebuild\n"
        payload += pickle.dumps(args, protocol=0)[:-1] + b"R."
        np.testing.assert_array_equal(pickle.loads(payload), [[2, 3], [4, 5]])
        canonical = importlib.import_module("jittor.serialization.load_pytorch")
        self.assertIs(
            canonical.UnpicklerWrapper(io.BytesIO()).find_class(
                "torch._utils", "_rebuild_tensor_v2"),
            canonical.jittor_rebuild,
        )

    def test_service_failure_propagates_without_caching_a_result(self):
        def fail():
            raise OSError("reader unavailable")

        self.services.register_runtime_module("load_pytorch", fail)
        legacy = importlib.import_module("jittor_utils.load_pytorch")
        with self.assertRaisesRegex(OSError, "reader unavailable"):
            legacy.load_pytorch
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            self.services.register_runtime_module("load_pytorch", lambda: None)

    def test_utility_endpoints_contain_no_runtime_import_or_algorithms(self):
        for name in MODULES + ("runtime_services",):
            tree = ast.parse((PYTHON_ROOT / "jittor_utils" / (name + ".py")).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(a.name == "jittor" or a.name.startswith("jittor.")
                                         for a in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertFalse(node.module == "jittor" or (node.module or "").startswith("jittor."))
            if name != "runtime_services":
                self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.ClassDef))
                                     for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()

"""Version and import-guard contracts, using only controlled fake packages."""
import builtins
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "adapters"))
from jittor_adapters import transformers, torchmetrics
from jittor_adapters._common import UnsupportedAdapterVersion

# Load the actual registration mechanism without importing the native runtime.
native = types.ModuleType("jittor")
native.__path__ = []
compat = types.ModuleType("_adapter_tests_compat")
compat.__path__ = [str(ROOT / "compat")]
sys.modules[compat.__name__] = compat
patcher = importlib.import_module(compat.__name__ + ".module_patcher")


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        self.modules = mock.patch.dict(sys.modules, {
            "jittor": native, "jittor.compat": compat,
            "jittor.compat.module_patcher": patcher,
        })
        self.modules.start()
        self.import_function = builtins.__import__
        self.saved_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()

    def tearDown(self):
        patcher.uninstall_module_patches()
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.saved_registry)
        for name in tuple(sys.modules):
            if name in ("transformers", "torchmetrics") or name.startswith(("transformers.", "torchmetrics.")):
                sys.modules.pop(name)
        self.assertIs(builtins.__import__, self.import_function)
        self.modules.stop()

    def fake_transformers(self, root, version):
        package = Path(root) / "transformers"
        (package / "utils").mkdir(parents=True)
        (package / "__init__.py").write_text(
            "__version__ = %r\nfrom .utils.import_utils import is_torch_npu_available\n"
            "probe_result = is_torch_npu_available(check_device=True)\n" % version)
        (package / "utils/__init__.py").write_text("")
        (package / "utils/import_utils.py").write_text(
            "def is_torch_npu_available(check_device=False):\n"
            "    raise RuntimeError('native torch_npu probe must not execute')\n")

    def test_transformers_npu_probe_rejects_real_pytorch_extension(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as root:
            self.fake_transformers(root, "4.56.2")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("transformers")
                self.assertFalse(module.probe_result)
                guard = sys.modules["transformers.utils.import_utils"].is_torch_npu_available
                self.assertFalse(guard())
                self.assertIs(module.is_torch_npu_available, guard)
                self.assertTrue(callable(guard.cache_clear))
            finally:
                sys.path.remove(root)

    def test_unsupported_transformers_version_fails_real_import(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as root:
            self.fake_transformers(root, "99.0.0")
            sys.path.insert(0, root)
            try:
                with self.assertRaisesRegex(UnsupportedAdapterVersion, "99.0.0"):
                    importlib.import_module("transformers")
                self.assertIn("99.0.0", patcher.last_module_patch_report().failures[0].detail)
            finally:
                sys.path.remove(root)

    def test_missing_adapter_is_reported_as_unavailable(self):
        with mock.patch.object(patcher, "_entry_points", return_value=()):
            report = patcher.install_module_patches(expected_entry_points=("jittor_transformers",))
        self.assertEqual([(entry.name, entry.status) for entry in report.results],
                         [("jittor_transformers", "unavailable")])

    def test_torchmetrics_version_guard_and_existing_bound_aliases(self):
        root = types.ModuleType("torchmetrics")
        root.__version__ = "1.7.4"
        data = types.ModuleType("torchmetrics.utilities.data")
        alias = types.ModuleType("torchmetrics.functional.classification")
        original = lambda value, minlength=None: ("original", minlength)
        data._bincount = alias._bincount = original
        data.dim_zero_cat = lambda value: value
        sys.modules.update({module.__name__: module for module in (root, data, alias)})
        torchmetrics.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        self.assertIs(alias._bincount, data._bincount)
        self.assertIsNot(data._bincount, original)
        self.assertEqual(data._bincount(object()), ("original", None))
        root.__version__ = "2.0.0"
        with self.assertRaisesRegex(UnsupportedAdapterVersion, "2.0.0"):
            torchmetrics.patch_data(data)


if __name__ == "__main__":
    unittest.main()

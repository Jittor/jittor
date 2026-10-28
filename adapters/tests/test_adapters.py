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
from jittor_adapters import transformers, transformers_qwen35, torchmetrics
from jittor_adapters._common import UnsupportedAdapterVersion, require_version

# Load the actual registration mechanism without importing the native runtime.
native = types.ModuleType("jittor")
native.__path__ = []
torch_frontend = types.ModuleType("torch")
torch_frontend.__torch_version__ = "2.11.0"
compat = types.ModuleType("_adapter_tests_compat")
compat.__path__ = [str(ROOT / "compat")]
sys.modules[compat.__name__] = compat
patcher = importlib.import_module(compat.__name__ + ".module_patcher")


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        self.modules = mock.patch.dict(sys.modules, {
            "jittor": native, "jittor.compat": compat,
            "jittor.compat.module_patcher": patcher,
            "torch": torch_frontend,
        })
        self.modules.start()
        # These cases build *fake* `transformers`/`torchmetrics` packages, and a
        # real one already in `sys.modules` wins over the fake however the temp
        # directory is put on `sys.path`: `importlib.import_module` returns the
        # cached module. Whether one is there depends on which other file the
        # session collected first -- measured, a module that imports the real
        # package at collection time made `..._npu_probe_rejects_real_pytorch_extension`
        # fail with `module transformers has no attribute probe_result` and
        # `..._unsupported_transformers_version_fails_real_import` fail with
        # `UnsupportedAdapterVersion not raised`, while the same file ran green on
        # its own. Evicted for the duration; `self.modules.stop()` restores them.
        for name in tuple(sys.modules):
            if name in ("transformers", "torchmetrics") or name.startswith(
                    ("transformers.", "torchmetrics.")):
                sys.modules.pop(name)
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
            "__version__ = %r\n"
            "from .utils.import_utils import (get_torch_version, is_torch_available, "
            "is_torch_npu_available, is_torchvision_available, is_torchvision_v2_available)\n"
            "probe_result = is_torch_npu_available(check_device=True)\n"
            "torch_available = is_torch_available()\n"
            "torch_version = get_torch_version()\n" % version)
        (package / "utils/__init__.py").write_text("")
        (package / "utils/import_utils.py").write_text(
            "from functools import lru_cache\n"
            "def _is_package_available(package, return_version=False):\n"
            "    result = package == 'torch'\n"
            "    return (result, '1.3.11.0' if result else 'N/A') if return_version "
            "else result\n"
            "@lru_cache()\n"
            "def is_torch_available():\n"
            "    found, version = _is_package_available('torch', return_version=True)\n"
            "    major, minor = (int(part) for part in version.split('.')[:2])\n"
            "    return found and (major, minor) >= (2, 4)\n"
            "@lru_cache()\n"
            "def get_torch_version():\n"
            "    return _is_package_available('torch', return_version=True)[1]\n"
            "def is_torch_npu_available(check_device=False):\n"
            "    raise RuntimeError('native torch_npu probe must not execute')\n"
            "def is_torchvision_available():\n"
            "    return False\n"
            "def is_torchvision_v2_available():\n"
            "    raise RuntimeError('unavailable torchvision version must not be parsed')\n"
            "BACKENDS_MAPPING = {'torchvision': (is_torchvision_available, 'torchvision unavailable')}\n")

    #: The shape transformers 5.x ships: the version is assigned while the
    #: package executes, then the module is handed to a ``_LazyModule`` and the
    #: namespace entry is dropped. Once the import is over the value is only
    #: reachable through ``__getattr__`` -- which is the state that was read.
    _LAZY_INIT = '''
import sys

__version__ = %r

from .utils.import_utils import is_torch_npu_available


class _LazyModule(type(sys)):
    _version = %r

    def __getattr__(self, name):
        if name == "__version__":
            return type(self)._version
        raise AttributeError(name)


sys.modules[__name__].__class__ = _LazyModule
del __version__
'''

    def fake_lazy_transformers(self, root, version):
        self.fake_transformers(root, version)
        (Path(root) / "transformers" / "__init__.py").write_text(
            self._LAZY_INIT % (version, version))

    def test_a_version_reachable_only_through_getattr_is_accepted(self):
        """The version lookup used to read the namespace dict, not the module.

        ``require_version`` did ``vars(module).get("__version__")``. Transformers
        5.x ships the package as a ``_LazyModule`` that carries the version in
        ``extra_objects`` and drops the namespace entry, so once the package has
        been imported that lookup reports ``None`` and the adapter rejects
        5.5.3 -- a version ``SUPPORTED_VERSIONS`` lists. The rejection is a
        ``@required_patch``, so it escapes ``install_module_patches`` and took
        five tests in ``compat/tests/torch/test_compat_mechanisms.py`` with it.

        The precondition asserted below is what gives this case teeth: if the
        fake ever keeps ``__version__`` in the dict, the case stops modelling
        the shape that was broken and the check below passes for the wrong
        reason.
        """
        with tempfile.TemporaryDirectory() as root:
            self.fake_lazy_transformers(root, "5.5.3")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("transformers")
                self.assertNotIn("__version__", vars(module))
                self.assertEqual(module.__version__, "5.5.3")
                self.assertEqual(
                    require_version("transformers", transformers.SUPPORTED_VERSIONS),
                    "5.5.3")
            finally:
                sys.path.remove(root)

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

    def test_transformers_513_probe_contract(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as root:
            self.fake_transformers(root, "5.13.0")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("transformers")
                probes = sys.modules["transformers.utils.import_utils"]
                self.assertFalse(module.is_torch_npu_available(check_device=True))
                self.assertIs(module.is_torch_npu_available, probes.is_torch_npu_available)
                self.assertIs(
                    probes.BACKENDS_MAPPING["torchvision"][0],
                    probes.is_torchvision_available,
                )
            finally:
                sys.path.remove(root)

    def test_transformers_recognizes_the_deployed_torchvision_facade(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as root:
            self.fake_transformers(root, "4.56.2")
            facade = Path(root) / "torchvision"
            facade.mkdir()
            (facade / "__init__.py").write_text("__version__ = '2.11.0'\n")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("transformers")
                import_utils = sys.modules["transformers.utils.import_utils"]
                self.assertTrue(module.is_torchvision_available())
                self.assertTrue(module.is_torchvision_v2_available())
                self.assertIs(module.is_torchvision_available, import_utils.is_torchvision_available)
                self.assertIs(module.is_torchvision_v2_available, import_utils.is_torchvision_v2_available)
                self.assertIs(
                    import_utils.BACKENDS_MAPPING["torchvision"][0],
                    import_utils.is_torchvision_available,
                )
                self.assertTrue(import_utils.BACKENDS_MAPPING["torchvision"][0]())
            finally:
                sys.path.remove(root)

    def test_transformers_uses_the_emulated_torch_api_version(self):
        transformers.register(patcher.register_module_patch)
        patcher.install_module_patches(load_entry_points=False)
        with tempfile.TemporaryDirectory() as root:
            self.fake_transformers(root, "5.9.0")
            sys.path.insert(0, root)
            try:
                module = importlib.import_module("transformers")
                self.assertTrue(module.torch_available)
                self.assertEqual(module.torch_version, "2.11.0")
                probe = sys.modules["transformers.utils.import_utils"]._is_package_available
                self.assertIs(probe("torch"), True)
                self.assertEqual(probe("torch", return_version=True), (True, "2.11.0"))
            finally:
                sys.path.remove(root)

    def test_qwen35_adapter_is_registered_and_version_locked(self):
        registrations = []
        transformers_qwen35.register(
            lambda path, callback: registrations.append((path, callback)))
        self.assertEqual(
            registrations,
            [("transformers.models.qwen3_5.modeling_qwen3_5",
              transformers_qwen35.patch_qwen35)],
        )

        package = types.ModuleType("transformers")
        package.__version__ = "5.8.0"
        module = types.SimpleNamespace()
        with mock.patch.dict(sys.modules, {"transformers": package}):
            with self.assertRaisesRegex(UnsupportedAdapterVersion, "5.8.0"):
                transformers_qwen35.patch_qwen35(module)

    def test_qwen35_adapter_rejects_changed_upstream_source(self):
        def changed_function():
            return None

        for name in transformers_qwen35.SOURCE_SHA256:
            with self.subTest(function=name):
                with self.assertRaisesRegex(UnsupportedAdapterVersion, "source changed"):
                    transformers_qwen35._verify_source(changed_function, name)


    def test_qwen35_partial_retirement_stub_identity_and_repeat(self):
        """Stub-only install contract; no model/tensor/device is executed."""
        def chunk():
            raise AssertionError("stub chunk must not execute")

        def recurrent():
            raise AssertionError("stub recurrent must not execute")

        def replacement():
            raise AssertionError("stub replacement must not execute")

        replacement._jittor_transformers_qwen35_functional = True
        module = types.SimpleNamespace(
            torch_chunk_gated_delta_rule=chunk,
            torch_recurrent_gated_delta_rule=recurrent,
        )
        package = types.ModuleType("transformers")
        package.__version__ = "5.9.0"

        def replace_stub(owner, name, value, expected):
            self.assertIs(owner, module)
            self.assertIs(getattr(owner, name), expected)
            setattr(owner, name, value)

        with mock.patch.dict(sys.modules, {"transformers": package}):
            with mock.patch.object(transformers_qwen35, "_verify_source") as verify:
                with mock.patch.object(transformers_qwen35, "_functional_chunk") as chunk_factory:
                    with mock.patch.object(transformers_qwen35, "_functional_recurrent", return_value=replacement) as recurrent_factory:
                        with mock.patch.object(transformers_qwen35, "replace", side_effect=replace_stub) as replace_mock:
                            self.assertIs(transformers_qwen35.patch_qwen35(module), True)
                            self.assertIs(module.torch_chunk_gated_delta_rule, chunk)
                            self.assertIs(module.torch_recurrent_gated_delta_rule, replacement)
                            self.assertEqual(verify.call_args_list, [
                                mock.call(chunk, "torch_chunk_gated_delta_rule"),
                                mock.call(recurrent, "torch_recurrent_gated_delta_rule"),
                            ])
                            self.assertIs(transformers_qwen35.patch_qwen35(module), False)
                            self.assertIs(module.torch_chunk_gated_delta_rule, chunk)
                            self.assertIs(module.torch_recurrent_gated_delta_rule, replacement)
                            self.assertEqual(verify.call_args_list, [
                                mock.call(chunk, "torch_chunk_gated_delta_rule"),
                                mock.call(recurrent, "torch_recurrent_gated_delta_rule"),
                                mock.call(chunk, "torch_chunk_gated_delta_rule"),
                            ])
                            chunk_factory.assert_not_called()
                            recurrent_factory.assert_called_once_with(module)
                            replace_mock.assert_called_once_with(
                                module, "torch_recurrent_gated_delta_rule", replacement, recurrent)

    def test_qwen35_partial_retirement_stub_repeated_registration(self):
        """Only the shared registry contract is exercised, not HF loading."""
        for _ in range(2):
            transformers_qwen35.register(patcher.register_module_patch)
        self.assertEqual(
            patcher.registered_module_patches(),
            {"transformers.models.qwen3_5.modeling_qwen3_5": (
                transformers_qwen35.patch_qwen35,)},
        )

    def test_qwen35_partial_retirement_stub_either_source_rejects_before_patch(self):
        """A failed source gate must precede either factory or any mutation."""
        for failed_name in transformers_qwen35.SOURCE_SHA256:
            with self.subTest(function=failed_name):
                originals = {name: (lambda: None) for name in transformers_qwen35.SOURCE_SHA256}
                module = types.SimpleNamespace(**originals)
                package = types.ModuleType("transformers")
                package.__version__ = "5.9.0"

                def verify_stub(function, name):
                    self.assertIs(function, originals[name])
                    if name == failed_name:
                        raise UnsupportedAdapterVersion("changed source")

                with mock.patch.dict(sys.modules, {"transformers": package}):
                    with mock.patch.object(transformers_qwen35, "_verify_source", side_effect=verify_stub):
                        with mock.patch.object(transformers_qwen35, "_functional_chunk") as chunk_factory:
                            with mock.patch.object(transformers_qwen35, "_functional_recurrent") as recurrent_factory:
                                with mock.patch.object(transformers_qwen35, "replace") as replace_mock:
                                    with self.assertRaises(UnsupportedAdapterVersion):
                                        transformers_qwen35.patch_qwen35(module)
                                    chunk_factory.assert_not_called()
                                    recurrent_factory.assert_not_called()
                                    replace_mock.assert_not_called()
                                    for name, original in originals.items():
                                        self.assertIs(getattr(module, name), original)

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

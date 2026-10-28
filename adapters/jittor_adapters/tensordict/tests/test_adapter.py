"""TensorDict adapter host contracts against the real rollback mechanism.

The fake leaves preserve object identity and reject NumPy conversion. They are
not numerical tensors and these tests are not hardware/autograd evidence.
"""
import builtins
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ADAPTERS = Path(__file__).resolve().parents[3]
REPOSITORY = ADAPTERS.parent
sys.path.insert(0, str(ADAPTERS))
sys.path.insert(0, str(ADAPTERS / "tests"))
from _adapter_boundary import boundary_errors
from jittor_adapters import tensordict as adapter
from jittor_adapters._common import UnsupportedAdapterVersion

compat = types.ModuleType("_tensordict_adapter_host_compat")
compat.__path__ = [os.environ.get("JITTOR_COMPAT_SOURCE", str(REPOSITORY / "compat"))]
sys.modules[compat.__name__] = compat
patcher = importlib.import_module(compat.__name__ + ".module_patcher")
transaction = importlib.import_module(compat.__name__ + ".transaction")


class Dense:
    def __init__(self, shape=(2, 3), device=None):
        self.shape, self.device = shape, device


class Jagged:
    is_nested = True

    def __init__(self, shape=(2, -1), device=None, payload=None):
        self.shape, self.device = shape, device
        self.payload = object() if payload is None else payload
        self.moves = []

    def numpy(self):
        raise AssertionError("jagged payload must never become a NumPy/dense leaf")

    def to(self, device, non_blocking=False):
        self.moves.append((device, non_blocking))
        return Jagged(self.shape, device, self.payload)


def fake_package(version="0.10.0"):
    root = types.ModuleType("tensordict")
    root.__version__, root.__path__ = version, []
    base = types.ModuleType("tensordict.base")
    base._ACCEPTED_CLASSES = (Dense,)
    base._device_recorder = types.SimpleNamespace(marked=False, record_transfer=mock.Mock())
    base._register_tensor_class = lambda cls: None

    class TensorDictBase:
        batch_size, batch_dims, device = (2,), 1, None

        def _convert_to_tensor(self, value):
            return value.numpy() if hasattr(value, "numpy") else ("converted", value)

        def _validate_value_generic(self, value, non_blocking=False, *, check_shape=True):
            if isinstance(value, Dense):
                is_tc = False
            elif not isinstance(value, base._ACCEPTED_CLASSES):
                return self._convert_to_tensor(value)
            # Faithful 0.10 failure: registered non-Tensor leaves don't assign is_tc.
            if check_shape and not is_tc:
                return value
            return value

        def _validate_value_devicefree(self, value, non_blocking=False, *, check_shape=True):
            return self._validate_value_generic(value, non_blocking, check_shape=check_shape)

        def _validate_value_batchfree(self, value, non_blocking=False, *, check_shape=True):
            if not isinstance(value, base._ACCEPTED_CLASSES):
                return self._convert_to_tensor(value)
            return value

        def _validate_value_batchfree_devicefree(self, value, non_blocking=False, *, check_shape=True):
            return self._validate_value_batchfree(value, non_blocking, check_shape=check_shape)

    base.TensorDictBase = TensorDictBase
    alias = types.ModuleType("tensordict._td")
    alias._ACCEPTED_CLASSES = (Dense, TensorDictBase)
    root.base = base
    return root, base, alias


class TestTensorDictAdapter(unittest.TestCase):
    def setUp(self):
        self.root, self.base, self.alias = fake_package()
        self.owner = self.base.TensorDictBase
        self.initial_registry = self.base._ACCEPTED_CLASSES
        self.initial_alias = self.alias._ACCEPTED_CLASSES
        self.original_validators = {name: getattr(self.owner, name) for name in adapter._VALIDATORS}
        frontend = types.ModuleType("jittor.compat.torch")
        frontend.nested_tensor_type = lambda: Jagged
        frontend.nested_tensor_rebuild = mock.Mock()
        self.modules = mock.patch.dict(sys.modules, {
            "jittor": types.ModuleType("jittor"), "jittor.compat": compat,
            "jittor.compat.torch": frontend, "jittor.compat.module_patcher": patcher,
            "tensordict": self.root, "tensordict.base": self.base, "tensordict._td": self.alias,
        })
        self.modules.start()
        # Remove real optional submodules temporarily, so tests cannot patch a
        # package imported by unrelated earlier test collection.
        for name in tuple(sys.modules):
            if name.startswith("tensordict.") and name not in ("tensordict.base", "tensordict._td"):
                sys.modules.pop(name)
        self.old_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()
        adapter.register(patcher.register_module_patch)
        self.hook = ("module_patch", "tensordict", adapter.patch_tensordict)

    def tearDown(self):
        transaction.release_runtime_hooks(self.hook)
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.old_registry)
        self.modules.stop()

    def apply(self):
        return patcher._apply_module_patches(self.root)

    def test_initial_lossy_conversion_is_reproduced(self):
        with self.assertRaisesRegex(AssertionError, "NumPy"):
            self.owner()._validate_value_generic(Jagged())
        self.base._ACCEPTED_CLASSES += (Jagged,)
        with self.assertRaises(UnboundLocalError):
            self.owner()._validate_value_generic(Jagged())

    def test_registration_preserves_exact_leaf_and_payload(self):
        self.apply()
        source = Jagged()
        for name in (*adapter._VALIDATORS, "_validate_value_batchfree", "_validate_value_batchfree_devicefree"):
            with self.subTest(name=name):
                result = getattr(self.owner(), name)(source)
                self.assertIs(result, source)
                self.assertIs(result.payload, source.payload)
        self.assertIn(Jagged, self.base._ACCEPTED_CLASSES)
        self.assertIn(Jagged, self.alias._ACCEPTED_CLASSES)
        self.assertIn(self.owner, self.alias._ACCEPTED_CLASSES)
        self.assertFalse(isinstance(source, Dense))

    def test_dense_and_non_tensor_inputs_keep_original_behavior(self):
        self.apply()
        owner, dense = self.owner(), Dense()
        self.assertIs(owner._validate_value_generic(dense), dense)
        self.assertEqual(owner._validate_value_generic(5), ("converted", 5))

    def test_batch_shape_mismatch_and_check_shape_false(self):
        self.apply()
        value = Jagged(shape=(3, -1))
        for name in adapter._VALIDATORS:
            with self.assertRaisesRegex(RuntimeError, "batch dimension mismatch"):
                getattr(self.owner(), name)(value)
            self.assertIs(getattr(self.owner(), name)(value, check_shape=False), value)

    def test_device_transfer_preserves_upstream_recording(self):
        self.apply()
        owner = self.owner()
        device = types.SimpleNamespace(type="npu", index=0)
        owner.device = device
        source = Jagged(device="cpu")
        self.base._device_recorder.marked = True
        result = owner._validate_value_generic(source, non_blocking=True)
        self.assertIs(result.device, device)
        self.assertIs(result.payload, source.payload)
        self.assertEqual(source.moves, [(device, True)])
        self.base._device_recorder.record_transfer.assert_called_once_with(device)
        self.assertIs(owner._validate_value_devicefree(source), source)

    def test_unsupported_version_or_contract_fails_before_any_write(self):
        self.root.__version__ = "0.11.0"
        with self.assertRaises(UnsupportedAdapterVersion):
            self.apply()
        self.assertIs(self.base._ACCEPTED_CLASSES, self.initial_registry)
        self.root.__version__ = "0.10.0"
        self.alias._ACCEPTED_CLASSES = [Dense]
        with self.assertRaises(UnsupportedAdapterVersion):
            self.apply()
        self.assertIs(self.base._ACCEPTED_CLASSES, self.initial_registry)

    def test_missing_validator_rejected_before_registration(self):
        self.owner._validate_value_devicefree = None
        with self.assertRaises(UnsupportedAdapterVersion):
            self.apply()
        self.assertIs(self.base._ACCEPTED_CLASSES, self.initial_registry)

    def test_idempotence_and_owned_rollback(self):
        self.apply()
        registered, validator = self.base._ACCEPTED_CLASSES, self.owner._validate_value_generic
        results = self.apply()
        self.assertEqual(results[0].status, "unchanged")
        self.assertIs(self.base._ACCEPTED_CLASSES, registered)
        self.assertIs(self.owner._validate_value_generic, validator)
        transaction.release_runtime_hooks(self.hook)
        self.assertIs(self.base._ACCEPTED_CLASSES, self.initial_registry)
        self.assertIs(self.alias._ACCEPTED_CLASSES, self.initial_alias)
        for name, original in self.original_validators.items():
            self.assertIs(getattr(self.owner, name), original)

    def test_partial_patch_failure_rolls_back_all_prior_writes(self):
        original = adapter.replace
        def fail(owner, name, replacement, expected):
            if name == "_validate_value_devicefree":
                raise RuntimeError("injected second-validator failure")
            return original(owner, name, replacement, expected)
        with mock.patch.object(adapter, "replace", fail), self.assertRaisesRegex(RuntimeError, "injected"):
            self.apply()
        self.assertIs(self.base._ACCEPTED_CLASSES, self.initial_registry)
        self.assertIs(self.alias._ACCEPTED_CLASSES, self.initial_alias)
        self.assertIs(self.owner._validate_value_generic, self.original_validators["_validate_value_generic"])

    def test_late_transport_source_failure_restores_root_patch(self):
        self._check_late_transport_failure('source_pin')

    def test_late_transport_alias_failure_restores_root_patch(self):
        self._check_late_transport_failure('alias_write')

    def _check_late_transport_failure(self, failure):
        from jittor_adapters.tensordict import jagged_rebuild, consolidated_transport
        sources = []
        for filename in ('_td_v010_rebuild_fixture.py', '_td_v010_transport_fixture.py'):
            spec = importlib.util.spec_from_file_location('_td_root_' + filename[:-3],
                                                         Path(__file__).with_name(filename))
            source = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(source)
            sources.append(source)
        utils = types.ModuleType(jagged_rebuild.MODULE)
        reductions = types.ModuleType(consolidated_transport.MODULE)
        originals = (
            (utils, jagged_rebuild.FUNCTION, sources[0]._rebuild_njt_from_njt),
            (self.base, jagged_rebuild.LEAF_FUNCTION, sources[0]._is_leaf_nontensor),
            (reductions, consolidated_transport.FUNCTION, sources[1]._rebuild_tensordict_files_consolidated),
        )
        for target, name, fixture in originals:
            setattr(target, name, types.FunctionType(fixture.__code__, vars(target), name))
        setattr(self.alias, jagged_rebuild.FUNCTION, getattr(utils, jagged_rebuild.FUNCTION))
        setattr(self.alias, consolidated_transport.FUNCTION,
                getattr(reductions, consolidated_transport.FUNCTION))
        targets = (self.root, self.base, self.alias, self.owner, utils, reductions)
        before = [(target, dict(vars(target))) for target in targets]
        original_alias_write = consolidated_transport.replace_bound_aliases
        def fail_aliases(*args):
            original_alias_write(*args)
            raise RuntimeError('injected late transport alias failure')
        injection = (mock.patch.object(consolidated_transport, 'FUNCTION_SHA256', '0' * 64)
                     if failure == 'source_pin' else
                     mock.patch.object(consolidated_transport, 'replace_bound_aliases', fail_aliases))
        exception = UnsupportedAdapterVersion if failure == 'source_pin' else RuntimeError
        with mock.patch.dict(sys.modules, {utils.__name__: utils, reductions.__name__: reductions}):
            with injection, self.assertRaises(exception):
                self.apply()
            for target, initial in before:
                self.assertEqual(set(vars(target)), set(initial))
                for name, value in initial.items():
                    self.assertIs(vars(target)[name], value, name)
            self.assertEqual(self.apply()[0].status, 'patched')
            self.assertEqual(self.apply()[0].status, 'unchanged')
            transaction.release_runtime_hooks(self.hook)
            for target, initial in before:
                self.assertEqual(set(vars(target)), set(initial))
                for name, value in initial.items():
                    self.assertIs(vars(target)[name], value, name)

    def test_external_replacement_survives_conflicting_rollback(self):
        self.apply()
        external = (Dense, Jagged, str)
        self.base._ACCEPTED_CLASSES = external
        with self.assertRaises(transaction.TransactionConflict):
            transaction.release_runtime_hooks(self.hook)
        self.assertIs(self.base._ACCEPTED_CLASSES, external)
        self.assertIs(self.alias._ACCEPTED_CLASSES, self.initial_alias)

    def test_registrar_does_not_import_optional_package_or_create_tensor(self):
        calls = []
        with mock.patch.object(builtins, "__import__", side_effect=AssertionError("unexpected import")):
            adapter.register(lambda name, callback: calls.append((name, callback)))
        self.assertEqual(calls, [("tensordict.utils", adapter.patch_deferred_rebuild),
                                 ("tensordict._reductions", adapter.patch_deferred_transport),
                                 ("tensordict", adapter.patch_tensordict)])

    def test_real_import_hook_waits_for_root_version_publication(self):
        for name in ("tensordict", "tensordict.base", "tensordict._td"):
            sys.modules.pop(name)
        # Supply host classes separately; the imported package itself only
        # publishes its version AFTER importing its base module, like 0.10.
        support = types.ModuleType("_td_import_host_fixture")
        support.owner = self.owner
        support.recorder = self.base._device_recorder
        support.Dense = Dense
        with mock.patch.dict(sys.modules, {support.__name__: support}), tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "tensordict"
            package.mkdir()
            (package / "__init__.py").write_text('from . import base\n__version__ = "0.10.0"\n')
            (package / "base.py").write_text(
                "from _td_import_host_fixture import owner as TensorDictBase, recorder as _device_recorder, Dense\n"
                "_ACCEPTED_CLASSES = (Dense,)\n"
                "def _register_tensor_class(cls): pass\n")
            sys.path.insert(0, directory)
            try:
                patcher.install_module_patches(load_entry_points=False)
                imported = importlib.import_module("tensordict")
                self.assertEqual(imported.__version__, "0.10.0")
                self.assertIn(Jagged, imported.base._ACCEPTED_CLASSES)
                source = Jagged()
                self.assertIs(imported.base.TensorDictBase()._validate_value_generic(source), source)
            finally:
                patcher.uninstall_module_patches()
                sys.path.remove(directory)


class TestRelocatableBoundary(unittest.TestCase):
    def test_adapter_uses_only_public_framework_access(self):
        for path in Path(adapter.__file__).parent.rglob("*.py"):
            if "tests" not in path.relative_to(Path(adapter.__file__).parent).parts:
                self.assertEqual(boundary_errors(path.read_text()), [], str(path))

    def test_gate_has_negative_controls(self):
        for source in ("from jittor.compat.torch.nested import _NestedTensor",
                       "torch.Tensor = object", "setattr(jittor, 'x', 1)", "torch.nested._internal"):
            self.assertTrue(boundary_errors(source), source)
        self.assertEqual(boundary_errors("from jittor.compat.torch import nested_tensor_type"), [])


if __name__ == "__main__":
    unittest.main()

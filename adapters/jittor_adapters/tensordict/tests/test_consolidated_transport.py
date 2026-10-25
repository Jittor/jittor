"""Exact private-helper rewrite and owned lifecycle; not numerical proof."""
import hashlib
from functools import wraps
import importlib.util
import inspect
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


support = _load('_td_transport_host_support', 'test_adapter.py')
compat, patcher, transaction = support.compat, support.patcher, support.transaction
fixture = _load('_td_transport_source', '_td_v010_transport_fixture.py')._rebuild_tensordict_files_consolidated
from jittor_adapters.tensordict import consolidated_transport as adapter
from jittor_adapters._common import UnsupportedAdapterVersion


class Storage:
    def __getitem__(self, key): return self
    def view(self, shape): return self
    def detach(self): return ('detached', self)


class Container:
    @classmethod
    def _from_dict_validated(cls, data, **kwargs):
        result = cls()
        result.data, result.metadata, result.locked = data, kwargs, False
        return result

    def lock_(self):
        self.locked = True
        return self


class TestConsolidatedTransport(unittest.TestCase):
    def setUp(self):
        self.root = types.ModuleType('tensordict')
        self.root.__version__ = '0.10.0'
        self.module = types.ModuleType(adapter.MODULE)
        self.original = types.FunctionType(fixture.__code__, vars(self.module), adapter.FUNCTION)
        setattr(self.module, adapter.FUNCTION, self.original)
        self.alias = types.ModuleType('tensordict.base')
        setattr(self.alias, adapter.FUNCTION, self.original)
        self.constructor = mock.Mock(side_effect=lambda *args, **kw: (args, kw))
        self.module.torch = types.SimpleNamespace(Size=tuple, device=lambda x: x,
            nested=types.SimpleNamespace(nested_tensor_from_jagged=self.constructor))
        self.module._STR_DTYPE_TO_DTYPE = {'float32': 'float32', 'int64': 'int64'}
        self.module._is_tensorclass = lambda cls: False
        self.module.CLS_MAP = {'Container': Container}
        self.modules = mock.patch.dict(sys.modules, {
            'jittor': types.ModuleType('jittor'), 'jittor.compat': compat,
            'jittor.compat.module_patcher': patcher, 'tensordict': self.root,
            adapter.MODULE: self.module, 'tensordict.base': self.alias,
        })
        self.modules.start()
        self.old_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()
        patcher.register_module_patch(adapter.MODULE, adapter.patch_deferred_transport)
        self.hook = ('module_patch', adapter.MODULE, adapter.patch_deferred_transport)

    def tearDown(self):
        transaction.release_runtime_hooks(self.hook)
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.old_registry)
        self.modules.stop()

    def apply(self): return patcher._apply_module_patches(self.module)

    def test_source_fixture_exact_hash(self):
        self.assertEqual(hashlib.sha256(inspect.getsource(fixture).encode()).hexdigest(), adapter.FUNCTION_SHA256)

    def test_live_globals_metadata_storage_and_detached_public_constructor(self):
        self.apply()
        function = getattr(self.module, adapter.FUNCTION)
        self.assertIs(function.__globals__, self.original.__globals__)
        self.assertEqual(inspect.signature(function), inspect.signature(self.original))
        metadata = dict(non_tensors={}, leaves={
            '<NJT_VALUES>n': ('float32', [7, 3], 0, 84, 0),
            '<NJT_LENGTHS>n': ('int64', [2], 84, 100, 0),
            '<NJT_OFFSETS>n': ('int64', [3], 100, 124, 0)},
            cls='Container', cls_metadata={'is_locked': True})
        storage = Storage()
        result = function(metadata, storage)
        self.assertTrue(result.locked)
        self.assertIs(result._consolidated['storage'], storage)
        self.assertIs(result._consolidated['metadata'], metadata)
        self.assertIn('n', result.data)
        args, kw = self.constructor.call_args
        self.assertEqual(args, (('detached', storage),))
        self.assertIs(kw['offsets'], storage)
        self.assertIs(kw['lengths'], storage)
        self.assertEqual(kw['jagged_dim'], 1)
        later = mock.Mock(return_value='later')
        self.module.torch.nested.nested_tensor_from_jagged = later
        self.assertEqual(function(metadata, storage).data['n'], 'later')

    def test_idempotence_aliases_and_owned_rollback(self):
        self.apply()
        current = getattr(self.module, adapter.FUNCTION)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), current)
        self.assertEqual(self.apply()[0].status, 'unchanged')
        transaction.release_runtime_hooks(self.hook)
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), self.original)

    def test_source_and_version_fail_before_write(self):
        self.root.__version__ = '0.11.0'
        with self.assertRaises(UnsupportedAdapterVersion): self.apply()
        self.root.__version__ = '0.10.0'
        with mock.patch.object(adapter, 'FUNCTION_SHA256', '0' * 64):
            with self.assertRaises(UnsupportedAdapterVersion): self.apply()
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)

    def test_ast_shape_guard_independent_of_hash(self):
        source = inspect.getsource(fixture)
        for changed in (source.replace('lengths=nested_lengths', 'lengths=None'),
                        source.replace('from torch.nested._internal.nested_tensor import NestedTensor', 'pass')):
            with self.subTest(source=changed), self.assertRaises(UnsupportedAdapterVersion):
                adapter._rewrite(self.original, changed)

    def test_foreign_function_rejected(self):
        setattr(self.module, adapter.FUNCTION, lambda *args: None)
        with self.assertRaises(UnsupportedAdapterVersion): self.apply()

    def test_copied_marker_does_not_approve_external_transport(self):
        self.apply()
        owned = getattr(self.module, adapter.FUNCTION)
        @wraps(owned)
        def foreign(*args, **kwargs):
            return 'not the adapter implementation'
        setattr(self.module, adapter.FUNCTION, foreign)
        try:
            with self.assertRaises(UnsupportedAdapterVersion):
                self.apply()
            self.assertIs(getattr(self.module, adapter.FUNCTION), foreign)
        finally:
            setattr(self.module, adapter.FUNCTION, owned)

    def test_external_replacement_not_overwritten(self):
        self.apply()
        external = lambda *args: None
        setattr(self.alias, adapter.FUNCTION, external)
        with self.assertRaises(transaction.TransactionConflict):
            transaction.release_runtime_hooks(self.hook)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), external)

    def test_root_version_published_late(self):
        del self.root.__version__
        self.assertEqual(self.apply()[0].status, 'unchanged')
        self.root.__version__ = '0.10.0'
        self.apply()
        self.assertIsNot(getattr(self.module, adapter.FUNCTION), self.original)

    def test_optional_module_absence_no_forced_import(self):
        sys.modules.pop(adapter.MODULE)
        with mock.patch('builtins.__import__', side_effect=AssertionError('forced import')):
            self.assertFalse(adapter.patch_loaded_transport())

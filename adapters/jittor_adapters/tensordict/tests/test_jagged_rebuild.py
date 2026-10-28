"""Owned private-dispatch contracts; numerical/device proof is a separate run."""
import hashlib
from functools import wraps
import inspect
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

# Sibling tests in other adapters also use test_adapter.py. Load the host
# support explicitly under a unique name; never depend on pytest's basename
# import mode or whichever unrelated test_adapter was collected first.
def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


support = _load_sibling('_td_rebuild_host_support', 'test_adapter.py')
compat, patcher, transaction, Jagged = support.compat, support.patcher, support.transaction, support.Jagged
source = _load_sibling('_td_rebuild_source_fixture', '_td_v010_rebuild_fixture.py')
fixture, leaf_fixture = source._rebuild_njt_from_njt, source._is_leaf_nontensor
from jittor_adapters.tensordict import jagged_rebuild as adapter
from jittor_adapters._common import UnsupportedAdapterVersion


class Leaf:
    def __init__(self, payload=None, requires_grad=True):
        self.payload = object() if payload is None else payload
        self.requires_grad = requires_grad

    def detach(self):
        return Leaf(self.payload, False)


class Symbol:
    """Inert metadata, not a framework or a concrete integer."""
    def __eq__(self, other):
        raise AssertionError('adapter must not compare symbolic dimensions')

    def __int__(self):
        raise AssertionError('adapter must not concretize symbolic dimensions')


SYMBOL = Symbol()


class Nested(Jagged):
    def __init__(self, shape=(2, SYMBOL, 3), lengths=None):
        super().__init__(shape)
        self.public_lengths = lengths

    def lengths(self):
        return self.public_lengths


class TestJaggedRebuild(unittest.TestCase):
    def setUp(self):
        self.root = types.ModuleType("tensordict")
        self.root.__version__ = "0.10.0"
        self.module = types.ModuleType(adapter.MODULE)
        self.original = types.FunctionType(fixture.__code__, vars(self.module), adapter.FUNCTION)
        setattr(self.module, adapter.FUNCTION, self.original)
        self.alias = types.ModuleType("tensordict.base")
        setattr(self.alias, adapter.FUNCTION, self.original)
        self.leaf_original = types.FunctionType(leaf_fixture.__code__, vars(self.alias), adapter.LEAF_FUNCTION)
        setattr(self.alias, adapter.LEAF_FUNCTION, self.leaf_original)
        self.alias._is_tensor_collection = lambda cls: cls is dict
        self.alias._pass_through_cls = lambda cls: False
        self.constructor = mock.Mock(side_effect=lambda values, **kw: (values, kw))
        torch = types.ModuleType("torch")
        torch.Tensor = Leaf
        self.alias.torch = torch
        torch.nested = types.SimpleNamespace(nested_tensor_from_jagged=self.constructor)
        frontend = types.ModuleType("jittor.compat.torch")
        frontend.nested_tensor_type = lambda: Nested
        def public_rebuild(template, values, offsets, *, lengths=None,
                           preserve_ragged_identity=False):
            axes = [i for i, size in enumerate(template.shape) if isinstance(size, Symbol)]
            if len(axes) != 1 or axes[0] == 0:
                raise UnsupportedAdapterVersion('invalid fixture public shape')
            self.last_template = template
            return self.constructor(values, offsets=offsets, lengths=lengths,
                jagged_dim=axes[0], preserve_ragged_identity=preserve_ragged_identity)
        frontend.nested_tensor_rebuild = public_rebuild
        self.modules = mock.patch.dict(sys.modules, {
            "jittor": types.ModuleType("jittor"), "jittor.compat": compat,
            "jittor.compat.torch": frontend, "jittor.compat.module_patcher": patcher,
            "torch": torch, "tensordict": self.root, adapter.MODULE: self.module,
            "tensordict.base": self.alias,
        })
        self.modules.start()
        self.old_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()
        patcher.register_module_patch(adapter.MODULE, adapter.patch_deferred_rebuild)
        patcher.register_module_patch('tensordict.base', adapter.patch_leaf_predicate)
        self.hook = ("module_patch", adapter.MODULE, adapter.patch_deferred_rebuild)
        self.leaf_hook = ("module_patch", 'tensordict.base', adapter.patch_leaf_predicate)

    def tearDown(self):
        transaction.release_runtime_hooks(self.hook)
        transaction.release_runtime_hooks(self.leaf_hook)
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.old_registry)
        self.modules.stop()

    def apply(self):
        return patcher._apply_module_patches(self.module)

    def test_exact_upstream_fixture_hash(self):
        self.assertEqual(hashlib.sha256(inspect.getsource(fixture).encode()).hexdigest(), adapter.FUNCTION_SHA256)
        self.assertEqual(hashlib.sha256(inspect.getsource(leaf_fixture).encode()).hexdigest(), adapter.LEAF_SHA256)

    def test_leaf_predicate_only_registers_exact_nested_family(self):
        self.assertFalse(self.leaf_original(Nested))
        patcher._apply_module_patches(self.alias)
        function = getattr(self.alias, adapter.LEAF_FUNCTION)
        self.assertTrue(function(Nested))
        self.assertTrue(function(type('Child', (Nested,), {})))
        self.assertTrue(function(Leaf))
        self.assertFalse(function(dict))
        self.assertFalse(function(str))
        transaction.release_runtime_hooks(self.leaf_hook)
        self.assertIs(getattr(self.alias, adapter.LEAF_FUNCTION), self.leaf_original)

    def test_leaf_predicate_source_fail_closed(self):
        with mock.patch.object(adapter, 'LEAF_SHA256', '0' * 64):
            with self.assertRaises(UnsupportedAdapterVersion):
                patcher._apply_module_patches(self.alias)
        self.assertIs(getattr(self.alias, adapter.LEAF_FUNCTION), self.leaf_original)

    def test_shared_payload_detaches_axis_and_lengths_publicly(self):
        self.apply()
        function = getattr(self.module, adapter.FUNCTION)
        for shape in ((2, SYMBOL, 3), (2, 3, SYMBOL)):
            for holes in (False, True):
                with self.subTest(shape=shape, holes=holes):
                    leaf, offsets, lengths = Leaf(), object(), object()
                    template = Nested(shape, object() if holes else None)
                    values, kw = function(template, leaf, offsets, lengths)
                    self.assertIs(self.last_template, template)
                    self.assertIs(values.payload, leaf.payload)
                    self.assertFalse(values.requires_grad)
                    self.assertTrue(leaf.requires_grad)
                    self.assertIs(kw['offsets'], offsets)
                    self.assertIs(kw['lengths'], lengths if holes else None)
                    self.assertEqual(kw['jagged_dim'], next(i for i, size in enumerate(shape)
                                                          if isinstance(size, Symbol)))
                    self.assertIs(kw['preserve_ragged_identity'], holes)

    def test_other_leaf_preserves_original_error(self):
        self.apply()
        with self.assertRaises(ImportError):
            getattr(self.module, adapter.FUNCTION)(object(), Leaf(), object(), None)
        self.constructor.assert_not_called()

    def test_invalid_public_shape_fails_closed(self):
        self.apply()
        for shape in ((2, 3), (-1, 2), (2, -1, 3), (SYMBOL, 2), (2, SYMBOL, SYMBOL)):
            with self.subTest(shape=shape), self.assertRaises(UnsupportedAdapterVersion):
                getattr(self.module, adapter.FUNCTION)(Nested(shape), Leaf(), object(), None)
        self.constructor.assert_not_called()

    def test_public_constructor_error_propagates(self):
        self.apply()
        self.constructor.side_effect = RuntimeError('device construction failed')
        with self.assertRaisesRegex(RuntimeError, 'device construction failed'):
            getattr(self.module, adapter.FUNCTION)(Nested(), Leaf(), object(), None)

    def test_holey_template_with_absent_new_lengths_uses_plain_metadata(self):
        self.apply()
        values, kw = getattr(self.module, adapter.FUNCTION)(
            Nested(lengths=object()), Leaf(), object(), None)
        self.assertIsNone(kw['lengths'])
        self.assertIs(kw['preserve_ragged_identity'], False)
        self.assertFalse(values.requires_grad)

    def test_unsupported_version_or_source_no_writes(self):
        self.root.__version__ = "0.11.0"
        with self.assertRaises(UnsupportedAdapterVersion): self.apply()
        self.root.__version__ = "0.10.0"
        with mock.patch.object(adapter, 'FUNCTION_SHA256', '0' * 64):
            with self.assertRaises(UnsupportedAdapterVersion): self.apply()
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), self.original)

    def test_foreign_function_ownership_rejected(self):
        setattr(self.module, adapter.FUNCTION, lambda *args: None)
        with self.assertRaises(UnsupportedAdapterVersion): self.apply()
        self.assertIs(getattr(self.alias, adapter.FUNCTION), self.original)

    def test_copied_marker_does_not_approve_external_rebuild(self):
        self._check_copied_marker(self.module, adapter.FUNCTION, self.apply)

    def test_copied_marker_does_not_approve_external_leaf_predicate(self):
        self._check_copied_marker(self.alias, adapter.LEAF_FUNCTION,
                                 lambda: patcher._apply_module_patches(self.alias))

    def _check_copied_marker(self, target, name, apply):
        apply()
        owned = getattr(target, name)
        @wraps(owned)
        def foreign(*args, **kwargs):
            return 'not the adapter implementation'
        setattr(target, name, foreign)
        try:
            with self.assertRaises(UnsupportedAdapterVersion):
                apply()
            self.assertIs(getattr(target, name), foreign)
        finally:
            # Restore only this fixture's own foreign write before releasing.
            setattr(target, name, owned)

    def test_idempotent_aliases_owned_rollback(self):
        self.apply()
        first = getattr(self.module, adapter.FUNCTION)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), first)
        self.assertEqual(self.apply()[0].status, 'unchanged')
        transaction.release_runtime_hooks(self.hook)
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), self.original)

    def test_external_replacement_preserved_on_conflict(self):
        self.apply()
        external = lambda *args: None
        setattr(self.alias, adapter.FUNCTION, external)
        with self.assertRaises(transaction.TransactionConflict):
            transaction.release_runtime_hooks(self.hook)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), external)
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)

    def test_late_root_version_defers_then_patches(self):
        del self.root.__version__
        self.assertEqual(self.apply()[0].status, 'unchanged')
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        self.root.__version__ = '0.10.0'
        self.apply()
        self.assertIsNot(getattr(self.module, adapter.FUNCTION), self.original)

    def test_missing_optional_module_does_not_import(self):
        sys.modules.pop(adapter.MODULE)
        with mock.patch('builtins.__import__', side_effect=AssertionError('forced import')):
            self.assertFalse(adapter.patch_loaded_rebuild())

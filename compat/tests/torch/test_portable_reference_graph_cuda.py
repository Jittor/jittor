"""CUDA save/load preserves object references and exact dense storage aliases.

Not a complete checkpoint-resume or general strided-storage proof.
"""
from collections import OrderedDict
from contextlib import ExitStack
import io
import os
import sys
import unittest

import numpy as np
import pytest

pytestmark = pytest.mark.cuda


class TestPortableReferenceGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jt = sys.modules.get('jittor')
        if cls.jt is not None:
            import jittor as jt
            import torch
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            cls.jt = jt
        else:
            import torch
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(k == 'jittor' or k.startswith('jittor.') for k in sys.modules)
        cls.torch = torch

    def setUp(self):
        if not self.torch.cuda.is_available():
            self.skipTest('CUDA unavailable; not CUDA evidence')
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        if self.jt is not None:
            self.assertTrue(self.jt.has_cuda)
            # use_acl is a legacy alias of use_cuda, not backend identity.
            self.assertTrue(self.jt.compiler.is_cuda)
            self.assertFalse(self.jt.compiler.has_acl)
        if self.jt is not None:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            self.stack.enter_context(self.jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error'))
            self.stack.enter_context(forbid_backend_fallbacks())
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _materialize(self, value, target='cuda'):
        self.assertEqual(value.device.type, target)
        if self.jt is None:
            if target == 'cuda': self.torch.cuda.synchronize(value.device)
        else:
            native = getattr(value, '_var', value); native.sync()
            index = 0 if target == 'cuda' else -1
            self.assertEqual(native.location(), 'device' if target == 'cuda' else 'cpu')
            self.assertEqual(int(native.device_id), index)
            self.assertEqual(int(native.placement_backend), int(target == 'cuda'))
            self.assertEqual(self.jt.core.dispatch_context([native]), (target, index))

    def _descriptor(self, value):
        # No sync here: a later materialization must not hide source migration.
        row = (value.device.type, value.device.index, str(value.dtype), tuple(value.shape),
               tuple(value.stride()), value.storage_offset(), bool(value.requires_grad),
               bool(value.is_leaf), value.grad_fn is None)
        if self.jt is not None:
            native = getattr(value, '_var', value)
            row += (native.location(), int(native.device_id), int(native.placement_backend),
                    self.jt.core.dispatch_context([native]))
        return row

    def _equals(self, value, expected):
        self._materialize(value, value.device.type)
        before = self._descriptor(value)
        with self.torch.no_grad():
            clone = value.clone()
            self._materialize(clone, value.device.type)
            actual = np.array(clone.cpu().numpy(), copy=True)
        self.assertEqual(self._descriptor(value), before)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(bool(np.array_equal(actual, expected)), 'complete public values differ')

    def _roundtrip(self, obj, target):
        stream = io.BytesIO()
        self.torch.save(obj, stream)
        self.assertLess(stream.tell(), 1024*1024, 'bounded public fixture')
        stream.seek(0)
        return self.torch.load(stream, map_location=target, weights_only=True)

    def test_cuda_shared_reference_and_ordered_metadata(self):
        torch = self.torch
        want = np.arange(12, dtype=np.float32).reshape(3,4)
        for target in (None, 'cuda:0'):
            with self.subTest(target=target):
                x = torch.tensor(want, device='cuda:0')
                different = x.clone()
                self._materialize(x); self._materialize(different)
                before = (self._descriptor(x), self._descriptor(different))
                shared = [x]
                state = OrderedDict(a=shared, b=shared, other=different)
                state._metadata = OrderedDict([('', {'version': 1})])
                state.extra = {'same': shared, 'tensor': x}
                graph = {'state':state, 'again':state, 'attrs':vars(state)}
                loaded = self._roundtrip(graph, target)
                self.assertEqual((self._descriptor(x),self._descriptor(different)),before)
                got = loaded['state']
                self.assertIs(type(got), OrderedDict)
                self.assertIs(got, loaded['again'])
                # Pickle BUILD updates __dict__, so its container identity is
                # not a public serialization guarantee; attribute value aliases are.
                self.assertIs(got.extra, loaded['attrs']['extra'])
                self.assertIs(got._metadata, loaded['attrs']['_metadata'])
                self.assertEqual(got._metadata, state._metadata)
                self.assertIs(got['a'], got['b']); self.assertIs(got.extra['same'], got['a'])
                self.assertIs(got.extra['tensor'], got['a'][0])
                self.assertIsNot(got['other'], got['a'][0])
                self._equals(got['a'][0], want); self._equals(got['other'], want)
                with torch.no_grad(): got['a'][0].fill_(9)
                self._equals(got['b'][0], np.full((3,4),9,np.float32))
                self._equals(got['other'], want)
                self._equals(x, want); self._equals(different, want)
        if self.jt is not None: self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def test_cuda_parameter_storage_alias_keeps_independent_autograd(self):
        torch = self.torch
        want = np.arange(12, dtype=np.float32).reshape(3, 4)
        for target, tensor_first in ((t, first) for t in (None, 'cuda:0') for first in (False, True)):
            with self.subTest(target=target, tensor_first=tensor_first):
                x = torch.tensor(want, device='cuda:0')
                a = torch.nn.Parameter(x.detach(), requires_grad=False)
                b = x.detach().requires_grad_(True)
                c = x.clone()
                for value in (x, a, b, c): self._materialize(value)
                original = tuple(self._descriptor(value) for value in (x, a, b, c))
                graph = dict(b=b, a=a, c=c, again=a) if tensor_first else dict(a=a, b=b, c=c, again=a)
                loaded = self._roundtrip(graph, target)
                self.assertEqual(tuple(self._descriptor(value) for value in (x, a, b, c)), original)
                self.assertIsInstance(loaded['a'], torch.nn.Parameter)
                self.assertFalse(loaded['a'].requires_grad)
                self.assertIs(type(loaded['b']), torch.Tensor)
                self.assertTrue(loaded['b'].requires_grad)
                self.assertTrue(loaded['b'].is_leaf)
                self.assertIsNone(loaded['b'].grad_fn)
                self.assertIs(loaded['again'], loaded['a'])
                self.assertIsNot(loaded['a'], loaded['b'])
                # Loading shared storage creates independent autograd holders.
                # A write through a different loaded holder must not invalidate
                # b's saved version; its backward observes the shared bytes.
                loss = (loaded['b'] * loaded['b']).sum()
                self._materialize(loss)
                self._equals(loss, np.asarray((want * want).sum(), dtype=np.float32))
                with torch.no_grad(): loaded['a'].fill_(29)
                self._equals(loaded['b'], np.full((3, 4), 29, dtype=np.float32))
                loss.backward()
                self._equals(loaded['b'].grad, np.full((3, 4), 58, dtype=np.float32))
                self.assertIsNone(loaded['a'].grad)
                self._equals(loaded['c'], want)
                self._equals(x, want); self._equals(b, want)
                self.assertIsNone(b.grad)
                self.assertEqual(tuple(self._descriptor(value) for value in (x, a, b, c)), original)
        if self.jt is not None: self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def test_cuda_standalone_loaded_trainable_tensor_is_leaf(self):
        torch = self.torch
        want = np.arange(12, dtype=np.float32).reshape(3, 4)
        for target in (None, 'cuda:0'):
            with self.subTest(target=target):
                source = torch.tensor(want, device='cuda:0', requires_grad=True)
                self._materialize(source)
                before = self._descriptor(source)
                loaded = self._roundtrip(source, target)
                self.assertTrue(loaded.requires_grad)
                self.assertTrue(loaded.is_leaf)
                self.assertIsNone(loaded.grad_fn)
                (loaded * loaded).sum().backward()
                self._equals(loaded.grad, 2 * want)
                self.assertIsNone(source.grad)
                self.assertEqual(self._descriptor(source), before)
        if self.jt is not None: self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def test_cuda_distinct_detach_holders_share_dense_region(self):
        import numpy as np
        torch = self.torch
        want = np.arange(12, dtype=np.float32).reshape(3, 4)
        shared_value = np.full((3, 4), 17, dtype=np.float32)
        loaded_value = np.full((3, 4), 29, dtype=np.float32)
        for target in (None, 'cuda:0'):
            with self.subTest(target=target):
                x = torch.tensor(want, device='cuda:0')
                a, b, c = x.detach(), x.detach(), x.clone()
                self.assertIsNot(a, b)
                self.assertIsNot(a, x); self.assertIsNot(b, x)
                self.assertIsNot(c, a); self.assertIsNot(c, b)
                sources = (x, a, b, c)
                for value in sources:
                    self._materialize(value)
                    self.assertEqual(tuple(value.shape), (3, 4))
                    self.assertEqual(tuple(value.stride()), (4, 1))
                    self.assertEqual(value.storage_offset(), 0)
                    self._equals(value, want)
                # First prove the original detached holders actually share storage.
                # No pointer/holder-id is used as physical storage evidence.
                with torch.no_grad():
                    a.fill_(17)
                self._equals(a, shared_value)
                self._equals(b, shared_value)
                self._equals(x, shared_value)
                self._equals(c, want)
                source_before = tuple(self._descriptor(value) for value in sources)
                loaded = self._roundtrip(dict(a=a, b=b, c=c, again=a), target)
                # Inspect ORIGINAL descriptors before any post-load source sync.
                self.assertEqual(tuple(self._descriptor(value) for value in sources), source_before)
                self.assertIs(loaded['a'], loaded['again'])  # existing same-object memo boundary
                self.assertIsNot(loaded['a'], loaded['b'])   # independent Python holder boundary
                self.assertIsNot(loaded['a'], loaded['c'])
                for key in ('a', 'b', 'c'):
                    self._materialize(loaded[key])
                self._equals(loaded['a'], shared_value)
                self._equals(loaded['b'], shared_value)
                self._equals(loaded['c'], want)
                loaded_before = tuple(self._descriptor(loaded[key]) for key in ('a', 'b', 'c'))
                with torch.no_grad():
                    loaded['a'].fill_(29)
                self._equals(loaded['a'], loaded_value)
                self._equals(loaded['again'], loaded_value)
                self._equals(loaded['b'], loaded_value)  # distinct-holder STORAGE sharing
                self._equals(loaded['c'], want)          # equal clone must remain independent
                self.assertEqual(tuple(self._descriptor(loaded[key]) for key in ('a', 'b', 'c')), loaded_before)
                self.assertEqual(tuple(self._descriptor(value) for value in sources), source_before)
                self._equals(x, shared_value)
                self._equals(a, shared_value)
                self._equals(b, shared_value)
                self._equals(c, want)
                self.assertEqual(tuple(self._descriptor(value) for value in sources), source_before)
        if self.jt is not None:
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)




    def test_cuda_independent_carrier_payload_and_dense_alias_roundtrip(self):
        torch = self.torch
        if self.jt is not None:
            self.assertEqual(torch.Tensor.__bases__, (torch._C.TensorBase,))
            self.assertEqual(torch._C.TensorBase.__bases__, (object,))
            self.assertFalse(issubclass(torch.Tensor, self.jt.core.Var))
        want = np.asarray([2**40 + .5, -(2**39) - .25, 3.125], dtype=np.float64)
        wide_integer = np.asarray([2**40 + 1, -(2**45) + 3, 0], dtype=np.int64)
        for target in (None, 'cuda:0', {'cuda:0': 'cuda:0'}):
            with self.subTest(target=target):
                x = torch.tensor(want, dtype=torch.float64, device='cuda:0', requires_grad=True)
                alias = x.detach()
                parameter = torch.nn.Parameter(x.detach(), requires_grad=False)
                equal_clone = x.clone()
                integer = torch.tensor(wide_integer, dtype=torch.int64, device='cuda:0')
                scalar = torch.tensor(2**40 + .5, dtype=torch.float64, device='cuda:0')
                empty = torch.empty((0, 3), dtype=torch.float64, device='cuda:0')
                sources = (x, alias, parameter, equal_clone, integer, scalar, empty)
                for value in sources:
                    self._materialize(value)
                    if self.jt is not None:
                        self.assertTrue(self.jt.core._is_tensor_carrier(value))
                        self.assertFalse(isinstance(value, self.jt.core.Var))
                if self.jt is not None:
                    # Native fetch must copy every actual carrier, not pass a
                    # sibling Tensor through as an ordinary Python argument.
                    marker = object()
                    fetched = []
                    self.jt.fetch(marker, *sources, lambda *items: fetched.append(items))
                    self.jt.sync_all(True)
                    self.assertEqual(len(fetched), 1)
                    self.assertIs(fetched[0][0], marker)
                    expected_fetch = (want, want, want, want, wide_integer,
                        np.asarray(2**40 + .5, dtype=np.float64),
                        np.empty((0, 3), dtype=np.float64))
                    self.assertEqual(len(fetched[0]), len(sources) + 1)
                    for actual, expected in zip(fetched[0][1:], expected_fetch):
                        self.assertIs(type(actual), np.ndarray)
                        self.assertEqual(actual.dtype, expected.dtype)
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertTrue(bool(np.array_equal(actual, expected)))
                    with self.assertRaises(TypeError):
                        self.jt.fetch([x], lambda result: None)
                # Observable write-through, not only equal values or pointer IDs.
                with torch.no_grad():
                    alias.fill_(17)
                shared = np.full(want.shape, 17, dtype=np.float64)
                for value in (x, alias, parameter):
                    self._equals(value, shared)
                self._equals(equal_clone, want)
                before = tuple(self._descriptor(value) for value in sources)
                state = OrderedDict(value=x, same=x, alias=alias, parameter=parameter,
                    equal_clone=equal_clone, integer=integer, scalar=scalar, empty=empty, absent=None)
                state._metadata = OrderedDict([('', {'version': 1, 'optional': None})])
                state.extra = {'tensor': x, 'optional': None}
                loaded = self._roundtrip(state, target)  # Default weights_only=True.
                self.assertEqual(tuple(self._descriptor(value) for value in sources), before)
                self.assertIs(type(loaded), OrderedDict)
                self.assertEqual(loaded._metadata, state._metadata)
                self.assertIsNone(loaded['absent'])
                self.assertIsNone(loaded.extra['optional'])
                self.assertIs(loaded.extra['tensor'], loaded['value'])
                self.assertIs(loaded['same'], loaded['value'])
                self.assertIsNot(loaded['alias'], loaded['value'])
                self.assertIsNot(loaded['parameter'], loaded['value'])
                self.assertIsInstance(loaded['parameter'], torch.nn.Parameter)
                self.assertFalse(loaded['parameter'].requires_grad)
                self.assertIsNone(loaded['parameter'].grad)
                self.assertTrue(loaded['value'].requires_grad)
                self.assertTrue(loaded['value'].is_leaf)
                self.assertIsNone(loaded['value'].grad_fn)
                self.assertIsNone(loaded['value'].grad)
                expected = dict(value=shared, alias=shared, parameter=shared, equal_clone=want,
                    integer=wide_integer, scalar=np.asarray(2**40 + .5, dtype=np.float64),
                    empty=np.empty((0, 3), dtype=np.float64))
                for key, array in expected.items():
                    self._materialize(loaded[key])
                    self.assertIs(loaded[key].dtype, state[key].dtype)
                    self.assertEqual(tuple(loaded[key].shape), tuple(array.shape))
                    self._equals(loaded[key], array)
                # Frozen Parameter and distinct Tensor holders must retain physical
                # storage sharing; the equally-valued clone stays independent.
                with torch.no_grad():
                    loaded['parameter'].fill_(29)
                mutated = np.full(want.shape, 29, dtype=np.float64)
                for key in ('value', 'same', 'alias', 'parameter'):
                    self._equals(loaded[key], mutated)
                self._equals(loaded['equal_clone'], want)
                self._equals(x, shared)
                self.assertEqual(tuple(self._descriptor(value) for value in sources), before)
        if self.jt is not None:
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def test_cuda_independent_carrier_callable_location_and_owned_return(self):
        torch = self.torch
        want = np.asarray([2**40 + .5, -2**39 - .25], dtype=np.float64)
        source = torch.tensor(want, dtype=torch.float64, device='cuda:0')
        self._materialize(source)
        before = self._descriptor(source)
        calls = []
        def location(storage, saved_device):
            calls.append(saved_device)
            return None  # Public Torch contract: retain original saved location.
        loaded = self._roundtrip(source, location)
        self.assertEqual(calls, ['cuda:0'])
        self.assertIs(loaded.dtype, torch.float64)
        self._materialize(loaded)
        self._equals(loaded, want)
        self.assertEqual(self._descriptor(source), before)
        if self.jt is not None:
            # Private existing owner contract ONLY: unchanged implementation accepts
            # a native carrier as a callback result. Native Torch's public callback
            # receives/returns Storage; do not claim this Tensor return is that API.
            from jittor.compat.torch.serialization import portable
            replacement = torch.tensor([31.25, -17.5], dtype=torch.float64, device='cuda:0')
            self._materialize(replacement)
            self.assertTrue(self.jt.core._is_tensor_carrier(replacement))
            self.assertFalse(isinstance(replacement, self.jt.core.Var))
            owner_calls = []
            def owned_result(value, saved_device):
                owner_calls.append((value is source, saved_device))
                return replacement
            result = portable._apply_map_location_impl(source, owned_result, 0,
                {id(source): 'cuda:0'}, {})
            self.assertEqual(owner_calls, [(True, 'cuda:0')])
            self.assertIs(result, replacement)
            self._materialize(result)
            self._equals(result, np.asarray([31.25, -17.5], dtype=np.float64))
            self.assertEqual(self._descriptor(source), before)
            self.assertEqual(self.jt.core.backend_fallback_count(), 0)

if __name__ == '__main__':
    unittest.main()

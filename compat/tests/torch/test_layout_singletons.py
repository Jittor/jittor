"""CUDA-only public layout identities and finite constructor regression.

Unsupported strided nested construction remains explicit.
Class-object immutability is not supported by the Python owner and is not
claimed here. This is not a complete native layout/factory conformance test. This
file does not install a shim or implement math. Public fixture snapshots remain
RAM-only; all complete-value/flag/storage assertions print check names only.
"""
import copy
from contextlib import ExitStack
import os
import pickle
import sys
import unittest


class TestLayoutSingletons(unittest.TestCase):
    def _resident(self, value):
        self.assertTrue(value.device.type == 'cuda' and value.device.index == 0,
                        'public CUDA0 placement')
        if self.jt is not None and value.numel():
            native = getattr(value, '_var', value)
            native.sync()
            self.assertTrue(self.jt.core.dispatch_context([native]) == ('cuda', 0),
                            'actual CUDA dispatch')
            self.assertTrue(native.device_id == 0 and native.placement_backend == 1
                            and native.location() == 'device', 'actual CUDA storage')
        # Empty tensors have no bytes/allocation to certify physically.

    def _signature(self, value):
        return (tuple(value.shape), tuple(value.stride()), value.dtype,
                value.device.type, value.device.index, value.requires_grad,
                value.is_leaf, value.grad_fn is None, value.retains_grad,
                value.data_ptr(), value.storage_offset())

    def _snapshot(self, value):
        self._resident(value)
        with self.torch.no_grad():
            independent = value.clone()
            self.assertTrue(independent is not value and not independent.requires_grad,
                            'independent stopped snapshot')
            self._resident(independent)
            result = self.np.array(independent.cpu().numpy(), copy=True, order='C')
        self._resident(value)
        self.assertTrue(bool(self.np.isfinite(result).all()), 'complete finite snapshot')
        return result

    def _same_array(self, left, right):
        self.assertTrue(left.dtype == right.dtype and left.shape == right.shape,
                        'complete dtype/shape preserved')
        self.assertTrue(left.tobytes(order='C') == right.tobytes(order='C'),
                        'complete value bytes preserved')

    def _dense(self, value):
        torch = self.torch
        self._resident(value)
        before = self._signature(value)
        full = self._snapshot(value)
        for _ in range(3):
            self.assertTrue(type(value.layout) is torch.layout,
                            'actual public layout type')
            self.assertTrue(value.layout is torch.strided,
                            'dense layout does not mean contiguous')
        with self.assertRaises(AttributeError):
            value.layout = torch.jagged
        self.assertTrue(self._signature(value) == before, 'metadata query preserved flags/storage')
        self._same_array(full, self._snapshot(value))

    def _constants(self):
        torch = self.torch
        self.assertTrue(type(torch.layout) is type, "ordinary public layout class type")
        for value, name in ((torch.strided, 'strided'), (torch.jagged, 'jagged')):
            self.assertTrue(type(value) is torch.layout and not isinstance(value, str),
                            'typed non-string canonical layout')
            self.assertTrue(str(value) == repr(value) == 'torch.' + name,
                            'public canonical layout display')
            self.assertTrue(value != name and name != value, 'no string identity substitution')
            self.assertTrue(type(hash(value)) is int and {value: True}[value], 'identity hashing')
            self.assertTrue(copy.copy(value) is value and copy.deepcopy(value) is value,
                            'RAM copy preserves canonical identity')
            for protocol in (2, pickle.HIGHEST_PROTOCOL):
                self.assertTrue(pickle.loads(pickle.dumps(value, protocol=protocol)) is value,
                                'RAM pickle restores canonical identity')
            for field in ('name', '_name'):
                with self.assertRaises((AttributeError, TypeError)):
                    setattr(value, field, 'jagged')
                with self.assertRaises((AttributeError, TypeError)):
                    object.__setattr__(value, field, 'jagged')
            self.assertFalse(hasattr(value, '__dict__'), 'no mutable instance dictionary')
        self.assertTrue(torch.strided is not torch.jagged, 'distinct canonical layouts')
        for arguments in ((), ('strided',), (None,), (0,)):
            with self.assertRaises(TypeError):
                torch.layout(*arguments)
        with self.assertRaises(TypeError):
            type('UnapprovedLayoutSubclass', (torch.layout,), {})
        try:
            forged = object.__new__(torch.layout)
        except TypeError:
            # Native opaque type disallows this allocation. A Python owner can
            # allocate an inert forgery, but it must never acquire a label.
            forged = None
        if forged is not None:
            for operation in (str, repr, copy.copy, pickle.dumps):
                with self.assertRaises(TypeError):
                    operation(forged)

    def _jagged(self, axis, holes, zero, dtype):
        torch = self.torch
        off_values = [0, 3, 8] if holes else ([0, 0, 2] if zero else [0, 2, 5])
        width = off_values[-1]
        shape = (width, 2) if axis == 1 else (2, width)
        values = torch.arange(width * 2, dtype=dtype, device='cuda:0').reshape(shape)
        offsets = torch.tensor(off_values, dtype=torch.int64, device='cuda:0')
        lengths = (torch.tensor([0 if zero else 2, 2], dtype=torch.int64, device='cuda:0')
                   if holes else None)
        nested = torch.nested.nested_tensor_from_jagged(
            values, offsets, lengths=lengths, jagged_dim=axis)
        rows = nested.unbind(0)
        self.assertTrue(len(rows) == 2 and nested.is_nested, 'complete public jagged batch')
        leaves = [values, offsets, nested.values(), nested.offsets(), *rows]
        if holes:
            leaves.extend((lengths, nested.lengths()))
        before = [self._signature(value) for value in leaves]
        full = [self._snapshot(value) for value in leaves]
        parent = (nested.requires_grad, nested.is_leaf, nested.grad_fn is None,
                  nested.retains_grad, nested.device.type, nested.device.index, nested.dtype)
        for _ in range(3):
            self.assertTrue(type(nested.layout) is torch.layout and nested.layout is torch.jagged,
                            'jagged canonical typed layout')
            self.assertTrue(not nested.requires_grad and nested.is_leaf
                            and nested.grad_fn is None and not nested.retains_grad,
                            'data-only jagged flags remain public')
        with self.assertRaises(AttributeError):
            nested.layout = torch.strided
        for value in leaves:
            self._dense(value)
        self.assertTrue([self._signature(value) for value in leaves] == before,
                        'jagged buffers/rows flags and storage preserved')
        for expected, value in zip(full, leaves):
            self._same_array(expected, self._snapshot(value))
        self.assertTrue((nested.requires_grad, nested.is_leaf, nested.grad_fn is None,
                         nested.retains_grad, nested.device.type, nested.device.index, nested.dtype)
                        == parent, 'jagged parent metadata preserved')

    def _factory_layouts(self):
        torch = self.torch
        class DisplayOnly:
            def __repr__(self):
                return 'torch.strided'
            __str__ = __repr__
        factories = (
            ('empty', lambda **kw: torch.empty((2, 3), **kw)),
            ('zeros', lambda **kw: torch.zeros((2, 3), **kw)),
            ('ones', lambda **kw: torch.ones((2, 3), **kw)),
            ('full', lambda **kw: torch.full((2, 3), 2, **kw)),
            ('arange', lambda **kw: torch.arange(6, **kw)),
            ('eye', lambda **kw: torch.eye(2, 3, **kw)),
            ('empty_strided', lambda **kw: torch.empty_strided((2, 3), (3, 1), **kw)),
        )
        for name, factory in factories:
            for label, extra in (('omitted', {}), ('none', {'layout': None}),
                                 ('strided', {'layout': torch.strided}),
                                 ('jagged', {'layout': torch.jagged})):
                with self.subTest(factory=name, layout=label):
                    result = factory(device='cuda:0', dtype=torch.float32, **extra)
                    self._resident(result)
                    self.assertTrue(result.layout is torch.strided and not result.is_nested,
                                    'reviewed dense factory layout')
                    self.assertTrue(not result.requires_grad and result.is_leaf
                                    and result.grad_fn is None, 'default stopped dense leaf')
                    if name not in ('empty', 'empty_strided'):
                        if name == 'arange':
                            initial = self.np.arange(6, dtype=self.np.float32)
                        elif name == 'eye':
                            initial = self.np.eye(2, 3, dtype=self.np.float32)
                        else:
                            fill = {'zeros': 0, 'ones': 1, 'full': 2}[name]
                            initial = self.np.full((2, 3), fill, dtype=self.np.float32)
                        self._same_array(initial, self._snapshot(result))
                    # Never inspect uninitialized empty/empty_strided bytes.
                    with torch.no_grad():
                        result.fill_(3)
                    expected = self.np.full(tuple(result.shape), 3, dtype=self.np.float32)
                    self._same_array(expected, self._snapshot(result))
                    self._dense(result)
            for wrong in ('strided', 'torch.strided', DisplayOnly()):
                with self.subTest(factory=name, rejected_type=type(wrong).__name__):
                    with self.assertRaises(TypeError):
                        factory(device='cuda:0', layout=wrong)
        for constructor in (torch.tensor, torch.as_tensor):
            self._dense(constructor([1, 2], device='cuda:0'))
            for argument in (None, torch.strided, torch.jagged, 'strided', DisplayOnly()):
                with self.subTest(constructor=constructor.__name__, layout_type=type(argument).__name__):
                    with self.assertRaises(TypeError):
                        constructor([1, 2], device='cuda:0', layout=argument)

    def _nested_layout_consumers(self):
        torch = self.torch
        rows = [torch.arange(2, dtype=torch.float32, device='cuda:0'),
                torch.arange(3, dtype=torch.float32, device='cuda:0')]
        before = [self._snapshot(row) for row in rows]
        for constructor in (torch.nested.nested_tensor, torch.nested.as_nested_tensor):
            with self.subTest(nested_constructor=constructor.__name__):
                value = constructor(rows, layout=torch.jagged)
                self.assertTrue(value.is_nested and value.layout is torch.jagged,
                                'explicit canonical jagged list constructor')
                actual = value.unbind(0)
                self.assertTrue(len(actual) == len(rows), 'all jagged rows present')
                for expected, row in zip(before, actual):
                    self._same_array(expected, self._snapshot(row))
                for options in ({}, {'layout': None}, {'layout': torch.strided}):
                    if self.jt is None:
                        native = constructor(rows, **options)
                        self.assertTrue(native.is_nested and native.layout is torch.strided,
                                        'native strided nested public baseline')
                    else:
                        with self.assertRaises(NotImplementedError):
                            constructor(rows, **options)
                for wrong in ('strided', 'jagged', object()):
                    with self.assertRaises(RuntimeError):
                        constructor(rows, layout=wrong)
        dense = torch.arange(8, dtype=torch.float32, device='cuda:0').reshape(2, 4)
        jagged = torch.nested.narrow(dense, 1, 0, 2, layout=torch.jagged)
        self.assertTrue(jagged.layout is torch.jagged, 'canonical jagged narrow')
        for original, actual in zip(dense.unbind(0), jagged.unbind(0)):
            self._same_array(self._snapshot(original[:2]), self._snapshot(actual))
        for options in ({}, {'layout': torch.strided}):
            if self.jt is None:
                self.assertTrue(torch.nested.narrow(dense, 0, 0, 1, **options).layout
                                is torch.strided, 'native default narrow is strided')
            else:
                with self.assertRaises(NotImplementedError):
                    torch.nested.narrow(dense, 0, 0, 1, **options)
        for wrong in (None, 'strided', 'jagged', object()):
            with self.assertRaises(RuntimeError):
                torch.nested.narrow(dense, 1, 0, 2, layout=wrong)
        with self.assertRaises(RuntimeError):
            torch.nested.narrow(dense, 0, 0, 1, layout=torch.jagged)
        for expected, row in zip(before, rows):
            self._same_array(expected, self._snapshot(row))

    def test_cuda(self):
        if os.environ.get('JITTOR_TEST_DEVICES', 'cpu') != 'cuda':
            self.skipTest('CUDA-only layout test not selected by JITTOR_TEST_DEVICES=cuda')
        import torch
        import numpy as np
        self.torch, self.np = torch, np
        self.jt = sys.modules.get('jittor')
        if self.jt is None:
            self.assertTrue(hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context'),
                            'independent Native Torch oracle')
        else:
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            self.assertTrue(namespace_owner(torch) is self.jt and activation_status(self.jt).active,
                            'actual candidate public namespace')
            self.assertTrue(torch.Tensor is not self.jt.Var, 'frontend Tensor is not native Var')
        self.assertTrue(torch.cuda.is_available() and torch.cuda.current_device() == 0,
                        'selected CUDA device must actually be available')
        with ExitStack() as stack:
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(
                    use_cuda=1, device_id=0, backend_fallback='error'))
                stack.enter_context(forbid_backend_fallbacks())
                before = self.jt.core.backend_fallback_count()
                self.assertTrue(type(before) is int and before == 0, 'initial fallback zero')
            self._constants()
            self._factory_layouts()
            self._nested_layout_consumers()
            base = torch.arange(12, dtype=torch.float32, device='cuda:0').reshape(3, 4)
            parameter = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float32, device='cuda:0'))
            frozen = torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float32, device='cuda:0'),
                                        requires_grad=False)
            dense = (base, base.transpose(0, 1), base[1:, 1:],
                     torch.empty((0, 3), dtype=torch.float32, device='cuda:0'),
                     torch.tensor(2.0, dtype=torch.float32, device='cuda:0'), parameter, frozen)
            for ordinal, value in enumerate(dense):
                with self.subTest(dense=ordinal):
                    self._dense(value)
            for axis in (1, 2):
                for holes in (False, True):
                    for zero in (False, True):
                        for dtype in (torch.int64, torch.float32):
                            with self.subTest(axis=axis, holes=holes, zero=zero, dtype=str(dtype)):
                                self._jagged(axis, holes, zero, dtype)
            torch.cuda.synchronize()
            if self.jt is not None:
                after = self.jt.core.backend_fallback_count()
                self.assertTrue(type(after) is int and after == before == 0, 'final fallback zero')


if __name__ == '__main__':
    unittest.main()

"""CUDA-only live packed-metadata regression.

Data-only shared/independent offsets and lengths; complete arrays stay in RAM.
No symbolic shape, trainable nested-node, holes padding or PPO acceptance claim.
"""
from contextlib import ExitStack
import os
import sys
import unittest


class TestNestedPackedMetadata(unittest.TestCase):
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

    def _flags(self, value):
        return (value.requires_grad, value.is_leaf, value.grad_fn is None, value.retains_grad)

    def _signature(self, value):
        return (id(value), tuple(value.shape), tuple(value.stride()), value.dtype,
                value.device.type, value.device.index, self._flags(value),
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

    def _same(self, expected, actual):
        self.assertTrue(expected.shape == actual.shape and expected.dtype == actual.dtype,
                        'complete shape/dtype match')
        self.assertTrue(expected.tobytes(order='C') == actual.tobytes(order='C'),
                        'complete value bytes match')

    def _check(self, nested, values, offsets, lengths, off_state, len_state, full, axis, holes):
        torch, np = self.torch, self.np
        leaves = [values, offsets] + ([] if lengths is None else [lengths])
        before = [self._signature(t) for t in leaves]
        parent = (id(nested), self._flags(nested), nested.dtype, nested.device.type,
                  nested.device.index, nested.layout)
        packed = nested.values()
        self.assertTrue(packed.data_ptr() - packed.storage_offset()*packed.element_size()
            == values.data_ptr() - values.storage_offset()*values.element_size(),
            'public packed view shares original caller storage')
        self.assertTrue(nested.offsets() is offsets, 'exact caller offsets retained')
        self.assertTrue(nested.lengths() is lengths, 'exact caller lengths retained')
        self.assertTrue(parent[1] == (False, True, True, False), 'data-only parent flags')
        self.assertTrue(nested.layout is torch.jagged and len(nested) == 2,
                        'original typed jagged batch')
        self._same(full, self._snapshot(nested.values()))
        self._same(np.array(off_state, dtype=np.int64), self._snapshot(nested.offsets()))
        if len_state is not None:
            self._same(np.array(len_state, dtype=np.int64), self._snapshot(nested.lengths()))
        rows = nested.unbind(0)
        iterated = tuple(iter(nested))
        self.assertTrue(len(rows) == len(iterated) == 2, 'all current rows present')
        for i, row in enumerate(rows):
            start = off_state[i]
            end = off_state[i+1] if len_state is None else start + len_state[i]
            slices = [slice(None), slice(None)]; slices[axis-1] = slice(start, end)
            expected = np.array(full[tuple(slices)], copy=True, order='C')
            for observed in (row, nested[i], iterated[i]):
                self.assertTrue(self._flags(observed) == (False, True, True, False),
                                'data-only current row flags')
                self.assertTrue(observed.layout is torch.strided, 'dense row layout')
                self._same(expected, self._snapshot(observed))
                if observed.numel():
                    self.assertTrue(observed.data_ptr() - observed.storage_offset()*observed.element_size()
                        == values.data_ptr() - values.storage_offset()*values.element_size(),
                        'current row shares original packed storage')
            if row.numel():
                # Use a public integer tuple, not reshape() of a strided row,
                # which may allocate a copy and cannot prove write-through.
                coordinate = [0, 0]; coordinate[axis-1] = start
                changed = full.copy(); changed[tuple(coordinate)] += 11
                with torch.no_grad(): row[0, 0].add_(11)
                try:
                    self._same(changed, self._snapshot(values))
                finally:
                    with torch.no_grad(): row[0, 0].sub_(11)
                self._same(full, self._snapshot(values))
        if not holes:
            # Native 2.8 explicitly refuses holes padding. Do not invent a
            # shared positive contract for that unsupported operation.
            width = full.shape[axis-1]
            shape = (2, width, 2) if axis == 1 else (2, 2, width)
            expected = np.full(shape, -7, dtype=full.dtype)
            for i in range(2):
                a, b = off_state[i:i+2]
                if axis == 1: expected[i, :b-a, :] = full[a:b, :]
                else: expected[i, :, :b-a] = full[:, a:b]
            padded = torch.nested.to_padded_tensor(nested, -7, output_size=shape)
            self._same(expected, self._snapshot(padded))
        self.assertTrue([self._signature(t) for t in leaves] == before,
                        'metadata/row/padding reads preserve caller identity flags and storage')
        self.assertTrue((id(nested), self._flags(nested), nested.dtype, nested.device.type,
                         nested.device.index, nested.layout) == parent, 'parent identity/flags unchanged')

    def _geometry(self, axis, holes, zero, dtype):
        torch, np = self.torch, self.np
        off_state = [0, 3, 8] if holes else ([0, 0, 2] if zero else [0, 2, 5])
        len_state = [0 if zero else 2, 2] if holes else None
        width = off_state[-1]
        shape = (width, 2) if axis == 1 else (2, width)
        npdtype = np.int64 if dtype is torch.int64 else np.float32
        full = np.arange(width*2, dtype=npdtype).reshape(shape)
        values = torch.tensor(full, dtype=dtype, device='cuda:0')
        offsets = torch.tensor(off_state, dtype=torch.int64, device='cuda:0')
        lengths = None if not holes else torch.tensor(len_state, dtype=torch.int64, device='cuda:0')
        offset_clone = offsets.clone()
        length_clone = None if lengths is None else lengths.clone()
        off_independent = off_state.copy()
        len_independent = None if len_state is None else len_state.copy()
        variants = [('base', offsets, lengths, off_state, len_state),
                    ('same_metadata', offsets, lengths, off_state, len_state),
                    ('independent_offsets', offset_clone, lengths, off_independent, len_state)]
        if holes:
            variants.append(('independent_lengths', offsets, length_clone, off_state, len_independent))
        nested = [(label, torch.nested.nested_tensor_from_jagged(
            values, off, lengths=lens, jagged_dim=axis), off, lens, host_off, host_lens)
            for label, off, lens, host_off, host_lens in variants]
        phases = ('baseline', 'values_write', 'values_restore', 'offsets_write', 'offsets_restore')
        if holes: phases += ('lengths_write', 'lengths_restore')
        for phase in phases:
            with torch.no_grad():
                if phase == 'values_write': values[0, 0].add_(7); full[0, 0] += 7
                elif phase == 'values_restore': values[0, 0].sub_(7); full[0, 0] -= 7
                elif phase == 'offsets_write':
                    delta = -1 if holes or not zero else 1
                    offsets[1].add_(delta); off_state[1] += delta
                elif phase == 'offsets_restore':
                    delta = 1 if holes or not zero else -1
                    offsets[1].add_(delta); off_state[1] += delta
                elif phase == 'lengths_write':
                    delta = 1 if zero else -1
                    lengths[0].add_(delta); len_state[0] += delta
                elif phase == 'lengths_restore':
                    delta = -1 if zero else 1
                    lengths[0].add_(delta); len_state[0] += delta
            for label, value, off, lens, host_off, host_lens in nested:
                with self.subTest(axis=axis, holes=holes, zero=zero, dtype=str(dtype), variant=label, phase=phase):
                    self._check(value, values, off, lens, host_off, host_lens, full, axis, holes)

    def _list_constructor(self, constructor):
        torch, np = self.torch, self.np
        rows = [torch.arange(4, dtype=torch.int64, device='cuda:0').reshape(2, 2),
                torch.arange(6, dtype=torch.int64, device='cuda:0').reshape(3, 2)]
        original = [self._snapshot(t) for t in rows]
        nested = constructor(rows, layout=torch.jagged)
        values, offsets = nested.values(), nested.offsets()
        full = np.concatenate(original, axis=0)
        off_state = [0, 2, 5]
        for phase, delta in (('baseline', 0), ('offsets_write', -1), ('offsets_restore', 1)):
            if delta:
                with torch.no_grad(): offsets[1].add_(delta)
                off_state[1] += delta
            with self.subTest(list_constructor=constructor.__name__, phase=phase):
                self._check(nested, values, offsets, None, off_state, None, full, 1, False)
        for expected, row in zip(original, rows):
            self._same(expected, self._snapshot(row))

    def test_cuda(self):
        if os.environ.get('JITTOR_TEST_DEVICES', 'cpu') != 'cuda':
            self.skipTest('CUDA-only packed metadata test not selected by JITTOR_TEST_DEVICES=cuda')
        import torch
        import numpy as np
        self.torch, self.np, self.jt = torch, np, sys.modules.get('jittor')
        if self.jt is None:
            self.assertTrue(hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context'),
                            'independent native oracle')
            self.assertTrue(not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules),
                            'no candidate modules in native oracle')
        else:
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            self.assertTrue(namespace_owner(torch) is self.jt and activation_status(self.jt).active,
                            'actual candidate namespace')
        self.assertTrue(torch.cuda.is_available() and torch.cuda.current_device() == 0, 'actual CUDA0 required')
        with ExitStack() as stack:
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error'))
                stack.enter_context(forbid_backend_fallbacks())
                before = self.jt.core.backend_fallback_count()
                self.assertTrue(type(before) is int and before == 0, 'initial zero fallback')
            for axis in (1, 2):
                for holes in (False, True):
                    for zero in (False, True):
                        for dtype in (torch.int64, torch.float32):
                            with self.subTest(fixture_axis=axis, fixture_holes=holes,
                                              fixture_zero=zero, fixture_dtype=str(dtype)):
                                self._geometry(axis, holes, zero, dtype)
            for constructor in (torch.nested.nested_tensor, torch.nested.as_nested_tensor):
                with self.subTest(fixture_constructor=constructor.__name__):
                    self._list_constructor(constructor)
            torch.cuda.synchronize()
            if self.jt is not None:
                after = self.jt.core.backend_fallback_count()
                self.assertTrue(type(after) is int and after == before == 0, 'final zero fallback')


if __name__ == '__main__': unittest.main()

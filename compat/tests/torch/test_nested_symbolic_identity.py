"""CUDA-only jagged metadata identity, lifetime and copying regression.

Paired execution and loaded-source/device attestation belong to the runner.
Only public synthetic CUDA inputs; no model, config, logs or tensor files.
Private owner-state tests prove the helper boundary, not complete installation.
"""
from contextlib import ExitStack
import copy
import gc
import os
import pickle
import sys
import unittest
import weakref

class TestJaggedSymIntIdentityCUDA(unittest.TestCase):
    def _setup_cuda(self):
        import torch
        self.torch, self.jt = torch, sys.modules.get('jittor')
        if self.jt is None:
            if not hasattr(torch, '_C') or any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules):
                raise RuntimeError('independent native oracle required')
        else:
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            if namespace_owner(torch) is not self.jt or not activation_status(self.jt).active:
                raise RuntimeError('actual candidate namespace required')
        if not torch.cuda.is_available() or torch.cuda.current_device() != 0:
            raise RuntimeError('actual CUDA0 required, no CPU substitute')

    def _scope(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        if self.jt is not None:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            self.stack.enter_context(self.jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error'))
            self.stack.enter_context(forbid_backend_fallbacks())
            self.assertTrue(self.jt.core.backend_fallback_count() == 0, 'initial fallback zero')
        self.addCleanup(self._healthy)

    def _healthy(self):
        self.torch.cuda.synchronize()
        if self.jt is not None:
            self.assertTrue(self.jt.core.backend_fallback_count() == 0, 'final fallback zero')

    def _resident(self, tensor):
        if self.jt is not None:
            native = getattr(tensor, '_var', tensor)
            native.sync()
            self.assertTrue(self.jt.core.dispatch_context([native]) == ('cuda', 0), 'actual dispatch')
            # A raw native Var may use FollowRuntime (-1), which is not CPU.
            # Keep actual allocator residency and dispatch separate from that
            # policy. Public Torch tensors still require explicit CUDA placement.
            follows_cuda = (type(tensor) is self.jt.core.Var
                            and native.placement_backend == -1 and self.jt.flags.use_cuda == 1)
            self.assertTrue(native.location() == 'device' and native.device_id == 0,
                            'actual device storage')
            self.assertTrue(native.placement_backend == 1 or follows_cuda,
                            'explicit CUDA or exact native CUDA runtime policy')
        if self.jt is None or type(tensor) is not self.jt.core.Var:
            self.assertTrue(tensor.device.type == 'cuda' and tensor.device.index == 0, 'actual CUDA0')

    def _signature(self, tensor):
        self._resident(tensor)
        legacy = self.jt is not None and type(tensor) is self.jt.core.Var
        flags = ((tensor.requires_grad, tensor.is_backward_leaf, tensor.grad_fn_node_id)
                 if legacy else (tensor.requires_grad, tensor.is_leaf, tensor.grad_fn is None, tensor.retains_grad))
        self.assertTrue((type(flags[0]) is bool and type(flags[1]) is bool and type(flags[2]) is int)
                        if legacy else all(type(flag) is bool for flag in flags), 'exact native/public flag types')
        pointer = tensor._storage_address if legacy else tensor.data_ptr()
        offset = tensor._storage_offset() if legacy else tensor.storage_offset()
        self.assertTrue(type(pointer) is int and type(offset) is int, 'exact pointer/offset integer metadata')
        if legacy:
            # Read a native Var directly, without constructing a frontend Tensor:
            # its constructor's default placement is not this native Var's API.
            # This is explicit D2H transfer only, never host Tensor arithmetic.
            host = tensor._copy_to_cpu()
            host.sync()
            self.assertTrue(host is not tensor and host.location() == 'cpu'
                            and host._storage_address != pointer, 'independent native host snapshot')
            body = host.numpy().tobytes(order='C')
        else:
            with self.torch.no_grad():
                body = tensor.detach().clone().cpu().numpy().tobytes(order='C')
        self._resident(tensor)
        after_flags = ((tensor.requires_grad, tensor.is_backward_leaf, tensor.grad_fn_node_id)
                       if legacy else (tensor.requires_grad, tensor.is_leaf, tensor.grad_fn is None, tensor.retains_grad))
        self.assertTrue(after_flags == flags and pointer == (tensor._storage_address if legacy else tensor.data_ptr())
            and offset == (tensor._storage_offset() if legacy else tensor.storage_offset()), 'snapshot leaves original flags/storage unchanged')
        return (id(tensor), pointer, tuple(tensor.shape), offset,
                flags, body)

    def _fixture(self, axis, holes, zero):
        torch = self.torch
        offsets = torch.tensor([0, 3, 8] if holes else ([0, 0, 2] if zero else [0, 2, 5]),
                               dtype=torch.int64, device='cuda:0')
        width = 8 if holes else (2 if zero else 5)
        values = torch.arange(width * 2, dtype=torch.int64, device='cuda:0').reshape(
            (width, 2) if axis == 1 else (2, width))
        lengths = torch.tensor([0 if zero else 2, 2], dtype=torch.int64, device='cuda:0') if holes else None
        return values, offsets, lengths

    def _public_key_identity_and_preservation(self):
        torch = self.torch
        self.assertTrue(torch.SymInt is not int and not isinstance(3, torch.SymInt), 'ordinary int exclusion')
        for axis in (1, 2):
            for holes in (False, True):
                for zero in (False, True):
                    with self.subTest(axis=axis, holes=holes, zero=zero):
                        values, offsets, lengths = self._fixture(axis, holes, zero)
                        cloned_offsets = offsets.clone()
                        cloned_lengths = lengths.clone() if holes else None
                        def make(off, lens):
                            return torch.nested.nested_tensor_from_jagged(values, off, lengths=lens, jagged_dim=axis)
                        base, same, offset_clone = make(offsets, lengths), make(offsets, lengths), make(cloned_offsets, lengths)
                        sources = [values, offsets, cloned_offsets] + ([lengths, cloned_lengths] if holes else [])
                        before = [self._signature(t) for t in sources]
                        term = base.shape[axis]
                        for repeated in (base.shape[axis], base.size(axis), base.size()[axis]):
                            self.assertTrue(type(repeated) is torch.SymInt and term is not repeated
                                and bool(term == repeated) and hash(term) == hash(repeated),
                                'same parent: fresh shape/size wrapper, equal logical identity/hash')
                        self.assertTrue(term is not same.shape[axis] and bool(term == same.shape[axis])
                            and hash(term) == hash(same.shape[axis]), 'same actual key, fresh wrapper')
                        self.assertTrue(bool(term == offset_clone.shape[axis]) is holes, 'lengths overrides offsets key')
                        if holes:
                            length_clone = make(offsets, cloned_lengths)
                            self.assertTrue(not bool(term == length_clone.shape[axis]), 'cloned lengths are independent')
                        self.assertTrue(type(term * 0) is torch.SymInt and type(2 * term) is torch.SymInt,
                                        'constant products remain symbolic')
                        self.assertTrue(bool(term) and bool(term >= 2) and not bool(term >= 2 * term), 'public constant relations')
                        self.assertTrue(before == [self._signature(t) for t in sources], 'all original bytes/flags/storage unchanged')

    def _public_weak_key_gc_old_term_survives(self):
        torch = self.torch
        def make_term():
            values, offsets, lengths = self._fixture(1, False, False)
            nested = torch.nested.nested_tensor_from_jagged(values, offsets, jagged_dim=1)
            term = nested.shape[1]
            return term, weakref.ref(offsets)
        term, key = make_term()
        gc.collect()
        self.assertTrue(key() is None, 'held term must not retain the key Tensor')
        self.assertTrue(bool(term == term) and bool(2 * term == term * 2), 'old term survives key collection')
        new_term, new_key = make_term()
        gc.collect()
        self.assertTrue(new_key() is None and not bool(term == new_term), 'fresh key never inherits an old token')

    def _public_symbol_and_Size_copy_contract(self):
        torch = self.torch
        values, offsets, lengths = self._fixture(1, False, False)
        nested = torch.nested.nested_tensor_from_jagged(values, offsets, jagged_dim=1)
        original = [self._signature(values), self._signature(offsets)]
        term = nested.shape[1]
        size = nested.shape
        cases = (('term', term, 1), ('times2', term * 2, 2), ('times0', term * 0, 0), ('Size', size, 1))
        copies = (('copy', copy.copy), ('deepcopy', copy.deepcopy),
                  ('pickle4', lambda x: pickle.loads(pickle.dumps(x, protocol=4))))
        count = 0
        for label, value, coefficient in cases:
            for method, function in copies:
                with self.subTest(symbol_case=label, copying=method):
                    result = function(value)
                    self.assertTrue(result is not value and bool(result == value)
                                    and hash(result) == hash(value), 'fresh copied metadata, same logical identity/hash')
                    copied_term = result[1] if label == 'Size' else result
                    self.assertTrue(type(copied_term) is torch.SymInt
                        and type(copied_term.node.nested_int_coeff()) is int
                        and copied_term.node.nested_int_coeff() == coefficient, 'complete copied coefficient')
                    count += 1
        self.assertTrue(count == 12, 'complete term/2s/0s/Size copy census')
        self.assertTrue(original == [self._signature(values), self._signature(offsets)], 'symbol copying leaves CUDA inputs unchanged')

    def _legacy(self):
        # Direct native constructor, not a fake Python class or frontend cast.
        native = self.jt.core.Var([0, 2, 5]).cast('int64').stop_grad()
        self.assertTrue(type(native) is self.jt.core.Var, 'actual legacy native Var')
        with self.assertRaises(TypeError):
            weakref.ref(native)
        self._resident(native)
        return native

    def _candidate_owner_and_legacy_cache_copy(self):
        if self.jt is None:
            return  # Candidate-private helper is not a Native oracle API.
        from jittor.compat.torch.fidelity import Fidelity, fidelity_of
        record = fidelity_of('torch.SymInt')
        self.assertTrue(record.implementation is self.torch.SymInt and record.level is Fidelity.APPROXIMATE,
                        'actual public finite SymInt fidelity owner')
        self.assertTrue('jagged' in record.detail and 'allocation' in record.detail
                        and 'generic' in record.detail, 'explicit finite metadata fidelity boundary')
        from jittor.compat.torch.tensor_state import TorchTensorState, get_tensor_state
        from jittor.compat.torch.tensor_object_state import get_tensor_object_state
        torch = self.torch
        sources = [torch.tensor([0, 2, 5], dtype=torch.int64, device='cuda:0'), self._legacy()]
        for source in sources:
            with self.subTest(legacy=(type(source) is self.jt.core.Var)):
                before = self._signature(source)
                first, second = TorchTensorState(), TorchTensorState()
                a, b = first.nested_symint(source), second.nested_symint(source)
                self.assertTrue(a is first.nested_symint(source) and not bool(a == b), 'opaque owner isolation')
                owner_ref = weakref.ref(first.nested_symbol_owner)
                del first
                gc.collect()
                self.assertTrue(owner_ref() is None and bool(a == a), 'symbol does not retain owner')
                # New holders must not inherit source identity through standard
                # clone/copy/deepcopy/pickle. Legacy rebuild has existing ambient
                # device semantics, so declare the CUDA construction context.
                with torch.device('cuda:0'):
                    copies = [source.clone(), copy.copy(source), copy.deepcopy(source),
                              pickle.loads(pickle.dumps(source, protocol=4))]
                for other in copies:
                    self._resident(other)
                    self.assertTrue(other is not source and not bool(second.nested_symint(other) == b),
                                    'new holder receives fresh identity')
                    self.assertTrue(self._signature(other)[-1] == before[-1], 'copied full public bytes')
                if type(source) is self.jt.core.Var:
                    values = torch.arange(10, dtype=torch.int64, device='cuda:0').reshape(5, 2)
                    original_values = self._signature(values)
                    clone_offsets = source.clone()
                    clone_before = self._signature(clone_offsets)
                    nested = torch.nested.nested_tensor_from_jagged(values, source, jagged_dim=1)
                    same = torch.nested.nested_tensor_from_jagged(values, source, jagged_dim=1)
                    cloned = torch.nested.nested_tensor_from_jagged(values, clone_offsets, jagged_dim=1)
                    term = nested.shape[1]
                    self.assertTrue(type(term) is torch.SymInt and term is not nested.size(1) and bool(term == nested.size(1))
                        and bool(term == get_tensor_state(self.jt).nested_symint(source)), 'legacy public shape stays supported')
                    self.assertTrue(term is not same.shape[1] and term is not nested.shape[1]
                        and term is not nested.size()[1] and bool(term == same.shape[1])
                        and hash(term) == hash(same.shape[1]) and not bool(term == cloned.shape[1]), 'legacy same/clone public key identity')
                    for item in (nested, same, cloned):
                        packed = item.values()
                        self._resident(packed)
                        self.assertTrue(bool(torch.equal(packed, values)), 'full legacy packed CUDA values')
                        rows = item.unbind(0)
                        self.assertTrue(len(rows) == 2, 'complete legacy row census')
                        for row, expected in zip(rows, (values[:2], values[2:5])):
                            self._resident(row)
                            self.assertTrue(tuple(row.shape) == tuple(expected.shape)
                                and bool(torch.equal(row, expected)), 'complete legacy CUDA row shape/values')
                    self.assertTrue(original_values == self._signature(values) and clone_before == self._signature(clone_offsets),
                                    'legacy public entry preserves values/clone bytes/flags/storage')
                    state = get_tensor_object_state(source)
                    cache = state.nested_symbol_cache
                    for restored in (copy.copy(cache), copy.deepcopy(cache), pickle.loads(pickle.dumps(cache, protocol=4))):
                        self.assertTrue(restored is not cache and restored.get(second.nested_symbol_owner) is None,
                                        'weak owner keys never persist through cache copy')
                    for restored_state in (copy.deepcopy(state), pickle.loads(pickle.dumps(state, protocol=4))):
                        self.assertTrue(restored_state.nested_symbol_cache.get(second.nested_symbol_owner) is None,
                                        'full slot state serialization resets runtime identity only')
                self.assertTrue(before == self._signature(source), 'source remains unchanged')

    def _candidate_legacy_holder_lifetime(self):
        if self.jt is None:
            return
        from jittor.compat.torch.tensor_state import get_tensor_state
        state = get_tensor_state(self.jt)
        self.jt.sync_all(); gc.collect()
        before = self.jt.core.number_of_hold_vars()
        def make_term():
            source = self._legacy()
            return state.nested_symint(source)
        term = make_term()
        self.jt.sync_all(); gc.collect()
        self.assertTrue(self.jt.core.number_of_hold_vars() == before, 'legacy term must not retain a native holder')
        self.assertTrue(bool(term == term) and bool(term >= 2), 'legacy term survives source collection')

    def _candidate_native_pickle_values(self):
        if self.jt is None:
            return  # Native Var serialization is not a public PyTorch API.
        import numpy as np
        from jittor.serialization.native import _reduce_var
        self.assertTrue(self.jt.core.Var.__reduce__ is _reduce_var, 'one native reducer owner')
        fixtures = (
            ('int64', [-2**50 - 3, 2**50 + 17, 0, 1, -1, 2**40 + 1]),
            ('float64', [1 + 2**-40, -2**40 - 0.25, 0., -0., 3.5, -2.25]),
            ('float32', [1.5, -2.25, 0., -0., 3.5, 7.]),
            ('bfloat16', [1.5, -2.25, 0., -0., 3.5, 7.]),
        )
        for dtype, values in fixtures:
            for geometry in ('vector', 'scalar', 'empty', 'transposed'):
                with self.subTest(native_pickle_dtype=dtype, geometry=geometry):
                    host_dtype = 'float32' if dtype == 'bfloat16' else dtype
                    data = np.asarray(values, dtype=host_dtype)
                    if geometry == 'scalar':
                        data = np.asarray(values[0], dtype=host_dtype)
                    elif geometry == 'empty':
                        data = np.empty((0, 2), dtype=host_dtype)
                    elif geometry == 'transposed':
                        data = data.reshape(2, 3)
                    # Host literals are fixture construction, not CPU Tensor
                    # arithmetic. All casts/views and restored values are CUDA.
                    with self.jt.flag_scope(auto_convert_64_to_32=0):
                        source = self.jt.core.Var(data, dtype).stop_grad()
                    if geometry == 'transposed':
                        source = source.transpose(1, 0)
                        data = data.transpose(1, 0)
                    self._resident(source)
                    def metadata(value):
                        return (id(value), value._storage_address, value._storage_offset(),
                            value.requires_grad, value.is_backward_leaf, value.grad_fn_node_id,
                            value.location(), value.device_id, value.placement_backend)
                    before = metadata(source)
                    old_policy = self.jt.flags.auto_convert_64_to_32
                    with self.torch.device('cuda:0'):
                        restored = pickle.loads(pickle.dumps(source, protocol=4))
                        repeated = pickle.loads(pickle.dumps((source, source), protocol=4))
                    self.assertTrue(repeated[0] is repeated[1], 'ordinary pickle memo identity')
                    self.assertTrue(self.jt.flags.auto_convert_64_to_32 == old_policy, 'restore factory policy')
                    self.assertTrue(type(restored) is self.jt.core.Var and restored is not source,
                                    'fresh exact native holder')
                    self._resident(restored)
                    self._resident(repeated[0])
                    self._resident(source)
                    self.assertTrue(str(restored.dtype) == dtype and tuple(restored.shape) == tuple(source.shape),
                                    'complete native dtype and shape')
                    for value in (source, restored, repeated[0]):
                        host = value._copy_to_cpu().stop_grad()
                        host.sync()
                        self.assertTrue(host is not value and host.location() == 'cpu', 'explicit host snapshot')
                        self.assertTrue(host.numpy().tobytes(order='C') == data.tobytes(order='C'),
                                        'every public fixture byte, including wide values and empty tails')
                    self.assertTrue(metadata(source) == before, 'pickle preserves original storage/flags/residency')

    def test_cuda(self):
        if os.environ.get('JITTOR_TEST_DEVICES') != 'cuda':
            self.skipTest('CUDA-only test explicitly requires cuda')
        self._setup_cuda()
        self._scope()
        anchor = self.torch.tensor([2, 3], dtype=self.torch.int64, device='cuda:0')
        self._resident(anchor)
        self.assertTrue(bool(self.torch.equal(anchor * 2 + 1,
            self.torch.tensor([5, 7], dtype=self.torch.int64, device='cuda:0'))), 'real CUDA arithmetic anchor')
        self._public_key_identity_and_preservation()
        self._public_weak_key_gc_old_term_survives()
        self._public_symbol_and_Size_copy_contract()
        self._candidate_owner_and_legacy_cache_copy()
        self._candidate_legacy_holder_lifetime()
        self._candidate_native_pickle_values()


if __name__ == '__main__':
    unittest.main()

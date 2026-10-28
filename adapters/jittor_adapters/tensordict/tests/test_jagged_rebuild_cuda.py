"""Real CUDA TensorDict jagged rebuild contract; no PPO acceptance claim."""
from contextlib import ExitStack
import os
import sys
import unittest


class TestTensorDictJaggedRebuildCUDA(unittest.TestCase):
    def resident(self, tensor):
        self.assertTrue(tensor.device.type == 'cuda' and tensor.device.index == 0)
        if self.jt is not None:
            native = getattr(tensor, '_var', tensor)
            native.sync()
            self.assertEqual(self.jt.core.dispatch_context([native]), ('cuda', 0))
            self.assertTrue(native.location() == 'device' and native.device_id == 0)

    def same_values(self, actual, expected):
        self.resident(actual)
        self.resident(expected)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertTrue(bool(self.torch.equal(actual, expected)))

    def test_cuda(self):
        if os.environ.get('JITTOR_TEST_DEVICES') != 'cuda':
            self.skipTest('explicit CUDA-only contract')
        import torch
        import tensordict
        from tensordict import TensorDict
        from tensordict.utils import _rebuild_njt_from_njt
        self.torch, self.jt = torch, sys.modules.get('jittor')
        self.assertEqual(tensordict.__version__, '0.10.0')
        self.assertTrue(torch.cuda.is_available() and torch.cuda.current_device() == 0)
        if self.jt is None:
            self.assertTrue(hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context'))
        with ExitStack() as stack:
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=1, device_id=0, backend_fallback='error'))
                stack.enter_context(forbid_backend_fallbacks())
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)
            anchor = torch.tensor([2., 3.], device='cuda:0')
            self.same_values(anchor * 2 + 1, torch.tensor([5., 7.], device='cuda:0'))
            for axis in (1, 2):
                for holes in (False, True):
                    with self.subTest(axis=axis, holes=holes):
                        values = torch.arange(21, dtype=torch.float32, device='cuda:0').reshape(
                            (7, 3) if axis == 1 else (3, 7))
                        offsets = torch.tensor([0, 3, 7], dtype=torch.int64, device='cuda:0')
                        lengths = torch.tensor([2, 3], dtype=torch.int64, device='cuda:0') if holes else None
                        nested = torch.nested.nested_tensor_from_jagged(
                            values, offsets, lengths=lengths, jagged_dim=axis)
                        self.assertEqual([i for i, size in enumerate(nested.shape)
                                          if isinstance(size, torch.SymInt)], [axis])
                        data = TensorDict({'nested': nested, 'dense': anchor}, batch_size=[2], device='cuda:0')
                        packed = data.consolidate()
                        self.assertTrue(packed.is_consolidated() and packed.is_locked)
                        rebuilt = packed['nested']
                        self.assertEqual([i for i, size in enumerate(rebuilt.shape)
                                          if isinstance(size, torch.SymInt)], [axis])
                        self.same_values(rebuilt.values(), values)
                        self.same_values(rebuilt.offsets(), offsets)
                        if holes:
                            self.same_values(rebuilt.lengths(), lengths)
                        storage = packed._consolidated['storage']
                        self.resident(storage)
                        start, end = storage.data_ptr(), storage.data_ptr() + storage.numel() * storage.element_size()
                        for leaf in (rebuilt.values(), rebuilt.offsets(), packed['dense']):
                            self.resident(leaf)
                            self.assertTrue(start <= leaf.data_ptr() < end)
                        # Consolidation need not keep the input's symbolic key:
                        # its upstream helper may receive an intermediate template.
                        # Native 0.10 rejects chunking noncontiguous holey NJTs.
                        # Native also rejects this packed axis-2 chunk case.
                        # Cover only the actual supported contiguous axis-1
                        # chunk path; retain consolidation on all four fixtures.
                        if axis == 1 and not holes:
                            for chunk in data.chunk(2):
                                again = chunk.consolidate()
                                self.assertTrue(again.is_consolidated())
                                self.resident(again['nested'].values())
                                self.resident(again['nested'].offsets())
                        # Isolate private rebuild from consolidation/chunk's
                        # legitimate mutations of the original metadata registry.
                        template = torch.nested.nested_tensor_from_jagged(
                            values, offsets.clone(), lengths=lengths.clone() if holes else None,
                            jagged_dim=axis)
                        template_symbol = template.shape[axis]
                        leaf = values.detach().clone().requires_grad_(True)
                        new_offsets = offsets.clone()
                        new_lengths = lengths.clone() if holes else None
                        previous = torch.nested.nested_tensor_from_jagged(
                            values, new_offsets, lengths=new_lengths, jagged_dim=axis)
                        previous_symbol = previous.shape[axis]
                        self.assertFalse(bool(previous_symbol == template_symbol))
                        view = _rebuild_njt_from_njt(template, leaf, new_offsets, new_lengths)
                        self.assertEqual(view.values().data_ptr(), leaf.data_ptr())
                        self.assertIs(view.offsets(), new_offsets)
                        self.assertFalse(view.requires_grad or view.values().requires_grad)
                        self.assertEqual(bool(view.shape[axis] == template_symbol), holes)
                        self.assertTrue(bool(previous.shape[axis] == previous_symbol))
                        self.assertEqual(bool(view.shape[axis] == previous_symbol), not holes)
                        again = torch.nested.nested_tensor_from_jagged(
                            values, new_offsets, lengths=new_lengths, jagged_dim=axis)
                        self.assertEqual(bool(again.shape[axis] == template_symbol), holes)
                        self.assertEqual(bool(previous.shape[axis] == again.shape[axis]), not holes)
                        # The template may itself be an old object whose key
                        # was rebound. Native uses that key's current registry
                        # identity, not this object's construction-time shape.
                        chained_offsets = offsets.clone()
                        chained_lengths = lengths.clone() if holes else None
                        chained_prior = torch.nested.nested_tensor_from_jagged(
                            values, chained_offsets, lengths=chained_lengths, jagged_dim=axis)
                        chained_symbol = chained_prior.shape[axis]
                        chained = _rebuild_njt_from_njt(
                            previous, leaf, chained_offsets, chained_lengths)
                        self.assertEqual(bool(chained.shape[axis] == template_symbol), holes)
                        self.assertEqual(bool(chained.shape[axis] == chained_symbol), not holes)
                        self.assertTrue(bool(chained_prior.shape[axis] == chained_symbol))
                        self.same_values(chained.values(), values)
                        if holes:
                            # An explicit new lengths=None selects the offsets
                            # key and must not copy the holey template's symbol.
                            plain_offsets = offsets.clone()
                            plain_prior = torch.nested.nested_tensor_from_jagged(
                                values, plain_offsets, jagged_dim=axis)
                            plain_symbol = plain_prior.shape[axis]
                            plain = _rebuild_njt_from_njt(template, leaf, plain_offsets, None)
                            self.assertIsNone(plain.lengths())
                            self.assertTrue(bool(plain.shape[axis] == plain_symbol))
                            self.assertFalse(bool(plain.shape[axis] == template_symbol))
                            self.assertFalse(plain.requires_grad or plain.values().requires_grad)
                            self.same_values(plain.values(), values)
                        self.assertIsNone(leaf.grad)
                        leaf.square().sum().backward()
                        self.same_values(leaf.grad, 2 * values)
            torch.cuda.synchronize()
            if self.jt is not None:
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)

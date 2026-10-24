"""Public autograd.grad duplicate-target and unused-gradient regression.

32 fresh cases per device. Full arrays remain in RAM; failure messages contain
counts/statistics only. Independent native Torch supplies the identity/materialization
contract. This does not test DTensor, FSDP or full PPO acceptance.
"""
from contextlib import ExitStack, contextmanager
import os
import sys
import unittest

import numpy as np


DTYPES = ('float32', 'float64')
X = np.array([1., 2., 4., 8.], dtype=np.float64)
COTANGENT = np.array([1/8, 3/16, 5/32, 7/64], dtype=np.float64)
ATOL = RTOL = 1e-5


class TestAutogradDuplicateTargets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime = os.environ.get('JITTOR_AUTOGRAD_DUPLICATE_RUNTIME',
                                 'candidate' if os.environ.get('JITTOR_TORCH_SHIM') == '1' else 'oracle')
        cls.jt = None
        if runtime == 'candidate':
            import jittor as jt
            import torch
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            cls.jt = jt
        elif runtime == 'oracle':
            import torch
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(name == 'jittor' or name.startswith('jittor.') for name in sys.modules)
        else:
            raise ValueError('JITTOR_AUTOGRAD_DUPLICATE_RUNTIME must be oracle or candidate')
        cls.torch = torch

    @contextmanager
    def _device(self, device):
        with ExitStack() as stack:
            if device == 'cuda':
                if not self.torch.cuda.is_available():
                    self.skipTest('CUDA unavailable; no CUDA support claim')
                self.assertEqual(self.torch.cuda.current_device(), 0)
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(device == 'cuda'), backend_fallback='error'))
                stack.enter_context(forbid_backend_fallbacks())
                before = self.jt.core.backend_fallback_count()
                self.assertEqual(before, 0)
            yield 'cuda:0' if device == 'cuda' else 'cpu'
            if self.jt is not None:
                self.jt.sync_all(True)
                self.assertEqual(self.jt.core.backend_fallback_count(), before)

    def _placement(self, tensor, target):
        device = target.split(':')[0]
        self.assertEqual(tensor.device.type, device)
        if self.jt is None:
            if device == 'cuda':
                self.assertEqual(tensor.device.index, 0)
                self.torch.cuda.synchronize(tensor.device)
            return
        native = getattr(tensor, '_var', tensor)
        native.sync()
        index = -1 if device == 'cpu' else 0
        self.assertEqual(self.jt.core.dispatch_context([native]), (device, index))
        self.assertEqual(int(native.device_id), index)
        self.assertEqual(int(native.placement_backend), 0 if device == 'cpu' else 1)
        self.assertEqual(native.location(), 'cpu' if device == 'cpu' else 'device')

    def _metadata(self, tensor, target):
        # Allocation/synchronization is allowed; uninitialized bytes are not read.
        self._placement(tensor, target)
        return dict(shape=tuple(tensor.shape), dtype=tensor.dtype,
                    requires_grad=bool(tensor.requires_grad), is_leaf=bool(tensor.is_leaf))

    def _snapshot_initialized(self, tensor, target):
        before = self._metadata(tensor, target)
        with self.torch.no_grad():
            clone = tensor.clone()
            self.assertIsNot(clone, tensor)
            self.assertFalse(clone.requires_grad)
            self._placement(clone, target)
            host = clone.cpu()
            self._placement(host, 'cpu')
            byte_view = host.contiguous().view(self.torch.uint8)
            self._placement(byte_view, 'cpu')
            raw = np.array(byte_view.numpy(), dtype=np.uint8, copy=True, order='C')
            widened = host.to(dtype=self.torch.float64)
            self._placement(widened, 'cpu')
            numbers = np.array(widened.numpy(), dtype=np.float64, copy=True, order='C')
        self.assertEqual(self._metadata(tensor, target), before)
        self.assertEqual(numbers.shape, before['shape'])
        self.assertTrue(np.isfinite(numbers).all(), 'nonfinite initialized contents')
        self.assertEqual(raw.size, numbers.size * tensor.element_size())
        return before, raw, numbers

    def _assert_numeric(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all(), 'nonfinite tensor')
        bad = ~np.isclose(actual, expected, atol=ATOL, rtol=RTOL)
        difference = np.abs(actual.astype(np.float64)-expected)
        self.assertFalse(bool(bad.any()),
                         'full-array comparison: failed_elements=%d max_abs=%g' %
                         (int(bad.sum()), float(difference.max(initial=0))))


    def _case(self, target, dtype_name, unused_present, allow, materialize, create):
        torch = self.torch
        dtype = getattr(torch, dtype_name)
        x = torch.tensor(X, dtype=dtype, device=target, requires_grad=True)
        unused = torch.tensor(X + 1, dtype=dtype, device=target, requires_grad=True)
        cot = torch.tensor(COTANGENT, dtype=dtype, device=target, requires_grad=False)
        inputs = (x, unused, cot)
        before = [self._snapshot_initialized(t, target) for t in inputs]
        targets = (x, unused, x) if unused_present else (x, x)
        loss = (cot * x * x * x).sum()
        self._placement(loss, target)
        options = dict(allow_unused=allow, materialize_grads=materialize, create_graph=create)
        error = ValueError if materialize and not allow else (
            RuntimeError if unused_present and not allow else None)
        gradients = None
        if error is not None:
            message = 'allow_unused' if error is ValueError else 'not have been used in the graph'
            with self.assertRaisesRegex(error, message):
                torch.autograd.grad(loss, targets, **options)
        else:
            gradients = torch.autograd.grad(loss, targets, **options)
            self.assertEqual(len(gradients), len(targets))
            with self.subTest(contract='duplicate_used_identity'):
                self.assertIsNotNone(gradients[0])
                self.assertIs(gradients[0], gradients[-1])
            for slot, gradient in enumerate(gradients):
                is_unused = unused_present and slot == 1
                with self.subTest(contract='returned_slot', slot=slot):
                    if is_unused and not materialize:
                        self.assertIsNone(gradient)
                        continue
                    self.assertIsInstance(gradient, torch.Tensor)
                    metadata, _, actual = self._snapshot_initialized(gradient, target)
                    self.assertEqual(metadata['shape'], (4,))
                    self.assertEqual(metadata['dtype'], dtype)
                    self.assertEqual(metadata['requires_grad'], True if is_unused else create)
                    self.assertEqual(metadata['is_leaf'], True if is_unused else not create)
                    self.assertEqual(gradient.grad_fn is not None, False if is_unused else create)
                    reference = np.zeros(4) if is_unused else 3 * COTANGENT * X * X
                    self._assert_numeric(actual, reference)
                    if create and not is_unused:
                        second, = torch.autograd.grad(gradient.sum(), (x,), retain_graph=True)
                        self.assertIsInstance(second, torch.Tensor)
                        _, _, actual_second = self._snapshot_initialized(second, target)
                        self._assert_numeric(actual_second, 6 * COTANGENT * X)
        after = [self._snapshot_initialized(t, target) for t in inputs]
        for name, initial, final in zip(('x', 'unused', 'cotangent'), before, after):
            with self.subTest(contract='source_unchanged', name=name):
                self.assertEqual(initial[0], final[0])
                self.assertTrue(np.array_equal(initial[1], final[1]), 'source complete bytes changed')
        self.assertIsNone(x.grad)
        self.assertIsNone(unused.grad)

    def _run_device(self, device):
        from itertools import product
        with self._device(device) as target:
            for dtype, unused, allow, materialize, create in product(
                    DTYPES, (False, True), (False, True), (False, True), (False, True)):
                with self.subTest(dtype=dtype, unused=unused, allow_unused=allow,
                                  materialize=materialize, create_graph=create):
                    self._case(target, dtype, unused, allow, materialize, create)

    def test_cpu(self):
        self._run_device('cpu')

    def test_cuda(self):
        self._run_device('cuda')


if __name__ == '__main__':
    unittest.main()

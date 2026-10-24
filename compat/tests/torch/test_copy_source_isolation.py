"""copy_ preserves source graphs and existing detached storage aliases.

Nine fixed FP32 cases, unchanged from the independent CPU/CUDA regression.
Select one device per process with JITTOR_TEST_DEVICES=cpu (default) or cuda;
run test_cpu / test_cuda in separate interpreters and compile caches. This is
public copy_ coverage, not full FSDP, model training or PPO acceptance.
"""
from contextlib import ExitStack, contextmanager
import os
import sys
import unittest
import numpy as np
import torch


class TestCopySourceIsolation(unittest.TestCase):
    @contextmanager
    def _device(self, device):
        selected = os.environ.get('JITTOR_TEST_DEVICES', 'cpu').strip()
        if selected not in ('cpu', 'cuda', 'npu', 'rocm'):
            raise ValueError('select one test device per interpreter: cpu/cuda/npu/rocm')
        if device != selected:
            self.skipTest('copy_ CPU/CUDA node not selected by JITTOR_TEST_DEVICES')
        jt = sys.modules.get('jittor')
        if device == 'cuda':
            if jt is not None:
                from _helpers.capability import check_accelerator
                capability = check_accelerator('cuda', backend=jt)
                if not capability.enabled:
                    self.skipTest('accelerator prerequisite: CUDA %s: %s' %
                                  (capability.state.value, capability.reason))
                self.assertTrue(torch.cuda.is_available(), 'enabled CUDA capability must execute')
            elif not torch.cuda.is_available():
                self.skipTest('CUDA unavailable; no CUDA execution claim')
        with ExitStack() as stack:
            if jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(jt.runtime.scope(use_cuda=int(device == 'cuda'), device_id=0 if device == 'cuda' else -1,
                                                    backend_fallback='error', cuda_kernel_math='strict'))
                stack.enter_context(forbid_backend_fallbacks())
                before = jt.core.backend_fallback_count()
                self.assertIs(type(before), int); self.assertEqual(before, 0)
            yield 'cuda:0' if device == 'cuda' else 'cpu'
            if jt is not None:
                after = jt.core.backend_fallback_count()
                self.assertIs(type(after), int); self.assertEqual(after, before)

    def _prove(self, value, target):
        kind = target.split(':')[0]
        self.assertEqual(value.device.type, kind)
        if kind == 'cuda': self.assertEqual(value.device.index, 0)
        jt = sys.modules.get('jittor')
        if jt is not None:
            native = getattr(value, '_var', value); native.sync()
            self.assertEqual(jt.core.dispatch_context([native]), (kind, -1 if kind == 'cpu' else 0))
            self.assertEqual(native.placement_backend, 0 if kind == 'cpu' else 1)
            self.assertEqual(native.location(), 'cpu' if kind == 'cpu' else 'device')
            count = jt.core.backend_fallback_count()
            self.assertIs(type(count), int); self.assertEqual(count, 0)
        elif kind == 'cuda': torch.cuda.synchronize(value.device)

    @staticmethod
    def _flags(value):
        return (tuple(value.shape), tuple(value.stride()), value.dtype, str(value.device),
                value.requires_grad, value.is_leaf, value.grad_fn is None)

    def _array(self, value, target):
        before = self._flags(value); self._prove(value, target)
        with torch.no_grad():
            owned = value.clone(); self.assertTrue(owned is not value, "snapshot_independent_object")
            self.assertFalse(owned.requires_grad); self._prove(owned, target)
            host = owned.cpu(); self._prove(host, 'cpu')
            result = np.array(host.numpy(), copy=True)
        self.assertEqual(self._flags(value), before); self._prove(value, target)
        self.assertTrue(np.isfinite(result).all())
        return result

    def _same_bytes(self, value, reference, target):
        array = self._array(value, target)
        self.assertEqual(array.dtype, reference.dtype); self.assertEqual(array.shape, reference.shape)
        self.assertTrue(array.tobytes() == reference.tobytes(), "complete_bytes_equal")

    def _allclose_arrays(self, actual, reference):
        self.assertTrue(type(actual) is np.ndarray and type(reference) is np.ndarray, "complete_array_types")
        self.assertTrue(actual.dtype == reference.dtype, "complete_dtype_equal")
        self.assertTrue(actual.shape == reference.shape, "complete_shape_equal")
        self.assertTrue(bool(np.isfinite(actual).all() and np.isfinite(reference).all()), "complete_arrays_finite")
        actual64 = actual.astype(np.float64); reference64 = reference.astype(np.float64)
        self.assertTrue(bool(np.all(np.abs(actual64 - reference64) <= 1e-5 + 1e-5 * np.abs(reference64))),
                        "complete_fixed_1e5")

    def _vjp(self, value, source, reference, target):
        cot = torch.tensor(self.cot, device=target)
        gradient, = torch.autograd.grad(value, (source,), grad_outputs=cot, create_graph=False,
                                       retain_graph=False, allow_unused=False)
        actual = self._array(gradient, target)
        self._allclose_arrays(actual, reference)
        self.assertTrue(source.grad is None, "source_grad_none")  # autograd.grad must not publish .grad

    def _case(self, target, style):
        p = torch.nn.Parameter(torch.tensor(self.fixture, device=target))
        if style in ('canonical_alias', 'view_alias', 'self'):
            cot = torch.tensor(self.cot, device=target)
            (p * cot).sum().backward(); old_grad = p.grad
            self.assertTrue(old_grad is not None, "original_grad_present")
            with torch.no_grad(): stored = p.detach()
            rhs = p.view(2, 3) if style == 'view_alias' else p
            destination = p if style == 'self' else stored
            flags = [self._flags(v) for v in (p, stored, rhs, destination)]
            self.assertEqual(p.data_ptr(), stored.data_ptr()); self.assertEqual(rhs.data_ptr(), p.data_ptr())
            with torch.no_grad(): returned = destination.copy_(rhs)
            self.assertTrue(returned is destination, "copy_returns_destination"); self.assertTrue(p.grad is old_grad, "original_grad_identity")
            self.assertEqual([self._flags(v) for v in (p, stored, rhs, destination)], flags)
            self.assertEqual(p.data_ptr(), stored.data_ptr()); self.assertEqual(rhs.data_ptr(), p.data_ptr())
            for value in (p, stored, rhs, destination): self._same_bytes(value, self.fixture, target)
            self._same_bytes(old_grad, self.cot, target)
            normal = p.view(2, 3); self.assertTrue(normal.requires_grad); self.assertFalse(normal.is_leaf)
            gradient, = torch.autograd.grad(normal, (p,), grad_outputs=cot, create_graph=False,
                                           retain_graph=False, allow_unused=False)
            self._allclose_arrays(self._array(gradient, target), self.cot)
            self.assertTrue(p.grad is old_grad, "original_grad_identity"); self._same_bytes(old_grad, self.cot, target)
        elif style == 'alias_changed':
            cot = torch.tensor(self.cot, device=target)
            (p * cot).sum().backward(); old_grad = p.grad
            with torch.no_grad(): stored = p.detach()
            expected = self.fixture + np.float32(.125)
            rhs = torch.nn.Parameter(torch.tensor(expected, device=target))
            flags = [self._flags(v) for v in (p, stored, rhs)]
            pointer = p.data_ptr()
            with torch.no_grad(): returned = stored.copy_(rhs)
            self.assertTrue(returned is stored, "copy_returns_stored"); self.assertTrue(p.grad is old_grad, "original_grad_identity")
            self.assertEqual([self._flags(v) for v in (p, stored, rhs)], flags)
            self.assertEqual(p.data_ptr(), pointer); self.assertEqual(stored.data_ptr(), pointer)
            self.assertNotEqual(rhs.data_ptr(), pointer)
            for value in (p, stored, rhs): self._same_bytes(value, expected, target)
            self._same_bytes(old_grad, self.cot, target); self.assertTrue(rhs.grad is None, "rhs_grad_none")
            normal = p.view(2, 3)
            gradient, = torch.autograd.grad(normal, (p,), grad_outputs=cot, create_graph=False,
                                           retain_graph=False, allow_unused=False)
            self._allclose_arrays(self._array(gradient, target), self.cot)
            self.assertTrue(p.grad is old_grad, "original_grad_identity"); self._same_bytes(old_grad, self.cot, target)
        elif style == 'strided_source':
            source = p.transpose(0, 1); flags = self._flags(source)
            destination = torch.zeros((3, 2), device=target)
            with torch.no_grad(): returned = destination.copy_(source)
            self.assertTrue(returned is destination, "copy_returns_destination"); self.assertFalse(destination.requires_grad)
            self.assertEqual(self._flags(source), flags); self.assertTrue(p.requires_grad)
            self.assertNotEqual(destination.data_ptr(), source.data_ptr())
            self._same_bytes(destination, self.fixture.T.copy(), target)
            self._same_bytes(source, self.fixture.T.copy(), target); self._same_bytes(p, self.fixture, target)
        elif style in ('distinct_stopped', 'distinct_parameter'):
            destination = (torch.zeros((2, 3), device=target) if style == 'distinct_stopped' else
                           torch.nn.Parameter(torch.zeros((2, 3), device=target)))
            source_flags, target_flags = self._flags(p), self._flags(destination)
            with torch.no_grad(): returned = destination.copy_(p)
            self.assertTrue(returned is destination, "copy_returns_destination")
            self.assertEqual(self._flags(p), source_flags); self.assertEqual(self._flags(destination), target_flags)
            self.assertNotEqual(destination.data_ptr(), p.data_ptr())
            self._same_bytes(p, self.fixture, target); self._same_bytes(destination, self.fixture, target)
            self.assertTrue(p.grad is None, "parameter_grad_none"); self.assertTrue(destination.grad is None, "destination_grad_none")
        elif style in ('grad_leaf', 'grad_nonleaf'):
            source = p if style == 'grad_leaf' else p.square()
            source_flags = self._flags(source); p_flags = self._flags(p)
            destination = torch.zeros((2, 3), device=target)
            returned = destination.copy_(source)
            self.assertTrue(returned is destination, "copy_returns_destination"); self.assertTrue(destination.requires_grad)
            self.assertFalse(destination.is_leaf); self.assertTrue(destination.grad_fn is not None, "destination_grad_fn_present")
            self.assertNotEqual(destination.data_ptr(), source.data_ptr())
            self.assertEqual(self._flags(source), source_flags); self.assertEqual(self._flags(p), p_flags)
            reference = self.fixture if style == 'grad_leaf' else self.fixture ** 2
            self._same_bytes(source, reference, target); self._same_bytes(destination, reference, target)
            self._same_bytes(p, self.fixture, target)
            expected = self.cot if style == 'grad_leaf' else 2 * self.fixture * self.cot
            self._vjp(destination, p, expected, target)
            self.assertEqual(self._flags(source), source_flags); self.assertEqual(self._flags(p), p_flags)
        else: raise ValueError('unrecognized case')

    def _check(self, device):
        self.fixture = np.array([[.25, -.5, .75], [-.125, .375, .5]], np.float32)
        self.cot = np.array([[.75, -1.125, 2.], [-.5, .25, 1.5]], np.float32)
        with self._device(device) as target:
            for style in ('canonical_alias', 'view_alias', 'self', 'alias_changed', 'strided_source',
                          'distinct_stopped', 'distinct_parameter', 'grad_leaf', 'grad_nonleaf'):
                with self.subTest(device=device, style=style): self._case(target, style)

    def test_cpu(self): self._check('cpu')
    def test_cuda(self): self._check('cuda')


if __name__ == '__main__': unittest.main()

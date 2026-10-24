"""Public rsqrt routing, numerical boundaries and higher-order gradients.

One CPU/CUDA device per process. Native oracle and candidate are independent
processes; NumPy formulas here are analytic checks, NOT an actual Torch/Jittor
RAM pair. A future independent official-CUDA full-array pair remains required.
The default/strict kernel-math selection applies to Jittor only; the independent
oracle always uses its unchanged public implementation.
CUDA default's subnormal FTZ is checked as the documented native contract,
not claimed as Torch parity; CPU, strict and all other boundaries stay precise.
No registration/override occurs here. CUDA F32 must use its native dispatch.
No full-PPO, precision-cause, performance, ACL/ROCm or all-dtype claim.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import os
import sys
import unittest

import numpy as np


def finite_reference(x, dy, direction):
    wide, cot, vec = (a.astype(np.float64) for a in (x, dy, direction))
    return (wide ** -.5, -.5 * cot * wide ** -1.5,
            .75 * vec * cot * wide ** -2.5, -.5 * vec * wide ** -1.5)


def mismatch_count(actual, expected):
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError('complete shape differs')
    failed = (~np.isfinite(actual) | ~np.isfinite(expected)
        | (np.abs(actual.astype(np.float64) - expected) > 1e-5 + 1e-5 * np.abs(expected)))
    return int(np.count_nonzero(failed))


class TestRsqrtPublicPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch, cls.jt = torch, None
        runtime = os.environ.get('JITTOR_RSQRT_RUNTIME',
            'candidate' if os.environ.get('JITTOR_TORCH_SHIM') == '1' else 'oracle')
        if runtime not in ('oracle', 'candidate'):
            raise ValueError('JITTOR_RSQRT_RUNTIME=oracle|candidate required')
        if runtime == 'oracle':
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules)
        else:
            import jittor as jt
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active and torch.Tensor is not jt.Var
            cls.jt = jt

    @contextmanager
    def _scope(self, device, mode):
        selected = os.environ.get('JITTOR_TEST_DEVICES', 'cpu')
        if selected not in ('cpu', 'cuda'):
            raise ValueError('exact single JITTOR_TEST_DEVICES=cpu|cuda required')
        if device != selected:
            self.skipTest('node excluded by explicit single-device process selection')
        with ExitStack() as stack:
            if device == 'cuda':
                self.assertTrue(self.torch.cuda.is_available(), 'requested actual CUDA unavailable')
                self.torch.cuda.set_device(0)
                self.assertEqual(self.torch.cuda.current_device(), 0)
                if self.jt is not None:
                    self.assertTrue(self.jt.compiler.has_cuda and self.jt.compiler.is_cuda)
                    self.assertFalse(any(getattr(self.jt.compiler, n, False) for n in ('has_acl', 'has_rocm', 'has_corex')))
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                stack.enter_context(self.jt.runtime.scope(use_cuda=int(device == 'cuda'),
                    device_id=0 if device == 'cuda' else -1, amp_reg=0,
                    backend_fallback='error', cuda_kernel_math=mode))
                stack.enter_context(forbid_backend_fallbacks())
                self.assertEqual(self.jt.flags.cuda_kernel_math, mode)
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)
            yield
            if self.jt is not None:
                self.assertEqual(self.jt.core.backend_fallback_count(), 0)

    def _resident(self, value, device):
        self.assertTrue(isinstance(value, self.torch.Tensor), 'public Tensor result required')
        self.assertEqual(value.device.type, device)
        self.assertEqual(value.device.index, 0 if device == 'cuda' else None)
        if self.jt is None:
            if device == 'cuda': self.torch.cuda.synchronize(value.device)
            return
        native = getattr(value, '_var', value)
        native.sync()
        ordinal = 0 if device == 'cuda' else -1
        self.assertEqual(self.jt.core.dispatch_context([native]), (device, ordinal))
        self.assertEqual(int(native.device_id), ordinal)
        self.assertEqual(int(native.placement_backend), int(device == 'cuda'))
        self.assertEqual(native.location(), 'device' if device == 'cuda' else 'cpu')

    def _snapshot(self, value, device):
        self._resident(value, device)  # Before any host read; original never numpy().
        flags = (value.requires_grad, value.is_leaf, tuple(value.shape), value.dtype)
        with self.torch.no_grad():
            clone = value.clone()
            self.assertTrue(clone is not value and clone.requires_grad is False, 'independent stopped snapshot required')
            self._resident(clone, device)
            host = clone.cpu()
            self._resident(host, 'cpu')
            array = np.array(host.numpy(), copy=True)
        self._resident(value, device)
        self.assertEqual(flags, (value.requires_grad, value.is_leaf, tuple(value.shape), value.dtype))
        return array

    def _route(self, x, device):
        if self.jt is None: return
        from jittor._runtime.dispatch import select_kernel
        from jittor.compat.torch.context import get_install_context
        from jittor.compat.torch.native_api import operation
        from jittor.ops import numerical
        self.assertTrue(self.jt.rsqrt is numerical.rsqrt and self.jt.Var.rsqrt is numerical.rsqrt,
                        'root and native method must share the numerical owner')
        self.assertTrue(self.torch.rsqrt is operation('rsqrt')
            and get_install_context(self.jt).state['native_torch_operations']['rsqrt'] is self.jt.rsqrt,
            'public Torch delegate must reach the actual shared native owner')
        kernel = select_kernel('tensor.rsqrt', x)
        if device == 'cuda' and x.dtype == self.torch.float32:
            self.assertTrue(kernel is not None, 'missing real CUDA F32 rsqrt dispatch')
            self.assertEqual(kernel.__module__, 'jittor.backends.cuda.kernels.misc.rsqrt_cuda')
            self.assertEqual(kernel.__name__, '_rsqrt_cuda')
        else:
            self.assertTrue(kernel is None, 'CPU/non-F32 must retain the old decomposition route')

    def _entry(self, x, spelling):
        if spelling == 'function': return self.torch.rsqrt(x)
        if spelling == 'method': return x.rsqrt()
        if spelling == 'native_root': return self.jt.rsqrt(x)
        if spelling == 'native_method': return self.jt.Var.rsqrt(x)
        raise ValueError('unknown declared spelling')

    def _finite(self, device, spelling, layout):
        torch = self.torch
        raw = np.array([[.25, .5, 1.], [2., 4., 16.]], dtype='float32')
        cot = np.array([[1., -.5, .75], [-1.25, 2., -.125]], dtype='float32')
        vec = np.array([[.5, -.75, 1.25], [-.25, .125, -1.]], dtype='float32')
        base = torch.tensor(raw.copy(), dtype=torch.float32, device=device, requires_grad=True)
        x = base if layout == 'dense' else base.transpose(0, 1)
        if layout != 'dense': raw, cot, vec = raw.T.copy(), cot.T.copy(), vec.T.copy()
        dy = torch.tensor(cot.copy(), dtype=torch.float32, device=device, requires_grad=True)
        vector = torch.tensor(vec.copy(), dtype=torch.float32, device=device, requires_grad=False)
        self._route(x, device)
        y = self._entry(x, spelling)
        dx, = torch.autograd.grad(y, (x,), grad_outputs=dy, create_graph=True)
        self.assertTrue(dx.requires_grad, 'first VJP must retain the actual higher-order graph')
        gx, gdy = torch.autograd.grad(dx, (x, dy), grad_outputs=vector, create_graph=False)
        reference = finite_reference(raw, cot, vec)
        for actual, expected in zip((y, dx, gx, gdy), reference):
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(mismatch_count(self._snapshot(actual, device), expected), 0,
                             'complete finite Y/VJP/second-derivative mismatch; arrays RAM-only')
        self.assertTrue(self._snapshot(x, device).tobytes() == raw.tobytes(), 'input full bytes changed')
        self.assertTrue(self._snapshot(dy, device).tobytes() == cot.tobytes(), 'cotangent full bytes changed')
        self.assertTrue(base.requires_grad and base.grad is None and dy.requires_grad and dy.grad is None,
                        'autograd.grad must not change leaf flags/accumulate gradients')

    def _flags_and_empty(self, device, spelling):
        torch = self.torch
        raw = np.array([.25, 1., 4.], dtype='float32')
        for requested in (False, True):
            for no_grad in (False, True):
                x = torch.tensor(raw.copy(), dtype=torch.float32, device=device, requires_grad=requested)
                self._route(x, device)
                with torch.no_grad() if no_grad else nullcontext():
                    y = self._entry(x, spelling)
                self.assertIs(y.requires_grad, requested and not no_grad)
                self._snapshot(y, device)
                self.assertTrue(x.requires_grad is requested and x.is_leaf and x.grad is None,
                                'normal/no_grad must preserve original leaf policy')
                self.assertTrue(self._snapshot(x, device).tobytes() == raw.tobytes(), 'flags test changed input bytes')
        for shape in ((), (0, 3)):
            raw = np.full(shape, 4., dtype='float32')
            x = torch.tensor(raw.copy(), dtype=torch.float32, device=device, requires_grad=True)
            dy = torch.tensor(np.full(shape, .5, dtype='float32'), dtype=torch.float32, device=device)
            y = self._entry(x, spelling)
            dx, = torch.autograd.grad(y, (x,), grad_outputs=dy)
            for value, expected in ((y, np.full(shape, .5)), (dx, np.full(shape, -.03125))):
                self.assertEqual(tuple(value.shape), shape)
                self.assertEqual(mismatch_count(self._snapshot(value, device), expected), 0,
                                 'scalar/empty complete forward or VJP mismatch')

    def _classified(self, device, spelling, mode):
        torch = self.torch
        raw = np.array([0., -0., -1., np.inf, -np.inf, np.nan, .25, 4.], dtype='float32')
        extreme = np.array([0x00800000, 0x007fffff, 0x00000001], dtype='<u4').view('<f4')
        raw = np.concatenate((raw, extreme))
        x = torch.tensor(raw.copy(), dtype=torch.float32, device=device, requires_grad=True)
        dy = torch.tensor(np.ones(raw.shape, dtype='float32'), dtype=torch.float32, device=device)
        y = self._entry(x, spelling)
        dx, = torch.autograd.grad(y, (x,), grad_outputs=dy)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            ref_y = (1. / np.sqrt(raw.astype(np.float64))).astype(np.float32)
            wide_y = ref_y.astype(np.float64)
            ref_dx = (-.5 * ((wide_y * wide_y) * wide_y)).astype(np.float32)
        if self.jt is not None and device == 'cuda' and mode == 'default':
            # KI-BACKEND-008: preserve the deliberate default FTZ policy.
            # Do not change original inputs, normal/special values, the oracle,
            # strict mode, any finite tolerance, or the residency assertions.
            subnormal = (raw > 0) & (raw < np.finfo(np.float32).tiny)
            self.assertEqual(int(np.count_nonzero(subnormal)), 2)
            ref_y[subnormal] = np.inf
            ref_dx[subnormal] = -np.inf
        for value, reference in ((y, ref_y), (dx, ref_dx)):
            actual = self._snapshot(value, device)
            for classifier in (np.isnan, np.isposinf, np.isneginf, np.isfinite):
                self.assertTrue(bool(np.array_equal(classifier(actual), classifier(reference))),
                                'full nonfinite classification differs; raw arrays omitted')
            finite = np.isfinite(reference)
            self.assertEqual(mismatch_count(actual[finite], reference[finite]), 0, 'finite subset differs')
            zeros = finite & (reference == 0)
            self.assertTrue(bool(np.array_equal(np.signbit(actual[zeros]), np.signbit(reference[zeros]))),
                            'signed zero category differs')
        self.assertTrue(self._snapshot(x, device).tobytes() == raw.tobytes(), 'classification changed input bytes')

    def _unchanged_routes(self, device):
        torch = self.torch
        for name in ('float64', 'float16', 'bfloat16'):
            x = torch.tensor([.25, 1., 4.], dtype=getattr(torch, name), device=device, requires_grad=False)
            self._resident(x, device)
            self._route(x, device)  # Low-precision eligibility, NOT a low-precision math claim.
        x = torch.tensor([.25, 1., 4., 16.], dtype=torch.float64, device=device, requires_grad=True)
        y = torch.rsqrt(x)
        dx, = torch.autograd.grad(y, (x,), grad_outputs=torch.tensor([1., -.5, 2., -2.], dtype=torch.float64, device=device))
        self.assertEqual(mismatch_count(self._snapshot(y, device), np.array([2., 1., .5, .25])), 0)
        self.assertEqual(mismatch_count(self._snapshot(dx, device), np.array([-4., .25, -.125, .015625])), 0)

    def _run(self, device):
        spellings = ('function', 'method') + (() if self.jt is None else ('native_root', 'native_method'))
        for mode in (('default', 'strict') if device == 'cuda' else ('default',)):
            with self._scope(device, mode):
                self._unchanged_routes(device)
                for spelling in spellings:
                    with self.subTest(device=device, mode=mode, spelling=spelling):
                        self._finite(device, spelling, 'dense')
                        self._finite(device, spelling, 'transpose')
                        self._flags_and_empty(device, spelling)
                        self._classified(device, spelling, mode)

    def test_cpu(self): self._run('cpu')

    def test_cuda(self): self._run('cuda')


if __name__ == '__main__': unittest.main()

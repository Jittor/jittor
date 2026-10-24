"""Scalar constants must not make stopped floating inputs trainable.

Run CPU before CUDA, in independent native Torch / Jittor interpreters. Set
JITTOR_SCALAR_DIVISION_RUNTIME=oracle|candidate, or use the existing shim flag.
Selectors: TestScalarDivisionAutogradFlags.test_cpu / test_cuda and
test_rope_flags_cpu / test_rope_flags_cuda. No dtype tolerance relaxation:
binary-exact inputs/cotangents/scalars permit the same 1e-5 full-array check.

Returned leaf.grad.requires_grad is OBSERVED as a separate existing gap, not
an assertion of this fresh-scalar fix. This test does not prove model parity.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import json
import os
import sys
import unittest

import numpy as np


class FloatSubclass(float):
    """Public scalar subtype bypasses candidate PyFloat_CheckExact fast path."""


DTYPES = ('float16', 'bfloat16', 'float32', 'float64')
STATES = ('stopped', 'trainable', 'no_grad')
X = np.array([1., 2., 4., 8.], dtype=np.float64)
COTANGENT = np.array([1/8, 3/16, 5/32, 7/64], dtype=np.float64)
SCALARS = (('int', 2), ('float', 2.0), ('float_subclass', FloatSubclass(2.0)))
ATOL = RTOL = 1e-5


def analytic(reverse):
    return (2.0 / X, -2.0 / (X * X) * COTANGENT) if reverse else (X / 2.0, COTANGENT / 2.0)


def operation(torch, jt, value, scalar, reverse, path):
    if path == 'direct_slow' and jt is not None:
        # Direct canonical helper unit coverage: no patch or alternate math.
        from jittor.compat.torch.installers.tensor.method_api import _true_division
        return _true_division(value, scalar, '__rtruediv__' if reverse else '__truediv__')
    if path not in ('public', 'direct_slow'):
        raise ValueError('unknown division path')
    return scalar / value if reverse else value / scalar


class TestScalarDivisionAutogradFlags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runtime = os.environ.get('JITTOR_SCALAR_DIVISION_RUNTIME',
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
            raise ValueError('JITTOR_SCALAR_DIVISION_RUNTIME must be candidate or oracle')
        cls.torch = torch

    @contextmanager
    def _device(self, device):
        with ExitStack() as stack:
            if device == 'cuda':
                if not self.torch.cuda.is_available():
                    self.skipTest('CUDA unavailable; no CUDA claim')
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

    def _array(self, tensor, target):
        self._placement(tensor, target)
        before = (tuple(tensor.shape), tensor.dtype, bool(tensor.requires_grad))
        with self.torch.no_grad():
            clone = tensor.clone()
            self.assertIsNot(clone, tensor)
            self._placement(clone, target)
            # Native NumPy has no bfloat16; exact representable FP32 conversion
            # applies only to the independent observation clone on its device.
            if tensor.dtype == self.torch.bfloat16:
                clone = clone.float()
                self._placement(clone, target)
            host = clone.cpu()
            self._placement(host, 'cpu')
            array = np.array(host.numpy(), copy=True)
        self._placement(tensor, target)
        self.assertEqual((tuple(tensor.shape), tensor.dtype, bool(tensor.requires_grad)), before)
        self.assertEqual(array.shape, before[0])
        self.assertTrue(np.isfinite(array).all())
        return array

    def _case(self, target, dtype_name, state, reverse, scalar, path):
        torch = self.torch
        dtype = getattr(torch, dtype_name)
        trainable = state != 'stopped'
        x = torch.tensor(X, dtype=dtype, device=target, requires_grad=trainable)
        cot = torch.tensor(COTANGENT, dtype=dtype, device=target, requires_grad=False)
        with torch.no_grad() if state == 'no_grad' else nullcontext():
            output = operation(torch, self.jt, x, scalar, reverse, path)
        self.assertEqual(output.dtype, dtype)
        self.assertEqual(tuple(output.shape), X.shape)
        self.assertEqual(bool(output.requires_grad), state == 'trainable')
        self.assertEqual(bool(x.requires_grad), trainable)
        self.assertFalse(cot.requires_grad)
        grad = None
        if state == 'trainable':
            loss = (output * cot).sum()
            loss.backward()
            grad = x.grad
            self.assertIsNotNone(grad, 'all four input gradients required')
            self.assertEqual(grad.dtype, dtype)
            self.assertEqual(tuple(grad.shape), X.shape)
            self._placement(loss, target)
        self._placement(x, target)
        self._placement(cot, target)
        forward, gradient = analytic(reverse)
        np.testing.assert_allclose(self._array(output, target), forward, atol=ATOL, rtol=RTOL)
        if grad is not None:
            np.testing.assert_allclose(self._array(grad, target), gradient, atol=ATOL, rtol=RTOL)
            # Explicitly separate the known returned-gradient graph-state gap.
            return bool(grad.requires_grad)
        self.assertIsNone(x.grad)
        return None

    def _check(self, device):
        observations = []
        with self._device(device) as target:
            for dtype in DTYPES:
                for state in STATES:
                    for reverse in (False, True):
                        for name, scalar in SCALARS + (('direct_slow_float', 2.0),):
                            path = 'direct_slow' if name == 'direct_slow_float' else 'public'
                            label = f'{dtype}/{state}/{reverse}/{name}'
                            with self.subTest(case=label, device=device):
                                requires_grad = self._case(target, dtype, state, reverse, scalar, path)
                                if requires_grad is not None:
                                    observations.append(dict(case=label, requires_grad=requires_grad,
                                                             expected_requires_grad=False))
        print('SCALAR_DIVISION_RETURNED_GRAD_FLAGS ' + json.dumps(dict(
            device=device, scope='known_separate_gap_not_asserted_by_fresh_scalar_fix',
            all_input_gradient_elements_checked=len(observations) == 32,
            successful_trainable_cases=len(observations), expected_trainable_cases=32,
            observed=observations,
            gap_count=sum(item['requires_grad'] for item in observations), L2_pass=False), sort_keys=True))

    def _rope(self, device):
        torch = self.torch
        with self._device(device) as target:
            for profile, dim, base in (('text', 64, 10000000), ('vision', 32, 10000.0)):
                for no_grad in (False, True):
                    with self.subTest(profile=profile, no_grad=no_grad, device=device):
                        with torch.no_grad() if no_grad else nullcontext():
                            if profile == 'text':
                                indices = torch.arange(0, dim, 2, dtype=torch.int64)
                                values = indices.to(device=None if device == 'cpu' else target, dtype=torch.float)
                            else:
                                kwargs = {} if device == 'cpu' else {'device': target}
                                values = torch.arange(0, dim, 2, dtype=torch.float, **kwargs)
                            exponent = values / dim
                            power = base ** exponent
                            inverse = 1.0 / power
                            cloned = inverse.clone()
                        for value in (values, exponent, power, inverse, cloned):
                            self.assertFalse(value.requires_grad)
                            self.assertEqual(value.dtype, torch.float32)
                            self.assertEqual(tuple(value.shape), (dim//2,))
                            self._placement(value, target)
                        # No byte-equality claim: reciprocal/pow numerics are
                        # intentionally unchanged by the constant-grad fix.

    def test_cpu(self):
        self._check('cpu')

    def test_cuda(self):
        self._check('cuda')

    def test_rope_flags_cpu(self):
        self._rope('cpu')

    def test_rope_flags_cuda(self):
        self._rope('cuda')


if __name__ == '__main__':
    unittest.main()

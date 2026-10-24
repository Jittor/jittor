"""Public where constants do not acquire gradient edges; real branches retain them.

Independent runtimes select the same CPU/CUDA methods. CUDA validation must
select test_cuda explicitly; no model/PPO completeness is implied by this unit.
"""
from contextlib import ExitStack, nullcontext
import os
import sys
import unittest

import numpy as np


class TestWhereScalarConstants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.jt = None
        if os.environ.get('JITTOR_TORCH_SHIM') == '1':
            import jittor as jt
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            cls.jt = jt
        else:
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules)

    def _resident(self, tensor, device):
        self.assertEqual(tensor.device.type, device)
        if device == 'cuda':
            self.assertEqual(tensor.device.index, 0)
        if self.jt is not None:
            native = getattr(tensor, '_var', tensor)
            native.sync()
            index = 0 if device == 'cuda' else -1
            self.assertEqual(self.jt.core.dispatch_context([native]), (device, index))
            self.assertEqual(int(native.device_id), index)
            self.assertEqual(int(native.placement_backend), int(device == 'cuda'))
            self.assertEqual(native.location(), 'device' if device == 'cuda' else 'cpu')

    def _values(self, value, expected, device):
        self._resident(value, device)
        before = (value.requires_grad, value.is_leaf, value.grad_fn is None)
        with self.torch.no_grad():
            copy = value.clone()
            self._resident(copy, device)
            actual = np.array(copy.cpu().numpy(), copy=True)
        self.assertEqual((value.requires_grad, value.is_leaf, value.grad_fn is None), before)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def _check(self, device):
        torch = self.torch
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest('CUDA unavailable; no CUDA claim')
        with ExitStack() as stack:
            if self.jt is not None:
                from jittor._runtime.fallback import forbid_backend_fallbacks
                flags = dict(use_cuda=int(device == 'cuda'), backend_fallback='error')
                if device == 'cuda':
                    flags['device_id'] = 0
                stack.enter_context(self.jt.runtime.scope(**flags))
                stack.enter_context(forbid_backend_fallbacks())
                before = self.jt.core.backend_fallback_count()
                self.assertEqual(before, 0)
            target = 'cuda:0' if device == 'cuda' else 'cpu'
            mask_array = np.array([[True, False], [False, True]], dtype=bool)
            cot_array = np.array([[.125, -.1875], [.15625, -.109375]], dtype=np.float32)
            for reverse in (False, True):
                for trainable in (False, True):
                    for no_grad in (False, True):
                        for scalar in (-2.0, 2):
                            with self.subTest(reverse=reverse, trainable=trainable, no_grad=no_grad, scalar=scalar):
                                mask = torch.tensor(mask_array, dtype=torch.bool, device=target)
                                branch = torch.tensor(0.0, dtype=torch.float32, device=target, requires_grad=trainable)
                                with torch.no_grad() if no_grad else nullcontext():
                                    result = (torch.where(mask, scalar, branch) if reverse else
                                              branch.where(mask, scalar))
                                expected_grad = trainable and not no_grad
                                self.assertEqual(result.requires_grad, expected_grad)
                                self.assertEqual(result.is_leaf, not expected_grad)
                                self.assertEqual(result.grad_fn is None, not expected_grad)
                                self.assertEqual(result.dtype, torch.float32)
                                self.assertEqual(branch.requires_grad, trainable)
                                self.assertTrue(branch.is_leaf)
                                self.assertIsNone(branch.grad)
                                expected = np.where(mask_array, scalar, 0.0) if reverse else np.where(mask_array, 0.0, scalar)
                                self._values(result, expected.astype(np.float32), device)
                                if expected_grad:
                                    cot = torch.tensor(cot_array, dtype=torch.float32, device=target)
                                    gradient, = torch.autograd.grad(result, (branch,), grad_outputs=cot)
                                    select = ~mask_array if reverse else mask_array
                                    self._values(gradient, np.array(cot_array[select].sum(), dtype=np.float32), device)
                                    self.assertIsNone(branch.grad)
                                self._values(branch, np.array(0.0, dtype=np.float32), device)
            # Exact downstream mask construction, with a paired explicit-Tensor control.
            for tensor_minimum in (False, True):
                mask = torch.tensor(mask_array, dtype=torch.bool, device=target)
                zero = torch.tensor(0.0, dtype=torch.float32, device=target)
                minimum = torch.finfo(torch.float32).min
                other = (torch.tensor(minimum, dtype=torch.float32, device=target)
                         if tensor_minimum else minimum)
                result = torch.where(mask, zero, other)
                self.assertFalse(result.requires_grad)
                self.assertTrue(result.is_leaf)
                self.assertIsNone(result.grad_fn)
                self._values(result, np.where(mask_array, 0.0, minimum).astype(np.float32), device)
            if self.jt is not None:
                self.jt.sync_all(True)
                self.assertEqual(self.jt.core.backend_fallback_count(), before)

    def test_cpu(self):
        self._check('cpu')

    def test_cuda(self):
        self._check('cuda')


if __name__ == '__main__':
    unittest.main()

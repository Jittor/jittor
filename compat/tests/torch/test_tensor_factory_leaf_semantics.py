"""Fresh tensor factories are leaves; ordinary casts retain their graph.

No repair/stop_grad/detach calls. CPU and CUDA selectors run in separate real
native/Jittor processes; host tests only validate this proposal's contracts.
All numeric comparisons use binary-exact four-element fixtures and 1e-5.
No arrays are written/printed. Returned-gradient requires_grad remains a
separate known gap, not part of the factory-leaf assertions.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import os
import sys
import unittest

import numpy as np

X = np.array([1., 2., 4., 8.], dtype=np.float32)
COT = np.array([1/8, 3/16, 5/32, 7/64], dtype=np.float32)
DTYPES = ('float16', 'bfloat16', 'float32', 'float64')


class TestTensorFactoryLeaf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mode = os.environ.get('JITTOR_FACTORY_LEAF_RUNTIME',
                              'candidate' if os.environ.get('JITTOR_TORCH_SHIM') == '1' else 'oracle')
        cls.jt = None
        if mode == 'candidate':
            import jittor as jt
            import torch
            from jittor.compat.torch import namespace_owner
            from jittor.compat.shim.runtime import activation_status
            assert namespace_owner(torch) is jt and activation_status(jt).active
            cls.jt = jt
        elif mode == 'oracle':
            import torch
            assert hasattr(torch, '_C') and not hasattr(torch, '_torch_compat_install_context')
            assert not any(n == 'jittor' or n.startswith('jittor.') for n in sys.modules)
        else:
            raise ValueError('JITTOR_FACTORY_LEAF_RUNTIME must be candidate or oracle')
        cls.torch = torch

    @contextmanager
    def _device(self, device):
        with ExitStack() as stack:
            if device == 'cuda':
                if not self.torch.cuda.is_available():
                    self.skipTest('CUDA unavailable: no CUDA claim')
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

    def _resident(self, tensor, target):
        device = target.split(':')[0]
        self.assertEqual(tensor.device.type, device)
        if self.jt is None:
            if device == 'cuda':
                self.assertEqual(tensor.device.index, 0)
                self.torch.cuda.synchronize(tensor.device)
        else:
            native = getattr(tensor, '_var', tensor)
            native.sync()
            index = -1 if device == 'cpu' else 0
            self.assertEqual(self.jt.core.dispatch_context([native]), (device, index))
            self.assertEqual(int(native.device_id), index)
            self.assertEqual(int(native.placement_backend), 0 if device == 'cpu' else 1)
            self.assertEqual(native.location(), 'cpu' if device == 'cpu' else 'device')

    def _compare(self, value, expected, target):
        self._resident(value, target)
        before = (tuple(value.shape), value.dtype, bool(value.requires_grad), bool(value.is_leaf))
        with self.torch.no_grad():
            copy = value.clone()
            self.assertIsNot(copy, value)
            self._resident(copy, target)
            if copy.dtype == self.torch.bfloat16:
                copy = copy.float()
                self._resident(copy, target)
            host = copy.cpu()
            self._resident(host, 'cpu')
            array = np.array(host.numpy(), copy=True)
        self._resident(value, target)
        self.assertEqual((tuple(value.shape), value.dtype, bool(value.requires_grad), bool(value.is_leaf)), before)
        self.assertEqual(array.shape, expected.shape)
        self.assertTrue(np.isfinite(array).all())
        self.assertTrue(np.allclose(array, expected, atol=1e-5, rtol=1e-5),
                        'complete array differs from fixed analytic reference')

    def _factory(self, device):
        torch = self.torch
        with self._device(device) as target:
            for dtype_name in DTYPES:
                for kind in ('list', 'numpy', 'tensor'):
                    for requires in (False, True):
                        for no_grad in (False, True):
                            with self.subTest(dtype=dtype_name, kind=kind, requires=requires, no_grad=no_grad):
                                source = None
                                if kind == 'tensor':
                                    source = torch.tensor(X, dtype=torch.float32, device=target, requires_grad=True)
                                    self.assertTrue(source.is_leaf)
                                    data = source
                                else:
                                    data = X.tolist() if kind == 'list' else X.copy()
                                with torch.no_grad() if no_grad else nullcontext():
                                    value = torch.tensor(data, dtype=getattr(torch, dtype_name),
                                                         device=target, requires_grad=requires)
                                # Factory requires_grad is an explicit exception to no_grad.
                                self.assertEqual(bool(value.requires_grad), requires)
                                self.assertTrue(value.is_leaf, 'fresh tensor must not retain internal cast history')
                                self.assertIsNone(value.grad_fn)
                                self.assertIsNone(value.grad)
                                if source is not None:
                                    self.assertIsNot(value, source)
                                    self.assertTrue(source.requires_grad)
                                if requires:
                                    cot = torch.tensor(COT, dtype=value.dtype, device=target)
                                    loss = (value * cot).sum()
                                    loss.backward()
                                    self.assertIsNotNone(value.grad, 'newly constructed leaf must receive all gradients')
                                    self.assertEqual(value.grad.dtype, value.dtype)
                                    self._compare(value.grad, COT, target)
                                    self._resident(loss, target)
                                self._compare(value, X, target)
                                if source is not None:
                                    self.assertIsNone(source.grad, 'torch.tensor(existing) must not backprop to existing')
                                    self.assertTrue(source.is_leaf)
                                    self._compare(source, X, target)

    def _cast_controls(self, device):
        torch = self.torch
        with self._device(device) as target:
            for constructor in ('arange', 'tensor'):
                for operation in ('to', 'as_tensor'):
                    with self.subTest(constructor=constructor, operation=operation):
                        source = (torch.arange(1, 5, dtype=torch.float32, device=target, requires_grad=True)
                                  if constructor == 'arange' else
                                  torch.tensor(X, dtype=torch.float32, device=target, requires_grad=True))
                        self.assertTrue(source.is_leaf)
                        converted = (source.to(dtype=torch.bfloat16) if operation == 'to' else
                                     torch.as_tensor(source, dtype=torch.bfloat16, device=target))
                        self.assertIsNot(converted, source)
                        self.assertTrue(converted.requires_grad)
                        self.assertFalse(converted.is_leaf, 'ordinary differentiable cast must remain connected')
                        self.assertIsNotNone(converted.grad_fn)
                        cot = torch.tensor(COT, dtype=torch.bfloat16, device=target)
                        (converted * cot).sum().backward()
                        self.assertIsNotNone(source.grad)
                        self._compare(source.grad, COT, target)
                        # Explicitly absent retain_grad: don't accidentally publish nonleaf.grad.
                        self.assertIsNone(converted.grad)
                        self._resident(converted, target)
            for requires in (False, True):
                with self.subTest(integer_arange_then_enable=requires):
                    integer = torch.arange(1, 5, dtype=torch.int64, device=target)
                    converted = integer.to(dtype=torch.float32).to(dtype=torch.bfloat16)
                    self.assertFalse(converted.requires_grad)
                    self.assertTrue(converted.is_leaf)
                    converted.requires_grad_(requires)
                    self.assertEqual(bool(converted.requires_grad), requires)
                    self.assertTrue(converted.is_leaf)
                    if requires:
                        cot = torch.tensor(COT, dtype=torch.bfloat16, device=target)
                        (converted * cot).sum().backward()
                        self.assertIsNotNone(converted.grad)
                        self._compare(converted.grad, COT, target)
                    self._resident(integer, target)
                    self._resident(converted, target)

    def test_owned_base_backward_cuda(self):
        """Normal public Tensor has a real root and publishes every CUDA grad."""
        torch = self.torch
        base = torch._C.TensorBase
        self.assertEqual(torch.Tensor.__bases__, (base,))
        self.assertEqual(base.__bases__, (object,))
        if self.jt is not None:
            self.assertFalse(issubclass(torch.Tensor, self.jt.core.Var))
        with self._device('cuda') as target:
            x = torch.tensor(X, dtype=torch.float32, device=target, requires_grad=True)
            loss = (x * x).sum()
            self._resident(loss, target)
            loss.backward()
            self.assertIsNotNone(x.grad)
            if self.jt is not None:
                from jittor._core.dtypes import var_dtype_name
                self.assertEqual(var_dtype_name(x), 'float32')
                self.assertEqual(var_dtype_name(x.grad), 'float32')
            self._compare(x.grad, np.array([2., 4., 8., 16.], dtype=np.float32), target)
            self._compare(x, X, target)
            self.assertTrue(x.requires_grad)
            self.assertTrue(x.is_leaf)
            host = x.detach().cpu()
            self._resident(host, 'cpu')
            restored = host.cuda()
            self._compare(restored, X, target)
            self._resident(host, 'cpu')
            self._compare(x, X, target)

    def test_factory_cpu(self):
        self._factory('cpu')

    def test_factory_cuda(self):
        self._factory('cuda')

    def test_cast_controls_cpu(self):
        self._cast_controls('cpu')

    def test_cast_controls_cuda(self):
        self._cast_controls('cuda')


if __name__ == '__main__':
    unittest.main()

"""CUDA global max/min values, complete input VJP, dtype and source contracts."""
from contextlib import nullcontext
import unittest

import numpy as np
import torch
import pytest

pytestmark = pytest.mark.cuda


class TestMaxMinGlobalCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():self.skipTest('actual CUDA required')
        if hasattr(torch, '_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope = forbid_backend_fallbacks()
        else:
            scope = nullcontext()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def resident(self, value):
        self.assertIs(type(value), torch.Tensor)
        self.assertEqual(str(value.device), 'cuda:0')
        if hasattr(torch, '_torch_compat_install_context'):
            import jittor as jt
            value.sync()
            self.assertEqual(jt.core.dispatch_context([value]), ('cuda', 0))
            self.assertEqual((value.location(), value.device_id, value.placement_backend), ('device', 0, 1))
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def check(self, value, expected):
        self.resident(value)
        actual = value.detach().clone().cpu().numpy().copy()
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def exercise(self, method):
        fixtures = (np.array([[-2., 3., 3.], [1., -2., 0.]], dtype=np.float32),
                    np.array([[-7., 2., 4.], [1., -3., 0.]], dtype=np.float32))
        for values in fixtures:
            for name, reference in (('max', np.max), ('min', np.min)):
                for nonleaf in (False, True):
                    with self.subTest(name=name, nonleaf=nonleaf, tied=values[0, 1] == values[0, 2]):
                        x = torch.tensor(values, dtype=torch.float32, device='cuda:0', requires_grad=True)
                        gain = 2 if nonleaf else 1
                        source = x * gain if nonleaf else x
                        y = getattr(source, name)() if method else getattr(torch, name)(source)
                        self.assertEqual(tuple(y.shape), ())
                        self.assertEqual(y.dtype, x.dtype)
                        self.assertTrue(y.requires_grad)
                        self.assertFalse(y.is_leaf)
                        target = np.asarray(reference(values * gain), dtype=np.float32)
                        self.check(y, target)
                        dy = torch.tensor(1.25, dtype=torch.float32, device='cuda:0')
                        dx, = torch.autograd.grad(y, (x,), grad_outputs=dy, create_graph=False)
                        mask = values == reference(values)
                        expected_dx = mask.astype(np.float32) * (1.25 * gain / int(mask.sum()))
                        self.check(dx, expected_dx)
                        self.assertEqual(dx.dtype, x.dtype)
                        self.assertFalse(dx.requires_grad)
                        self.assertIsNone(x.grad)
                        self.check(x, values)
                        self.check(source, values * gain)

    def test_cuda_functional_global_max_min_full_input_vjp(self):
        self.exercise(False)

    def test_cuda_method_global_max_min_full_input_vjp(self):
        self.exercise(True)

    def test_cuda_no_grad_global_max_min_dtype_and_source(self):
        for dtype, host_dtype in ((torch.float32, np.float32), (torch.int64, np.int64)):
            values = np.asarray([[-2, 4], [3, 1]], dtype=host_dtype)
            for name, reference in (('max', np.max), ('min', np.min)):
                with self.subTest(dtype=dtype, name=name):
                    x = torch.tensor(values, dtype=dtype, device='cuda:0', requires_grad=dtype is torch.float32)
                    before = (x.requires_grad, x.is_leaf, x.grad_fn is None)
                    with torch.no_grad():
                        a = getattr(x, name)()
                        b = getattr(torch, name)(x)
                    for value in (a, b):
                        self.assertEqual(value.dtype, dtype)
                        self.assertEqual(tuple(value.shape), ())
                        self.assertFalse(value.requires_grad)
                        self.check(value, np.asarray(reference(values), dtype=host_dtype))
                    self.check(x, values)
                    self.assertEqual((x.requires_grad, x.is_leaf, x.grad_fn is None), before)

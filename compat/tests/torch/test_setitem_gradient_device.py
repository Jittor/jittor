"""Public CUDA CopySlices contracts; run Native and compat independently."""
from contextlib import contextmanager, nullcontext
import unittest

import numpy as np
import pytest

pytestmark = pytest.mark.cuda


class TestSetitemGradientDevice(unittest.TestCase):
    @contextmanager
    def cuda(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('real CUDA required')
        torch.cuda.set_device(0); jt = None
        if hasattr(torch, '_torch_compat_install_context'):
            import jittor as jt
            self.assertEqual(jt.core.backend_fallback_count(), 0)
        self.jt = jt
        scope = nullcontext() if jt is None else jt.runtime.scope(use_cuda=1, device_id=0,
            backend_fallback='error', cuda_kernel_math='strict')
        with scope:
            yield torch
            if jt is not None: self.assertEqual(jt.core.backend_fallback_count(), 0)

    def flags(self, value):
        return (value.requires_grad, value.is_leaf, value.grad_fn is None, value.grad is None)

    def array(self, torch, value):
        self.assertEqual(value.device, torch.device('cuda:0'))
        before = self.flags(value)
        if self.jt is not None:
            native = getattr(value, '_var', value); native.sync()
            self.assertEqual(self.jt.core.dispatch_context([native]), ('cuda', 0))
            self.assertEqual(native.location(), 'device')
        with torch.no_grad(): result = np.array(value.detach().cpu().numpy(), copy=True)
        self.assertEqual(self.flags(value), before)
        self.assertTrue(np.isfinite(result).all()); return result

    def compare(self, torch, value, expected):
        actual = self.array(torch, value); expected = np.asarray(expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_cuda_stopped_root_and_view_gain_edge(self):
        with self.cuda() as torch:
            shape = (1, 2, 2, 3, 4); j = np.arange(48).reshape(shape)
            a_np = (((j*7+3) % 31-15)/23).astype(np.float32)
            b_np = (((j*11+5) % 29-14)/19).astype(np.float32)
            dy_np = (((j*13+7) % 37-18)/17).astype(np.float32)
            for view in (False, True):
                a = torch.tensor(a_np, device='cuda:0', requires_grad=True)
                b = torch.tensor(b_np, device='cuda:0', requires_grad=True)
                dy = torch.tensor(dy_np, device='cuda:0'); out = torch.zeros_like(a)
                if view: out = out.view(-1).view(shape)
                held = out; self.assertEqual(self.flags(out), (False, True, True, True))
                for i in range(2):
                    rhs = a[:, :, i]*1.25 + a[:, :, i]*b[:, :, i]
                    out[:, :, i] = rhs
                    self.assertIs(out, held)
                    self.assertEqual(self.flags(out), (True, False, False, True))
                da, db = torch.autograd.grad(out, (a, b), grad_outputs=dy, allow_unused=False,
                    create_graph=False, retain_graph=False)
                self.compare(torch, out, (a_np.astype(np.float64)*(1.25+b_np)).astype(np.float32))
                self.compare(torch, da, (dy_np.astype(np.float64)*(1.25+b_np)).astype(np.float32))
                self.compare(torch, db, (dy_np.astype(np.float64)*a_np).astype(np.float32))
                self.assertIsNone(a.grad); self.assertIsNone(b.grad)
                self.compare(torch, a, a_np); self.compare(torch, b, b_np)

    def test_cuda_partial_nonleaf_broadcast_preserves_old_edge(self):
        with self.cuda() as torch:
            x_np = (np.arange(12).reshape(3, 4)/23).astype(np.float32)
            rhs_np = np.array([[3., -2., 7., 1.]], dtype=np.float32)/19
            dy_np = ((np.arange(12).reshape(3, 4)*3-11)/17).astype(np.float32)
            x = torch.tensor(x_np, device='cuda:0', requires_grad=True)
            rhs = torch.tensor(rhs_np, device='cuda:0', requires_grad=True)
            out = x*2; out[1:3] = rhs
            dx, drhs = torch.autograd.grad(out, (x, rhs), grad_outputs=torch.tensor(dy_np, device='cuda:0'))
            want = x_np*2; want[1:3] = rhs_np
            dx_np = dy_np*2; dx_np[1:3] = 0
            self.compare(torch, out, want); self.compare(torch, dx, dx_np)
            self.compare(torch, drhs, dy_np[1:3].sum(axis=0, keepdims=True))
            self.assertIsNone(x.grad); self.assertIsNone(rhs.grad)

    def test_cuda_leaf_and_leaf_view_inplace_rejected(self):
        with self.cuda() as torch:
            values = np.array([1., -2., 3., -4.], dtype=np.float32)
            for view in (False, True):
                base = torch.tensor(values, device='cuda:0', requires_grad=True)
                target = base.view(2, 2) if view else base
                with self.assertRaises(RuntimeError): target[...] = 2.0
                self.compare(torch, base, values)
                self.assertEqual(self.flags(base), (True, True, True, True))

    def test_cuda_no_grad_and_frozen_constant_writes(self):
        with self.cuda() as torch:
            original = np.array([1., -2., 3., -4.], dtype=np.float32)
            trainable = torch.tensor(original, device='cuda:0', requires_grad=True)
            frozen = torch.tensor(original, device='cuda:0', requires_grad=False)
            frozen[1:3] = 0.5
            self.assertEqual(self.flags(frozen), (False, True, True, True))
            with torch.no_grad():
                stopped = torch.zeros_like(trainable); stopped[:2] = trainable[1:3]
                trainable[1:3] = -0.25
            self.assertEqual(self.flags(stopped), (False, True, True, True))
            self.assertEqual(self.flags(trainable), (True, True, True, True))
            want = original.copy(); want[1:3] = 0.5
            self.compare(torch, frozen, want)
            self.compare(torch, stopped, np.array([-2., 3., 0., 0.], dtype=np.float32))
            want = original.copy(); want[1:3] = -0.25
            self.compare(torch, trainable, want)

    def test_cuda_scalar_mask_region_broadcast_and_complex(self):
        with self.cuda() as torch:
            mask = torch.tensor([True, False, True], device='cuda:0')
            out = torch.zeros((3, 2), device='cuda:0', dtype=torch.float32); out[mask] = 2.0
            self.compare(torch, out, np.array([[2., 2.], [0., 0.], [2., 2.]], dtype=np.float32))
            self.assertEqual(self.flags(out), (False, True, True, True))
            rhs_np = np.array([[[1., -2.], [3., -4.]]], dtype=np.float32)
            rhs = torch.tensor(rhs_np, device='cuda:0', requires_grad=True)
            out[mask] = rhs
            dy_np = np.arange(6, dtype=np.float32).reshape(3, 2)/7
            gradient, = torch.autograd.grad(out, (rhs,), grad_outputs=torch.tensor(dy_np, device='cuda:0'))
            self.compare(torch, out, np.array([[1., -2.], [0., 0.], [3., -4.]], dtype=np.float32))
            self.compare(torch, gradient, dy_np[[0, 2]][None])
            z = torch.zeros(3, device='cuda:0', dtype=torch.complex64); z[1] = 1+2j
            self.compare(torch, z, np.array([0, 1+2j, 0], dtype=np.complex64))
            self.assertEqual(self.flags(z), (False, True, True, True))

    def test_cuda_reversibly_frozen_destination_gains_rhs_edge(self):
        with self.cuda() as torch:
            values = np.arange(6, dtype=np.float32).reshape(3, 2)/17
            rhs_values = np.array([[2., -3.]], dtype=np.float32)/19
            dy_values = np.arange(6, dtype=np.float32).reshape(3, 2)/23
            for as_view in (False, True):
                base = torch.tensor(values, device='cuda:0', requires_grad=True)
                base.requires_grad_(False)
                old = base*3
                target = base.view(-1).view(3, 2) if as_view else base
                rhs = torch.tensor(rhs_values, device='cuda:0', requires_grad=True)
                target[1:2] = rhs
                self.assertTrue(target.requires_grad)
                self.assertFalse(target.is_leaf)
                gradient, = torch.autograd.grad(target, (rhs,),
                    grad_outputs=torch.tensor(dy_values, device='cuda:0'))
                expected = values.copy(); expected[1:2] = rhs_values
                self.compare(torch, target, expected)
                self.compare(torch, gradient, dy_values[1:2])
                self.compare(torch, old, values*3)
                self.assertFalse(old.requires_grad)
                self.assertIsNone(rhs.grad)

    def test_cuda_integer_destination_does_not_gain_rhs_edge(self):
        with self.cuda() as torch:
            rhs = torch.tensor([1.5, -2.5], device='cuda:0', requires_grad=True)
            target = torch.zeros(4, device='cuda:0', dtype=torch.int64)
            target[1:3] = rhs
            self.assertFalse(target.requires_grad)
            self.assertTrue(target.is_leaf)
            self.compare(torch, target, np.array([0, 1, -2, 0], dtype=np.int64))
            self.assertTrue(rhs.requires_grad)
            self.assertIsNone(rhs.grad)

    def test_cuda_CPU_rhs_staging_keeps_CUDA_destination(self):
        with self.cuda() as torch:
            x_np = np.arange(6, dtype=np.float32).reshape(3, 2)/23
            host_np = np.array([[7., -3.]], dtype=np.float32)/19
            # A stopped host staging value, not a CPU training/computation path.
            host = torch.tensor(host_np, device='cpu', requires_grad=False)
            x = torch.tensor(x_np, device='cuda:0', requires_grad=True)
            out = x*2; out[1:2] = host
            dx, = torch.autograd.grad(out, (x,), grad_outputs=torch.ones_like(out))
            want = x_np*2; want[1:2] = host_np
            dx_np = np.full((3, 2), 2., dtype=np.float32); dx_np[1:2] = 0
            self.compare(torch, out, want); self.compare(torch, dx, dx_np)
            self.assertEqual(host.device, torch.device('cpu'))
            np.testing.assert_array_equal(host.numpy(), host_np)

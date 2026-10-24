"""LayerNorm uses native math with owned CUDA tensors and intact full VJPs."""
from contextlib import nullcontext
import unittest

import numpy as np
import torch


class TestLayerNormCarrierCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("real CUDA required")
        if hasattr(torch, "_torch_compat_install_context"):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope = forbid_backend_fallbacks()
        else:
            scope = nullcontext()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def check(self, value, expected):
        self.assertIs(type(value), torch.Tensor)
        self.assertEqual(str(value.device), "cuda:0")
        if hasattr(torch, "_torch_compat_install_context"):
            import jittor as jt
            value.sync()
            self.assertEqual(jt.core.dispatch_context([value]), ("cuda", 0))
            self.assertEqual((value.location(), value.device_id, value.placement_backend), ("device", 0, 1))
            self.assertEqual(jt.core.backend_fallback_count(), 0)
        actual = value.detach().clone().cpu().numpy().copy()
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def fixture(self, shape=(2, 4), rank=1):
        x = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 3) / 13
        normalized = shape[-rank:]
        w = (np.arange(np.prod(normalized), dtype=np.float32).reshape(normalized) + 1) / 7
        b = (np.arange(np.prod(normalized), dtype=np.float32).reshape(normalized) - 2) / 11
        dy = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 4) / 19
        return x, w, b, dy

    def reference(self, x, w, b, dy, rank=1):
        axes = tuple(range(x.ndim - rank, x.ndim))
        centered = x - x.mean(axis=axes, keepdims=True)
        inv = 1 / np.sqrt((centered * centered).mean(axis=axes, keepdims=True) + np.float32(1e-5))
        xhat = centered * inv
        weighted = dy if w is None else dy * w
        dx = inv * (weighted - weighted.mean(axis=axes, keepdims=True)
                    - xhat * (weighted * xhat).mean(axis=axes, keepdims=True))
        outer = tuple(range(x.ndim - rank))
        dw = (dy * xhat).sum(axis=outer) if outer else dy * xhat
        db = dy.sum(axis=outer) if outer else dy
        return xhat * (1 if w is None else w) + (0 if b is None else b), dx, dw, db

    def exercise(self, shape, rank=1, affine=(True, True), module=False, nonleaf=False):
        x, w, b, dy = self.fixture(shape, rank)
        tx = torch.tensor(x, device="cuda:0", requires_grad=True)
        source = tx * 2 if nonleaf else tx
        w = w if affine[0] else None
        b = b if affine[1] else None
        tw = None if w is None else torch.tensor(w, device="cuda:0", requires_grad=True)
        tb = None if b is None else torch.tensor(b, device="cuda:0", requires_grad=True)
        if module:
            layer = torch.nn.LayerNorm(shape[-rank:], elementwise_affine=affine[0], bias=affine[1], device="cuda:0")
            if tw is not None:
                layer.weight = torch.nn.Parameter(tw)
                tw = layer.weight
            if tb is not None:
                layer.bias = torch.nn.Parameter(tb)
                tb = layer.bias
            y = layer(source)
        else:
            y = torch.nn.functional.layer_norm(source, shape[-rank:], tw, tb, 1e-5)
        self.assertTrue(y.requires_grad)
        self.assertFalse(y.is_leaf)
        tensors = [tx] + ([tw] if tw is not None else []) + ([tb] if tb is not None else [])
        grads = torch.autograd.grad(y, tuple(tensors), grad_outputs=torch.tensor(dy, device="cuda:0"), allow_unused=False)
        expected, dx, dw, db = self.reference(x * (2 if nonleaf else 1), w, b, dy, rank)
        self.check(y, expected)
        refs = [dx * (2 if nonleaf else 1)] + ([dw] if tw is not None else []) + ([db] if tb is not None else [])
        for got, ref in zip(grads, refs):
            self.check(got, ref)
        self.check(tx, x)
        for tensor in tensors:
            self.assertIsNone(tensor.grad)
            self.assertTrue(tensor.requires_grad)

    def test_no_grad_tensor_affine_forward_full_values(self):
        x, w, b, dy = self.fixture()
        tensors = [torch.tensor(v, device="cuda:0", requires_grad=True) for v in (x, w, b)]
        with torch.no_grad():
            output = torch.nn.functional.layer_norm(tensors[0], (4,), tensors[1], tensors[2], 1e-5)
        self.check(output, self.reference(x, w, b, dy)[0])
        self.assertFalse(output.requires_grad)
        for tensor, expected in zip(tensors, (x, w, b)):
            self.check(tensor, expected)
            self.assertTrue(tensor.requires_grad)
            self.assertIsNone(tensor.grad)

    def test_functional_full_input_weight_bias_vjp(self):
        for shape, rank in (((4,), 1), ((2, 4), 1), ((2, 3, 4), 1), ((2, 3, 4), 2)):
            with self.subTest(shape=shape, rank=rank):
                self.exercise(shape, rank)

    def test_functional_optional_affine_and_nonleaf_vjp(self):
        for affine in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(affine=affine):
                self.exercise((2, 4), affine=affine, nonleaf=True)

    def test_owned_module_full_input_weight_bias_vjp(self):
        for affine in ((True, True), (True, False), (False, False)):
            with self.subTest(affine=affine):
                self.exercise((2, 4), affine=affine, module=True)

    def test_noncontiguous_input_full_vjp(self):
        original, w, b, unused = self.fixture((2, 4, 3), 1)
        x = original.transpose(0, 2, 1).copy()
        w = np.asarray([.75, 1.25, -.5, .25], dtype=np.float32)
        b = np.asarray([.1, -.2, .05, -.1], dtype=np.float32)
        dy = np.arange(x.size, dtype=np.float32).reshape(x.shape) / 17
        source = torch.tensor(original, device="cuda:0", requires_grad=True)
        tw = torch.tensor(w, device="cuda:0", requires_grad=True)
        tb = torch.tensor(b, device="cuda:0", requires_grad=True)
        y = torch.nn.functional.layer_norm(source.transpose(-1, -2), (4,), tw, tb, 1e-5)
        grads = torch.autograd.grad(y, (source, tw, tb), grad_outputs=torch.tensor(dy, device="cuda:0"))
        output, dx, dw, db = self.reference(x, w, b, dy)
        self.check(y, output)
        for got, expected in zip(grads, (dx.transpose(0, 2, 1).copy(), dw, db)):
            self.check(got, expected)

    def test_rejects_shape_mismatch(self):
        x = torch.ones((2, 4), device="cuda:0")
        for shape in ((), (3,), (1, 4), (2, 2, 4)):
            with self.subTest(shape=shape), self.assertRaises(RuntimeError):
                torch.nn.functional.layer_norm(x, shape)
        for weight, bias in ((torch.ones((1, 4), device="cuda:0"), None),
                             (None, torch.ones((3,), device="cuda:0"))):
            with self.assertRaises(RuntimeError):
                torch.nn.functional.layer_norm(x, (4,), weight, bias)

"""CUDA triangular transforms retain public autograd and native math ownership."""
from contextlib import nullcontext
import unittest

import numpy as np
import torch


class TestTriangularAutogradCUDA(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "requires actual CUDA")
        if hasattr(torch, "_torch_compat_install_context"):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            self.scope = forbid_backend_fallbacks()
        else:
            self.scope = nullcontext()
        self.scope.__enter__()
        self.addCleanup(self.scope.__exit__, None, None, None)

    def resident(self, value):
        self.assertIs(type(value), torch.Tensor)
        self.assertEqual(str(value.device), "cuda:0")
        if hasattr(torch, "_torch_compat_install_context"):
            import jittor as jt
            value.sync()
            self.assertEqual(jt.core.dispatch_context([value]), ("cuda", 0))
            self.assertEqual((value.location(), value.device_id, value.placement_backend), ("device", 0, 1))
            self.assertEqual(jt.core.backend_fallback_count(), 0)

    def check(self, value, expected):
        self.resident(value)
        actual = value.detach().clone().cpu().numpy().copy()
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def exercise(self, method):
        for name, reference in (("tril", np.tril), ("triu", np.triu)):
            for shape in ((3, 4), (2, 3, 4), (1, 1)):
                values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 19 - .3
                weights = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 23 - .2
                for diagonal in (-1, 0, 1):
                    for nonleaf in (False, True):
                        with self.subTest(name=name, shape=shape, diagonal=diagonal, nonleaf=nonleaf):
                            x = torch.tensor(values, device="cuda:0", requires_grad=True)
                            source = x * 2 if nonleaf else x
                            y = getattr(source, name)(diagonal) if method else getattr(torch, name)(input=source, diagonal=diagonal)
                            self.assertTrue(y.requires_grad)
                            self.assertFalse(y.is_leaf)
                            self.check(y, reference(values * (2 if nonleaf else 1), k=diagonal))
                            dy = torch.tensor(weights, device="cuda:0")
                            grad, = torch.autograd.grad(y, (x,), grad_outputs=dy, allow_unused=False)
                            self.check(grad, reference(weights, k=diagonal) * (2 if nonleaf else 1))
                            self.check(x, values)
                            self.assertIsNone(x.grad)

    def test_functional_values_and_full_input_vjp(self):
        self.exercise(False)

    def test_methods_values_and_full_input_vjp(self):
        self.exercise(True)

    def test_no_grad_dtype_and_input_unchanged(self):
        for dtype in (torch.float32, torch.int64, torch.bool):
            for name, reference in (("tril", np.tril), ("triu", np.triu)):
                x = torch.ones((2, 3), dtype=dtype, device="cuda:0", requires_grad=dtype is torch.float32)
                with torch.no_grad():
                    y = getattr(x, name)()
                self.assertFalse(y.requires_grad)
                self.assertEqual(y.dtype, dtype)
                self.check(y, reference(np.ones((2, 3))))
                self.check(x, np.ones((2, 3)))

    def test_signature_rejects_invalid_options(self):
        x = torch.ones((2, 3), device="cuda:0")
        for name in ("tril", "triu"):
            fn = getattr(torch, name)
            for invalid in (1.5, True):
                with self.assertRaises(TypeError):
                    fn(x, diagonal=invalid)
            for invalid in ({"requires_grad": False}, {"device": "cuda"}, {"dtype": torch.float32}):
                with self.assertRaises(TypeError):
                    fn(x, **invalid)
            with self.assertRaises(RuntimeError):
                fn(torch.ones((3,), device="cuda:0"))
            self.check(fn(x, diagonal=np.int64(1)),
                       (np.tril if name == "tril" else np.triu)(np.ones((2, 3)), k=1))
            if hasattr(torch, "_torch_compat_install_context"):
                # Unsupported writes must fail explicitly, not silently ignore out.
                out = torch.ones((2, 3), device="cuda:0")
                with self.assertRaises(NotImplementedError):
                    fn(x, out=out)
                self.check(out, np.ones((2, 3)))

    def test_qwen_decay_mask_chain_full_input_vjp(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 97 - .1
        weights = np.arange(96, dtype=np.float32).reshape(2, 3, 4, 4) / 131 - .2
        x = torch.tensor(values, device="cuda:0", requires_grad=True)
        g = x.cumsum(dim=-1)
        y = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        ref_g = np.cumsum(values, axis=-1).astype(np.float64)
        expected = np.exp(ref_g[..., :, None] - ref_g[..., None, :]) * np.tril(np.ones((4, 4)))
        self.check(y, expected)
        grad, = torch.autograd.grad(y, (x,), grad_outputs=torch.tensor(weights, device="cuda:0"))
        weighted = weights.astype(np.float64) * expected
        dg = weighted.sum(-1) - weighted.sum(-2)
        self.check(grad, np.flip(np.cumsum(np.flip(dg, axis=-1), axis=-1), axis=-1))
        self.check(x, values)

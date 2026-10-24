"""Owned gather argument spellings and complete CUDA VJPs."""
from contextlib import nullcontext
import unittest
import numpy as np
import torch


class TestGatherCarrierCUDA(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("real CUDA required")
        scope = nullcontext()
        if hasattr(torch, '_torch_compat_install_context'):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope = forbid_backend_fallbacks()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def check(self, value, expected):
        self.assertIs(type(value), torch.Tensor)
        self.assertEqual(str(value.device), 'cuda:0')
        if hasattr(torch, '_torch_compat_install_context'):
            import jittor as jt
            value.sync()
            self.assertEqual(jt.core.dispatch_context([value]), ('cuda', 0))
            self.assertEqual(jt.core.backend_fallback_count(), 0)
        got = value.detach().clone().cpu().numpy().copy()
        self.assertEqual(got.shape, expected.shape)
        self.assertEqual(got.dtype, expected.dtype)
        if expected.dtype.kind in 'biu':
            np.testing.assert_array_equal(got, expected)
        else:
            np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)

    def gather_vjp(self, method):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 17
        indices = np.array([[[1, 1], [2, 0], [3, 1]],
                            [[0, 0], [2, 2], [1, 3]]], dtype=np.int64)
        seed = (np.arange(12, dtype=np.float32).reshape(2, 3, 2) - 4) / 19
        x = torch.tensor(values, device='cuda:0', requires_grad=True)
        ix = torch.tensor(indices, device='cuda:0')
        y = x.gather(dim=-1, index=ix) if method else torch.gather(input=x, dim=-1, index=ix)
        self.check(y, np.take_along_axis(values, indices, axis=-1))
        grad, = torch.autograd.grad(y, (x,), grad_outputs=torch.tensor(seed, device='cuda:0'))
        expected = np.zeros_like(values)
        for coord in np.ndindex(indices.shape):
            expected[coord[:-1] + (indices[coord],)] += seed[coord]
        self.check(grad, expected)
        self.check(x, values)
        self.assertIsNone(x.grad)

    def test_public_function_full_input_vjp(self):
        self.gather_vjp(False)

    def test_public_method_full_input_vjp(self):
        self.gather_vjp(True)

    def test_strided_input_full_vjp_and_integer_empty(self):
        source = np.arange(12, dtype=np.float32).reshape(3, 4) / 13
        ix = np.array([[2, 0], [1, 1], [0, 2], [2, 1]], dtype=np.int64)
        tx = torch.tensor(source, device='cuda:0', requires_grad=True)
        y = torch.gather(tx.transpose(0, 1), 1, torch.tensor(ix, device='cuda:0'))
        self.check(y, np.take_along_axis(source.T, ix, axis=1))
        grad, = torch.autograd.grad(y.sum(), (tx,))
        expected = np.zeros_like(source)
        for row, col in np.ndindex(ix.shape):
            expected[ix[row, col], row] += 1
        self.check(grad, expected)
        integer = torch.tensor(np.arange(12, dtype=np.int64).reshape(3, 4), device='cuda:0')
        ids = np.array([[1, 1, 0, 2], [2, 0, 1, 1]], dtype=np.int64)
        self.check(torch.gather(integer, 0, torch.tensor(ids, device='cuda:0')),
                   np.take_along_axis(np.arange(12, dtype=np.int64).reshape(3, 4), ids, axis=0))
        empty = torch.tensor(np.empty((3, 0), dtype=np.int64), device='cuda:0')
        self.check(integer.gather(1, empty), np.empty((3, 0), dtype=np.int64))

    def test_bad_index_rank_dimension_dtype_and_type(self):
        x = torch.ones((2, 3), device='cuda:0')
        ids = torch.zeros((2, 1), device='cuda:0', dtype=torch.int64)
        with self.assertRaises(IndexError):
            torch.gather(x, 2, ids)
        with self.assertRaises(RuntimeError):
            torch.gather(x, 1, ids.reshape(-1))
        with self.assertRaises(RuntimeError):
            torch.gather(x, 1, torch.zeros((3, 1), device='cuda:0', dtype=torch.int64))
        with self.assertRaises((TypeError, RuntimeError)):
            torch.gather(x, 1, ids.float())
        with self.assertRaises(TypeError):
            torch.gather(x, 1, [[0], [1]])
        if hasattr(torch, '_torch_compat_install_context'):
            with self.assertRaises(NotImplementedError):
                torch.gather(x, 1, ids, sparse_grad=True)
            with self.assertRaises(NotImplementedError):
                torch.gather(x, 1, ids, out=x)

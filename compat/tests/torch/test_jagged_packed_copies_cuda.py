"""CUDA jagged copies preserve complete packed storage, metadata and all VJPs."""
from contextlib import nullcontext
import unittest
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cuda


class TestJaggedPackedCopiesCUDA(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        scope = nullcontext()
        if hasattr(torch, "_torch_compat_install_context"):
            from jittor._runtime.fallback import forbid_backend_fallbacks
            scope = forbid_backend_fallbacks()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def check(self, value, expected):
        self.assertEqual(value.device, torch.device("cuda:0"))
        if hasattr(torch, "_torch_compat_install_context"):
            import jittor as jt
            value.sync()
            self.assertEqual(jt.core.dispatch_context([value]), ("cuda", 0))
            self.assertEqual(jt.core.backend_fallback_count(), 0)
        else:
            torch.cuda.synchronize()
        actual = value.detach().clone().cpu().numpy().copy()
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def holes(self, axis=1):
        shape = (8, 2) if axis == 1 else (2, 8)
        expected = np.arange(16, dtype=np.float32).reshape(shape) / 17
        values = torch.tensor(expected, device="cuda:0", requires_grad=True)
        offsets = torch.tensor([0, 3, 8], device="cuda:0", dtype=torch.int64)
        lengths = torch.tensor([1, 2], device="cuda:0", dtype=torch.int64)
        nested = torch.nested.nested_tensor_from_jagged(
            values, offsets=offsets, lengths=lengths, jagged_dim=axis)
        return expected, values, offsets, lengths, nested

    def test_preserve_clone_keeps_complete_holes_and_full_vjp(self):
        for axis in (1, 2):
            with self.subTest(axis=axis):
                expected, values, offsets, lengths, nested = self.holes(axis)
                cloned = nested.clone(memory_format=torch.preserve_format)
                self.assertIs(cloned.offsets(), offsets)
                self.assertIs(cloned.lengths(), lengths)
                self.assertFalse(cloned.is_contiguous())
                self.assertEqual(cloned.shape, nested.shape)
                self.check(cloned.values(), expected)
                self.assertNotEqual(cloned.values().data_ptr(), values.data_ptr())
                cotangent = np.arange(16, dtype=np.float32).reshape(expected.shape) / 19
                gradient, = torch.autograd.grad(
                    cloned.values(), values,
                    grad_outputs=torch.tensor(cotangent, device="cuda:0"))
                self.check(gradient, cotangent)
                self.check(values, expected)

    def test_dtype_cast_keeps_metadata_and_all_packed_gradients(self):
        for axis in (1, 2):
            with self.subTest(axis=axis):
                expected, values, offsets, lengths, nested = self.holes(axis)
                converted = nested.to(dtype=torch.float64)
                self.assertIs(converted.offsets(), offsets)
                self.assertIs(converted.lengths(), lengths)
                self.assertFalse(converted.is_contiguous())
                self.assertEqual(converted.shape, nested.shape)
                self.assertIs(converted.dtype, torch.float64)
                self.check(converted.values(), expected.astype(np.float64))
                cotangent = np.arange(16, dtype=np.float64).reshape(expected.shape) / 19
                gradient, = torch.autograd.grad(
                    converted.values(), values,
                    grad_outputs=torch.tensor(cotangent, device="cuda:0"))
                self.check(gradient, cotangent.astype(np.float32))
                self.check(values, expected)

    def test_same_dtype_to_is_identity_and_copy_is_independent(self):
        expected, values, offsets, lengths, nested = self.holes()
        self.assertIs(nested.to(dtype=torch.float32), nested)
        copied = nested.to(dtype=torch.float32, copy=True)
        self.assertIsNot(copied, nested)
        self.assertIs(copied.offsets(), offsets)
        self.assertIs(copied.lengths(), lengths)
        self.assertEqual(copied.shape, nested.shape)
        self.assertNotEqual(copied.values().data_ptr(), values.data_ptr())
        self.check(copied.values(), expected)

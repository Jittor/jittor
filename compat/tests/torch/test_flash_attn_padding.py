"""Padding facade contracts against independent NumPy values and gradients.

Run the same file with official Torch using JITTOR_FLASH_PADDING_RUNTIME=oracle.
JITTOR_FLASH_PADDING_DEVICE defaults to cpu; npu/cuda must be explicitly chosen.
Loading this pure-Torch submodule directly avoids needing an official CUDA
flash_attn extension merely to test its public padding helpers.
"""

from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
padding = None


class TestFlashAttnPadding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global padding
        spec = importlib.util.spec_from_file_location(
            "flash_padding_under_test", ROOT / "compat/shim/resources/stubs/flash_attn/bert_padding.py")
        padding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(padding)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.device = os.environ.get("JITTOR_FLASH_PADDING_DEVICE", "cpu")
        self.jt = None
        if os.environ.get("JITTOR_FLASH_PADDING_RUNTIME") == "oracle":
            self.assertTrue(hasattr(torch, "_C"))
            self.assertFalse(hasattr(torch, "_torch_compat_install_context"))
            if self.device == "npu":
                import torch_npu  # noqa: F401
        else:
            import jittor as jt
            from jittor.compat.shim.runtime import activation_status
            from jittor.compat.torch.publication import namespace_owner
            from jittor._runtime.fallback import forbid_backend_fallbacks
            self.assertIs(namespace_owner(torch), jt)
            self.assertTrue(activation_status(jt).active)
            self.jt = jt
            self.stack.enter_context(jt.runtime.scope(use_cuda=int(self.device != "cpu")))
            self.stack.enter_context(forbid_backend_fallbacks())
            self.fallback_before = jt.core.backend_fallback_count()
            self.addCleanup(self.check_fallback)

    def check_fallback(self):
        self.jt.sync_all(self.device != "cpu")
        self.assertEqual(self.jt.core.backend_fallback_count(), self.fallback_before)

    def tensor(self, value, grad=False):
        return torch.tensor(value, device=self.device, requires_grad=grad)

    def array(self, value):
        if self.jt is not None:
            value.sync()
            self.assertEqual(value.location(), "cpu" if self.device == "cpu" else "device")
        else:
            self.assertEqual(value.device.type, self.device)
        return value.detach().cpu().numpy()

    def test_index_first_axis_duplicate_gradient_accumulates(self):
        raw = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        indices = np.array([2, 0, 2], dtype=np.int64)
        weights = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
        source = self.tensor(raw, grad=True)
        selected = padding.index_first_axis(source, self.tensor(indices))
        np.testing.assert_array_equal(self.array(selected), raw[indices])
        (selected * self.tensor(weights)).sum().backward()
        expected = np.zeros_like(raw)
        np.add.at(expected, indices, weights)
        self.assertIsNotNone(source.grad)
        np.testing.assert_array_equal(self.array(source.grad), expected)

    def test_pad_forward_and_packed_gradient(self):
        raw = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
        indices = np.array([0, 3, 5], dtype=np.int64)
        source = self.tensor(raw, grad=True)
        output = padding.pad_input(source, self.tensor(indices), 2, 3)
        expected = np.zeros((6, 2, 3), dtype=np.float32)
        expected[indices] = raw
        np.testing.assert_array_equal(self.array(output), expected.reshape(2, 3, 2, 3))
        weights = np.arange(1, 37, dtype=np.float32).reshape(2, 3, 2, 3)
        (output * self.tensor(weights)).sum().backward()
        self.assertIsNotNone(source.grad)
        np.testing.assert_array_equal(self.array(source.grad), weights.reshape(6, 2, 3)[indices])

    def test_unpad_metadata_roundtrip_and_gradient(self):
        raw = np.arange(48, dtype=np.float32).reshape(2, 4, 2, 3)
        used = np.array([[0, 1, 1, 0], [1, 0, 0, 0]], dtype=np.int64)
        unused = np.array([[0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.int64)
        for extra in (None, unused):
            for mask_dtype in (np.int64, np.bool_):
                with self.subTest(unused=extra is not None, dtype=str(mask_dtype)):
                    source = self.tensor(raw, grad=True)
                    packed, indices, offsets, maximum, used_lengths = padding.unpad_input(
                        source, self.tensor(used.astype(mask_dtype)),
                        None if extra is None else self.tensor(extra.astype(mask_dtype)))
                    mask = used if extra is None else used + extra
                    selected = np.flatnonzero(mask.reshape(-1))
                    counts = mask.sum(axis=1, dtype=np.int32)
                    np.testing.assert_array_equal(self.array(packed), raw.reshape(8, 2, 3)[selected])
                    np.testing.assert_array_equal(self.array(indices), selected)
                    np.testing.assert_array_equal(self.array(offsets), np.r_[0, np.cumsum(counts)])
                    np.testing.assert_array_equal(self.array(used_lengths), used.sum(axis=1))
                    self.assertEqual(offsets.dtype, torch.int32)
                    self.assertEqual(used_lengths.dtype, torch.int32)
                    self.assertEqual(maximum, int(counts.max()))
                    restored = padding.pad_input(packed, indices, 2, 4)
                    np.testing.assert_array_equal(self.array(restored), raw * mask[:, :, None, None])
                    weights = np.arange(1, 49, dtype=np.float32).reshape(raw.shape)
                    (restored * self.tensor(weights)).sum().backward()
                    self.assertIsNotNone(source.grad)
                    np.testing.assert_array_equal(self.array(source.grad), weights * mask[:, :, None, None])

    def test_all_padding_retains_zero_gradient_path(self):
        source = self.tensor(np.ones((2, 3, 2), dtype=np.float32), grad=True)
        packed, indices, offsets, maximum, used = padding.unpad_input(
            source, self.tensor(np.zeros((2, 3), dtype=np.int64)))
        self.assertEqual(tuple(packed.shape), (0, 2))
        self.assertEqual(tuple(indices.shape), (0,))
        self.assertEqual(maximum, 0)
        np.testing.assert_array_equal(self.array(offsets), [0, 0, 0])
        np.testing.assert_array_equal(self.array(used), [0, 0])
        restored = padding.pad_input(packed, indices, 2, 3)
        np.testing.assert_array_equal(self.array(restored), np.zeros((2, 3, 2)))
        restored.sum().backward()
        self.assertIsNotNone(source.grad)
        np.testing.assert_array_equal(self.array(source.grad), np.zeros((2, 3, 2)))


if __name__ == "__main__":
    unittest.main()

"""Torch eye(out=) writes to the requested tensor on CPU and CUDA."""
import unittest

import torch


class TestEyeOut(unittest.TestCase):
    def test_out_identity_placement_dtype(self):
        for device in ("cpu", "cuda"):
            for dtype in (torch.float32, torch.float64, torch.int64):
                with self.subTest(device=device, dtype=dtype):
                    out = torch.empty((3, 4), device=device, dtype=dtype)
                    result = torch.eye(3, 4, out=out)
                    self.assertIs(result, out)
                    self.assertEqual(result.device.type, device)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.cpu().tolist(),
                                     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    def test_empty_out_resize(self):
        out = torch.empty((0,), device="cuda", dtype=torch.float64)
        self.assertIs(torch.eye(2, out=out), out)
        self.assertEqual(tuple(out.shape), (2, 2))
        self.assertEqual(out.cpu().tolist(), [[1, 0], [0, 1]])

    def test_explicit_mismatch(self):
        out = torch.empty((2, 2), device="cuda")
        with self.assertRaises(RuntimeError):
            torch.eye(2, out=out, dtype=torch.float64)
        with self.assertRaises(RuntimeError):
            torch.eye(2, out=out, device="cpu")

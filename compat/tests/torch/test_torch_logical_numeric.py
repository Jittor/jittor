import unittest
import torch


class TestLogicalNumeric(unittest.TestCase):
    def test_numeric_truth_reductions(self):
        for device in ("cpu", "cuda"):
            for dtype in (torch.float16, torch.bfloat16, torch.float32,
                          torch.float64, torch.int32, torch.int64, torch.bool):
                with self.subTest(device=device, dtype=dtype):
                    x = torch.tensor([[0, 0, 0], [0, 2, -4], [2, 4, 8]],
                                     dtype=dtype, device=device)
                    for function, method in ((torch.any, x.any), (torch.all, x.all)):
                        expected = [False, True, True] if function is torch.any else [False, False, True]
                        for result in (function(x, dim=1), method(dim=1)):
                            self.assertEqual(result.device.type, device)
                            self.assertEqual(result.dtype, torch.bool)
                            self.assertEqual(result.cpu().tolist(), expected)

    def test_nan_and_cache_occupancy(self):
        for device in ("cpu", "cuda"):
            for dtype in (torch.float32, torch.float64):
                x = torch.zeros((128, 64), dtype=dtype, device=device)
                self.assertEqual(int(x.any(dim=-1).sum()), 0)
                x[0, 0] = float("nan")
                x[127, 63] = -1
                self.assertEqual(int(x.any(dim=-1).sum()), 2)
                self.assertTrue(bool(torch.any(x)))
                self.assertFalse(bool(torch.all(x)))


class TestCudaIntegerTruth(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_nonzero_integer_values_reduce_as_truth_values(self):
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[2, 4, 8], [0, 2, 4]], dtype=dtype, device="cuda")
                for result in (torch.all(x, dim=1), x.all(dim=1)):
                    self.assertEqual(result.device.type, "cuda")
                    self.assertEqual(result.dtype, torch.bool)
                    self.assertEqual(result.cpu().tolist(), [True, False])

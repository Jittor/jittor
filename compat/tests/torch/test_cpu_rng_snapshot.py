"""CPU RNG snapshots preserve position and do not reset CUDA state."""
import unittest
import numpy as np
import torch

class TestCPURNGSnapshot(unittest.TestCase):
    def test_mixed_draws_resume_and_malformed_state_is_atomic(self):
        torch.manual_seed(317)
        torch.rand(13, device="cpu").numpy()
        torch.randn(7, device="cpu").numpy()
        state = torch.get_rng_state()
        self.assertEqual(state.dtype, torch.uint8)
        self.assertEqual(state.device.type, "cpu")
        expected = (torch.rand(9, device="cpu").numpy().copy(),
                    torch.randn(5, device="cpu").numpy().copy())
        torch.rand(23, device="cpu").numpy()
        torch.set_rng_state(state)
        actual = (torch.rand(9, device="cpu").numpy().copy(),
                  torch.randn(5, device="cpu").numpy().copy())
        for x, y in zip(actual, expected):
            np.testing.assert_array_equal(x, y)
        before = torch.get_rng_state().numpy().copy()
        with self.assertRaises(RuntimeError):
            torch.set_rng_state(torch.tensor([0, 1, 2], dtype=torch.uint8))
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)

    def test_restoring_cpu_does_not_reseed_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        torch.manual_seed(29)
        torch.rand(5, device="cuda").cpu().numpy()
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state().cpu().numpy().copy()
        torch.set_rng_state(cpu_state)
        np.testing.assert_array_equal(
            torch.cuda.get_rng_state().cpu().numpy(), cuda_state)

    def test_cpu_int64_random_under_cuda_scope_preserves_cuda_rng(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        torch.manual_seed(123)
        cpu_before = torch.get_rng_state().numpy().copy()
        cuda_before = torch.cuda.get_rng_state().cpu().numpy().copy()
        value = torch.empty((), dtype=torch.int64, device="cpu").random_()
        self.assertEqual(value.device.type, "cpu")
        self.assertEqual(value.dtype, torch.int64)
        self.assertGreaterEqual(int(value.item()), 0)
        self.assertLess(int(value.item()), 1 << 63)
        self.assertFalse(np.array_equal(torch.get_rng_state().numpy(), cpu_before))
        np.testing.assert_array_equal(
            torch.cuda.get_rng_state().cpu().numpy(), cuda_before)
        state = torch.get_rng_state()
        expected = int(torch.empty((), dtype=torch.int64, device="cpu").random_().item())
        torch.set_rng_state(state)
        actual = int(torch.empty((), dtype=torch.int64, device="cpu").random_().item())
        self.assertEqual(actual, expected)

    def test_random_uses_target_device_for_float_and_cuda_integer(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        torch.manual_seed(521)
        cuda_before = torch.cuda.get_rng_state().cpu().numpy().copy()
        cpu_float = torch.empty(8, dtype=torch.float32, device="cpu").random_()
        self.assertEqual(cpu_float.device.type, "cpu")
        np.testing.assert_array_equal(cpu_float.numpy(), np.floor(cpu_float.numpy()))
        np.testing.assert_array_equal(torch.cuda.get_rng_state().cpu().numpy(), cuda_before)
        cpu_before = torch.get_rng_state().numpy().copy()
        cuda_integer = torch.empty(8, dtype=torch.int64, device="cuda").random_()
        self.assertEqual(cuda_integer.device.type, "cuda")
        self.assertEqual(cuda_integer.dtype, torch.int64)
        self.assertTrue(np.any(cuda_integer.cpu().numpy() > 0))
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), cpu_before)
        self.assertFalse(np.array_equal(torch.cuda.get_rng_state().cpu().numpy(), cuda_before))

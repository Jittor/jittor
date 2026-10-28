"""Complete legacy CPU RNG decoding during CUDA checkpoint recovery."""
import unittest
import numpy as np
import torch


class TestLegacyCPURNGState(unittest.TestCase):
    def test_legacy_complete_cpu_state_restores_stream_and_preserves_cuda(self):
        # JTCPU1 checkpoints stored the signed 32-bit seed and the complete
        # default_random_engine stream before the runtime RNG API refactor.
        for seed in (317, -1):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                torch.rand(13, device="cpu").numpy()
                torch.randn(7, device="cpu").numpy()
                state = torch.get_rng_state()
                fields = state.numpy().tobytes().decode("ascii").split()
                self.assertEqual(fields[:3], ["JITTOR_RNG_1", "cpu", "0"])
                legacy = ("JTCPU1 " + str(seed) + " " + " ".join(fields[7:])).encode("ascii")
                saved = torch.tensor(list(legacy), dtype=torch.uint8, device="cpu")
                expected = (torch.rand(9, device="cpu").numpy().copy(),
                            torch.randn(5, device="cpu").numpy().copy())
                cuda = (torch.cuda.get_rng_state().numpy().copy()
                        if torch.cuda.is_available() else None)
                torch.set_rng_state(saved)
                np.testing.assert_array_equal(torch.get_rng_state().numpy(), state.numpy())
                if cuda is not None:
                    np.testing.assert_array_equal(torch.cuda.get_rng_state().numpy(), cuda)
                actual = (torch.rand(9, device="cpu").numpy().copy(),
                          torch.randn(5, device="cpu").numpy().copy())
                for value, reference in zip(actual, expected):
                    np.testing.assert_array_equal(value, reference)

    def test_malformed_legacy_cpu_state_is_atomic(self):
        torch.manual_seed(29)
        before = torch.get_rng_state().numpy().copy()
        cuda = (torch.cuda.get_rng_state().numpy().copy()
                if torch.cuda.is_available() else None)
        for payload in (b"JTCPU1 29", b"JTCPU1 2147483648 1",
                        b"JTCPU1 -2147483649 1", b"JTCPU1 29 1 trailing",
                        b"JITTOR_RNG_1 acl 0 29 0 ACL_COUNTER"):
            with self.subTest(payload=payload):
                with self.assertRaises(RuntimeError):
                    torch.set_rng_state(torch.tensor(list(payload), dtype=torch.uint8, device="cpu"))
                np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)
                if cuda is not None:
                    np.testing.assert_array_equal(torch.cuda.get_rng_state().numpy(), cuda)

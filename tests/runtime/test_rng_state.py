"""Native CPU engine snapshots preserve stream position without global reseed."""

import unittest

import jittor as jt
import numpy as np


class TestNativeCPURNGState(unittest.TestCase):
    def test_full_seed_metadata_and_legacy_getter_boundary(self):
        try:
            jt.set_cpu_seed((1 << 64) - 1)
            self.assertEqual(jt.get_cpu_initial_seed(), (1 << 64) - 1)
            with self.assertRaisesRegex(RuntimeError, "cannot represent"):
                jt.get_seed()
            state = jt.get_cpu_rng_state()
            jt.set_seed(42)
            jt.set_cpu_rng_state(state)
            self.assertEqual(jt.get_cpu_initial_seed(), (1 << 64) - 1)
            with self.assertRaisesRegex(RuntimeError, "cannot represent"):
                jt.get_seed()
            jt.set_seed(-1)
            self.assertEqual(jt.get_seed(), -1)
            self.assertEqual(jt.get_cpu_initial_seed(), (1 << 64) - 1)
        finally:
            jt.set_seed(1729)

    def test_serialized_engine_continues_mixed_random_draws(self):
        with jt.flag_scope(use_cuda=0):
            jt.set_seed(1729)
            jt.rand(7).sync()
            jt.randn(9).sync()
            state = jt.get_cpu_rng_state()
            expected = jt.randn(11, dtype="float64").numpy().copy()
            jt.set_seed(42)
            jt.set_cpu_rng_state(state)
            self.assertEqual(jt.get_seed(), 1729)
            np.testing.assert_array_equal(jt.randn(11, dtype="float64").numpy(), expected)

    def test_invalid_native_state_is_rejected_atomically(self):
        with jt.flag_scope(use_cuda=0):
            jt.set_seed(1729)
            jt.rand(7).sync()
            state = jt.get_cpu_rng_state()
            for invalid in ("", "bad", state + "trailing", state.replace("1729", "999999999999999999999"),
                            "JITTOR_CPU_RNG_V2\n42 42 1 -1\n1\n"):
                with self.subTest(state=invalid):
                    with self.assertRaises(RuntimeError):
                        jt.set_cpu_rng_state(invalid)
                    self.assertEqual(jt.get_cpu_rng_state(), state)
                    self.assertEqual(jt.get_seed(), 1729)


if __name__ == "__main__":
    unittest.main()

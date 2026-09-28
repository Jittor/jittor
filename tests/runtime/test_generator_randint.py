import unittest
from unittest import mock

import numpy as np
import jittor as jt


class TestGeneratorRandint(unittest.TestCase):
    def test_seed_reseeds_and_resets_counter(self):
        generator = jt.Generator("cpu").manual_seed(11)
        jt.randint(0, 10, (7,), generator=generator)
        with mock.patch("jittor._core.generator.secrets.randbits",
                        return_value=(1 << 63) + 17):
            seed = generator.seed()
        self.assertEqual(seed, (1 << 63) + 17)
        self.assertEqual(generator.initial_seed(), seed)
        self.assertEqual(generator.get_state()[-1], 0)

    def test_cpu_state_restore_lazy_reservation_and_global_isolation(self):
        generator = jt.Generator("cpu").manual_seed(777)
        initial_global = jt.get_cpu_rng_state()

        first = jt.randint(-7, 19, (17,), dtype="int64", generator=generator)
        saved = generator.get_state()
        second = jt.randint(0, 1000, (11,), dtype="int32", generator=generator)
        expected_first = first.numpy().copy()
        expected_second = second.numpy().copy()

        self.assertEqual(jt.get_cpu_rng_state(), initial_global)
        generator.set_state(saved)
        replay = jt.randint(0, 1000, (11,), dtype="int32", generator=generator)
        np.testing.assert_array_equal(replay.numpy(), expected_second)
        np.testing.assert_array_equal(first.numpy(), expected_first)
        self.assertEqual(jt.get_cpu_rng_state(), initial_global)

    def test_independent_streams_and_invalid_calls_do_not_advance(self):
        left = jt.Generator().manual_seed(31)
        right = jt.Generator().manual_seed(31)
        baseline = left.get_state()
        with self.assertRaises(ValueError):
            jt.randint(5, 5, (2,), generator=left)
        self.assertEqual(left.get_state(), baseline)
        with self.assertRaises(TypeError):
            jt.randint(0, 5, (2.5,), generator=left)
        self.assertEqual(left.get_state(), baseline)
        with self.assertRaises(TypeError):
            jt.randint(0.5, 5, (2,), generator=left)
        self.assertEqual(left.get_state(), baseline)
        empty = jt.randint(0, 5, (0,), generator=left)
        self.assertEqual(tuple(empty.shape), (0,))
        self.assertEqual(left.get_state(), baseline)
        np.testing.assert_array_equal(
            jt.randint(0, 100, (32,), generator=left).numpy(),
            jt.randint(0, 100, (32,), generator=right).numpy())
        boundary = jt.randint((1 << 31) - 1, 1 << 31, (3,),
                              dtype="int32", generator=left).numpy()
        np.testing.assert_array_equal(boundary,
                                      np.full((3,), (1 << 31) - 1, np.int32))

    def test_wide_int64_range_full_seed_and_malformed_state(self):
        generator = jt.Generator().manual_seed((1 << 64) - 1)
        original = generator.get_state()
        values = jt.randint(-(1 << 63), 1, (257,), dtype="int64",
                            generator=generator).numpy()
        self.assertTrue(np.all(values >= np.iinfo(np.int64).min))
        self.assertTrue(np.all(values < 1))
        advanced = generator.get_state()
        for invalid in (
                None,
                ("JITTOR_GENERATOR_V0", "cpu", None, 0, 0),
                ("JITTOR_GENERATOR_V1", "cuda", 0, 0, 0),
                ("JITTOR_GENERATOR_V1", "cpu", None, 0, -1)):
            with self.assertRaises(ValueError):
                generator.set_state(invalid)
            self.assertEqual(generator.get_state(), advanced)
        generator.set_state(original)
        replay = jt.randint(-(1 << 63), 1, (257,), dtype="int64",
                            generator=generator).numpy()
        np.testing.assert_array_equal(replay, values)

    def test_lazy_reverse_evaluation_uses_reserved_counters(self):
        expected_generator = jt.Generator().manual_seed(123)
        expected_first = jt.randint(0, 1000, (64,), generator=expected_generator).numpy()
        expected_second = jt.randint(0, 1000, (64,), generator=expected_generator).numpy()
        actual_generator = jt.Generator().manual_seed(123)
        actual_first = jt.randint(0, 1000, (64,), generator=actual_generator)
        actual_second = jt.randint(0, 1000, (64,), generator=actual_generator)
        np.testing.assert_array_equal(actual_second.numpy(), expected_second)
        np.testing.assert_array_equal(actual_first.numpy(), expected_first)

    def test_split_draws_preserve_odd_offsets_and_rejection_mapping(self):
        for low, high in ((-101, 211), (-(1 << 63), 1)):
            with self.subTest(low=low, high=high):
                whole_generator = jt.Generator("cpu").manual_seed(2026)
                whole = jt.randint(low, high, (19,), dtype="int64",
                                   generator=whole_generator).numpy()
                split_generator = jt.Generator("cpu").manual_seed(2026)
                chunks = [
                    jt.randint(low, high, (size,), dtype="int64",
                               generator=split_generator).numpy()
                    for size in (1, 7, 11)
                ]
                np.testing.assert_array_equal(np.concatenate(chunks), whole)

    @unittest.skipUnless(jt.has_cuda, "CUDA is unavailable")
    def test_cuda_executes_on_device_and_restores_state(self):
        generator = jt.Generator("cuda:0").manual_seed(99)
        first = jt.randint(-100, 100, (257,), dtype="int64", generator=generator)
        first.sync()
        self.assertEqual(first.location(), "device")
        self.assertEqual(first.device_id, 0)
        state = generator.get_state()
        expected = jt.randint(0, 17, (129,), dtype="int32", generator=generator)
        expected_values = expected.numpy().copy()
        generator.set_state(state)
        replay = jt.randint(0, 17, (129,), dtype="int32", generator=generator)
        replay.sync()
        self.assertEqual(replay.location(), "device")
        self.assertEqual(replay.device_id, 0)
        np.testing.assert_array_equal(replay.numpy(), expected_values)

        cpu_generator = jt.Generator("cpu").manual_seed(99)
        cpu_values = jt.randint(-(1 << 63), 1, (129,), dtype="int64",
                                generator=cpu_generator).numpy()
        cuda_generator = jt.Generator("cuda:0").manual_seed(99)
        cuda_values = jt.randint(-(1 << 63), 1, (129,), dtype="int64",
                                 generator=cuda_generator).numpy()
        np.testing.assert_array_equal(cuda_values, cpu_values)

        cpu_generator = jt.Generator("cpu").manual_seed(99)
        with jt.flag_scope(use_cuda=1):
            cpu_values = jt.randint(0, 17, (9,), generator=cpu_generator)
            cpu_values.sync()
        self.assertEqual(cpu_values.location(), "cpu")


if __name__ == "__main__":
    unittest.main()

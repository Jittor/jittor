import unittest
from unittest import mock

import numpy as np

import jittor as jt


class TestBenchmark(unittest.TestCase):

    def test_materializes_every_nested_output_and_excludes_warmup(self):
        calls = {"left": 0, "right": 0}
        seen_slots = []

        def make_output(value, name):
            def forward(np_module, data):
                calls[name] += 1
                np_module.copyto(data["outputs"][0], data["inputs"][0] + 1)

            return jt.numpy_code(value.shape, value.dtype, [value], forward)

        input_pool = [jt.array(np.full(4, index, dtype="float32")) for index in range(2)]
        original_pool = tuple(input_pool)

        def operation(value):
            # Mutating the caller's list must not change the pool captured by
            # benchmark(). Both results must stay live until synchronization.
            seen_slots.append(value)
            input_pool.append(value)
            return {
                "left": make_output(value, "left"),
                "nested": [make_output(value, "right")],
            }

        clock = iter((1.0, 1.2, 2.0, 2.3, 3.0, 3.4))
        with mock.patch("jittor.benchmarking._timer", side_effect=lambda: next(clock)):
            result = jt.benchmark(operation, input_pool, warmup=2, repeat=3)

        self.assertIsInstance(result, jt.BenchmarkResult)
        for actual, expected in zip(result.samples, (0.2, 0.3, 0.4)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result.median, 0.3)
        self.assertEqual(result.warmup, 2)
        self.assertEqual(result.repeat, 3)
        self.assertEqual(result.input_count, len(original_pool))
        self.assertEqual(calls, {"left": 5, "right": 5})
        self.assertEqual(
            [id(value) for value in seen_slots],
            [id(original_pool[index % 2]) for index in range(5)],
        )

    def test_rejects_measurements_without_materialized_vars(self):
        with self.assertRaisesRegex(TypeError, "at least one jittor.Var"):
            jt.benchmark(lambda value: value + 1, [1, 2], warmup=1, repeat=1)

    def test_requires_a_nonempty_concrete_pool_and_positive_counts(self):
        cases = (
            (((value for value in [1]),), {"warmup": 1, "repeat": 1}),
            (([],), {"warmup": 1, "repeat": 1}),
            (([jt.ones(1)],), {"warmup": 0, "repeat": 1}),
            (([jt.ones(1)],), {"warmup": 1, "repeat": 0}),
        )
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                jt.benchmark(lambda value: value, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()

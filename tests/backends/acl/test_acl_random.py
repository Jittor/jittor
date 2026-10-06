"""Native random/arg-reduce provider registration must execute on real ACL."""

import unittest

import numpy as np
import jittor as jt

from _helpers import capability
from _helpers.child_process import run_child_script
from jittor._runtime.fallback import forbid_backend_fallbacks


@unittest.skipIf(
    not capability.check_accelerator("acl", backend=jt).enabled, "No ACL found")
class TestACLNativeRandom(unittest.TestCase):
    def setUp(self):
        # Only reseed Jittor's engine; preserve NumPy/Python RNGs owned by the
        # test-input fixture, and restore the original native seed on cleanup.
        self.addCleanup(jt.set_seed, jt.get_seed())

    def assert_acl_resident(self, value):
        value.sync()
        self.assertEqual(value.location(), "device")
        self.assertTrue(jt.compiler.has_acl)
        self.assertGreaterEqual(value.device_id, 0)
        # Native tensors may follow runtime placement rather than pin a backend.
        self.assertIn(value.placement_backend, (-1, 2))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_random_distributions_and_dtypes_without_fallback(self):
        jt.set_seed(20260910)
        for dtype in ("float16", "bfloat16", "float32"):
            for distribution in ("uniform", "normal"):
                with self.subTest(dtype=dtype, distribution=distribution):
                    with forbid_backend_fallbacks():
                        # Use the core constructor: jt.random can generate fp32
                        # and cast, which would not exercise every ACL dtype.
                        value = jt.core.ops.random((32768,), dtype, distribution)
                        self.assertEqual(str(value.dtype), dtype)
                        self.assert_acl_resident(value)
                        actual = value.float32().numpy()
                    self.assertTrue(np.isfinite(actual).all())
                    if distribution == "uniform":
                        # Rounding to BF16/FP16 can produce exactly 1.0.
                        self.assertTrue((actual >= 0).all())
                        self.assertTrue((actual <= 1).all())
                        self.assertAlmostEqual(float(actual.mean()), 0.5, delta=0.02)
                        self.assertAlmostEqual(float(actual.var()), 1 / 12, delta=0.01)
                    else:
                        self.assertAlmostEqual(float(actual.mean()), 0.0, delta=0.04)
                        self.assertAlmostEqual(float(actual.var()), 1.0, delta=0.06)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_seed_replays_sequence_and_advances_offset_without_fallback(self):
        for distribution in ("uniform", "normal"):
            with self.subTest(distribution=distribution):
                def sequence(seed):
                    jt.set_seed(seed)
                    values = []
                    for _ in range(2):
                        value = jt.core.ops.random((257,), "float32", distribution)
                        self.assert_acl_resident(value)
                        values.append(value.numpy().copy())
                    return values

                with forbid_backend_fallbacks():
                    first = sequence(20260910)
                    replay = sequence(20260910)
                    changed = sequence(20260911)
                for actual, expected in zip(replay, first):
                    np.testing.assert_array_equal(actual, expected)
                self.assertFalse(np.array_equal(first[0], first[1]))
                self.assertFalse(np.array_equal(first[0], changed[0]))

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_single_draw_multinomial_distribution_and_replay(self):
        probabilities = np.tile(
            np.asarray([0.2, 0.3, 0.5], dtype=np.float32), (4096, 1))
        with forbid_backend_fallbacks():
            jt.set_seed(20260925)
            first = jt.multinomial(jt.array(probabilities), 1)
            self.assert_acl_resident(first)
            actual = first.numpy().copy()
            jt.set_seed(20260925)
            replay = jt.multinomial(jt.array(probabilities), 1).numpy()
            one_hot = jt.multinomial(
                jt.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                         dtype="float32"), 1).numpy()
        self.assertEqual(actual.shape, (4096, 1))
        self.assertEqual(str(first.dtype), "int64")
        np.testing.assert_array_equal(actual, replay)
        np.testing.assert_array_equal(one_hot, [[2], [1]])
        frequencies = np.bincount(actual.ravel(), minlength=3) / 4096
        np.testing.assert_allclose(frequencies, [0.2, 0.3, 0.5], atol=0.03)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_large_vocabulary_multinomial_matches_torch_npu_rng(self):
        # Independent torch_npu 2.7.1.post4 on Ascend 910B3, seed 20260925.
        # The old inverse-CDF ACL path returned index 3837 and advanced the
        # stream differently; native CANN multinomial returns 47764.
        weights = np.zeros((1, 151936), dtype=np.float32)
        weights[0, [1773, 3837, 47764]] = [0.1, 0.4, 0.5]
        expected_after = np.asarray(
            [0.6222606897354126, 0.597446084022522,
             0.9208430051803589, 0.9523041248321533,
             0.022185802459716797, 0.9120728969573975,
             0.5148977041244507, 0.6582885980606079],
            dtype=np.float32)
        with forbid_backend_fallbacks():
            jt.set_seed(20260925)
            sampled = jt.multinomial(jt.array(weights), 1)
            self.assert_acl_resident(sampled)
            actual = sampled.numpy()
            following = jt.rand((8,))
            self.assert_acl_resident(following)
            after = following.numpy()
        self.assertEqual(str(sampled.dtype), "int64")
        np.testing.assert_array_equal(actual, [[47764]])
        np.testing.assert_array_equal(after, expected_after)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_random_float64_declines_acl_with_explicit_fallback_error(self):
        result = run_child_script(
            """
import os
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
assert jt.compiler.has_acl
jt.flags.use_acl = 1
jt.flags.use_cuda = 1
before = jt.core.backend_fallback_count()
with forbid_backend_fallbacks():
    try:
        jt.core.ops.random((16,), "float64", "normal").sync()
    except RuntimeError as error:
        message = str(error).lower()
        assert "fallback" in message, message
        assert "random" in message, message
        assert jt.core.backend_fallback_count() > before
        print("ACL-RANDOM-FLOAT64-REJECTED", flush=True)
        # The deliberately failed graph must not be retried during teardown.
        os._exit(0)
    raise AssertionError("ACL unexpectedly accepted float64 random")
""",
            text=True, timeout=180, crash_isolated=True,
            without_torch_mode=True, name="acl_random_float64_rejection",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ACL-RANDOM-FLOAT64-REJECTED", result.stdout)

    @jt.flag_scope(use_acl=1, use_cuda=1)
    def test_arg_reduce_provider_outputs_and_gradient_without_fallback(self):
        source_np = np.asarray([[1, 5, 3, 5], [-2, -4, 7, 0]], dtype=np.float32)
        for operation, axis, keepdims in (("max", 1, False), ("min", 0, True)):
            with self.subTest(operation=operation, axis=axis):
                reference_indices = (np.argmax if operation == "max" else np.argmin)(
                    source_np, axis=axis)
                indices_with_axis = np.expand_dims(reference_indices, axis)
                reference_values = np.take_along_axis(source_np, indices_with_axis, axis)
                reference_gradient = np.zeros_like(source_np)
                np.put_along_axis(reference_gradient, indices_with_axis, 1, axis)
                if keepdims:
                    reference_indices = indices_with_axis
                else:
                    reference_values = np.squeeze(reference_values, axis)
                with forbid_backend_fallbacks():
                    source = jt.array(source_np)
                    indices, values = jt.arg_reduce(source, operation, axis, keepdims)
                    gradient = jt.grad(values.sum(), source)
                    for tensor in (indices, values, gradient):
                        self.assert_acl_resident(tensor)
                    actual = jt.fetch_sync([indices, values, gradient])
                for value, reference in zip(
                    actual, (reference_indices, reference_values, reference_gradient)
                ):
                    np.testing.assert_array_equal(value, reference)

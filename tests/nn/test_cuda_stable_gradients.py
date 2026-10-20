"""Oracle and reproducibility contracts for CUDA linear/embedding gradients."""

import unittest

import numpy as np

import jittor as jt
from jittor._runtime.dispatch import try_dispatch


@unittest.skipUnless(jt.has_cuda, "requires real CUDA")
class TestCUDAStableGradients(unittest.TestCase):
    def setUp(self):
        self.scope = jt.flag_scope(use_cuda=1, amp_reg=0)
        self.scope.__enter__()

    def tearDown(self):
        jt.sync_all(True)
        self.scope.__exit__(None, None, None)

    def test_bias_cancellation_repeatability_and_oracle(self):
        rng = np.random.RandomState(42)
        half = rng.normal(size=(64, 384)).astype("float32") * 1e-5
        cotangent = np.concatenate([half, -half], axis=0).reshape(2, 64, 384)
        results = []
        for _ in range(12):
            bias = jt.zeros((384,))
            output = try_dispatch("nn.linear_bias", jt.zeros(cotangent.shape), bias)
            self.assertIsNotNone(output)
            gradient = jt.grad((output * jt.array(cotangent)).sum(), bias).numpy()
            results.append(gradient)
        for actual in results[1:]:
            np.testing.assert_array_equal(actual, results[0])
        np.testing.assert_allclose(results[0], cotangent.sum((0, 1), dtype="float64"),
                                   atol=2e-10, rtol=2e-6)

    def test_public_linear_first_and_second_derivatives(self):
        # Small exact values let the same analytic oracle cover CPU and CUDA.
        for device in (0, 1):
            with jt.flag_scope(use_cuda=device):
                x = jt.array(np.arange(24, dtype="float32").reshape(2, 4, 3) / 8)
                weight = jt.array(np.eye(3, dtype="float32"))
                bias = jt.array(np.array([0.125, -0.25, 0.5], dtype="float32"))
                output = jt.nn.linear(x, weight, bias)
                first = jt.grad((output ** 2).sum(), bias, retain_graph=True)
                second, cross = jt.grad(first.sum(), [bias, x])
                expected = 2 * (x.numpy() + bias.numpy()).sum((0, 1))
                np.testing.assert_allclose(first.numpy(), expected, atol=2e-6, rtol=2e-6)
                np.testing.assert_array_equal(second.numpy(), np.full((3,), 16))
                np.testing.assert_array_equal(cross.numpy(), np.full(x.shape, 2))

    def test_embedding_repeated_signed_indices_and_second_derivatives(self):
        for dtype in ("int32", "int64"):
            for raw in ([1, -1, 1, 6, -7, 1], [2] * 1025, list(range(7))):
                indices = np.array(raw, dtype=dtype).reshape(1, -1)
                canonical = indices % 7
                counts = np.bincount(canonical.reshape(-1), minlength=7)[:, None]
                initial = np.arange(35, dtype="float32").reshape(7, 5) / 32
                results = []
                for _ in range(3):
                    weight = jt.array(initial)
                    output = jt.nn.embedding(jt.array(indices), weight)
                    first = jt.grad((output ** 2).sum(), weight, retain_graph=True)
                    second = jt.grad(first.sum(), weight)
                    np.testing.assert_array_equal(output.numpy(), initial[canonical])
                    np.testing.assert_array_equal(first.numpy(), 2 * counts * initial)
                    np.testing.assert_array_equal(second.numpy(), np.broadcast_to(2 * counts, (7, 5)))
                    results.append(first.numpy())
                np.testing.assert_array_equal(results[0], results[-1])

    def test_embedding_stable_sort_order_and_noncontiguous_cotangent(self):
        rng = np.random.RandomState(13)
        indices = rng.randint(0, 17, size=4096).astype("int64")
        # Pin CUB equal-key stability, on which segmented summation relies.
        order, keys = jt.argsort(jt.array(indices))
        order, keys = order.numpy(), keys.numpy()
        for token in range(17):
            np.testing.assert_array_equal(order[keys == token], np.flatnonzero(indices == token))
        cotangent = rng.normal(size=(33, 4096)).astype("float32") * 1e-5
        expected = np.zeros((17, 33), dtype="float64")
        np.add.at(expected, indices, cotangent.astype("float64").T)
        results = []
        for _ in range(6):
            weight = jt.zeros((17, 33))
            output = jt.nn.embedding(jt.array(indices), weight)
            first = jt.grad((output * jt.array(cotangent).transpose()).sum(), weight)
            results.append(first.numpy())
        for actual in results:
            np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-6)
            np.testing.assert_array_equal(actual, results[0])

    def test_embedding_chunk_boundaries_and_adjoint(self):
        from jittor.backends.cuda.kernels.nn.embedding_cuda import _EmbeddingSum
        rng = np.random.RandomState(77)
        # Segment starts/ends straddle 64-position chunk boundaries; short
        # and long segments coexist, including the final incomplete chunk.
        for counts in ((63, 2, 64), (64, 1, 64), (65, 63, 129),
                       (253, 4, 256), (1, 4096, 1)):
            indices = np.repeat(np.arange(len(counts)), counts).astype("int64")
            values = rng.normal(0, 1e-5, (len(indices), 33)).astype("float32")
            expected = np.zeros((7, 33), dtype="float64")
            np.add.at(expected, indices, values.astype("float64"))
            copies = []
            for _ in range(3):
                cotangent = jt.array(values)
                output = _EmbeddingSum.apply(cotangent, jt.array(indices), 7)
                adjoint = jt.grad((output ** 2).sum(), cotangent)
                copies.append(output.numpy())
                np.testing.assert_allclose(copies[-1], expected, atol=2e-8, rtol=2e-6)
                np.testing.assert_allclose(adjoint.numpy(), 2 * expected[indices],
                                           atol=4e-8, rtol=2e-6)
            np.testing.assert_array_equal(copies[0], copies[-1])

    def test_embedding_scalar_layout_empty_and_fallback(self):
        initial = np.arange(35, dtype="float32").reshape(5, 7)
        weight = jt.array(initial).transpose()
        indices = jt.array(np.array([[0, 2, 6], [1, 2, 3]], dtype="int32")).transpose()
        np.testing.assert_array_equal(jt.nn.embedding(indices, weight).numpy(), initial.T[indices.numpy()])
        scalar = jt.array(np.array(-1, dtype="int64"))
        output = jt.nn.embedding(scalar, weight)
        np.testing.assert_array_equal(output.numpy(), initial.T[-1])
        scalar_grad = jt.grad((jt.nn.embedding(scalar, weight) ** 2).sum(), weight).numpy()
        scalar_expected = np.zeros((7, 5), dtype="float32")
        scalar_expected[-1] = 2 * initial.T[-1]
        np.testing.assert_array_equal(scalar_grad, scalar_expected)
        empty = jt.array(np.empty((0,), dtype="int32"))
        self.assertEqual(int(jt.nn.embedding(empty, weight).numel()), 0)
        # Unsupported optional semantics retain their existing implementation.
        padded = jt.nn.embedding(jt.array([1, 2]), weight, padding_idx=1)
        grad = jt.grad(padded.sum(), weight).numpy()
        expected = np.zeros((7, 5), dtype="float32")
        expected[2] = 1
        np.testing.assert_array_equal(grad, expected)

    def test_bias_tile_boundaries_and_noncontiguous_cotangent(self):
        from jittor.backends.cuda.kernels.nn.linear_bias_cuda import _BiasSum
        rng = np.random.RandomState(51)
        for rows, channels in ((1, 1), (255, 31), (256, 32), (257, 33),
                               (257, 255), (257, 256), (257, 257)):
            values = rng.normal(0, 1e-3, (channels, rows)).astype("float32")
            cotangent = jt.array(values).transpose()
            output = _BiasSum.apply(cotangent)
            adjoint = jt.grad((output ** 2).sum(), cotangent)
            expected = values.sum(1, dtype="float64")
            np.testing.assert_allclose(output.numpy(), expected, atol=2e-8, rtol=2e-6)
            np.testing.assert_allclose(adjoint.numpy(), np.broadcast_to(2 * expected, (rows, channels)),
                                       atol=4e-8, rtol=2e-6)

    def test_amp_and_frozen_weight_keep_dtype_and_forward(self):
        initial = np.arange(35, dtype="float32").reshape(7, 5) / 32
        for dtype in ("float16", "bfloat16", "float32"):
            weight = jt.array(initial).cast(dtype)
            indices = jt.array([1, 1, 6])
            output = jt.nn.embedding(indices, weight)
            self.assertEqual(str(output.dtype), dtype)
            np.testing.assert_array_equal(output.numpy().astype("float32"), initial[[1, 1, 6]])
        for register in (jt.amp_flags.prefer16, jt.amp_flags.prefer32):
            with jt.flag_scope(amp_reg=register):
                weight = jt.array(initial)
                output = jt.nn.embedding(jt.array([1, 1, 6]), weight)
                grad = jt.grad(output.sum(), weight)
                expected = np.zeros_like(initial); expected[1] = 2; expected[6] = 1
                np.testing.assert_array_equal(grad.numpy(), expected)
        with jt.no_grad():
            frozen = jt.array(initial).stop_grad()
            np.testing.assert_array_equal(jt.nn.embedding(jt.array([1, 6]), frozen).numpy(), initial[[1, 6]])


if __name__ == "__main__":
    unittest.main()

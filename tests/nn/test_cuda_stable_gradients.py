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
        from jittor._core.dtypes import var_dtype_name
        for device in (0, 1):
            with jt.flag_scope(use_cuda=device):
                for dtype in ("float16", "bfloat16", "float32"):
                    raw_x = np.arange(24, dtype="float32").reshape(2, 4, 3) / 8
                    raw_bias = np.array([0.125, -0.25, 0.5], dtype="float32")
                    x = jt.array(raw_x).cast(dtype)
                    weight = jt.array(np.eye(3, dtype="float32")).cast(dtype)
                    bias = jt.array(raw_bias).cast(dtype)
                    output = jt.nn.linear(x, weight, bias)
                    first = jt.grad((output ** 2).sum(), bias, retain_graph=True)
                    second, cross = jt.grad(first.sum(), [bias, x])
                    expected = 2 * (raw_x + raw_bias).sum((0, 1))
                    self.assertEqual(var_dtype_name(first), dtype)
                    self.assertEqual(var_dtype_name(second), dtype)
                    np.testing.assert_array_equal(first.float32().numpy(), expected)
                    np.testing.assert_array_equal(second.float32().numpy(), np.full((3,), 16))
                    np.testing.assert_array_equal(cross.float32().numpy(), np.full(x.shape, 2))

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

    def test_reduced_bias_boundaries_dtype_and_adjoint(self):
        from jittor._core.dtypes import var_dtype_name
        from jittor.backends.cuda.kernels.nn.linear_bias_cuda import _BiasSum
        # Integer multiples of 1/64 keep the exact sum and its adjoint
        # representable in both reduced dtypes, even across chunk boundaries.
        for dtype in ("float16", "bfloat16"):
            for rows in (1, 255, 256, 257, 513):
                values = ((np.arange(33 * rows).reshape(33, rows) % 7) - 3).astype("float32") / 64
                cotangent = jt.array(values).cast(dtype).transpose()
                output = _BiasSum.apply(cotangent)
                adjoint = jt.grad((output ** 2).sum(), cotangent)
                self.assertEqual(var_dtype_name(output), dtype)
                self.assertEqual(var_dtype_name(adjoint), dtype)
                expected = values.sum(1, dtype="float64")
                np.testing.assert_array_equal(output.float32().numpy(), expected)
                np.testing.assert_array_equal(adjoint.float32().numpy(),
                                              np.broadcast_to(2 * expected, (rows, 33)))

    def test_reduced_linear_bias_cancellation_repeatability_and_master_grad(self):
        from jittor._core.dtypes import var_dtype_name
        from jittor._runtime.dispatch import select_kernel
        rng = np.random.RandomState(99137)
        half = rng.normal(0, 1e-5, (46, 3072)).astype("float32")
        half[::5] *= 64
        values = np.concatenate((half, -half)).reshape(2, 46, 3072)
        for dtype, register in (("float16", jt.amp_flags.prefer16),
                                ("bfloat16", jt.amp_flags.prefer_bfloat16)):
            results = []
            for _ in range(12):
                master = jt.zeros((3072,))
                compute_bias = master.cast(dtype)
                x = jt.zeros(values.shape).cast(dtype)
                self.assertIsNotNone(select_kernel("nn.linear_bias", x, compute_bias))
                output = try_dispatch("nn.linear_bias", x, compute_bias)
                cotangent = jt.array(values).cast(dtype)
                gradient = jt.grad((output * cotangent).sum(), master)
                self.assertEqual(var_dtype_name(output), dtype)
                self.assertEqual(var_dtype_name(gradient), "float32")
                results.append(gradient.numpy())
            for actual in results[1:]:
                np.testing.assert_array_equal(actual, results[0])
            # Rounded opposite rows have an exact zero FP64 oracle. The
            # nonzero residual only comes from FP32 accumulation rounding.
            np.testing.assert_allclose(results[0], np.zeros((3072,)), atol=2e-8, rtol=0)
            with jt.flag_scope(amp_reg=register):
                x = jt.zeros((2, 46, 3))
                weight = jt.zeros((3072, 3))
                master = jt.zeros((3072,))
                output = jt.nn.linear(x, weight, master)
                self.assertEqual(var_dtype_name(output), dtype)
                gradient = jt.grad((output * jt.array(values).cast(dtype)).sum(), master)
                self.assertEqual(var_dtype_name(gradient), "float32")
                np.testing.assert_array_equal(gradient.numpy(), results[0])

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


    def test_embedding_padding_repeat_and_chunked_accumulation(self):
        rng = np.random.RandomState(931)
        indices_np = rng.randint(0, 17, size=(1, 384)).astype("int64")
        values = rng.normal(0, 1e-5, size=(1, 384, 33)).astype("float32")
        values[:, 192:] = -values[:, :192]
        indices_np[:, 192:] = indices_np[:, :192]
        initial = np.arange(17 * 33, dtype="float32").reshape(17, 33) / 32
        initial[0] = 0.375
        expected = np.zeros((17, 33), dtype="float64")
        keep = indices_np.reshape(-1) != 0
        np.add.at(expected, indices_np.reshape(-1)[keep], values.reshape(-1, 33)[keep].astype("float64"))
        copies = []
        from jittor._runtime.dispatch import select_kernel
        for _ in range(8):
            indices, weight = jt.array(indices_np), jt.array(initial)
            self.assertIsNotNone(select_kernel("nn.embedding", indices, weight, 0, None, 2.0, False, False))
            output = jt.nn.embedding(indices, weight, padding_idx=0)
            copies.append(jt.grad((output * jt.array(values)).sum(), weight).numpy())
            np.testing.assert_array_equal(output.numpy(), initial[indices_np])
            np.testing.assert_allclose(copies[-1], expected, atol=2e-8, rtol=2e-6)
        for actual in copies[1:]:
            np.testing.assert_array_equal(actual, copies[0])

    def test_embedding_padding_first_second_grad_and_native_boundaries(self):
        initial = np.arange(35, dtype="float32").reshape(7, 5) / 32 + 0.125
        # Native functional compares the raw padding argument. Preserve this
        # pre-existing boundary instead of silently normalizing it here.
        cases = (("int32", 129, 0), ("int64", 385, 2),
                 ("int64", 7, -1), ("int64", 7, 7), ("int64", 7, -8))
        for index_dtype, count, padding in cases:
            raw = np.resize(np.array([0, 2, 2, -1, 6, 0, 1], dtype=index_dtype), count).reshape(1, -1)
            canonical = raw % 7
            active = raw != padding
            counts = np.bincount(canonical[active], minlength=7)[:, None]
            weight = jt.array(initial)
            output = jt.nn.embedding(jt.array(raw), weight, padding_idx=padding)
            first = jt.grad((output ** 2).sum(), weight, retain_graph=True)
            second = jt.grad(first.sum(), weight)
            np.testing.assert_array_equal(output.numpy(), initial[canonical])
            np.testing.assert_array_equal(first.numpy(), 2 * counts * initial)
            np.testing.assert_array_equal(second.numpy(), np.broadcast_to(2 * counts, initial.shape))
        module = jt.nn.Embedding(7, 5, padding_idx=-1, _weight=jt.array(initial))
        self.assertEqual(module.padding_idx, 6)
        for invalid in (7, -8):
            with self.assertRaisesRegex(AssertionError, "within num_embeddings"):
                jt.nn.Embedding(7, 5, padding_idx=invalid)


if __name__ == "__main__":
    unittest.main()

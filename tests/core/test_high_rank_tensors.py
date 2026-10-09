"""Tensors above ten dimensions, up to the 32-dimension rank limit.

The packed shape vector used to stop at ten entries and 64 value bits, so a
rank-11 tensor (a dense 10-qubit state plus one contraction axis) could not be
built at all. Lifting that exposed every place that had silently assumed it:
JIT keys that wrote per-axis names in hex while templates expand them in
decimal, 16-bit reduce/broadcast masks, ten-slot scratch vectors, and a packed
encoding that corrupted the entry after one ending exactly at bit 64. Each case
below exercises one of those paths against NumPy on every available device.
"""

import unittest

import numpy as np
import jittor as jt

from _helpers.common import JittorTestCase, get_all_device_types, use_cuda_for


RANKS = (11, 12, 17)


class TestHighRankTensors(JittorTestCase):
    def _cases(self, body):
        for device in get_all_device_types():
            for rank in RANKS:
                with self.subTest(device=device, rank=rank):
                    with jt.flag_scope(use_cuda=use_cuda_for(device)):
                        rng = np.random.default_rng(rank)
                        a = rng.standard_normal([2] * rank).astype("float32")
                        b = rng.standard_normal([2] * rank).astype("float32")
                        body(rank, a, b, rng)

    def assert_close(self, actual, expected, tol=1e-4):
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)

    def test_construction_and_elementwise(self):
        def body(rank, a, b, rng):
            ja, jb = jt.array(a), jt.array(b)
            self.assertEqual(tuple(ja.shape), a.shape)
            self.assert_close((ja * jb + ja).numpy(), a * b + a)
            self.assert_close(jt.ones([2] * rank).numpy(), np.ones([2] * rank))
            self.assert_close(jt.where(ja > 0, ja, jb).numpy(), np.where(a > 0, a, b))
        self._cases(body)

    def test_reductions(self):
        # Masks past bit 15 were truncated: a rank-17 full sum returned shape [2].
        def body(rank, a, b, rng):
            ja = jt.array(a)
            self.assertEqual(tuple(ja.sum().shape), ())
            self.assert_close(ja.sum().numpy(), a.sum(), 1e-3)
            axes = (0, rank // 2, rank - 1)
            self.assert_close(ja.sum(dims=list(axes)).numpy(), a.sum(axis=axes))
            self.assert_close(ja.max(dim=rank - 1, keepdims=True).numpy(),
                              a.max(axis=rank - 1, keepdims=True))
            self.assert_close(jt.argmax(ja, dim=rank - 1)[0].numpy(), a.argmax(axis=rank - 1))
            self.assert_close(jt.argsort(ja, dim=rank - 1)[1].numpy(), np.sort(a, axis=rank - 1))
        self._cases(body)

    def test_transpose_reshape_broadcast(self):
        def body(rank, a, b, rng):
            ja = jt.array(a)
            perm = [int(p) for p in rng.permutation(rank)]
            self.assert_close(ja.permute(*perm).numpy(), a.transpose(perm))
            self.assert_close((ja.permute(*perm) + 1).numpy(), a.transpose(perm) + 1)
            self.assert_close(ja.reshape(2 ** (rank // 2), -1).numpy(),
                              a.reshape(2 ** (rank // 2), -1))
            # A zero-stride view whose strides spill out of the packed form.
            self.assert_close((ja + jt.array(a[:1])).numpy(), a + a[:1])
            self.assert_close(jt.concat([ja, ja], dim=rank - 1).numpy(),
                              np.concatenate([a, a], axis=rank - 1))
        self._cases(body)

    def test_indexing(self):
        def body(rank, a, b, rng):
            ja = jt.array(a)
            self.assert_close(ja[1, ..., 0].numpy(), a[1, ..., 0])
            self.assert_close(ja[:, 1:, ..., ::-1].numpy(), a[:, 1:, ..., ::-1])
            index = np.array([1, 0, 1])
            self.assert_close(ja[:, jt.array(index)].numpy(), a[:, index])
            c = ja.clone()
            c[1, ..., 0] = 5.0
            expected = a.copy()
            expected[1, ..., 0] = 5.0
            self.assert_close(c.numpy(), expected)
        self._cases(body)

    def test_matmul_grad_and_complex(self):
        def body(rank, a, b, rng):
            ja, jb = jt.array(a), jt.array(b)
            self.assert_close(jt.matmul(ja, jb).numpy(), np.matmul(a, b), 1e-3)
            perm = [int(p) for p in rng.permutation(rank)]
            loss = (ja.permute(*perm) * jb.permute(*perm)).sum()
            self.assert_close(jt.grad(loss, ja).numpy(), b)
            self.assert_close((ja * (1 + 2j) * jb).numpy(), a * (1 + 2j) * b)
        self._cases(body)

    def test_rank_limit_is_a_clear_error(self):
        self.assertEqual(tuple(jt.ones([1] * 32).shape), (1,) * 32)
        with self.assertRaisesRegex(RuntimeError, "at most 32 dimensions"):
            jt.ones([1] * 33)


class TestShapeVectorSpill(JittorTestCase):
    def test_spilled_shape_vector_round_trips(self):
        values = [0, 1024, 512, 256, 128, 64, 16, 8, 4, 2, 1]
        vector = jt.NanoVector(values)
        self.assertEqual(list(vector), values)
        self.assertEqual(len(vector), 11)
        wide = jt.NanoVector([1 << 40, 1 << 40, -(1 << 40)])
        self.assertEqual(list(wide), [1 << 40, 1 << 40, -(1 << 40)])
        self.assertEqual(wide, jt.NanoVector([1 << 40, 1 << 40, -(1 << 40)]))
        self.assertEqual(hash(wide), hash((1 << 40, 1 << 40, -(1 << 40))))
        appended = jt.NanoVector(list(range(10)))
        appended.append(10)
        self.assertEqual(list(appended), list(range(11)))


if __name__ == "__main__":
    unittest.main()

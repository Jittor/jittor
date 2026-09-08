"""Independent numerical tests for shape helpers exported by ``jittor.misc``."""

import itertools
import unittest

import numpy as np

import jittor as jt
import jittor.misc as misc


class TestMiscShape(unittest.TestCase):
    def assert_close(self, actual, expected):
        np.testing.assert_allclose(
            actual.numpy(), np.asarray(expected), rtol=1e-6, atol=1e-6,
        )

    def test_atleast_variants_values_identity_and_gradients(self):
        self.assertEqual(misc.atleast_1d(), ())
        self.assertEqual(misc.atleast_2d(), ())
        self.assertEqual(misc.atleast_3d(), ())

        vector = jt.array([1.0, 2.0])
        matrix = jt.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertIs(misc.atleast_1d(vector), vector)
        self.assertIs(misc.atleast_2d(matrix), matrix)
        self.assert_close(misc.atleast_2d(vector), [[1.0, 2.0]])
        self.assert_close(misc.atleast_3d(vector), [[[1.0], [2.0]]])
        self.assert_close(
            misc.atleast_3d(matrix),
            [[[1.0], [2.0]], [[3.0], [4.0]]],
        )
        multiple = misc.atleast_2d(vector, [5.0, 6.0])
        self.assertIsInstance(multiple, tuple)
        self.assertEqual(len(multiple), 2)
        self.assert_close(multiple[0], [[1.0, 2.0]])
        self.assert_close(multiple[1], [[5.0, 6.0]])

        grad = jt.grad(misc.atleast_3d(vector).sum(), vector)
        self.assert_close(grad, np.ones(2, dtype=np.float32))

    def test_cartesian_product_values_identity_errors_and_gradients(self):
        a = jt.array([1.0, 2.0])
        b = jt.array([10.0, 20.0, 30.0])
        self.assertIs(misc.cartesian_prod(a), a)
        expected = np.asarray(
            list(itertools.product([1.0, 2.0], [10.0, 20.0, 30.0])),
            dtype=np.float32,
        )
        product = misc.cartesian_prod(a, b)
        self.assert_close(product, expected)
        self.assert_close(
            misc.cartesian_prod([1, 2], [3, 4]),
            [[1, 3], [1, 4], [2, 3], [2, 4]],
        )
        weights = jt.array(np.arange(1, 13, dtype=np.float32).reshape(6, 2))
        grad_a, grad_b = jt.grad((product * weights).sum(), [a, b])
        self.assert_close(grad_a, [9.0, 27.0])
        self.assert_close(grad_b, [10.0, 14.0, 18.0])
        with self.assertRaisesRegex(AssertionError, "only accepts 1-D Vars"):
            misc.cartesian_prod(jt.ones((1, 1)), b)

    def test_block_diag_values_empty_errors_and_gradients(self):
        empty = misc.block_diag()
        self.assertEqual(empty.shape, [0, 0])
        self.assertEqual(str(empty.dtype), "float32")

        matrix = jt.array([[1.0, 2.0], [3.0, 4.0]])
        vector = jt.array([5.0, 6.0, 7.0])
        result = misc.block_diag(matrix, vector, 8.0)
        expected = np.asarray([
            [1, 2, 0, 0, 0, 0],
            [3, 4, 0, 0, 0, 0],
            [0, 0, 5, 6, 7, 0],
            [0, 0, 0, 0, 0, 8],
        ], dtype=np.float32)
        self.assert_close(result, expected)
        weights = jt.array(np.arange(1, 25, dtype=np.float32).reshape(4, 6))
        grad_matrix, grad_vector = jt.grad((result * weights).sum(), [matrix, vector])
        self.assert_close(grad_matrix, [[1.0, 2.0], [7.0, 8.0]])
        self.assert_close(grad_vector, [15.0, 16.0, 17.0])
        with self.assertRaisesRegex(ValueError, "must have at most 2 dimensions"):
            misc.block_diag(jt.ones((1, 1, 1)))

    def test_repeat_chunk_and_expand_values_and_gradients(self):
        repeated_input = jt.array([[1.0], [2.0]])
        repeated = misc.repeat(repeated_input, 2, 3)
        self.assert_close(repeated, np.tile([[1.0], [2.0]], (2, 3)))
        repeat_weights = jt.array(np.arange(1, 13, dtype=np.float32).reshape(4, 3))
        self.assert_close(
            jt.grad((repeated * repeat_weights).sum(), repeated_input),
            [[30.0], [48.0]],
        )

        chunk_input = jt.array(np.arange(10, dtype=np.float32))
        chunks = misc.chunk(chunk_input, 3)
        self.assertEqual([part.shape[0] for part in chunks], [4, 4, 2])
        for part, expected in zip(chunks, np.split(np.arange(10), [4, 8])):
            self.assert_close(part, expected)
        chunk_loss = sum((index + 1) * part.sum() for index, part in enumerate(chunks))
        self.assert_close(
            jt.grad(chunk_loss, chunk_input),
            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
        )

        short_input = jt.array([7.0, 8.0])
        short_chunks = misc.chunk(short_input, 4)
        self.assertEqual(len(short_chunks), 2)
        self.assert_close(short_chunks[0], [7.0])
        self.assert_close(short_chunks[1], [8.0])
        short_loss = 2 * short_chunks[0].sum() + 5 * short_chunks[1].sum()
        self.assert_close(jt.grad(short_loss, short_input), [2.0, 5.0])

        expanded_input = jt.array([[1.0], [2.0], [3.0]])
        expanded = misc.expand(expanded_input, -1, 4)
        self.assert_close(expanded, np.broadcast_to([[1.0], [2.0], [3.0]], (3, 4)))
        expand_weights = jt.array(np.arange(1, 13, dtype=np.float32).reshape(3, 4))
        self.assert_close(
            jt.grad((expanded * expand_weights).sum(), expanded_input),
            [[10.0], [26.0], [42.0]],
        )


if __name__ == "__main__":
    unittest.main()

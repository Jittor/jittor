"""Cross-closure regressions caught while splitting the Torch installer."""

from __future__ import print_function

import unittest

import numpy as np

import jittor as jt


class TestInstallerSplitRegressions(unittest.TestCase):
    def test_integer_getitem_remains_a_basic_index(self):
        value = jt.array([10, 20, 30])[1]
        self.assertEqual(int(value.item()), 20)

    def test_single_argument_where_uses_native_nonzero(self):
        indices = jt.where(jt.array([0, 1, 0, 1]).bool())
        self.assertIsInstance(indices, tuple)
        self.assertEqual(len(indices), 1)
        np.testing.assert_array_equal(
            indices[0].numpy().reshape(-1), np.array([1, 3])
        )

    def test_svd_returns_the_three_torch_values(self):
        u, singular_values, v = jt.svd(
            jt.array([[3.0, 0.0], [0.0, 2.0]])
        )
        self.assertEqual(tuple(u.shape), (2, 2))
        self.assertEqual(tuple(singular_values.shape), (2,))
        self.assertEqual(tuple(v.shape), (2, 2))

    @staticmethod
    def _nested_mask_vmap(mask_fn):
        mapped = jt.vmap(mask_fn, in_dims=(None, None, None, 0))
        mapped = jt.vmap(mapped, in_dims=(None, None, 0, None))
        mapped = jt.vmap(mapped, in_dims=(None, 0, None, None))
        return jt.vmap(mapped, in_dims=(0, None, None, None))

    def test_nested_mask_vmap_matches_loop_and_broadcasts_batch_heads(self):
        from torch._dynamo._trace_wrapped_higher_order_op import (
            TransformGetItemToIndex,
        )

        padding = jt.array([
            [True, True, True, False, False],
            [True, True, True, True, False],
        ])
        call_count = 0

        def mask(batch, _head, query, key):
            nonlocal call_count
            call_count += 1
            return (key <= query) & padding[batch, key]

        mapped = self._nested_mask_vmap(mask)
        args = (jt.arange(2), jt.arange(3), jt.arange(4), jt.arange(5))
        expected = mapped(*args).numpy()
        self.assertEqual(call_count, 2 * 3 * 4 * 5)
        with TransformGetItemToIndex():
            actual = mapped(*args).numpy()
        self.assertEqual(call_count, 2 * 3 * 4 * 5 + 1)

        self.assertEqual(actual.shape, (2, 3, 4, 5))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[:, 0], actual[:, 2])

    def test_transform_getitem_context_restores_depth_after_exception(self):
        from torch._dynamo._trace_wrapped_higher_order_op import (
            TransformGetItemToIndex,
        )

        self.assertEqual(
            getattr(jt, "_transform_getitem_to_index_depth", 0), 0)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with TransformGetItemToIndex():
                self.assertEqual(jt._transform_getitem_to_index_depth, 1)
                raise RuntimeError("expected")
        self.assertEqual(jt._transform_getitem_to_index_depth, 0)


if __name__ == "__main__":
    unittest.main()

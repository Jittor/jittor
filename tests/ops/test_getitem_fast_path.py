from _helpers import capability as _test_capability
import unittest
from unittest import mock

import jittor as jt
import numpy as np
from jittor.ops import indexing


def _cases():
    rng = np.random.RandomState(0)
    x3 = rng.rand(6, 5, 4).astype(np.float32)
    x1 = rng.rand(10).astype(np.float32)
    x2 = rng.rand(4, 3).astype(np.float32)
    rows = np.array([1, 0, 1, 1, 0, 0], bool)

    def constant_index():
        index = jt.full((4,), 2, "int64")
        index._jittor_constant_index_value = 2
        return index

    return [
        ("int", x3, lambda: 0),
        ("negative int", x3, lambda: -1),
        ("slice", x3, lambda: slice(1, 4)),
        ("stepped slice", x3, lambda: slice(None, None, 2)),
        ("negative slice", x1, lambda: slice(-3, None)),
        ("None", x2, lambda: None),
        ("Ellipsis", x2, lambda: Ellipsis),
        ("tuple of ints", x3, lambda: (0, 1)),
        ("slice and int", x3, lambda: (slice(1, 3), 2)),
        ("Ellipsis and slice", x3, lambda: (Ellipsis, slice(None, 2))),
        ("None in tuple", x3, lambda: (0, None, slice(None))),
        ("index Var", x3, lambda: jt.array(np.array([0, 3, 5, 1, 3], np.int64))),
        ("negative index Var", x1, lambda: jt.array(np.array([-1, 0, -2], np.int64))),
        ("2-d int32 index Var", x2, lambda: jt.array(np.array([[0, 2], [1, 3]], np.int32))),
        ("0-d index Var", x3, lambda: jt.array(np.int64(2))),
        ("bool mask", x2, lambda: jt.array(x2 > 0.5)),
        ("bool row mask", x3, lambda: jt.array(rows)),
        ("two index Vars", x3, lambda: (jt.array(np.array([0, 3, 5], np.int64)),
                                        jt.array(np.array([0, 1, 4], np.int64)))),
        ("index Var and slice", x3, lambda: (slice(None), jt.array(np.array([4, 0], np.int64)))),
        ("Ellipsis and index Var", x3, lambda: (Ellipsis, jt.array(np.array([3, 1], np.int64)))),
        ("index Var and int", x3, lambda: (jt.array(np.array([0, 3], np.int64)), 0)),
        # Declined by the fast path: these must keep their full-path meaning.
        ("uint8 mask", x1, lambda: jt.array((x1 > 0.5).astype(np.uint8))),
        ("bool mask in tuple", x3, lambda: (jt.array(rows), 0)),
        ("uint8 mask in tuple", x3, lambda: (jt.array(rows.astype(np.uint8)), slice(None))),
        ("range", x3, lambda: range(2)),
        ("list", x3, lambda: [0, 2]),
        ("numpy int", x3, lambda: np.int64(1)),
        ("numpy int in slice", x3, lambda: slice(np.int64(1), 3)),
        ("constant index", x3, constant_index),
    ]


def _observe(array, make_index):
    base = jt.array(array)
    result = base[make_index()]
    observed = (result.numpy(), tuple(result.shape), str(result.dtype))
    result.assign(result + 100.0)
    return observed + (base.numpy(),)


def _full_path(x, slices):
    return indexing._SLOW


class TestGetitemFastPath(unittest.TestCase):
    """`Var.__getitem__`'s fast path must give exactly what the full path gives.

    `indexing._fast_getitem` answers the common index forms without the
    Torch-compatible rewrites in `_getitem_result`, on the claim that those
    rewrites are no-ops for them. Each case here is evaluated twice -- once as
    `x[index]`, once with the fast path switched off -- and the values, shapes,
    dtypes and view behaviour (whether a write through the result reaches `x`)
    must agree. Forms the fast path declines are included too, so a rewrite added
    to the full path that one of the fast forms should have taken shows up here.
    """

    def _check(self):
        for name, array, make_index in _cases():
            with self.subTest(index=name):
                fast = _observe(array, make_index)
                with mock.patch.object(indexing, "_fast_getitem", _full_path):
                    full = _observe(array, make_index)
                np.testing.assert_array_equal(fast[0], full[0])
                self.assertEqual(fast[1:3], full[1:3])
                np.testing.assert_array_equal(fast[3], full[3], err_msg="view behaviour differs")

    def test_matches_full_path_cpu(self):
        with jt.flag_scope(use_cuda=0):
            self._check()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No cuda found")
    def test_matches_full_path_cuda(self):
        with jt.flag_scope(use_cuda=1):
            self._check()

    def test_basic_index_result_is_a_view(self):
        base = jt.array(np.zeros((3, 4), np.float32))
        base[1].assign(jt.ones((4,)))
        np.testing.assert_array_equal(base.numpy()[1], np.ones(4, np.float32))

    def test_errors_are_unchanged(self):
        with self.assertRaises(IndexError):
            jt.ones((2, 3))[5]


if __name__ == "__main__":
    unittest.main()

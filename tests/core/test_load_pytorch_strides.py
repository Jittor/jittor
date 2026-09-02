"""``jittor_utils.load_pytorch`` must honour a saved tensor's offset and stride.

``torch.save`` writes the whole *storage* once and records, per tensor,
``(storage_offset, size, stride)``.  A tensor that is a view -- a transpose, a
slice, one head of a fused QKV weight -- therefore travels as a non-contiguous
description of a larger buffer.

Both readers in this file used to lose that:

* ``jittor_rebuild`` (the modern zip format) narrowed the storage to
  ``[offset : offset + prod(size)]`` and *then* reindexed the narrowed slice
  with the original strides.  Those strides index the full storage, so every
  element past the narrow window fell outside the source and ``reindex`` filled
  it with 0.  Right shape, no warning, zeros where weights should be.
* ``jittor_rebuild_direct`` (the legacy format) dropped ``storage_offset``
  entirely, so every view of a shared storage started at that storage's front,
  and a 1-D stride was ignored outright.

The reference here is numpy performing the same strided read, which is what
torch itself does.
"""
import unittest

import numpy as np

import jittor as jt
from jittor_utils import load_pytorch


def _reference(base, offset, size, stride):
    """What torch reconstructs: the strided read, done by numpy."""
    if not size:
        return base[offset:offset + 1]
    return np.ascontiguousarray(np.lib.stride_tricks.as_strided(
        base[offset:], shape=size,
        strides=tuple(s * base.itemsize for s in stride)))


# (offset, size, stride) triples, as torch records them for
#   base = arange(24).reshape(4, 6)
_VIEWS = {
    "contiguous":  (0, (4, 6), (6, 1)),
    "transpose":   (0, (6, 4), (1, 6)),
    "column_view": (2, (4, 3), (6, 1)),
    "row_step":    (0, (2, 6), (12, 1)),
    "narrow_3d":   (4, (2, 2, 2), (12, 4, 2)),
    "expanded":    (0, (3, 2), (0, 1)),   # torch's expand() records stride 0
    "vector_step": (1, (4,), (5,)),       # 1-D, and not contiguous
    "scalar":      (13, (), ()),
}


class TestZipFormatRebuildHonoursStride(unittest.TestCase):
    """``jittor_rebuild`` is what the modern zip archive's pickle calls."""

    def setUp(self):
        self.base = np.arange(24, dtype="float32")

    def _rebuild(self, offset, size, stride):
        return load_pytorch.jittor_rebuild(
            jt.array(self.base), offset, size, stride, False, None)

    def test_every_view_reads_its_own_elements(self):
        for name, (offset, size, stride) in _VIEWS.items():
            with self.subTest(view=name):
                got = self._rebuild(offset, size, stride)
                want = _reference(self.base, offset, size, stride)
                np.testing.assert_array_equal(
                    got.numpy().reshape(want.shape), want)

    def test_a_view_that_runs_past_the_storage_is_refused(self):
        # Silently short-reading is how a truncated or mismatched checkpoint
        # became wrong numbers instead of an error.
        with self.assertRaises(ValueError) as caught:
            self._rebuild(0, (5, 6), (6, 1))
        self.assertIn("truncated", str(caught.exception))

    def test_a_negative_stride_is_refused(self):
        with self.assertRaises(ValueError):
            self._rebuild(23, (4,), (-1,))

    def test_a_stride_of_the_wrong_rank_is_refused(self):
        with self.assertRaises(ValueError):
            self._rebuild(0, (4, 6), (6,))


class TestLegacyFormatRebuildHonoursStride(unittest.TestCase):
    """The legacy format defers materialization until the bytes have been read."""

    def setUp(self):
        self.base = np.arange(24, dtype="float32")

    def _materialize(self, offset, size, stride):
        wrapper = load_pytorch.jittor_rebuild_direct(
            self.base, offset, size, stride, False, None)
        return load_pytorch.materialize_wrappers({"w": wrapper})["w"]

    def test_the_wrapper_carries_the_offset(self):
        wrapper = load_pytorch.jittor_rebuild_direct(
            self.base, 2, (4, 3), (6, 1), False, None)
        self.assertEqual(wrapper.storage_offset, 2)

    def test_every_view_reads_its_own_elements(self):
        for name, (offset, size, stride) in _VIEWS.items():
            if not size:
                continue    # the legacy rebuilder is only reached for tensors
            with self.subTest(view=name):
                got = self._materialize(offset, size, stride)
                want = _reference(self.base, offset, size, stride)
                np.testing.assert_array_equal(
                    got.numpy().reshape(want.shape), want)

    def test_a_parameter_keeps_the_tensor_description_it_wraps(self):
        # This used to read a global named `storage` that does not exist, so
        # any legacy checkpoint holding an nn.Parameter raised NameError.
        inner = load_pytorch.jittor_rebuild_direct(
            self.base, 2, (4, 3), (6, 1), None, None)
        wrapper = load_pytorch.jittor_rebuild_var_direct(inner, True, None)
        self.assertIs(wrapper.storage, self.base)
        self.assertEqual(wrapper.storage_offset, 2)
        got = load_pytorch.materialize_wrappers({"w": wrapper})["w"]
        np.testing.assert_array_equal(
            got.numpy(), _reference(self.base, 2, (4, 3), (6, 1)))


class TestExpectedStride(unittest.TestCase):
    def test_it_matches_torch_for_ordinary_shapes(self):
        self.assertEqual(load_pytorch.expected_stride((4, 6)), (6, 1))
        self.assertEqual(load_pytorch.expected_stride((2, 3, 4)), (12, 4, 1))
        self.assertEqual(load_pytorch.expected_stride((5,)), (1,))
        self.assertEqual(load_pytorch.expected_stride(()), ())

    def test_a_zero_length_dimension_does_not_collapse_the_stride(self):
        # torch uses max(size, 1) here, so an empty dimension keeps the
        # stride its neighbours would otherwise lose.
        self.assertEqual(load_pytorch.expected_stride((3, 0, 4)), (4, 4, 1))


if __name__ == "__main__":
    unittest.main()

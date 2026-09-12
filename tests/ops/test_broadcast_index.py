# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""An index Var that is a strided view must be read through its strides.

A broadcast is a *storage descriptor*: ``BroadcastToOp::infer_shape`` gives the
expanded axes a zero stride and shares the producer's allocation, so a
``(4, 5)`` int64 index Var built by expanding a five-element row is backed by
**forty bytes**, not one hundred and sixty.

This file used to build that view as ``jt.zeros(shape, dtype)``, which was
``array(0).broadcast(shape)`` and therefore backed by *eight* bytes. That is no
longer so, and the reason is not a change of mind about expands: a one-element
source now keeps the computed form, because describing it as a view made
``jt.zeros`` a tensor whose buffer was not its shape, and anything writing
through its pointer -- a cuSPARSE dense output, say -- silently lost the write
(see ``tests/core/test_constant_tensor_storage.py``). So the premise moved to a
multi-element source, which is an expand there is a reason to describe as a
view: the overshoot of a dense walk is three quarters of the index rather than
nineteen twentieths, and everything below still distinguishes the defect.

The ``getitem``/``setitem`` kernels read index Vars as if they were dense --
``vp[i0*oshape1 + i1]``, with the strides derived from the *output* shape --
so such an index made the kernel walk 20 elements off the end of a one-element
buffer and use whatever it found as an index (KI-OPS-009). The guard that was
supposed to prevent this, ``adapt_index_storage``, existed but was never
emitted into the generated ``make_getitem``/``make_setitem``.

Why the sizes matter
--------------------
The overshoot for a ``(4, 5)`` index is 152 bytes. Whether that is garbage or
zeroes is a property of the heap, not of the code: the same defect returned the
right answer on one build and an out-of-bounds index on another, which is how
KI-OPS-009 was first recorded as build-specific. It is not. Larger index shapes
overshoot further and fail on any build, so they are in this file too.

``test_a_broadcast_index_reads_its_own_element`` is the case that does not
depend on the heap at all: it broadcasts a one-element *view* of a buffer whose
next 19 elements are a known pattern, so a dense walk returns that pattern and
nothing else can. Before the fix it returned ``[0, 1, 2, 3, 0, 1, ...]`` where
every logical index is zero.
"""

import unittest

import numpy as np

from _helpers import capability as _test_capability

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: Index shapes whose second dimension decides how far past the one-element
#: buffer a dense walk runs: 152 bytes, 2040 bytes, 16376 bytes.
WIDTHS = (5, 64, 512)


class TestBroadcastIndexCpu(unittest.TestCase):

    device_flag = 0

    def _source(self, width):
        """``a[r, c] == 100*r + c``, so a value names the row it came from."""
        return (100 * np.arange(4)[:, None] + np.arange(width)[None, :]).astype("float32")

    def _zero_index(self, width, dtype="int64"):
        """A ``(4, width)`` all-zero index that really is a strided view.

        Expanding a *row* of zeros, not a single zero: the one-element expand
        keeps the computed form now, so building the index the way this file
        used to would hand every case below a dense buffer and test nothing.
        """
        return jt.array(np.zeros(width, dtype=dtype)).broadcast((4, width))

    def test_a_pure_broadcast_is_a_strided_view(self):
        """The premise. If this stops holding, the rest of the file stops testing.

        ``broadcast`` is what makes an index Var non-contiguous, and the whole
        defect is that a non-contiguous index Var reached the kernel. A build
        where ``broadcast`` materialised instead would pass every case below
        without exercising anything.
        """
        with jt.flag_scope(use_cuda=self.device_flag):
            idx = self._zero_index(5)
            self.assertFalse(idx._storage_is_contiguous(),
                             "broadcast no longer produces a strided view; "
                             "this file's premise is gone")
            # the expanded axis carries the zero stride; the kept axis is dense
            self.assertEqual(list(idx._storage_strides()), [0, 1])

    def test_a_one_element_expand_is_not_a_view(self):
        """The other half of the premise, so the move above cannot be undone
        silently: `jt.zeros` owns a buffer of its own shape, which is why this
        file may no longer build its index out of one."""
        with jt.flag_scope(use_cuda=self.device_flag):
            dense = jt.zeros((4, 5), "int64")
            self.assertTrue(dense._storage_is_contiguous())
            self.assertEqual(list(dense._storage_strides() or [5, 1]), [5, 1])

    def test_gather_through_a_broadcast_index(self):
        for width in WIDTHS:
            for dtype in ("int64", "int32"):
                with self.subTest(width=width, dtype=dtype):
                    with jt.flag_scope(use_cuda=self.device_flag):
                        a_np = self._source(width)
                        got = jt.gather(jt.array(a_np), 0,
                                        self._zero_index(width, dtype)).numpy()
                    np.testing.assert_array_equal(
                        got, np.broadcast_to(a_np[0], (4, width)),
                        "gather read a row other than 0 for an all-zero "
                        "broadcast index (KI-OPS-009)")

    def test_a_broadcast_index_reads_its_own_element(self):
        """Heap-independent: the wrong answer is a *known* pattern, not garbage.

        ``base[0:5]`` is a five-element view of a 20-element buffer holding
        ``[0,1,2,3,0,1,...]``. Expanding it to ``(4, 5)`` makes every row equal
        to ``[0,1,2,3,0]``, so every row of the gather must be that. A kernel
        walking the index densely instead reads ``base[r*5 + c]`` and returns a
        *different* row pattern for each ``r`` -- no heap state can make the two
        agree.
        """
        with jt.flag_scope(use_cuda=self.device_flag):
            a_np = self._source(5)
            base = jt.array((np.arange(20) % 4).astype("int64"))
            idx = base[0:5].broadcast((4, 5))
            self.assertFalse(idx._storage_is_contiguous())
            rows = (jt.gather(jt.array(a_np), 0, idx).numpy() // 100).astype(int)
        np.testing.assert_array_equal(
            rows, np.tile((np.arange(5) % 4).astype(int), (4, 1)),
            "the index kernel walked the index Var's neighbours in memory "
            "instead of its own five elements (KI-OPS-009)")

    def test_setitem_through_a_broadcast_index(self):
        """The write side, which was an out-of-bounds *write* before the check."""
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.zeros((4, 5), "float32")
            columns = jt.array(np.tile(np.arange(5), (4, 1)))
            x[self._zero_index(5), columns] = jt.ones((4, 5), "float32")
            got = x.numpy()
        expect = np.zeros((4, 5), "float32")
        expect[0, :] = 1.0
        np.testing.assert_array_equal(
            got, expect, "setitem wrote somewhere other than row 0 for an "
                         "all-zero broadcast index (KI-OPS-009)")

    def test_scatter_add_through_a_broadcast_index(self):
        """``index_add`` builds its index by broadcasting, so this is the
        spelling real code reaches the defect through."""
        with jt.flag_scope(use_cuda=self.device_flag):
            got = jt.zeros((4, 5), "float32").scatter_add(
                0, self._zero_index(5, "int32"), jt.ones((4, 5), "float32")).numpy()
        expect = np.zeros((4, 5), "float32")
        expect[0, :] = 4.0
        np.testing.assert_array_equal(
            got, expect, "scatter_add accumulated into rows other than 0 "
                         "(KI-OPS-009)")

    def test_a_dense_index_is_unchanged(self):
        """The fix must not move the path that was already right."""
        with jt.flag_scope(use_cuda=self.device_flag):
            a_np = self._source(5)
            idx = np.array([[0, 1, 2, 3, 0], [3, 2, 1, 0, 3],
                            [1, 1, 1, 1, 1], [2, 0, 2, 0, 2]])
            got = jt.gather(jt.array(a_np), 0, jt.array(idx)).numpy()
        np.testing.assert_array_equal(got, np.take_along_axis(a_np, idx, axis=0))


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestBroadcastIndexCuda(TestBroadcastIndexCpu):
    """The same body on the device.

    The kernels share one template, so a fix applied to the index expression
    alone would show up here; a fix applied before the op is built has to be
    shown not to depend on the device it later runs on.
    """

    device_flag = 1


if __name__ == "__main__":
    unittest.main()

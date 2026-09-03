# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Everything that returns a position or a count returns int64.

Torch types every index this way, and the reason is not cosmetic: an index is
an offset into a buffer, and int32 stops being able to name one at 2**31
elements -- a 8 GiB float32 tensor. Jittor returned int32 from ``where`` /
``nonzero`` / ``randperm`` / ``topk`` and from the pooling ``return_indices``
kernels, so those indices were both the wrong dtype for a torch consumer and
unable to address a large tensor.

The expected dtypes below were read off torch 2.12.1 in a separate interpreter
(the ``torch`` importable next to Jittor is Jittor's own shim, so it cannot be
an oracle -- see the ``jittor-op-parity-oracle`` skill). They are frozen here as
constants so the suite does not need torch installed:

    nonzero int64, where(cond) int64, randperm int64, topk indices int64
    (empty and non-empty), max_pool{2,3}d indices int64,
    adaptive_max_pool{2,3}d indices int64, searchsorted int64 (already was)

``jt.argsort``, ``jt.argmax``/``jt.argmin`` and ``jt.arange`` still return
int32 where torch returns int64. Same defect, but they are C++ op defaults with
CUB paths underneath and, for arange, a blast radius across the whole library;
they are out of scope for this task and recorded in its commit message.
"""

import unittest

import numpy as np
import pytest

import jittor as jt


class _IndexDtype:
    """Shared body; the CPU and CUDA subclasses pick ``use_cuda``."""

    use_cuda = 0

    def _flag(self):
        return jt.flag_scope(use_cuda=self.use_cuda)

    def test_where_returns_int64(self):
        cond = jt.array(np.array([[0, 1, 0], [1, 0, 2]], dtype="int32"))
        with self._flag():
            outs = jt.where(cond)
            dtypes = [str(v.dtype) for v in outs]
            values = [v.numpy() for v in outs]
        self.assertEqual(dtypes, ["int64", "int64"])
        np.testing.assert_array_equal(values[0], [0, 1, 1])
        np.testing.assert_array_equal(values[1], [1, 0, 2])

    def test_where_is_int64_past_the_cub_threshold(self):
        """>4096 elements and ndim>1 take the CUB kernel, a separate output."""
        raw = np.zeros((8, 1024), dtype="int32")
        raw[3, 700] = 1
        raw[7, 1023] = 1
        cond = jt.array(raw)
        with self._flag():
            rows, cols = jt.where(cond)
            got = (str(rows.dtype), str(cols.dtype),
                   rows.numpy(), cols.numpy())
        self.assertEqual(got[:2], ("int64", "int64"))
        np.testing.assert_array_equal(got[2], [3, 7])
        np.testing.assert_array_equal(got[3], [700, 1023])

    def test_nonzero_returns_int64(self):
        x = jt.array(np.array([[0, 1, 0], [1, 0, 2]], dtype="int32"))
        with self._flag():
            idx = x.nonzero()
            dtype, value = str(idx.dtype), idx.numpy()
        self.assertEqual(dtype, "int64")
        np.testing.assert_array_equal(value, [[0, 1], [1, 0], [1, 2]])

    def test_nonzero_1d_returns_int64(self):
        x = jt.array(np.array([0, 5, 0, 7], dtype="int32"))
        with self._flag():
            idx = x.nonzero()
            dtype, value = str(idx.dtype), idx.numpy()
        self.assertEqual(dtype, "int64")
        np.testing.assert_array_equal(value.reshape(-1), [1, 3])

    def test_randperm_returns_int64(self):
        with self._flag():
            perm = jt.randperm(16)
            dtype, value = str(perm.dtype), perm.numpy()
        self.assertEqual(dtype, "int64")
        np.testing.assert_array_equal(np.sort(value), np.arange(16))

    def test_randperm_still_honours_an_explicit_dtype(self):
        with self._flag():
            self.assertEqual(str(jt.randperm(4, dtype="int32").dtype), "int32")

    def test_topk_indices_are_int64_empty_and_not(self):
        with self._flag():
            values, indices = jt.topk(jt.float32([3.0, 1.0, 2.0]), 2)
            filled = (str(indices.dtype), values.numpy(), indices.numpy())
            empty_values, empty_indices = jt.topk(jt.float32([]), 1)
            empty = (str(empty_indices.dtype), str(empty_values.dtype))
        # Both branches, because they are two different code paths: the empty
        # one builds its own array, the filled one carries argsort's output.
        self.assertEqual(filled[0], "int64")
        np.testing.assert_allclose(filled[1], [3.0, 2.0])
        np.testing.assert_array_equal(filled[2], [0, 2])
        self.assertEqual(empty, ("int64", "float32"))

    def test_max_pool_indices_are_int64(self):
        # 2D and 3D only: jt.nn.MaxPool1d accepts return_indices and returns a
        # bare Var anyway (a silently-ignored argument, task 5.19's class, not
        # this one), so there is no 1D index to type.
        x2 = jt.float32(np.arange(16).reshape(1, 1, 4, 4))
        x3 = jt.float32(np.arange(64).reshape(1, 1, 4, 4, 4))
        with self._flag():
            _, i2 = jt.nn.MaxPool2d(2, stride=2, return_indices=True)(x2)
            _, i3 = jt.nn.MaxPool3d(2, stride=2, return_indices=True)(x3)
            got = [(str(v.dtype), v.numpy()) for v in (i2, i3)]
        self.assertEqual([g[0] for g in got], ["int64"] * 2)
        # torch 2.12.1 gives the same encoding for the 2D case
        np.testing.assert_array_equal(got[0][1].reshape(-1), [5, 7, 13, 15])

    def test_adaptive_max_pool_indices_are_int64(self):
        x2 = jt.float32(np.arange(16).reshape(1, 1, 4, 4))
        x3 = jt.float32(np.arange(64).reshape(1, 1, 4, 4, 4))
        with self._flag():
            _, i2 = jt.nn.AdaptiveMaxPool2d(2, return_indices=True)(x2)
            _, i3 = jt.nn.AdaptiveMaxPool3d(2, return_indices=True)(x3)
            got = [str(i2.dtype), str(i3.dtype)]
        self.assertEqual(got, ["int64", "int64"])

    def test_max_unpool_round_trip_still_works(self):
        """The indices feed reindex expressions; int64 must not break them."""
        raw = np.arange(1, 26, dtype="float32").reshape(1, 1, 5, 5)
        raw[0, 0, 4, :] = 0
        raw[0, 0, :, 4] = 0
        with self._flag():
            x = jt.array(raw)
            pooled, indices = jt.nn.MaxPool2d(2, stride=2,
                                              return_indices=True)(x)
            restored = jt.nn.MaxUnpool2d(2, stride=2)(
                pooled, indices, output_size=x.shape).numpy()
        expected = np.zeros((1, 1, 5, 5), dtype="float32")
        for r, c in ((1, 1), (1, 3), (3, 1), (3, 3)):
            expected[0, 0, r, c] = raw[0, 0, r, c]
        np.testing.assert_array_equal(restored, expected)

    def test_an_int64_index_survives_arithmetic_that_overflows_int32(self):
        """Why the dtype is the bug and not a label.

        Jittor promotes by byte width, so ``index * scalar`` stays in the
        index's own dtype. An int32 index therefore wrapped as soon as the
        arithmetic left 2**31 -- and a flat offset into a big tensor is exactly
        that arithmetic. With four elements and a stride of 10**9 the int32
        answer was -1294967296.
        """
        mask = jt.array(np.array([1, 1, 1, 1], dtype="int32"))
        with self._flag():
            idx = mask.nonzero().reshape((-1,))
            offsets = (idx * 1000000000).numpy()
        np.testing.assert_array_equal(
            offsets, [0, 1000000000, 2000000000, 3000000000])


class TestIndexDtypeCPU(_IndexDtype, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestIndexDtypeCUDA(_IndexDtype, unittest.TestCase):
    use_cuda = 1


class TestRepeatInterleaveCounts(unittest.TestCase):
    """``repeat_interleave``'s CUDA fast path used to refuse 2**31 outputs.

    The refusal was honest -- ``repeats`` was cast to int32, the prefix sum ran
    in int32 and the kernel's ``out_row`` was an ``int`` -- so the assertion was
    holding back real overflow rather than being over-cautious. Removing it
    means the path had to be made 64-bit first, which is what the opt-in probe
    below checks.
    """

    def test_fast_path_agrees_with_the_generic_path(self):
        rng = np.random.default_rng(20260903)
        x = rng.standard_normal((7, 3)).astype("float32")
        repeats = np.array([0, 2, 1, 3, 0, 1, 4], dtype="int64")
        expected = np.repeat(x, repeats, axis=0)
        for use_cuda in ((0, 1) if jt.has_cuda else (0,)):
            with self.subTest(use_cuda=use_cuda):
                with jt.flag_scope(use_cuda=use_cuda):
                    got = jt.repeat_interleave(
                        jt.array(x), jt.array(repeats, dtype="int64"),
                        dim=0, output_size=int(repeats.sum())).numpy()
                np.testing.assert_allclose(got, expected)

    @pytest.mark.manual
    @unittest.skipIf(not jt.has_cuda, "No CUDA found")
    def test_more_than_two_billion_output_rows(self):
        """Opt-in probe: it allocates 2 GiB on the device.

        Run it with ``JITTOR_TEST_MANUAL=1`` when touching this kernel.
        ``output_size`` above 2**31 used to raise AssertionError; removing that
        assertion without the int64 rewrite would return the wrong rows past
        element 2**31 instead, because ``int out_row = linear / inner`` went
        negative and the binary search then always chose row 0.
        """
        total = (1 << 31) + 5
        n = 4
        repeats = np.full(n, total // n, dtype="int64")
        repeats[-1] += total - int(repeats.sum())
        x = np.arange(1, n + 1, dtype="int8")
        with jt.flag_scope(use_cuda=1):
            out = jt.repeat_interleave(
                jt.array(x), jt.array(repeats, dtype="int64"),
                dim=0, output_size=total)
            head = out[:3].numpy()
            tail = out[total - 3:].numpy()
        np.testing.assert_array_equal(head, [1, 1, 1])
        np.testing.assert_array_equal(tail, [n, n, n])


if __name__ == "__main__":
    unittest.main()

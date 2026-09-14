# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``gather``/``scatter``/``index_select`` have to check the index they are given.

``gather`` builds a reindex expression from ``index.shape`` and one ``i{k}`` per
remaining axis. When ``index`` has a rank the input does not, that expression is
still *well formed* -- it just describes a different gather from the one the
caller asked for -- so nothing raised:

* ``jt.ones((3, 4)).gather(0, jt.array([0, 1, 2]))`` returned a ``[3, 4]`` var.
  torch rejects it: ``Index tensor must have the same number of dimensions as
  input tensor``.
* ``jt.ones((3, 4)).gather(0, jt.zeros((3, 6)))`` returned a ``[3, 6]`` var, two
  columns of which the input does not have.
* ``jt.ones((3, 4)).index_select(0, jt.zeros((2, 2)))`` returned ``[2, 2, 4]``,
  folding the index's own rank into the output. torch requires a vector.

A wrong answer with no error is the expensive kind: it is found, if at all, far
from where it was made. The three cases above are the ones a caller can hit with
public arguments, so they are checked at the frontend, where the argument still
has a name.

The dim argument was separately answered by ``indexes[dim] = index`` -- a list
assignment, whose ``IndexError: list assignment index out of range`` names
neither the operation nor the rank nor the bound.

Reference behaviour was read from a binary PyTorch 2.12 build in a separate
process; the values below are plain numpy.
"""

import unittest

import numpy as np

import jittor as jt


class TestGatherIndexArgument(unittest.TestCase):

    def test_an_index_of_the_wrong_rank_is_rejected(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.ones((3, 4)).gather(0, jt.array([0, 1, 2]))
        text = str(caught.exception)
        self.assertIn("gather", text)
        self.assertIn("1-D", text)
        self.assertIn("[3]", text)
        self.assertIn("2-D", text)
        self.assertIn("[3, 4]", text)

    def test_an_index_wider_than_the_input_is_rejected(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.ones((3, 4)).gather(0, jt.zeros((3, 6), dtype="int32"))
        text = str(caught.exception)
        self.assertIn("gather", text)
        self.assertIn("[3, 6]", text)
        self.assertIn("[3, 4]", text)
        self.assertIn("dim 1", text)

    def test_the_gathered_dim_itself_may_be_wider_than_the_input(self):
        # Only the *other* dims are bounded: gathering 7 rows out of 3 with
        # repeats is exactly what gather is for.
        raw = np.array([[0, 1], [2, 0], [1, 1], [0, 2],
                        [2, 2], [1, 0], [0, 0]], dtype="int32")
        source = np.arange(6, dtype="float32").reshape(3, 2)
        out = jt.array(source).gather(0, jt.array(raw))
        self.assertEqual(list(out.shape), [7, 2])
        np.testing.assert_allclose(
            out.data, np.take_along_axis(source, raw.astype("int64"), 0))

    def test_a_dim_out_of_range_names_the_op_and_the_rank(self):
        with self.assertRaises(IndexError) as caught:
            jt.ones((3, 4)).gather(5, jt.zeros((3, 4), dtype="int32"))
        text = str(caught.exception)
        self.assertIn("gather", text)
        self.assertIn("5", text)
        self.assertIn("[-2, 1]", text)
        self.assertNotIn("list assignment index out of range", text)

    def test_a_float_index_is_a_type_error(self):
        with self.assertRaises(TypeError) as caught:
            jt.ones((3, 4)).gather(0, jt.zeros((3, 4)))
        self.assertIn("integer dtype", str(caught.exception))

    def test_gather_still_gathers(self):
        source = np.array([[1, 2], [3, 4]], dtype="float32")
        x = jt.array(source)
        np.testing.assert_allclose(
            x.gather(1, jt.array([[0, 0], [1, 0]])).data, [[1, 1], [4, 3]])
        np.testing.assert_allclose(
            x.gather(0, jt.array([[0, 0], [1, 0]])).data, [[1, 2], [3, 2]])
        np.testing.assert_allclose(
            x.gather(-1, jt.array([[0, 0], [1, 0]])).data, [[1, 1], [4, 3]])


class TestScatterIndexArgument(unittest.TestCase):

    def test_an_index_of_the_wrong_rank_is_rejected(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.zeros((3, 4)).scatter(0, jt.array([0, 1, 2]), jt.ones(3))
        text = str(caught.exception)
        self.assertIn("scatter", text)
        self.assertIn("1-D", text)
        self.assertIn("2-D", text)

    def test_a_dim_out_of_range_names_the_op_and_the_rank(self):
        with self.assertRaises(IndexError) as caught:
            jt.zeros((3, 4)).scatter(5, jt.zeros((3, 4), dtype="int32"),
                                     jt.ones((3, 4)))
        self.assertIn("scatter", str(caught.exception))

    def test_scatter_still_scatters(self):
        src = jt.arange(1, 11).reshape((2, 5)).float32()
        index = jt.array([[0, 1, 2, 0]])
        out = jt.zeros((3, 5)).scatter_(0, index, src)
        np.testing.assert_allclose(out.data,
                                   [[1, 0, 0, 4, 0],
                                    [0, 2, 0, 0, 0],
                                    [0, 0, 3, 0, 0]])

    def test_index_add_still_accumulates(self):
        out = jt.ones((5, 3)).index_add(
            0, jt.array([0, 4, 2]),
            jt.array(np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9]], "float32")))
        np.testing.assert_allclose(out.data,
                                   [[2, 3, 4], [1, 1, 1], [8, 9, 10],
                                    [1, 1, 1], [5, 6, 7]])


class TestIndexSelectIndexArgument(unittest.TestCase):

    def test_a_non_vector_index_is_rejected(self):
        with self.assertRaises(IndexError) as caught:
            jt.ones((3, 4)).index_select(0, jt.zeros((2, 2), dtype="int32"))
        text = str(caught.exception)
        self.assertIn("index_select", text)
        self.assertIn("2-D", text)
        self.assertIn("[2, 2]", text)

    def test_index_select_still_selects(self):
        source = np.arange(12, dtype="float32").reshape(3, 4)
        x = jt.array(source)
        np.testing.assert_allclose(
            jt.index_select(x, 0, jt.array([2, 1])).data, source[[2, 1]])
        np.testing.assert_allclose(
            jt.index_select(x, 1, jt.array([2, 1])).data, source[:, [2, 1]])


if __name__ == "__main__":
    unittest.main()

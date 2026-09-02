# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``scatter`` / ``scatter_`` in-place contract.

``Var.scatter`` (no trailing underscore) used to be in-place: it forwarded
straight to ``setitem``, which mutates its target, so ``y = x.scatter(...)``
silently rewrote ``x``.  torch's ``Tensor.scatter`` is out-of-place and
``Tensor.scatter_`` is the in-place one; the repo itself worked around the
mismatch by cloning by hand inside ``scatter_add``.

Reference values are plain numpy; the semantics were checked once against a
binary PyTorch 2.12 build in a separate process.
"""

import unittest

import numpy as np

import jittor as jt


def _np_scatter(x, dim, index, src, reduce="void"):
    out = x.copy()
    for position in np.ndindex(*index.shape):
        target = list(position)
        target[dim] = int(index[position])
        value = src[position] if np.ndim(src) else src
        if reduce == "void":
            out[tuple(target)] = value
        elif reduce == "add":
            out[tuple(target)] += value
        elif reduce == "multiply":
            out[tuple(target)] *= value
        else:
            raise ValueError(reduce)
    return out


class TestScatterIsOutOfPlace(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((3, 5), dtype="float32")
        self.src = np.arange(1, 11, dtype="float32").reshape(2, 5)
        self.index = np.array([[0, 1, 2, 0, 0]], dtype="int32")

    def test_scatter_leaves_the_input_unchanged(self):
        x = jt.array(self.base)
        before = x.numpy().copy()
        out = x.scatter(0, jt.array(self.index), jt.array(self.src))
        # The write only happens when the lazy graph runs, so `out` has to be
        # materialized BEFORE `x` is inspected -- reading `x` first passes even
        # against the in-place implementation.
        out.sync()
        np.testing.assert_array_equal(x.numpy(), before)
        self.assertGreater(float(np.abs(out.numpy() - before).max()), 0)

    def test_scatter_value_matches_numpy(self):
        out = jt.array(self.base).scatter(
            0, jt.array(self.index), jt.array(self.src))
        expected = _np_scatter(self.base, 0, self.index, self.src[:1])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_scatter_underscore_writes_in_place_and_returns_self(self):
        x = jt.array(self.base)
        out = x.scatter_(0, jt.array(self.index), jt.array(self.src))
        expected = _np_scatter(self.base, 0, self.index, self.src[:1])
        np.testing.assert_allclose(x.numpy(), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6, atol=1e-6)

    def test_reduce_add_is_out_of_place(self):
        base = np.full((2, 4), 2.0, dtype="float32")
        index = np.array([[2], [3]], dtype="int32")
        src = np.array([[1.5], [1.5]], dtype="float32")
        x = jt.array(base)
        out = x.scatter(1, jt.array(index), jt.array(src), reduce="add")
        out.sync()
        np.testing.assert_array_equal(x.numpy(), base)
        np.testing.assert_allclose(
            out.numpy(), _np_scatter(base, 1, index, src, "add"),
            rtol=1e-6, atol=1e-6)

    def test_reduce_multiply_is_out_of_place(self):
        base = np.full((2, 4), 2.0, dtype="float32")
        index = np.array([[2], [3]], dtype="int32")
        src = np.array([[1.23], [1.23]], dtype="float32")
        x = jt.array(base)
        out = x.scatter(1, jt.array(index), jt.array(src), reduce="multiply")
        out.sync()
        np.testing.assert_array_equal(x.numpy(), base)
        np.testing.assert_allclose(
            out.numpy(), _np_scatter(base, 1, index, src, "multiply"),
            rtol=1e-5, atol=1e-5)

    def test_scalar_src_is_out_of_place(self):
        base = np.zeros((3, 4), dtype="float32")
        index = np.array([[0, 1, 2, 0]], dtype="int32")
        x = jt.array(base)
        out = x.scatter(0, jt.array(index), 9.0)
        out.sync()
        np.testing.assert_array_equal(x.numpy(), base)
        self.assertAlmostEqual(float(out.numpy().sum()), 36.0, places=4)

    def test_repeated_calls_compose_from_the_same_base(self):
        """The clearest symptom of the old behaviour: results accumulated."""
        x = jt.array(np.zeros((3, 4), dtype="float32"))
        index = jt.array(np.array([[0, 1, 2, 0]], dtype="int32"))
        first = x.scatter(0, index, 1.0)
        first.sync()
        second = x.scatter(0, index, 2.0)
        second.sync()
        # Both start from the same zeros; the in-place version made the second
        # call read a base the first call had already rewritten.
        self.assertAlmostEqual(float(first.numpy().sum()), 4.0, places=4)
        self.assertAlmostEqual(float(second.numpy().sum()), 8.0, places=4)
        np.testing.assert_array_equal(x.numpy(), np.zeros((3, 4), "float32"))

    def test_scatter_add_and_scatter_add_underscore(self):
        base = np.zeros((3, 4), dtype="float32")
        # duplicate indices: scatter_add accumulates, unlike plain scatter
        index = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype="int32")
        src = np.ones((2, 4), dtype="float32")
        x = jt.array(base)
        out = x.scatter_add(0, jt.array(index), jt.array(src))
        out.sync()
        np.testing.assert_array_equal(x.numpy(), base)
        np.testing.assert_allclose(
            out.numpy(), _np_scatter(base, 0, index, src, "add"),
            rtol=1e-6, atol=1e-6)
        y = jt.array(base)
        y.scatter_add_(0, jt.array(index), jt.array(src))
        np.testing.assert_allclose(y.numpy(), out.numpy(), rtol=1e-6, atol=1e-6)

    def test_gradient_flows_through_the_copy(self):
        src = jt.array(self.src)
        base = jt.array(self.base)
        out = base.scatter(0, jt.array(self.index), src)
        grad_src, grad_base = jt.grad(out.sum(), [src, base])
        # every scattered position takes its value from src, so those cells
        # contribute 1 to src and 0 to base.
        expected_src = np.zeros_like(self.src)
        expected_src[0] = 1.0
        np.testing.assert_allclose(
            grad_src.numpy(), expected_src, rtol=1e-6, atol=1e-6)
        overwritten = _np_scatter(
            np.zeros_like(self.base), 0, self.index,
            np.ones_like(self.src[:1]))
        np.testing.assert_allclose(
            grad_base.numpy(), 1.0 - overwritten, rtol=1e-6, atol=1e-6)


class TestScatterIsOutOfPlaceCuda(TestScatterIsOutOfPlace):
    @unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
    def setUp(self):
        super().setUp()
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = 0


if __name__ == "__main__":
    unittest.main()

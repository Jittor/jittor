# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The CUDA backends' user-input boundaries, exercised rather than counted.

Task 2.19 splits reporting in two: a caller who handed in the wrong shape,
dtype or layout gets an exception it can catch (``USER_CHECK`` /
``USER_CHECKop``), while a state the implementation says is impossible stays an
assertion.  Most of the CUDA half of that migration was accepted on evidence
that could not decide the question it was asked: a count of ``USER_CHECK``
occurrences plus ``nvcc -c`` on the translation unit.  Both of those are happy
with a check that aborts the process, or one that no call from Python can ever
reach.

Only a run can tell the difference, so this file runs them.  Each case asserts
three things:

* the boundary is *reachable* -- some ordinary call from Python arrives at it;
* it raises, and the exception crosses pyjt as a catchable ``RuntimeError``
  naming the operand rather than killing the interpreter;
* the runtime is still usable afterwards, which is what separates "raised" from
  "raised and left the graph in a state where nothing else works".

The third one is not decoration.  The failure this file was written after was a
release path that reached ``std::terminate``: the process died between two
tests, with no failing test to point at and every later file in the directory
unrun (see ``test_var_holder_teardown.py``).

The ops are called directly rather than through ``jt.nn``.  The Python wrappers
validate first, so a test that goes through them proves the wrapper checks and
leaves the C++ boundary -- the thing 2.19 changed -- untouched.
"""
import unittest

import numpy as np

import jittor as jt

from _helpers.assertions import expect_error


def _ops(name):
    module = getattr(jt.compile_extern, name, None)
    if module is None:
        raise unittest.SkipTest("%s is not available in this build" % name)
    return module


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class CudaBoundaryCase(unittest.TestCase):
    """Base: every rejection is followed by a real computation.

    A boundary that raises and then leaves the runtime unusable would satisfy
    ``assertRaises`` just as well as one that recovers, and the difference is
    the whole point of separating user errors from internal invariants.
    """

    def rejects(self, make, match):
        error = expect_error(lambda: make().sync(),
                             exc_type=RuntimeError, match=match)
        self.assertNotIsInstance(error, SystemExit)
        with jt.flag_scope(use_cuda=1):
            self.assertEqual(float((jt.ones((4, 4)) * 2).sum().item()), 32.0,
                             "the runtime did not survive %r" % match)
        return error


class TestCudnnConvBoundaries(CudaBoundaryCase):
    def setUp(self):
        self.cudnn = _ops("cudnn_ops")

    @jt.flag_scope(use_cuda=1)
    def test_input_rank(self):
        self.rejects(
            lambda: self.cudnn.cudnn_conv(
                jt.random((4, 8, 8)), jt.random((4, 4, 3, 3)),
                1, 1, 0, 0, 1, 1, 1),
            r"x->shape\.size\(\)\(3\) == 4")

    @jt.flag_scope(use_cuda=1)
    def test_weight_rank(self):
        self.rejects(
            lambda: self.cudnn.cudnn_conv(
                jt.random((1, 4, 8, 8)), jt.random((4, 4, 3)),
                1, 1, 0, 0, 1, 1, 1),
            r"w->shape\.size\(\)\(3\) == 4")

    @jt.flag_scope(use_cuda=1)
    def test_grouped_channels_must_match(self):
        self.rejects(
            lambda: self.cudnn.cudnn_conv(
                jt.random((1, 4, 8, 8)), jt.random((4, 3, 3, 3)),
                1, 1, 0, 0, 1, 1, 1),
            r"wci \* groups\(3\) == xc\(4\)")

    @jt.flag_scope(use_cuda=1)
    def test_a_format_string_that_names_no_axis(self):
        """The layout is a user-supplied string, so a typo in it is a user
        error.  Unrefused it indexed ``shape[3]`` for an axis that is not in
        the format at all and convolved whatever that happened to be."""
        self.rejects(
            lambda: self.cudnn.cudnn_conv(
                jt.random((1, 4, 8, 8)), jt.random((4, 4, 3, 3)),
                1, 1, 0, 0, 1, 1, 1, "NCXW", "oihw", ""),
            r"Not a valid format")


class TestCudnnConv3dBoundaries(CudaBoundaryCase):
    def setUp(self):
        self.cudnn = _ops("cudnn_ops")

    @jt.flag_scope(use_cuda=1)
    def test_input_rank(self):
        self.rejects(
            lambda: self.cudnn.cudnn_conv3d(
                jt.random((4, 4, 4, 4)), jt.random((4, 4, 3, 3, 3)),
                1, 1, 1, 0, 0, 0, 1, 1, 1, 1),
            r"x->shape\.size\(\)\(4\) == 5")

    @jt.flag_scope(use_cuda=1)
    def test_weight_rank(self):
        self.rejects(
            lambda: self.cudnn.cudnn_conv3d(
                jt.random((1, 4, 4, 4, 4)), jt.random((4, 4, 3, 3)),
                1, 1, 1, 0, 0, 0, 1, 1, 1, 1),
            r"w->shape\.size\(\)\(4\) == 5")

    @jt.flag_scope(use_cuda=1)
    def test_grouped_channels_must_match(self):
        self.rejects(
            lambda: self.cudnn.cudnn_conv3d(
                jt.random((1, 4, 4, 4, 4)), jt.random((4, 3, 3, 3, 3)),
                1, 1, 1, 0, 0, 0, 1, 1, 1, 1),
            r"wci \* groups\(3\) == xc\(4\)")


class TestCudnnRnnBoundaries(CudaBoundaryCase):
    def setUp(self):
        self.cudnn = _ops("cudnn_ops")

    def _rnn(self, x, input_size=8, proj_size=0):
        # The flat weight is oversized on purpose: every boundary below fires
        # before the weight is laid out, and sizing it exactly would make the
        # test depend on the layout rather than on the boundary.
        return lambda: self.cudnn.cudnn_rnn(
            x, jt.random((1, 4, 8)), jt.random((1, 4, 8)), jt.random((4096,)),
            "lstm", input_size, 8, 1, proj_size, 0.0, True, False, False)[0]

    @jt.flag_scope(use_cuda=1)
    def test_input_rank(self):
        self.rejects(self._rnn(jt.random((5, 8))),
                     r"x->shape\.size\(\)\(2\) == 3")

    @jt.flag_scope(use_cuda=1)
    def test_input_channels_must_match_input_size(self):
        self.rejects(self._rnn(jt.random((5, 4, 9))),
                     r"x->shape\[2\]\(9\) == input_size\(8\)")

    @jt.flag_scope(use_cuda=1)
    def test_projected_lstm_is_refused_by_name(self):
        """cuDNN has no projected LSTM on this path.  Unrefused this reached
        ``cudnnSetRNNDescriptor_v6`` and came back CUDNN_STATUS_NOT_SUPPORTED,
        which does not say which of its many arguments was the problem."""
        self.rejects(self._rnn(jt.random((5, 4, 8)), proj_size=2),
                     r"proj_size\(2\) == 0")


class TestCubSegmentBoundaries(CudaBoundaryCase):
    """``offsets`` describes the segmentation and comes from the caller.

    CUB reads it as ``int32`` and as ``n + 1`` entries.  Handed anything else
    it used to read past the end of the array, or reinterpret ``int64`` pairs
    as segment starts, and sort by whatever that produced -- a wrong answer,
    not an error.
    """

    def setUp(self):
        self.cub = _ops("cub_ops")
        self.x = jt.random((2, 3, 4))
        self.indexes = jt.zeros((2, 3, 4), "int32")
        self.offsets = jt.array(np.arange(7).astype("int32"))

    @jt.flag_scope(use_cuda=1)
    def test_argsort_offsets_dtype(self):
        self.rejects(
            lambda: self.cub.cub_argsort(
                self.x, self.indexes, jt.zeros((7,), "int64"),
                False, "int32")[0],
            r"offsets->dtype\(\)==ns_int32")

    @jt.flag_scope(use_cuda=1)
    def test_argsort_indexes_rank(self):
        self.rejects(
            lambda: self.cub.cub_argsort(
                self.x, jt.zeros((2, 3), "int32"), self.offsets,
                False, "int32")[0],
            r"x->shape\.size\(\) == indexes->shape\.size\(\)")

    @jt.flag_scope(use_cuda=1)
    def test_argsort_indexes_shape(self):
        self.rejects(
            lambda: self.cub.cub_argsort(
                self.x, jt.zeros((2, 3, 5), "int32"), self.offsets,
                False, "int32")[0],
            r"x->shape\[i\] == indexes->shape\[i\]")

    @jt.flag_scope(use_cuda=1)
    def test_argsort_offsets_rank(self):
        self.rejects(
            lambda: self.cub.cub_argsort(
                self.x, self.indexes, jt.zeros((7, 1), "int32"),
                False, "int32")[0],
            r"offsets->shape\.size\(\)\(2\) == 1")

    @jt.flag_scope(use_cuda=1)
    def test_argsort_offsets_length(self):
        self.rejects(
            lambda: self.cub.cub_argsort(
                self.x, self.indexes, jt.zeros((5,), "int32"),
                False, "int32")[0],
            r"offsets->shape\[0\]\(5\) == n \+ 1\(7\)")

    @jt.flag_scope(use_cuda=1)
    def test_arg_reduce_offsets_dtype(self):
        self.rejects(
            lambda: self.cub.cub_arg_reduce(
                self.x, jt.zeros((7,), "int64"), "min", False)[0],
            r"offsets->dtype\(\)==ns_int32")

    @jt.flag_scope(use_cuda=1)
    def test_arg_reduce_offsets_rank(self):
        self.rejects(
            lambda: self.cub.cub_arg_reduce(
                self.x, jt.zeros((7, 1), "int32"), "min", False)[0],
            r"offsets->shape\.size\(\)\(2\) == 1")

    @jt.flag_scope(use_cuda=1)
    def test_arg_reduce_offsets_length(self):
        self.rejects(
            lambda: self.cub.cub_arg_reduce(
                self.x, jt.zeros((5,), "int32"), "min", False)[0],
            r"offsets->shape\[0\]\(5\) == n \+ 1\(7\)")


class TestCurandBoundaries(CudaBoundaryCase):
    def setUp(self):
        self.curand = _ops("curand_ops")

    @jt.flag_scope(use_cuda=1)
    def test_distribution_name(self):
        self.rejects(
            lambda: self.curand.curand_random((4, 4), "float32", "abs"),
            r"type == ns_normal \|\| type == ns_uniform")

    @jt.flag_scope(use_cuda=1)
    def test_dtype_curand_cannot_draw(self):
        """curand draws float and double only.  Anything else used to expand
        to a ``curandGenerate*`` against a pointer of the wrong type and fail
        inside nvcc, at compile time, naming a generated file."""
        error = self.rejects(
            lambda: self.curand.curand_random((4, 4), "int32", "uniform"),
            r"dtype == ns_float32 \|\| dtype == ns_float64")
        self.assertIn("Draw float32 and cast", str(error))


class TestCufftBoundaries(CudaBoundaryCase):
    @jt.flag_scope(use_cuda=1)
    def test_unsupported_dtype_names_itself(self):
        """This one is checked in ``jit_prepare``, which runs on a compile
        worker, so the parallel compiler re-raises it as its own
        ``RuntimeError`` and the ``UserError`` class is lost on the way out.
        The message survives, which is what a caller reads; the class does not,
        which is why this asserts the text and not the type.
        """
        cufft = _ops("cufft_ops")
        self.rejects(
            lambda: cufft.cufft_fft(jt.zeros((1, 4, 4, 2), "float16"), False),
            r"not supported fft dtype")


if __name__ == "__main__":
    unittest.main()

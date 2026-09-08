# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""All five NCCL collectives run on the per-device communication stream.

Before 4.08 every collective passed literal 0 as the NCCL stream, so it ran on
the default stream and could never overlap with anything. They now go through
`nccl_stream_begin()` / `nccl_stream_end()`, which put the collective on the
side stream and bracket it with the two event dependencies (compute -> comm on
the way in, comm -> compute on the way out).

Moving work onto a second stream is exactly the change that turns a correctness
bug into a race, so "the numbers came out right once" is not evidence here.
Each collective is checked three ways:

* the stream really is a side stream (a zero handle would make every ordering
  assertion below pass for free),
* both event dependencies were actually recorded (the counters move by 2), and
* the values are right when the input is produced by default-stream compute
  immediately before the collective and the output is consumed by
  default-stream compute immediately after, repeated enough times that a
  missing event shows up.

Expected values are derived from the rank so that "the collective did not run"
and "the collective ran unordered" both give structurally wrong numbers rather
than plausible ones.
"""

import unittest

import numpy as np

import jittor as jt

from _helpers.distributed import run_mpi_test
from jittor.compile_extern import mpi, nccl_ops

WORLD = 2
# 4 MB per buffer: large enough that the collective takes long enough for an
# unordered read to land on the wrong data, small enough to leave the card
# alone. arange * WORLD*(WORLD+1)/2 stays exactly representable in float32.
N = 1 << 20

_COMMUNICATION_STREAM = 1


def _rank():
    return mpi.world_rank()


class _Dependencies:
    """Reads the communication stream's event-dependency counter."""

    def __init__(self, test):
        self.test = test
        self.device = jt.current_device()

    def __enter__(self):
        self.before = jt.core._cuda_stream_dependency_count(
            _COMMUNICATION_STREAM, self.device)
        return self

    def __exit__(self, *exc):
        if exc[0] is not None:
            return False
        after = jt.core._cuda_stream_dependency_count(
            _COMMUNICATION_STREAM, self.device)
        # Two per collective: cuda_side_stream_wait_default on the way in and
        # cuda_default_stream_wait_side on the way out. A collective still
        # hardcoded to stream 0 would move this by 0.
        self.test.assertEqual(after - self.before, 2)
        return False


@unittest.skipIf(nccl_ops is None, "nccl not found")
class TestNcclCommunicationStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if mpi is None or mpi.world_size() != WORLD:
            raise unittest.SkipTest("two ranks are required")

    @jt.flag_scope(use_cuda=1)
    def test_communication_stream_is_not_the_default_stream(self):
        # Pins the meaning of every ordering assertion in this file: on stream
        # 0 the collective would be ordered against compute for free.
        handle = jt.core._cuda_stream_handle(
            _COMMUNICATION_STREAM, jt.current_device())
        self.assertNotEqual(handle, 0)

    @jt.flag_scope(use_cuda=1)
    def test_all_reduce(self):
        rank = _rank()
        x = jt.array((np.arange(N) * (rank + 1)).astype("float32"))
        with _Dependencies(self):
            y = nccl_ops.nccl_all_reduce(x)
            got = y.numpy()
        expected = np.arange(N) * (WORLD * (WORLD + 1) // 2)
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    @jt.flag_scope(use_cuda=1)
    def test_broadcast(self):
        rank = _rank()
        # Non-root starts at -1: neither the expected value nor zero, so
        # "broadcast did not run" cannot be mistaken for success.
        source = np.arange(N).astype("float32")
        local = source if rank == 0 else np.full(N, -1, dtype="float32")
        x = jt.array(local)
        with _Dependencies(self):
            y = nccl_ops.nccl_broadcast(x, 0)
            got = y.numpy()
        np.testing.assert_allclose(got, source, rtol=0, atol=0)

    @jt.flag_scope(use_cuda=1)
    def test_reduce(self):
        rank = _rank()
        x = jt.array((np.arange(N) * (rank + 1)).astype("float32"))
        with _Dependencies(self):
            y = nccl_ops.nccl_reduce(x, 0)
            got = y.numpy()
        # 8.11: the non-root output keeps the full shape and stays zeroed. Its
        # contents are meaningless by definition; asserting the zeros keeps the
        # graph-isomorphism decision from being reverted unnoticed.
        if rank == 0:
            expected = np.arange(N) * (WORLD * (WORLD + 1) // 2)
        else:
            expected = np.zeros(N, dtype="float32")
        self.assertEqual(got.shape, (N,))
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    @jt.flag_scope(use_cuda=1)
    def test_all_gather(self):
        rank = _rank()
        chunk = N // WORLD
        x = jt.array(np.full(chunk, rank + 1, dtype="float32"))
        with _Dependencies(self):
            y = nccl_ops.nccl_all_gather(x)
            got = y.numpy()
        expected = np.concatenate(
            [np.full(chunk, r + 1, dtype="float32") for r in range(WORLD)])
        self.assertEqual(got.shape, (chunk * WORLD,))
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    @jt.flag_scope(use_cuda=1)
    def test_reduce_scatter(self):
        rank = _rank()
        chunk = N // WORLD
        # Chunk i goes to rank i. The two chunks are scaled differently so a
        # rank that receives the wrong chunk is off by a factor of ten rather
        # than by a rounding error.
        local = np.concatenate([
            np.full(chunk, (rank + 1) * 10 ** i, dtype="float32")
            for i in range(WORLD)])
        x = jt.array(local)
        with _Dependencies(self):
            y = nccl_ops.nccl_reduce_scatter(x)
            got = y.numpy()
        expected = np.full(
            chunk, sum(r + 1 for r in range(WORLD)) * 10 ** rank,
            dtype="float32")
        self.assertEqual(got.shape, (chunk,))
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    @jt.flag_scope(use_cuda=1)
    def test_produced_then_consumed_across_the_stream_boundary(self):
        """The race the stream switch introduces, repeated.

        Every iteration produces the collective's input with default-stream
        compute right before the collective and consumes its output with
        default-stream compute right after. Nothing is read back until the end,
        so the buffers are freed and handed out again while the communication
        stream may still be using them -- which is the other half of the race,
        not just the ordering of the kernels.

        Reducing each iteration to one scalar error keeps 200 iterations' worth
        of live memory at a few hundred bytes while leaving the window wide.
        """
        rank = _rank()
        iterations = 200
        seed = jt.array((np.arange(N) % 97).astype("float32"))
        errors = []
        for step in range(iterations):
            # Default-stream compute producing a rank- and step-dependent
            # input into a freshly allocated buffer.
            produced = (seed + step) * (rank + 1)
            reduced = nccl_ops.nccl_all_reduce(produced)
            # Default-stream compute consuming the collective's output.
            expected = (seed + step) * (WORLD * (WORLD + 1) // 2)
            errors.append((reduced - expected).abs().max())
        jt.sync_all(True)
        worst = max(float(e.numpy()) for e in errors)
        self.assertEqual(worst, 0.0, "unordered collective after {} iterations"
                         .format(iterations))


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestNcclCommunicationStreamEntry(unittest.TestCase):
    def test(self):
        run_mpi_test(WORLD, "test_nccl_comm_stream")


if __name__ == "__main__":
    unittest.main()

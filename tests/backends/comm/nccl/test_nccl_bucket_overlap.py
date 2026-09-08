# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""NCCL bucketing (`ncclGroupStart`/`ncclGroupEnd`) and the deferred join.

8.02's second half. Two things a collective could not do before:

* be **grouped** with its neighbours into one submission instead of one launch
  per tensor, and
* let default-stream compute **run ahead of it**, which is impossible while
  every collective joins the default stream back the instant it is enqueued.

The overlap itself is not something a wall clock proves on this class of
machine -- see the note at the bottom -- so what is asserted here is the
structure that makes overlap possible and, more importantly, the correctness it
puts at risk. The timeline evidence is produced separately with a profiler; the
recipe is in `agent/skills/jittor-distributed-verification/SKILL.md`.

The dependency counter is what distinguishes the three shapes, because each
`nccl_stream_begin` contributes one compute->comm dependency and each join
contributes one comm->compute dependency:

| shape                     | dependencies for N collectives |
| ------------------------- | ------------------------------ |
| ungrouped                 | 2N   (one join per collective) |
| grouped, joins at the end | N+1  (one join per bucket)     |
"""

import unittest

import numpy as np

import jittor as jt
import jittor.distributed as jd

from _helpers.distributed import run_mpi_test
from jittor.compile_extern import mpi, nccl_ops

WORLD = 2
NB = 4                  # collectives per bucket
NUMEL = 1 << 18         # 1 MB each
_COMMUNICATION_STREAM = 1

POISON = -12345.0


def _rank():
    return mpi.world_rank()


def _reduced(x):
    return x * (WORLD * (WORLD + 1) // 2)


@unittest.skipIf(nccl_ops is None, "nccl not found")
class TestNcclBucket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if mpi is None or mpi.world_size() != WORLD:
            raise unittest.SkipTest("two ranks are required")

    def setUp(self):
        self.device = jt.current_device()
        # A leaked deferred join would make the next test's counters and
        # pending flag meaningless, so never inherit one.
        jd.comm_wait()
        jt.sync_all(True)

    def tearDown(self):
        jd.comm_wait()
        jt.sync_all(True)

    def _deps(self):
        return jt.core._cuda_stream_dependency_count(
            _COMMUNICATION_STREAM, self.device)

    def _inputs(self, step=0):
        rank = _rank()
        return [jt.array(
                    (np.arange(NUMEL) % 97 + step).astype("float32")
                    * (rank + 1) * (i + 1))
                for i in range(NB)]

    def _expected(self, step=0):
        return [_reduced((np.arange(NUMEL) % 97 + step).astype("float32")
                         * (i + 1))
                for i in range(NB)]

    @jt.flag_scope(use_cuda=1)
    def test_ungrouped_collectives_join_one_by_one(self):
        xs = self._inputs()
        jt.sync_all(True)
        before = self._deps()
        ys = [nccl_ops.nccl_all_reduce(x) for x in xs]
        jt.sync(ys)
        jt.sync_all(True)
        # The baseline the bucket is measured against: 2 per collective.
        self.assertEqual(self._deps() - before, 2 * NB)
        self.assertFalse(jd.join_pending())
        for y, want in zip(ys, self._expected()):
            np.testing.assert_allclose(y.numpy(), want, rtol=0, atol=0)

    @jt.flag_scope(use_cuda=1)
    def test_bucket_submits_once_and_joins_once(self):
        xs = self._inputs()
        jt.sync_all(True)
        before = self._deps()
        with jd.bucket_scope(defer_join=False):
            ys = [nccl_ops.nccl_all_reduce(x) for x in xs]
            jt.sync(ys)
        jt.sync_all(True)
        # One join for the whole bucket instead of one per collective.
        self.assertEqual(self._deps() - before, NB + 1)
        # A synchronous bucket owes nothing: the join happened at scope exit,
        # after ncclGroupEnd(). This is the assertion that catches the join
        # being placed before the group is submitted, where it orders nothing.
        self.assertFalse(jd.join_pending())
        for y, want in zip(ys, self._expected()):
            np.testing.assert_allclose(y.numpy(), want, rtol=0, atol=0)

    @jt.flag_scope(use_cuda=1)
    def test_deferred_bucket_leaves_the_join_outstanding(self):
        xs = self._inputs()
        jt.sync_all(True)
        before = self._deps()
        with jd.bucket_scope(defer_join=True):
            ys = [nccl_ops.nccl_all_reduce(x) for x in xs]
            jt.sync(ys)
        # The window: the collectives are submitted, the default stream is not
        # ordered behind them yet. Without this the "overlap" could just be a
        # zero-length window that happened to give the right answer.
        self.assertTrue(jd.join_pending())
        self.assertEqual(self._deps() - before, NB)
        self.assertTrue(jd.comm_wait())
        self.assertFalse(jd.join_pending())
        self.assertEqual(self._deps() - before, NB + 1)
        # comm_wait is idempotent: nothing left to wait for.
        self.assertFalse(jd.comm_wait())
        for y, want in zip(ys, self._expected()):
            np.testing.assert_allclose(y.numpy(), want, rtol=0, atol=0)

    @jt.flag_scope(use_cuda=1)
    def test_a_second_bucket_before_the_wait_is_refused(self):
        xs = self._inputs()
        jt.sync_all(True)
        with jd.bucket_scope(defer_join=True):
            ys = [nccl_ops.nccl_all_reduce(x) for x in xs]
            jt.sync(ys)
        self.assertTrue(jd.join_pending())
        # Stacking buckets would keep growing the set of reserved blocks and
        # leave it ambiguous which bucket a later wait covers. Refused loudly
        # rather than allowed to accumulate.
        with self.assertRaises(Exception):
            with jd.bucket_scope(defer_join=True):
                pass
        jd.comm_wait()
        for y, want in zip(ys, self._expected()):
            np.testing.assert_allclose(y.numpy(), want, rtol=0, atol=0)

    @jt.flag_scope(use_cuda=1)
    def test_deferred_bucket_keeps_its_buffers_out_of_the_allocator(self):
        """The race the deferred join introduces, aimed at the allocator.

        While the join is outstanding the collective is still reading its
        inputs and writing its outputs, and the default stream is free to run
        ahead -- which includes the allocator handing those exact blocks to
        the next op. Every iteration drops all references to the inputs and
        then immediately allocates same-sized buffers full of a poison value,
        so a block that was not reserved gets overwritten with something that
        cannot be mistaken for a plausible result.
        """
        iterations = 40
        errors = []
        for step in range(iterations):
            xs = self._inputs(step)
            with jd.bucket_scope(defer_join=True):
                ys = [nccl_ops.nccl_all_reduce(x) for x in xs]
                jt.sync(ys)
            self.assertTrue(jd.join_pending())
            # Drop every reference to the inputs; their blocks are now
            # reclaimable as far as the allocator is concerned.
            del xs
            poison = [jt.full((NUMEL,), POISON, dtype="float32")
                      for _ in range(2 * NB)]
            jt.sync(poison)
            del poison
            jd.comm_wait()
            for y, want in zip(ys, self._expected(step)):
                errors.append((y - jt.array(want)).abs().max())
            del ys
        jt.sync_all(True)
        worst = max(float(e.numpy()) for e in errors)
        self.assertEqual(worst, 0.0,
                         "deferred bucket lost data after {} iterations"
                         .format(iterations))

    @jt.flag_scope(use_cuda=1)
    def test_bucket_matches_the_ungrouped_path_elementwise(self):
        xs = self._inputs(7)
        jt.sync_all(True)
        plain = [nccl_ops.nccl_all_reduce(x).numpy() for x in xs]
        with jd.bucket_scope(defer_join=False):
            sync_bucket = [nccl_ops.nccl_all_reduce(x) for x in xs]
            jt.sync(sync_bucket)
        sync_bucket = [y.numpy() for y in sync_bucket]
        with jd.bucket_scope(defer_join=True):
            async_bucket = [nccl_ops.nccl_all_reduce(x) for x in xs]
            jt.sync(async_bucket)
        jd.comm_wait()
        async_bucket = [y.numpy() for y in async_bucket]
        for a, b, c in zip(plain, sync_bucket, async_bucket):
            np.testing.assert_allclose(b, a, rtol=0, atol=0)
            np.testing.assert_allclose(c, a, rtol=0, atol=0)


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestNcclBucketEntry(unittest.TestCase):
    def test(self):
        run_mpi_test(WORLD, "test_nccl_bucket_overlap")


if __name__ == "__main__":
    unittest.main()

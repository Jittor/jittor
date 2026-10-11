# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A held Var that still has consumers must read back correctly.

`sync_all` sweeps only holders whose Var has no output edges:

    if (!v->var->_outputs.size())
        vars.push_back(v->var);

so a Var that is held by Python *and* still feeds a consumer is never a target
of that batch -- it is an intermediate, free to be fused away and to have the
buffers it was computed from released. Reading it later finds it unfinished and
recomputes it, and a recompute whose inputs have been recycled returns whatever
now occupies that memory.

This shape is not hypothetical. It is the one the MiniMax-H3 investigation
(section 48 of `docs/results/2026-09-14-vllm-omni-h3-enablement.md`) arrived at:
that decode is correct when the Var is synced explicitly and uniform random
bytes when it is left to the pipeline's own `sync_all`, and a fetch-time probe
showed `finished=0 mem_ptr=0` -- so it *is* recomputed, and the recompute is
what goes wrong.

These cases do **not** reproduce that failure; they passed on first run, in both
half and single precision, with and without allocation churn. They are here as
the invariant rather than as a reproduction: the property is one the executor is
supposed to hold, nothing else in `tests/core` states it, and the H3 evidence
says this is the neighbourhood. A regression here would be silent otherwise --
the value is wrong, not missing, and nothing raises.
"""
import unittest

import numpy as np

import jittor as jt


def _chain(n, dtype):
    """A Var with enough arithmetic behind it to be worth fusing."""
    base = jt.array(np.arange(n * n, dtype=np.float32).reshape(n, n) % 7.0)
    if dtype != "float32":
        base = base.cast(dtype)
    mid = base * 2.0 + 1.0
    for _ in range(6):
        mid = mid * 1.01 + 0.5
    return mid


def _expected(n):
    mid = np.arange(n * n, dtype=np.float32).reshape(n, n) % 7.0
    mid = mid * 2.0 + 1.0
    for _ in range(6):
        mid = mid * 1.01 + 0.5
    return mid


def _churn():
    """Allocate and drop large buffers so a released block would be reused.

    Without this a released buffer may simply still hold its old contents, and
    the test would pass for the wrong reason.
    """
    keep = [jt.random((4 * 1024 * 1024,)).float32() * 3.0 for _ in range(4)]
    jt.sync_all(True)
    del keep
    jt.gc()


class TestNonSinkHolderEvaluation(unittest.TestCase):

    # Half precision first: H3 fails only under autocast, and it is the dtype
    # where fusion has no boundary to stop at (section 47).
    DTYPES = ("float16", "float32")
    TOL = {"float16": 5e-3, "float32": 1e-5}

    def _check(self, mid, n, dtype, what):
        got = mid.numpy().astype(np.float64)
        want = _expected(n).astype(np.float64)
        rel = np.abs(got - want).max() / max(1e-9, np.abs(want).max())
        self.assertLess(
            rel, self.TOL[dtype],
            "%s: non-sink Var read back wrong (dtype=%s n=%d) -- "
            "got mean=%.4f std=%.4f, want mean=%.4f std=%.4f"
            % (what, dtype, n, got.mean(), got.std(), want.mean(), want.std()))

    def test_a_non_sink_holder_survives_sync_all(self):
        for dtype in self.DTYPES:
            for n in (256, 512):
                with self.subTest(dtype=dtype, n=n):
                    mid = _chain(n, dtype)
                    sink = mid.sum()          # gives mid an output edge
                    jt.sync_all(True)         # mid is not a target of this
                    self._check(mid, n, dtype, "sync_all")
                    del mid, sink
                    jt.gc()

    def test_a_non_sink_holder_survives_sync_all_then_reuse(self):
        for dtype in self.DTYPES:
            for n in (256, 512):
                with self.subTest(dtype=dtype, n=n):
                    mid = _chain(n, dtype)
                    sink = mid.sum()
                    jt.sync_all(True)
                    _churn()
                    self._check(mid, n, dtype, "sync_all + reuse")
                    del mid, sink
                    jt.gc()

    def test_a_non_sink_holder_survives_reuse_with_no_barrier(self):
        # Closest to H3, where the only barriers are the pipeline's own and the
        # Var is still pending when the allocator has moved on.
        for dtype in self.DTYPES:
            for n in (256, 512):
                with self.subTest(dtype=dtype, n=n):
                    mid = _chain(n, dtype)
                    sink = mid.sum()
                    _churn()
                    self._check(mid, n, dtype, "reuse, no barrier")
                    del mid, sink
                    jt.gc()

    def test_an_explicit_sync_is_not_what_makes_it_correct(self):
        # The control. If this were the only passing case, the three above
        # would be measuring their own read rather than the executor.
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                mid = _chain(256, dtype)
                sink = mid.sum()
                mid.sync(True)
                self._check(mid, 256, dtype, "explicit sync")
                del mid, sink
                jt.gc()


if __name__ == "__main__":
    unittest.main()

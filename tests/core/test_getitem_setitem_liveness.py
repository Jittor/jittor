# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Reproduction attempt for a P0 use-after-free flowey's own fork found and
fixed in getitem/setitem under lazy execution (a `Node::live_consumers`
graph-structural liveness counter, since master's `Node::liveness.{forward,
backward,pending}` are driven by Python-level VarHolder construction/
destruction, which flowey found can legitimately drop to 0 well before a
structurally-reachable, already-constructed consumer op actually runs under
lazy execution -- `lazy_execution` defaults to on, `src/core/executor.cc`).

Master's executor is a different, from-scratch rewrite (this file predates
that fix landing here at all), and separately has its own ASAN-verified fix
for an analogous relay/fusion use-after-free (KI-COMPILER-005 in
`agent/manuals/known-issues.md`: `VarRelayManager::add_relay_group` freed
inputs a fused op still pointed at; fixed by making `removed_input_vars` own
its vars via `VarPtr` instead of borrowing raw pointers) -- ownership rather
than a manual counter. Master's `GetitemOp`/`SetitemOp` also hold no cached
raw `Var*` members of their own (see `src/ops/composite/getitem_op.h`),
unlike the flowey-fork code the original bug report describes, which cached
`GetitemOp::x`/`SetitemOp::x/y` specifically to work around a *different*
executor's `_inputs()`/`input(N)` going stale under a revisited op.

This is a stress attempt to trigger the SAME failure mode (a Python-level
reference to an operand dropped, and garbage-collected, before the lazy op
consuming it actually executes) against master's own executor -- not a
transplant of flowey's fix, which targets internals this tree does not have.
Many iterations, varying shapes, explicit `gc.collect()`, and a large
throwaway allocation between construction and sync (to make it likely that
any memory freed too early gets overwritten before it's read, so a bug shows
up as a wrong *value*, not just rarely as a crash).
"""

import gc
import unittest

import numpy as np

from _helpers import capability as _test_capability

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


def _pressure():
    """Allocate and force a real device write, so anything freed too early
    is likely to be overwritten by the time it might get read again."""
    jt.random((1024, 1024)).sync()


class _GetitemSetitemLivenessContract:

    device_flag = 0

    def test_getitem_survives_dropped_source_before_sync(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            rng = np.random.RandomState(0)
            for _ in range(150):
                shape = (int(rng.randint(2, 12)), int(rng.randint(2, 12)))
                a_np = rng.randn(*shape).astype("float32")
                idx = int(rng.randint(0, shape[0]))

                a = jt.array(a_np)
                b = a[idx]  # lazy: GetitemOp constructed, not yet run
                del a
                gc.collect()
                _pressure()
                np.testing.assert_allclose(b.numpy(), a_np[idx], atol=1e-6)

    def test_setitem_survives_dropped_operands_before_sync(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            rng = np.random.RandomState(1)
            for _ in range(150):
                shape = (int(rng.randint(2, 12)), int(rng.randint(2, 12)))
                a_np = rng.randn(*shape).astype("float32")
                v_np = rng.randn(shape[1]).astype("float32")
                idx = int(rng.randint(0, shape[0]))

                a = jt.array(a_np)
                v = jt.array(v_np)
                a[idx] = v  # lazy: SetitemOp constructed, not yet run
                del v
                gc.collect()
                _pressure()
                expected = a_np.copy()
                expected[idx] = v_np
                np.testing.assert_allclose(a.numpy(), expected, atol=1e-6)

    def test_shared_getitem_result_survives_partial_sync_and_gc(self):
        # One getitem result feeds two independent downstream consumers.
        # Sync branch 1 first (which may retire/fuse its own ops), then drop
        # every remaining Python reference except branch 2's output, force
        # GC and allocation pressure, and only THEN sync branch 2 -- close to
        # the "an op is revisited across two separate scheduling passes"
        # shape flowey's original bug depended on: branch 2's not-yet-run op
        # still holds a graph edge to `shared`'s Var after every Python name
        # for it (and its own source `a`) is gone.
        with jt.flag_scope(use_cuda=self.device_flag):
            rng = np.random.RandomState(3)
            for _ in range(80):
                n = int(rng.randint(4, 16))
                a_np = rng.randn(n, n).astype("float32")
                a = jt.array(a_np)
                row = int(rng.randint(0, n))
                shared = a[row]  # GetitemOp, feeds two branches
                branch1 = shared * 2 + 1
                branch2 = shared - 3
                del a
                r1 = branch1.numpy()  # sync branch 1 only
                del branch1, shared
                gc.collect()
                _pressure()
                np.testing.assert_allclose(r1, a_np[row] * 2 + 1, atol=1e-6)
                np.testing.assert_allclose(branch2.numpy(), a_np[row] - 3, atol=1e-6)

    def test_chained_getitem_setitem_across_many_deferred_syncs(self):
        # The shape of workload that originally surfaced flowey's bug
        # (TensorCircuit's MPSCircuit.sample()/measure() loop): many
        # short-lived intermediate vars and heavy getitem/setitem traffic,
        # with syncs interleaved so ops are sometimes constructed well
        # before -- and potentially fused across -- the sync that actually
        # runs them.
        with jt.flag_scope(use_cuda=self.device_flag):
            rng = np.random.RandomState(2)
            acc = jt.zeros((8,), "float32")
            acc_np = np.zeros((8,), dtype="float32")
            for i in range(250):
                src_np = rng.randn(8).astype("float32")
                src = jt.array(src_np)
                j = int(rng.randint(0, 8))
                picked = src[j]
                del src
                acc[j] = acc[j] + picked
                acc_np[j] = acc_np[j] + src_np[j]
                if i % 17 == 0:
                    gc.collect()
                    _pressure()
                if i % 5 == 0:
                    np.testing.assert_allclose(acc.numpy(), acc_np, atol=1e-5)
            np.testing.assert_allclose(acc.numpy(), acc_np, atol=1e-5)


class TestGetitemSetitemLivenessCpu(_GetitemSetitemLivenessContract, unittest.TestCase):
    device_flag = 0


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestGetitemSetitemLivenessCuda(_GetitemSetitemLivenessContract, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()

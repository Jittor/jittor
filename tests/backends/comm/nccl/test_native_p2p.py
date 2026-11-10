"""Native NCCL point-to-point regression; run with ``nox -s nccl``."""

import os
import unittest

import numpy as np

import jittor as jt
from _helpers import capability


def _payload(rank):
    return np.arange(8, dtype="float32") + 100 * rank


@unittest.skipUnless(
    capability.check_accelerator("cuda", backend=jt).enabled and int(jt.world_size) >= 2,
    "requires the multi-rank NCCL nox gate",
)
class TestNativePointToPoint(unittest.TestCase):
    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_blocking_send_then_recv_both_directions(self):
        from jittor import distributed as dist

        self.assertNotEqual(os.environ.get("JITTOR_TORCH_SHIM"), "1")
        self.assertEqual(dist.get_backend(), "nccl")
        rank, world = int(jt.rank), int(jt.world_size)
        before = jt.core.backend_fallback_count()

        # A chain, one hop at a time, so no rank blocks in a send whose
        # matching receive has not been reached yet. Forward, then backward:
        # the second direction is what the official example needs.
        for direction in (1, -1):
            for step in range(world - 1):
                source, destination = (
                    (step, step + 1) if direction == 1 else (step + 1, step))
                buffer = jt.array(
                    np.zeros(8, dtype="float32")).cuda(0).stop_grad()
                if rank == source:
                    value = jt.array(_payload(source)).cuda(0).stop_grad()
                    dist.send(value, destination)
                    value.sync()
                elif rank == destination:
                    dist.recv(buffer, source)
                    np.testing.assert_array_equal(
                        buffer.float32().numpy(), _payload(source),
                        err_msg="direction %d step %d on rank %d" % (
                            direction, step, rank))
                dist.barrier()
        self.assertEqual(jt.core.backend_fallback_count(), before)

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_isend_irecv_one_directional(self):
        from jittor import distributed as dist

        rank, world = int(jt.rank), int(jt.world_size)
        self.assertEqual(dist.get_backend(), "nccl")
        for source, destination in ((0, 1), (1, 0)):
            if rank == source:
                value = jt.array(_payload(source)).cuda(0).stop_grad()
                work = dist.isend(value, destination)
                self.assertFalse(work.is_completed())
                work.wait()
                self.assertTrue(work.is_completed())
            elif rank == destination:
                buffer = jt.array(
                    np.zeros(8, dtype="float32")).cuda(0).stop_grad()
                work = dist.irecv(buffer, source)
                work.wait()
                np.testing.assert_array_equal(
                    buffer.float32().numpy(), _payload(source))
            dist.barrier()

    @jt.flag_scope(use_cuda=1, use_parallel_op_compiler=0, backend_fallback="error")
    def test_batch_isend_irecv_bidirectional_exchange(self):
        from jittor import distributed as dist

        rank = int(jt.rank)
        self.assertEqual(dist.get_backend(), "nccl")
        # The documented exchange: both ranks send and receive at once. Doing
        # this with separate isend/irecv calls stalls each communication
        # stream on its own send; one group is what lets both progress.
        if rank in (0, 1):
            peer = 1 - rank
            outgoing = jt.array(_payload(rank)).cuda(0).stop_grad()
            incoming = jt.array(
                np.zeros(8, dtype="float32")).cuda(0).stop_grad()
            works = dist.batch_isend_irecv([
                (True, outgoing, peer, 0),
                (False, incoming, peer, 0),
            ])
            self.assertEqual(len(works), 2)
            for work in works:
                work.wait()
            np.testing.assert_array_equal(
                incoming.float32().numpy(), _payload(peer))
        dist.barrier()

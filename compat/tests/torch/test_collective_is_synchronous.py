"""`async_op=False` has to mean the collective has been issued.

Jittor is lazy: `mpi_broadcast`/`mpi_all_reduce` express the operation, and the
NCCL call happens whenever something later forces a flush. Inside one process
that is free. Across a process group it is not: the order collectives are
issued in is a contract between peers, and no single process's scheduler can
see it.

MiniMax-H3 ref2va on four TP ranks deadlocked on exactly this -- three ranks
inside `_encode_visual_conditions`, one still inside `encode_prompt`, both in
`_broadcast_tensor`'s `.tolist()`, identical twenty minutes apart. Rank 0
carries the real tensor and so has a different op graph from its peers, so its
flush ran one collective while theirs had moved on to the next.

These run single-process, so no real collective is issued; what they pin is the
contract at the return point -- that a synchronous call flushes and an async
one does not, and that the short-circuit paths stay free.
"""
import unittest
from unittest import mock

import jittor as jt

from jittor.compat.torch.installers import distributed as D


class TestCollectiveResultContract(unittest.TestCase):
    def test_a_synchronous_issued_collective_is_flushed(self):
        var = jt.ones((4,))
        with mock.patch.object(D, "_sync_collective") as sync:
            self.assertIsNone(D._collective_result(var, False, issued=True))
        sync.assert_called_once_with(var)

    def test_an_async_collective_is_not_flushed(self):
        # torch's async_op=False is the blocking one; async_op=True hands back
        # a work handle and must not stall the caller.
        var = jt.ones((4,))
        with mock.patch.object(D, "_sync_collective") as sync:
            work = D._collective_result(var, True, issued=True)
        sync.assert_not_called()
        self.assertIsNotNone(work)

    def test_a_short_circuited_call_costs_nothing(self):
        # Single-member group, or a rank outside the group: no collective was
        # expressed, so there is nothing to flush and no reason to pay a sync.
        var = jt.ones((4,))
        with mock.patch.object(D, "_sync_collective") as sync:
            D._collective_result(var, False, issued=False)
        sync.assert_not_called()

    def test_sync_collective_handles_a_list(self):
        # all_gather returns the output list, not a single tensor.
        calls = []

        class Fake:
            def sync(self):
                calls.append(self)

        D._sync_collective([Fake(), Fake()])
        self.assertEqual(len(calls), 2)

    def test_sync_collective_ignores_things_without_sync(self):
        D._sync_collective([None, 5, "x"])

    def test_every_collective_marks_whether_it_issued(self):
        # The regression is structural: a collective that forgets `issued`
        # silently goes back to being lazy. Every return point must say.
        import ast
        import inspect
        source = inspect.getsource(D)
        tree = ast.parse(source)
        bare = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", "") != "_collective_result":
                continue
            if not any(kw.arg == "issued" for kw in node.keywords):
                bare.append(node.lineno)
        self.assertEqual(bare, [], "these _collective_result calls do not say "
                                   "whether a collective was issued: %s" % bare)


if __name__ == "__main__":
    unittest.main()

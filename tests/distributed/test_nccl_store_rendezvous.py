"""Ordering contract for the NCCL store rendezvous, without a GPU or a network.

`_init_nccl_from_store` exchanges the NCCL unique id through a store, then calls
`nccl_init_with_unique_id`. That call is a *collective*: it parks until every rank
arrives, and its pyjt wrapper holds the GIL for the whole of it. The process that
hosts the store -- rank 0 runs its server threads -- therefore cannot answer a
peer that is still in the store while it parks there, and the peer's `get` waits
out its timeout. Observed on TP2 as an intermittent
"NCCL store rendezvous timeout: rank 1 waited 120 s" (roughly one start in three),
with rank 0 at 100% CPU past its own `get` and rank 1 blocked in the store client.

The fix is order, not extra work: every rank records that it has read the id, and
nobody enters the collective until every rank has. This test pins that order with
a fake store, so a later edit cannot put the collective back before the barrier.

Run: python -m pytest tests/distributed/test_nccl_store_rendezvous.py
"""
import os
import unittest
from unittest.mock import patch

import jittor  # noqa: F401  (the module under test lives under jittor)
from jittor.build import compile_extern

_UNIQUE_ID = bytes(range(128))


class _Recorder:
    def __init__(self):
        self.calls = []


class _FakeStore:
    """A store whose `wait` is satisfied by whatever is already set."""

    def __init__(self, recorder, preloaded):
        self.recorder = recorder
        self.keys = dict(preloaded)

    def set(self, key, value):
        self.recorder.calls.append(("set", key))
        self.keys[key] = value

    def get(self, key, timeout=None):
        self.recorder.calls.append(("get", key))
        return self.keys[key]

    def wait(self, keys, timeout=None):
        self.recorder.calls.append(("wait", tuple(keys)))
        missing = [key for key in keys if key not in self.keys]
        self.assert_none(missing)

    @staticmethod
    def assert_none(missing):
        if missing:
            raise AssertionError(
                "wait called for keys nobody set: %r" % (missing,))


class _FakeNccl:
    """Stands in for the core module, recording when the collective is entered."""

    def __init__(self, recorder):
        self.recorder = recorder

    def nccl_get_unique_id(self):
        self.recorder.calls.append(("unique_id",))
        return _UNIQUE_ID

    def nccl_init_with_unique_id(self, unique_id):
        self.recorder.calls.append(("collective", tuple(unique_id)))


def _preload(world_size):
    keys = {"jittor/nccl/world/unique_id": _UNIQUE_ID}
    for rank in range(world_size):
        keys["jittor/nccl/world/unique_id_read/%d" % rank] = b"1"
        keys["jittor/nccl/world/initialized/%d" % rank] = b"1"
    return keys


def _run(world_size, rank):
    recorder = _Recorder()
    store = _FakeStore(recorder, _preload(world_size))
    with patch.dict(os.environ, {
            "JT_NCCL_WORLD_SIZE": str(world_size),
            "JT_NCCL_RANK": str(rank)}, clear=False):
        compile_extern._init_nccl_from_store(_FakeNccl(recorder), store=store)
    return recorder.calls


class TestNcclStoreRendezvousOrder(unittest.TestCase):
    def test_the_read_barrier_precedes_the_collective(self):
        for world_size in (1, 2, 3):
            calls = _run(world_size, 0)
            ops = [call[0] for call in calls]
            barrier = [i for i, call in enumerate(calls)
                       if call[0] == "wait"
                       and any("unique_id_read" in key for key in call[1])]
            self.assertTrue(barrier, "the pre-collective barrier disappeared")
            self.assertLess(
                max(barrier), ops.index("collective"),
                "the collective is entered before every rank has read the id")

    def test_every_rank_waits_for_every_other_rank(self):
        world_size = 3
        calls = _run(world_size, 1)
        barrier = [call for call in calls
                   if call[0] == "wait"
                   and any("unique_id_read" in key for key in call[1])]
        self.assertEqual(len(barrier), 1, "expected one read barrier")
        waited = set(barrier[0][1])
        for rank in range(world_size):
            self.assertIn("jittor/nccl/world/unique_id_read/%d" % rank, waited)

    def test_the_id_is_read_before_anyone_enters_the_collective(self):
        calls = _run(2, 0)
        ops = [call[0] for call in calls]
        self.assertLess(ops.index("get"), ops.index("collective"))
        self.assertLess(ops.index("wait"), ops.index("collective"))


if __name__ == "__main__":
    unittest.main()

"""Ordering contract for the NCCL store rendezvous, without a GPU or a network.

`_init_nccl_from_store` exchanges the NCCL unique id through a store, then calls
`nccl_init_with_unique_id`. That call is a *collective*: it parks until every rank
arrives, and its pyjt wrapper holds the GIL for the whole of it. The process that
hosts the store -- rank 0 runs its server threads -- therefore cannot answer a
peer that is still in the store while it parks there, and the peer's `get` waits
out its timeout. Observed on TP2 as an intermittent
"NCCL store rendezvous timeout: rank 1 waited 120 s" (roughly one start in three),
with rank 0 at 100% CPU past its own `get` and rank 1 blocked in the store client.

The fix is order, not extra work, and it is one-sided: every rank announces its
arrival with `Store.arrive` -- whose reply is flushed *before* the marker becomes
visible -- and only the rank that hosts the store waits, because it is the only
one that must not enter the collective early. A barrier of `set` + `wait` on both
sides cannot say what is needed, however many phases it has: a peer's own `wait`
is a request the host still owes a reply to, so each phase closes the previous
window and opens an identical one.

The fake-store tests below pin that order. `TestTheStoreHostDoesNotStrandAPeer`
drives the real thing with two processes, and is the one that fails when the
window is open.

Run: python -m pytest tests/distributed/test_nccl_store_rendezvous.py
"""
import json
import os
import socket
import tempfile
import unittest
from unittest.mock import patch

from _helpers.child_process import PYTHON, default_timeout, shell

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

    def arrive(self, key):
        self.recorder.calls.append(("arrive", key))
        self.keys[key] = b"1"

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
        keys["jittor/nccl/world/arrived/%d" % rank] = b"1"
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
    def test_arrival_is_announced_before_the_collective(self):
        """Every rank says it is here, and says it with `arrive`, not `set`.

        `arrive` is the whole fix: the store flushes its reply before the marker
        becomes visible, so a marker the host can see is a peer it has already
        answered. A `set` would make the marker visible first and leave the peer
        blocked on a reply that the host, once inside the GIL-holding
        collective, can no longer write.
        """
        for world_size in (1, 2, 3):
            for rank in range(world_size):
                calls = _run(world_size, rank)
                operations = [call[0] for call in calls]
                announcements = [
                    index for index, call in enumerate(calls)
                    if call[0] == "arrive"
                    and call[1] == "jittor/nccl/world/arrived/%d" % rank]
                self.assertEqual(
                    len(announcements), 1,
                    "rank %d of %d did not announce its arrival exactly once"
                    % (rank, world_size))
                self.assertLess(
                    announcements[0], operations.index("collective"),
                    "the collective is entered before this rank has arrived")

    def test_the_host_waits_for_every_rank_before_the_collective(self):
        world_size = 3
        calls = _run(world_size, 0)
        operations = [call[0] for call in calls]
        barrier = [(index, call) for index, call in enumerate(calls)
                   if call[0] == "wait"
                   and any("arrived" in key for key in call[1])]
        self.assertEqual(len(barrier), 1, "expected one arrival barrier")
        index, call = barrier[0]
        self.assertLess(
            index, operations.index("collective"),
            "the host enters the collective before every rank has arrived")
        for rank in range(world_size):
            self.assertIn("jittor/nccl/world/arrived/%d" % rank, set(call[1]))

    def test_a_peer_waits_for_nobody_before_the_collective(self):
        """The peers' own waits are what re-opened the window, so they are gone.

        A peer that waits leaves a request outstanding on the host's server, and
        the host -- released by the very marker that peer just set -- can be
        inside the collective before that request is read. Nothing is lost by
        dropping it: the collective is itself the barrier.
        """
        for world_size in (2, 3):
            for rank in range(1, world_size):
                calls = _run(world_size, rank)
                operations = [call[0] for call in calls]
                collective = operations.index("collective")
                self.assertEqual(
                    [call for call in calls[:collective] if call[0] == "wait"],
                    [],
                    "rank %d of %d waits on the store before the collective"
                    % (rank, world_size))

    def test_the_id_is_read_before_anyone_enters_the_collective(self):
        for rank in (0, 1):
            operations = [call[0] for call in _run(2, rank)]
            self.assertLess(
                operations.index("get"), operations.index("collective"))


#: One rank of the rendezvous, against a real TCPStore. The collective is
#: replaced by a call that holds the GIL for its whole duration, which is what
#: the pyjt wrapper around `nccl_init_with_unique_id` does.
_RANK_CHILD_SOURCE = r'''
import json, os, sys, time

rank = int(sys.argv[1])
world_size = int(sys.argv[2])
port = sys.argv[3]
gate = sys.argv[4]
hold = float(sys.argv[5])
report_path = sys.argv[6]

import jittor  # noqa: F401  -- imported before JT_NCCL_* is set, so no real rendezvous
from jittor.build import compile_extern

report = {"rank": rank, "collective": None, "error": None}


class _Nccl:
    def nccl_get_unique_id(self):
        return bytes(range(128))

    def nccl_init_with_unique_id(self, unique_id):
        report["collective"] = time.time()
        if rank == 0:
            # One C call, so the interpreter never reaches a bytecode boundary
            # and the GIL is held for the whole of it -- ncclCommInitRank's shape.
            sum(range(int(hold * 110_000_000)))


if rank:
    # A peer can be descheduled between two store calls. Make it certain, so the
    # interleaving is the same on every run instead of a coin flip.
    import jittor.distributed.store as store_module

    _request = store_module._TCPStoreClient.request

    def _descheduled(self, request, _request=_request):
        time.sleep(0.1)
        return _request(self, request)

    store_module._TCPStoreClient.request = _descheduled

os.environ.update({
    "JT_NCCL_WORLD_SIZE": str(world_size), "JT_NCCL_RANK": str(rank),
    "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": port,
    "JT_RENDEZVOUS_TIMEOUT_S": "60",
})

# Both ranks start the rendezvous together, whatever each one paid to import.
open(os.path.join(gate, "ready.%d" % rank), "w").close()
deadline = time.time() + 300
while time.time() < deadline:
    if all(os.path.exists(os.path.join(gate, "ready.%d" % peer))
           for peer in range(world_size)):
        break
    time.sleep(0.01)

try:
    compile_extern._init_nccl_from_store(_Nccl())
except BaseException as error:  # noqa: BLE001  -- reported, not raised
    report["error"] = "%s: %s" % (type(error).__name__, error)
with open(report_path, "w") as handle:
    json.dump(report, handle)
'''


class TestTheStoreHostDoesNotStrandAPeer(unittest.TestCase):
    """The rank that hosts the store must not enter the collective too early.

    Rank 0 runs the TCPStore server in Python threads of its own process, and
    the collective holds the GIL for its whole duration, so while rank 0 is
    inside it no thread of that process can read a peer's request or write a
    peer's reply. A barrier made of `set` + `wait` cannot close that window,
    however many phases it has: the store answers a peer's `set` and *then*
    makes the key visible, so rank 0 -- released by that very key -- can be in
    the collective while the peer is still waiting to be spoken to. The peer
    then sits in `readline` for as long as the collective lasts, which on a real
    run is forever, because the collective is waiting for that peer.

    This drives it with a real TCPStore, two child processes and a fake
    collective that holds the GIL the same way, and asks the only question that
    matters: did the peer get into the collective while rank 0 was still out of
    it?
    """

    HOLD_SECONDS = 6.0

    def test_the_peer_is_not_left_waiting_on_a_process_that_cannot_answer(self):
        directory = tempfile.mkdtemp(prefix="jittor-rendezvous-")
        source = os.path.join(directory, "rank.py")
        with open(source, "w") as handle:
            handle.write(_RANK_CHILD_SOURCE)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        reports = [os.path.join(directory, "report.%d" % rank) for rank in (0, 1)]

        command = " & ".join(
            '"%s" "%s" %d 2 %d "%s" %s "%s"'
            % (PYTHON, source, rank, port, directory, self.HOLD_SECONDS,
               reports[rank])
            for rank in (0, 1)) + " & wait"
        result = shell(command, timeout=default_timeout(), merge_stderr=True)

        collected = []
        for rank in (0, 1):
            self.assertTrue(
                os.path.exists(reports[rank]),
                "rank %d never finished the rendezvous.\n%s"
                % (rank, result.stdout[-4000:]))
            with open(reports[rank]) as handle:
                collected.append(json.load(handle))
        for report in collected:
            self.assertIsNotNone(
                report["collective"],
                "rank %d never reached the collective: %s"
                % (report["rank"], report["error"]))

        lag = collected[1]["collective"] - collected[0]["collective"]
        self.assertLess(
            lag, self.HOLD_SECONDS / 2,
            "rank 1 reached the collective %.1f s after rank 0, which is the "
            "length of rank 0's GIL-holding collective: it spent that time "
            "blocked on a store request that rank 0's process could not answer. "
            "The pre-collective barrier has to leave every peer with nothing "
            "outstanding before the host enters the collective." % lag)


if __name__ == "__main__":
    unittest.main()

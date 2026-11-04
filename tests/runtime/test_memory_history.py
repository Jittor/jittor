# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`jt.profiling.record_memory_history` / `memory_snapshot`: PyTorch's
snapshot, from Jittor's device pools.

The numbers are checked against what does not come from the history: the
pools' own live and reserved counters, a tensor whose size and building line
the test knows, the line that drops it, and an allocation the device cannot
satisfy. The format is checked against the keys PyTorch 2.11's
``torch.cuda.memory._snapshot()`` writes, so that PyTorch's memory_viz reads it.
"""

import inspect
import os
import pickle
import tempfile
import unittest

import jittor as jt
from jittor import profiling

from _helpers import capability as _test_capability

HAS_CUDA = bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)
NO_CUDA = "needs a CUDA device: the pools under test are the device's"

# What torch.cuda.memory._snapshot() writes (PyTorch 2.11).
SNAPSHOT_KEYS = ["segments", "device_traces", "allocator_settings", "external_annotations"]
SEGMENT_KEYS = ["device", "address", "total_size", "allocated_size", "active_size",
                "requested_size", "stream", "segment_type", "segment_pool_id", "is_expandable",
                "frames", "blocks"]
BLOCK_KEYS = ["address", "size", "requested_size", "state", "frames"]
TRACE_KEYS = ["action", "addr", "size", "stream", "time_us", "compile_context", "user_metadata",
              "frames"]
FRAME_KEYS = ["name", "filename", "line"]


def _here():
    """The caller's line number."""
    return inspect.currentframe().f_back.f_lineno


def _lines_in_this_file(frames):
    return [f["line"] for f in frames if f["filename"] == __file__]


def _active_blocks(snapshot):
    return [b for s in snapshot["segments"] for b in s["blocks"] if b["state"] == "active_allocated"]


@unittest.skipIf(not HAS_CUDA, NO_CUDA)
class TestMemoryHistory(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        jt.sync_all(True)
        jt.gc()
        self.device = int(jt.core.current_device())

    def tearDown(self):
        profiling.record_memory_history(None)
        jt.flags.use_cuda = self._use_cuda

    def test_the_snapshot_has_pytorchs_keys(self):
        profiling.record_memory_history()
        x = jt.randn(64, 64)
        x.sync()
        snapshot = profiling.memory_snapshot()
        self.assertEqual(list(snapshot), SNAPSHOT_KEYS)
        segment = snapshot["segments"][0]
        self.assertEqual(list(segment), SEGMENT_KEYS)
        self.assertEqual(list(segment["blocks"][0]), BLOCK_KEYS)
        self.assertIn(segment["segment_type"], ("small", "large"))
        trace = snapshot["device_traces"][self.device]
        self.assertEqual(list(trace[0]), TRACE_KEYS)
        block = next(b for b in _active_blocks(snapshot) if b["frames"])
        self.assertEqual(list(block["frames"][0]), FRAME_KEYS)

    def test_the_segments_add_up_to_the_pools_own_counters(self):
        profiling.record_memory_history()
        keep = [jt.randn(n, 1000) for n in (1, 300, 3000)]
        jt.sync_all(True)
        segments = [s for s in profiling.memory_snapshot()["segments"] if s["device"] == self.device]
        self.assertEqual(sum(s["allocated_size"] for s in segments),
                         jt.core.device_memory_used(self.device))
        self.assertEqual(sum(s["total_size"] for s in segments),
                         jt.core.device_memory_reserved(self.device))
        for s in segments:
            self.assertEqual(sum(b["size"] for b in s["blocks"]), s["total_size"])
        del keep

    def test_a_tensors_block_carries_its_size_and_the_line_that_built_it(self):
        profiling.record_memory_history()
        x = jt.randn(1000, 1000); line = _here()
        x.sync()
        snapshot = profiling.memory_snapshot()
        mine = [b for b in _active_blocks(snapshot) if b["requested_size"] == 4000000]
        self.assertEqual(len(mine), 1, [b["requested_size"] for b in _active_blocks(snapshot)])
        self.assertGreaterEqual(mine[0]["size"], 4000000)
        self.assertIn(line, _lines_in_this_file(mine[0]["frames"]))
        # Innermost frame first, as in PyTorch's snapshot: this test comes
        # before the unittest machinery that called it.
        files = [f["filename"] for f in mine[0]["frames"]]
        runner = [i for i, name in enumerate(files) if name.endswith(os.path.join("unittest", "case.py"))]
        self.assertTrue(runner, files)
        self.assertLess(files.index(__file__), runner[0])
        trace = snapshot["device_traces"][self.device]
        allocs = [e for e in trace if e["action"] == "alloc" and e["addr"] == mine[0]["address"]]
        self.assertEqual([e["size"] for e in allocs], [4000000])
        self.assertIn(line, _lines_in_this_file(allocs[0]["frames"]))

    def test_a_free_is_requested_and_completed_at_the_line_that_dropped_the_tensor(self):
        profiling.record_memory_history(context="all")
        x = jt.randn(1000, 1000)
        x.sync()
        address = next(b["address"] for b in _active_blocks(profiling.memory_snapshot())
                       if b["requested_size"] == 4000000)
        del x; line = _here()
        trace = profiling.memory_snapshot()["device_traces"][self.device]
        actions = [e["action"] for e in trace if e["addr"] == address]
        self.assertEqual(actions[-2:], ["free_requested", "free_completed"])
        free = [e for e in trace if e["addr"] == address and e["action"] == "free_completed"][-1]
        self.assertEqual(free["size"], 4000000)
        self.assertIn(line, _lines_in_this_file(free["frames"]))
        self.assertNotIn(address, [b["address"] for b in _active_blocks(profiling.memory_snapshot())])

    def test_replaying_the_trace_gives_the_live_blocks(self):
        profiling.record_memory_history()
        keep = []
        for n in (10, 200, 3000, 50):
            keep.append(jt.randn(n, 1000))
            jt.randn(n, 999).sync()          # dropped at once
        jt.sync_all(True)
        snapshot = profiling.memory_snapshot()
        live = set()
        for e in snapshot["device_traces"][self.device]:
            if e["action"] == "alloc":
                live.add(e["addr"])
            elif e["action"] == "free_completed":
                live.discard(e["addr"])
        active = {b["address"] for b in _active_blocks(snapshot)}
        self.assertTrue(live, "nothing was allocated under the history")
        self.assertTrue(live <= active, live - active)
        del keep

    def test_segments_returned_to_the_driver_are_traced(self):
        profiling.record_memory_history()
        x = jt.randn(4096, 4096)
        x.sync()
        del x
        jt.gc()
        trace = profiling.memory_snapshot()["device_traces"][self.device]
        grown = {e["addr"]: e["size"] for e in trace if e["action"] == "segment_alloc"}
        freed = {e["addr"]: e["size"] for e in trace if e["action"] == "segment_free"}
        self.assertTrue(grown, "a 64 MiB tensor came from no new segment")
        self.assertTrue(set(grown) & set(freed))
        for addr in set(grown) & set(freed):
            self.assertEqual(grown[addr], freed[addr])

    def test_an_allocation_the_device_refuses_is_an_oom_entry(self):
        profiling.record_memory_history()
        with self.assertRaises(Exception):
            jt.empty((1 << 38,), "float32").sync()
        entries = [e for e in profiling.memory_snapshot()["device_traces"][self.device]
                   if e["action"] == "oom"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["size"], 4 << 38)
        self.assertGreater(entries[0]["device_free"], 0)

    def test_stopping_forgets_the_trace_and_max_entries_keeps_the_newest(self):
        profiling.record_memory_history(max_entries=5)
        for _ in range(10):
            jt.randn(100).sync()
        trace = profiling.memory_snapshot()["device_traces"][self.device]
        self.assertEqual(len(trace), 5)
        self.assertEqual(trace[-1]["action"], "free_completed")
        # Taking a snapshot is itself an entry, seen by the next snapshot.
        trace = profiling.memory_snapshot()["device_traces"][self.device]
        self.assertEqual(trace[-1]["action"], "snapshot")
        profiling.record_memory_history(None)
        self.assertEqual(profiling.memory_snapshot()["device_traces"][self.device], [])

    def test_state_only_keeps_block_stacks_and_no_trace(self):
        profiling.record_memory_history("state")
        x = jt.randn(1000, 1000); line = _here()
        x.sync()
        snapshot = profiling.memory_snapshot()
        self.assertEqual(snapshot["device_traces"][self.device], [])
        mine = [b for b in _active_blocks(snapshot) if b["requested_size"] == 4000000]
        self.assertIn(line, _lines_in_this_file(mine[0]["frames"]))

    def test_record_function_ranges_are_external_annotations(self):
        profiling.record_memory_history()
        with profiling.record_function("load batch"):
            jt.randn(10).sync()
        annotations = profiling.memory_snapshot()["external_annotations"]
        self.assertEqual([(a["stage"], a["name"]) for a in annotations],
                         [("START", "load batch"), ("END", "load batch")])
        self.assertLessEqual(annotations[0]["time_us"], annotations[1]["time_us"])

    def test_the_pickle_round_trips(self):
        profiling.record_memory_history()
        jt.randn(10).sync()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snapshot.pickle")
            profiling.dump_memory_snapshot(path)
            with open(path, "rb") as f:
                data = pickle.load(f)
        self.assertEqual(list(data), SNAPSHOT_KEYS)
        self.assertTrue(data["device_traces"][self.device])

    def test_bad_arguments_are_refused(self):
        for kwargs in ({"enabled": "yes"}, {"context": "everything"}, {"stacks": "cpp"}):
            with self.assertRaises(ValueError):
                profiling.record_memory_history(**kwargs)


if __name__ == "__main__":
    unittest.main()

# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`jt.profiling.memory_stats` / `memory_summary`: PyTorch's allocator
statistics and table, from Jittor's device pools.

The statistics are counted on every allocation, free and segment change; the
checks hold their current values against a walk of the pools' segments and
blocks as they are now (`memory_snapshot`), which does not use the counters,
and their totals against allocations the test makes. The key set is the one
PyTorch 2.11's ``torch.cuda.memory_stats`` returns.
"""

import unittest

import jittor as jt
from jittor import profiling

from _helpers import capability as _test_capability

HAS_CUDA = bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)
NO_CUDA = "needs a CUDA device: the pools under test are the device's"

_POOLED = ("active", "active_bytes", "allocated_bytes", "allocation", "inactive_split",
           "inactive_split_bytes", "requested_bytes", "reserved_bytes", "segment")
# torch.cuda.memory_stats() keys (PyTorch 2.11).
TORCH_KEYS = sorted(
    ["%s.%s.%s" % (stat, pool, part) for stat in _POOLED
     for pool in ("all", "large_pool", "small_pool")
     for part in ("allocated", "current", "freed", "peak")]
    + ["%s.%s" % (stat, part) for stat in ("oversize_allocations", "oversize_segments")
       for part in ("allocated", "current", "freed", "peak")]
    + ["max_split_size", "num_alloc_retries", "num_device_alloc", "num_device_free", "num_ooms",
       "num_sync_all_streams"])


def _walk(device):
    """What the counters' current values must be, from the pools' segments."""
    want = {"%s.%s" % (k, p): 0 for k in _POOLED for p in ("all", "small_pool", "large_pool")}
    for segment in profiling.memory_snapshot()["segments"]:
        if segment["device"] != device:
            continue
        pools = ("all", segment["segment_type"] + "_pool")
        split = len(segment["blocks"]) > 1
        for pool in pools:
            want["segment." + pool] += 1
            want["reserved_bytes." + pool] += segment["total_size"]
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    want["allocation." + pool] += 1
                    want["active." + pool] += 1
                    want["allocated_bytes." + pool] += block["size"]
                    want["active_bytes." + pool] += block["size"]
                    want["requested_bytes." + pool] += block["requested_size"]
                elif split:
                    want["inactive_split." + pool] += 1
                    want["inactive_split_bytes." + pool] += block["size"]
    return want


@unittest.skipIf(not HAS_CUDA, NO_CUDA)
class TestMemoryStats(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1
        jt.sync_all(True)
        self.device = int(jt.core.current_device())

    def tearDown(self):
        profiling.record_memory_history(None)
        jt.flags.use_cuda = self._use_cuda

    def _check_current(self):
        stats = profiling.memory_stats(self.device)
        want = _walk(self.device)
        got = {k: stats[k + ".current"] for k in want}
        self.assertEqual(got, want)
        return stats

    def test_the_keys_are_pytorchs(self):
        self.assertEqual(list(profiling.memory_stats(self.device)), TORCH_KEYS)

    def test_current_values_match_a_walk_of_the_pools(self):
        profiling.record_memory_history("state")
        keep = []
        for n in (1, 7, 300, 3000, 12000):
            keep.append(jt.randn(n, 250))
            jt.randn(n, 249).sync()
        jt.sync_all(True)
        stats = self._check_current()
        self.assertEqual(stats["allocated_bytes.all.current"], jt.core.device_memory_used(self.device))
        self.assertEqual(stats["reserved_bytes.all.current"], jt.core.device_memory_reserved(self.device))
        self.assertGreater(stats["inactive_split.all.current"], 0)
        del keep[1::2]
        jt.sync_all(True)
        self._check_current()
        del keep
        jt.gc()
        self._check_current()

    def test_totals_and_peaks_count_what_was_allocated(self):
        profiling.reset_accumulated_memory_stats(self.device)
        jt.core.reset_device_memory_peak(self.device)
        base = profiling.memory_stats(self.device)
        self.assertEqual(base["allocation.all.allocated"], 0)
        self.assertEqual(base["allocated_bytes.all.peak"], base["allocated_bytes.all.current"])
        xs = [jt.randn(1000, 1000) for _ in range(3)]
        jt.sync_all(True)
        live = profiling.memory_stats(self.device)["allocated_bytes.all.current"]
        del xs
        stats = profiling.memory_stats(self.device)
        self.assertGreaterEqual(stats["allocation.all.allocated"], 3)
        self.assertGreaterEqual(stats["requested_bytes.all.allocated"], 3 * 4000000)
        self.assertGreaterEqual(stats["requested_bytes.all.freed"], 3 * 4000000)
        self.assertGreaterEqual(stats["allocated_bytes.all.peak"], live)
        self.assertGreaterEqual(live - base["allocated_bytes.all.current"], 3 * 4000000)
        for key in ("allocation.all", "allocated_bytes.all", "requested_bytes.small_pool",
                    "segment.large_pool", "inactive_split_bytes.all"):
            self.assertEqual(stats[key + ".current"],
                             base[key + ".current"] + stats[key + ".allocated"] - stats[key + ".freed"],
                             key)

    def test_a_refused_allocation_counts_an_oom_and_a_retry(self):
        before = profiling.memory_stats(self.device)
        with self.assertRaises(Exception):
            jt.empty((1 << 38,), "float32").sync()
        after = profiling.memory_stats(self.device)
        self.assertEqual(after["num_ooms"], before["num_ooms"] + 1)
        self.assertEqual(after["num_alloc_retries"], before["num_alloc_retries"] + 1)

    def test_segments_from_the_driver_are_counted(self):
        before = profiling.memory_stats(self.device)
        x = jt.randn(4096, 4096)
        x.sync()
        del x
        jt.gc()
        after = profiling.memory_stats(self.device)
        self.assertGreater(after["num_device_alloc"], before["num_device_alloc"])
        self.assertGreater(after["num_device_free"], before["num_device_free"])
        self.assertEqual(after["segment.all.allocated"] - after["segment.all.freed"],
                         after["segment.all.current"])

    def test_the_summary_is_pytorchs_table(self):
        jt.randn(100, 100).sync()
        stats = profiling.memory_stats(self.device)
        text = profiling.memory_summary(self.device)
        lines = text.splitlines()
        self.assertTrue(all(len(line) == 77 for line in lines), [len(l) for l in lines])
        self.assertIn("Jittor CUDA memory summary, device ID %d" % self.device, lines[1])
        self.assertTrue(lines[5].startswith("|        Metric         | Cur Usage  | Peak Usage |"))
        row = next(line for line in lines if line.startswith("| Allocations "))
        self.assertEqual(int(row.split("|")[2].split()[0]), stats["allocation.all.current"])
        self.assertEqual(len(profiling.memory_summary(self.device, abbreviated=True).splitlines()),
                         len(lines) - 2 * 9)


if __name__ == "__main__":
    unittest.main()

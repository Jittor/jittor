"""``torch.cuda.memory_stats`` has torch's keys and ``memory_summary`` is torch's
table.

``memory_stats`` used to return five of torch's ~120 keys, so code reading
``num_ooms`` or ``active_bytes.all.peak`` raised KeyError, and
``memory_summary`` printed a five-row table of jittor's own that tools parsing
torch's could not read. The table is torch's layout titled as jittor's.
"""

import unittest

import torch

from _helpers import capability as _test_capability

import jittor as jt

HAS_CUDA = bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)


@unittest.skipIf(not HAS_CUDA, "needs a CUDA device: the pools under test are the device's")
class TestTorchMemorySummary(unittest.TestCase):
    def test_memory_stats_answers_what_the_memory_functions_answer(self):
        x = torch.randn(1000, 1000, device="cuda")
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        self.assertEqual(stats["allocated_bytes.all.current"], torch.cuda.memory_allocated())
        self.assertEqual(stats["reserved_bytes.all.current"], torch.cuda.memory_reserved())
        self.assertEqual(stats["allocated_bytes.all.peak"], torch.cuda.max_memory_allocated())
        self.assertGreaterEqual(stats["requested_bytes.all.current"], 4000000)
        self.assertEqual(stats["allocation.all.current"],
                         stats["allocation.small_pool.current"] + stats["allocation.large_pool.current"])
        self.assertEqual(list(stats), sorted(stats))
        for key in ("num_ooms", "num_alloc_retries", "active_bytes.all.peak",
                    "inactive_split_bytes.all.current", "oversize_allocations.current",
                    "max_split_size", "num_sync_all_streams"):
            self.assertIn(key, stats)
        del x

    def test_memory_summary_is_torchs_table(self):
        torch.randn(100, 100, device="cuda").sum().item()
        lines = torch.cuda.memory_summary().splitlines()
        self.assertEqual(lines[0], "|" + "=" * 75 + "|")
        self.assertEqual(lines[1], "|" + " " * 18 + "Jittor CUDA memory summary, device ID %-18d |"
                         % torch.cuda.current_device())
        self.assertEqual(lines[5], "|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |")
        labels = [line[1:24].strip() for line in lines[6:] if line.startswith("| ")]
        self.assertEqual([l for l in labels if not l.startswith("from")], [
            "Allocated memory", "Active memory", "Requested memory", "GPU reserved memory",
            "Non-releasable memory", "Allocations", "Active allocs", "GPU reserved segments",
            "Non-releasable allocs", "Oversize allocations", "Oversize GPU segments"])
        self.assertEqual(len(lines), 47)
        self.assertEqual(len(torch.cuda.memory_summary(abbreviated=True).splitlines()), 29)

    def test_reset_accumulated_memory_stats_restarts_the_totals(self):
        torch.randn(100, device="cuda").sum().item()
        torch.cuda.reset_accumulated_memory_stats()
        stats = torch.cuda.memory_stats()
        self.assertEqual(stats["allocation.all.allocated"], 0)
        self.assertEqual(stats["num_device_alloc"], 0)


if __name__ == "__main__":
    unittest.main()

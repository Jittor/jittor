"""``torch.profiler`` events, averages, table and exports follow PyTorch's.

The table used to be a layout of jittor's own with "self" equal to "total";
kernels were not events; memory columns, input-shape and stack grouping,
``export_stacks`` and ``export_memory_timeline`` did nothing. The checks hold
the numbers against each other the way PyTorch's arithmetic does (a range's
total is its own time plus its children's, an operator's self device time is
its kernels') and the text against PyTorch 2.11's layout.
"""

import gzip
import inspect
import json
import os
import re
import tempfile
import unittest

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from _helpers import capability as _test_capability

import jittor as jt
from jittor.profiling import _cupti

HAS_CUDA = bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)
HAS_CUPTI = HAS_CUDA and _cupti.load() is not None


def _profile(**kwargs):
    model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(),
                                torch.nn.Linear(512, 10)).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(64, 256, device="cuda")

    def step():
        with record_function("forward"):
            loss = model(x).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], **kwargs) as prof:
        step()
        torch.cuda.synchronize()
    return prof


@unittest.skipIf(not HAS_CUPTI, "needs a CUDA device and CUPTI for device events")
class TestTorchProfilerTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prof = _profile(record_shapes=True, profile_memory=True, with_stack=True)

    def test_kernels_are_cuda_events_charged_to_their_operator(self):
        events = self.prof.events()
        kernels = [e for e in events if str(e.device_type) == "DeviceType.CUDA"]
        self.assertTrue(kernels)
        for k in kernels:
            self.assertEqual(k.cpu_time_total, 0)
            self.assertEqual(k.self_device_time_total, k.time_range.elapsed_us())
        ops = [e for e in events if e.kernels]
        self.assertTrue(ops)
        for op in ops:
            self.assertAlmostEqual(op.self_device_time_total,
                                   sum(k.time_range.elapsed_us() for k in op.kernels), places=6)

    def test_a_range_totals_its_own_time_and_the_operators_built_in_it(self):
        forward = [e for e in self.prof.events() if e.name == "forward"]
        self.assertEqual(len(forward), 1)
        forward = forward[0]
        self.assertTrue(forward.cpu_children, "nothing built in the range was charged to it")
        self.assertAlmostEqual(forward.cpu_time_total,
                               forward.self_cpu_time_total
                               + sum(c.cpu_time_total for c in forward.cpu_children), places=6)
        self.assertGreater(forward.device_time_total, 0)
        self.assertAlmostEqual(forward.device_time_total,
                               sum(c.device_time_total for c in forward.cpu_children), places=6)

    def test_operators_belong_to_the_range_that_built_them_not_the_one_that_ran_them(self):
        x = torch.randn(64, 64, device="cuda")
        (x @ x).relu().sum().item()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("build"):
                y = (x @ x).relu().sum()      # lazily: nothing runs here
            with record_function("run"):
                y.item()                      # the graph executes here
            torch.cuda.synchronize()
        events = prof.events()
        build = next(e for e in events if e.name == "build")
        run = next(e for e in events if e.name == "run")
        ops = [e for e in events if e.cpu_parent is build]
        self.assertTrue(ops)
        # Only the executor's own work (planning-time compiles) is the run's.
        self.assertFalse([e for e in events if e.cpu_parent is run and e.name != "jit_compile"])
        self.assertTrue(all(run.time_range.start <= e.time_range.start for e in ops))
        # The range they ran in does not count them as its own time, and the
        # one that built them totals them.
        inside = sum(e.cpu_time_total for e in events if e.cpu_parent is not None
                     and e.cpu_parent.name in ("build", "run")
                     and run.time_range.start <= e.time_range.start
                     and e.time_range.end <= run.time_range.end)
        self.assertAlmostEqual(run.self_cpu_time_total, run.time_range.elapsed_us() - inside, places=3)
        self.assertAlmostEqual(build.cpu_time_total,
                               build.self_cpu_time_total + sum(e.cpu_time_total for e in ops), places=3)
        self.assertGreater(build.device_time_total, 0)

    def test_a_first_run_compile_is_its_own_event_under_the_operator(self):
        x = torch.randn(64, 64, device="cuda")
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
            a = (x * 0.4142135); first = inspect.currentframe().f_lineno   # a new constant
            b = a - 0.7320508; second = inspect.currentframe().f_lineno
            b.exp().sum().item()
        events = prof.events()
        compiles = [e for e in events if e.name == "jit_compile"]
        self.assertTrue(compiles)
        for c in compiles:
            # Compiled while the executor planned, or by the operator as it
            # first ran -- then the operator's own time leaves it out.
            op = c.cpu_parent
            if op is None:
                continue
            self.assertIn(c, op.cpu_children)
            self.assertAlmostEqual(op.self_cpu_time_total,
                                   op.time_range.elapsed_us() - c.time_range.elapsed_us(), places=3)
        # The fused kernel was built on two lines; its stack names both.
        fused = [e for e in events if e.name.startswith("fused") and e.stack]
        self.assertTrue(fused)
        here = [s for s in fused[0].stack if os.path.basename(__file__) in s]
        self.assertGreaterEqual(len(here), 2, fused[0].stack)
        # For a flame graph each of those lines is its own call stack, not a
        # caller of the other.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "stacks.txt")
            prof.export_stacks(path, "self_cpu_time_total")
            lines = open(path).read().splitlines()
        marks = ["(%d):" % first, "(%d):" % second]
        self.assertTrue(any(marks[0] in line for line in lines), lines)
        self.assertTrue(any(marks[1] in line for line in lines), lines)
        self.assertFalse([line for line in lines if marks[0] in line and marks[1] in line], lines)

    def test_the_table_is_pytorchs(self):
        averages = self.prof.key_averages()
        text = averages.table(row_limit=10)
        lines = text.splitlines()
        sep = lines[0]
        self.assertTrue(re.fullmatch(r"(-+  )+", sep), sep)
        widths = [len(c) for c in sep.split("  ") if c]
        self.assertTrue(all(w == 12 for w in widths[1:]), widths)
        self.assertLessEqual(widths[0], 55)
        header = [h.strip() for h in re.split(r"\s{2,}", lines[1].strip())]
        self.assertEqual(header, ["Name", "Self CPU %", "Self CPU", "CPU total %", "CPU total",
                                  "CPU time avg", "Self CUDA", "Self CUDA %", "CUDA total",
                                  "CUDA time avg", "CPU Mem", "Self CPU Mem", "CUDA Mem",
                                  "Self CUDA Mem", "# of Calls"])
        self.assertEqual(lines[2], sep)
        self.assertEqual(len(lines), 3 + 10 + 3)
        self.assertEqual(lines[-3], sep)
        total_cpu = sum(e.self_cpu_time_total for e in averages)
        total_cuda = sum(e.self_device_time_total for e in averages
                         if str(e.device_type) == "DeviceType.CUDA")
        self.assertTrue(lines[-2].startswith("Self CPU time total: "))
        self.assertTrue(lines[-1].startswith("Self CUDA time total: "))
        self.assertIn("%.3f" % (total_cuda if total_cuda < 1e3 else total_cuda / 1e3), lines[-1])
        self.assertIn("%.3f" % (total_cpu if total_cpu < 1e3 else total_cpu / 1e3), lines[-2])
        for line in lines[3:-3]:
            self.assertEqual(len(line), len(sep), line)
            name = line[:widths[0]].strip()
            self.assertLessEqual(len(name), 55)

    def test_sort_keys_order_the_rows(self):
        rows = self.prof.key_averages()
        for key in ("cuda_time_total", "self_cpu_time_total", "self_cuda_memory_usage", "count"):
            text = rows.table(sort_by=key, row_limit=-1)
            ordered = sorted(rows, key=lambda e: getattr(e, key.replace("cuda", "device")), reverse=True)
            first = text.splitlines()[3].strip().split("  ")[0]
            self.assertTrue(ordered[0].key.startswith(first.rstrip(".")), (key, first, ordered[0].key))

    def test_operators_own_the_memory_they_allocate(self):
        rows = self.prof.key_averages()
        self.assertTrue(any(e.self_device_memory_usage > 0 for e in rows))
        matmul = max(rows, key=lambda e: e.self_device_memory_usage)
        self.assertGreaterEqual(matmul.self_device_memory_usage, 64 * 512 * 4)

    def test_input_shapes_group_and_show_as_lists(self):
        text = self.prof.key_averages(group_by_input_shape=True).table(row_limit=-1)
        self.assertIn("Input Shapes", text.splitlines()[1])
        self.assertIn("[[64, 256], [512, 256]]", text)
        self.assertNotIn("Input Shapes", self.prof.key_averages().table().splitlines()[1])

    def test_stacks_name_this_file(self):
        text = self.prof.key_averages(group_by_stack_n=2).table(row_limit=5)
        self.assertIn("Source Location", text.splitlines()[1])
        self.assertIn(os.path.basename(__file__), text)
        events = [e for e in self.prof.events() if e.stack]
        self.assertTrue(events)
        for entry in events[0].stack:
            self.assertRegex(entry, r"^.+\(\d+\): .+$")
            # jittor's own frames (its Torch layer here) are left out.
            self.assertNotIn(os.path.join("compat", "torch", ""), entry)
            self.assertNotIn(os.path.join("python", "jittor", ""), entry)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "stacks.txt")
            self.prof.export_stacks(path, "self_cuda_time_total")
            lines = open(path).read().splitlines()
        self.assertTrue(lines)
        for line in lines:
            frames, value = line.rsplit(" ", 1)
            self.assertGreater(int(value), 0)
            self.assertIn(os.path.basename(__file__), frames.split(";")[-1])

    def test_the_chrome_trace_is_kinetos(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.json.gz")
            self.prof.export_chrome_trace(path)
            with gzip.open(path, "rt") as f:
                trace = json.load(f)
        for key in ("schemaVersion", "deviceProperties", "with_stack", "profile_memory",
                    "record_shapes", "displayTimeUnit", "baseTimeNanoseconds", "traceEvents",
                    "traceName"):
            self.assertIn(key, trace)
        events = trace["traceEvents"]
        cats = {e.get("cat") for e in events}
        self.assertTrue({"cpu_op", "user_annotation", "kernel", "ac2g", "cpu_instant_event"} <= cats, cats)
        kernel = next(e for e in events if e.get("cat") == "kernel")
        for key in ("External id", "device", "stream", "correlation", "grid", "block",
                    "registers per thread", "shared memory"):
            self.assertIn(key, kernel["args"])
        starts = {e["id"] for e in events if e.get("ph") == "s"}
        ends = {e["id"] for e in events if e.get("ph") == "f"}
        self.assertEqual(starts, ends)
        labels = [e["args"]["labels"] for e in events if e.get("name") == "process_labels"]
        self.assertIn("CPU", labels)
        self.assertIn("GPU %d" % kernel["pid"], labels)

    def test_the_memory_timeline_is_pytorchs_shape(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "timeline.json")
            self.prof.export_memory_timeline(path, device="cuda:0")
            times, sizes = json.load(open(path))
        self.assertEqual(len(times), len(sizes))
        self.assertTrue(all(len(row) == 9 and row[0] == 0 for row in sizes))
        self.assertEqual(times, sorted(times))
        self.assertTrue(any(row[-1] > 0 for row in sizes))


if __name__ == "__main__":
    unittest.main()

"""``torch.profiler``/``torch.autograd.profiler`` record, and the memory
readings they sit next to mean what torch means.

Both profiler namespaces used to be inert: ``profile`` recorded nothing,
``key_averages`` was missing or empty and ``export_chrome_trace`` wrote no
file. ``max_memory_reserved`` returned the *allocated* high-water, a number
below ``memory_reserved()``.
"""

import json
import os
import tempfile
import unittest

import torch

from _helpers import capability as _test_capability

import jittor as jt

HAS_CUDA = bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)


def _model_and_step():
    model = torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.GELU(),
                                torch.nn.Linear(1024, 256)).cuda()
    opt = torch.optim.AdamW(model.parameters())
    x = torch.randn(64, 256, device="cuda")

    def step():
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    for _ in range(2):
        step()
    torch.cuda.synchronize()
    return step


@unittest.skipIf(not HAS_CUDA, "needs a CUDA device")
class TestTorchProfiler(unittest.TestCase):
    def test_profile_records_operators_and_exports_a_trace(self):
        step = _model_and_step()
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            with torch.profiler.record_function("train_step"):
                step()
        rows = prof.key_averages()
        names = [row.key for row in rows]
        self.assertIn("cublas_matmul", names)
        matmul = rows[names.index("cublas_matmul")]
        self.assertGreater(matmul.count, 0)
        self.assertGreater(matmul.cpu_time_total, 0)
        table = rows.table(sort_by="self_cpu_time_total", row_limit=5)
        self.assertIn("cublas_matmul", prof.key_averages().table(sort_by="count", row_limit=50))
        self.assertIn("Self CPU time total", table)
        path = os.path.join(tempfile.mkdtemp(), "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
        self.assertTrue(any(e.get("name") == "cublas_matmul" for e in events))
        self.assertTrue(any(e.get("name") == "train_step" for e in events))

    def test_schedule_saves_once_per_active_window(self):
        step = _model_and_step()
        saved = []
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)
        with torch.profiler.profile(schedule=schedule,
                                    on_trace_ready=lambda p: saved.append(len(p.events()))) as prof:
            for _ in range(6):
                step()
                prof.step()
        self.assertEqual(len(saved), 1)
        self.assertGreater(saved[0], 0)

    def test_autograd_profiler_records(self):
        step = _model_and_step()
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            step()
        self.assertTrue(len(prof.function_events) > 0)
        self.assertIsInstance(prof.key_averages(), torch.autograd.profiler.EventList)
        self.assertIn("cublas_matmul", prof.table(row_limit=50))


class TestTorchProfilerHost(unittest.TestCase):
    def test_disabled_profiler_returns_empty_event_list(self):
        with jt.flag_scope(use_cuda=0):
            with torch.profiler.profile(enabled=False) as prof:
                (torch.randn(8, 8) + 1).sum().item()
            with torch.autograd.profiler.profile(enabled=False) as aprof:
                (torch.randn(8, 8) + 1).sum().item()
        self.assertEqual(len(prof.events()), 0)
        self.assertEqual(len(prof.key_averages()), 0)
        self.assertEqual(prof.key_averages().table(), "")
        self.assertIsInstance(aprof.key_averages(), torch.autograd.profiler.EventList)
        self.assertEqual(len(aprof.key_averages()), 0)

    def test_exception_inside_profile_skips_on_trace_ready_and_cleans_up(self):
        called = []
        with jt.flag_scope(use_cuda=0):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                            on_trace_ready=lambda p: called.append(p)):
                    raise RuntimeError("boom")
        self.assertEqual(called, [])
        self.assertEqual(jt.core.step_trace_active(), 0)

    def test_negative_skip_first_wait_rejected(self):
        with self.assertRaises(ValueError):
            torch.profiler.schedule(wait=0, warmup=0, active=1, skip_first_wait=-1)

    def test_user_annotation_flag_and_demangle(self):
        from jittor.compat.torch import profiler_events
        self.assertEqual(profiler_events._demangle("_Z8kernel_0PfS_i"),
                         "kernel_0(float*, float*, int)")
        with jt.flag_scope(use_cuda=0):
            x = torch.randn(16, 16)
            x.sum().item()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        profile_memory=True) as prof:
                with torch.profiler.record_function("cpu_step"):
                    (x * 2 + 1).sum().item()
        events = prof.events()
        ann = [e for e in events if e.name == "cpu_step"]
        self.assertEqual(len(ann), 1)
        self.assertTrue(ann[0].is_user_annotation)
        self.assertTrue(all(not c.is_user_annotation for c in ann[0].cpu_children))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "trace.json")
            prof.export_chrome_trace(path)
            trace = json.load(open(path, encoding="utf-8"))["traceEvents"]
        for me in (e for e in trace if e.get("name") == "[memory]"):
            self.assertLessEqual(me["args"]["Total Allocated"], me["args"]["Total Reserved"], me)


@unittest.skipIf(not HAS_CUDA, "needs a CUDA device")
class TestTorchCudaMemoryReadings(unittest.TestCase):
    def test_max_memory_reserved_is_the_reserved_high_water(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        big = torch.empty(64 << 20, device="cuda")      # 256 MiB
        big.fill_(1.0)
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved()
        self.assertGreaterEqual(torch.cuda.max_memory_reserved(), reserved)
        self.assertGreaterEqual(torch.cuda.max_memory_allocated(), 256 << 20)
        stats = torch.cuda.memory_stats()
        self.assertGreaterEqual(stats["reserved_bytes.all.peak"], reserved)
        self.assertIn("| GPU reserved memory   |", torch.cuda.memory_summary())
        del big


if __name__ == "__main__":
    unittest.main()

# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""`jt.profile`: the step profile has to be right, not just present.

Each case checks a number against something that does not come from the
profiler itself: the host split against the wall clock, device kernel time
against a device-bound wall measurement, the traced memory peak against the
pools' own high-water counter, the live-at-peak set against a tensor the test
knows is live, and the graph-launch count against `GraphReplay.stats`.

The device-time cases need CUPTI (the ``nvidia-cuda-cupti-cu12`` wheel or
``$JITTOR_CUPTI_LIB``); they skip with that reason when it cannot be loaded.
"""

import json
import os
import tempfile
import unittest
import warnings

import jittor as jt
from jittor import nn
from jittor.profiling import _cupti

from _helpers import capability as _test_capability

HAS_CUDA = bool(_test_capability.check_accelerator("cuda", backend=jt).enabled)
NO_CUDA = "needs a CUDA device: the pools and kernels under test are the device's"
NO_CUPTI = "CUPTI (the CUDA profiling library) cannot be loaded; set JITTOR_CUPTI_LIB"
MiB = 1 << 20


def _cupti_loaded():
    return HAS_CUDA and _cupti.load() is not None


def _mlp_step():
    model = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))
    opt = jt.optim.AdamW(model.parameters(), lr=1e-3)
    x = jt.randn(128, 256)

    def step():
        opt.step((model(x) ** 2).mean())
    for _ in range(3):
        step()
    jt.sync_all(True)
    return step


class TestHostOnly(unittest.TestCase):
    def test_a_cpu_region_is_split_and_its_host_pool_peak_is_exact(self):
        with jt.flag_scope(use_cuda=0):
            model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64))
            opt = jt.optim.SGD(model.parameters(), lr=0.1)
            x = jt.randn(32, 64)

            def step():
                opt.step((model(x) ** 2).mean())
            step()
            jt.sync_all(True)
            with jt.profile() as prof:
                step()
                big = jt.randn(1024, 1024)      # 4 MiB on the host, live only here
                (big + 1).sync()
                del big
        self.assertIsNone(prof.device)
        self.assertTrue(prof.bound[0].startswith("host"))
        h = prof.host
        parts = (h["python_and_graph_build"] + h["executor_planning"] + h["executor_compile"]
                 + h["operator_launch"] + h["device_wait"] + h["graph_launch"])
        self.assertLess(abs(parts - prof.wall_ns), 0.02 * prof.wall_ns + 20000, h)
        mem = prof.memory
        self.assertEqual(mem.device, -1)
        self.assertEqual(mem.peak_allocated, jt.core.device_memory_peak(-1))
        self.assertGreaterEqual(mem.peak_allocated - mem.start[0], 4 * MiB)
        self.assertIn("runs on the host", prof.summary())


@unittest.skipIf(not HAS_CUDA, NO_CUDA)
class TestHostSplit(unittest.TestCase):
    def test_the_host_parts_sum_to_the_wall_time(self):
        with jt.flag_scope(use_cuda=1):
            step = _mlp_step()
            with jt.profile(device=False, memory=False) as prof:
                step()
        h = prof.host
        parts = (h["python_and_graph_build"] + h["executor_planning"] + h["executor_compile"]
                 + h["operator_launch"] + h["device_wait"] + h["graph_launch"])
        self.assertGreater(h["operators"], 5)
        self.assertGreaterEqual(h["batches"], 1)
        # The split is a partition of the wall time, not a sample of it.
        self.assertLess(abs(parts - prof.wall_ns), 0.02 * prof.wall_ns + 20000, h)
        for op in prof.op_records:
            self.assertLessEqual(op["t_start"], op["t_alloc"])
            self.assertLessEqual(op["t_alloc"], op["t_end"])
        names = {row.name for row in prof.op_stats()}
        self.assertIn("cublas_matmul", names)
        # Every operator carries the Python line that built it: this file.
        sites = prof.site_stats()
        self.assertTrue(any("test_step_profile.py" in site for site in sites), sorted(sites))

    def test_profiles_do_not_nest(self):
        with jt.flag_scope(use_cuda=1):
            with jt.profile(device=False, memory=False):
                with self.assertRaisesRegex(RuntimeError, "do not nest"):
                    jt.profile(device=False, memory=False).__enter__()

    def test_the_region_does_not_change_results(self):
        with jt.flag_scope(use_cuda=1):
            a = jt.randn(64, 64)
            b = jt.randn(64, 64)
            expected = (a @ b).numpy()
            with jt.profile():
                got = (a @ b).numpy()
        self.assertEqual(expected.tolist(), got.tolist())


@unittest.skipIf(not HAS_CUDA, NO_CUDA)
@unittest.skipIf(not _cupti_loaded(), NO_CUPTI)
class TestDeviceTime(unittest.TestCase):
    def test_kernels_are_attributed_to_the_operators_that_launched_them(self):
        with jt.flag_scope(use_cuda=1):
            step = _mlp_step()
            with jt.profile(memory=False) as prof:
                step()
        d = prof.device
        self.assertGreater(d["kernels"], 5)
        self.assertLessEqual(d["busy"], prof.wall_ns)
        self.assertGreaterEqual(d["attributed"], 0.95 * (d["kernel"] + d["copy"]))
        matmul = [row for row in prof.op_stats() if row.name == "cublas_matmul"]
        self.assertTrue(matmul and matmul[0].device_ns > 0 and matmul[0].kernels > 0)

    def test_device_time_agrees_with_a_device_bound_wall_clock(self):
        # 16 back-to-back 2048^3 matmuls keep the device saturated, so the wall
        # time per matmul is its device time plus a sliver of launch overhead.
        import time
        n = 16
        with jt.flag_scope(use_cuda=1):
            a = jt.randn(2048, 2048)
            b = jt.randn(2048, 2048)
            for _ in range(3):
                (a @ b).sync()
            jt.sync_all(True)
            outs = []
            start = time.perf_counter()
            for _ in range(n):
                outs.append(a @ b)
            jt.sync(outs)
            jt.sync_all(True)
            wall = (time.perf_counter() - start) / n
            with jt.profile(memory=False) as prof:
                outs = [a @ b for _ in range(n)]
                jt.sync(outs)
        rows = [row for row in prof.op_stats() if row.name == "cublas_matmul"]
        self.assertEqual(rows[0].calls, n)
        per_call = rows[0].device_ns / n / 1e9
        self.assertGreater(per_call, 0.5 * wall, (per_call, wall))
        self.assertLess(per_call, 1.1 * wall, (per_call, wall))

    def test_chrome_trace_has_host_operators_and_device_kernels(self):
        with jt.flag_scope(use_cuda=1):
            step = _mlp_step()
            with jt.profile() as prof:
                step()
        path = os.path.join(tempfile.mkdtemp(), "step.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
        host = [e for e in events if e.get("ph") == "X" and e["pid"] == 1 and e["tid"] == 2]
        device = [e for e in events if e.get("ph") == "X" and e["pid"] == 2]
        self.assertEqual(len(host), prof.host["operators"])
        self.assertEqual(len(device), prof.device["kernels"] + prof.device["copies"])
        self.assertTrue(any(e["name"] == "device memory (MiB)" for e in events))


@unittest.skipIf(not HAS_CUDA, NO_CUDA)
class TestMemory(unittest.TestCase):
    def test_the_traced_peak_is_the_pools_peak_and_names_what_was_live(self):
        with jt.flag_scope(use_cuda=1):
            x = jt.randn(1024, 1024)
            jt.sync_all(True)
            with jt.profile(device=False) as prof:
                big = jt.randn(8192, 8192)      # 256 MiB, live only inside the region
                (big * 2).sum().sync()
                del big
        mem = prof.memory
        self.assertEqual(mem.peak_allocated, jt.core.device_memory_peak(jt.core.current_device()))
        self.assertGreaterEqual(mem.peak_allocated - mem.start[0], 256 * MiB)
        tensors = dict((label, nbytes) for label, nbytes, _ in mem.live_by("tensor"))
        self.assertGreaterEqual(tensors.get("float32[8192,8192]", 0), 256 * MiB, tensors)
        sites = mem.live_by("site")
        self.assertTrue(any("test_step_profile.py" in label and nbytes >= 256 * MiB
                            for label, nbytes, _ in sites), sites)
        # What was already live is reported as such, not lost.
        origins = dict((label, nbytes) for label, nbytes, _ in mem.live_by("origin"))
        self.assertGreaterEqual(origins.get("allocated before profile", 0), 4 * MiB)
        self.assertGreaterEqual(mem.pool_peak_reserved, mem.peak_allocated)
        del x

    def test_the_reserved_high_water_is_counted_and_reset(self):
        with jt.flag_scope(use_cuda=1):
            device = jt.core.current_device()
            jt.sync_all(True)
            jt.gc()
            jt.core.reset_device_memory_peak(device)
            reserved = jt.core.device_memory_reserved(device)
            self.assertEqual(jt.core.device_memory_reserved_peak(device), reserved)
            big = jt.randn(8192, 8192)
            big.sync()
            grown = jt.core.device_memory_reserved(device)
            self.assertGreaterEqual(grown - reserved, 256 * MiB)
            del big
            jt.sync_all(True)
            jt.gc()
            self.assertGreaterEqual(jt.core.device_memory_reserved_peak(device), grown)
            self.assertLess(jt.core.device_memory_reserved(device), grown)

    def test_a_new_legacy_memory_session_starts_a_new_record(self):
        # `profile_memory_enable` used to keep the previous session's peak, so
        # a second, smaller session reported the first one's numbers.
        with jt.flag_scope(use_cuda=1):
            with jt.flag_scope(profile_memory_enable=1):
                (jt.randn(8192, 8192) * 2).sync()
                first = jt.core.get_peak_device_used_memory(0)
            jt.sync_all(True)
            jt.gc()
            with jt.flag_scope(profile_memory_enable=1):
                (jt.randn(256, 256) * 2).sync()
                second = jt.core.get_peak_device_used_memory(0)
        self.assertGreaterEqual(first - second, 256 * MiB, (first, second))


@unittest.skipIf(not HAS_CUDA, NO_CUDA)
class TestReplay(unittest.TestCase):
    def _replay(self):
        m = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
        m.eval()
        x = jt.randn(32, 256)
        replay = jt.graph_replay(m, x)
        for _ in range(6):
            replay(x).sync()
        jt.sync_all(True)
        if not replay.stats["graph"]:
            self.skipTest("this build did not record a device graph: %s" % replay._graph_refused)
        return replay, x

    def test_a_replayed_step_shows_its_graph_launch(self):
        with jt.flag_scope(use_cuda=1):
            replay, x = self._replay()
            with jt.profile(memory=False) as prof:
                replay(x).sync()
        self.assertEqual(prof.replay["graph_launches"], 1)
        launches = [row for row in prof.op_stats() if row.name == "cuda_graph_launch"]
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0].calls, 1)
        deltas = [delta for name, delta, _ in prof.replay["sources"] if name.startswith("GraphReplay")]
        self.assertEqual(deltas, [{"captured": 0, "replayed": 1, "rebuilt": 0, "graph": 1}])
        if prof.device is not None:
            self.assertGreater(launches[0].device_ns, 0)
            self.assertGreater(launches[0].kernels, 0)
        self.assertIn("device-graph launches", prof.summary())

    def test_profile_scope_names_the_graph_launch_it_cannot_see(self):
        with jt.flag_scope(use_cuda=1):
            replay, x = self._replay()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with jt.profile_scope():
                    replay(x).sync()
                    jt.sync_all(True)
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("device-graph" in m and "jt.profile" in m for m in messages), messages)


if __name__ == "__main__":
    unittest.main()

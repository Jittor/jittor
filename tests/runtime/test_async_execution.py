"""Batches executed on a worker thread (src/runtime/async_exec.h).

Under ``no_grad`` on CUDA -- and on ACL, where the worker also binds the
device the batch was posted from and takes a training forward as well -- an
auto-flushed batch runs on one worker thread while Python keeps building the
graph. What has to hold: the numbers are the ones a
batch on the calling thread gives, every read of a Var's data sees the
worker's writes, a flag assignment does not overtake a batch that reads flags,
a gradient is handed over only where the backend's policy says a flush costs
a training step nothing, and a second Python thread building meanwhile
neither deadlocks nor corrupts the graph.

Each case says whether the worker took a batch (``async_exec_batches``); where
this process cannot place a worker -- no second core in the calling thread's
last-level cache domain -- the cases that need one skip.
"""

import time
import unittest

import numpy as np

import jittor as jt
from _helpers import capability as _test_capability
from _helpers.child_process import run_child_script


def _worker_backend():
    """The accelerator the worker runs on here, or None.

    ACL takes a batch only under ``backend_fallback=error`` (see
    `may_run_async`), so the cases set that policy there.
    """
    for name in ("cuda", "acl"):
        if _test_capability.check_accelerator(name, backend=jt).enabled:
            return name
    return None


def _chain(x, w, steps):
    # Enough operators to cross the flush count several times, with
    # elementwise runs (fused) between products. A flush takes the results
    # Python holds that nothing reads yet and that are written out -- never one
    # an elementwise operator is still to compute -- so every step also keeps
    # a product of its own, which any flush can take.
    kept = []
    for i in range(steps):
        y = jt.tanh(x * 0.5 + 0.25) + x * 0.125
        kept.append(jt.matmul(y, w))
        x = jt.matmul(y, w)
    return x + sum(k[:1, :1] for k in kept[-3:])


_FORK_MID_BATCH = """
import subprocess
import threading
import numpy as np
import jittor as jt

jt.flags.use_cuda = 1
jt.flags.auto_graph_replay = 0
jt.flags.auto_flush_ops = 23
jt.flags.backend_fallback = "error"
rng = np.random.RandomState(7)
x0 = jt.array(rng.randn(64, 64).astype("float32"))
w = jt.array((rng.randn(64, 64) / 8).astype("float32"))
jt.sync_all(True)

def chain(x, steps):
    for i in range(steps):
        y = jt.tanh(x * 0.5 + 0.25) + x * 0.125
        x = jt.matmul(y, w)
    return x

forked = []

def fork_now():
    # Nothing of jittor's: this thread only forks, the way a data loader or a
    # `nvidia-smi`/`npu-smi` sampler does.
    forked.append(subprocess.call(["true"]))

with jt.no_grad():
    before = jt.core.async_exec_batches()
    x = chain(x0, 60)
    thread = threading.Thread(target=fork_now)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), "the fork handler did not return"
    ran = jt.core.async_exec_batches() - before
    got = chain(x, 60).numpy()
with jt.flag_scope(async_execution=0), jt.no_grad():
    want = chain(chain(x0, 60), 60).numpy()
assert forked == [0], forked
assert np.array_equal(got, want)
print("FORK_OK", ran)
"""

_TWO_THREADS = """
import threading
import numpy as np
import jittor as jt

jt.flags.use_cuda = 1
jt.flags.auto_graph_replay = 0
jt.flags.auto_flush_ops = 23
jt.flags.backend_fallback = "error"
rng = np.random.RandomState(7)
x0 = jt.array(rng.randn(64, 64).astype("float32"))
w = jt.array((rng.randn(64, 64) / 8).astype("float32"))
jt.sync_all(True)

def chain(x, steps):
    kept = []
    for i in range(steps):
        y = jt.tanh(x * 0.5 + 0.25) + x * 0.125
        kept.append(jt.matmul(y, w))
        x = jt.matmul(y, w)
    return x

with jt.flag_scope(async_execution=0), jt.no_grad():
    want = chain(chain(x0, 60), 60).numpy()
result = []

def other():
    result.append(float(jt.array(np.arange(4, dtype="float32")).sum().item()))

with jt.no_grad():
    before = jt.core.async_exec_batches()
    half = chain(x0, 60)
    # A batch is likely in flight now: the second thread's bindings take the
    # graph lock while it runs, and its sync waits for it.
    thread = threading.Thread(target=other)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), "the second thread did not finish"
    middle = jt.core.async_exec_batches()
    got = chain(half, 60).numpy()
    after = jt.core.async_exec_batches()
assert result == [6.0], result
assert np.array_equal(got, want)
# The worker stands down once a second thread has used jittor.
assert after == middle, (before, middle, after)
print("TWO_THREADS_OK", middle - before)
"""


_MIGRATE_ALIASES = """
import ctypes
import threading
import time
import numpy as np
import jittor as jt

jt.flags.backend_fallback = "error"
jt.flags.auto_graph_replay = 0
jt.flags.auto_flush_ops = 23
jt.flags.async_execution = 1
jt.flags.async_min_caller_us = 0
with jt.flag_scope(use_cuda=0):
    host = jt.array(np.arange(12, dtype="int32").reshape(3, 4))
    alias = host._storage_permute((1, 0))
    alias.sync()
assert host.location() == alias.location() == "cpu"
assert host._storage_address == alias._storage_address
jt.flags.use_cuda = 1
x = jt.array(np.eye(64, dtype="float32"))
w = jt.array(np.eye(64, dtype="float32"))
jt.sync_all(True)
fallbacks = jt.core.backend_fallback_count()
result = []
errors = []

def migrate():
    try:
        result.append(alias.device_ptr_ready)
    except BaseException as error:
        errors.append(repr(error))

with jt.no_grad():
    before = jt.core.async_exec_batches()
    for _ in range(60):
        # Keep this batch entirely on CUDA: a native integer scalar add
        # introduces host work, which correctly prevents worker admission.
        x = jt.matmul(x, w)
    jt.submit_pending(x)
    deadline = time.monotonic() + 30
    while jt.core.async_exec_batches() == before and time.monotonic() < deadline:
        time.sleep(.001)
    thread = threading.Thread(target=migrate)
    thread.start()
    thread.join(30)
    assert not thread.is_alive(), "alias migration did not finish"
    x.sync()
    ran = jt.core.async_exec_batches() - before
assert not errors, errors
assert len(result) == 1 and result[0]
assert host.location() == alias.location() == "device"
assert host._storage_address == alias._storage_address == result[0]
library = ctypes.CDLL("libcuda.so.1")
device = ctypes.c_int(-1)
assert library.cuPointerGetAttribute(ctypes.byref(device), 9,
        ctypes.c_void_p(result[0])) == 0
assert device.value == 0
np.testing.assert_array_equal(alias.numpy(), np.arange(12).reshape(3, 4).T)
np.testing.assert_array_equal(host.numpy(), np.arange(12).reshape(3, 4))
assert jt.core.backend_fallback_count() == fallbacks
print("MIGRATE_ALIASES_OK", ran)
"""


@unittest.skipIf(not _worker_backend(), "No accelerator the worker runs on")
class TestAsyncExecution(unittest.TestCase):
    def setUp(self):
        # A prime count, so the crossings fall on different operators of the
        # chain and some on a product.
        self.flags = jt.flag_scope(use_cuda=1, async_execution=1, auto_flush_ops=23,
                                   auto_graph_replay=0, backend_fallback="error")
        self.flags.__enter__()
        rng = np.random.RandomState(7)
        self.x = jt.array(rng.randn(64, 64).astype("float32"))
        self.w = jt.array((rng.randn(64, 64) / 8).astype("float32"))
        jt.sync_all(True)

    def tearDown(self):
        jt.sync_all(True)
        self.flags.__exit__(None, None, None)

    def _run(self, async_on, steps=40):
        with jt.flag_scope(async_execution=async_on), jt.no_grad():
            before = jt.core.async_exec_batches()
            out = _chain(self.x, self.w, steps).numpy()
            return out, jt.core.async_exec_batches() - before

    def _need_worker(self, ran):
        if not ran:
            self.skipTest("no core to place the worker on next to this thread")

    def test_the_worker_gives_the_calling_thread_s_numbers(self):
        want, ran_sync = self._run(0)
        got, ran = self._run(1)
        self.assertEqual(ran_sync, 0)
        self._need_worker(ran)
        np.testing.assert_array_equal(got, want)

    @unittest.skipUnless(_worker_backend() == "cuda", "requires CUDA streams")
    def test_direct_storage_copy_is_published_to_the_next_worker(self):
        # Delay only the framework's copy stream. The worker's compute stream
        # can finish its batch, and the calling thread can queue the D2D copy,
        # while those bytes are still pending. No host/device-wide wait is the
        # fix: the next worker must inherit the copy's completion event.
        header = r'''
        #include "runtime/backend_streams.h"
        #include "runtime/device.h"
        __global__ void delay_copy(unsigned long long cycles) {
            auto started = clock64();
            while (clock64() - started < cycles) {}
        }
        __global__ void mark_ready(int* output) { output[0] = 1; }
        '''
        source = r'''
        auto stream = jittor::accelerator_backend_stream(
            jittor::current_device(), jittor::BackendStreamKind::Copy);
        delay_copy<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(200000000);
        mark_ready<<<1, 1>>>(out0_p);
        '''
        with jt.flag_scope(auto_flush_ops=1, auto_flush_bytes=0,
                           async_flush_ops=1), jt.no_grad():
            target = jt.zeros((4,), dtype="int32")
            values = jt.array(np.arange(4, dtype="int32"))
            jt.sync_all(True)
            before = jt.core.async_exec_batches()
            ready = jt.code((1,), "int32", [], cuda_header=header, cuda_src=source)
            jt.submit_pending(ready)
            deadline = time.monotonic() + 30
            while jt.core.async_exec_batches() == before and time.monotonic() < deadline:
                time.sleep(.001)
            self._need_worker(jt.core.async_exec_batches() - before)
            target._copy_into(values)
            # Native clone shares storage. Use copy so the worker actually
            # reads the destination bytes into an independent allocation.
            result = target.copy()
            jt.submit_pending(result)
            result.sync()
            self.assertEqual(result.location(), "device")
            np.testing.assert_array_equal(result.numpy(), np.arange(4, dtype="int32"))

    @unittest.skipUnless(_worker_backend() == "cuda", "requires CUDA graphs")
    def test_graph_launch_is_published_to_the_next_worker(self):
        # The recorded kernel writes only legal integers. Its delay leaves
        # enough time for an unordered worker copy to read the preceding zeros.
        header = r'''
        __global__ void delayed_range(int* output) {
            auto started = clock64();
            while (clock64() - started < 200000000) {}
            for (int i = 0; i < 4; ++i) output[i] = i + 1;
        }
        '''
        graph = 0
        try:
            with jt.flag_scope(auto_flush_ops=0), jt.no_grad():
                output = jt.zeros((4,), dtype="int32")
                output.sync()
                with jt.flag_scope(keep_graph=2, async_execution=0):
                    computed = jt.code((4,), "int32", [], cuda_header=header,
                                       cuda_src="delayed_range<<<1, 1>>>(out0_p);")
                    computed.sync()
                    self.assertTrue(jt.core.graph_capture_begin())
                    try:
                        computed.sync(False, False)
                        # The compiled-step recorder uses this same stable
                        # preallocated-output pattern. Reading output back here
                        # would migrate it and invalidate the recorded address.
                        output._copy_into(computed, False)
                    finally:
                        graph = jt.core.graph_capture_end()
                    self.assertNotEqual(graph, 0)
                del computed
                jt.sync_all(True)
                pointer = output._storage_address
                self.assertEqual(output.location(), "device")
                dummy = jt.ones((4,), dtype="int32")
                dummy.sync()
                with jt.flag_scope(auto_flush_ops=1, auto_flush_bytes=0,
                                   async_flush_ops=1):
                    before = jt.core.async_exec_batches()
                    establish = dummy.copy()
                    jt.submit_pending(establish)
                    deadline = time.monotonic() + 30
                    while jt.core.async_exec_batches() == before and time.monotonic() < deadline:
                        time.sleep(.001)
                    self._need_worker(jt.core.async_exec_batches() - before)
                    self.assertEqual(output._storage_address, pointer)
                    jt.core.graph_launch(graph)
                    result = output.copy()
                    jt.submit_pending(result)
                    result.sync()
                    self.assertEqual(result.location(), "device")
                    self.assertNotEqual(result._storage_address, pointer)
                    np.testing.assert_array_equal(result.numpy(), np.arange(1, 5, dtype="int32"))
        finally:
            if graph:
                # Cleanup waits after the observation, including on a failed
                # assertion, so the captured buffers outlive their launches.
                jt.core.graph_wait()
                jt.core.graph_release(graph)

    def test_intermediates_read_back_while_the_worker_runs(self):
        with jt.no_grad():
            before = jt.core.async_exec_batches()
            values = []
            x = self.x
            for i in range(30):
                x = _chain(x, self.w, 2)
                if i % 7 == 3:
                    values.append(x.numpy())
            self._need_worker(jt.core.async_exec_batches() - before)
        with jt.flag_scope(async_execution=0), jt.no_grad():
            x = self.x
            want = []
            for i in range(30):
                x = _chain(x, self.w, 2)
                if i % 7 == 3:
                    want.append(x.numpy())
        for got, ref in zip(values, want):
            np.testing.assert_array_equal(got, ref)

    def _train(self, async_on, steps=20):
        w = jt.array(self.w.numpy())
        with jt.flag_scope(async_execution=async_on, async_min_caller_us=0):
            before = jt.core.async_exec_batches()
            out = _chain(self.x, w, steps)
            grad = jt.grad(out.sum(), w)
            grad.sync()
            return grad.numpy(), jt.core.async_exec_batches() - before

    def test_a_training_forward_goes_to_the_worker_only_where_it_is_free(self):
        """Whether training is handed over is the backend's call.

        An auto-flush costs a training step an intermediate written out
        instead of fused into its readers; a backend that fuses nothing loses
        nothing by it and sets `async_admits_grad`.
        """
        want, ran_sync = self._train(0)
        got, ran = self._train(1)
        self.assertEqual(ran_sync, 0)
        if _worker_backend() == "acl":
            self._need_worker(ran)
        else:
            self.assertEqual(ran, 0)
        np.testing.assert_array_equal(got, want)

    def test_a_python_owned_tape_released_by_the_worker(self):
        """A `jt.Function`'s backward is a Python callable the tape owns.

        The worker has no GIL, so dropping the last reference to one as a
        batch winds down used to be a segfault rather than a refcount; the
        release goes back to this thread instead
        (`async_exec_defer_python_release`). Training on ACL builds these
        constantly -- its convolutions and norms are `jt.Function`s.
        """
        class Scale(jt.Function):
            def execute(self, x):
                self.saved = x
                return x * 2.0

            def grad(self, grad_output):
                return grad_output * 2.0

        def run(async_on):
            w = jt.array(self.w.numpy())
            with jt.flag_scope(async_execution=async_on, async_min_caller_us=0):
                before = jt.core.async_exec_batches()
                x = self.x
                for _ in range(12):
                    x = Scale()(_chain(x, w, 3))
                grad = jt.grad(x.sum(), w)
                grad.sync()
                return grad.numpy(), jt.core.async_exec_batches() - before

        want, _ = run(0)
        got, ran = run(1)
        if _worker_backend() == "acl":
            self._need_worker(ran)
        np.testing.assert_array_equal(got, want)
        # The deferred releases are run on this thread at the next executor
        # entry; reaching here without a crash is the assertion.
        jt.sync_all(True)

    def test_a_flag_assignment_waits_for_the_batch(self):
        with jt.no_grad():
            before = jt.core.async_exec_batches()
            out = _chain(self.x, self.w, 40)
            ran = jt.core.async_exec_batches() - before
            # Whatever is in flight finishes before the flag moves.
            with jt.flag_scope(use_cuda=1, reuse_dying_inputs=jt.flags.reuse_dying_inputs ^ 1):
                pass
            got = out.numpy()
        self._need_worker(ran or jt.core.async_exec_batches() - before)
        want, _ = self._run(0)
        np.testing.assert_array_equal(got, want)

    def _chain_peak(self, async_on, steps=60):
        # Each product is 8 MiB and read once, by the next one: what a batch
        # keeps of them is how promptly it lets go of what it has used last.
        rng = np.random.RandomState(3)
        x0 = jt.array(rng.randn(2048, 1024).astype("float32") / 32)
        w = jt.array((np.eye(1024) + rng.randn(1024, 1024) / 64).astype("float32"))
        jt.sync_all(True)
        with jt.flag_scope(async_execution=async_on), jt.no_grad():
            before = jt.core.async_exec_batches()
            live = jt.core._device_memory_window_start(0)
            x = x0
            for _ in range(steps):
                x = jt.matmul(jt.tanh(x), w)
            got = x.numpy()
            peak = jt.core._device_memory_window_peak(0) - live
            return got, peak, jt.core.async_exec_batches() - before

    def test_the_worker_lets_go_of_what_a_batch_has_used_as_it_goes(self):
        # The worker applies the graph's bookkeeping behind its launches, and
        # what that holds back is bounded by `async_release_lag_bytes`; a batch
        # that kept everything it used until its end would hold ~23 products.
        want, sync_peak, _ = self._chain_peak(0)
        got, peak, ran = self._chain_peak(1)
        self._need_worker(ran)
        np.testing.assert_array_equal(got, want)
        # The lag, one more product, and a megabyte for the per-allocation
        # padding a backend adds (ACL pads to 32 bytes, and the worker has
        # more blocks live at once than the calling thread does). A batch
        # that let go of nothing would be at ~184 MB, not ~41 MB.
        self.assertLessEqual(peak,
                             sync_peak + jt.flags.async_release_lag_bytes + (9 << 20),
                             (peak, sync_peak))

    def test_a_second_thread_may_fork_while_a_batch_is_in_flight(self):
        # `subprocess.Popen` forks with the GIL held, and the fork handler
        # waits for the batch. Waiting there while holding the GIL -- or the
        # graph lock -- deadlocks three ways: the worker wants the graph lock
        # between its segments, the thread holding it wants the GIL back, and
        # the GIL is held by the thread doing the waiting.
        done = run_child_script(_FORK_MID_BATCH, text=True, merge_stderr=True,
                                name="async_fork_mid_batch", timeout=600)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        self.assertIn("FORK_OK", done.stdout)

    def test_a_second_python_thread_may_build_meanwhile(self):
        # In a child: a second thread's bindings stand the worker down for the
        # rest of the process, which would leave the other cases nothing to
        # test.
        done = run_child_script(_TWO_THREADS, text=True, merge_stderr=True,
                                name="async_two_threads", timeout=600)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        self.assertIn("TWO_THREADS_OK", done.stdout)

    @unittest.skipUnless(_worker_backend() == "cuda", "requires CUDA migration")
    def test_a_second_thread_can_migrate_a_host_alias_group(self):
        # A migration waits for the worker before locking the alias ring.
        # Keep the second Python thread in a child: using a second binding
        # thread deliberately stands the worker down for that process.
        done = run_child_script(_MIGRATE_ALIASES, text=True, merge_stderr=True,
                                name="async_migrate_aliases", timeout=120)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        self.assertIn("MIGRATE_ALIASES_OK", done.stdout)
        ran = int(done.stdout.rsplit("MIGRATE_ALIASES_OK ", 1)[1].split()[0])
        self._need_worker(ran)


if __name__ == "__main__":
    unittest.main()

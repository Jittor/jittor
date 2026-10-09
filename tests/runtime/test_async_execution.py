"""Inference batches executed on a worker thread (src/runtime/async_exec.h).

Under ``no_grad`` on CUDA an auto-flushed batch runs on one worker thread while
Python keeps building the graph. What has to hold: the numbers are the ones a
batch on the calling thread gives, every read of a Var's data sees the
worker's writes, a flag assignment does not overtake a batch that reads flags,
anything that needs a gradient stays on the calling thread, and a second
Python thread building meanwhile neither deadlocks nor corrupts the graph.

Each case says whether the worker took a batch (``async_exec_batches``); where
this process cannot place a worker -- no second core in the calling thread's
last-level cache domain -- the cases that need one skip.
"""

import unittest

import numpy as np

import jittor as jt
from _helpers import capability as _test_capability
from _helpers.child_process import run_child_script


def _cuda():
    return _test_capability.check_accelerator("cuda", backend=jt).enabled


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


_TWO_THREADS = """
import threading
import numpy as np
import jittor as jt

jt.flags.use_cuda = 1
jt.flags.auto_graph_replay = 0
jt.flags.auto_flush_ops = 23
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


@unittest.skipIf(not _cuda(), "No CUDA found")
class TestAsyncExecution(unittest.TestCase):
    def setUp(self):
        # A prime count, so the crossings fall on different operators of the
        # chain and some on a product.
        self.flags = jt.flag_scope(use_cuda=1, async_execution=1, auto_flush_ops=23,
                                   auto_graph_replay=0)
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

    def test_training_stays_on_the_calling_thread(self):
        w = jt.array(self.w.numpy())
        before = jt.core.async_exec_batches()
        out = _chain(self.x, w, 20)
        grad = jt.grad(out.sum(), w)
        grad.sync()
        self.assertEqual(jt.core.async_exec_batches(), before)
        self.assertTrue(np.isfinite(grad.numpy()).all())

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
        self.assertLessEqual(peak, sync_peak + jt.flags.async_release_lag_bytes + (8 << 20),
                             (peak, sync_peak))

    def test_a_second_python_thread_may_build_meanwhile(self):
        # In a child: a second thread's bindings stand the worker down for the
        # rest of the process, which would leave the other cases nothing to
        # test.
        done = run_child_script(_TWO_THREADS, text=True, merge_stderr=True,
                                name="async_two_threads", timeout=600)
        self.assertEqual(done.returncode, 0, done.stdout[-3000:])
        self.assertIn("TWO_THREADS_OK", done.stdout)


if __name__ == "__main__":
    unittest.main()

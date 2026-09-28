# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A graph one thread issued, finished by another.

The compute stream is the backend's per-thread default stream -- on CUDA,
`cudaStreamPerThread` -- so each thread that issues work gets a different one.
The graph, the Vars and their buffers are process-global and carry no stream
affinity, so without an explicit handoff the second thread's operators are
free to run while the first thread's are still writing the buffers they read.
Nothing reports it: no error, no warning, just wrong numbers.

The shape below is the one a serving loop produces. One thread issues a heavy
prefix and does *not* wait for the device (`sync()` defaults to
`device_sync=False`, which is exactly what a lazy graph does when it flushes).
Another thread then evaluates the rest, which reads that prefix. The prefix has
to be big enough to still be in flight when the second thread gets there --
hence the matmul chain, which is milliseconds of device time against the tens
of microseconds it takes to hand a Var through a queue.

Measured on the H3 video VAE before the handoff existed: every frame of every
decode came back around 4e7 where the answer ranges over +-6, and at longer
clip lengths the whole tensor was NaN. `CUDA_LAUNCH_BLOCKING=1` alone restored
it, which is what identified the asynchrony as the mechanism.

The comparison is against the *same-thread* answer, not against a fixed
constant: this arithmetic is not bit-reproducible run to run, so only the gap
between one thread and two means anything.
"""
import queue
import threading
import unittest

import numpy as np

import jittor as jt


def _chain(src):
    a = jt.array(src)
    y = a
    for _ in range(40):
        y = jt.matmul(y, a) * 0.5 + 0.01
    return y


@unittest.skipIf(not jt.has_cuda, "no CUDA device")
class TestCrossThreadComputeStream(unittest.TestCase):
    def test_second_thread_finishes_a_graph_the_first_issued(self):
        with jt.flag_scope(use_cuda=1):
            rs = np.random.RandomState(0)
            src = (rs.randn(1024, 1024) * 0.05).astype("float32")

            def build():
                head = _chain(src)
                # Issue the prefix and return without waiting for the device,
                # leaving it in flight on *this* thread's compute stream.
                head.sync()
                # Built here, deliberately not executed: whoever fetches it
                # runs these, and they read `head`.
                return (head * 2.0 + 1.0).sqrt()

            reference = build().numpy()

            work, done = queue.Queue(), queue.Queue()

            def consumer():
                while True:
                    var = work.get()
                    if var is None:
                        break
                    done.put(var.numpy())

            thread = threading.Thread(target=consumer, daemon=True)
            thread.start()
            try:
                worst = 0.0
                for _ in range(8):
                    work.put(build())
                    got = done.get()
                    self.assertFalse(np.isnan(got).any(),
                                     "cross-thread evaluation produced NaN")
                    scale = max(1e-9, float(np.abs(reference).max()))
                    worst = max(worst, float(np.abs(got - reference).max()) / scale)
            finally:
                work.put(None)
                thread.join()

            # Same-thread run-to-run spread, measured on this build rather than
            # assumed, so the bound cannot quietly drift with the arithmetic.
            spread = 0.0
            scale = max(1e-9, float(np.abs(reference).max()))
            for _ in range(4):
                again = build().numpy()
                spread = max(spread, float(np.abs(again - reference).max()) / scale)

            self.assertLessEqual(
                worst, max(spread * 4, 1e-5),
                "a graph issued on one thread and finished on another differs "
                "from the same-thread answer by more than that answer varies "
                "between runs (cross-thread %.6g, same-thread %.6g)"
                % (worst, spread))


if __name__ == "__main__":
    unittest.main()

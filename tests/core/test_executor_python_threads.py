# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Driving jittor from more than one Python thread.

`use_threading` is declared as "Allow to use python threading with jittor", so
this is a configuration jittor claims to support. It does not: with a held
non-sink Var being built on one thread while others are inside the executor, the
process segfaults inside `parallel_compile_all_ops` in roughly every run, with
no CUDA involved and in a few seconds.

Why a subprocess. The failure is a **segfault**, so it cannot be caught in
process -- an assertion here would take the whole test session with it. The
child is given a short deadline and the parent reads its exit status.

Why this is skipped by default. It is a known-failing case, not a regression
guard: it documents the bug and gives a fast way to tell whether a change moves
it. Run it with `JT_TEST_THREAD_RACE=1`.

Cost, for whoever runs it: about 10s per attempt, of which ~8s is importing
jittor. See `docs/results/2026-09-14-vllm-omni-h3-enablement.md` for what this
race does to a real workload -- there it is silent wrong data rather than a
crash.
"""
import os
import textwrap
import unittest

from _helpers.child_process import run_python_child


CHILD = textwrap.dedent("""
    import threading, time, sys
    import jittor as jt
    jt.flags.use_cuda = 0
    N, SECS = 16, float(sys.argv[1])
    stop = threading.Event()

    def rival():
        while not stop.is_set():
            try:
                x = jt.random((N, N)).float32()
                y = x.sum()
                jt.sync_all(True)
                del x, y
            except Exception:
                return

    for i in range(3):
        threading.Thread(target=rival, daemon=True).start()

    t0, rounds = time.time(), 0
    while time.time() - t0 < SECS:
        a = jt.random((N, N)).float32()
        mid = a * 2.0 + 1.0
        for _ in range(4):
            mid = mid * 1.01 + 0.5
        sink = mid.sum()        # an output edge, so `mid` is not a sink
        jt.sync_all(True)
        rounds += 1
        del a, mid, sink
    stop.set()
    print("SURVIVED", rounds)
""")


@unittest.skipUnless(os.environ.get("JT_TEST_THREAD_RACE"),
                     "known-failing; set JT_TEST_THREAD_RACE=1 to run")
class TestExecutorUnderPythonThreads(unittest.TestCase):

    def _run(self, secs):
        # Through the shared helper, not a hand-rolled PYTHONPATH. This test
        # launches the interpreter, and every such launch has to import the
        # tree *this session* imports -- the hand-rolled version pinned the
        # repository root rather than `python/`, so a child could have imported
        # an installed jittor instead (tests/structure/test_child_process_contract).
        # stderr is merged because a segfault's message is on whichever stream
        # the runtime chose; the exit status is what the case asserts on.
        return run_python_child(["-c", CHILD, str(secs)],
                                timeout=secs + 180, merge_stderr=True)

    def test_a_held_non_sink_var_survives_other_threads_in_the_executor(self):
        proc = self._run(8.0)
        self.assertGreaterEqual(
            proc.returncode, 0,
            "jittor died on signal %d while three other Python threads were "
            "inside the executor. `use_threading` advertises this "
            "configuration. Last child output:\n%s"
            % (-proc.returncode, proc.stdout[-2000:]))


if __name__ == "__main__":
    unittest.main()

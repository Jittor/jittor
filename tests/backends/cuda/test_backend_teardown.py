# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Library handle teardown reports failures; it never aborts the process.

Every ``*_wrapper.cc`` destroyed its library handle from a static destructor
using ``checkCudaErrors``, which ``LOGf``s, which throws.  Destructors are
noexcept, so a teardown the library refuses called ``std::terminate`` and the
process died on SIGABRT.

That matters because of *when* a teardown gets refused: after a CUDA error has
already made the context sticky-invalid.  So the run that has something to say
is exactly the run that aborts on the way out, and the last thing it prints is
``terminate called after throwing an instance of 'std::runtime_error' ...
CUDNN_STATUS_INTERNAL_ERROR cudnnDestroy`` -- a message about the cleanup,
standing in for the message about the fault.  The test below reproduces that
sequence and asserts the real error is the one that survives.

Note when re-checking this against the old code: the child's SIGABRT used to
take the *runner* with it.  jittor installs a SIGCHLD handler that
quick_exit(1)s the parent whenever a direct child dies other than by a clean
exit or SIGTERM (``utils/log.cc``), so a pre-fix run of this file ended as
``pytest`` exiting 1 after one dot, not as a reported failure (6.C31).
``crash_isolated=True`` below puts a shell in between, so the abort is now
reported as this test failing.
"""
import textwrap
import unittest

import jittor as jt

from _helpers.child_process import run_python_child


def _run_child(body):
    """Run ``body`` in a fresh interpreter against *this* jittor tree.

    ``crash_isolated``: the whole point of this file is a child that aborts,
    and without the shell in between jittor's SIGCHLD handler deletes pytest
    instead of letting the abort be asserted.
    """
    return run_python_child(
        ["-c", textwrap.dedent(body)], timeout=1800, crash_isolated=True)


# Creates the cublas / cudnn / curand handles whose teardown is under test.
_TOUCH_BACKENDS = """
    import ctypes
    import jittor as jt
    jt.flags.use_cuda = 1
    a = jt.random((16, 16), "float32")     # curand
    jt.matmul(a, a).sync()                 # cublas
    print("BODY-DONE", flush=True)
"""

# A one-thread kernel writing a gigabyte past its output. The launch itself
# succeeds; the fault lands asynchronously and leaves the context in the
# sticky-error state every later CUDA call inherits -- including the Destroys
# in the static destructors. This is the shape of the real thing (an async
# error surfacing late), made deterministic.
#
# `sync_all(True)` rather than `y.sync()`: `Var.sync` only waits for the op to
# be issued, so on hardware it returns cleanly and the fault is still in
# flight. This probe used to stop there and swallow the exception it assumed
# it would get, which left it asserting that a message it never produced
# outlived the teardown noise -- the assertion passed nowhere and was only
# reasoned about. `sync_all(True)` device-synchronizes, which is where jittor
# checks and reports, so the fault the test is about is now really raised.
_POISON_CONTEXT = """
    import sys
    x = jt.zeros((1,), "float32")
    y = jt.code(x.shape, x.dtype, [x], cuda_src=\"\"\"
        __global__ void jt_out_of_bounds_write(float* p) { p[1<<28] = 1.0f; }
        jt_out_of_bounds_write<<<1,1>>>(out0_p);
    \"\"\")
    try:
        y.sync()
        jt.sync_all(True)
    except Exception as fault:
        print("THE-FAULT: %s" % fault, file=sys.stderr, flush=True)
    else:
        raise AssertionError("the out-of-bounds write was never reported")
    assert ctypes.CDLL(None).cudaDeviceSynchronize() == 700  # cudaErrorIllegalAddress
    print("POISONED", flush=True)
"""


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestBackendTeardown(unittest.TestCase):
    def test_clean_exit_reports_nothing(self):
        """Control: an ordinary exit reports no teardown error at all.

        Without this, the next test would also pass for a build that reported
        a teardown failure on every single run.
        """
        proc = _run_child(_TOUCH_BACKENDS)
        self.assertIn("BODY-DONE", proc.stdout)
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        self.assertNotIn("teardown error", proc.stderr)
        self.assertNotIn("terminate called", proc.stderr)

    def test_failing_teardown_is_reported_not_fatal(self):
        proc = _run_child(_TOUCH_BACKENDS + _POISON_CONTEXT)
        self.assertIn("POISONED", proc.stdout)
        tail = proc.stderr[-8000:]

        # Before the fix: SIGABRT (returncode 134) and 'terminate called after
        # throwing an instance of std::runtime_error' out of cudnn_wrapper.cc.
        self.assertNotIn("terminate called", proc.stderr)
        self.assertEqual(proc.returncode, 0, tail)

        # Not throwing is only half of it: the refused teardown must still be
        # on the record rather than swallowed.
        self.assertIn("CUDA teardown error", proc.stderr)
        self.assertIn("cudnnDestroy", proc.stderr)

        # ... and the point of the whole exercise: the fault that actually
        # broke the run outlives the cleanup noise instead of being replaced
        # by it. Both orderings are on the record -- the fault while the
        # process was still running, the teardown error after it.
        self.assertIn("cudaErrorIllegalAddress", proc.stderr)
        self.assertLess(proc.stderr.index("cudaErrorIllegalAddress"),
                        proc.stderr.index("CUDA teardown error"),
                        tail)


if __name__ == "__main__":
    unittest.main()

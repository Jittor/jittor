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

Note when re-checking this against the old code: the child's SIGABRT takes the
*runner* with it.  jittor installs a SIGCHLD handler that quick_exit(1)s the
parent whenever a child dies other than by a clean exit or SIGTERM
(``utils/log.cc``), so a pre-fix run of this file ends as ``pytest`` exiting 1
after one dot, not as a reported failure.  Run the child body directly to see
the abort.
"""
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import jittor as jt


def _run_child(body):
    """Run ``body`` in a fresh interpreter against *this* jittor tree."""
    # The tree under test is the one imported here, not whatever the
    # site-packages .pth points at: a child that picks up a different checkout
    # tests code this test never touched, and says nothing about it.
    python_root = Path(jt.__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.fspath(python_root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        env=env, capture_output=True, text=True, timeout=1800,
    )


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
_POISON_CONTEXT = """
    x = jt.zeros((1,), "float32")
    y = jt.code(x.shape, x.dtype, [x], cuda_src=\"\"\"
        __global__ void jt_out_of_bounds_write(float* p) { p[1<<28] = 1.0f; }
        jt_out_of_bounds_write<<<1,1>>>(out0_p);
    \"\"\")
    try:
        y.sync()
    except Exception:
        pass
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
        # by it.
        self.assertIn("cudaErrorIllegalAddress", proc.stderr)


if __name__ == "__main__":
    unittest.main()

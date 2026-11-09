# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Library handle teardown must report independent failures without aborting.

A previously reported sticky CUDA fault is not logged again during teardown;
the original fault remains visible. An independent teardown error on a healthy
context must still be reported without throwing from a destructor.
"""

from _helpers import capability as _test_capability
import textwrap
import unittest

import jittor as jt

from _helpers.child_process import run_python_child


def _run_child(body):
    """Run ``body`` in a fresh interpreter against *this* jittor tree.

    ``crash_isolated``: the whole point of this file is a child that aborts, and
    the shell turns that into the ``128 + signo`` status the assertions below
    expect.  It is not what keeps pytest alive any more -- ``64350894`` fixed
    the handler that used to delete it.
    """
    return run_python_child(
        ["-c", textwrap.dedent(body)], timeout=1800, crash_isolated=True)


# Creates the cublas / cudnn / curand handles whose teardown is under test.
_TOUCH_BACKENDS = """
import ctypes
import jittor as jt
from contextlib import ExitStack as _ProcessPolicyStack
import atexit as _process_policy_atexit
_process_policy_scopes = _ProcessPolicyStack()
_process_policy_atexit.register(_process_policy_scopes.close)
_process_policy_scopes.enter_context(jt.runtime.scope(use_cuda=1))
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
# dedent-ed here, at the definition. _TOUCH_BACKENDS is written flush left
# and this block indented, and _run_child dedents the *concatenation* --
# which removes nothing when the first line has no indent, so the child
# died at `import sys` with IndentationError before running anything, and
# the test then reported 'POISONED' not found in '' with no hint why.
_POISON_CONTEXT = textwrap.dedent("""
    import sys
    x = jt.zeros((1,), "float32")
    # Address 16, not p past its end: under HMM a GPU write to any mapped host
    # page succeeds, and `import jittor` raises RLIMIT_STACK to 512 MiB, which a
    # child inherits as the size of each thread stack -- p[1<<28] landed in one
    # and nothing faulted. Page 0 is never mapped, by any process.
    y = jt.code(x.shape, x.dtype, [x], cuda_src=\"\"\"
        __global__ void jt_out_of_bounds_write(float* p) { *((volatile float*)16) = 1.0f; }
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
""")


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
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

    def test_reported_sticky_fault_exits_without_duplicate_teardown(self):
        proc = _run_child(_TOUCH_BACKENDS + _POISON_CONTEXT)
        self.assertIn("POISONED", proc.stdout)
        tail = proc.stderr[-8000:]

        # Before the fix: SIGABRT (returncode 134) and 'terminate called after
        # throwing an instance of std::runtime_error' out of cudnn_wrapper.cc.
        self.assertNotIn("terminate called", proc.stderr)
        self.assertEqual(proc.returncode, 0, tail)

        # Once reported, a sticky context error is not duplicated as a
        # teardown error. The original fault must remain visible instead.
        self.assertIn("cudaErrorIllegalAddress", proc.stderr)
        self.assertNotIn("CUDA teardown error", proc.stderr)

    def test_independent_teardown_error_is_reported_without_throwing(self):
        # Inject a nonzero status without poisoning the CUDA context. This
        # exercises the actual production macro and must log and continue.
        body = r'''
        a = jt.zeros((1,), "float32")
        b = jt.code([1], a.dtype, [a],
            cuda_header="#include \"helper_cuda.h\"",
            cuda_src="""
        __global__ void kernel(float32* a, float32* b) { b[0] = a[0]; }
        kernel<<<1,1>>>(in0_p, out0_p);
        peekCudaErrorsAlways(cudaErrorInvalidValue);
        """)
        b.sync()
        print("HEALTHY-TEARDOWN-PEEK", flush=True)
        '''
        # _TOUCH_BACKENDS has no indentation, so dedent the appended block
        # separately, as with _POISON_CONTEXT.
        proc = _run_child(_TOUCH_BACKENDS + textwrap.dedent(body))
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        self.assertIn("HEALTHY-TEARDOWN-PEEK", proc.stdout)
        self.assertIn("CUDA teardown error", proc.stderr)
        self.assertIn("cudaErrorInvalidValue", proc.stderr)
        self.assertNotIn("terminate called", proc.stderr)



if __name__ == "__main__":
    unittest.main()

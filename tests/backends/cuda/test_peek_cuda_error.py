# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``peekCudaErrors`` must keep reporting after the first failure.

``peek()`` in ``extern/cuda/inc/helper_cuda.h`` used a process-wide boolean
latch (``jittor::peek_logged``): the *first* asynchronous CUDA error anywhere in
the process was printed and every later one -- from any call site, of any kind,
for the rest of the run -- was dropped.  Since ``peek`` sits on the teardown and
async-free paths, the errors that matter most for diagnosing a corrupted stream
are exactly the ones that came after.

The latch is replaced by per-(call site, error code) rate limiting with
exponential backoff, so a hot loop cannot flood the log while a *new* failure
still gets through.

The probe injects ``peekCudaErrors`` calls into a ``jt.code`` op, which is the
only way to reach the macro from Python.  Note what makes this a proof that the
new header was actually compiled in rather than just "still green": the assert
counts *three* reports where the old binary could only ever produce one, and the
old core exports no ``peek_should_log`` at all, so a stale core would fail the
JIT op's link rather than pass quietly (see the ``cuda-backend-choice-proof``
skill on changes under ``extern/cuda/inc``).

Run::  python -m pytest tests/backends/cuda/test_peek_cuda_error.py
"""

import unittest

import jittor as jt

from _helpers.child_process import run_child_script


PROBE = r'''
import jittor as jt
jt.flags.use_cuda = 1
a = jt.zeros(1)
b = jt.code([1], a.dtype, [a],
cuda_header="""
#include "helper_cuda.h"
""",
cuda_src="""
__global__ void kernel(float32* a, float32* b) { b[0] = a[0]; }
kernel<<<1,1>>>(in0_p, out0_p);
peekCudaErrors(cudaErrorInvalidValue);
peekCudaErrors(cudaErrorInvalidValue);
peekCudaErrors(cudaErrorInvalidDevice);
""")
b.sync()
print("DONE")
'''


@unittest.skipIf(not jt.has_cuda, "No cuda found")
class TestPeekCudaError(unittest.TestCase):
    def test_every_call_site_is_reported(self):
        done = run_child_script(PROBE, text=True, merge_stderr=True,
                                name="peek_cuda_error")
        output = done.stdout
        self.assertEqual(done.returncode, 0, output[-4000:])
        self.assertIn("DONE", output, output[-4000:])
        # Three peeks: two distinct call sites with the same code plus one with
        # a different code. The old process-wide latch printed exactly one.
        peeks = [line for line in output.splitlines() if "Peek CUDA error" in line]
        self.assertGreaterEqual(len(peeks), 3, output[-4000:])


if __name__ == "__main__":
    unittest.main()

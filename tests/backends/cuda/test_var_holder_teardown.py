# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Dropping a Var reports a failed release; it never aborts the process.

``~VarHolder`` drops the last reference to a ``Var``, which runs the liveness
propagation, which frees nodes, which reaches the allocator.  Every step on
that path reports by throwing: the liveness counters assert their own
invariants and the allocator checks its CUDA calls.

A destructor is implicitly ``noexcept``, so an error that escapes one is
``std::terminate`` *at the destructor's own frame*.  The generated
``tp_dealloc`` does wrap the ``~VarHolder()`` call in a ``try`` (see
``tests/core/test_pyjt_compiler_parser.py``), but that catch sits below the
frame where ``terminate`` is called and never runs -- which is why "the
destructor half is done" needed a run on real hardware before it could be
believed.

The trigger below is a cuDNN LSTM trained on GPU.  Its backward pass releases
backward liveness once more than it took, the counter's own invariant fires,
and before the fix the whole interpreter went down on SIGABRT -- taking
``tests/backends/cuda`` with it at whichever file happened to run next, with no
failing test to point at.

The unbalanced release itself is a separate defect: it is upstream behaviour
that used to underflow an ``int`` in silence, and it is only what this test
currently leans on to make a destructor fail.  So the assertions below are
about the contract (no ``terminate``, ordinary exit) and not about the
trigger's message: when the release is balanced this test stops exercising
anything, and that should surface as someone re-pointing it rather than as a
green run that proves nothing.

``crash_isolated=True``: jittor installs a process-level ``SIGCHLD`` handler,
so a child that aborts takes the runner with it and pytest vanishes with no
output at all (6.C31).  The shell in between turns the abort into an ordinary
non-zero exit that can be asserted.
"""
import textwrap
import unittest

import jittor as jt

from _helpers.child_process import run_python_child


def _run_child(body):
    return run_python_child(
        ["-c", textwrap.dedent(body)], timeout=1800, crash_isolated=True)


_CUDA_LSTM_BACKWARD = """
    import numpy as np
    import jittor as jt

    rs = np.random.RandomState(7)
    x = (rs.randn(5, 4, 8) * 0.5).astype("float32")
    with jt.flag_scope(use_cuda=1):
        module = jt.nn.LSTM(8, 8, num_layers=2)
        module.train()
        out = module(jt.array(x))[0]
        params = list(module.parameters())
        for g in jt.grad(out.float32().sum(), params):
            g.float32().numpy()
    print("BODY-DONE", flush=True)
"""

_CUDA_FORWARD_ONLY = """
    import jittor as jt
    with jt.flag_scope(use_cuda=1):
        a = jt.random((16, 16), "float32")
        jt.matmul(a, a).sync()
    print("BODY-DONE", flush=True)
"""


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestVarHolderTeardown(unittest.TestCase):
    def test_ordinary_teardown_is_quiet(self):
        """Control: without this, the next test would pass on a build that
        aborted on every run, because "did not abort" would never have been
        shown to mean anything here."""
        proc = _run_child(_CUDA_FORWARD_ONLY)
        self.assertIn("BODY-DONE", proc.stdout)
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        self.assertNotIn("terminate called", proc.stderr)

    def test_a_failing_release_does_not_abort_the_process(self):
        proc = _run_child(_CUDA_LSTM_BACKWARD)
        tail = proc.stderr[-8000:]
        # Before the fix: 'terminate called after throwing an instance of
        # std::runtime_error ... backward liveness release without a matching
        # owner', and returncode 134.
        self.assertNotIn("terminate called", proc.stderr)
        self.assertIn("BODY-DONE", proc.stdout)
        self.assertEqual(proc.returncode, 0, tail)


if __name__ == "__main__":
    unittest.main()

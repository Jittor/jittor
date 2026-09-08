# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Dead-code elimination keeps "(void)x;" out of generated kernels.

Op sources end their JIT body with statements like

    (void)count, (void)rcount, (void)yshape0, (void)ystride0;

(``ops/reduce_op.cc``; ``ops/transpose_op.cc`` and ``ops/fuse_transpose_op.cc``
do the same). They exist to silence an unused-variable warning when the file is
compiled normally. In the generated kernel they must not count as uses -- every
variable named there would otherwise survive dead-code elimination -- and they
must not be emitted.

KernelIR decides that by matching the shape of the statement at parse time. The
counterpart, that a statement which merely mentions ``void`` is left alone, is
covered by the C++ unit tests ``kernel_ir_void_discard`` and
``kernel_ir_void_discard_shapes`` (``src/tests/test_kernel_ir.cc``, reachable
from ``test_jit_tests.py``): no op in the tree emits a void cast into a fused
kernel today, which is exactly why deleting those statements went unnoticed.
"""
import unittest

import numpy as np

import jittor as jt


class TestVoidDiscardElimination(unittest.TestCase):
    def _fused_source(self, build, tag):
        with jt.profile_scope(compile_options={"test_void_discard": tag}) as rep:
            value = build()
        self.assertGreaterEqual(len(rep), 2)
        with open(rep[1][1]) as source:
            return value, source.read()

    def test_reduce_kernel_has_no_discard_markers(self):
        a = jt.random([64, 128])
        a.sync()
        expected = a.numpy().sum(axis=0)
        got, source = self._fused_source(
            lambda: jt.reduce(a, "add", (0,)).data, 1)
        np.testing.assert_allclose(got, expected, rtol=1e-4)
        self.assertNotIn("(void)", source)
        # the variables the markers named are dead and must be gone with them
        for name in ("count", "rcount"):
            self.assertNotIn(" %s =" % name, source)
            self.assertNotIn(" %s=" % name, source)

    def test_fused_elementwise_and_reduce_has_no_discard_markers(self):
        a = jt.random([32, 96])
        a.sync()
        expected = (a.numpy() * 2.0).sum(axis=1)
        got, source = self._fused_source(
            lambda: jt.reduce(a * 2.0, "add", (1,)).data, 2)
        np.testing.assert_allclose(got, expected, rtol=1e-4)
        self.assertNotIn("(void)", source)

    def test_a_standalone_op_keeps_its_marker(self):
        """Only fused ops run the pass pipeline, so only they strip the marker.

        ``transpose`` is compiled on its own -- ``OpCompiler`` runs PassManager
        only when the op is a fused op -- so its ``(void)xshape0;`` reaches the
        C++ compiler, where it means what it says and costs nothing. Pinned so
        that the difference reads as a fact about where the pass runs rather
        than as an inconsistency.
        """
        a = jt.random([32, 48])
        a.sync()
        expected = a.numpy().transpose()
        got, source = self._fused_source(
            lambda: jt.transpose(a, [1, 0]).data, 3)
        np.testing.assert_allclose(got, expected, rtol=1e-5)
        self.assertIn("(void)xshape0;", source)


if __name__ == "__main__":
    unittest.main()

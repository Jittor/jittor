# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""FloatAtomicFixPass: the ordered-int representation float atomics need.

CUDA has no atomicMax for float, so ``cuda_atomic_max(float*)`` runs an integer
atomicMax over the buffer reinterpreted as *ordered ints*
(``misc/cuda_atomic.h``).  That is only correct if the buffer is in that
representation, which is this pass's job: it rewrites the initialisation into
``__int_as_float(floatToOrderedInt(...))`` and appends ``fix_float()`` to
convert back at the end.

So the pass is not an optimisation -- skipping a statement it does not
understand leaves an integer atomicMax running over raw float bit patterns that
are never converted back.  It used to do exactly that, in three places: a shape
mismatch, a target whose name does not end in ``p``, and ``catch (...)``.
"""
import os
import unittest

import numpy as np

import jittor as jt


def _kernels(build, tag, options=None):
    """Run ``build`` under a fresh jit key; return (value, generated sources)."""
    compile_options = dict(options or {})
    compile_options["_float_atomic_fix"] = tag
    with jt.profile_scope(compile_options=compile_options, enable_tuner=0) as rep:
        got = build()
    src = ""
    for row in rep[1:]:
        for cell in row:
            if isinstance(cell, str) and cell.endswith(".cc") and os.path.exists(cell):
                with open(cell) as f:
                    src += f.read()
    return got, src


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestFloatAtomicFix(unittest.TestCase):
    def setUp(self):
        self._use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.sync_all()
        jt.flags.use_cuda = self._use_cuda

    def test_a_float_atomic_reduce_is_converted_and_converted_back(self):
        """Both halves have to be there, or the buffer is left as ordered ints."""
        tag = 0
        for dtype in ("float32", "float64"):
            for op, reference in (("maximum", np.max), ("minimum", np.min)):
                tag += 1
                a = jt.random([64, 128]).cast(dtype)
                a.sync()
                got, src = _kernels(
                    lambda a=a, op=op: jt.reduce(a, op, (0,)).numpy(), tag)
                np.testing.assert_allclose(
                    got, reference(a.numpy(), axis=0), rtol=1e-5, atol=1e-5,
                    err_msg="%s %s" % (dtype, op))
                if "cuda_atomic_" not in src:
                    # this shape did not need an atomic; nothing to check
                    continue
                assert "floatToOrderedInt(" in src, (
                    "%s %s: atomic without the ordered-int initialisation" %
                    (dtype, op))
                assert "fix_float(" in src, (
                    "%s %s: ordered ints are never converted back" %
                    (dtype, op))


    def test_a_scatter_max_still_uses_the_raw_ieee_atomic(self):
        """setitem's cuda_atomic_max_rmw is a different function.

        It shares the ``cuda_atomic_max`` prefix, and this pass used to claim
        any statement with that prefix and then quietly drop it when the shape
        did not match. It must not be claimed at all: setitem's output is a raw
        memcpy copy with no ordered-int pass (see ops/setitem_op.cc).
        """
        x = jt.zeros([32])
        v = jt.random([64])
        index = jt.array(np.arange(64) % 32)
        x.sync(); v.sync(); index.sync()
        got = x.setitem(index, v, "maximum").numpy()
        reference = np.zeros(32, "float32")
        for i, j in enumerate(index.numpy()):
            reference[j] = max(reference[j], v.numpy()[i])
        np.testing.assert_allclose(got, reference, rtol=1e-5, atol=1e-5)

    def test_a_name_the_compiler_cannot_resolve_is_named(self):
        """``restride`` renames op0_yp to op0_y_new, and RestridePass then asks
        the compiler to resolve that name.

        The lookup failed with ``Check failed: found && opvar_id < ...
        Something wrong... Could you please report this issue?`` -- an assert
        with neither the name nor a hint. It now says which name it was.
        """
        a = jt.random([64, 128])
        a.sync()
        with self.assertRaises(Exception) as caught:
            with jt.flag_scope(compile_options={"restride": 1,
                                                "_float_atomic_fix": 99}):
                jt.reduce(a, "maximum", (0,)).sync()
        message = str(caught.exception)
        assert "not a var member of this fused op" in message, message
        assert "op0_" in message, message


if __name__ == "__main__":
    unittest.main()

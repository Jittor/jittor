# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""ACL ops must not silently widen their input to float32 (6.B11).

Six ops in ``jittor/extern/acl/aclops`` opened with ``x = x.float32()``. That
is not a conversion for the kernel's benefit: the result var keeps the promoted
dtype, so a bf16 or fp16 model quietly became fp32 at silu / softmax / sigmoid /
relu / leaky_relu / layernorm and stayed fp32 for the rest of the graph. It
disagrees with torch, it costs bandwidth, and nothing reported it. The other 28
op files in the same directory already pass the dtype through, and two of them
(adamw_op.py, getitem_op.py) already declare the supported set explicitly.

NOTE ON VERIFICATION: there is no Ascend hardware here, so no ACL kernel can be
built or run. What is checkable -- and what this test checks -- is the Python
layer that decides which dtype reaches the kernel: the op classes import fine
without CANN, so the kernel launcher is replaced with a recorder and the dtype
handed to it is inspected directly. Whether every ACL kernel accepts fp16 and
bf16 is NOT verified here; the in-repo evidence is that other ops in this
directory already send them (the norm kernels are named
``grouped_bfloat16_rms_norm``).
"""
import unittest

import numpy as np

import jittor as jt

from jittor.extern.acl.aclops import _code as acl_code_mod
from jittor.extern.acl.aclops import (norms_op, relu_op, sigmoid_op, silu_op,
                                      softmax_op)


class _Recorder:
    """Stands in for the ACL kernel launcher and records what it was handed."""

    def __init__(self):
        self.seen_dtype = None
        self.seen_output_dtypes = None

    def __call__(self, name, inputs, output_dtypes=None, output_shapes=None,
                 attr_code="", attr_header="", outputs=None, **kwargs):
        self.seen_dtype = str(inputs[0].dtype)
        if outputs is not None:
            self.seen_output_dtypes = [str(o.dtype) for o in outputs]
            return list(outputs)
        self.seen_output_dtypes = [str(d) for d in (output_dtypes or [])]
        return [jt.empty(s, d) for s, d in zip(output_shapes, output_dtypes)]


class TestAclDtypePreservation(unittest.TestCase):

    # (module, launcher attribute, callable building the op invocation)
    CASES = [
        ("silu", silu_op, "silu_cmd", lambda m, x: m.SiLUACL()(x)),
        ("softmax", softmax_op, "softmax_cmd", lambda m, x: m.SoftmaxACL().execute(x, -1)),
        ("sigmoid", sigmoid_op, "sigmoid_cmd", lambda m, x: m.SigmoidACL().execute(x)),
        ("relu", relu_op, "acl_code", lambda m, x: m.ReLUACL()(x)),
        ("leaky_relu", relu_op, "acl_code", lambda m, x: m.LeakyReLUACL()(x, 0.01)),
    ]

    def _run(self, module, attr, invoke, x):
        recorder = _Recorder()
        original = getattr(module, attr)
        setattr(module, attr, recorder)
        try:
            invoke(module, x)
        finally:
            setattr(module, attr, original)
        return recorder

    def test_input_dtype_reaches_the_kernel_unchanged(self):
        for dtype in ("float16", "bfloat16", "float32"):
            for name, module, attr, invoke in self.CASES:
                with self.subTest(op=name, dtype=dtype):
                    x = jt.ones([2, 4], dtype="float32").cast(dtype)
                    self.assertEqual(str(x.dtype), dtype)
                    recorder = self._run(module, attr, invoke, x)
                    self.assertEqual(
                        recorder.seen_dtype, dtype,
                        "{} widened its input to {}".format(name, recorder.seen_dtype))

    def test_output_dtype_is_not_widened_either(self):
        # softmax also allocated its output with jt.empty(shape) -- no dtype --
        # which defaults to float32 and put the widening straight back.
        for dtype in ("float16", "bfloat16"):
            for name, module, attr, invoke in self.CASES:
                with self.subTest(op=name, dtype=dtype):
                    x = jt.ones([2, 4], dtype="float32").cast(dtype)
                    recorder = self._run(module, attr, invoke, x)
                    self.assertTrue(
                        all(d == dtype for d in recorder.seen_output_dtypes),
                        "{} allocated {} outputs for a {} input".format(
                            name, recorder.seen_output_dtypes, dtype))

    def test_layernorm_keeps_its_input_dtype(self):
        for dtype in ("float16", "bfloat16", "float32"):
            with self.subTest(dtype=dtype):
                x = jt.ones([2, 4], dtype="float32").cast(dtype)
                w = jt.ones([4], dtype="float32").cast(dtype)
                b = jt.zeros([4], dtype="float32").cast(dtype)
                recorder = _Recorder()
                original = norms_op.norms_cmd
                norms_op.norms_cmd = recorder
                try:
                    norms_op.LayerNormACL([4])(x, w, b)
                finally:
                    norms_op.norms_cmd = original
                self.assertEqual(recorder.seen_dtype, dtype)

    def test_unsupported_dtype_is_rejected_not_widened(self):
        # float64 / integers have no ACL float kernel. Saying so beats turning
        # them into float32 behind the user's back.
        for dtype in ("float64", "int32"):
            with self.subTest(dtype=dtype):
                x = jt.ones([2, 4], dtype=dtype)
                with self.assertRaises(TypeError):
                    acl_code_mod.check_acl_float_dtype(x, "silu")

    def test_supported_set_is_declared_once(self):
        self.assertEqual(acl_code_mod.ACL_FLOAT_DTYPES,
                         ("float16", "bfloat16", "float32"))
        x = jt.ones([2], dtype="float32")
        self.assertIs(acl_code_mod.check_acl_float_dtype(x, "silu"), x)


if __name__ == "__main__":
    unittest.main()

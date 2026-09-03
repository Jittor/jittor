# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``Module.half()`` converts the module, not half of it.

``.half()``/``.float16()``/``.bfloat16()``/``.float32()``/``.float64()`` walked
``parameters()`` only, so a BatchNorm that had been made half held a float16
weight next to a float32 running_mean -- and a model "converted to fp16" still
normalised against float32 statistics. torch's ``Module.to(dtype)`` casts
parameters and buffers alike (``_apply`` walks both), leaving the integer and
bool buffers -- num_batches_tracked, attention masks -- as they are.

The same methods also carried a dead amp branch: ``self._amp_level = -1``
immediately followed by ``if self._amp_level >= 0:``. Its body installed
``__half_call__`` onto the CLASS, so had it ever run, calling ``.half()`` on one
module would have wrapped the forward of every other instance of that class.
"""

import unittest

import numpy as np

import jittor as jt


class _WithBuffers(jt.Module):
    def __init__(self):
        self.weight = jt.ones((3,))
        self.register_buffer("scale", jt.ones((3,)))
        self.register_buffer("steps", jt.array(np.zeros(1, "int32")))
        self.register_buffer("mask", jt.array(np.ones(3, "bool")))
        self.register_buffer("tmp", jt.ones((3,)), persistent=False)

    def execute(self, x):
        return x * self.weight * self.scale


class TestModuleDtypeCast(unittest.TestCase):

    def _dtypes(self, m):
        return {n: str(v.dtype) for n, v in
                m.named_parameters() + m.named_buffers()}

    def test_half_converts_float_buffers(self):
        m = _WithBuffers().half()
        self.assertEqual(self._dtypes(m), {
            "weight": "float16", "scale": "float16", "tmp": "float16",
            "steps": "int32", "mask": "bool"})

    def test_batchnorm_running_stats_follow_the_weight(self):
        """The case the audit names: bn.half() left running_mean float32."""
        bn = jt.nn.BatchNorm(4).half()
        self.assertEqual(str(bn.weight.dtype), "float16")
        self.assertEqual(str(bn.running_mean.dtype), "float16")
        self.assertEqual(str(bn.running_var.dtype), "float16")
        # counters stay integral
        self.assertEqual(str(bn.num_batches_tracked.dtype), "int32")

    def test_bfloat16_float32_float64_all_reach_the_buffers(self):
        for method, want in (("bfloat16", "bfloat16"),
                             ("float32", "float32"),
                             ("float64", "float64")):
            bn = getattr(jt.nn.BatchNorm(4), method)()
            self.assertEqual(str(bn.running_var.dtype), want, method)

    def test_a_round_trip_through_half_and_back(self):
        bn = jt.nn.BatchNorm(4)
        bn.running_var.assign(jt.ones(4) * 2)
        bn.half().float32()
        self.assertEqual(str(bn.running_var.dtype), "float32")
        np.testing.assert_allclose(bn.running_var.numpy(), 2.0)

    def test_half_does_not_rewrite_the_class(self):
        """The dead branch installed __half_call__ on ``type(self)``.

        A per-instance conversion must not change how every other instance of
        the same class is called.
        """
        before = jt.nn.BatchNorm.__call__
        one = jt.nn.BatchNorm(4)
        one.half()
        self.assertIs(jt.nn.BatchNorm.__call__, before)
        other = jt.nn.BatchNorm(4)
        out = other(jt.ones((2, 4, 2, 2)))
        self.assertEqual(str(out.dtype), "float32")

    def test_no_amp_level_state_is_left_behind(self):
        m = _WithBuffers().half()
        self.assertNotIn("_amp_level", m.__dict__)

    def test_nested_modules_are_converted_whole(self):
        net = jt.nn.Sequential(jt.nn.Linear(4, 4), jt.nn.BatchNorm(4))
        net.half()
        self.assertEqual(str(net[1].running_mean.dtype), "float16")


if __name__ == "__main__":
    unittest.main()

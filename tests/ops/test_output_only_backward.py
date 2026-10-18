"""Unary ops whose gradient reads only their output keep only their output.

``relu``, ``exp``, ``sqrt`` and ``sigmoid`` differentiate from ``y`` alone
(`UnaryOp::grad`), so after the forward their input is dead: nothing in the
backward reads it. ``relu`` used to be written as ``ternary(x > 0, x, 0)``,
whose backward kept the sign mask -- 586 MB of a ResNet-50 training step's
saved tensors -- and a ``jt.Function`` version kept nothing extra but tapped
its input, which stopped the residual ``add`` from fusing into it.

Run::  python -m pytest tests/ops/test_output_only_backward.py
"""
import unittest

import numpy as np

import jittor as jt
from jittor import nn

HAS_CUDA = bool(jt.has_cuda)
MiB = 1 << 20


class TestReluValues(unittest.TestCase):
    def check(self, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda):
            a = np.random.RandomState(0).randn(257).astype("float32") * 3
            a[:3] = [0.0, -0.0, np.nan]
            weight = np.arange(257, dtype="float32")
            for dtype in ("float32", "float16", "bfloat16", "float64"):
                x = jt.array(a).cast(dtype)
                y = nn.relu(x)
                self.assertEqual(y.dtype, x.dtype)
                expect = np.where(a > 0, a, 0)
                tol = 1e-2 if dtype in ("float16", "bfloat16") else 1e-6
                np.testing.assert_allclose(y.float32().numpy(), expect, rtol=tol, atol=tol)
                g = jt.grad((nn.relu(x) * jt.array(weight).cast(dtype)).sum(), x)
                np.testing.assert_allclose(g.float32().numpy(), np.where(a > 0, weight, 0),
                                           rtol=tol)
            ints = jt.array(np.array([-2, 0, 3], dtype="int32"))
            np.testing.assert_array_equal(nn.relu(ints).numpy(), [0, 0, 3])

    def test_cpu(self):
        self.check(0)

    @unittest.skipIf(not HAS_CUDA, "needs CUDA")
    def test_cuda(self):
        self.check(1)

    def test_fused_residual_add(self):
        # relu(a + b) is one fused expression; its gradient reaches both addends.
        a = jt.randn(4, 8)
        b = jt.randn(4, 8)
        y = nn.relu(a + b)
        ga, gb = jt.grad((y * y).sum(), [a, b])
        ref = np.maximum(a.numpy() + b.numpy(), 0)
        np.testing.assert_allclose(y.numpy(), ref, rtol=1e-6)
        np.testing.assert_allclose(ga.numpy(), 2 * ref, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gb.numpy(), 2 * ref, rtol=1e-6, atol=1e-6)


@unittest.skipIf(not HAS_CUDA, "needs CUDA")
class TestInputNotKept(unittest.TestCase):
    def kept_bytes(self, fn):
        """Device bytes still held once ``fn``'s forward has run.

        The input is synced, so it is a real buffer: left lazy it would fuse
        into ``fn`` and there would be nothing to keep or drop.
        """
        def used():
            jt.sync_all(True)
            jt.gc()
            return jt.core.device_memory_used(0)
        with jt.flag_scope(use_cuda=1):
            x = jt.randn(1024, 1024)
            base = used()
            h = x * 3.0
            h.sync()
            y = fn(h)
            y.sync()
            del h
            kept = used() - base
            del x, y
        return kept

    def test_output_only_ops_drop_their_input(self):
        for name, fn in (("relu", nn.relu), ("exp", jt.exp), ("sqrt", jt.sqrt),
                         ("sigmoid", jt.sigmoid)):
            # The output alone: 4 MiB.
            self.assertEqual(self.kept_bytes(fn), 4 * MiB, name)

    def test_an_op_that_needs_its_input_still_keeps_it(self):
        # The control: sin differentiates through cos(x), so x stays.
        self.assertEqual(self.kept_bytes(jt.sin), 8 * MiB)


if __name__ == "__main__":
    unittest.main()

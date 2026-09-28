# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A rank>2 left operand reaches the 2-D GEMM without being reshaped.

``matmul`` and ``matmul_transpose`` used to route a batched left operand through
the rank-2 cuBLAS kernel by flattening it and un-flattening the result. Both
reshapes are pure views that generate no code, but each is a graph node, and
``nn.Linear`` on a batched input pays two of them on every forward -- measured
at 64 of the 408 nodes a transformer decode step builds.

A dense row-major ``(d0, .., dn-1, m)`` *is* its ``(d0*..*dn-1, m)`` flattening:
same pointer, same leading dimension. So the kernel can take the rank it is
given. What has to be tested is that "same buffer" really is all that changed --
a wrong flattening is a silently wrong GEMM, not a crash -- so this covers the
forward values, the output shape, both gradients, the transposed spelling the
backward itself builds, and the strided operands that must keep the old route
because the flattening does not describe them.
"""
import unittest

import numpy as np

from _helpers import capability as _test_capability

import jittor as jt
from jittor import nn


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


class TestMatmulHigherRankCpu(unittest.TestCase):

    device_flag = 0

    def setUp(self):
        self.rs = np.random.RandomState(20260912)

    def _close(self, got, want, tol=2e-4):
        got, want = np.asarray(got), np.asarray(want)
        self.assertEqual(got.shape, want.shape)
        scale = max(float(np.abs(want).max()), 1e-30)
        self.assertLessEqual(float(np.abs(got - want).max()) / scale, tol)

    def test_forward_matches_numpy_for_every_rank(self):
        for ash, bsh in [((4, 3), (3, 5)), ((2, 4, 3), (3, 5)),
                         ((2, 3, 4, 3), (3, 5)), ((2, 3, 4, 5, 6), (6, 7)),
                         ((1, 1, 8), (8, 4)), ((7, 1, 3), (3, 3))]:
            with self.subTest(a=ash, b=bsh):
                an = self.rs.rand(*ash).astype("float32")
                bn = self.rs.rand(*bsh).astype("float32")
                with jt.flag_scope(use_cuda=self.device_flag):
                    got = jt.matmul(jt.array(an), jt.array(bn)).numpy()
                self._close(got, an @ bn)

    def test_matmul_transpose_matches_numpy_for_every_rank(self):
        """The spelling `nn.Linear` uses."""
        for ash, bsh in [((4, 3), (5, 3)), ((2, 4, 3), (5, 3)),
                         ((2, 3, 4, 3), (5, 3)), ((1, 1, 8), (4, 8))]:
            with self.subTest(a=ash, b=bsh):
                an = self.rs.rand(*ash).astype("float32")
                bn = self.rs.rand(*bsh).astype("float32")
                with jt.flag_scope(use_cuda=self.device_flag):
                    got = jt.nn.matmul_transpose(jt.array(an), jt.array(bn)).numpy()
                self._close(got, an @ bn.T)

    def test_a_rank_1_left_operand_still_works(self):
        """It reaches the same branch as rank > 2 and must not take its path.

        A rank-1 `a` flattens to `(1, m)`, which is a row the reshape adds
        rather than a rank the kernel can read off the buffer; handing it
        straight to the 2-D kernel made `nn.Linear` on a 1-D input raise.
        """
        an = self.rs.rand(10).astype("float32")
        bn = self.rs.rand(6, 10).astype("float32")
        with jt.flag_scope(use_cuda=self.device_flag):
            self._close(jt.nn.matmul_transpose(jt.array(an), jt.array(bn)).numpy(),
                        an @ bn.T)
            lin = nn.Linear(10, 6)
            wn = lin.weight.numpy()
            bias = lin.bias.numpy()
            self._close(lin(jt.array(an)).numpy(), an @ wn.T + bias)
        # and the matmul spelling, whose branch is guarded separately
        bn2 = self.rs.rand(10, 6).astype("float32")
        with jt.flag_scope(use_cuda=self.device_flag):
            self._close(jt.matmul(jt.array(an), jt.array(bn2)).numpy(), an @ bn2)

    def test_a_strided_left_operand_is_still_right(self):
        """The flattening does not describe these, so they keep the reshape."""
        base = self.rs.rand(4, 8, 6).astype("float32")
        bn = self.rs.rand(3, 5).astype("float32")
        with jt.flag_scope(use_cuda=self.device_flag):
            sliced = jt.array(base)[:, :, 0:6:2]
            self.assertFalse(sliced._storage_is_contiguous())
            self._close(jt.matmul(sliced, jt.array(bn)).numpy(),
                        base[:, :, 0:6:2] @ bn)
        moved = self.rs.rand(4, 3, 8).astype("float32")
        with jt.flag_scope(use_cuda=self.device_flag):
            view = jt.array(moved).transpose(0, 2, 1)
            self._close(jt.matmul(view, jt.array(bn)).numpy(),
                        moved.transpose(0, 2, 1) @ bn)

    def test_both_gradients(self):
        for ash, bsh in [((2, 4, 3), (3, 5)), ((8, 16, 32), (32, 64)),
                         ((2, 3, 4, 3), (3, 5))]:
            with self.subTest(a=ash, b=bsh):
                an = self.rs.rand(*ash).astype("float32")
                bn = self.rs.rand(*bsh).astype("float32")
                cot = self.rs.rand(*(list(ash[:-1]) + [bsh[1]])).astype("float32")
                with jt.flag_scope(use_cuda=self.device_flag):
                    x, w = jt.array(an), jt.array(bn)
                    loss = (jt.matmul(x, w) * jt.array(cot)).sum()
                    gx, gw = jt.grad(loss, [x, w])
                    gx, gw = gx.numpy(), gw.numpy()
                a2 = an.reshape(-1, ash[-1])
                c2 = cot.reshape(-1, bsh[1])
                self._close(gx, (c2 @ bn.T).reshape(ash))
                self._close(gw, a2.T @ c2)

    def test_linear_on_a_batched_input(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            lin = nn.Linear(32, 64)
            wn, bn = lin.weight.numpy(), lin.bias.numpy()
            xn = self.rs.rand(4, 7, 32).astype("float32")
            cot = self.rs.rand(4, 7, 64).astype("float32")
            x = jt.array(xn)
            y = lin(x)
            loss = (y * jt.array(cot)).sum()
            gx, gw, gb = jt.grad(loss, [x, lin.weight, lin.bias])
            y, gx, gw, gb = y.numpy(), gx.numpy(), gw.numpy(), gb.numpy()
        self._close(y, xn @ wn.T + bn)
        self._close(gx, cot @ wn)
        self._close(gw, cot.reshape(-1, 64).T @ xn.reshape(-1, 32))
        self._close(gb, cot.reshape(-1, 64).sum(0))

    def test_dtypes(self):
        # jt.array narrows a float64 host array under auto_convert_64_to_32, so
        # the dtype is asked for explicitly and the reference read back from it
        for dtype, tol in (("float32", 2e-4), ("float64", 1e-12)):
            with self.subTest(dtype=dtype):
                with jt.flag_scope(use_cuda=self.device_flag):
                    a = jt.array(self.rs.rand(2, 8, 16).astype("float32")).cast(dtype)
                    b = jt.array(self.rs.rand(16, 8).astype("float32")).cast(dtype)
                    self.assertEqual(str(a.dtype), dtype)
                    got = jt.matmul(a, b).numpy()
                    an, bn = a.numpy().astype("float64"), b.numpy().astype("float64")
                self._close(got, an @ bn, tol)


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestMatmulHigherRankCuda(TestMatmulHigherRankCpu):
    """The same body on the device, which is where the rank>2 kernel path is."""

    device_flag = 1


if __name__ == "__main__":
    unittest.main()

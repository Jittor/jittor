# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import numpy as np
import jittor as jt


CASES_UNARY = [
    ("i->", (5,)),
    ("ij->", (3, 4)),
    ("ij->ji", (3, 4)),
    ("ii->", (4, 4)),
    ("ii->i", (4, 4)),
    ("iii->i", (3, 3, 3)),
]

CASES_BINARY = [
    ("ij,ij->ij", (3, 4), (3, 4)),
    ("i,i->", (5,), (5,)),
    ("i,j->ij", (3,), (4,)),
    ("ij,j->i", (3, 4), (4,)),
    ("i,ij->j", (3,), (3, 4)),
    ("ij,jk->ik", (3, 4), (4, 5)),
    ("ji,jk->ik", (4, 3), (4, 5)),
    ("ij,kj->ik", (3, 4), (5, 4)),
    ("bij,bjk->bik", (2, 3, 4), (2, 4, 5)),
    ("...ij,...jk->...ik", (2, 3, 3, 4), (2, 3, 4, 5)),
    ("ij,kl->ijkl", (2, 3), (4, 5)),
    ("ii,ii->", (4, 4), (4, 4)),
    ("...i,...i->...", (2, 3, 4), (2, 3, 4)),
    ("ij,ij->", (3, 4), (3, 4)),
    ("ijk,ikl->ijl", (2, 3, 4), (2, 4, 5)),
    # Broadcast cases on shared labels (numpy-style size-1 expansion).
    ("i,i->", (1,), (5,)),
    ("i,i->", (5,), (1,)),
    ("...i,...i->...", (1, 3, 4), (2, 3, 4)),
    ("...i,...i->...", (2, 1, 4), (2, 3, 4)),
    ("bij,bjk->bik", (1, 2, 3), (4, 3, 5)),
    ("bij,bjk->bik", (4, 2, 3), (1, 3, 5)),
    ("ij,jk->ik", (2, 3), (1, 5)),
    ("ij,jk->ik", (2, 1), (4, 3)),
    ("bi,bj->bij", (1, 3), (4, 2)),
]

CASES_TERNARY = [
    ("ij,jk,kl->il", (2, 3), (3, 4), (4, 5)),
    ("bij,bjk,bkl->bil", (2, 3, 4), (2, 4, 5), (2, 5, 6)),
    ("i,j,k->ijk", (2,), (3,), (4,)),
]


def _rand(shape, seed):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32) if shape else np.array(rng.randn(), dtype=np.float32)


class TestLinalgEinsum(unittest.TestCase):
    def _check(self, eq, *shapes):
        arrs = [_rand(s, 17 + i) for i, s in enumerate(shapes)]
        expected = np.einsum(eq, *arrs)
        got = jt.linalg.einsum(eq, *[jt.array(a) for a in arrs]).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5,
            err_msg=f"einsum({eq!r}) mismatch")

    def test_unary(self):
        for eq, *shapes in CASES_UNARY:
            with self.subTest(eq=eq):
                self._check(eq, *shapes)

    def test_binary(self):
        for eq, *shapes in CASES_BINARY:
            with self.subTest(eq=eq):
                self._check(eq, *shapes)

    def test_ternary(self):
        for eq, *shapes in CASES_TERNARY:
            with self.subTest(eq=eq):
                self._check(eq, *shapes)

    def test_grad_dot(self):
        a_np = _rand((5,), 1)
        b_np = _rand((5,), 2)
        a = jt.array(a_np); b = jt.array(b_np)
        y = jt.linalg.einsum("i,i->", a, b)
        ga, gb = jt.grad(y, [a, b])
        np.testing.assert_allclose(ga.numpy(), b_np, rtol=1e-4)
        np.testing.assert_allclose(gb.numpy(), a_np, rtol=1e-4)

    def test_grad_matmul(self):
        a_np = _rand((3, 4), 1)
        b_np = _rand((4, 5), 2)
        a = jt.array(a_np); b = jt.array(b_np)
        y = jt.linalg.einsum("ij,jk->ik", a, b).sum()
        ga, gb = jt.grad(y, [a, b])
        np.testing.assert_allclose(ga.numpy(),
            np.ones((3, 5)) @ b_np.T, rtol=1e-4)
        np.testing.assert_allclose(gb.numpy(),
            a_np.T @ np.ones((3, 5)), rtol=1e-4)

    def test_grad_batched(self):
        a_np = _rand((2, 3, 4), 1)
        b_np = _rand((2, 4, 5), 2)
        a = jt.array(a_np); b = jt.array(b_np)
        y = jt.linalg.einsum("bij,bjk->bik", a, b).sum()
        ga, gb = jt.grad(y, [a, b])
        self.assertEqual(tuple(ga.shape), (2, 3, 4))
        self.assertEqual(tuple(gb.shape), (2, 4, 5))
        # Compare to einsum-based reference for the gradient
        ones = np.ones((2, 3, 5), dtype=np.float32)
        ref_ga = np.einsum("bik,bjk->bij", ones, b_np)
        ref_gb = np.einsum("bij,bik->bjk", a_np, ones)
        np.testing.assert_allclose(ga.numpy(), ref_ga, rtol=1e-4)
        np.testing.assert_allclose(gb.numpy(), ref_gb, rtol=1e-4)

    def test_grad_ternary(self):
        shapes = [(2, 3), (3, 4), (4, 5)]
        arrs_np = [_rand(s, i + 7) for i, s in enumerate(shapes)]
        arrs = [jt.array(a) for a in arrs_np]
        y = jt.linalg.einsum("ij,jk,kl->il", *arrs).sum()
        grads = jt.grad(y, list(arrs))
        for g, s in zip(grads, shapes):
            self.assertEqual(tuple(g.shape), s)

    def test_bf16_dot(self):
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        a = jt.array(a_np).cast("bfloat16")
        b = jt.array(b_np).cast("bfloat16")
        out = jt.linalg.einsum("i,i->", a, b).cast("float32").numpy()
        np.testing.assert_allclose(out, np.array(32.0), rtol=1e-2)

    def test_bf16_matmul(self):
        a_np = _rand((3, 4), 11)
        b_np = _rand((4, 5), 12)
        a_bf = jt.array(a_np).cast("bfloat16")
        b_bf = jt.array(b_np).cast("bfloat16")
        out = jt.linalg.einsum("ij,jk->ik", a_bf, b_bf).cast("float32").numpy()
        np.testing.assert_allclose(out, a_np @ b_np, rtol=2e-2, atol=2e-2)

    def test_fp16_pair(self):
        a_np = _rand((2, 3, 4), 21)
        b_np = _rand((2, 4, 5), 22)
        a_h = jt.array(a_np).cast("float16")
        b_h = jt.array(b_np).cast("float16")
        out = jt.linalg.einsum("bij,bjk->bik", a_h, b_h).cast("float32").numpy()
        ref = np.einsum("bij,bjk->bik", a_np, b_np)
        np.testing.assert_allclose(out, ref, rtol=5e-3, atol=5e-3)

    def test_shape_mismatch_raises(self):
        a = jt.array(_rand((3,), 1))
        b = jt.array(_rand((4,), 2))
        with self.assertRaises((ValueError, AssertionError, RuntimeError)):
            jt.linalg.einsum("i,i->", a, b).sync()


@unittest.skipIf(not jt.has_cuda, "no cuda")
class TestLinalgEinsumCuda(TestLinalgEinsum):
    def setUp(self):
        self._old = jt.flags.use_cuda
        jt.flags.use_cuda = 1

    def tearDown(self):
        jt.flags.use_cuda = self._old


if __name__ == "__main__":
    unittest.main()

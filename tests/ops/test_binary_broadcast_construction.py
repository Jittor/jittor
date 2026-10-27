from _helpers import capability as _test_capability
import unittest

import jittor as jt
import numpy as np


class TestBinaryBroadcastConstruction(unittest.TestCase):
    """A broadcasting binary op is built without a forwarding BinaryOp.

    `make_binary` asks `BinaryOp::forward_before_construct` first, which builds
    the broadcast operands' binary op directly instead of constructing a
    BinaryOp only to have it forward. Everything a caller can observe must stay
    as it was: values, dtypes, shapes, gradients, dynamic shapes, and the error
    for operands that do not broadcast.
    """

    def _check(self):
        rng = np.random.RandomState(0)
        cases = [
            ((4, 1, 3), (5, 1)), ((3,), (2, 3)), ((2, 3), (1,)), ((1,), (4, 4)),
            ((2, 1), (1, 2)), ((3, 4), (3, 4)), ((), (2, 2)),
        ]
        for xs, ys in cases:
            for dtype in ("float32", "float64", "int32"):
                with self.subTest(x=xs, y=ys, dtype=dtype):
                    a = np.asarray(rng.rand(*xs) * 10).astype(dtype)
                    b = np.asarray(rng.rand(*ys) * 10 + 1).astype(dtype)
                    for op, ref in (("__add__", np.add), ("__mul__", np.multiply),
                                    ("__sub__", np.subtract), ("__lt__", np.less)):
                        out = getattr(jt.array(a), op)(jt.array(b))
                        # The oracle never broadcasts inside the binary op: its
                        # operands already have the result shape.
                        shape = list(np.broadcast_shapes(xs, ys))
                        oracle = getattr(jt.array(a).broadcast(shape), op)(jt.array(b).broadcast(shape))
                        self.assertEqual(tuple(out.shape), tuple(oracle.shape))
                        self.assertEqual(str(out.dtype), str(oracle.dtype))
                        np.testing.assert_array_equal(out.numpy(), oracle.numpy())
                        np.testing.assert_allclose(out.numpy(), ref(a, b), rtol=1e-5, atol=1e-6)
        x = jt.array(rng.rand(4, 3).astype("float32"))
        np.testing.assert_allclose((x * 2.0 + 1.0).numpy(), x.numpy() * 2.0 + 1.0, rtol=1e-6)

    def _check_grad(self):
        a = jt.array(np.random.RandomState(1).rand(4, 1, 3).astype("float32"))
        b = jt.array(np.random.RandomState(2).rand(5, 1).astype("float32"))
        ga, gb = jt.grad((a * b).sum(), [a, b])
        np.testing.assert_allclose(ga.numpy(), np.broadcast_to(b.numpy().sum(0), (4, 1, 1)) * np.ones((4, 1, 3)), rtol=1e-5)
        np.testing.assert_allclose(gb.numpy(), np.full((5, 1), a.numpy().sum()), rtol=1e-5)

    def _check_dynamic(self):
        x = jt.array(np.array([0.1, 0.9, 0.7, 0.2, 0.8], "float32"))
        picked = x[x > 0.5]                     # unknown extent until it runs
        np.testing.assert_allclose((picked * jt.array([2.0])).numpy(), [1.8, 1.4, 1.6], rtol=1e-6)
        np.testing.assert_allclose((jt.array([1.0]) + picked).numpy(), [1.9, 1.7, 1.8], rtol=1e-6)

    def _check_mismatch(self):
        with self.assertRaises(Exception) as caught:
            (jt.ones((2, 3)) + jt.ones((4, 3))).sync()
        self.assertIn("Shape not match for binary op", str(caught.exception))

    def _all(self):
        self._check()
        self._check_grad()
        self._check_dynamic()
        self._check_mismatch()

    def test_cpu(self):
        with jt.flag_scope(use_cuda=0):
            self._all()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No cuda found")
    def test_cuda(self):
        with jt.flag_scope(use_cuda=1):
            self._all()


if __name__ == "__main__":
    unittest.main()

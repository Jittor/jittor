# ***************************************************************
# torch-level regression test (#6): finite-difference gradient check
# for core ops (forward+backward correctness), no torch dependency.
#   python -m jittor.test.test_op_gradcheck
# ***************************************************************
import unittest, numpy as np
import jittor as jt
jt.flags.use_cuda = 0


def _numjac(fwd, x_np, eps=1e-3):
    x0 = x_np.astype(np.float64); g = np.zeros_like(x0)
    for idx in np.ndindex(*x0.shape):
        xp = x0.copy(); xp[idx] += eps
        xm = x0.copy(); xm[idx] -= eps
        g[idx] = (float(fwd(jt.array(xp.astype(np.float32))).numpy())
                  - float(fwd(jt.array(xm.astype(np.float32))).numpy())) / (2 * eps)
    return g


def _relerr(fwd, x_np):
    x = jt.array(x_np.astype(np.float32)); y = fwd(x)
    a = jt.grad(y, [x])[0].numpy().astype(np.float64)
    n = _numjac(fwd, x_np)
    return float(np.max(np.abs(a - n)) / (np.max(np.abs(n)) + 1e-6))


class TestOpGradcheck(unittest.TestCase):
    def test_grads(self):
        np.random.seed(0)
        x = np.random.randn(5); xp = np.abs(np.random.randn(5)) + 0.5
        cases = [
            ("mod",      lambda v: (v % 3.0).sum(),            x),
            ("maximum",  lambda v: jt.maximum(v, 0.0).sum(),   x),
            ("minimum",  lambda v: jt.minimum(v, 0.0).sum(),   x),
            ("pow",      lambda v: (jt.abs(v) ** 2.5).sum(),    x),
            ("mean",     lambda v: v.mean(),                   x),
            ("div",      lambda v: (v / 2.0).sum(),            x),
            ("exp",      lambda v: v.exp().sum(),              x),
            ("sqrt",     lambda v: (jt.abs(v) + 0.5).sqrt().sum(), xp),
            ("sigmoid",  lambda v: v.sigmoid().sum(),          x),
            ("relu",     lambda v: jt.nn.relu(v).sum(),        x),
        ]
        for name, fwd, xv in cases:
            with self.subTest(op=name):
                self.assertLess(_relerr(fwd, xv), 2e-2, f"{name} grad mismatch vs finite-diff")


if __name__ == '__main__':
    unittest.main()

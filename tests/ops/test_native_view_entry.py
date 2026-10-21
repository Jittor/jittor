"""``Var.view``/``reshape`` built natively agree with the Python path.

A contiguous plain Var reshaped to exact ints goes to ``_fast_view``
(``src/bindings/pyjt/py_compat_fast.h``); the oracle is the Python body of
``view``, reached by making the native entry decline.
"""
from _helpers import capability as _test_capability

import contextlib
import unittest

import numpy as np

import jittor as jt
from jittor._core import var as var_module


@contextlib.contextmanager
def python_path():
    saved = var_module._fast_view
    var_module._fast_view = lambda x, shape: None
    try:
        yield
    finally:
        var_module._fast_view = saved


_SHAPES = [((6, 4),), (-1,), (4, -1), ((24,),), ([2, 12],), (2, 3, 4), (1, 24, 1), ((-1, 2),)]


class _View:
    use_cuda = 0

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=self.use_cuda)
        self._scope.__enter__()

    def tearDown(self):
        self._scope.__exit__(None, None, None)

    def test_values_shapes_and_dtypes(self):
        for dtype in ("float32", "int32", "bool"):
            raw = (np.arange(24) % 5).reshape(2, 3, 4).astype(dtype)
            for args in _SHAPES:
                for method in ("view", "reshape"):
                    with self.subTest(dtype=dtype, args=args, method=method):
                        got = getattr(jt.array(raw), method)(*args)
                        with python_path():
                            want = getattr(jt.array(raw), method)(*args)
                        self.assertEqual(tuple(got.shape), tuple(want.shape))
                        self.assertEqual(str(got.dtype), str(want.dtype))
                        np.testing.assert_array_equal(got.numpy(), want.numpy())

    def test_write_through_the_view_reaches_the_base(self):
        for path in (contextlib.nullcontext, python_path):
            with self.subTest(path=path.__name__):
                with path():
                    x = jt.array(np.arange(6, dtype="float32"))
                    v = x.view(2, 3)
                    v.assign(v * 0 + 5)
                self.assertEqual(x.numpy().tolist(), [5.0] * 6)

    def test_gradient(self):
        x = jt.array(np.random.RandomState(0).rand(2, 3, 4).astype("float32"))
        w = jt.array(np.arange(24, dtype="float32").reshape(6, 4))
        g = jt.grad((x.reshape(6, 4) * w).sum(), x)
        with python_path():
            g2 = jt.grad((x.reshape(6, 4) * w).sum(), x)
        np.testing.assert_array_equal(g.numpy(), g2.numpy())

    def test_declined_forms(self):
        x = jt.array(np.arange(24, dtype="float32").reshape(4, 6))
        t = x.transpose(1, 0)
        for value, args in ((t, (24,)), (x, (np.int64(6), 4)), (x, (jt.array(6), 4)), (x, ((6, 4),))):
            with self.subTest(args=args):
                got = value.reshape(*args)
                with python_path():
                    want = value.reshape(*args)
                np.testing.assert_array_equal(got.numpy(), want.numpy())
        with self.assertRaises(Exception):
            x.view(5, 5)


class TestNativeViewCPU(_View, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestNativeViewCUDA(_View, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()

"""``x[basic]`` and ``jt.concat`` reach the native paths and change nothing.

Both used to be built in Python on every call; they now try the native
builders the Torch frontend uses (``src/bindings/pyjt/py_compat_fast.h``)
first, and fall back to the Python path whenever the native one declines.
The oracle here is that Python path itself, reached by hiding the native
entries: values, dtypes, shapes, gradients and the view a basic index records
(a write through it reaches the base) must all agree.
"""

from _helpers import capability as _test_capability

import contextlib
import unittest

import numpy as np

import jittor as jt
from jittor._runtime import dispatch


@contextlib.contextmanager
def python_paths():
    saved = (getattr(dispatch.try_dispatch, "_native_basic_getitem", None),
             getattr(dispatch.select_kernel, "_native_concat", None))
    dispatch.try_dispatch._native_basic_getitem = None
    dispatch.select_kernel._native_concat = None
    try:
        yield
    finally:
        dispatch.try_dispatch._native_basic_getitem, dispatch.select_kernel._native_concat = saved


_INDICES = [3, -1, slice(1, 4), slice(None, None, 2), slice(-3, None), (1, 2), (slice(0, 2), 1),
            (None, 1), (Ellipsis, 1), (1, Ellipsis), (slice(1, 3), None, slice(None, None, -1)),
            (), Ellipsis, None]


class _Entries:
    use_cuda = 0

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=self.use_cuda)
        self._scope.__enter__()

    def tearDown(self):
        self._scope.__exit__(None, None, None)

    def _base(self, dtype="float32"):
        return jt.array(np.arange(60).reshape(4, 5, 3).astype(dtype))

    def test_entries_are_bound(self):
        self.assertIsNotNone(getattr(dispatch.try_dispatch, "_native_basic_getitem", None))
        self.assertIsNotNone(getattr(dispatch.select_kernel, "_native_concat", None))

    def test_basic_getitem_matches_the_python_path(self):
        for dtype in ("float32", "int32", "bool"):
            for index in _INDICES:
                with self.subTest(dtype=dtype, index=index):
                    got = self._base(dtype)[index]
                    with python_paths():
                        want = self._base(dtype)[index]
                    self.assertEqual(str(got.dtype), str(want.dtype))
                    self.assertEqual(tuple(got.shape), tuple(want.shape))
                    np.testing.assert_array_equal(got.numpy(), want.numpy())

    def test_a_write_through_a_basic_view_reaches_the_base(self):
        for index in (1, slice(1, 3), (slice(None), 2)):
            with self.subTest(index=index):
                x = self._base()
                view = x[index]
                view.assign(view * 0 - 7)
                with python_paths():
                    y = self._base()
                    view2 = y[index]
                    view2.assign(view2 * 0 - 7)
                np.testing.assert_array_equal(x.numpy(), y.numpy())

    def test_basic_getitem_gradient(self):
        x = jt.array(np.random.RandomState(0).rand(4, 5).astype("float32"))
        g = jt.grad((x[1:3, ::2] * 3).sum() + x[0].sum(), x)
        with python_paths():
            g2 = jt.grad((x[1:3, ::2] * 3).sum() + x[0].sum(), x)
        np.testing.assert_array_equal(g.numpy(), g2.numpy())

    def test_indices_the_native_path_declines_still_work(self):
        x = self._base()
        mask = x > 30
        idx = jt.array([0, 2])
        for index in (mask, idx, (idx, 1), [0, 2], np.int64(1)):
            with self.subTest(index=type(index).__name__):
                got = x[index]
                with python_paths():
                    want = x[index]
                np.testing.assert_array_equal(got.numpy(), want.numpy())

    def test_concat_matches_the_python_path(self):
        rng = np.random.RandomState(1)
        cases = [
            ([(2, 3), (4, 3)], 0, "float32"), ([(2, 3), (2, 1), (2, 5)], 1, "float32"),
            ([(2, 3), (2, 3)], -1, "int32"), ([(3,), (0,), (2,)], 0, "float32"),
            ([(2, 2)] * 70, 0, "float32"), ([(2, 3), (2, 3)], 0, "uint8"),
            ([(1, 2, 3), (1, 2, 3)], 2, "float16"),
        ]
        for shapes, dim, dtype in cases:
            with self.subTest(shapes=shapes[:3], dim=dim, dtype=dtype):
                arrays = [rng.rand(*s).astype(dtype) * 10 for s in shapes]
                got = jt.concat([jt.array(a) for a in arrays], dim)
                with python_paths():
                    want = jt.concat([jt.array(a) for a in arrays], dim)
                self.assertEqual(str(got.dtype), str(want.dtype))
                np.testing.assert_array_equal(got.numpy(), want.numpy())
        # mixed dtypes promote exactly as before
        a, b = jt.array([1, 2], "int32"), jt.array([0.5], "float32")
        with python_paths():
            want = jt.concat([a, b])
        got = jt.concat([a, b])
        self.assertEqual(str(got.dtype), str(want.dtype))
        np.testing.assert_array_equal(got.numpy(), want.numpy())

    def test_concat_and_stack_gradients(self):
        x = jt.array(np.random.RandomState(2).rand(3, 4).astype("float32"))
        y = jt.array(np.random.RandomState(3).rand(2, 4).astype("float32"))
        loss = lambda: (jt.concat([x, y]) * jt.arange(5).reshape(5, 1)).sum() + (jt.stack([x, x]) ** 2).sum()
        gx, gy = jt.grad(loss(), [x, y])
        with python_paths():
            gx2, gy2 = jt.grad(loss(), [x, y])
        np.testing.assert_allclose(gx.numpy(), gx2.numpy(), rtol=1e-6)
        np.testing.assert_allclose(gy.numpy(), gy2.numpy(), rtol=1e-6)

    def test_concat_rejects_what_it_rejected(self):
        for args in (([jt.ones((2, 3)), jt.ones((2, 4))], 0), ([jt.ones((2, 3))], 5), ([], 0)):
            with self.subTest(args=len(args[0])):
                with python_paths():
                    try:
                        jt.concat(*args)
                    except Exception as error:
                        want = (type(error), str(error))
                    else:
                        want = None
                try:
                    jt.concat(*args)
                except Exception as error:
                    got = (type(error), str(error))
                else:
                    got = None
                self.assertEqual(got, want)


class TestNativeEntriesCPU(_Entries, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestNativeEntriesCUDA(_Entries, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()

"""``x[index]``, ``x[index] = v`` and ``jt.concat`` reach the native paths
and change nothing.

They used to be built in Python on every call; they now try the native
builders (``src/bindings/pyjt/py_compat_fast.h``) first, and fall back to the
Python path whenever the native one declines. The oracle here is that Python
path itself, reached by hiding the native entries: values, dtypes, shapes,
gradients and the view a basic index records (a write through it reaches the
base) must all agree. Index Vars -- a bool mask, an integer index, a tuple
holding them -- take the native path too, and so do assignments.
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
             getattr(dispatch.select_kernel, "_native_concat", None),
             getattr(dispatch.try_dispatch, "_native_setitem", None))
    dispatch.try_dispatch._native_basic_getitem = None
    dispatch.select_kernel._native_concat = None
    dispatch.try_dispatch._native_setitem = None
    try:
        yield
    finally:
        (dispatch.try_dispatch._native_basic_getitem, dispatch.select_kernel._native_concat,
         dispatch.try_dispatch._native_setitem) = saved


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

    def _gather_indices(self):
        # Every array is drawn here, once: each index is built twice, for the
        # native and the Python path, and both must see the same values.
        rng = np.random.RandomState(4)
        m1 = rng.rand(4) > 0.5
        m2 = rng.rand(4, 5) > 0.5
        m3 = rng.rand(4, 5, 3) > 0.5
        idx = np.array([3, 0, 2, 2, -1])
        jdx = np.array([4, 1, 0, 0, 2])
        return [
            ("mask1", lambda: jt.array(m1)), ("mask2", lambda: jt.array(m2)),
            ("mask3", lambda: jt.array(m3)),
            ("mask_none", lambda: jt.array(np.zeros(4, bool))),
            ("mask_all", lambda: jt.array(np.ones((4, 5), bool))),
            ("index", lambda: jt.array(idx)), ("index32", lambda: jt.array(idx.astype("int32"))),
            ("index2d", lambda: jt.array(idx[:4].reshape(2, 2))),
            ("tuple", lambda: (jt.array(idx), jt.array(jdx))),
            ("tuple_slice", lambda: (jt.array(idx), slice(1, 4))),
            ("slice_tuple", lambda: (slice(None, None, 2), jt.array(jdx))),
            ("int_tuple", lambda: (1, jt.array(jdx))),
            ("none_tuple", lambda: (None, jt.array(idx))),
            ("ellipsis_tuple", lambda: (Ellipsis, jt.array(np.array([2, 0])))),
        ]

    def test_gather_getitem_matches_the_python_path(self):
        for dtype in ("float32", "int32", "bool"):
            for name, index in self._gather_indices():
                with self.subTest(dtype=dtype, index=name):
                    got = self._base(dtype)[index()]
                    with python_paths():
                        want = self._base(dtype)[index()]
                    self.assertEqual(str(got.dtype), str(want.dtype))
                    self.assertEqual(tuple(got.shape), tuple(want.shape))
                    np.testing.assert_array_equal(got.numpy(), want.numpy())

    def test_gather_getitem_gradient(self):
        for name, index in self._gather_indices():
            with self.subTest(index=name):
                def grad():
                    x = jt.array(np.random.RandomState(0).rand(4, 5, 3).astype("float32"))
                    return jt.grad((x[index()] * 3).sum(), x).numpy()
                got = grad()
                with python_paths():
                    want = grad()
                np.testing.assert_array_equal(got, want)

    def test_gather_getitem_follows_a_replaced_getitem(self):
        # `_fast_getitem` hands index Vars to `Var.getitem` only while it is
        # jittor's own; a replacement must still see every such call.
        from jittor.ops import indexing
        seen = []
        original = jt.Var.getitem

        def recording(x, slices, *rest):
            seen.append(type(slices).__name__)
            return original(x, slices, *rest)

        jt.Var.getitem = recording
        try:
            x = self._base()
            out = x[x > 30]
        finally:
            jt.Var.getitem = original
        self.assertIs(jt.Var.getitem, indexing.var_getitem)
        self.assertTrue(seen)
        np.testing.assert_array_equal(out.numpy(), np.arange(31, 60, dtype="float32"))

    def _assignments(self):
        # As in `_gather_indices`: values are drawn once and only wrapped per
        # call, so both paths assign the same numbers.
        rng = np.random.RandomState(5)
        m1 = rng.rand(4) > 0.5
        m3 = rng.rand(4, 5, 3) > 0.5
        idx = np.array([0, 3, 3])
        drawn = {}

        def f(*shape):
            if shape not in drawn:
                drawn[shape] = rng.rand(*shape).astype("float32")
            return jt.array(drawn[shape])
        return [
            ("slice", lambda x: x.__setitem__(slice(1, 3), f(2, 5, 3))),
            ("int_scalar", lambda x: x.__setitem__(2, 7.5)),
            ("tuple_int", lambda x: x.__setitem__((1, slice(0, 3)), 3)),
            ("none_ellipsis", lambda x: x.__setitem__((None, Ellipsis, 1), -2.0)),
            ("index_bool_value", lambda x: x.__setitem__(jt.array(idx), True)),
            ("index_var", lambda x: x.__setitem__(jt.array(np.array([1, 2])), f(2, 5, 3))),
            ("index_tuple", lambda x: x.__setitem__((jt.array(idx), jt.array(np.array([1, 1, 4]))), f(3, 3))),
            ("mask_full_float", lambda x: x.__setitem__(jt.array(m3), 0.25)),
            ("mask_full_int", lambda x: x.__setitem__(jt.array(m3), 4)),
            ("mask_full_one", lambda x: x.__setitem__(jt.array(m3), f(1))),
            ("mask_full_values", lambda x: x.__setitem__(jt.array(m3), jt.arange(int(m3.sum())).float32())),
            ("mask_rows_values", lambda x: x.__setitem__(jt.array(m1), f(int(m1.sum()), 5, 3))),
            ("mask_rows_scalar", lambda x: x.__setitem__(jt.array(m1), -1.0)),
        ]

    def test_setitem_matches_the_python_path(self):
        for dtype in ("float32", "int32"):
            for name, assign in self._assignments():
                with self.subTest(dtype=dtype, case=name):
                    x = self._base(dtype)
                    assign(x)
                    with python_paths():
                        y = self._base(dtype)
                        assign(y)
                    self.assertEqual(str(x.dtype), str(y.dtype))
                    np.testing.assert_array_equal(x.numpy(), y.numpy())

    def test_setitem_gradient_and_view_write(self):
        def scenario():
            rng = np.random.RandomState(6)
            x = jt.array(rng.rand(6, 5).astype("float32"))
            w = jt.array(rng.rand(3, 5).astype("float32"))
            y = x * 1.0
            y[jt.array(np.array([0, 2, 4]))] = w * 2
            y[y > 0.9] = 0.0
            gx, gw = jt.grad((y * y).sum(), [x, w])
            base = jt.zeros((4, 4))
            view = base[1:3]
            view[jt.array(np.array([1]))] = 5.0
            return y.numpy(), gx.numpy(), gw.numpy(), base.numpy()
        got = scenario()
        with python_paths():
            want = scenario()
        for a, b in zip(got, want):
            np.testing.assert_array_equal(a, b)

    def test_assignments_the_native_path_declines_still_work(self):
        for name, index, value in (
                ("uint8 mask", lambda: jt.array(np.array([1, 0, 1, 0], "uint8")), 1.0),
                ("list", lambda: [0, 2], 2.0),
                ("numpy value", lambda: slice(0, 2), np.ones((2, 5, 3), "float32")),
                ("mask in tuple", lambda: (jt.array(np.array([True, False, True, False])), 1), 3.0)):
            with self.subTest(case=name):
                x = self._base()
                x[index()] = value
                with python_paths():
                    y = self._base()
                    y[index()] = value
                np.testing.assert_array_equal(x.numpy(), y.numpy())

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

    def test_views_written_in_place_while_recording_gradients(self):
        """The native paths read operands as the bindings do (`grad_input`).

        With gradients recorded, a view whose source was written in place is
        brought up to date when a binding reads it; reading the raw Var instead
        would build on what the view was before the write.
        """
        def scenario(op):
            source = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
            source.start_grad()
            s = source * 1.0
            v = s[1:3]
            s[1:2] = jt.array(np.full((1, 4), 100.0, "float32"))
            out = op(v)
            return out.numpy(), jt.grad(out.sum(), source).numpy()

        ops = {
            "getitem": lambda v: v[0] * 2,
            "concat": lambda v: jt.concat([v, v * 3], 0),
            "clamp": lambda v: v.clamp(2.0, 50.0),
        }
        for name, op in ops.items():
            with self.subTest(op=name):
                got = scenario(op)
                with python_paths():
                    saved = getattr(dispatch.try_dispatch, "_native_clamp", None)
                    dispatch.try_dispatch._native_clamp = None
                    try:
                        want = scenario(op)
                    finally:
                        dispatch.try_dispatch._native_clamp = saved
                np.testing.assert_array_equal(got[0], want[0])
                np.testing.assert_array_equal(got[1], want[1])

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

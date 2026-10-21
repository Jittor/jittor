"""``Var.clamp`` built natively gives what ``_clamp_cpu`` gives.

With only the default kernel registered, ``clamp`` hands Python int/float
bounds to the native builder (``_fast_clamp`` in
``src/bindings/pyjt/py_compat_fast.h``), which renders ``_clamp_cpu``'s graph
op for op. The oracle is ``_clamp_cpu`` itself, reached by hiding the native
entry: dtype, values (NaN positions included) and gradients must agree for
every dtype and every bound form the native path accepts, and the forms it
declines must still work.
"""

from _helpers import capability as _test_capability

import contextlib
import math
import unittest

import numpy as np

import jittor as jt
from jittor._runtime import dispatch


@contextlib.contextmanager
def python_path():
    saved = getattr(dispatch.try_dispatch, "_native_clamp", None)
    dispatch.try_dispatch._native_clamp = None
    try:
        yield
    finally:
        dispatch.try_dispatch._native_clamp = saved


_FLOATS = ("float16", "float32", "float64")
_INTS = ("int8", "int32", "int64", "uint8")
_BOUNDS = [(0, 5), (0.5, 4.25), (None, 3), (2, None), (None, 2.5), (1.5, None), (4, 1),
           (3.5, 0.5), (0, 0), (-1e9, 1e9), (2, 2.75), (2.25, 6), (float("nan"), 3),
           (1, float("nan")), (float("-inf"), float("inf")), (-3, 100)]


def _data(dtype):
    raw = np.array([-2.0, 0.0, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 9.0])
    if dtype in _FLOATS:
        raw = np.concatenate([raw, [np.nan, np.inf, -np.inf]])
    if dtype == "uint8":
        raw = np.abs(raw)
    return raw.astype(dtype)


class _Clamp:
    use_cuda = 0

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=self.use_cuda)
        self._scope.__enter__()

    def tearDown(self):
        self._scope.__exit__(None, None, None)

    def test_native_entry_is_bound_and_taken(self):
        self.assertIsNotNone(getattr(dispatch.try_dispatch, "_native_clamp", None))
        x = jt.array(_data("float32"))
        self.assertIsNot(jt.core._fast_clamp(x, 0.0, 1.0), NotImplemented)

    def test_values_and_dtypes_match_the_python_path(self):
        for dtype in _FLOATS + _INTS:
            for lo, hi in _BOUNDS:
                with self.subTest(dtype=dtype, lo=lo, hi=hi):
                    raw = _data(dtype)
                    got = jt.array(raw).clamp(lo, hi)
                    with python_path():
                        want = jt.array(raw).clamp(lo, hi)
                    self.assertEqual(str(got.dtype), str(want.dtype))
                    np.testing.assert_array_equal(got.numpy(), want.numpy())

    def test_gradients_match_the_python_path(self):
        for dtype in ("float32", "float64"):
            for lo, hi in ((0.5, 4.0), (2.0, 2.0), (None, 3.0), (1, None), (5, 1)):
                with self.subTest(dtype=dtype, lo=lo, hi=hi):
                    x = jt.array(_data(dtype)[:11])
                    w = jt.array(np.arange(11).astype(dtype))
                    g = jt.grad((x.clamp(lo, hi) * w).sum(), x)
                    with python_path():
                        g2 = jt.grad((x.clamp(lo, hi) * w).sum(), x)
                    np.testing.assert_array_equal(g.numpy(), g2.numpy())

    def test_nan_input_gradient_matches(self):
        x = jt.array(np.array([np.nan, 1.0, 9.0], "float32"))
        g = jt.grad(x.clamp(0.0, 5.0).sum(), x)
        with python_path():
            g2 = jt.grad(x.clamp(0.0, 5.0).sum(), x)
        np.testing.assert_array_equal(g.numpy(), g2.numpy())

    def test_declined_forms_still_work(self):
        x = jt.array(_data("float32"))
        cases = [(jt.array(1.0), 4.0), (np.float32(1), 3), (True, 3), (0, 2 ** 40),
                 (None, None)]
        for lo, hi in cases:
            with self.subTest(lo=type(lo).__name__, hi=type(hi).__name__):
                got = x.clamp(lo, hi)
                with python_path():
                    want = x.clamp(lo, hi)
                np.testing.assert_array_equal(got.numpy(), want.numpy())
        b = jt.array(np.array([True, False]))
        np.testing.assert_array_equal(b.clamp(0, 1).numpy(), jt.array(np.array([True, False])).clamp(0, 1).numpy())
        empty = jt.zeros((0,), "float32")
        self.assertIs(empty.clamp(0, 1), empty)

    def test_a_registered_kernel_is_still_dispatched(self):
        calls = []

        def kernel(x, lo, hi):
            calls.append((lo, hi))
            return x

        with dispatch.override_kernel("clamp", "cpu" if not self.use_cuda else "cuda", kernel):
            x = jt.array(_data("float32"))
            self.assertIs(x.clamp(0, 1), x)
        self.assertEqual(calls, [(0, 1)])


class TestNativeClampCPU(_Clamp, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestNativeClampCUDA(_Clamp, unittest.TestCase):
    use_cuda = 1


class TestToSpellingCache(unittest.TestCase):

    def test_spellings_give_what_parsing_gives(self):
        from jittor.ops import tensor_protocol
        x = jt.array(np.arange(4, dtype="float32"))
        for spelling in ("int64", "float32", "cpu", jt.int64, jt.float32, jt.bool):
            with self.subTest(spelling=spelling):
                for _ in range(2):  # the second call is answered from the table
                    out = x.to(spelling)
                    self.assertEqual(tensor_protocol._parse_to((spelling,), {}),
                                     tensor_protocol._TO_SPELLINGS.get(spelling,
                                     tensor_protocol._parse_to((spelling,), {})))
                self.assertEqual(out.numpy().tolist(), x.numpy().astype(
                    str(out.dtype) if str(out.dtype) != "bool" else bool).tolist())

    def test_a_rejected_spelling_is_not_remembered(self):
        from jittor.ops import tensor_protocol
        x = jt.array([1.0])
        for _ in range(2):
            with self.assertRaises(TypeError):
                x.to("cuda1")
        self.assertNotIn("cuda1", tensor_protocol._TO_SPELLINGS)

    def test_bound_builtins_are_not_cached(self):
        from jittor.ops import tensor_protocol
        x = jt.array([1.0])
        self.assertFalse(tensor_protocol._cacheable_spelling(x.numpy))
        self.assertTrue(tensor_protocol._cacheable_spelling(jt.int64))


if __name__ == "__main__":
    unittest.main()

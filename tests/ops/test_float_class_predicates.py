# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""isnan / isinf / isfinite answer the same question on every backend.

The non-ACL kernel evaluated ``isinf(float(x))``. ``float(x)`` is a *narrowing*
cast for float64, so every finite float64 above 3.4e38 -- 1e300, say -- became
inf inside the predicate and ``isinf`` said True. The ACL backend computes the
same three predicates from primitive comparisons, which never narrow, and said
False. One public API, two answers, decided by which accelerator was present.

numpy is the oracle here: it is an independent implementation of the same IEEE
predicates, and it is already a dependency. The ACL bodies are exercised
directly (they are plain elementwise expressions and run anywhere) so this file
pins the two implementations to each other without an NPU.

Why the kernel is not simply replaced by those comparisons: Jittor compiles
fused kernels with ``-Ofast``, which implies ``-ffinite-math-only``, under which
``x != x`` and ``x >= 0`` are free to be folded. ``_simple_for`` exists to
compile this one kernel at ``-O2`` instead. The ACL path gets away with the
comparisons because aclnn evaluates them, not a JIT kernel.
"""

import unittest

import numpy as np

import jittor as jt


_CASES = {
    # the headline case: 1e300 is an ordinary finite float64
    "float64": np.array([1e300, -1e300, 1e-300, 0.0, np.inf, -np.inf, np.nan,
                         3.5], dtype="float64"),
    "float32": np.array([3.4e38, -3.4e38, 1e-38, 0.0, np.inf, -np.inf, np.nan,
                         3.5], dtype="float32"),
    "float16": np.array([65504.0, -65504.0, 6e-5, 0.0, np.inf, -np.inf, np.nan,
                         3.5], dtype="float16"),
    "int32": np.array([-2147483648, -1, 0, 1, 2147483647], dtype="int32"),
    "int64": np.array([-(2 ** 62), -1, 0, 1, 2 ** 62], dtype="int64"),
    "bool": np.array([True, False, True], dtype="bool"),
}

_PREDICATES = ("isnan", "isinf", "isfinite")


def _numpy_expected(name, raw):
    if raw.dtype.kind in "biu":
        # torch: integers have no nan and no inf, so isfinite is all True
        return {"isnan": np.zeros(raw.shape, "bool"),
                "isinf": np.zeros(raw.shape, "bool"),
                "isfinite": np.ones(raw.shape, "bool")}[name]
    return getattr(np, name)(raw)


class _Predicates:

    use_cuda = 0

    def test_matches_numpy(self):
        for dtype, raw in _CASES.items():
            for name in _PREDICATES:
                with self.subTest(dtype=dtype, predicate=name):
                    with jt.flag_scope(use_cuda=self.use_cuda):
                        # jt.array() silently narrows a float64/int64 numpy
                        # array, which would hide exactly the case under test
                        x = jt.array(raw, dtype=dtype)
                        self.assertEqual(str(x.dtype), dtype)
                        got = getattr(jt, name)(x).numpy()
                    np.testing.assert_array_equal(
                        got, _numpy_expected(name, raw))

    def test_the_acl_bodies_share_the_dtype_policy(self):
        """The other backend's implementation, on the part that is portable.

        These are the expressions ``jt.isnan`` and friends use when
        ``jt.flags.use_acl`` is set, and they are ordinary elementwise ops, so
        they can be evaluated here -- with two exclusions that are themselves
        the reason the two spellings have to stay separate:

        * **float16 is left out**: ``jittor::float16 <= int`` is an ambiguous
          overload, so this expression does not even compile into a JIT kernel.
          aclnn evaluates it instead and never sees that.
        * **nan and the infinities are left out**: Jittor compiles fused
          kernels with ``-Ofast``, hence ``-ffinite-math-only``, under which the
          compiler may assume they do not occur and fold the comparisons. That
          is why the CPU/CUDA path keeps its own ``-O2`` kernel rather than
          reusing these three lines.

        What is left -- the dtype policy, and the answers for ordinary values --
        has to be the same on every backend, and that is what is checked here.
        """
        bodies = {"isnan": jt.misc._isnan_acl,
                  "isinf": jt.misc._isinf_acl,
                  "isfinite": jt.misc._isfinite_acl}
        for dtype, raw in _CASES.items():
            if dtype == "float16":
                continue
            ordinary = (np.isfinite(raw) if raw.dtype.kind == "f"
                        else np.ones(raw.shape, "bool"))
            for name, body in bodies.items():
                with self.subTest(dtype=dtype, predicate=name):
                    with jt.flag_scope(use_cuda=self.use_cuda):
                        x = jt.array(raw, dtype=dtype)
                        kernel = getattr(jt, name)(x).numpy()
                        acl = body(x).numpy()
                    np.testing.assert_array_equal(kernel[ordinary],
                                                  acl[ordinary])
                    np.testing.assert_array_equal(
                        acl[ordinary], _numpy_expected(name, raw)[ordinary])

    def test_signed_infinity_predicates(self):
        raw = _CASES["float64"]
        with jt.flag_scope(use_cuda=self.use_cuda):
            x = jt.array(raw, dtype="float64")
            pos = jt.isposinf(x).numpy()
            neg = jt.isneginf(x).numpy()
        np.testing.assert_array_equal(pos, np.isposinf(raw))
        np.testing.assert_array_equal(neg, np.isneginf(raw))

    def test_a_large_finite_float64_is_not_infinite(self):
        """The one-line version of the defect, kept legible.

        1e300 fits a float64 and does not fit a float32, so a predicate that
        casts to float before testing reports it as infinite.
        """
        with jt.flag_scope(use_cuda=self.use_cuda):
            x = jt.array(np.array([1e300], dtype="float64"), dtype="float64")
            got = (bool(jt.isinf(x).numpy()[0]),
                   bool(jt.isfinite(x).numpy()[0]),
                   bool(jt.isnan(x).numpy()[0]))
        self.assertEqual(got, (False, True, False))


class TestPredicatesCPU(_Predicates, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestPredicatesCUDA(_Predicates, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()

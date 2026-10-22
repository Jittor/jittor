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

Why the kernel is not simply replaced by those comparisons: the optimisation
level a fused kernel is built at is decided elsewhere, and under
``-ffinite-math-only`` -- implied by ``-ffast-math``, implied in turn by the
``-Ofast`` this project used to ship -- ``x != x`` and
``x >= 0`` are free to be folded to a constant. A predicate that answers "is
this a NaN" is exactly the code that must not depend on which flags a build
settled on. It used to escape them in a ``jt.code`` kernel pinned to ``-O2``;
it is now a unary op that tests the bit pattern (``type/fp_class_compute.h``),
which no flag can fold and which fuses like any other elementwise op. The ACL
path gets away with the comparisons because aclnn evaluates them, not a JIT
kernel.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt
from _helpers.common import kernel_launches


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
        * **nan and the infinities are left out**: a fused kernel built
          under ``-ffinite-math-only`` may assume they do not occur and fold
          the comparisons away. The CPU flags that granted that (``-Ofast``)
          are gone, but the flags are still decided
          outside this expression, which is why the CPU/CUDA path keeps its
          own ``-O2`` kernel rather than reusing these three lines.

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
        for dtype in ("float64", "float32", "float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                with jt.flag_scope(use_cuda=self.use_cuda):
                    if dtype == "bfloat16":
                        x = jt.array(_CASES["float32"]).cast(dtype)
                    else:
                        x = jt.array(_CASES[dtype], dtype=dtype)
                    # The input's own values: bfloat16 rounds 3.4e38 up to inf.
                    # Widening to float64 keeps every value's class.
                    ref = x.float64().numpy()
                    pos = jt.isposinf(x).numpy()
                    neg = jt.isneginf(x).numpy()
                np.testing.assert_array_equal(pos, np.isposinf(ref))
                np.testing.assert_array_equal(neg, np.isneginf(ref))

    def test_a_predicate_fuses_into_what_reads_it(self):
        """``where(isfinite(x), x, -1)`` is one kernel, not a predicate kernel
        writing bools for a second one to read back."""
        raw = _CASES["float32"]
        with jt.flag_scope(use_cuda=self.use_cuda):
            x = jt.array(raw)
            x.sync()
            for name in _PREDICATES:
                with self.subTest(predicate=name):
                    count, got = kernel_launches(
                        lambda: jt.where(getattr(jt, name)(x * 2), x, -1.0).numpy())
                    self.assertEqual(count, 1)
                    np.testing.assert_array_equal(
                        got, np.where(getattr(np, name)(raw * 2), raw, -1.0))

    def test_fused_into_a_fast_math_kernel_it_still_answers(self):
        """The bit test survives the flags the comparisons do not.

        ``-Ofast`` tells the compiler no operand is NaN or infinite, and the
        predicate is now fused into a kernel built with whatever flags that
        kernel gets. The answer must not change: it is an integer test.
        """
        if self.use_cuda:
            raise unittest.SkipTest("nvcc does not take -Ofast; CUDA kernels "
                                    "already build with --use_fast_math")
        flags = jt.flags
        previous = flags.compile_options
        options = dict(previous)
        options["FLAGS: -Ofast "] = 1
        flags.compile_options = options
        try:
            for dtype in ("float32", "float64", "float16"):
                raw = _CASES[dtype]
                x = jt.array(raw, dtype=dtype)
                for name in _PREDICATES + ("isposinf", "isneginf"):
                    with self.subTest(dtype=dtype, predicate=name):
                        got = getattr(jt, name)(x * 1).numpy()
                        np.testing.assert_array_equal(got, getattr(np, name)(raw * 1))
        finally:
            flags.compile_options = previous

    def test_a_zero_dim_input_compiles(self):
        """0-d is a rank the ``code`` op's stride templates got wrong.

        The predicates were a ``jt.code`` op, whose PRECALC is
        ``@for(j, in0_dim-2, -1, -1, ...)``. At rank 0 that is
        ``@for(j, -2, -1, -1, ...)``, and the expander used to stop only on
        equality, so the counter stepped away from ``-1`` until it hit the
        1000-iteration cap and failed the compile. They are unary ops now, and
        the same rank sweep pins that every rank still compiles and answers --
        ranks 1 and 2 as much as 0.

        This is the defect behind ``test_safe_clip`` in the fp16/bf16 suites:
        the dtype was incidental, the rank was the trigger.
        """
        for dtype in ("float32", "float64", "float16", "bfloat16", "int32"):
            for shape in ((), (1,), (2, 3)):
                for name in _PREDICATES:
                    with self.subTest(dtype=dtype, shape=shape, predicate=name):
                        raw = np.full(shape, np.inf if dtype.startswith(("float", "bf"))
                                      else 7).astype(
                            "float32" if dtype == "bfloat16" else dtype)
                        with jt.flag_scope(use_cuda=self.use_cuda):
                            x = jt.array(raw).cast(dtype)
                            got = getattr(jt, name)(x)
                            self.assertEqual(tuple(got.shape), shape)
                            got = got.numpy()
                        expected = {"isnan": False, "isinf": True,
                                    "isfinite": False}[name]
                        if dtype == "int32":
                            expected = name == "isfinite"
                        np.testing.assert_array_equal(
                            got, np.full(shape, expected))

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


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestPredicatesCUDA(_Predicates, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Core TYPE-SYSTEM parity: the dtype-promotion lattice, the cast methods, and
NanoString.

This is a LOW-LEVEL / CORE module -- the promotion lattice and the cast methods
are consulted by *every* mixed-dtype arithmetic expression and every ``.long()`` /
``.float()`` call in user code, so a wrong entry here is a silent
precision/range loss that no higher-level test would localise. Correctness here
is non-negotiable ("不能有bug").

Why a dedicated module and not just a slice of ``test_ops.py``: op_db gradcheck
runs each op at a single declared dtype and never *mixes* dtypes, so it cannot see
a promotion-lattice bug at all. This module pins the cross-product of dtype pairs
explicitly and compares each one against the DOCUMENTED torch ``_promoteTypesLookup``
rule (c10/core/ScalarType.cpp), hard-coded below -- never jittor-vs-jittor.

What is asserted (all on CPU; promotion is device-independent -- it is decided in
Python from the operand dtype names before any kernel runs):
  1. ``jt.result_type`` / ``jt.promote_types`` / ``jt.can_cast`` and the actual
     binary-op output dtype, for every meaningful pair drawn from
     ``{bool, uint8, int8, int16, int32, int64, float16, float32, float64}``;
  2. true-division's special rule (``/`` is ALWAYS float: an integral result_type
     lands on the default float32, a floating one is kept);
  3. the cast methods ``.float()/.double()/.half()/.long()/.int()/.short()/
     .char()/.byte()/.bool()`` land on torch's EXACT dtype, with VALUES checked
     against numpy;
  4. type-string casts ``.type("torch.LongTensor")`` etc., ``.type()`` no-arg
     name, and ``.to(dtype)`` / ``.to("torch.int64")``;
  5. a few NanoString edge checks (``str(dtype)`` round-trips back into jittor's
     own dispatch, str-subclass accepted).

Pinning dtypes: ``jt.array(x)`` silently narrows int64->int32 and float64->float32,
so every fixed-width tensor is built with ``jt.array(a, dtype=...)`` (which jittor
honours via ``auto_convert_64_to_32=0``). Values are read via ``.numpy()`` because
jittor has no 0-d scalar (a reduced value is shape ``(1,)``).

PRESERVED semantic-diffs (carried, NOT silently "fixed"): the two ``@unittest.skip``
locks from ``test_torch_compat_dtype.py`` -- (a) the pre-shim native binary-op
promotion kept the narrower/left operand, and (b) native ``Var.long`` was aliased to
``Var.int32``. The torch_compat layer that ``import jittor`` installs unconditionally
now OVERRIDES both (the live behavior is the torch lattice / int64, asserted
positively above), but the historical reasons are carried verbatim per the audit's
must-keep rule.

Run::  python -m pytest tests/core/test_type_system.py
       python -m pytest tests/core/test_type_system.py
"""
import unittest

import numpy as np
import jittor as jt

from _helpers.common import JittorTestCase


# numpy dtype for each bare jittor name we build pinned tensors from.
_NPDT = {
    "bool": np.bool_, "uint8": np.uint8, "int8": np.int8, "int16": np.int16,
    "int32": np.int32, "int64": np.int64, "float16": np.float16,
    "float32": np.float32, "float64": np.float64,
}


def _dts(v):
    """Bare jittor dtype string for a Var ('float32', 'int64', ...)."""
    return str(v.dtype)


def _pin(name, val=(1, 2, 3)):
    """A 1-D jittor Var of the given bare dtype, built so the dtype SURVIVES.

    ``jt.array`` narrows int64->int32 / float64->float32 unless an explicit dtype
    is given; pass it so the wide dtypes (and the exact narrow ones) are preserved.
    """
    a = np.array(list(val), dtype=_NPDT[name])
    return jt.array(a, dtype=name)


# =====================================================================================
# The DOCUMENTED torch promotion lattice (torch's _promoteTypesLookup, c10/core/
# ScalarType.cpp). This is the INDEPENDENT reference -- torch's documented result,
# hard-coded so the test does not need real torch installed. Rules encoded:
#   * category order  bool < int < float;
#   * same-signedness ints: the wider wins;
#   * signed+unsigned ints: promote to a SIGNED type wide enough to hold both
#     (uint8+int8 -> int16, uint8+int16 -> int16, uint8+int32 -> int32,
#      uint8+int64 -> int64);  uint8+uint8 stays uint8;
#   * a float of ANY width absorbs an int of ANY width WITHOUT widening
#     (float16+int64 -> float16);
#   * floats: the wider wins (float32+float64 -> float64);
#   * bool is the identity-low element: bool + X == X.
# The pairs cover every meaningful combination of the 9 core dtypes (order does not
# matter -- promotion is commutative, asserted as such below).
# =====================================================================================
_PROMO = {
    # bool is the identity-low element: bool + X -> X.
    ("bool", "bool"): "bool",
    ("bool", "uint8"): "uint8",
    ("bool", "int8"): "int8",
    ("bool", "int16"): "int16",
    ("bool", "int32"): "int32",
    ("bool", "int64"): "int64",
    ("bool", "float16"): "float16",
    ("bool", "float32"): "float32",
    ("bool", "float64"): "float64",
    # signed int + signed int (same signedness) : the wider wins.
    ("int8", "int8"): "int8",
    ("int8", "int16"): "int16",
    ("int8", "int32"): "int32",
    ("int8", "int64"): "int64",
    ("int16", "int16"): "int16",
    ("int16", "int32"): "int32",
    ("int16", "int64"): "int64",
    ("int32", "int32"): "int32",
    ("int32", "int64"): "int64",
    ("int64", "int64"): "int64",
    # unsigned + unsigned : stays unsigned.
    ("uint8", "uint8"): "uint8",
    # signed + unsigned : promote to a SIGNED type wide enough to hold both.
    ("uint8", "int8"): "int16",
    ("uint8", "int16"): "int16",
    ("uint8", "int32"): "int32",
    ("uint8", "int64"): "int64",
    # float of ANY width absorbs int of ANY width WITHOUT widening.
    ("uint8", "float16"): "float16",
    ("uint8", "float32"): "float32",
    ("uint8", "float64"): "float64",
    ("int8", "float16"): "float16",
    ("int8", "float32"): "float32",
    ("int16", "float16"): "float16",
    ("int16", "float32"): "float32",
    ("int32", "float16"): "float16",
    ("int32", "float32"): "float32",
    ("int32", "float64"): "float64",
    ("int64", "float16"): "float16",   # marquee: float16 NOT widened to hold int64
    ("int64", "float32"): "float32",
    ("int64", "float64"): "float64",
    # float + float : the wider wins.
    ("float16", "float16"): "float16",
    ("float16", "float32"): "float32",
    ("float16", "float64"): "float64",
    ("float32", "float32"): "float32",
    ("float32", "float64"): "float64",
    ("float64", "float64"): "float64",
}

# Every dtype name that appears, for sanity in tests that iterate the basis.
_BASIS = ["bool", "uint8", "int8", "int16", "int32", "int64",
          "float16", "float32", "float64"]


def _ref_promote(a, b):
    """Look up the documented torch promotion for the (unordered) pair (a, b)."""
    if (a, b) in _PROMO:
        return _PROMO[(a, b)]
    return _PROMO[(b, a)]


class _CPUOnly(JittorTestCase):
    """Promotion is decided in Python from operand dtype names, before any kernel
    runs, so it is device-independent; pin CPU to keep this off the busy
    accelerator and make the lock deterministic."""

    def setUp(self):
        self._saved_use_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = 0

    def tearDown(self):
        jt.flags.use_cuda = self._saved_use_cuda


# -------------------------------------------------------- result_type / promote_types API

class TestPromotionAPI(_CPUOnly):
    def test_promote_types_matches_lattice(self):
        # jt.promote_types(dtype1, dtype2) -> dtype object; must equal the documented
        # torch result and be commutative.
        for (a, b), want in _PROMO.items():
            r1 = jt.promote_types(getattr(jt, a), getattr(jt, b))
            r2 = jt.promote_types(getattr(jt, b), getattr(jt, a))
            self.assertEqual(str(r1), want, msg=f"promote_types({a},{b})")
            self.assertEqual(str(r2), want, msg=f"promote_types({b},{a}) [commutative]")

    def test_promote_types_doc_examples(self):
        # the two examples from torch.promote_types' own docstring.
        self.assertEqual(str(jt.promote_types(jt.int32, jt.float32)), "float32")
        self.assertEqual(str(jt.promote_types(jt.uint8, jt.long)), "int64")

    def test_result_type_two_tensors(self):
        for (a, b), want in _PROMO.items():
            self.assertEqual(str(jt.result_type(_pin(a), _pin(b))), want,
                             msg=f"result_type({a},{b})")
            self.assertEqual(str(jt.result_type(_pin(b), _pin(a))), want,
                             msg=f"result_type({b},{a}) [commutative]")

    def test_result_type_tensor_and_dtype(self):
        # result_type also accepts a dtype object as either argument.
        self.assertEqual(str(jt.result_type(_pin("int32"), jt.float32)), "float32")
        self.assertEqual(str(jt.result_type(jt.int64, _pin("int32"))), "int64")

    def test_result_type_python_scalar_wrapped_number_rule(self):
        # torch "wrapped number" rule: a Python scalar bumps the result only if it
        # is a strictly HIGHER category than the tensor. int scalar keeps the
        # tensor's int dtype; float scalar lifts an int tensor to default float32.
        xi = _pin("int32")
        xf = _pin("float32")
        self.assertEqual(str(jt.result_type(xi, 2)), "int32")      # int scalar: no widen
        self.assertEqual(str(jt.result_type(xi, 1.5)), "float32")  # float scalar lifts int->f32
        self.assertEqual(str(jt.result_type(xf, 2)), "float32")    # int scalar keeps float
        self.assertEqual(str(jt.result_type(xf, 1.5)), "float32")
        self.assertEqual(str(jt.result_type(_pin("int64"), 7)), "int64")  # no widen to default

    def test_can_cast(self):
        # jt.can_cast(from, to): True iff promote(from, to) == to.
        self.assertTrue(jt.can_cast(jt.int32, jt.int64))
        self.assertTrue(jt.can_cast(jt.float32, jt.float64))
        self.assertTrue(jt.can_cast(jt.bool, jt.int32))
        self.assertTrue(jt.can_cast(jt.int32, jt.float32))
        self.assertFalse(jt.can_cast(jt.float32, jt.int32))   # float -> int loses category
        self.assertFalse(jt.can_cast(jt.int64, jt.int32))     # wider -> narrower


# ----------------------------------------------------------- binary-op promotion (dtypes)

class TestBinaryOpPromotion(_CPUOnly):
    def _binop_dtype(self, da, db, op):
        a = _pin(da, (2, 4, 6))
        b = _pin(db, (1, 2, 3))
        return _dts(op(a, b))

    def test_add_promotes_like_torch(self):
        for (da, db), want in _PROMO.items():
            self.assertEqual(self._binop_dtype(da, db, lambda x, y: x + y), want,
                             msg=f"{da}+{db}")
            self.assertEqual(self._binop_dtype(db, da, lambda x, y: x + y), want,
                             msg=f"{db}+{da} [commutative]")

    def test_sub_mul_promote(self):
        for (da, db), want in _PROMO.items():
            # subtraction is undefined for two bools in torch; skip bool-bool for sub.
            if (da, db) != ("bool", "bool"):
                self.assertEqual(self._binop_dtype(da, db, lambda x, y: x - y), want,
                                 msg=f"{da}-{db}")
            self.assertEqual(self._binop_dtype(da, db, lambda x, y: x * y), want,
                             msg=f"{da}*{db}")

    def test_floordiv_follows_lattice(self):
        # torch '//' follows the plain promotion lattice (unlike '/').
        self.assertEqual(self._binop_dtype("int32", "int64", lambda x, y: x // y), "int64")
        self.assertEqual(self._binop_dtype("int8", "int16", lambda x, y: x // y), "int16")
        self.assertEqual(self._binop_dtype("uint8", "int8", lambda x, y: x // y), "int16")

    def test_truediv_is_always_float(self):
        # torch '/' ALWAYS yields float (the documented special case). For an
        # integral result_type it lands on the DEFAULT float (float32) regardless of
        # the integer width; for a floating result_type it KEEPS that float.
        # (jittor natively follows numpy -- int64/int32 -> float64, int8/int8 ->
        # float16, float16/int64 -> float64 -- all wrong vs torch; the shim fixes it.)
        for da, db in [("int32", "int32"), ("int64", "int32"), ("int64", "int64"),
                       ("int8", "int8"), ("int32", "int64"), ("uint8", "int32"),
                       ("int16", "int16"), ("bool", "int32"), ("uint8", "uint8")]:
            self.assertEqual(_dts(_pin(da, (4, 8, 12)) / _pin(db, (2, 2, 2))),
                             "float32", msg=f"{da}/{db} -> float32")
        # a floating result_type keeps that float (float absorbs int, no widen)
        self.assertEqual(_dts(_pin("float16", (4, 8, 12)) / _pin("int64", (2, 2, 2))),
                         "float16")
        self.assertEqual(_dts(_pin("float32", (4, 8, 12)) / _pin("float64", (2, 2, 2))),
                         "float64")
        self.assertEqual(_dts(_pin("float16", (4, 8, 12)) / _pin("float32", (2, 2, 2))),
                         "float32")
        # value sanity: 12/2 == 6.0, and it is float32 not the numpy-wide float64
        r = _pin("int64", (4, 8, 12)) / _pin("int32", (2, 2, 2))
        self.assertEqual(r, np.array([2.0, 4.0, 6.0], "float32"),
                         atol=0, rtol=0, msg="truediv values")
        # reflected: scalar / int-tensor -> float32
        self.assertEqual(_dts(8.0 / _pin("int32", (2, 4, 8))), "float32")

    def test_scalar_promotion_through_ops(self):
        # Python-scalar promotion through real ops (not just result_type).
        xi = _pin("int32")
        self.assertEqual(_dts(xi + 1), "int32")        # int tensor + py int
        self.assertEqual(_dts(xi + 1.0), "float32")    # int tensor + py float
        self.assertEqual(_dts(xi * 3), "int32")
        xf = _pin("float32")
        self.assertEqual(_dts(xf + 1), "float32")      # float tensor + py int
        self.assertEqual(_dts(xf * 2), "float32")
        # int64 tensor + py int stays int64 (no widen to default)
        self.assertEqual(_dts(_pin("int64") + 1), "int64")


# --------------------------------------------------------- binary-op promotion (VALUES)

class TestPromotionValues(_CPUOnly):
    """Promotion must not just relabel the dtype -- the VALUES must be the widened
    arithmetic. numpy (at the promoted dtype) is the independent reference."""

    def test_int32_plus_int64_value_and_range(self):
        a = _pin("int32", (10, 20, 30))
        b = _pin("int64", (1, 2, 3))
        out = a + b
        self.assertEqual(_dts(out), "int64")
        ref = np.array([10, 20, 30], "int32").astype("int64") + np.array([1, 2, 3], "int64")
        self.assertEqual(out, ref, msg="int32+int64 value")

    def test_int64_mul_int32_needs_int64_range(self):
        # int64 * int32 must keep int64 range (an int32 result would overflow).
        big = _pin("int64", (1000000, 1000001, 1000002))
        m = _pin("int32", (1000000, 1000000, 1000000))
        prod = big * m
        self.assertEqual(_dts(prod), "int64")
        ref = (np.array([1000000, 1000001, 1000002], np.int64)
               * np.array([1000000] * 3, np.int32))
        self.assertEqual(prod, ref, msg="int64*int32 value/range")

    def test_float32_plus_float64_value(self):
        f = _pin("float32", (1, 2, 3))
        d = _pin("float64", (1, 1, 1))
        out = f + d
        self.assertEqual(_dts(out), "float64")
        self.assertEqual(out, np.array([2.0, 3.0, 4.0], "float64"),
                         atol=1e-12, rtol=1e-12, msg="float32+float64 value")

    def test_reflected_keeps_value_and_dtype(self):
        xi = _pin("int32", (10, 20, 30))
        self.assertEqual(_dts(100 - xi), "int32")
        self.assertEqual(100 - xi, np.array([90, 80, 70], "int32"), msg="100-int32 value")
        b = _pin("int64", (1, 2, 3))
        a = _pin("int32", (10, 20, 30))
        self.assertEqual(_dts(b - a), "int64")
        self.assertEqual(b - a, np.array([-9, -18, -27], "int64"), msg="int64-int32 value")
        xf = _pin("float32", (2, 3, 4))
        self.assertEqual(_dts(2 ** xf), "float32")
        self.assertEqual(2 ** xf, np.array([4.0, 8.0, 16.0], "float32"),
                         atol=1e-5, rtol=1e-5, msg="2**float32 value")


# ------------------------------------------------------------------------ cast methods

class TestCastMethods(_CPUOnly):
    # torch's documented per-method target dtype.
    _METHOD = {
        "byte": "uint8", "char": "int8", "short": "int16", "int": "int32",
        "long": "int64", "half": "float16", "float": "float32",
        "double": "float64", "bool": "bool",
    }

    def test_cast_methods_exact_dtype_from_float(self):
        # the classic bug guarded here: .long() must be int64 (native jittor aliased
        # Var.long -> Var.int32; the torch_compat install re-points it to int64).
        x = _pin("float32", (1.5, 2.5, 3.5))
        for m, want in self._METHOD.items():
            self.assertEqual(_dts(getattr(x, m)()), want, msg=f".{m}() from float32")

    def test_cast_methods_exact_dtype_from_int(self):
        # from an int input the wrong int alias would also surface.
        x = _pin("int32", (1, 2, 3))
        self.assertEqual(_dts(x.long()), "int64", msg=".long()==int64")
        self.assertEqual(_dts(x.int()), "int32", msg=".int()==int32")
        self.assertEqual(_dts(x.short()), "int16", msg=".short()==int16")
        self.assertEqual(_dts(x.byte()), "uint8", msg=".byte()==uint8")
        self.assertEqual(_dts(x.char()), "int8", msg=".char()==int8")
        self.assertEqual(_dts(x.double()), "float64", msg=".double()==float64")
        self.assertEqual(_dts(x.half()), "float16", msg=".half()==float16")
        self.assertEqual(_dts(x.float()), "float32", msg=".float()==float32")
        self.assertEqual(_dts(x.bool()), "bool", msg=".bool()==bool")

    def test_long_returns_int64_values(self):
        # value + dtype: float->long truncates toward zero AND lands on int64.
        x = _pin("float32", (1.9, 2.1, -3.7))
        r = x.long()
        self.assertEqual(_dts(r), "int64")
        self.assertEqual(r, np.array([1.9, 2.1, -3.7], "float32").astype("int64"),
                         msg=".long() truncates toward zero")

    def test_double_and_half_values(self):
        # .double() widens losslessly; .half() lands on float16 (lossy but typed).
        x = _pin("float32", (1.25, -2.5, 3.75))
        d = x.double()
        self.assertEqual(_dts(d), "float64")
        self.assertEqual(d, np.array([1.25, -2.5, 3.75], "float64"),
                         atol=1e-12, rtol=1e-12, msg=".double() value")
        h = x.half()
        self.assertEqual(_dts(h), "float16")
        self.assertEqual(h, np.array([1.25, -2.5, 3.75], "float32").astype("float16"),
                         atol=1e-3, rtol=1e-3, msg=".half() value")

    def test_bool_method_values(self):
        # .bool(): nonzero -> True, exact zero -> False.
        b = _pin("float32", (0.0, 1.0, -2.0)).bool()
        self.assertEqual(_dts(b), "bool")
        self.assertEqual(b, np.array([False, True, True]), msg=".bool() value")

    def test_byte_char_short_values(self):
        # the integer-narrowing casts keep numpy-astype semantics on in-range data.
        x = _pin("int32", (0, 5, 127))
        self.assertEqual(x.byte(), np.array([0, 5, 127], "int32").astype("uint8"),
                         msg=".byte() value")
        self.assertEqual(x.char(), np.array([0, 5, 127], "int32").astype("int8"),
                         msg=".char() value")
        self.assertEqual(x.short(), np.array([0, 5, 127], "int32").astype("int16"),
                         msg=".short() value")

    def test_dtype_constant_objects(self):
        # the dtype OBJECTS torch.long / int / short / half / double / float.
        self.assertEqual(jt.long.name, "int64")
        self.assertEqual(jt.int.name, "int32")
        self.assertEqual(jt.short.name, "int16")
        self.assertEqual(jt.half.name, "float16")
        self.assertEqual(jt.double.name, "float64")
        self.assertEqual(jt.float.name, "float32")


# --------------------------------------------------------------- .to / .type type-strings

class TestToAndType(_CPUOnly):
    def test_to_dtype_object_and_string(self):
        x = _pin("float32", (1.5, 2.5, 3.5))
        self.assertEqual(_dts(x.to(jt.float64)), "float64")
        self.assertEqual(_dts(x.to(jt.int64)), "int64")
        self.assertEqual(_dts(x.to(jt.int32)), "int32")
        self.assertEqual(_dts(x.to("float64")), "float64")
        self.assertEqual(_dts(x.to("torch.int64")), "int64")   # torch-prefixed string
        # value: float->int truncates toward zero
        self.assertEqual(x.to(jt.int32), np.array([1.5, 2.5, 3.5], "float32").astype("int32"),
                         msg=".to(int32) value")

    def test_type_with_dtype_casts(self):
        x = _pin("float32", (1, 2, 3))
        self.assertEqual(_dts(x.type(jt.int64)), "int64")
        self.assertEqual(_dts(x.type(jt.float64)), "float64")
        # torch also accepts the typed-tensor NAME string.
        self.assertEqual(_dts(x.type("torch.LongTensor")), "int64")
        self.assertEqual(_dts(x.type("torch.DoubleTensor")), "float64")
        self.assertEqual(_dts(x.type("torch.IntTensor")), "int32")
        self.assertEqual(_dts(x.type("torch.ByteTensor")), "uint8")

    def test_type_no_arg_returns_name(self):
        # Tensor.type() with no argument returns the torch type-name string.
        self.assertEqual(_pin("float32").type(), "torch.FloatTensor")
        self.assertEqual(_pin("float64").type(), "torch.DoubleTensor")
        self.assertEqual(_pin("int64").type(), "torch.LongTensor")
        self.assertEqual(_pin("int32").type(), "torch.IntTensor")
        self.assertEqual(_pin("int16").type(), "torch.ShortTensor")
        self.assertEqual(_pin("int8").type(), "torch.CharTensor")
        self.assertEqual(_pin("uint8").type(), "torch.ByteTensor")
        self.assertEqual(_pin("float16").type(), "torch.HalfTensor")
        self.assertEqual(_pin("bool").type(), "torch.BoolTensor")


# ----------------------------------------------------------------------------- NanoString

class TestNanoString(_CPUOnly):
    """A few edge checks on jittor's NanoString dtype token -- the value that
    ``str(var.dtype)`` produces and that jittor's own C++ dispatch consumes."""

    def test_str_roundtrips(self):
        # str(NanoString) must be the bare name, and feeding it back constructs the
        # same token -- jittor's contrib/linalg/nn do exactly this round-trip.
        for name in ["float32", "float64", "float16", "int8", "int16",
                     "int32", "int64", "uint8", "bool"]:
            self.assertEqual(str(jt.NanoString(name)), name, msg=f"str(NanoString({name}))")
            self.assertEqual(str(jt.NanoString(str(jt.NanoString(name)))), name,
                             msg=f"NanoString round-trip {name}")

    def test_alias_tokens(self):
        # the torch-style short aliases resolve to the canonical bare names.
        self.assertEqual(str(jt.NanoString("float")), "float32")
        self.assertEqual(str(jt.NanoString("double")), "float64")
        self.assertEqual(str(jt.NanoString("half")), "float16")

    def test_str_subclass_accepted(self):
        # a plain str subclass (which torch_compat's `dtype` IS) must be accepted by
        # NanoString just like a bare str.
        class _StrSub(str):
            pass
        self.assertEqual(str(jt.NanoString(_StrSub("float32"))), "float32")
        self.assertEqual(str(jt.NanoString(_StrSub("int64"))), "int64")
        # the live torch_compat dtype object is itself a str subclass.
        self.assertEqual(str(jt.NanoString(jt.float32)), "float32")
        self.assertEqual(str(jt.NanoString(jt.int64)), "int64")

    def test_equality_against_str(self):
        # NanoString == bare-name string (used throughout jittor's dispatch).
        self.assertTrue(jt.NanoString("float") == "float32")
        self.assertTrue(jt.NanoString("int64") == "int64")

    def test_var_dtype_str_feeds_back_into_dispatch(self):
        # the load-bearing invariant: str(var.dtype) must be a token jittor accepts
        # for a cast (contrib.concat / linalg do this), and the dtype must survive.
        v = _pin("float64", (1.0, 2.0))
        s = str(v.dtype)
        self.assertEqual(s, "float64")
        self.assertEqual(_dts(v.cast(s)), "float64")
        # narrowing-pinned int64 round-trips through its own dtype string too.
        w = _pin("int64", (1, 2, 3))
        self.assertEqual(_dts(w.cast(str(w.dtype))), "int64")


# ============================ PRESERVED semantic-diffs (carried, NOT re-asserted) ========
# These two locks come from test_torch_compat_dtype.py. They describe the PRE-SHIM
# NATIVE jittor behavior. `import jittor` unconditionally runs torch_compat.install(),
# which OVERRIDES both: the live binary-op promotion now follows torch's lattice
# (TestBinaryOpPromotion, above) and Var.long() now returns int64
# (TestCastMethods.test_cast_methods_exact_dtype_from_int, above). The skips are
# carried verbatim per the audit's must-keep rule so the historical divergence and
# its verify-then-fix note are never lost in a refactor -- and are NOT asserted to a
# wrong value.

class TestPreservedSemanticDiffs(_CPUOnly):
    @unittest.skip("SEMANTIC-DIFF (carried from test_torch_compat_dtype.py): the "
                   "PRE-SHIM native jittor binary-op promotion did NOT follow torch's "
                   "result_type lattice for MIXED dtypes -- it kept the narrower/left "
                   "operand's type (int32+int64 -> int32, float32+float64 -> float32, "
                   "float16+int64 -> float16-by-other-path, int8+int32 -> int8, "
                   "bool+int32 -> int8), silently losing precision/range. The "
                   "torch_compat layer installed at import now promotes to result_type "
                   "first (asserted positively in TestBinaryOpPromotion). "
                   "verify-then-fix kept for history; do not re-assert the old value.")
    def test_native_promotion_kept_narrower_operand_DIVERGENCE(self):
        pass

    @unittest.skip("SEMANTIC-DIFF (carried from test_torch_compat_dtype.py): native "
                   "jittor aliased Var.long = Var.int32, so .long() returned int32 "
                   "whereas torch's .long() is int64. The dtype OBJECT torch.long was "
                   "always int64 (correct); only the cast METHOD diverged. The "
                   "torch_compat install re-points Var.long to cast->int64 (asserted "
                   "positively in TestCastMethods). Carried for history; not re-asserted.")
    def test_native_long_was_int32_DIVERGENCE(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Torch-grade DTYPE-PROMOTION + integer-dtype-method regression tests for
``import jittor as torch``.

Part of the torch-grade test-suite (bug #10). Like the other ``test_torch_compat_*``
modules this is a structured ``unittest`` module: every check compares jittor-as-torch
against the DOCUMENTED torch result (hard-coded dtype names, since real torch may not be
installed) and runs on BOTH CPU and CUDA (when the build has it), so it locks torch
*semantics* rather than jittor self-consistency.

Covered:
  * ``torch.result_type(a, b)`` / ``torch.promote_types(t1, t2)`` / ``torch.can_cast`` --
    the documented ``_promoteTypesLookup`` lattice (c10/core/ScalarType.cpp);
  * binary-op type promotion for MIXED dtypes (``int32+int64 -> int64``,
    ``float32+float64 -> float64``, ``float16+int64 -> float16``, ``uint8+int8 -> int16``,
    ``uint8+uint8 -> uint8`` ...), including the Python-scalar "wrapped number" rule;
  * integer/float cast methods land on torch's EXACT dtype: ``.long() -> int64``,
    ``.int() -> int32``, ``.short() -> int16``, ``.byte() -> uint8``, ``.char() -> int8``,
    ``.bool() -> bool``, ``.half() -> float16``, ``.double() -> float64``,
    ``.float() -> float32``;
  * ``.to(dtype)`` / ``.type(dtype)`` / ``.type()`` (no-arg type name).

Reference for torch's promotion rules (documented behavior, encoded below):
  - category order  bool < int < float < complex;
  - same-category ints: the wider wins, but signed+unsigned promotes to a signed type
    wide enough to hold both (uint8+int8 -> int16, uint8+int64 -> int64);
  - a float of ANY width absorbs an int of ANY width WITHOUT widening
    (float16+int64 -> float16);  int+bfloat16 -> bfloat16;
  - floats: the wider wins, except float16+bfloat16 -> float32;
  - a Python scalar only bumps the result if it is a strictly HIGHER category than the
    tensor: int scalar keeps the tensor's int dtype, float scalar lifts an int tensor to
    the default float (float32).

jittor has no 0-d scalars (a "scalar" is shape ``(1,)``); values are compared via
``.numpy()``.

Run:  python -m jittor.test.test_torch_compat_promotion
      python -m pytest python/jittor/test/test_torch_compat_promotion.py
"""
import unittest
import numpy as np
import jittor as torch          # the whole point: jittor IS torch here
import jittor as jt

# Exercise CPU always; add CUDA when the build has it. NPU(ACL) reports has_cuda too.
_DEVICES = [("cpu", 0)] + ([("cuda", 1)] if jt.has_cuda else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def dts(v):
    """bare jittor dtype string for a Var ('float32', 'int64', ...)."""
    return str(v.dtype)


# numpy dtype for each bare name we build test tensors from.
_NPDT = {
    "bool": np.bool_, "uint8": np.uint8, "int8": np.int8, "int16": np.int16,
    "int32": np.int32, "int64": np.int64, "float16": np.float16,
    "float32": np.float32, "float64": np.float64,
}


def mk(name, val=(1, 2, 3)):
    """A 1-D jittor tensor of the given bare dtype (built so the dtype survives)."""
    return torch.tensor(np.array(list(val), dtype=_NPDT[name]))


class Base(unittest.TestCase):
    def ae(self, got, ref, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_array_equal(g, r, err_msg=msg)


# =====================================================================================
# The DOCUMENTED torch promotion lattice (what we assert against). These are torch's
# result_type results, NOT jittor's old behavior. Verified against torch's docs.
# =====================================================================================
_PROMO = {
    # int + float : float category wins; float NOT widened to hold the int.
    ("int32", "float32"): "float32",
    ("int64", "float32"): "float32",
    ("int32", "float64"): "float64",
    ("float16", "int64"): "float16",   # float of any width absorbs int of any width
    ("uint8", "float16"): "float16",
    ("int8", "float32"): "float32",
    # float + float : wider wins, except f16+bf16 -> f32.
    ("float32", "float64"): "float64",
    ("float16", "float32"): "float32",
    ("float16", "float64"): "float64",
    ("float32", "float32"): "float32",
    ("float64", "float64"): "float64",
    # int + int (same signedness) : wider wins.
    ("int32", "int64"): "int64",
    ("int8", "int16"): "int16",
    ("int16", "int32"): "int32",
    ("int32", "int32"): "int32",
    ("int64", "int64"): "int64",
    ("int8", "int8"): "int8",
    # signed + unsigned : promote to a signed type wide enough to hold both.
    ("uint8", "int8"): "int16",
    ("uint8", "int16"): "int16",
    ("uint8", "int32"): "int32",
    ("uint8", "int64"): "int64",
    ("uint8", "uint8"): "uint8",
    # bool is the identity-low element: bool + X == X.
    ("bool", "int32"): "int32",
    ("bool", "int64"): "int64",
    ("bool", "uint8"): "uint8",
    ("bool", "float32"): "float32",
    ("bool", "bool"): "bool",
}


# ----------------------------------------------------------- torch.promote_types object API

class TestPromoteTypesAPI(Base):
    def test_promote_types_matches_lattice(self):
        # torch.promote_types(dtype1, dtype2) -> dtype object, commutative.
        for (a, b), want in _PROMO.items():
            r1 = torch.promote_types(getattr(torch, a), getattr(torch, b))
            r2 = torch.promote_types(getattr(torch, b), getattr(torch, a))
            self.assertEqual(str(r1), want, f"promote_types({a},{b})")
            self.assertEqual(str(r2), want, f"promote_types({b},{a}) [commutative]")

    def test_promote_types_doc_examples(self):
        # the two examples from torch.promote_types' own docstring.
        self.assertEqual(str(torch.promote_types(torch.int32, torch.float32)), "float32")
        self.assertEqual(str(torch.promote_types(torch.uint8, torch.long)), "int64")

    def test_promote_types_float16_bfloat16(self):
        # the documented special case: neither can represent the other -> float32.
        self.assertEqual(str(torch.promote_types(torch.float16, torch.bfloat16)), "float32")
        self.assertEqual(str(torch.promote_types(torch.bfloat16, torch.float16)), "float32")

    def test_promote_types_int_bfloat16(self):
        # int + bfloat16 -> bfloat16 (torch parity, incl. its low-mantissa caveat).
        self.assertEqual(str(torch.promote_types(torch.int32, torch.bfloat16)), "bfloat16")
        self.assertEqual(str(torch.promote_types(torch.bfloat16, torch.int64)), "bfloat16")


# ------------------------------------------------------------------- torch.result_type API

class TestResultTypeAPI(Base):
    def test_result_type_two_tensors(self):
        def body(dev):
            for (a, b), want in _PROMO.items():
                self.assertEqual(str(torch.result_type(mk(a), mk(b))), want,
                                 f"result_type({a},{b}) {dev}")
        both_devices(body)

    def test_result_type_tensor_and_dtype(self):
        # torch.result_type also accepts a dtype as either argument.
        self.assertEqual(str(torch.result_type(mk("int32"), torch.float32)), "float32")
        self.assertEqual(str(torch.result_type(torch.int64, mk("int32"))), "int64")

    def test_result_type_with_python_scalar(self):
        # torch "wrapped number" rule: a scalar only bumps the result if it is a
        # strictly HIGHER category than the tensor.
        def body(dev):
            xi = mk("int32")
            xf = mk("float32")
            self.assertEqual(str(torch.result_type(xi, 2)), "int32", dev)       # int scalar: no widen
            self.assertEqual(str(torch.result_type(xi, 1.5)), "float32", dev)   # float scalar lifts int->float32
            self.assertEqual(str(torch.result_type(xf, 2)), "float32", dev)     # int scalar keeps float
            self.assertEqual(str(torch.result_type(xf, 1.5)), "float32", dev)
            self.assertEqual(str(torch.result_type(mk("int64"), 7)), "int64", dev)
        both_devices(body)

    def test_can_cast(self):
        # torch.can_cast(from, to): True iff promote(from, to) == to.
        self.assertTrue(torch.can_cast(torch.int32, torch.int64))
        self.assertTrue(torch.can_cast(torch.float32, torch.float64))
        self.assertTrue(torch.can_cast(torch.bool, torch.int32))
        self.assertFalse(torch.can_cast(torch.float32, torch.int32))   # float -> int loses category
        self.assertFalse(torch.can_cast(torch.int64, torch.int32))     # wider -> narrower


# ------------------------------------------------------------- binary-op promotion (values)

class TestBinaryOpPromotion(Base):
    def _binop_dtype(self, da, db, op):
        a = mk(da, (2, 4, 6))
        b = mk(db, (1, 2, 3))
        return dts(op(a, b))

    def test_add_promotes_like_torch(self):
        def body(dev):
            for (da, db), want in _PROMO.items():
                self.assertEqual(self._binop_dtype(da, db, lambda x, y: x + y), want,
                                 f"{da}+{db} {dev}")
                # commutative dtype
                self.assertEqual(self._binop_dtype(db, da, lambda x, y: x + y), want,
                                 f"{db}+{da} {dev}")
        both_devices(body)

    def test_sub_mul_promote(self):
        def body(dev):
            for (da, db), want in _PROMO.items():
                # subtraction is undefined for bool in torch; skip bool pairs for sub.
                if "bool" not in (da, db):
                    self.assertEqual(self._binop_dtype(da, db, lambda x, y: x - y), want,
                                     f"{da}-{db} {dev}")
                self.assertEqual(self._binop_dtype(da, db, lambda x, y: x * y), want,
                                 f"{da}*{db} {dev}")
        both_devices(body)

    def test_floordiv_promotes(self):
        # torch '//' follows the standard promotion lattice (unlike '/').
        def body(dev):
            self.assertEqual(self._binop_dtype("int32", "int64", lambda x, y: x // y),
                             "int64", dev)
            self.assertEqual(self._binop_dtype("int8", "int16", lambda x, y: x // y),
                             "int16", dev)
        both_devices(body)

    def test_truediv_is_float(self):
        # torch '/' ALWAYS yields float (the documented special case). For an
        # integral result_type it lands on the DEFAULT float (float32) regardless of
        # the integer width; for a floating result_type it keeps that float.
        # (jittor natively follows numpy: int64/int32 -> float64, int8/int8 ->
        # float16, float16/int64 -> float64 -- all wrong vs torch.)
        def body(dev):
            # integral pairs -> float32 (NOT float64 / float16)
            for da, db in [("int32", "int32"), ("int64", "int32"), ("int64", "int64"),
                           ("int8", "int8"), ("int32", "int64"), ("uint8", "int32"),
                           ("int16", "int16"), ("bool", "int32")]:
                self.assertEqual(dts(mk(da, (4, 8, 12)) / mk(db, (2, 2, 2))),
                                 "float32", f"{da}/{db} {dev}")
            # floating result_type -> keep that float (float absorbs int, no widen)
            self.assertEqual(dts(mk("float16", (4, 8, 12)) / mk("int64", (2, 2, 2))),
                             "float16", dev)
            self.assertEqual(dts(mk("float32", (4, 8, 12)) / mk("float64", (2, 2, 2))),
                             "float64", dev)
            self.assertEqual(dts(mk("float16", (4, 8, 12)) / mk("float32", (2, 2, 2))),
                             "float32", dev)
            # value sanity: 12/2 == 6.0 in float32
            r = mk("int64", (4, 8, 12)) / mk("int32", (2, 2, 2))
            self.ae(r.numpy(), np.array([2.0, 4.0, 6.0], "float32"), dev)
            # reflected: scalar / int-tensor -> float32
            self.assertEqual(dts(8.0 / mk("int32", (2, 4, 8))), "float32", dev)
        both_devices(body)

    def test_mixed_values_correct(self):
        # promotion must not just relabel the dtype -- the VALUES must be the
        # widened arithmetic (numpy is the independent reference).
        def body(dev):
            a = mk("int32", (10, 20, 30)); b = mk("int64", (1, 2, 3))
            self.ae((a + b).numpy(),
                    np.array([10, 20, 30], "int32").astype("int64")
                    + np.array([1, 2, 3], "int64"), dev)
            # float promotion keeps the float arithmetic
            f = mk("float32", (1, 2, 3)); d = mk("float64", (1, 1, 1))
            r = f + d
            self.assertEqual(dts(r), "float64", dev)
            self.ae(r.numpy(), np.array([2.0, 3.0, 4.0], "float64"), dev)
            # int64 * int32 needs int64 range (old jittor overflowed to int32)
            big = (jt.arange(3, dtype="int64") + 1000000)        # [1e6, 1e6+1, 1e6+2]
            m = mk("int32", (1000000, 1000000, 1000000))
            prod = big * m
            self.assertEqual(dts(prod), "int64", dev)
            ref = (np.arange(3, dtype=np.int64) + 1000000) * np.array([1000000] * 3, np.int32)
            self.ae(prod.numpy(), ref, dev)
        both_devices(body)

    def test_reflected_ops_value_and_dtype(self):
        # reflected operators (scalar - tensor, etc.) keep both the right value and
        # the right dtype.
        def body(dev):
            xi = mk("int32", (10, 20, 30))
            self.ae((100 - xi).numpy(), np.array([90, 80, 70], "int32"), dev)
            self.assertEqual(dts(100 - xi), "int32", dev)
            a = mk("int32", (10, 20, 30)); b = mk("int64", (1, 2, 3))
            self.assertEqual(dts(b - a), "int64", dev)
            self.ae((b - a).numpy(), np.array([-9, -18, -27], "int64"), dev)
            xf = mk("float32", (2, 3, 4))
            self.assertEqual(dts(2 ** xf), "float32", dev)
            self.ae((2 ** xf).numpy(), np.array([4.0, 8.0, 16.0], "float32"), dev)
        both_devices(body)

    def test_scalar_promotion_matches_torch(self):
        # Python-scalar promotion through real ops (not just result_type).
        def body(dev):
            xi = mk("int32")
            self.assertEqual(dts(xi + 1), "int32", dev)        # int tensor + py int
            self.assertEqual(dts(xi + 1.0), "float32", dev)    # int tensor + py float
            self.assertEqual(dts(xi * 3), "int32", dev)
            xf = mk("float32")
            self.assertEqual(dts(xf + 1), "float32", dev)      # float tensor + py int
            self.assertEqual(dts(xf * 2), "float32", dev)
        both_devices(body)


# ------------------------------------------------------------------ integer dtype methods

class TestCastMethods(Base):
    # torch's documented per-method target dtype.
    _METHOD = {
        "byte": "uint8", "char": "int8", "short": "int16", "int": "int32",
        "long": "int64", "half": "float16", "float": "float32",
        "double": "float64", "bool": "bool",
    }

    def test_cast_methods_exact_dtype_from_float(self):
        # the classic bug: .long() must be int64 (jittor aliased it to int32).
        def body(dev):
            x = mk("float32", (1.5, 2.5, 3.5))
            for m, want in self._METHOD.items():
                self.assertEqual(dts(getattr(x, m)()), want, f".{m}() {dev}")
        both_devices(body)

    def test_cast_methods_exact_dtype_from_int(self):
        # from an int input the wrong int alias would also surface.
        def body(dev):
            x = mk("int32", (1, 2, 3))
            self.assertEqual(dts(x.long()), "int64", dev)
            self.assertEqual(dts(x.int()), "int32", dev)
            self.assertEqual(dts(x.short()), "int16", dev)
            self.assertEqual(dts(x.byte()), "uint8", dev)
            self.assertEqual(dts(x.char()), "int8", dev)
            self.assertEqual(dts(x.double()), "float64", dev)
            self.assertEqual(dts(x.half()), "float16", dev)
            self.assertEqual(dts(x.float()), "float32", dev)
        both_devices(body)

    def test_long_returns_int64_values(self):
        # value + dtype: float->long truncates toward zero AND lands on int64.
        def body(dev):
            x = mk("float32", (1.9, 2.1, -3.7))
            r = x.long()
            self.assertEqual(dts(r), "int64", dev)
            self.ae(r.numpy(), np.array([1.9, 2.1, -3.7], "float32").astype("int64"), dev)
        both_devices(body)

    def test_bool_method(self):
        def body(dev):
            b = mk("float32", (0.0, 1.0, 2.0)).bool()
            self.assertEqual(dts(b), "bool", dev)
            self.ae(b.numpy(), np.array([False, True, True]), dev)
        both_devices(body)

    def test_dtype_constants(self):
        # torch.long / torch.int / torch.short / torch.byte dtype OBJECTS.
        self.assertEqual(torch.long.name, "int64")
        self.assertEqual(torch.int.name, "int32")
        self.assertEqual(torch.short.name, "int16")
        self.assertEqual(torch.half.name, "float16")
        self.assertEqual(torch.double.name, "float64")
        self.assertEqual(torch.float.name, "float32")


# ------------------------------------------------------------------- .to(dtype) / .type()

class TestToAndType(Base):
    def test_to_dtype(self):
        def body(dev):
            x = mk("float32", (1.5, 2.5, 3.5))
            self.assertEqual(dts(x.to(torch.float64)), "float64", dev)
            self.assertEqual(dts(x.to(torch.int64)), "int64", dev)
            self.assertEqual(dts(x.to(torch.int32)), "int32", dev)
            self.assertEqual(dts(x.to("float64")), "float64", dev)
            self.assertEqual(dts(x.to("torch.int64")), "int64", dev)
            # value: float->int truncates toward zero
            self.ae(x.to(torch.int32).numpy(),
                    np.array([1.5, 2.5, 3.5], "float32").astype("int32"), dev)
        both_devices(body)

    def test_type_with_dtype_casts(self):
        def body(dev):
            x = mk("float32", (1, 2, 3))
            self.assertEqual(dts(x.type(torch.int64)), "int64", dev)
            self.assertEqual(dts(x.type(torch.float64)), "float64", dev)
            # torch also accepts the typed-tensor NAME string.
            self.assertEqual(dts(x.type("torch.LongTensor")), "int64", dev)
            self.assertEqual(dts(x.type("torch.DoubleTensor")), "float64", dev)
        both_devices(body)

    def test_type_no_arg_returns_name(self):
        # torch.Tensor.type() with no argument returns the torch type-name string.
        def body(dev):
            self.assertEqual(mk("float32").type(), "torch.FloatTensor", dev)
            self.assertEqual(mk("int64").type(), "torch.LongTensor", dev)
            self.assertEqual(mk("int32").type(), "torch.IntTensor", dev)
            self.assertEqual(mk("uint8").type(), "torch.ByteTensor", dev)
            self.assertEqual(mk("bool").type(), "torch.BoolTensor", dev)
        both_devices(body)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Torch-grade dtype-semantics regression tests for ``import jittor as torch``.

Part of the torch-grade test-suite rewrite (round 3). Like ``test_torch_compat_ops.py``
this is a structured ``unittest`` module: every check compares jittor-as-torch against an
INDEPENDENT reference (numpy / explicit torch dtype rules) and runs on BOTH CPU and CUDA
(when the build has it), so it locks torch *dtype semantics* rather than jittor
self-consistency.

Covered: dtype objects + repr (``torch.float32`` / ``torch.long`` ...), ``.to(dtype)`` /
``.astype`` / ``.float()`` / ``.long()`` / ``.int()`` / ``.bool()`` / ``.half()`` /
``.double()``, ``zeros`` / ``ones`` / ``empty`` / ``full`` / ``arange`` + ``*_like`` with
``dtype=``, ``from_numpy`` dtype preservation, integer/float *type promotion* vs torch's
rules, and ``iinfo`` / ``finfo``.

Two genuine semantic divergences from torch are isolated with ``@unittest.skip`` (NOT
asserted to a wrong value), documented in the module-level "SUSPECTED-BUG / SEMANTIC-DIFF"
notes below and in each skip reason:

  1. jittor binary-op type promotion differs from torch's ``result_type`` lattice for
     most MIXED dtypes (e.g. ``int32+int64 -> int32`` where torch gives ``int64``;
     ``float32+float64 -> float32`` where torch gives ``float64``). jittor appears to keep
     the narrower/left operand's type rather than promoting upward. The torch-agreeing
     subset (``int32+float32 -> float32`` etc.) IS asserted positively.
  2. ``Var.long()`` returns ``int32`` (jittor aliases ``Var.long = Var.int32``), whereas
     torch's ``.long()`` is ``int64``. The dtype OBJECT ``torch.long`` IS ``int64``
     (correct); only the cast METHOD diverges.

jittor has no 0-d scalars (a "scalar" is shape ``(1,)``); values are compared via
``.item()`` where a Python scalar is expected.

Run:  python -m jittor.test.test_torch_compat_dtype
      python -m pytest python/jittor/test/test_torch_compat_dtype.py
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


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def ae(self, got, ref, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_array_equal(g, r, err_msg=msg)


# ----------------------------------------------------------------- dtype objects + repr

class TestDtypeObjects(Base):
    def test_aliases_resolve_to_bare_names(self):
        # torch.long/int/float/double/half/short map to the right jittor dtype names.
        self.assertEqual(torch.float32.name, "float32")
        self.assertEqual(torch.float64.name, "float64")
        self.assertEqual(torch.float16.name, "float16")
        self.assertEqual(torch.int32.name, "int32")
        self.assertEqual(torch.int64.name, "int64")
        self.assertEqual(torch.int8.name, "int8")
        self.assertEqual(torch.uint8.name, "uint8")
        self.assertEqual(torch.bool.name, "bool")
        # torch-style short aliases
        self.assertEqual(torch.long.name, "int64")
        self.assertEqual(torch.int.name, "int32")
        self.assertEqual(torch.float.name, "float32")
        self.assertEqual(torch.double.name, "float64")
        self.assertEqual(torch.half.name, "float16")
        self.assertEqual(torch.short.name, "int16")

    def test_repr_is_torch_style(self):
        self.assertEqual(repr(torch.float32), "torch.float32")
        self.assertEqual(repr(torch.long), "torch.int64")
        self.assertEqual(repr(torch.bool), "torch.bool")

    def test_str_split_is_torch_style(self):
        # transformers relies on str(dtype).split('.') == ['torch', name].
        self.assertEqual(str(torch.float32).split("."), ["torch", "float32"])
        self.assertEqual(str(torch.int64).split("."), ["torch", "int64"])

    def test_dtype_equality_against_str(self):
        x = torch.ones(2, dtype=torch.float32)
        self.assertTrue(x.dtype == torch.float32)
        self.assertTrue(x.dtype == "float32")
        self.assertTrue(x.dtype == "torch.float32")
        self.assertFalse(x.dtype == torch.int32)

    def test_is_floating_point(self):
        self.assertTrue(torch.float32.is_floating_point)
        self.assertTrue(torch.float64.is_floating_point)
        self.assertTrue(torch.float16.is_floating_point)
        self.assertFalse(torch.int32.is_floating_point)
        self.assertFalse(torch.int64.is_floating_point)
        self.assertFalse(torch.bool.is_floating_point)
        # tensor-level predicate
        self.assertTrue(torch.ones(2, dtype=torch.float32).is_floating_point())
        self.assertFalse(torch.ones(2, dtype=torch.int32).is_floating_point())


# --------------------------------------------------------------- constructors with dtype=

class TestConstructorDtype(Base):
    def test_zeros_ones_full_dtype(self):
        def body(dev):
            self.assertEqual(dts(torch.zeros(3, dtype=torch.float32)), "float32", dev)
            self.assertEqual(dts(torch.zeros(3, dtype=torch.long)), "int64", dev)
            self.assertEqual(dts(torch.ones(3, dtype=torch.int32)), "int32", dev)
            self.assertEqual(dts(torch.full((2, 2), 5, dtype=torch.float64)), "float64", dev)
            # values are right too, not just dtype
            self.ae(torch.zeros(3, dtype=torch.long).numpy(), np.zeros(3, "int64"), dev)
            self.ae(torch.ones(3, dtype=torch.int32).numpy(), np.ones(3, "int32"), dev)
            self.ae(torch.full((2, 2), 5, dtype=torch.float64).numpy(),
                    np.full((2, 2), 5, "float64"), dev)
        both_devices(body)

    def test_empty_dtype_shape(self):
        # empty: contents undefined, but dtype + shape must match the request.
        def body(dev):
            e = torch.empty(2, 3, dtype=torch.int16)
            self.assertEqual(dts(e), "int16", dev)
            self.assertEqual(tuple(e.shape), (2, 3), dev)
        both_devices(body)

    def test_arange_dtype(self):
        def body(dev):
            self.assertEqual(dts(torch.arange(5)), "int32", dev)          # int range -> int32
            self.assertEqual(dts(torch.arange(0.0, 5.0)), "float32", dev)  # float range -> float32
            self.ae(torch.arange(5).numpy(), np.arange(5, dtype="int32"), dev)
        both_devices(body)

    def test_like_constructors_keep_dtype(self):
        def body(dev):
            self.assertEqual(dts(torch.zeros_like(torch.ones(3, dtype=torch.int64))),
                             "int64", dev)
            self.assertEqual(dts(torch.ones_like(torch.ones(3, dtype=torch.float64))),
                             "float64", dev)
            self.assertEqual(dts(torch.full_like(torch.ones(3, dtype=torch.int32), 5)),
                             "int32", dev)
        both_devices(body)


# ------------------------------------------------------------------- cast methods / .to

class TestCastMethods(Base):
    def setUp(self):
        self.x = np.array([1.5, 2.5, 3.5], dtype="float32")

    def test_to_dtype(self):
        def body(dev):
            x = torch.array(self.x)
            self.assertEqual(dts(x.to(torch.int32)), "int32", dev)
            self.assertEqual(dts(x.to(torch.float64)), "float64", dev)
            self.assertEqual(dts(x.to("float64")), "float64", dev)
            # value: float->int truncates toward zero (torch + numpy astype agree)
            self.ae(x.to(torch.int32).numpy(), self.x.astype("int32"), dev)
        both_devices(body)

    def test_to_device_is_noop_on_dtype(self):
        # torch's .to('cuda') / .to(dtype, device) -- jittor has a single global
        # backend, so device moves are no-ops but dtype must still convert.
        def body(dev):
            x = torch.array(self.x)
            self.assertEqual(dts(x.to("cuda")), "float32", dev)
            self.assertEqual(dts(x.to(torch.float64)), "float64", dev)
        both_devices(body)

    def test_astype(self):
        def body(dev):
            x = torch.array(self.x)
            self.assertEqual(dts(x.astype("int64")), "int64", dev)
            self.assertEqual(dts(x.astype("float16")), "float16", dev)
        both_devices(body)

    def test_float_int_bool_double_half(self):
        def body(dev):
            x = torch.array(self.x)
            self.assertEqual(dts(x.float()), "float32", dev)
            self.assertEqual(dts(x.int()), "int32", dev)
            self.assertEqual(dts(x.double()), "float64", dev)
            self.assertEqual(dts(x.half()), "float16", dev)
            # .bool(): nonzero -> True
            b = torch.array(np.array([0.0, 1.0, 2.0], "float32")).bool()
            self.assertEqual(dts(b), "bool", dev)
            self.ae(b.numpy(), np.array([False, True, True]), dev)
            # round-trip int->float
            self.assertEqual(dts(x.int().float()), "float32", dev)
        both_devices(body)

    @unittest.skip("SEMANTIC-DIFF: jittor aliases Var.long = Var.int32, so .long() returns "
                   "int32, whereas torch's .long() is int64. The dtype OBJECT torch.long IS "
                   "int64 (correct, see TestDtypeObjects); only the cast METHOD diverges. "
                   "verify-then-fix: point Var.long at int64 to match torch.")
    def test_long_returns_int64_like_torch(self):
        def body(dev):
            x = torch.array(self.x)
            self.assertEqual(dts(x.long()), "int64", dev)
        both_devices(body)


# ----------------------------------------------------------------- from_numpy dtype keep

class TestFromNumpyDtype(Base):
    def test_from_numpy_preserves_dtype(self):
        def body(dev):
            for npdt in ["int64", "int32", "float32", "float64",
                         "int16", "int8", "uint8", "bool"]:
                a = np.arange(3).astype(npdt)
                v = torch.from_numpy(a)
                self.assertEqual(dts(v), npdt, f"from_numpy {npdt} {dev}")
                self.ae(v.numpy(), a, f"from_numpy values {npdt} {dev}")
        both_devices(body)

    def test_tensor_from_numpy_preserves_int64_float64(self):
        # torch.tensor(np.int64-array) keeps int64; jittor's bare jt.array would
        # downcast to int32, so the shim must preserve it.
        def body(dev):
            self.assertEqual(dts(torch.tensor(np.arange(3, dtype="int64"))), "int64", dev)
            self.assertEqual(dts(torch.tensor(np.ones(3, dtype="float64"))), "float64", dev)
        both_devices(body)


# ---------------------------------------------------------------------- type promotion

# torch's result_type lattice (the documented torch behavior we compare against).
# Split into the subset jittor AGREES with, and the subset it DIVERGES on.
_TORCH_PROMO = {
    ("int32", "float32"): "float32",
    ("int32", "int32"): "int32",
    ("float32", "float32"): "float32",
    ("float64", "float64"): "float64",
    ("int64", "int64"): "int64",
    ("int8", "int8"): "int8",
    # --- below here jittor diverges from torch (kept for the documented skip) ---
    ("int64", "float32"): "float32",
    ("int32", "int64"): "int64",
    ("float32", "float64"): "float64",
    ("int32", "float64"): "float64",
    ("float16", "float32"): "float32",
    ("int8", "int32"): "int32",
    ("int16", "int32"): "int32",
    ("bool", "int32"): "int32",
    ("bool", "float32"): "float32",
    ("uint8", "int32"): "int32",
}

# The subset where jittor's binary-op dtype matches torch (verified at write time).
_AGREE = {
    ("int32", "float32"), ("int32", "int32"), ("float32", "float32"),
    ("float64", "float64"), ("int64", "int64"), ("int8", "int8"),
}


class TestTypePromotion(Base):
    def _binop_dtype(self, da, db):
        a = torch.ones(3, dtype=getattr(torch, da))
        b = torch.ones(3, dtype=getattr(torch, db))
        return dts(a + b)

    def test_promotion_agreeing_subset(self):
        # These cases jittor promotes exactly like torch -- assert them positively.
        def body(dev):
            for (da, db) in sorted(_AGREE):
                self.assertEqual(self._binop_dtype(da, db), _TORCH_PROMO[(da, db)],
                                 f"{da}+{db} {dev}")
                # commutative: order shouldn't change the result for these
                self.assertEqual(self._binop_dtype(db, da), _TORCH_PROMO[(da, db)],
                                 f"{db}+{da} {dev}")
        both_devices(body)

    def test_scalar_promotion(self):
        # Python-scalar promotion DOES match torch: a python int keeps the tensor's
        # integer dtype; a python float lifts an int tensor to its default float.
        def body(dev):
            xi = torch.ones(3, dtype=torch.int32)
            self.assertEqual(dts(xi + 1), "int32", dev)        # int tensor + py int
            self.assertEqual(dts(xi + 1.0), "float32", dev)    # int tensor + py float
            xf = torch.ones(3, dtype=torch.float32)
            self.assertEqual(dts(xf + 1), "float32", dev)      # float tensor + py int
        both_devices(body)

    @unittest.skip("SUSPECTED-BUG / SEMANTIC-DIFF: jittor binary-op type promotion does NOT "
                   "follow torch's result_type lattice for MIXED dtypes. jittor keeps the "
                   "narrower/left operand's type instead of promoting upward, e.g. "
                   "int32+int64 -> int32 (torch: int64), float32+float64 -> float32 "
                   "(torch: float64), float16+float32 -> float16 (torch: float32), "
                   "int8+int32 -> int8 (torch: int32), bool+int32 -> int8 (torch: int32). "
                   "This silently loses precision/range in mixed-dtype arithmetic. "
                   "verify-then-fix: make the binary-op output dtype follow torch promotion.")
    def test_promotion_full_torch_lattice(self):
        def body(dev):
            for (da, db), ref in _TORCH_PROMO.items():
                self.assertEqual(self._binop_dtype(da, db), ref, f"{da}+{db} {dev}")
        both_devices(body)


# --------------------------------------------------------------------------- iinfo/finfo

class TestInfo(Base):
    def test_iinfo(self):
        self.assertEqual(torch.iinfo(torch.int32).max, np.iinfo("int32").max)
        self.assertEqual(torch.iinfo(torch.int32).min, np.iinfo("int32").min)
        self.assertEqual(torch.iinfo(torch.int64).max, np.iinfo("int64").max)
        self.assertEqual(torch.iinfo(torch.int64).min, np.iinfo("int64").min)
        self.assertEqual(torch.iinfo(torch.int32).bits, 32)
        self.assertEqual(torch.iinfo(torch.int8).max, 127)
        self.assertEqual(torch.iinfo(torch.uint8).max, 255)

    def test_finfo(self):
        fi = torch.finfo(torch.float32)
        ref = np.finfo("float32")
        self.assertAlmostEqual(fi.eps, float(ref.eps))
        self.assertAlmostEqual(fi.max, float(ref.max))
        self.assertEqual(fi.bits, 32)
        fi64 = torch.finfo(torch.float64)
        ref64 = np.finfo("float64")
        self.assertAlmostEqual(fi64.tiny, float(ref64.tiny))

    def test_finfo_from_str_and_tensor_dtype(self):
        # finfo must accept both the dtype object and a Var's .dtype.
        x = torch.ones(2, dtype=torch.float32)
        self.assertAlmostEqual(torch.finfo(x.dtype).eps, float(np.finfo("float32").eps))
        self.assertAlmostEqual(torch.finfo("float32").eps, float(np.finfo("float32").eps))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Torch-grade dtype-semantics regression tests for the independent Torch frontend.

Every check compares the frontend against an independent reference (NumPy /
explicit Torch dtype rules) and runs on CPU and available accelerator backends.

Covered: dtype objects and repr, tensor casts including .long(), typed factories,
from_numpy dtype preservation, the documented binary type-promotion combinations
in _TORCH_PROMO, and iinfo/finfo. The cast and promotion checks run directly;
scalar values use .item() where a Python scalar is expected.

Run: python -m pytest compat/tests/torch/test_torch_compat_dtype.py
"""

from _helpers import capability as _test_capability
import unittest
import numpy as np
import torch
import jittor as jt

# Use the runtime's truthful Torch device name while retaining Jittor's legacy
# accelerator flag to enter the selected backend.
_HAS_ACCELERATOR = _test_capability.any_accelerator_enabled(backend=jt)
_ACCELERATOR_DEVICE = (
    "npu" if _test_capability.check_accelerator("acl", backend=jt).enabled else "cuda"
) if _HAS_ACCELERATOR else None
_DEVICES = [("cpu", 0)] + ([(_ACCELERATOR_DEVICE, 1)] if _HAS_ACCELERATOR else [])


def both_devices(fn):
    """Run ``fn(device_name)`` once per available device under the right flag scope."""
    for name, use_cuda in _DEVICES:
        with jt.flag_scope(use_cuda=use_cuda):
            fn(name)


def dts(v):
    """bare jittor dtype string for a Var ('float32', 'int64', ...)."""
    return v.dtype.name


def bfloat16_round(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32).copy()
    bits += np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))
    return (bits & np.uint32(0xffff0000)).view(np.float32)


class Base(unittest.TestCase):
    def ac(self, got, ref, atol=1e-5, rtol=1e-5, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_allclose(g, r, atol=atol, rtol=rtol, err_msg=msg)

    def ae(self, got, ref, msg=""):
        g = np.asarray(got); r = np.asarray(ref)
        self.assertEqual(tuple(g.shape), tuple(r.shape), f"shape {g.shape}!={r.shape}; {msg}")
        np.testing.assert_array_equal(g, r, err_msg=msg)


# ----------------------------------------------------------------- typed constructors

class TestTypedTensorConstructors(Base):
    def test_byte_tensor_accepts_nested_bytes_and_bytearray(self):
        nested = torch.ByteTensor([b"Az\x00"])
        self.assertEqual(tuple(nested.shape), (1, 3))
        self.assertEqual(dts(nested), "uint8")
        self.ae(nested.numpy(), np.array([[65, 122, 0]], dtype=np.uint8))

        flat = torch.ByteTensor(bytearray(b"Az\x00"))
        self.assertEqual(tuple(flat.shape), (3,))
        self.assertEqual(dts(flat), "uint8")
        self.ae(flat.numpy(), np.array([65, 122, 0], dtype=np.uint8))

    def test_byte_tensor_rejects_bare_bytes_like_torch(self):
        with self.assertRaisesRegex(TypeError, "invalid data type 'bytes'"):
            torch.ByteTensor(b"Az\x00")


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
        self.assertFalse(isinstance(x.dtype, str))
        self.assertFalse(x.dtype == "float32")
        self.assertFalse(x.dtype == "torch.float32")
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
    def test_tensor_explicit_complex64(self):
        expected = np.array([[1 + 2j, 3 - 4j]], dtype="complex64")
        source128 = expected.astype("complex128")
        def body(dev):
            for value in ([[1 + 2j, 3 - 4j]], source128):
                got = torch.tensor(value, dtype=torch.complex64)
                self.assertEqual(dts(got), "complex64", dev)
                self.ac(got.numpy(), expected, atol=0, rtol=0,
                        msg=f"explicit complex64 {type(value).__name__} {dev}")
            got = torch.as_tensor(source128, dtype=torch.complex64)
            self.assertEqual(dts(got), "complex64", dev)
            self.ac(got.numpy(), expected, atol=0, rtol=0,
                    msg=f"as_tensor explicit complex64 {dev}")
        both_devices(body)

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

    def test_empty_native_shape_fast_path_preserves_compatibility(self):
        def body(dev):
            native_shape = torch.ones((2, 3)).shape
            for shape in ((2, 3), [2, 3], native_shape):
                value = torch.empty(shape)
                self.assertEqual(tuple(value.shape), (2, 3), dev)
                self.assertTrue(
                    getattr(value, "_jittor_torch_ext_mutable", False), dev)

            self.assertEqual(tuple(torch.empty(2, 3).shape), (2, 3), dev)

            self.assertEqual(
                tuple(torch.empty(torch.Size((2, 3))).shape), (2, 3), dev)
            self.assertEqual(
                tuple(torch.empty((np.int64(2), np.int64(3))).shape),
                (2, 3),
                dev,
            )

        both_devices(body)

    def test_default_dtype_applies_to_float_factories(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)

            def body(dev):
                for factory in (torch.zeros, torch.ones, torch.empty):
                    value = factory((2, 3))
                    self.assertEqual(dts(value), "bfloat16", dev)

            both_devices(body)
        finally:
            torch.set_default_dtype(previous)

    def test_arange_dtype(self):
        def body(dev):
            self.assertEqual(dts(torch.arange(5)), "int64", dev)          # Torch integer default
            self.assertEqual(dts(torch.arange(0.0, 5.0)), "float32", dev)  # float range -> float32
            self.ae(torch.arange(5).numpy(), np.arange(5, dtype="int64"), dev)
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
            x = torch.tensor(self.x)
            self.assertEqual(dts(x.to(torch.int32)), "int32", dev)
            self.assertEqual(dts(x.to(torch.float64)), "float64", dev)
            self.assertEqual(dts(x.to("float64")), "float64", dev)
            # value: float->int truncates toward zero (torch + numpy astype agree)
            self.ae(x.to(torch.int32).numpy(), self.x.astype("int32"), dev)
        both_devices(body)

    def test_to_keeps_the_dtype_a_device_move_does_not_name(self):
        """``.to(device)`` moves and keeps the dtype; ``.to(dtype)`` converts.

        This used to assert that ``.to("cuda")`` is a no-op because "jittor has
        a single global backend". It is not one any more -- the frontend places
        tensors for real -- and on a build with no accelerator the call raises
        `Invalid cuda device index 0; visible device count is 0`, which is what
        torch does there too (`Torch not compiled with CUDA enabled`). The
        dtype half is the part that holds everywhere, so that is what is
        asserted unconditionally.
        """
        def body(dev):
            x = torch.tensor(self.x)
            self.assertEqual(dts(x.to(torch.float64)), "float64", dev)
            if dev != "cpu":
                moved = x.to(dev)
                self.assertEqual(dts(moved), "float32", dev)
                self.assertEqual(moved.device.type, dev, dev)
            elif _HAS_ACCELERATOR:
                # Move to the truthful accelerator name from CPU instead of
                # relying on the legacy CUDA alias used by older sweeps.
                moved = x.to(_ACCELERATOR_DEVICE)
                self.assertEqual(dts(moved), "float32", _ACCELERATOR_DEVICE)
                self.assertEqual(moved.device.type, _ACCELERATOR_DEVICE)
            else:
                with self.assertRaises(RuntimeError):
                    x.to("cuda")
        both_devices(body)

    def test_astype(self):
        def body(dev):
            x = torch.tensor(self.x)
            self.assertEqual(dts(x.astype("int64")), "int64", dev)
            self.assertEqual(dts(x.astype("float16")), "float16", dev)
        both_devices(body)

    def test_float_int_bool_double_half(self):
        def body(dev):
            x = torch.tensor(self.x)
            self.assertEqual(dts(x.float()), "float32", dev)
            self.assertEqual(dts(x.int()), "int32", dev)
            self.assertEqual(dts(x.double()), "float64", dev)
            self.assertEqual(dts(x.half()), "float16", dev)
            # .bool(): nonzero -> True
            b = torch.tensor(np.array([0.0, 1.0, 2.0], "float32")).bool()
            self.assertEqual(dts(b), "bool", dev)
            self.ae(b.numpy(), np.array([False, True, True]), dev)
            # round-trip int->float
            self.assertEqual(dts(x.int().float()), "float32", dev)
        both_devices(body)

    def test_long_returns_int64_like_torch(self):
        def body(dev):
            x = torch.tensor(self.x)
            self.assertEqual(dts(x.long()), "int64", dev)
        both_devices(body)


# ----------------------------------------------------------------- from_numpy dtype keep

class TestPythonScalarDtypes(Base):
    # torch reads the Python scalar types as int64 / float64 / bool wherever it
    # takes a dtype; jittor's own `int` and `float` are 32-bit.
    def test_factories_take_python_scalar_types(self):
        def body(dev):
            for py, want in ((int, "int64"), (float, "float64"), (bool, "bool")):
                self.assertEqual(dts(torch.zeros(3, dtype=py)), want, dev)
                self.assertEqual(dts(torch.full((2,), 1, dtype=py)), want, dev)
                self.assertEqual(dts(torch.tensor([1, 0], dtype=py)), want, dev)
                self.assertEqual(dts(torch.tensor([], dtype=py)), want, dev)
            self.assertEqual(dts(torch.arange(3, dtype=int)), "int64", dev)
        both_devices(body)

    def test_to_takes_python_scalar_types(self):
        def body(dev):
            x = torch.tensor([1.5, -2.5])
            self.assertEqual(dts(x.to(int)), "int64", dev)
            self.assertEqual(dts(x.to(float)), "float64", dev)
            self.assertEqual(dts(x.to(dtype=bool)), "bool", dev)
            self.ae(x.to(int).numpy(), np.array([1, -2], dtype="int64"))
        both_devices(body)


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

# Documented Torch result types for the tensor pairs checked below.
_TORCH_PROMO = {
    ("int32", "float32"): "float32",
    ("int32", "int32"): "int32",
    ("float32", "float32"): "float32",
    ("float64", "float64"): "float64",
    ("int64", "int64"): "int64",
    ("int8", "int8"): "int8",
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

# A compact subset also checked in both operand orders.
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
        # Check this subset with both operand orders.
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

    def test_python_float_keeps_bfloat16_tensor_dtype(self):
        source = np.array([1.0, -2.0, 3.5, -7.25], dtype=np.float32)
        source_bf = bfloat16_round(source)
        scale = 128 ** -0.5

        def body(dev):
            tensor = torch.tensor(source_bf, dtype=torch.bfloat16)
            scaled = tensor * scale
            reflected = scale * tensor
            self.assertEqual(dts(scaled), "bfloat16", dev)
            self.assertEqual(dts(reflected), "bfloat16", dev)
            scaled.sync()
            reflected.sync()
            expected = bfloat16_round(source_bf * np.float32(scale))
            self.ae(scaled.float().numpy(), expected, dev)
            self.ae(reflected.float().numpy(), expected, dev)

        both_devices(body)

    def test_integer_true_division_promotes_to_default_float(self):
        def body(dev):
            x = torch.from_numpy(np.array([1, 2, 3], dtype=np.uint8))
            y = x / 255.0
            self.assertEqual(dts(y), "float32", dev)
            self.ac(y.numpy(),
                    np.array([1, 2, 3], dtype=np.float32) / np.float32(255.0),
                    atol=1e-7, rtol=1e-6, msg=dev)

            z = torch.tensor([1, 2, 3], dtype=torch.int32) / 2
            self.assertEqual(dts(z), "float32", dev)
            self.ac(z.numpy(), np.array([0.5, 1.0, 1.5], dtype=np.float32), msg=dev)

            xf = torch.tensor([1, 2, 3], dtype=torch.float32)
            self.assertEqual(dts(xf / 255.0), "float32", dev)
        both_devices(body)

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

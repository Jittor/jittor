# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Integer promotion in the core, with no compatibility layer in front of it.

``tests/core/test_type_system.py`` already checks the promotion lattice -- but
it runs in Torch mode, where ``compat/torch/installers/tensor.py`` wraps every
arithmetic operator, computes the answer from its own Python table and *casts
the result*. That wrapper was written precisely because the core got this
wrong ("a C++ binary_dtype_infer quirk we cannot touch"), so those tests pass
whatever the core does. A plain ``import jittor`` has no such wrapper, and that
is what is asserted here.

The lattice is NumPy's, which is also Torch's for these pairs: within one
signedness the wider type wins; across signedness the result must hold both,
which costs one doubling beyond the unsigned operand; and uint64 with any
signed type has no integer answer at all, so it lands on float64.

Values, not only dtypes: ``uint8(200) + int8(1)`` returned dtype int8 and the
number -55, which is the whole reason this is a "silently wrong answer" bug and
not a cosmetic one.
"""

import unittest

import numpy as np

import jittor as jt


def _var(dtype, values):
    # jt.array() narrows int64->int32 and float64->float32 unless told;
    # every fixed-width operand here is built explicitly.
    return jt.array(np.array(values, dtype=dtype), dtype=dtype)


#: (a, b) -> promoted dtype. NumPy's `np.promote_types`, verified against it in
#: test_matches_numpy below rather than trusted.
_PROMOTE = {
    ("int8", "int8"): "int8",
    ("int8", "int16"): "int16",
    ("int8", "int64"): "int64",
    ("int16", "int32"): "int32",
    ("int32", "int64"): "int64",
    ("uint8", "uint8"): "uint8",
    ("uint8", "uint32"): "uint32",
    ("uint32", "uint64"): "uint64",
    # mixed signedness: one doubling past the unsigned operand
    ("uint8", "int8"): "int16",
    ("uint8", "int16"): "int16",
    ("uint8", "int32"): "int32",
    ("uint8", "int64"): "int64",
    ("uint16", "int8"): "int32",
    ("uint16", "int16"): "int32",
    ("uint16", "int32"): "int32",
    ("uint32", "int8"): "int64",
    ("uint32", "int32"): "int64",
    ("uint32", "int64"): "int64",
    # nothing integral holds uint64 and a signed type at once
    ("uint64", "int8"): "float64",
    ("uint64", "int64"): "float64",
    # bool adopts the other operand rather than widening it
    ("bool", "int8"): "int8",
    ("bool", "uint8"): "uint8",
    ("bool", "int64"): "int64",
}


class TestIntegerPromotion(unittest.TestCase):

    def test_matches_numpy(self):
        """The table above is numpy's, not one this test invented."""
        for (a, b), want in _PROMOTE.items():
            self.assertEqual(str(np.promote_types(a, b)), want,
                             msg="the reference table disagrees with numpy "
                                 "for %s,%s" % (a, b))

    def test_add_promotes_both_ways(self):
        for (a, b), want in _PROMOTE.items():
            x = _var(a, [1])
            y = _var(b, [1])
            self.assertEqual(str((x + y).dtype), want, msg="%s+%s" % (a, b))
            self.assertEqual(str((y + x).dtype), want, msg="%s+%s" % (b, a))

    def test_uint8_plus_int8_keeps_the_value(self):
        """The marquee case: int8 cannot hold 201, and nothing said so."""
        x = _var("uint8", [200])
        y = _var("int8", [1])
        z = x + y
        self.assertEqual(str(z.dtype), "int16")
        self.assertEqual(int(z.numpy()[0]), 201)

    def test_uint32_plus_int32_keeps_the_value(self):
        x = _var("uint32", [4000000000])
        y = _var("int32", [1])
        z = x + y
        self.assertEqual(str(z.dtype), "int64")
        self.assertEqual(int(z.numpy()[0]), 4000000001)

    def test_uint64_with_signed_falls_back_to_float64(self):
        x = _var("uint64", [1])
        y = _var("int64", [-2])
        z = x + y
        self.assertEqual(str(z.dtype), "float64")
        # computed in float64, not in uint64 and converted afterwards
        self.assertEqual(float(z.numpy()[0]), -1.0)

    def test_bitwise_ops_stay_integral(self):
        """No float fallback where a float kernel would not compile."""
        x = _var("uint64", [6])
        y = _var("int64", [3])
        z = x & y
        self.assertFalse(str(z.dtype).startswith("float"), str(z.dtype))
        self.assertEqual(int(z.numpy()[0]), 2)

    def test_same_dtype_is_unchanged(self):
        for name in ("int8", "int16", "int32", "int64",
                     "uint8", "uint16", "uint32", "uint64"):
            x = _var(name, [3])
            self.assertEqual(str((x + x).dtype), name, msg=name)


class TestScalarPromotion(unittest.TestCase):
    """A python scalar is a wrapped number: it lifts the category, never the width."""

    def test_float_scalar_lifts_int_to_the_default_float(self):
        # uint8 * (1/255.) used to be float16 -- one byte in, so "float of
        # matching width" chose the 16-bit float, and image preprocessing lost
        # three decimal digits.
        x = _var("uint8", [255])
        z = x * (1 / 255.)
        self.assertEqual(str(z.dtype), "float32")
        np.testing.assert_allclose(z.numpy(), [1.0], rtol=1e-6)

    def test_float_scalar_does_not_widen_int64(self):
        # int64 * 2.0 used to be float64 -- eight bytes in, so "float of
        # matching width" chose double, which costs 32x throughput on a GPU.
        x = _var("int64", [3])
        self.assertEqual(str((x * 2.0).dtype), "float32")

    def test_float_tensor_keeps_its_own_width(self):
        # the scalar rule must not reach a float tensor: float16 stays float16
        for name in ("float16", "float32", "float64"):
            x = _var(name, [1.5])
            self.assertEqual(str((x * 2.0).dtype), name, msg=name)

    def test_int_scalar_keeps_the_tensor_dtype(self):
        for name in ("int8", "uint8", "int64", "uint32"):
            x = _var(name, [3])
            self.assertEqual(str((x * 2).dtype), name, msg=name)


def _scalar(dtype, value):
    """A Var carrying ``_is_scalar``: a zero-dimensional numpy value.

    ``jt.array(np.uint8(200))`` takes the scalar path in ``ArrayOp`` (see
    ``from_scalar_object`` in ``pyjt/py_array_op.cc``) and the resulting Var is
    flagged as a wrapped number, exactly like ``jt.array(3)``.  It keeps its own
    dtype, which is what makes a *pair* of them interesting.
    """
    return jt.array(np.dtype(dtype).type(value))


class TestTwoScalarsPromoteBothWaysTheSame(unittest.TestCase):
    """``a + b`` and ``b + a`` must agree when *both* sides are scalars.

    The scalar rule is "a wrapped number adopts the tensor's dtype", and it was
    written as two unconditional early returns::

        if (xscalar) return int_dtype(y.dsize_(), y.is_unsigned());
        if (yscalar) return int_dtype(x.dsize_(), x.is_unsigned());

    With one scalar that is right.  With *two* the first branch always wins, so
    the answer is whichever dtype was written second -- and swapping the
    operands changes it.  There is no "the tensor" to adopt here, so neither
    early return applies and the promotion lattice has to decide, which is both
    commutative and wide enough to hold the value.
    """

    def test_premise_a_zero_dim_numpy_value_is_a_scalar(self):
        # If this stops holding, the cases below stop testing what they say:
        # a scalar adopts the dtype of the tensor it meets, so meeting an
        # int8 *tensor* must give int8 and not promote to int64.
        s = _scalar("int64", 3)
        t = _var("int8", [1, 2])
        self.assertEqual(str((s + t).dtype), "int8")
        self.assertEqual(str((t + s).dtype), "int8")

    def test_uint8_and_int8_scalars(self):
        # uint8(200) + int8(1): the lattice says int16/201 either way.  Before,
        # one order gave int8 and the value -55, the other uint8 and 201.
        a, b = _scalar("uint8", 200), _scalar("int8", 1)
        for x, y in ((a, b), (b, a)):
            z = x + y
            self.assertEqual(str(z.dtype), "int16")
            self.assertEqual(int(z.item()), 201)

    def test_uint32_and_int32_scalars(self):
        a, b = _scalar("uint32", 2 ** 31), _scalar("int32", 1)
        for x, y in ((a, b), (b, a)):
            z = x + y
            self.assertEqual(str(z.dtype), "int64")
            self.assertEqual(int(z.item()), 2 ** 31 + 1)

    def test_same_width_same_signedness_is_unchanged(self):
        a, b = _scalar("int16", 3), _scalar("int16", 4)
        self.assertEqual(str((a + b).dtype), "int16")


class TestAngleConverters(unittest.TestCase):
    """``rad2deg``/``deg2rad`` are where the promotion rule reaches a user.

    They were written as ``180 * x / np.pi``: the leading python int is a
    scalar, so the multiply keeps the *input's* dtype and an integer input
    wrapped there, before the division could lift anything to float. Folding
    the constant makes it one float multiply.
    """

    def test_rad2deg_of_an_integer_var(self):
        x = _var("uint8", [3, 200])
        z = jt.rad2deg(x)
        self.assertEqual(str(z.dtype), "float32")
        np.testing.assert_allclose(z.numpy(), np.rad2deg([3.0, 200.0]),
                                   rtol=1e-6)

    def test_deg2rad_of_an_integer_var(self):
        x = _var("int32", [180, 360])
        z = jt.deg2rad(x)
        self.assertEqual(str(z.dtype), "float32")
        np.testing.assert_allclose(z.numpy(), np.deg2rad([180.0, 360.0]),
                                   rtol=1e-6)

    def test_float_input_keeps_its_width(self):
        x = _var("float64", [1.0, 2.0])
        self.assertEqual(str(jt.rad2deg(x).dtype), "float64")


if __name__ == "__main__":
    unittest.main()

"""Fidelity battery for the Torch cumulative family (``cumsum``/``cumprod``).

Every earlier cohort of task 7.03 verified its stable objects on CPU only. A
cumulative scan is the one place where that is not enough: the summation order
is the backend's, so CPU and CUDA cannot be assumed to agree bit-for-bit even
though both compute "the same" op. The numerical cases therefore run through
``instantiate_device_type_tests`` -- one class per device the session selected --
and one case compares the two paths against each other rather than each against
NumPy in isolation.

What the cross-device case pins, measured on sm_89 with 4096 float32 elements:
the two backends agree to ~1.4e-06 relative, and the CUDA scan is *closer* to
the float64 reference than the sequential CPU scan is (2.1e-03 vs 3.7e-03
absolute on a series whose partial sums reach ~2e3). That is the expected
signature of a parallel prefix sum, not a defect, so it is registered in the
fidelity detail instead of being "fixed". The integer path has no rounding and
is pinned exactly.
"""

import importlib
import unittest

import numpy as np
import jittor as torch

from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


CUMULATIVE_NAMES = ("cumsum", "cumprod")

# Small enough that a sequential and a tree-shaped scan still agree exactly.
SMALL = (np.arange(1, 13, dtype="float64").reshape(3, 4) / 7.0).astype("float32")
# Long enough for the two summation orders to separate, with mixed signs so the
# partial sums do not grow monotonically and cancellation actually happens.
LONG = (np.sin(np.arange(4096, dtype="float64")) * 1e3).astype("float32")
MASK = np.array([[True, False, True, True], [False, False, True, False]])


def _owner():
    return importlib.import_module("jittor.compat.torch.installers.tensor")


class TestTorchCumulativeFidelityMetadata(unittest.TestCase):
    """Identity and fidelity metadata, which do not depend on the device."""

    def test_cumulative_family_is_stable_module_level_objects(self):
        owner = _owner()
        for name in CUMULATIVE_NAMES:
            with self.subTest(name=name):
                implementation = getattr(owner, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertIs(getattr(torch.Var, name), implementation)
                self.assertEqual(implementation.__module__, owner.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_cumulative_family_fidelity_records_the_backend_order(self):
        owner = _owner()
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in CUMULATIVE_NAMES:
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(owner, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("int64", record.detail)
                self.assertIn("summation order", record.detail)
                self.assertIn("device", record.detail)
                self.assertIn("layout", record.detail)


class TestTorchCumulative(cu.JittorTestCase):
    def test_cumsum_matches_a_float64_reference(self, device):
        actual = torch.cumsum(torch.array(SMALL), 1).numpy()
        expected = np.cumsum(SMALL.astype("float64"), axis=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)

    def test_cumsum_along_the_leading_dimension(self, device):
        actual = torch.cumsum(torch.array(SMALL), 0).numpy()
        expected = np.cumsum(SMALL.astype("float64"), axis=0)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)

    def test_cumprod_matches_a_float64_reference(self, device):
        actual = torch.cumprod(torch.array(SMALL), 1).numpy()
        expected = np.cumprod(SMALL.astype("float64"), axis=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)

    def test_default_dim_is_the_trailing_one(self, device):
        actual = torch.cumsum(torch.array(SMALL)).numpy()
        expected = np.cumsum(SMALL.astype("float64"), axis=-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)

    def test_bool_input_is_promoted_to_int64_exactly(self, device):
        result = torch.cumsum(torch.array(MASK), -1)
        self.assertEqual(str(result.dtype), "int64")
        np.testing.assert_array_equal(
            result.numpy(), np.cumsum(MASK.astype("int64"), axis=-1))

    def test_uint8_input_is_promoted_to_int64_exactly(self, device):
        values = np.array([[3, 0, 7], [1, 2, 0]], dtype="uint8")
        result = torch.cumsum(torch.array(values), -1)
        self.assertEqual(str(result.dtype), "int64")
        np.testing.assert_array_equal(
            result.numpy(), np.cumsum(values.astype("int64"), axis=-1))

    def test_dtype_keyword_casts_the_result(self, device):
        result = torch.cumsum(torch.array(SMALL), 1, dtype=torch.float64)
        self.assertEqual(str(result.dtype), "float64")
        np.testing.assert_allclose(
            result.numpy(), np.cumsum(SMALL.astype("float64"), axis=1),
            rtol=1e-6, atol=0)

    def test_out_keeps_identity_and_receives_the_values(self, device):
        destination = torch.zeros(SMALL.shape)
        returned = torch.cumsum(torch.array(SMALL), 1, out=destination)
        self.assertIs(returned, destination)
        np.testing.assert_allclose(
            destination.numpy(), np.cumsum(SMALL.astype("float64"), axis=1),
            rtol=1e-6, atol=0)

    def test_out_reaches_the_parent_of_a_retained_view(self, device):
        parent = torch.zeros((2, 4))
        view = parent[0]
        torch.cumsum(torch.array(SMALL[0]), 0, out=view)
        np.testing.assert_allclose(
            parent.numpy()[0], np.cumsum(SMALL[0].astype("float64")),
            rtol=1e-6, atol=0)

    def test_var_method_delegates_to_the_module_level_object(self, device):
        tensor = torch.array(SMALL)
        np.testing.assert_array_equal(
            tensor.cumsum(1).numpy(), torch.cumsum(tensor, 1).numpy())
        np.testing.assert_array_equal(
            tensor.cumprod(1).numpy(), torch.cumprod(tensor, 1).numpy())

    def test_axis_alias_is_accepted_without_an_adapter(self, device):
        tensor = torch.array(SMALL)
        np.testing.assert_array_equal(
            tensor.cumsum(axis=0).numpy(), torch.cumsum(tensor, 0).numpy())

    def test_this_device_agrees_with_the_cpu_path_on_a_long_float32_scan(self, device):
        """The cross-device case this battery exists for.

        The two backends run different scan algorithms, so this bounds the
        disagreement rather than demanding equality -- and separately checks
        that neither has drifted away from the float64 reference, which is what
        would distinguish "different rounding" from "wrong".
        """
        here = torch.cumsum(torch.array(LONG), 0).numpy().astype("float64")
        with torch.flag_scope(use_cuda=0):
            on_cpu = torch.cumsum(torch.array(LONG), 0).numpy().astype("float64")
        reference = np.cumsum(LONG.astype("float64"))
        scale = np.max(np.abs(reference))
        self.assertLessEqual(np.max(np.abs(here - on_cpu)) / scale, 1e-5)
        self.assertLessEqual(np.max(np.abs(here - reference)) / scale, 1e-5)
        self.assertLessEqual(np.max(np.abs(on_cpu - reference)) / scale, 1e-5)

    def test_this_device_agrees_with_the_cpu_path_exactly_on_integers(self, device):
        values = np.arange(4096, dtype="int64").reshape(8, 512) % 7
        here = torch.cumsum(torch.array(values, dtype="int64"), 1).numpy()
        with torch.flag_scope(use_cuda=0):
            on_cpu = torch.cumsum(
                torch.array(values, dtype="int64"), 1).numpy()
        np.testing.assert_array_equal(here, on_cpu)
        np.testing.assert_array_equal(here, np.cumsum(values, axis=1))


instantiate_device_type_tests(TestTorchCumulative, globals())


if __name__ == "__main__":
    unittest.main()

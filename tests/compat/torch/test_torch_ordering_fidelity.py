"""Fidelity battery for the Torch ordering family (sort/argsort/topk/median).

Run on every device the session selected, because this family is the second
place (after the cumulative scan) where the two backends genuinely disagree:
the sort is not stable, so on rows with duplicate keys CPU and CUDA return
*different indices* for the *same* values. Measured on sm_89 with 8x512 rows
drawn from ``arange % 97`` (about five duplicates per key): the sorted values
are bit-identical between the two, the argsort indices are not.

The cases below therefore pin the invariant that actually holds -- values are
exact, and the indices returned are consistent with the values they claim to
address -- and pin exact index equality only where the keys are distinct.
Asserting index equality across devices on a tie would be pinning an accident
of the backend, and demanding stability would be pinning a contract Torch does
not offer either (``torch.sort`` defaults to ``stable=False``).
"""

import importlib
import unittest

import numpy as np
import jittor as torch

from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


ORDERING_NAMES = ("sort", "argsort", "topk", "median")

DISTINCT = np.array([[3.0, 1.0, 4.0, 1.5], [9.0, 2.0, 6.0, 5.0]], dtype="float32")
# ~5 duplicates per key per row: enough for two sort implementations to disagree
# about which equal element comes first.
DUPLICATED = (np.arange(4096, dtype="float32") % 97).reshape(8, 512)


def _owner():
    return importlib.import_module("jittor.compat.torch.installers.tensor")


class TestTorchOrderingFidelityMetadata(unittest.TestCase):
    """Identity and fidelity metadata, which do not depend on the device."""

    def test_ordering_family_is_stable_module_level_objects(self):
        owner = _owner()
        for name in ORDERING_NAMES:
            with self.subTest(name=name):
                implementation = getattr(owner, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertIs(getattr(torch.Var, name), implementation)
                self.assertEqual(implementation.__module__, owner.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_ordering_family_fidelity_records_the_tie_behaviour(self):
        owner = _owner()
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ORDERING_NAMES:
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(owner, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("not stable", record.detail)
                self.assertIn("int64", record.detail)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)


class TestTorchOrdering(cu.JittorTestCase):
    def test_sort_returns_values_then_indices(self, device):
        result = torch.sort(torch.array(DISTINCT), dim=1)
        self.assertEqual(result._fields, ("values", "indices"))
        np.testing.assert_array_equal(
            result.values.numpy(), np.sort(DISTINCT, axis=1))
        np.testing.assert_array_equal(
            result.indices.numpy(), np.argsort(DISTINCT, axis=1))
        self.assertEqual(str(result.indices.dtype), "int64")

    def test_sort_descending_reverses_the_order(self, device):
        actual = torch.sort(torch.array(DISTINCT), dim=1, descending=True)
        np.testing.assert_array_equal(
            actual.values.numpy(), np.sort(DISTINCT, axis=1)[:, ::-1])

    def test_argsort_returns_only_int64_indices(self, device):
        actual = torch.argsort(torch.array(DISTINCT), dim=1)
        self.assertEqual(str(actual.dtype), "int64")
        np.testing.assert_array_equal(
            actual.numpy(), np.argsort(DISTINCT, axis=1))

    def test_topk_returns_the_largest_and_the_smallest(self, device):
        largest = torch.topk(torch.array(DISTINCT), 2, dim=1)
        smallest = torch.topk(torch.array(DISTINCT), 2, dim=1, largest=False)
        np.testing.assert_array_equal(
            largest.values.numpy(), np.sort(DISTINCT, axis=1)[:, ::-1][:, :2])
        np.testing.assert_array_equal(
            smallest.values.numpy(), np.sort(DISTINCT, axis=1)[:, :2])
        self.assertEqual(str(largest.indices.dtype), "int64")

    def test_topk_indices_address_the_values_they_returned(self, device):
        result = torch.topk(torch.array(DISTINCT), 3, dim=1)
        gathered = np.take_along_axis(
            DISTINCT, result.indices.numpy().astype("int64"), axis=1)
        np.testing.assert_array_equal(gathered, result.values.numpy())

    def test_median_takes_the_lower_of_two_middles(self, device):
        values = np.array([[4.0, 1.0, 3.0, 2.0]], dtype="float32")
        result = torch.median(torch.array(values), dim=1)
        np.testing.assert_array_equal(result.values.numpy(), np.array([2.0]))
        np.testing.assert_array_equal(result.indices.numpy(), np.array([3]))
        self.assertEqual(str(result.indices.dtype), "int64")

    def test_median_keepdim_and_full_reduction(self, device):
        tensor = torch.array(DISTINCT)
        keepdim = torch.median(tensor, dim=1, keepdim=True)
        self.assertEqual(tuple(keepdim.values.shape), (2, 1))
        np.testing.assert_array_equal(
            keepdim.values.numpy().reshape(-1),
            torch.median(tensor, dim=1).values.numpy())
        full = torch.median(tensor).numpy()
        np.testing.assert_array_equal(
            full, np.sort(DISTINCT, axis=None)[(DISTINCT.size - 1) // 2])

    def test_median_rejects_a_dimension_out_of_range(self, device):
        with self.assertRaises(IndexError):
            torch.median(torch.array(DISTINCT), dim=5)

    def test_var_methods_delegate_to_the_module_level_objects(self, device):
        tensor = torch.array(DISTINCT)
        np.testing.assert_array_equal(
            tensor.sort(dim=1).values.numpy(),
            torch.sort(tensor, dim=1).values.numpy())
        np.testing.assert_array_equal(
            tensor.argsort(dim=1).numpy(), torch.argsort(tensor, dim=1).numpy())
        np.testing.assert_array_equal(
            tensor.topk(2, dim=1).values.numpy(),
            torch.topk(tensor, 2, dim=1).values.numpy())
        np.testing.assert_array_equal(
            tensor.median(dim=1).values.numpy(),
            torch.median(tensor, dim=1).values.numpy())

    def test_this_device_agrees_with_the_cpu_path_on_the_sorted_values(self, device):
        """Values are the contract; indices under ties are not.

        The same rows are sorted here and on CPU. The values must match exactly
        -- a sort has no rounding to hide behind -- and each device's indices
        must address its own values, which is the property a caller can rely on
        when the keys are not unique.
        """
        here = torch.sort(torch.array(DUPLICATED), dim=1)
        here_values = here.values.numpy()
        here_indices = here.indices.numpy().astype("int64")
        with torch.flag_scope(use_cuda=0):
            on_cpu = torch.sort(torch.array(DUPLICATED), dim=1)
            cpu_values = on_cpu.values.numpy()
        np.testing.assert_array_equal(here_values, cpu_values)
        np.testing.assert_array_equal(
            here_values, np.sort(DUPLICATED, axis=1))
        np.testing.assert_array_equal(
            np.take_along_axis(DUPLICATED, here_indices, axis=1), here_values)

    def test_this_device_agrees_with_the_cpu_path_exactly_on_distinct_keys(self, device):
        keys = (np.arange(4096, dtype="float32") * 1.5).reshape(8, 512)
        keys = keys[:, ::-1].copy()
        here = torch.argsort(torch.array(keys), dim=1).numpy()
        with torch.flag_scope(use_cuda=0):
            on_cpu = torch.argsort(torch.array(keys), dim=1).numpy()
        np.testing.assert_array_equal(here, on_cpu)
        np.testing.assert_array_equal(here, np.argsort(keys, axis=1))


instantiate_device_type_tests(TestTorchOrdering, globals())


if __name__ == "__main__":
    unittest.main()

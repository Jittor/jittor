import importlib
import unittest

import numpy as np
import jittor as torch


STACKING_NAMES = (
    "vstack", "row_stack", "hstack", "dstack", "column_stack",
)


class TestTorchNumericalFidelity(unittest.TestCase):
    def test_eye_is_a_stable_module_level_object(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        self.assertIs(torch.eye, numerical.eye)

    def test_eye_fidelity_is_queryable(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch.eye")
        self.assertIs(record.implementation, numerical.eye)
        self.assertEqual(record.level, fidelity.Fidelity.APPROXIMATE)
        self.assertIn("device", record.detail)

    def test_eye_cpu_values_and_dtype(self):
        with torch.flag_scope(use_cuda=0):
            actual = torch.eye(2, 3, dtype=torch.float64)
            np.testing.assert_array_equal(
                actual.numpy(), np.eye(2, 3, dtype=np.float64))

    def test_stacking_is_stable_module_level_and_keeps_public_identity(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        for name in STACKING_NAMES:
            with self.subTest(name=name):
                implementation = getattr(numerical, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__module__, numerical.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_stacking_fidelity_is_queryable_and_conservative(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in STACKING_NAMES:
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(numerical, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)

    def test_stacking_cpu_1d_matches_numpy(self):
        with torch.flag_scope(use_cuda=0):
            a = torch.array([1.0, 2.0, 3.0])
            b = torch.array([4.0, 5.0, 6.0])
            np_a = np.array([1.0, 2.0, 3.0])
            np_b = np.array([4.0, 5.0, 6.0])
            np.testing.assert_array_equal(
                torch.vstack([a, b]).numpy(), np.vstack([np_a, np_b]))
            np.testing.assert_array_equal(
                torch.row_stack([a, b]).numpy(), np.row_stack([np_a, np_b]))
            np.testing.assert_array_equal(
                torch.hstack([a, b]).numpy(), np.hstack([np_a, np_b]))
            np.testing.assert_array_equal(
                torch.dstack([a, b]).numpy(), np.dstack([np_a, np_b]))
            np.testing.assert_array_equal(
                torch.column_stack([a, b]).numpy(),
                np.column_stack([np_a, np_b]))

    def test_stacking_cpu_2d_and_mixed_inputs_match_numpy(self):
        with torch.flag_scope(use_cuda=0):
            a = torch.array([[1.0, 2.0, 3.0]])
            b = torch.array([[4.0, 5.0, 6.0]])
            one_d = torch.array([7.0, 8.0, 9.0])
            np_a = np.array([[1.0, 2.0, 3.0]])
            np_b = np.array([[4.0, 5.0, 6.0]])
            np_one_d = np.array([7.0, 8.0, 9.0])
            np.testing.assert_array_equal(
                torch.vstack([a, b]).numpy(), np.vstack([np_a, np_b]))
            np.testing.assert_array_equal(
                torch.hstack([a, b]).numpy(), np.hstack([np_a, np_b]))
            np.testing.assert_array_equal(
                torch.dstack([a, b]).numpy(), np.dstack([np_a, np_b]))
            np.testing.assert_array_equal(
                torch.column_stack([a, b]).numpy(),
                np.column_stack([np_a, np_b]))
            np.testing.assert_array_equal(
                torch.vstack([one_d, a]).numpy(),
                np.vstack([np_one_d, np_a]))


if __name__ == "__main__":
    unittest.main()

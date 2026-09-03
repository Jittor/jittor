import importlib
import unittest

import numpy as np
import jittor as torch


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


if __name__ == "__main__":
    unittest.main()

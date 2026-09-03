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

    def test_movedim_and_moveaxis_are_stable_module_level_objects(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        for name in ("movedim", "moveaxis"):
            with self.subTest(name=name):
                implementation = getattr(numerical, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__module__, numerical.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_movedim_and_moveaxis_fidelity_is_queryable(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("movedim", "moveaxis"):
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(numerical, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)

    def test_movedim_cpu_positive_and_negative_single_axis_matches_numpy(self):
        values = np.arange(24).reshape(2, 3, 4).astype("float32")
        with torch.flag_scope(use_cuda=0):
            actual = torch.movedim(torch.array(values), 0, 2).numpy()
            negative = torch.movedim(torch.array(values), -1, 0).numpy()
        np.testing.assert_array_equal(actual, np.moveaxis(values, 0, 2))
        np.testing.assert_array_equal(negative, np.moveaxis(values, -1, 0))

    def test_moveaxis_cpu_multi_axis_matches_numpy(self):
        values = np.arange(24).reshape(2, 3, 4).astype("float32")
        with torch.flag_scope(use_cuda=0):
            actual = torch.moveaxis(
                torch.array(values), (0, 1), (2, 0)).numpy()
        np.testing.assert_array_equal(
            actual, np.moveaxis(values, (0, 1), (2, 0)))

    def test_movedim_var_methods_use_the_family_internal_implementation(self):
        values = np.arange(24).reshape(2, 3, 4).astype("float32")
        with torch.flag_scope(use_cuda=0):
            tensor = torch.array(values)
            actual = tensor.movedim(0, 2).numpy()
            negative = tensor.moveaxis(-1, 0).numpy()
        np.testing.assert_array_equal(actual, np.moveaxis(values, 0, 2))
        np.testing.assert_array_equal(negative, np.moveaxis(values, -1, 0))

    def test_shape_helpers_are_stable_module_level_objects(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        for name in ("unflatten", "swapaxes", "swapdims", "ravel"):
            with self.subTest(name=name):
                implementation = getattr(numerical, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__module__, numerical.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_shape_helpers_fidelity_is_queryable_and_conservative(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("unflatten", "swapaxes", "swapdims", "ravel"):
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(numerical, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)

    def test_shape_helpers_cpu_match_numpy(self):
        values = np.arange(24).reshape(2, 3, 4).astype("float32")
        with torch.flag_scope(use_cuda=0):
            tensor = torch.array(values)
            flat_tensor = tensor.reshape(2, 12)
            unflattened = torch.unflatten(flat_tensor, 1, (3, 4))
            swapped = torch.swapaxes(tensor, 0, -1)
            swapdim_alias = torch.swapdims(tensor, 0, 2)
            flattened = torch.ravel(tensor)
        np.testing.assert_array_equal(
            unflattened.numpy(), values.reshape(2, 3, 4))
        np.testing.assert_array_equal(
            swapped.numpy(), np.swapaxes(values, 0, -1))
        np.testing.assert_array_equal(
            swapdim_alias.numpy(), np.swapaxes(values, 0, 2))
        np.testing.assert_array_equal(flattened.numpy(), values.ravel())
        np.testing.assert_array_equal(
            flat_tensor.unflatten(1, (3, 4)).numpy(), values)
        np.testing.assert_array_equal(
            tensor.swapdims(0, 2).numpy(), np.swapaxes(values, 0, 2))
        np.testing.assert_array_equal(tensor.ravel().numpy(), values.ravel())

    def test_elementwise_sign_family_is_stable_module_level_objects(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        for name in ("copysign", "xlogy", "heaviside", "signbit"):
            with self.subTest(name=name):
                implementation = getattr(numerical, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__module__, numerical.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_elementwise_sign_family_fidelity_is_queryable(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("copysign", "xlogy", "heaviside", "signbit"):
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(numerical, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)

    def test_elementwise_sign_family_cpu_copysign_and_xlogy_matches_numpy(self):
        magnitude = np.array([1.0, 2.0, 3.0], dtype="float32")
        signs = np.array([-1.0, 0.0, 1.0], dtype="float32")
        x_values = np.array([1.0, 2.0, 0.0], dtype="float32")
        y_values = np.array([2.0, 3.0, 0.0], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            actual_sign = torch.copysign(
                torch.array(magnitude), torch.array(signs)).numpy()
            actual_xlogy = torch.xlogy(
                torch.array(x_values), torch.array(y_values)).numpy()
            method_sign = torch.array(magnitude).copysign(torch.array(signs)).numpy()
        np.testing.assert_array_equal(actual_sign, np.copysign(magnitude, signs))
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_xlogy = np.where(
                x_values == 0, 0.0, x_values * np.log(y_values))
        np.testing.assert_allclose(actual_xlogy, expected_xlogy, rtol=1e-6)
        np.testing.assert_array_equal(method_sign, np.copysign(magnitude, signs))

    def test_elementwise_sign_family_cpu_heaviside_and_signbit_matches_numpy(self):
        values = np.array([-1.0, 0.0, 2.0], dtype="float32")
        steps = np.array([3.0, 4.0, 5.0], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            actual_step = torch.heaviside(
                torch.array(values), torch.array(steps)).numpy()
            actual_signbit = torch.signbit(torch.array(values)).numpy()
            method_step = torch.array(values).heaviside(torch.array(steps)).numpy()
            method_signbit = torch.array(values).signbit().numpy()
        np.testing.assert_array_equal(actual_step, np.heaviside(values, steps))
        np.testing.assert_array_equal(actual_signbit, np.signbit(values))
        np.testing.assert_array_equal(method_step, np.heaviside(values, steps))
        np.testing.assert_array_equal(method_signbit, np.signbit(values))


if __name__ == "__main__":
    unittest.main()

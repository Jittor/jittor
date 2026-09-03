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

    def test_matrix_family_is_stable_module_level_objects(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        for name in ("trace", "diag_embed", "diagflat"):
            with self.subTest(name=name):
                implementation = getattr(numerical, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__module__, numerical.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_matrix_family_fidelity_is_queryable(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("trace", "diag_embed", "diagflat"):
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(numerical, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)

    def test_matrix_family_cpu_trace_and_diag_embed_match_numpy(self):
        matrix = np.arange(9).reshape(3, 3).astype("float32")
        rows = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            actual_trace = torch.trace(torch.array(matrix)).numpy()
            actual_embed = torch.diag_embed(torch.array(rows)).numpy()
        np.testing.assert_array_equal(actual_trace, np.trace(matrix))
        np.testing.assert_array_equal(
            actual_embed, np.stack([np.diag(row) for row in rows]))

    def test_matrix_family_cpu_diagflat_and_var_methods_match_numpy(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        rows = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            tensor = torch.array(values)
            actual_diagflat = torch.diagflat(tensor).numpy()
            actual_trace = torch.array(values).trace().numpy()
            actual_embed = torch.array(rows).diag_embed().numpy()
        np.testing.assert_array_equal(actual_diagflat, np.diagflat(values))
        np.testing.assert_array_equal(actual_trace, np.trace(values))
        np.testing.assert_array_equal(
            actual_embed, np.stack([np.diag(row) for row in rows]))

    def test_float_power_is_a_stable_module_level_object(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        self.assertTrue(callable(numerical.float_power))
        self.assertIs(torch.float_power, numerical.float_power)
        self.assertEqual(numerical.float_power.__module__, numerical.__name__)
        self.assertEqual(numerical.float_power.__name__, "float_power")

    def test_float_power_fidelity_is_queryable_and_conservative(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch.float_power")
        self.assertIs(record.implementation, numerical.float_power)
        self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
        self.assertIn("device", record.detail)
        self.assertIn("out", record.detail)

    def test_float_power_cpu_matches_numpy_values_and_float64_dtype(self):
        base = np.array([1.5, 2.0, 3.0], dtype="float32")
        exponent = np.array([2.0, 0.5, 3.0], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            actual_scalar = torch.float_power(torch.array(base), 2.0)
            actual_tensor = torch.float_power(
                torch.array(base), torch.array(exponent))
            actual_method = torch.array(base).float_power(2.0)
        np.testing.assert_allclose(
            actual_scalar.numpy(), np.float_power(base, 2.0), rtol=1e-6)
        np.testing.assert_allclose(
            actual_tensor.numpy(), np.float_power(base, exponent), rtol=1e-6)
        np.testing.assert_allclose(
            actual_method.numpy(), np.float_power(base, 2.0), rtol=1e-6)
        self.assertEqual(str(actual_scalar.dtype), "float64")
        self.assertEqual(str(actual_tensor.dtype), "float64")

    def test_close_family_is_stable_module_level_objects(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        for name in ("isclose", "allclose"):
            with self.subTest(name=name):
                implementation = getattr(numerical, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__module__, numerical.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_close_family_fidelity_is_queryable_and_conservative(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("isclose", "allclose"):
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(numerical, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)

    def test_close_family_cpu_values_equal_nan_and_allclose_bool(self):
        left = np.array([1.0, 2.0, np.nan], dtype="float32")
        right = np.array([1.0, 2.00001, np.nan], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            actual = torch.isclose(
                torch.array(left), torch.array(right), equal_nan=True).numpy()
            all_false = torch.allclose(
                torch.array(left), torch.array(right), equal_nan=False)
            all_true = torch.allclose(
                torch.array(left), torch.array(right), equal_nan=True)
        np.testing.assert_array_equal(actual, np.isclose(left, right, equal_nan=True))
        self.assertIs(type(all_false), bool)
        self.assertIs(type(all_true), bool)
        self.assertEqual(all_false, np.allclose(left, right, equal_nan=False))
        self.assertEqual(all_true, np.allclose(left, right, equal_nan=True))

    def test_pairwise_search_family_is_stable_module_level_objects(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        for name in ("cdist", "bucketize"):
            with self.subTest(name=name):
                implementation = getattr(numerical, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__module__, numerical.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_pairwise_search_family_fidelity_is_queryable(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("cdist", "bucketize"):
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(numerical, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)

    def test_pairwise_search_family_cpu_cdist_p1_p2_matches_numpy(self):
        left = np.array([[0.0, 1.0], [2.0, 3.0]], dtype="float32")
        right = np.array([[1.0, 1.0], [4.0, 5.0], [-1.0, 2.0]], dtype="float32")
        delta = left[:, None, :] - right[None, :, :]
        with torch.flag_scope(use_cuda=0):
            actual_p1 = torch.cdist(
                torch.array(left), torch.array(right), p=1).numpy()
            actual_p2 = torch.cdist(
                torch.array(left), torch.array(right), p=2).numpy()
        np.testing.assert_allclose(
            actual_p1, np.abs(delta).sum(axis=-1), rtol=1e-6)
        np.testing.assert_allclose(
            actual_p2, np.sqrt((delta * delta).sum(axis=-1)), rtol=1e-6)

    def test_pairwise_search_family_cpu_bucketize_sides_match_numpy(self):
        values = np.array([0.0, 1.0, 3.0, 5.0], dtype="float32")
        boundaries = np.array([1.0, 3.0, 4.0], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            actual_left = torch.bucketize(
                torch.array(values), torch.array(boundaries), right=False)
            actual_right = torch.bucketize(
                torch.array(values), torch.array(boundaries), right=True,
                out_int32=True)
        np.testing.assert_array_equal(
            actual_left.numpy(), np.searchsorted(boundaries, values, side="left"))
        np.testing.assert_array_equal(
            actual_right.numpy(), np.searchsorted(boundaries, values, side="right"))
        self.assertEqual(str(actual_left.dtype), "int64")
        self.assertEqual(str(actual_right.dtype), "int32")

    def test_nan_reduction_family_is_stable_module_level_objects(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        for name in ("nansum", "nanmean"):
            with self.subTest(name=name):
                implementation = getattr(numerical, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__module__, numerical.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_nan_reduction_family_fidelity_is_queryable(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("nansum", "nanmean"):
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(numerical, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)

    def test_nan_reduction_family_cpu_full_and_dim_keepdim_matches_numpy(self):
        values = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            actual_sum = torch.nansum(torch.array(values)).numpy()
            actual_mean = torch.nanmean(torch.array(values)).numpy()
            sum_dim = torch.nansum(
                torch.array(values), dim=0, keepdim=True).numpy()
            mean_dim = torch.nanmean(
                torch.array(values), dim=1, keepdim=False).numpy()
        np.testing.assert_allclose(actual_sum, np.nansum(values), rtol=1e-6)
        np.testing.assert_allclose(actual_mean, np.nanmean(values), rtol=1e-6)
        np.testing.assert_allclose(
            sum_dim, np.nansum(values, axis=0, keepdims=True), rtol=1e-6)
        np.testing.assert_allclose(
            mean_dim, np.nanmean(values, axis=1), rtol=1e-6)

    def test_nan_reduction_family_var_methods_keep_nan_count(self):
        values = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            tensor = torch.array(values)
            actual_sum = tensor.nansum(dim=0).numpy()
            actual_mean = tensor.nanmean(dim=1, keepdim=True).numpy()
        np.testing.assert_allclose(actual_sum, np.nansum(values, axis=0), rtol=1e-6)
        np.testing.assert_allclose(
            actual_mean, np.nanmean(values, axis=1, keepdims=True), rtol=1e-6)

    def test_aminmax_is_a_stable_module_level_object(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        self.assertTrue(callable(numerical.aminmax))
        self.assertIs(torch.aminmax, numerical.aminmax)
        self.assertEqual(numerical.aminmax.__module__, numerical.__name__)
        self.assertEqual(numerical.aminmax.__name__, "aminmax")

    def test_aminmax_fidelity_is_queryable_and_conservative(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch.aminmax")
        self.assertIs(record.implementation, numerical.aminmax)
        self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
        self.assertIn("device", record.detail)
        self.assertIn("out", record.detail)

    def test_aminmax_cpu_full_dim_keepdim_and_var_match_numpy(self):
        values = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            tensor = torch.array(values)
            full = torch.aminmax(tensor)
            dim = torch.aminmax(tensor, dim=1, keepdim=True)
            method = tensor.aminmax(dim=0)
        self.assertEqual(tuple(full.min.shape), ())
        self.assertEqual(tuple(full.max.shape), ())
        np.testing.assert_array_equal(full.min.numpy(), np.asarray(values.min()))
        np.testing.assert_array_equal(full.max.numpy(), np.asarray(values.max()))
        np.testing.assert_array_equal(
            dim.min.numpy(), np.min(values, axis=1, keepdims=True))
        np.testing.assert_array_equal(
            dim.max.numpy(), np.max(values, axis=1, keepdims=True))
        np.testing.assert_array_equal(method.min.numpy(), np.min(values, axis=0))
        np.testing.assert_array_equal(method.max.numpy(), np.max(values, axis=0))

    def test_pdist_is_a_stable_module_level_object(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        self.assertTrue(callable(numerical.pdist))
        self.assertIs(torch.pdist, numerical.pdist)
        self.assertEqual(numerical.pdist.__module__, numerical.__name__)
        self.assertEqual(numerical.pdist.__name__, "pdist")

    def test_pdist_fidelity_is_queryable_and_conservative(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch.pdist")
        self.assertIs(record.implementation, numerical.pdist)
        self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
        self.assertIn("device", record.detail)
        self.assertIn("out", record.detail)

    def test_pdist_cpu_p1_p2_shape_and_var_method_match_numpy(self):
        values = np.array(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 1.0], [1.0, 5.0]],
            dtype="float32")
        expected_p1 = np.array([
            np.abs(values[i] - values[j]).sum()
            for i in range(len(values)) for j in range(i + 1, len(values))])
        expected_p2 = np.array([
            np.linalg.norm(values[i] - values[j])
            for i in range(len(values)) for j in range(i + 1, len(values))])
        with torch.flag_scope(use_cuda=0):
            tensor = torch.array(values)
            actual_p1 = torch.pdist(tensor, p=1)
            actual_p2 = torch.pdist(tensor, p=2)
            actual_method = tensor.pdist(p=2)
        self.assertEqual(tuple(actual_p1.shape), (6,))
        np.testing.assert_allclose(actual_p1.numpy(), expected_p1, rtol=1e-6)
        np.testing.assert_allclose(actual_p2.numpy(), expected_p2, rtol=1e-6)
        np.testing.assert_allclose(actual_method.numpy(), expected_p2, rtol=1e-6)

    def test_logcumsumexp_is_a_stable_module_level_object(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        self.assertTrue(callable(numerical.logcumsumexp))
        self.assertIs(torch.logcumsumexp, numerical.logcumsumexp)
        self.assertEqual(
            numerical.logcumsumexp.__module__, numerical.__name__)
        self.assertEqual(numerical.logcumsumexp.__name__, "logcumsumexp")

    def test_logcumsumexp_fidelity_is_queryable_and_conservative(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch.logcumsumexp")
        self.assertIs(record.implementation, numerical.logcumsumexp)
        self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
        self.assertIn("device", record.detail)
        self.assertIn("out", record.detail)

    def test_logcumsumexp_cpu_1d_2d_dims_and_var_method_match_numpy(self):
        values_1d = np.array([-1.0, 0.5, 2.0], dtype="float32")
        values_2d = np.array(
            [[-1.0, 0.5, 2.0], [1.5, -0.5, 3.0]], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            one_d = torch.logcumsumexp(torch.array(values_1d), 0).numpy()
            two_d = torch.logcumsumexp(torch.array(values_2d), 1).numpy()
            method = torch.array(values_2d).logcumsumexp(0).numpy()
        np.testing.assert_allclose(
            one_d, np.log(np.cumsum(np.exp(values_1d))), rtol=1e-5)
        np.testing.assert_allclose(
            two_d, np.log(np.cumsum(np.exp(values_2d), axis=1)), rtol=1e-5)
        np.testing.assert_allclose(
            method, np.log(np.cumsum(np.exp(values_2d), axis=0)), rtol=1e-5)

    def test_quantile_is_a_stable_module_level_object(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        self.assertTrue(callable(numerical.quantile))
        self.assertIs(torch.quantile, numerical.quantile)
        self.assertEqual(numerical.quantile.__module__, numerical.__name__)
        self.assertEqual(numerical.quantile.__name__, "quantile")

    def test_quantile_fidelity_is_queryable_and_cpu_only(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch.quantile")
        self.assertIs(record.implementation, numerical.quantile)
        self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
        self.assertIn("NumPy CPU", record.detail)
        self.assertIn("device", record.detail)

    def test_quantile_cpu_q_values_dim_keepdim_match_numpy(self):
        values = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            tensor = torch.array(values)
            actual = [torch.quantile(tensor, q).numpy() for q in (0.0, 0.5, 1.0)]
            dim = torch.quantile(tensor, 0.5, dim=1, keepdim=True).numpy()
            dim_no_keep = torch.quantile(tensor, 0.5, dim=0).numpy()
            tensor_q = torch.quantile(tensor, torch.array(0.5)).numpy()
        for got, q in zip(actual, (0.0, 0.5, 1.0)):
            np.testing.assert_allclose(got, np.quantile(values, q), rtol=1e-6)
        np.testing.assert_allclose(
            dim, np.quantile(values, 0.5, axis=1, keepdims=True), rtol=1e-6)
        np.testing.assert_allclose(
            dim_no_keep, np.quantile(values, 0.5, axis=0), rtol=1e-6)
        np.testing.assert_allclose(tensor_q, np.quantile(values, 0.5), rtol=1e-6)
        self.assertEqual(str(tensor_q.dtype), "float32")

    def test_nanquantile_is_a_stable_module_level_object(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        self.assertTrue(callable(numerical.nanquantile))
        self.assertIs(torch.nanquantile, numerical.nanquantile)
        self.assertEqual(numerical.nanquantile.__module__, numerical.__name__)
        self.assertEqual(numerical.nanquantile.__name__, "nanquantile")

    def test_nanquantile_fidelity_is_queryable_and_cpu_only(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch.nanquantile")
        self.assertIs(record.implementation, numerical.nanquantile)
        self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
        self.assertIn("NumPy CPU", record.detail)
        self.assertIn("device", record.detail)

    def test_nanquantile_cpu_nan_q_values_dim_keepdim_match_numpy(self):
        values = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 6.0]], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            tensor = torch.array(values)
            actual = [torch.nanquantile(tensor, q).numpy()
                      for q in (0.0, 0.5, 1.0)]
            dim = torch.nanquantile(tensor, 0.5, dim=1, keepdim=True).numpy()
            dim_no_keep = torch.nanquantile(tensor, 0.5, dim=0).numpy()
            tensor_q = torch.nanquantile(tensor, torch.array(0.5)).numpy()
        for got, q in zip(actual, (0.0, 0.5, 1.0)):
            np.testing.assert_allclose(got, np.nanquantile(values, q), rtol=1e-6)
        np.testing.assert_allclose(
            dim, np.nanquantile(values, 0.5, axis=1, keepdims=True), rtol=1e-6)
        np.testing.assert_allclose(
            dim_no_keep, np.nanquantile(values, 0.5, axis=0), rtol=1e-6)
        np.testing.assert_allclose(tensor_q, np.nanquantile(values, 0.5), rtol=1e-6)
        self.assertEqual(str(tensor_q.dtype), "float32")

    def test_std_mean_family_is_stable_module_level_objects(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        for name in ("std_mean", "var_mean"):
            with self.subTest(name=name):
                implementation = getattr(numerical, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertEqual(implementation.__module__, numerical.__name__)
                self.assertEqual(implementation.__name__, name)

    def test_std_mean_family_fidelity_records_current_limitations(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("std_mean", "var_mean"):
            with self.subTest(name=name):
                record = fidelity.fidelity_of("torch." + name)
                self.assertIs(record.implementation, getattr(numerical, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("correction", record.detail)
                self.assertIn("keepdim", record.detail)

    def test_std_mean_family_cpu_values_and_tuple_shapes_match_numpy(self):
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
        with torch.flag_scope(use_cuda=0):
            tensor = torch.array(values)
            std_full, mean_full = torch.std_mean(tensor)
            var_dim, mean_dim = torch.var_mean(tensor, dim=1)
            std_dim, mean_keep = torch.std_mean(
                tensor, dim=1, keepdim=True)
        self.assertEqual(tuple(std_full.shape), ())
        self.assertEqual(tuple(mean_full.shape), ())
        np.testing.assert_allclose(
            std_full.numpy(), np.std(values, ddof=1), rtol=1e-6)
        np.testing.assert_allclose(
            mean_full.numpy(), np.mean(values), rtol=1e-6)
        np.testing.assert_allclose(
            var_dim.numpy(), np.var(values, axis=1, ddof=1), rtol=1e-6)
        np.testing.assert_allclose(
            mean_dim.numpy(), np.mean(values, axis=1), rtol=1e-6)
        np.testing.assert_allclose(
            std_dim.numpy(), np.std(values, axis=1, ddof=1), rtol=1e-6)
        np.testing.assert_allclose(
            mean_keep.numpy(), np.mean(values, axis=1, keepdims=True), rtol=1e-6)

    def test_mv_is_a_stable_module_level_object(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        self.assertTrue(callable(numerical.mv))
        self.assertIs(torch.mv, numerical.mv)
        self.assertEqual(numerical.mv.__module__, numerical.__name__)
        self.assertEqual(numerical.mv.__name__, "mv")

    def test_mv_fidelity_is_queryable_and_conservative(self):
        numerical = importlib.import_module(
            "jittor.compat.torch.installers.numerical")
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch.mv")
        self.assertIs(record.implementation, numerical.mv)
        self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
        self.assertIn("out", record.detail)
        self.assertIn("device", record.detail)

    def test_mv_cpu_value_out_identity_and_var_delegate_match_numpy(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        vector = np.array([2.0, -1.0], dtype="float32")
        expected = np.matmul(matrix, vector)
        with torch.flag_scope(use_cuda=0):
            matrix_tensor = torch.array(matrix)
            vector_tensor = torch.array(vector)
            actual = torch.mv(matrix_tensor, vector_tensor)
            out = torch.zeros(2)
            returned = torch.mv(matrix_tensor, vector_tensor, out=out)
            method = matrix_tensor.mv(vector_tensor)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6)
        self.assertIs(returned, out)
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)
        np.testing.assert_allclose(method.numpy(), expected, rtol=1e-6)

    def test_mv_invalid_rank_and_size_raise(self):
        with torch.flag_scope(use_cuda=0):
            with self.assertRaisesRegex(RuntimeError, "expected a 2-D"):
                torch.mv(torch.ones((1, 2, 3)), torch.ones(3))
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                torch.mv(torch.ones((2, 3)), torch.ones(2))


if __name__ == "__main__":
    unittest.main()

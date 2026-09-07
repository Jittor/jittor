"""Kaiming keeps Torch's random draw and retained-view assignment semantics."""

import math

import jittor as jt
import numpy as np

from _helpers import common as cu
from _helpers.device_types import dtypes, instantiate_device_type_tests


class TestTorchInitPackage(cu.JittorTestCase):
    @dtypes("float32", "float64")
    def test_kaiming_uses_torch_uniform_draw(self, device, dtype):
        actual = jt.zeros((3, 4), dtype)
        expected = jt.zeros((3, 4), dtype)
        actual.sync()
        expected.sync()
        bound = math.sqrt(3.0) * jt.init.calculate_std(
            actual, "fan_in", "leaky_relu", 0,
        )
        jt.set_global_seed(17)
        self.assertIs(jt.nn.init.kaiming_uniform_(actual), actual)
        values = actual.numpy()
        jt.set_global_seed(17)
        jt.nn.init.uniform_(expected, -bound, bound)
        np.testing.assert_array_equal(values, expected.numpy())
        self.assertEqual(str(actual.dtype), str(expected.dtype))

    @dtypes("float32", "float64")
    def test_kaiming_writes_back_to_retained_view(self, device, dtype):
        parent = jt.zeros((3, 8), dtype).start_grad()
        expected = jt.zeros((3, 4), dtype)
        parent.sync()
        expected.sync()
        view = parent[:, 1:5]
        view.sync()
        bound = math.sqrt(3.0) * jt.init.calculate_std(
            view, "fan_out", "leaky_relu", 0.2,
        )
        jt.set_global_seed(17)
        self.assertIs(jt.nn.init.kaiming_uniform_(view, a=0.2, mode="fan_out"), view)
        values = parent.numpy()
        jt.set_global_seed(17)
        jt.nn.init.uniform_(expected, -bound, bound)
        np.testing.assert_array_equal(values[:, 1:5], expected.numpy())
        np.testing.assert_array_equal(values[:, :1], np.zeros((3, 1)))
        np.testing.assert_array_equal(values[:, 5:], np.zeros((3, 3)))
        self.assertFalse(view.is_stop_grad())

    def test_kaiming_generator_is_not_silently_ignored(self, device):
        weight = jt.zeros((3, 4))
        with self.assertRaisesRegex(NotImplementedError, "generator"):
            jt.nn.init.kaiming_uniform_(weight, generator=object())


instantiate_device_type_tests(TestTorchInitPackage, globals())

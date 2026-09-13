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

    def test_kaiming_generator_advances_without_changing_global_rng(self, device):
        if device != "cpu":
            self.skipTest("explicit CUDA Generator is not implemented")
        import torch
        generator = torch.Generator().manual_seed(23)
        replay_generator = torch.Generator().manual_seed(23)
        first, second, replay = (torch.zeros((3, 4)) for _ in range(3))
        global_before = jt.get_seed()
        torch.nn.init.kaiming_uniform_(first, generator=generator)
        state = generator.get_state()
        torch.nn.init.kaiming_uniform_(second, generator=generator)
        torch.nn.init.kaiming_uniform_(replay, generator=replay_generator)
        oracle = np.array([
            [-0.1757310033, -0.5171789527, -0.1900150031, -0.3501487374],
            [1.1211271286, -0.9554151893, -0.5062193871, 1.0301240683],
            [0.2146068215, -0.9065476060, 0.4234038293, -0.9730401039],
        ], dtype=np.float32)
        np.testing.assert_allclose(first.numpy(), oracle, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(first.numpy(), replay.numpy())
        self.assertFalse(np.array_equal(first.numpy(), second.numpy()))
        restored = torch.Generator().set_state(state)
        restored_value = torch.zeros((3, 4))
        torch.nn.init.kaiming_uniform_(restored_value, generator=restored)
        np.testing.assert_array_equal(second.numpy(), restored_value.numpy())
        self.assertEqual(jt.get_seed(), global_before)

    def test_generator_uniform_split_draw_matches_single_draw(self, device):
        if device != "cpu":
            self.skipTest("explicit CUDA Generator is not implemented")
        import torch
        split_generator = torch.Generator().manual_seed(101)
        whole_generator = torch.Generator().manual_seed(101)
        left, right, whole = (torch.zeros((2, 3)) for _ in range(3))
        torch.nn.init.uniform_(left, -0.25, 0.75, generator=split_generator)
        torch.nn.init.uniform_(right, -0.25, 0.75, generator=split_generator)
        whole = torch.zeros((12,))
        torch.nn.init.uniform_(whole, -0.25, 0.75, generator=whole_generator)
        joined = np.concatenate((left.numpy().reshape(-1), right.numpy().reshape(-1)))
        np.testing.assert_array_equal(joined, whole.numpy())

    def test_kaiming_generator_matches_pytorch_low_precision(self, device):
        if device != "cpu":
            self.skipTest("explicit CUDA Generator is not implemented")
        import torch
        expected = {
            torch.float16: [-0.2032470703125, -0.278076171875, 0.376220703125, 0.11480712890625],
            torch.bfloat16: [0.2109375, -0.388671875, -0.052734375, 0.306640625],
        }
        for dtype, oracle in expected.items():
            value = torch.empty((2, 1, 16), dtype=dtype)
            torch.nn.init.kaiming_uniform_(value, generator=torch.Generator().manual_seed(0))
            np.testing.assert_array_equal(value.float().reshape(-1)[:4].numpy(), oracle)


instantiate_device_type_tests(TestTorchInitPackage, globals())

"""Real TensorDict behavior through the deployed Jittor Torch shim."""

import importlib.util
import unittest

import numpy as np

import jittor as torch


_HAS_TENSORDICT = importlib.util.find_spec("tensordict") is not None


@unittest.skipUnless(_HAS_TENSORDICT, "tensordict is not installed")
@unittest.skipUnless(torch.compiler.has_cuda, "CUDA is required")
class TestTensorDictCompat(unittest.TestCase):
    def test_cpu_conversion_uses_device_objects(self):
        from tensordict import TensorDict

        parsed_device = torch._C._nn._parse_to("cpu")[0]
        self.assertIsInstance(parsed_device, torch.device)
        self.assertEqual(parsed_device.type, "cpu")
        self.assertIsNone(parsed_device.index)

        with torch.flag_scope(use_cuda=1):
            tensordict = TensorDict(
                {"value": torch.ones((2,), device="cuda")}, batch_size=[2]
            )
            cpu_tensordict = tensordict.cpu()

        self.assertEqual(cpu_tensordict.device.type, "cpu")
        self.assertFalse(cpu_tensordict["value"].is_cuda)

    def test_cuda_construct_update_and_lazy_stack(self):
        from tensordict import TensorDict, lazy_stack

        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        with torch.flag_scope(use_cuda=1):
            value = torch.tensor(expected, device="cuda")
            tensordict = TensorDict({"value": value}, batch_size=[3])
            tensordict["double"] = tensordict["value"] * 2
            stacked = lazy_stack([tensordict, tensordict], dim=0)
            self.assertTrue(tensordict["double"].is_cuda)
            self.assertEqual(tuple(stacked.batch_size), (2, 3))
            got_value = tensordict["value"].numpy()
            got_double = stacked[1]["double"].numpy()

        np.testing.assert_array_equal(got_value, expected)
        np.testing.assert_array_equal(got_double, expected * 2)

    def test_jittor_tensor_indices_patch_real_tensordict(self):
        from tensordict import TensorDict, lazy_stack
        from tensordict.base import TensorDictBase

        expected = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        with torch.flag_scope(use_cuda=1):
            tensordict = TensorDict(
                {"value": torch.tensor(expected, device="cuda")}, batch_size=[3]
            )
            integer_index = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
            boolean_index = torch.tensor(
                [True, False, True], dtype=torch.bool, device="cuda"
            )
            selected = tensordict[integer_index]["value"].numpy()
            masked = tensordict[boolean_index]["value"].numpy()
            stacked = lazy_stack([tensordict, tensordict], dim=0)
            lazy_index = torch.tensor([1, 0], dtype=torch.int64, device="cuda")
            lazy_selected = stacked[lazy_index][0]["value"].numpy()

        self.assertTrue(getattr(TensorDictBase, "_jittor_index_compat", False))
        np.testing.assert_array_equal(selected, expected[[2, 0]])
        np.testing.assert_array_equal(masked, expected[[0, 2]])
        np.testing.assert_array_equal(lazy_selected, expected)


if __name__ == "__main__":
    unittest.main()

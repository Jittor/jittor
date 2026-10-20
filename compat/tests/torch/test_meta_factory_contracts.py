"""Shape-only meta construction and explicit real materialization contracts.

Jittor's accepted representation remains storage-backed; these tests do not
claim no-storage meta allocation or memory-capacity offload.
"""

import os
import unittest

import numpy as np
import torch


DEVICE = os.environ.get("RELEASE_META_DEVICE", "cpu")


class TestMetaFactoryContracts(unittest.TestCase):
    def test_context_meta_data_constructors_override_real_input_without_relabeling_it(self):
        source = torch.tensor([1.0, 2.0], device=DEVICE, requires_grad=True)
        with torch.device("meta"):
            copied = torch.tensor(source)
            converted = torch.as_tensor(source)
            self.assertTrue(copied.is_meta and converted.is_meta)
            self.assertIsNot(converted, source)
            self.assertFalse(copied.requires_grad)
            self.assertTrue(converted.requires_grad)
            self.assertIs(torch.as_tensor(source, device=DEVICE), source)
            self.assertEqual(torch.empty_like(source).device.type, DEVICE)
            self.assertEqual(source.new_empty((2,)).device.type, DEVICE)
        self.assertFalse(source.is_meta)
        np.testing.assert_array_equal(source.detach().cpu().numpy(), [1.0, 2.0])

    def test_meta_input_inherits_meta_but_real_data_constructor_request_fails_closed(self):
        source = torch.empty(2, device="meta")
        with torch.device("meta"):
            self.assertTrue(torch.tensor(source).is_meta)
            self.assertIs(torch.as_tensor(source), source)
        for constructor in (torch.tensor, torch.as_tensor):
            with self.subTest(constructor=constructor.__name__), self.assertRaisesRegex(
                    NotImplementedError, "Cannot copy out of meta"):
                constructor(source, device=DEVICE)
            with torch.device(DEVICE), self.assertRaisesRegex(
                    NotImplementedError, "Cannot copy out of meta"):
                constructor(source)
        self.assertTrue(source.is_meta)

    def test_explicit_empty_metadata(self):
        value = torch.empty((2, 3), dtype=torch.float64, device="meta", requires_grad=True)
        self.assertTrue(value.is_meta)
        self.assertEqual(value.device.type, "meta")
        self.assertEqual(tuple(value.shape), (2, 3))
        self.assertEqual(value.dtype, torch.float64)
        self.assertTrue(value.requires_grad)

    def test_context_is_per_constructed_tensor_and_nests_with_real(self):
        previous = torch.ones(2, device=DEVICE)
        with torch.device("meta"):
            contextual = torch.tensor([])
            placeholder = torch.zeros(2)
            explicit = torch.zeros(2, device=DEVICE)
            self.assertEqual(previous.device.type, DEVICE)
            self.assertFalse(explicit.is_meta)
            with torch.device(DEVICE):
                nested = torch.empty(2)
                self.assertFalse(nested.is_meta)
            self.assertTrue(torch.empty(2).is_meta)
        self.assertTrue(contextual.is_meta)
        self.assertTrue(placeholder.is_meta)
        self.assertFalse(torch.empty(2).is_meta)

    def test_like_parameter_clone_detach_and_dtype_preserve_meta(self):
        value = torch.empty((2, 3), device="meta")
        for output in (torch.empty_like(value), value.new_empty((3, 2)),
                       torch.nn.Parameter(value), value.clone(), value.detach(),
                       value.to(dtype=torch.float64)):
            self.assertTrue(output.is_meta)
        self.assertFalse(torch.empty_like(value).requires_grad)

    def test_explicit_real_empty_like_does_not_relabel_source(self):
        source = torch.empty((2, 3), device="meta")
        output = torch.empty_like(source, device=DEVICE)
        self.assertFalse(output.is_meta)
        self.assertEqual(output.device.type, DEVICE)
        self.assertTrue(source.is_meta)
        output.fill_(3)
        self.assertEqual(float(output.sum().item()), 18)

    def test_empty_like_gradient_request_is_independent_of_real_or_meta_input(self):
        for device in (DEVICE, "meta"):
            for source_requires_grad in (False, True):
                source = torch.empty((2, 3), device=device,
                                     requires_grad=source_requires_grad)
                for requested in (None, False, True):
                    with self.subTest(device=device, source_requires_grad=source_requires_grad,
                                      requested=requested):
                        kwargs = {} if requested is None else {"requires_grad": requested}
                        output = torch.empty_like(source, **kwargs)
                        self.assertEqual(bool(output.requires_grad), bool(requested))
                        self.assertEqual(bool(source.requires_grad), source_requires_grad)
                        self.assertEqual(output.device.type, torch.device(device).type)

    def test_as_tensor_explicit_meta_preserves_real_source_and_normal_alias(self):
        source = torch.tensor([1.0, 2.0], device=DEVICE, requires_grad=True)
        self.assertIs(torch.as_tensor(source), source)
        self.assertIs(torch.as_tensor(source, device=DEVICE), source)
        placeholder = torch.as_tensor(source, device="meta")
        self.assertIsNot(placeholder, source)
        self.assertTrue(placeholder.is_meta)
        self.assertTrue(placeholder.requires_grad)
        self.assertFalse(source.is_meta)
        self.assertEqual(source.device.type, DEVICE)
        np.testing.assert_array_equal(source.detach().cpu().numpy(), [1.0, 2.0])
        self.assertIs(torch.as_tensor(placeholder, device="meta"), placeholder)

    def test_module_context_and_assign_materialize_real_weights(self):
        with torch.device("meta"):
            module = torch.nn.Linear(3, 2)
        self.assertTrue(all(parameter.is_meta for parameter in module.parameters()))
        weights = {
            "weight": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=DEVICE),
            "bias": torch.tensor([0.5, -0.5], device=DEVICE),
        }
        module.load_state_dict(weights, assign=True)
        self.assertTrue(all(not parameter.is_meta for parameter in module.parameters()))
        self.assertTrue(all(parameter.device.type == DEVICE for parameter in module.parameters()))
        result = module(torch.ones((1, 3), device=DEVICE))
        np.testing.assert_array_equal(result.detach().cpu().numpy(), [[6.5, 14.5]])

    def test_module_to_empty_materializes_meta_metadata(self):
        with torch.device("meta"):
            module = torch.nn.Linear(3, 2)
        module.to_empty(device=DEVICE)
        self.assertTrue(all(not parameter.is_meta for parameter in module.parameters()))
        self.assertTrue(all(parameter.device.type == DEVICE for parameter in module.parameters()))


if __name__ == "__main__":
    unittest.main()

"""Numerical checks for the autograd-safe Qwen3.5 gated-delta adapter."""

import importlib.util
import unittest

try:
    import torch
except ImportError:  # The lightweight adapter unit-test environment has no torch.
    torch = None


@unittest.skipIf(torch is None, "official PyTorch is not installed")
class Qwen35FunctionalTest(unittest.TestCase):
    def test_chunk_forward_and_all_input_gradients(self):
        original_find_spec = importlib.util.find_spec
        self.addCleanup(setattr, importlib.util, "find_spec", original_find_spec)
        importlib.util.find_spec = lambda name, package=None: (
            None
            if name.split(".", 1)[0] in {"torch_npu", "torchvision"}
            else original_find_spec(name, package)
        )
        from jittor_adapters.transformers_qwen35 import _functional_chunk
        from transformers.models.qwen3_5 import modeling_qwen3_5

        reference = modeling_qwen3_5.torch_chunk_gated_delta_rule
        candidate = _functional_chunk(modeling_qwen3_5)
        for sequence_length in (8, 32, 65):
            with self.subTest(sequence_length=sequence_length):
                torch.manual_seed(20260928 + sequence_length)
                shapes = (
                    (1, sequence_length, 2, 4),
                    (1, sequence_length, 2, 4),
                    (1, sequence_length, 2, 3),
                    (1, sequence_length, 2),
                    (1, sequence_length, 2),
                )
                # Projection-sized activations keep the triangular recurrence
                # in its intended stable regime instead of manufacturing an
                # ill-conditioned closure from unit-variance keys.
                source = [
                    torch.randn(shape, dtype=torch.float32) * 0.25
                    for shape in shapes
                ]
                source[3] = -source[3].abs() * 0.1
                source[4] = source[4].sigmoid()
                reference_inputs = [value.detach().requires_grad_() for value in source]
                candidate_inputs = [value.detach().clone().requires_grad_() for value in source]

                expected_output, expected_state = reference(
                    *reference_inputs,
                    output_final_state=True,
                )
                actual_output, actual_state = candidate(
                    *candidate_inputs,
                    output_final_state=True,
                )
                expected_loss = expected_output.square().sum() + expected_state.square().sum()
                actual_loss = actual_output.square().sum() + actual_state.square().sum()
                expected_loss.backward()
                actual_loss.backward()

                torch.testing.assert_close(actual_output, expected_output, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-6)
                for actual, expected in zip(candidate_inputs, reference_inputs):
                    self.assertIsNotNone(actual.grad)
                    self.assertIsNotNone(expected.grad)
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-5, atol=5e-6)


if __name__ == "__main__":
    unittest.main()

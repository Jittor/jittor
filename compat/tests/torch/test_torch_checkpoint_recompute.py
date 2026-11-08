"""Functional tests for Jittor-backed Torch activation checkpointing."""

import unittest
import functools as _functools

import numpy as np
import jittor as jt
import torch
from torch.utils.checkpoint import checkpoint


class TestCheckpointRecompute(unittest.TestCase):
    def test_recompute_nested_outputs_captured_parameter_and_rng(self):
        calls = []
        random_values = []
        x = jt.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        x.start_grad()
        weight = jt.array(np.array([2.0], dtype=np.float32))
        weight.start_grad()

        def layer(value, scale=0.0):
            sample = jt.rand(value.shape, dtype=value.dtype)
            jt.sync_all()
            random_values.append(sample.numpy().tolist())
            calls.append(int(jt.flags.no_grad))
            hidden = value * weight + scale + sample
            return hidden, {"square": value * value}

        jt.set_seed(821)
        hidden, aux = checkpoint(layer, x, scale=2.0, use_reentrant=False)
        self.assertIsInstance(aux, dict)
        self.assertEqual(tuple(aux), ("square",))
        hidden.numpy()
        jt.rand((5,)).numpy()
        state_before_backward = jt.core.rng_state("cpu")
        gx, gweight = jt.grad(hidden.sum() + aux["square"].sum(), [x, weight])

        self.assertEqual(calls, [1, 0])
        self.assertEqual(random_values[0], random_values[1])
        self.assertEqual(state_before_backward, jt.core.rng_state("cpu"))
        np.testing.assert_allclose(gx.numpy(), [4.0, 6.0, 8.0], atol=1e-6)
        np.testing.assert_allclose(gweight.numpy(), [6.0], atol=1e-6)

    def test_no_grad_runs_function_without_recompute(self):
        calls = []
        value = jt.array(np.array([1.0, 2.0], dtype=np.float32))

        def layer(x):
            calls.append(int(jt.flags.no_grad))
            return x * 3

        with jt.no_grad():
            output = checkpoint(layer, value)
        np.testing.assert_allclose(output.numpy(), [3.0, 6.0])
        self.assertEqual(calls, [1])

    def test_bound_module_parameter_receives_checkpoint_gradient(self):
        calls = []

        class Layer(jt.Module):
            def __init__(self):
                super().__init__()
                self.weight = jt.array(np.array([3.0], dtype=np.float32))
                self.weight.start_grad()

            def execute(self, value):
                calls.append(int(jt.flags.no_grad))
                return value * self.weight

        layer = Layer()
        value = jt.array(np.array([2.0, 4.0], dtype=np.float32))
        value.start_grad()
        output = checkpoint(layer.execute, value)
        grad_value, grad_weight = jt.grad(
            output.sum(), [value, layer.weight])

        self.assertEqual(calls, [1, 0])
        np.testing.assert_allclose(grad_value.numpy(), [3.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(grad_weight.numpy(), [6.0], atol=1e-6)

    def test_checkpoint_partial_super_method_captures_module_parameters(self):
        calls = []

        class Layer(jt.Module):
            def __init__(self):
                super().__init__()
                self.weight = jt.array(np.array([3.0], dtype=np.float32))
                self.weight.start_grad()

            def execute(self, value):
                calls.append(int(jt.flags.no_grad))
                return value * self.weight

        layer = Layer()
        value = jt.array(np.array([2.0, 4.0], dtype=np.float32))
        value.start_grad()
        function = _functools.partial(super(Layer, layer).__call__)
        output = checkpoint(function, value, use_reentrant=False)
        grad_value, grad_weight = jt.grad(
            output.sum(), [value, layer.weight])

        self.assertEqual(calls, [1, 0])
        np.testing.assert_allclose(grad_value.numpy(), [3.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(grad_weight.numpy(), [6.0], atol=1e-6)

    def test_llama_decoder_layer_recomputes_with_input_and_weight_gradients(self):
        try:
            from transformers import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        except Exception as exc:
            self.skipTest("local Transformers Llama layer unavailable: %s" % exc)

        config = LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8,
            attention_dropout=0.0,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        layer = LlamaDecoderLayer(config, layer_idx=0)
        layer.train()
        layer.gradient_checkpointing = True

        calls = []
        checkpoint_outputs = []
        checkpoint_fn = _functools.partial(checkpoint, use_reentrant=False)

        def checkpoint_probe(function, *args, **kwargs):
            def traced_function(*inner_args, **inner_kwargs):
                calls.append(int(jt.flags.no_grad))
                result = function(*inner_args, **inner_kwargs)
                if isinstance(result, jt.Var):
                    checkpoint_outputs.append(result.numpy().copy())
                return result
            return checkpoint_fn(traced_function, *args, **kwargs)

        object.__setattr__(layer, "_gradient_checkpointing_func", checkpoint_probe)
        original_forward = layer.forward
        data = np.arange(16, dtype=np.float32).reshape(1, 2, 8) / 10.0
        value = torch.tensor(data, requires_grad=True)
        positions = torch.tensor([[0, 1]], dtype=torch.long)
        cosine = torch.ones((1, 2, 4), dtype=torch.float32)
        sine = torch.zeros((1, 2, 4), dtype=torch.float32)
        q_weight = layer.self_attn.q_proj.weight
        output = layer(
            value,
            position_ids=positions,
            position_embeddings=(cosine, sine),
            use_cache=False,
        )
        grad_value, grad_weight = jt.grad(
            output.square().sum(), [value, q_weight])

        self.assertEqual(calls, [1, 0])
        self.assertEqual(len(checkpoint_outputs), 2)
        np.testing.assert_allclose(
            checkpoint_outputs[0], checkpoint_outputs[1],
            rtol=0.0, atol=0.0)
        self.assertTrue(np.isfinite(grad_value.numpy()).all())
        self.assertTrue(np.isfinite(grad_weight.numpy()).all())
        self.assertGreater(float(np.max(np.abs(grad_value.numpy()))), 0.0)
        self.assertGreater(float(np.max(np.abs(grad_weight.numpy()))), 0.0)
        checkpoint_grad_value = grad_value.numpy().copy()
        checkpoint_grad_weight = grad_weight.numpy().copy()

        baseline_value = torch.tensor(data, requires_grad=True)
        baseline_output = original_forward(
            baseline_value,
            position_ids=positions,
            position_embeddings=(cosine, sine),
            use_cache=False,
        )
        baseline_grad_value, baseline_grad_weight = jt.grad(
            baseline_output.square().sum(), [baseline_value, q_weight])
        np.testing.assert_allclose(
            checkpoint_grad_value, baseline_grad_value.numpy(),
            rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(
            checkpoint_grad_weight, baseline_grad_weight.numpy(),
            rtol=2e-4, atol=2e-5)

    def test_transformers_gradient_checkpointing_layer_protocol(self):
        try:
            from transformers.modeling_layers import GradientCheckpointingLayer
        except Exception as exc:
            self.skipTest("local Transformers checkpoint layer unavailable: %s" % exc)

        calls = []

        class Layer(GradientCheckpointingLayer):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.tensor([3.0], dtype=torch.float32))

            def forward(self, value):
                calls.append(int(jt.flags.no_grad))
                return value * self.weight

        layer = Layer()
        layer.train()
        layer.gradient_checkpointing = True
        object.__setattr__(
            layer, "_gradient_checkpointing_func",
            _functools.partial(checkpoint, use_reentrant=False))
        value = torch.tensor([2.0, 4.0], requires_grad=True)
        output = layer(value)
        grad_value, grad_weight = jt.grad(
            output.sum(), [value, layer.weight])

        self.assertTrue(output.requires_grad)
        self.assertEqual(calls, [1, 0])
        np.testing.assert_allclose(grad_value.numpy(), [3.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(grad_weight.numpy(), [6.0], atol=1e-6)

if __name__ == "__main__":
    unittest.main()

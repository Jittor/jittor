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

        jt.set_cpu_seed(821)
        hidden, aux = checkpoint(layer, x, scale=2.0, use_reentrant=False)
        self.assertIsInstance(aux, dict)
        self.assertEqual(tuple(aux), ("square",))
        hidden.numpy()
        jt.rand((5,)).numpy()
        state_before_backward = jt.get_cpu_rng_state()

        gx, gweight = jt.grad(hidden.sum() + aux["square"].sum(), [x, weight])

        self.assertEqual(calls, [1, 0])
        self.assertEqual(random_values[0], random_values[1])
        self.assertEqual(state_before_backward, jt.get_cpu_rng_state())
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

"""Weight-normalization behavior and independent Torch parity."""

from __future__ import print_function

import warnings
import unittest

import jittor as jt
import numpy as np

from _helpers.torch_runtime import import_torch_modules, modules_available
from jittor.nn.utils.weight_norm import remove_weight_norm, weight_norm


class JittorModule(jt.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.linear = jt.array(weight)

    def execute(self, value):
        return jt.matmul(self.linear, value)


class TestWeightNorm(unittest.TestCase):
    def setUp(self):
        self.weight = (
            np.arange(1, 13, dtype=np.float32).reshape(3, 4) / np.float32(7.0)
        )
        self.value = (
            np.arange(1, 21, dtype=np.float32).reshape(4, 5) / np.float32(11.0)
        )

    def test_parameter_shapes_match_legacy_torch_contract(self):
        expected = {
            -2: (3, 1),
            # Jittor has no native 0-D Var; Torch's scalar gain is represented
            # by the established one-element shape.
            -1: (1,),
            0: (3, 1),
            1: (1, 4),
            None: (1,),
        }
        for dim, gain_shape in expected.items():
            with self.subTest(dim=dim):
                module = JittorModule(self.weight)
                self.assertIs(weight_norm(module, "linear", dim), module)
                self.assertEqual(tuple(module.linear_g.shape), gain_shape)
                self.assertEqual(tuple(module.linear_v.shape), self.weight.shape)
                np.testing.assert_allclose(module.linear.numpy(), self.weight, rtol=1e-6)
                self.assertEqual(
                    sorted(name for name, _ in module.named_parameters()),
                    ["linear_g", "linear_v"],
                )

    def test_forward_backward_recompute_and_remove(self):
        module = JittorModule(self.weight)
        weight_norm(module, "linear", -1)
        value = jt.array(self.value)

        output = module(value)
        np.testing.assert_allclose(
            output.numpy(), self.weight @ self.value, rtol=1e-5, atol=1e-6
        )
        gradient = jt.grad(output.sum(), value).numpy()
        expected_gradient = np.repeat(
            self.weight.sum(axis=0).reshape(4, 1),
            self.value.shape[1],
            axis=1,
        )
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=1e-5, atol=1e-6
        )

        module.linear_g.assign(module.linear_g * 2)
        doubled = module(jt.array(self.value)).numpy()
        np.testing.assert_allclose(
            doubled, 2 * (self.weight @ self.value), rtol=1e-5, atol=1e-6
        )

        self.assertIs(remove_weight_norm(module, "linear"), module)
        self.assertFalse(hasattr(module, "linear_g"))
        self.assertFalse(hasattr(module, "linear_v"))
        self.assertEqual(
            [name for name, _ in module.named_parameters()],
            ["linear"],
        )
        np.testing.assert_allclose(
            module(jt.array(self.value)).numpy(),
            doubled,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_duplicate_registration_and_missing_remove_fail_loudly(self):
        module = JittorModule(self.weight)
        weight_norm(module, "linear", 0)
        with self.assertRaisesRegex(RuntimeError, "two weight_norm hooks"):
            weight_norm(module, "linear", 0)
        remove_weight_norm(module, "linear")
        with self.assertRaisesRegex(ValueError, "weight_norm of 'linear' not found"):
            remove_weight_norm(module, "linear")

    def test_existing_pre_forward_hook_is_preserved(self):
        module = JittorModule(self.weight)
        calls = []

        def existing(owner, args):
            calls.append((owner, len(args)))

        module.register_pre_forward_hook(existing)
        weight_norm(module, "linear", 0)
        module(jt.array(self.value)).sync()
        remove_weight_norm(module, "linear")
        module(jt.array(self.value)).sync()
        self.assertEqual(calls, [(module, 1), (module, 1)])

    @unittest.skipUnless(
        modules_available("torch"),
        "independent binary Torch was not preloaded",
    )
    def test_independent_torch_forward_backward_parity(self):
        torch, torch_nn = import_torch_modules("torch", "torch.nn")

        class TorchModule(torch_nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.linear = torch_nn.Parameter(torch.from_numpy(weight.copy()))

            def forward(self, value):
                return torch.matmul(self.linear, value)

        for dim in (-1, 0, 1):
            with self.subTest(dim=dim):
                jittor_module = JittorModule(self.weight)
                torch_module = TorchModule(self.weight)
                weight_norm(jittor_module, "linear", dim)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    torch_nn.utils.weight_norm(torch_module, "linear", dim)

                jittor_value = jt.array(self.value)
                torch_value = torch.from_numpy(self.value.copy()).requires_grad_(True)
                jittor_output = jittor_module(jittor_value)
                torch_output = torch_module(torch_value)
                np.testing.assert_allclose(
                    jittor_output.numpy(),
                    torch_output.detach().numpy(),
                    rtol=1e-5,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    jt.grad(jittor_output.sum(), jittor_value).numpy(),
                    torch.autograd.grad(torch_output.sum(), torch_value)[0].numpy(),
                    rtol=1e-5,
                    atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()

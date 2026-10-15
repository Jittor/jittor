"""A weighted RMS fast path must preserve optional/partial norm semantics."""
import os
import unittest

import numpy as np
import torch


class OptionalWeightRMSNorm(torch.nn.Module):
    def __init__(self, device, *, has_weight=True, pass_weight=True,
                 variance_size_override=None, unused_weight_on_cpu=False):
        super().__init__()
        self.variance_epsilon = 0.1
        self.has_weight = has_weight
        self.pass_weight = pass_weight
        self.variance_size_override = variance_size_override
        weight_device = 'cpu' if unused_weight_on_cpu else device
        self.weight = torch.nn.Parameter(
            torch.tensor([2., 3., 4., 5.], device=weight_device))
        self.forward_calls = 0

    def forward(self, value):
        self.forward_calls += 1
        width = getattr(self, 'variance_size_override', None) or value.shape[-1]
        variance = value[..., :width].pow(2).mean(dim=-1, keepdim=True)
        normalized = value * torch.rsqrt(variance + self.variance_epsilon)
        if getattr(self, 'has_weight', True) and getattr(self, 'pass_weight', True):
            normalized = normalized * self.weight
        return normalized


class TestRMSNormOptionalWeight(unittest.TestCase):
    def _check(self, *, has_weight=True, pass_weight=True,
               variance_size_override=None, unused_weight_on_cpu=False, omit_options=False):
        device = os.environ.get('JITTOR_TEST_DEVICE', 'cpu')
        model = OptionalWeightRMSNorm(
            device, has_weight=has_weight, pass_weight=pass_weight,
            variance_size_override=variance_size_override,
            unused_weight_on_cpu=unused_weight_on_cpu)
        if omit_options:
            for name in ('has_weight', 'pass_weight', 'variance_size_override'):
                delattr(model, name)
        source = np.array([[1., -2., .5, 3.], [-1., 2.5, -.25, 1.5]], dtype='float32')
        cotangent = np.array([[.2, -.7, 1.1, .4], [.8, .3, -.5, 1.2]], dtype='float32')
        value = torch.tensor(source, device=device, requires_grad=True)
        output = model(value)
        (output * torch.tensor(cotangent, device=device)).sum().backward()

        # Independent analytic first derivatives, evaluated in NumPy float64.
        x = source.astype('float64')
        width = variance_size_override or x.shape[-1]
        inverse = 1 / np.sqrt(np.mean(x[:, :width] ** 2, axis=-1, keepdims=True) + .1)
        weighted = has_weight and pass_weight
        weights = np.array([2., 3., 4., 5.]) if weighted else np.ones(4)
        expected = x * inverse * weights
        scaled_cotangent = cotangent * weights
        expected_gradient = scaled_cotangent * inverse
        expected_gradient[:, :width] -= (
            x[:, :width] * (scaled_cotangent * x).sum(axis=-1, keepdims=True)
            * inverse ** 3 / width)
        self.assertTrue(str(output.device).startswith(device))
        self.assertIsNotNone(value.grad)
        self.assertTrue(str(value.grad.device).startswith(device))
        np.testing.assert_allclose(output.detach().cpu().numpy(), expected, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(value.grad.detach().cpu().numpy(), expected_gradient,
                                   rtol=2e-5, atol=2e-6)
        if weighted:
            self.assertIsNotNone(model.weight.grad)
            self.assertTrue(str(model.weight.grad.device).startswith(device))
            np.testing.assert_allclose(
                model.weight.grad.detach().cpu().numpy(),
                (cotangent * x * inverse).sum(axis=0), rtol=2e-5, atol=2e-6)
        else:
            self.assertIsNone(model.weight.grad)
        if not weighted or variance_size_override is not None:
            self.assertEqual(model.forward_calls, 1)
        # Repeated calls use the native dispatch shortcut after initial leaf
        # publication. Its inference fast path must honor the same contract.
        for _ in range(2):
            with torch.no_grad():
                inference = model(torch.tensor(source, device=device))
            self.assertTrue(str(inference.device).startswith(device))
            np.testing.assert_allclose(inference.detach().cpu().numpy(), expected,
                                       rtol=2e-5, atol=2e-6)

    def test_has_weight_false_does_not_consume_nonunit_weight(self):
        self._check(has_weight=False)

    def test_pass_weight_false_does_not_consume_host_weight(self):
        self._check(pass_weight=False, unused_weight_on_cpu=True)

    def test_partial_variance_preserves_forward_and_both_gradients(self):
        self._check(variance_size_override=2)

    def test_missing_optional_flags_preserve_standard_weighted_semantics(self):
        self._check(omit_options=True)

    def test_standard_weighted_neighbor_preserves_both_gradients(self):
        self._check()


if __name__ == '__main__':
    unittest.main()

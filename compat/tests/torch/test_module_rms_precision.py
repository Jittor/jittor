"""Optional RMS module fusion must retain the forward owner's mixed dtype."""
from contextlib import nullcontext
import unittest

import numpy as np
import torch

from jittor._runtime import forbid_backend_fallbacks
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


def _portable(value, weight, epsilon):
    original_dtype = value.dtype
    wide = value.float()
    normalized = wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + epsilon)
    return weight * normalized.to(original_dtype)


class _PromotingRMSNorm(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = 1e-6

    def forward(self, value):
        return _portable(value, self.weight, self.variance_epsilon)


class _NestedRMS(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, value):
        return self.norm(value)


class TestModuleRMSPrecision(cu.JittorTestCase):
    def test_frozen_and_trainable_inputs_keep_mixed_weight_promotion(self, device):
        source = np.linspace(-.01, .01, 1024, dtype=np.float32).reshape(2, 4, 128)
        weights = np.linspace(.5, 1.5, 128, dtype=np.float32)
        seed = torch.tensor(np.random.RandomState(17832).normal(size=source.shape).astype(np.float32), device=device)
        with forbid_backend_fallbacks():
            for precision in (torch.float16, torch.bfloat16):
                for enabled in (False, True):
                    for requires_grad, no_grad in ((False, False), (True, False), (True, True)):
                        weight = torch.tensor(weights, device=device)
                        model = _NestedRMS(_PromotingRMSNorm(weight)).to(device)
                        for call in range(2):
                            with self.subTest(precision=str(precision), enabled=enabled,
                                              requires_grad=requires_grad, no_grad=no_grad, call=call):
                                x = torch.tensor(source, device=device).to(precision).requires_grad_(requires_grad)
                                reference_x = torch.tensor(source, device=device).to(precision).requires_grad_(requires_grad)
                                with torch.no_grad() if no_grad else nullcontext():
                                    with torch.autocast(device, dtype=precision, enabled=enabled):
                                        expected = _portable(reference_x, weight, 1e-6)
                                        y = model(x)
                                self.assertEqual(str(y.dtype), 'torch.float32')
                                self.assertEqual(y.requires_grad, requires_grad and not no_grad)
                                y.sync()
                                self.assertEqual(y.location(), 'cpu' if device == 'cpu' else 'device')
                                np.testing.assert_array_equal(y.detach().cpu().numpy(), expected.detach().cpu().numpy())
                                if requires_grad and not no_grad:
                                    (y * seed).sum().backward()
                                    (expected * seed).sum().backward()
                                    x.grad.sync()
                                    self.assertEqual(str(x.grad.dtype), str(precision))
                                    self.assertEqual(x.grad.location(), 'cpu' if device == 'cpu' else 'device')
                                    np.testing.assert_array_equal(x.grad.float().cpu().numpy(), reference_x.grad.float().cpu().numpy())

    def test_same_dtype_fp32_cold_and_warm_calls_keep_precise_owner(self, device):
        if device != 'cuda':
            self.skipTest('CUDA native module shortcut')
        # Independent PyTorch 2.8 CUDA weighted RMS values. A candidate
        # portable expression is deliberately not the numerical reference.
        source = [[.1, -.2, .3, -.4, .5, -.6, .7, -.8]]
        weights = [.5, .6428571343421936, .7857142686843872, .9285714626312256,
                   1.0714285373687744, 1.2142857313156128, 1.3571428060531616, 1.5]
        expected = np.array([[.09901455789804459, -.2546088695526123,
                              .46678292751312256, -.7355367541313171,
                              1.0608701705932617, -1.4427835941314697,
                              1.8812764883041382, -2.376349449157715]], dtype='float32')
        model = _NestedRMS(_PromotingRMSNorm(torch.tensor(weights, device=device))).to(device)
        # Publish the ordinary module inventory, as a real training/final
        # evaluation does, then exercise the native call cache repeatedly.
        self.assertEqual(len(list(model.named_parameters())), 1)
        value = torch.tensor(source, device=device, dtype=torch.float32)
        with torch.no_grad(), forbid_backend_fallbacks():
            for _ in range(4):
                output = model(value)
                output.sync()
                self.assertEqual(str(output.dtype), 'torch.float32')
                self.assertEqual(output.location(), 'device')
                np.testing.assert_array_equal(output.cpu().numpy(), expected)

    def test_standard_torch_rmsnorm_keeps_its_own_output_dtype(self, device):
        source = np.linspace(-.01, .01, 1024, dtype=np.float32).reshape(2, 4, 128)
        with forbid_backend_fallbacks():
            for precision in (torch.float16, torch.bfloat16, torch.float32):
                norm = torch.nn.RMSNorm(128, eps=1e-6).to(device)
                for enabled in (False, True):
                    amp_dtype = torch.float16 if precision == torch.float32 else precision
                    with torch.autocast(device, dtype=amp_dtype, enabled=enabled):
                        y = norm(torch.tensor(source, device=device).to(precision))
                    y.sync()
                    self.assertEqual(str(y.dtype), str(precision))
                    self.assertEqual(y.location(), 'cpu' if device == 'cpu' else 'device')


instantiate_device_type_tests(TestModuleRMSPrecision, globals())

if __name__ == '__main__':
    unittest.main()

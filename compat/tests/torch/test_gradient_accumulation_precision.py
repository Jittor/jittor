"""Torch backward entries select consumer order without changing native defaults."""
import unittest
from unittest import mock

import jittor as jt
import numpy as np
import torch

from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests
from jittor._runtime import forbid_backend_fallbacks


def _loss(value):
    intermediate = value.sin()
    first = intermediate * 1e8
    second = intermediate * -1e8
    last = intermediate * 1.
    return ((first + second) + last).sum()


class TestGradientAccumulationPrecision(cu.JittorTestCase):
    def test_both_backward_entries_restore_native_order(self, device):
        original = jt.flags.gradient_accumulation_order
        self.assertEqual(original, 0)
        with forbid_backend_fallbacks():
            for entry in ('grad', 'backward'):
                value = torch.tensor([.2, -.3], device=device, requires_grad=True)
                loss = _loss(value)
                if entry == 'grad':
                    gradient, = torch.autograd.grad(loss, value, create_graph=True)
                    second, = torch.autograd.grad(gradient.sum(), value)
                    np.testing.assert_array_equal(second.cpu().numpy(), np.zeros(2))
                else:
                    loss.backward()
                    gradient = value.grad
                np.testing.assert_array_equal(gradient.detach().cpu().numpy(), np.zeros(2))
                self.assertEqual(jt.flags.gradient_accumulation_order, original)

    def test_both_backward_entries_restore_order_after_error(self, device):
        original = jt.flags.gradient_accumulation_order
        for entry in ('grad', 'backward'):
            value = torch.tensor([.2], device=device, requires_grad=True)
            loss = _loss(value)
            with mock.patch.object(jt.core, 'grad_optional', side_effect=RuntimeError('test backward sentinel')):
                with self.assertRaisesRegex(RuntimeError, 'test backward sentinel'):
                    if entry == 'grad':
                        torch.autograd.grad(loss, value)
                    else:
                        loss.backward()
            self.assertEqual(jt.flags.gradient_accumulation_order, original)


instantiate_device_type_tests(TestGradientAccumulationPrecision, globals())

if __name__ == '__main__':
    unittest.main()

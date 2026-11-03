"""Third-party optimizers keep custom defaults and per-parameter state.

`torch.optim.Optimizer.__init__(params, defaults)` is the base contract that
schedulefree and similar libraries build on: every default key must appear in
each parameter group and ``optimizer.state[p]`` must be a mutable mapping.
"""

from __future__ import print_function

import unittest

import torch


class CustomStep(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, custom_flag=True):
        defaults = dict(lr=lr, custom_flag=custom_flag)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                with torch.no_grad():
                    parameter.add_(parameter.grad, alpha=-group['lr'])
                state = self.state[parameter]
                state['custom_counter'] = state.get('custom_counter', 0) + 1


class TestOptimizerCustomDefaults(unittest.TestCase):
    def test_custom_defaults_propagate_to_param_groups(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        optimizer = CustomStep([parameter], lr=0.5)
        self.assertEqual(optimizer.defaults['custom_flag'], True)
        self.assertEqual(optimizer.param_groups[0]['custom_flag'], True)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.5)

    def test_state_mapping_and_step_work_for_third_party_optimizer(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        parameter.grad = torch.ones(2)
        optimizer = CustomStep([parameter], lr=0.5)
        self.assertEqual(optimizer.state[parameter], {})
        optimizer.state[parameter]['custom_counter'] = 0
        optimizer.step()
        self.assertEqual(optimizer.state[parameter]['custom_counter'], 1)
        self.assertTrue(bool((parameter.detach() == torch.full((2,), 0.5)).all().item()))


if __name__ == '__main__':
    unittest.main()

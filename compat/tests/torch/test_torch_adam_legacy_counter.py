"""Initialized legacy Adam state remains present when its numeric counter is zero."""
import unittest

import numpy as np
import torch

from jittor._runtime import forbid_backend_fallbacks
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


class TestAdamLegacyCounter(cu.JittorTestCase):
    def test_numeric_zero_checkpoint_retains_initialized_state(self, device):
        with forbid_backend_fallbacks():
            _check_numeric_zero(self, device)


def _check_numeric_zero(self, device):
    for optimizer_class in (torch.optim.Adam, torch.optim.AdamW):
        with self.subTest(optimizer=optimizer_class.__name__):
            initial = np.array([1., -2., .3], dtype=np.float32)
            parameter = torch.nn.Parameter(torch.tensor(initial, device=device))
            control = torch.nn.Parameter(torch.tensor(initial, device=device))
            settings = dict(lr=.01, betas=(.7, .93), eps=1e-6, weight_decay=.1)
            if optimizer_class is torch.optim.AdamW:
                settings['fused'] = False
            optimizer = optimizer_class([parameter], **settings)
            fresh = optimizer_class([control], **settings)
            checkpoint = optimizer.state_dict()
            index = checkpoint['param_groups'][0]['params'][0]
            checkpoint['state'][index] = dict(step=0., exp_avg=torch.zeros_like(parameter),
                                             exp_avg_sq=torch.zeros_like(parameter))
            optimizer.load_state_dict(checkpoint)
            self.assertIn(parameter, optimizer.state)
            state = optimizer.state[parameter]
            self.assertSetEqual(set(state), {'step', 'exp_avg', 'exp_avg_sq'})
            counter = state['step']
            self.assertTrue(torch.is_tensor(counter))
            self.assertEqual(str(counter.dtype), 'torch.float32')
            self.assertEqual(counter.shape, ())
            self.assertEqual(counter.item(), 0.)
            self.assertIs(optimizer.state_dict()['state'][index]['step'], counter)
            for value, instance in ((parameter, optimizer), (control, fresh)):
                instance.zero_grad()
                value.square().sum().backward()
                instance.step()
            self.assertIs(optimizer.state[parameter]['step'], counter)
            self.assertEqual(counter.item(), 1.)
            np.testing.assert_array_equal(parameter.detach().cpu().numpy(), control.detach().cpu().numpy())
            for key in ('exp_avg', 'exp_avg_sq'):
                np.testing.assert_array_equal(optimizer.state[parameter][key].cpu().numpy(),
                                              fresh.state[control][key].cpu().numpy())

instantiate_device_type_tests(TestAdamLegacyCounter, globals())

if __name__ == '__main__':
    unittest.main()

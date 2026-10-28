"""Opt-in real registered VERL components, full gradients and three-step training."""

import os
from pathlib import Path
import unittest

from _ecosystem_harness import (REAL_TORCH_PYTHON, EcosystemComparison, _distributions_available,
                                _enabled, _torch_shim_is_active)
from _ecosystem_verl_training_harness import compare_training


@unittest.skipUnless(_enabled('JITTOR_ECOSYSTEM_VERL_TRAINING'), 'opt-in source-bound VERL training gate')
class EcosystemVERLTraining(EcosystemComparison):
    def setUp(self):
        super().setUp()
        self.assertTrue(REAL_TORCH_PYTHON, 'VERL requires an independent binary Torch interpreter')
        self.assertTrue(_torch_shim_is_active(), 'VERL candidate must run the Torch shim')
        self.assertTrue(_distributions_available(('verl', 'numpy', 'transformers', 'ray', 'tensordict')),
                        'Selected VERL gate requires all actual dependencies')
        self.device = os.environ.get('JITTOR_TEST_DEVICES', 'cpu')
        self.assertIn(self.device, ('cpu', 'cuda'))
        checkout, wheel = (os.environ.get(name, '') for name in ('JITTOR_VERL_UPSTREAM', 'JITTOR_VERL_WHEEL'))
        self.assertTrue(checkout and Path(checkout).is_dir(), 'JITTOR_VERL_UPSTREAM must name the clean pin')
        self.assertTrue(wheel and Path(wheel).is_file(), 'JITTOR_VERL_WHEEL must name the actual installed wheel')
        self.checkout, self.wheel = Path(checkout).resolve(), Path(wheel).resolve()

    def test_verl_ppo_gae_vanilla(self):
        compare_training(self, 'verl_ppo_gae_vanilla', self.checkout, self.wheel)

    def test_verl_distillation_k3(self):
        compare_training(self, 'verl_distillation_k3', self.checkout, self.wheel)

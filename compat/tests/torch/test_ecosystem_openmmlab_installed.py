"""Opt-in full original OpenMMLab training with actual installed-wheel provenance."""

import os
from pathlib import Path
import unittest

from _ecosystem_harness import (REAL_TORCH_PYTHON, SPEED_RATIO, EcosystemComparison,
                                _distributions_available, _enabled, _torch_shim_is_active)
from _ecosystem_openmmlab_installed_harness import compare_training


@unittest.skipUnless(_enabled('JITTOR_ECOSYSTEM_OPENMMLAB_INSTALLED'), 'opt-in source-bound OpenMMLab training gate')
class EcosystemOpenMMLabInstalled(EcosystemComparison):
    def setUp(self):
        super().setUp()
        self.assertTrue(REAL_TORCH_PYTHON, 'OpenMMLab needs an independent binary Torch interpreter')
        self.assertTrue(_torch_shim_is_active(), 'Candidate must run the actual Torch shim')
        self.assertFalse(SPEED_RATIO, 'Tiny OpenMMLab correctness has no performance ratio gate')
        self.assertTrue(_distributions_available(('mmengine', 'mmcv', 'numpy')), 'Actual OpenMMLab packages are required')
        self.device = os.environ.get('JITTOR_TEST_DEVICES', 'cpu')
        self.assertIn(self.device, ('cpu', 'cuda'))
        self.checkouts, self.wheels = {}, {}
        for repo in ('mmengine', 'mmcv'):
            for kind, store in (('CHECKOUT', self.checkouts), ('WHEEL', self.wheels)):
                name = 'JITTOR_' + repo.upper() + '_' + kind
                value = os.environ.get(name, '')
                self.assertTrue(value and Path(value).exists(), name + ' must name the clean source or installed wheel')
                store[repo] = Path(value).resolve()

    def test_mmengine_train_step(self):
        compare_training(self, 'mmengine_train_step', self.checkouts, self.wheels)

    def test_mmcv_convmodule_train_step(self):
        compare_training(self, 'mmcv_convmodule_train_step', self.checkouts, self.wheels)

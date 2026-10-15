"""Opt-in actual Swift loader/template/Trainer training, independently on CPU/CUDA."""

import os
from pathlib import Path
import unittest

from _ecosystem_harness import (REAL_TORCH_PYTHON, SPEED_RATIO, EcosystemComparison,
    _distributions_available, _enabled, _torch_shim_is_active)
from _ecosystem_swift_harness import compare_training


@unittest.skipUnless(_enabled('JITTOR_ECOSYSTEM_SWIFT'), 'opt-in real Swift Trainer gate')
class EcosystemSwift(EcosystemComparison):
    def setUp(self):
        super().setUp()
        self.assertTrue(REAL_TORCH_PYTHON, 'Swift Trainer requires an independent real Torch oracle')
        self.assertTrue(_torch_shim_is_active(), 'Swift candidate must run the Torch shim')
        self.assertFalse(SPEED_RATIO, 'Swift Trainer correctness gate has no speed ratio')
        self.assertTrue(_distributions_available(('swift', 'transformers', 'peft', 'accelerate')),
                        'Selected real Swift gate requires all actual downstream dependencies')
        self.device = os.environ.get('JITTOR_TEST_DEVICES', 'cpu')
        self.assertIn(self.device, ('cpu', 'cuda'))
        upstream, wheel = (os.environ.get(name, '') for name in
                           ('JITTOR_SWIFT_UPSTREAM', 'JITTOR_SWIFT_WHEEL'))
        self.assertTrue(upstream and Path(upstream).is_dir(), 'JITTOR_SWIFT_UPSTREAM must name a clean pinned checkout')
        self.assertTrue(wheel and Path(wheel).is_file(), 'JITTOR_SWIFT_WHEEL must name the actual installed wheel')
        self.upstream, self.wheel = Path(upstream).resolve(), Path(wheel).resolve()

    def test_ms_swift_trainer_full_llama(self):
        compare_training(self, self.upstream, self.wheel)

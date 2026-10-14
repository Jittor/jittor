"""Opt-in CPU/CUDA gates for real MMEngine train_step and MMCV ConvModule.

Run ``nox -s ecosystem_openmmlab`` with separately provisioned oracle/candidate
interpreters. No compiled MMCV operator or production performance is claimed.
"""

import os
import unittest

from _ecosystem_harness import (
    REAL_TORCH_PYTHON,
    SPEED_RATIO,
    EcosystemComparison,
    _distributions_available,
    _enabled,
    _torch_shim_is_active,
)
from _ecosystem_openmmlab_cases import REQUIRED_GRADIENTS
from _ecosystem_openmmlab_harness import compare_training


@unittest.skipUnless(_enabled("JITTOR_ECOSYSTEM_OPENMMLAB"), "opt-in real OpenMMLab training gate")
class EcosystemOpenMMLab(EcosystemComparison):
    def setUp(self):
        super().setUp()
        self.assertTrue(REAL_TORCH_PYTHON, "OpenMMLab requires an independent real Torch oracle")
        self.assertTrue(_torch_shim_is_active(), "OpenMMLab candidate must run the Torch shim")
        self.assertFalse(SPEED_RATIO, "OpenMMLab correctness gate requires an empty speed ratio")
        self.device = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        self.assertIn(self.device, ("cpu", "cuda"), "select exactly one actual CPU/CUDA device")
        self.repeats = 1

    def test_mmengine_train_step(self):
        self.assertTrue(
            _distributions_available(("mmengine",)),
            "selected MMEngine gate requires available mmengine",
        )
        self._compare(
            "mmengine_train_step", required_gradients=REQUIRED_GRADIENTS["mmengine_train_step"]
        )
        compare_training(self, "mmengine_train_step")

    def test_mmcv_convmodule_train_step(self):
        self.assertTrue(
            _distributions_available(("mmcv", "mmengine")),
            "selected MMCV gate requires available mmcv and mmengine",
        )
        self._compare(
            "mmcv_convmodule_train_step",
            required_gradients=REQUIRED_GRADIENTS["mmcv_convmodule_train_step"],
        )
        compare_training(self, "mmcv_convmodule_train_step")

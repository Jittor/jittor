"""Opt-in real 32768-token local/global Longformer parity.

Set JITTOR_ECOSYSTEM_LONGFORMER_32K=1, JITTOR_TEST_DEVICES=cpu or cuda,
REAL_TORCH_PYTHON, and an empty JITTOR_ECOSYSTEM_SPEED_RATIO. This tiny-width
correctness protocol does not establish production-model speed.
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
from _ecosystem_longformer_cases import REQUIRED_GRADIENTS


@unittest.skipUnless(_enabled("JITTOR_ECOSYSTEM_LONGFORMER_32K"),
                     "opt-in Longformer 32768-token gate")
class EcosystemLongformer(EcosystemComparison):
    def setUp(self):
        super().setUp()
        self.assertTrue(REAL_TORCH_PYTHON, "Longformer requires an independent real Torch oracle")
        self.assertTrue(_torch_shim_is_active(), "Longformer candidate must run the Torch shim")
        self.assertTrue(_distributions_available(("transformers",)),
                        "Longformer gate requires available Transformers")
        self.assertFalse(SPEED_RATIO, "32K tiny-width correctness gate requires an empty speed ratio")
        self.device = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        self.assertIn(self.device, ("cpu", "cuda"), "select exactly one actual CPU/CUDA device")
        self.repeats = 1
        if self.device == "cuda":
            self.forward_tolerance = 5e-3
            self.backward_tolerance = 2e-2

    def test_transformers_longformer_32768(self):
        self._compare("transformers_longformer_32768", required_gradients=REQUIRED_GRADIENTS)

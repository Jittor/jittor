"""Opt-in pinned vLLM native-path component parity, CPU first then CUDA.

Run ``nox -s ecosystem_vllm`` with two provisioned interpreters. This gate has
no trainable parameters and does not validate model families or an engine.
"""
import os
import unittest

from _ecosystem_harness import (
    REAL_TORCH_PYTHON, SPEED_RATIO, EcosystemComparison, _distributions_available,
    _enabled, _torch_shim_is_active,
)
from _ecosystem_vllm_cases import REQUIRED_GRADIENTS


@unittest.skipUnless(_enabled("JITTOR_ECOSYSTEM_VLLM"), "opt-in vLLM component gate")
class EcosystemVLLM(EcosystemComparison):
    def setUp(self):
        super().setUp()
        self.assertTrue(REAL_TORCH_PYTHON, "vLLM requires an independent real Torch oracle")
        self.assertTrue(_torch_shim_is_active(), "vLLM candidate must run the Torch shim")
        self.assertTrue(_distributions_available(("vllm", "transformers")),
                        "vLLM component gate requires actual vLLM and Transformers")
        self.assertFalse(SPEED_RATIO, "vLLM tiny inference primitives are a correctness gate")
        self.device = os.environ.get("JITTOR_TEST_DEVICES", "cpu")
        self.assertIn(self.device, ("cpu", "cuda"), "select one actual CPU or CUDA device")
        self.repeats = 1
        # Identical conservative correctness thresholds on both devices.
        self.forward_tolerance = 2e-3
        self.backward_tolerance = 1e-2

    def test_vllm_decoder_primitives(self):
        self._compare("vllm_decoder_primitives", required_gradients=REQUIRED_GRADIENTS)


if __name__ == "__main__":
    unittest.main()

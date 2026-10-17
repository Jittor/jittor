"""The optional Qwen2 V2 runner policy must stay model-specific and reversible."""

import os
from types import ModuleType
import unittest
from unittest.mock import patch
import uuid

from jittor.compat import transaction as tx
from jittor_adapters.vllm.runner_policy import allow_experimental_qwen2_v2_runner


class Qwen2V2PolicyTests(unittest.TestCase):
    def test_disabled_keeps_vllm_default(self):
        module = ModuleType("fake_vllm_config")
        original = frozenset({"Qwen3ForCausalLM", "LlamaForCausalLM"})
        module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = original
        with patch.dict(os.environ, {"JITTOR_VLLM_EXPERIMENTAL_QWEN2_V2": "0"}):
            self.assertFalse(allow_experimental_qwen2_v2_runner(module))
        self.assertIs(module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES, original)

    def test_opt_in_adds_only_qwen2_and_rolls_back(self):
        module = ModuleType("fake_vllm_config")
        original = frozenset({"Qwen3ForCausalLM", "LlamaForCausalLM"})
        module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = original
        with patch.dict(os.environ, {"JITTOR_VLLM_EXPERIMENTAL_QWEN2_V2": "1"}):
            with tx.runtime_hook("qwen2-v2-policy-" + uuid.uuid4().hex) as hook:
                self.assertTrue(allow_experimental_qwen2_v2_runner(module))
                self.assertFalse(allow_experimental_qwen2_v2_runner(module))
                self.assertEqual(
                    module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES,
                    original | {"Qwen2ForCausalLM"},
                )
                self.assertNotIn(
                    "Qwen2VLForConditionalGeneration",
                    module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES,
                )
            hook.rollback()
        self.assertIs(module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES, original)

    def test_unknown_upstream_policy_fails_closed(self):
        module = ModuleType("fake_vllm_config")
        with patch.dict(os.environ, {"JITTOR_VLLM_EXPERIMENTAL_QWEN2_V2": "1"}):
            with self.assertRaisesRegex(RuntimeError, "policy changed"):
                allow_experimental_qwen2_v2_runner(module)


if __name__ == "__main__":
    unittest.main()

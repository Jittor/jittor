"""Numerical parity with PyTorch for the downstream libraries Jittor targets.

The existing downstream tests assert that a model *runs* under
``import torch`` -> Jittor and that gradients appear.  That is not the same as
agreeing with PyTorch.  This module runs the same case twice -- once in a real
PyTorch interpreter, once in this Jittor one -- from identical weights, and
compares the forward output, every parameter gradient and every input gradient.

The comparison itself lives in ``_ecosystem_harness``, together with the
``REAL_TORCH_PYTHON`` / ``JITTOR_ECOSYSTEM_SPEED_RATIO`` configuration; this
module only selects the parity cases.
"""

import unittest

from _ecosystem_harness import (
    REAL_TORCH_PYTHON,
    EcosystemComparison,
    _cuda_is_available,
    _enabled,
    _npu_is_available,
    _torch_shim_is_active,
)


@unittest.skipUnless(REAL_TORCH_PYTHON, "REAL_TORCH_PYTHON is not configured")
@unittest.skipUnless(_torch_shim_is_active(), "this interpreter does not run torch as Jittor")
class EcosystemParity(EcosystemComparison):
    """Compare a downstream library's numbers between PyTorch and Jittor."""

    def test_transformers_gpt2(self):
        self._compare("transformers_gpt2")

    def test_transformers_llama(self):
        self._compare("transformers_llama")

    def test_transformers_bert(self):
        self._compare("transformers_bert")

    def test_transformers_vit(self):
        self._compare("transformers_vit")

    def test_diffusers_unet2d(self):
        self._compare("diffusers_unet2d")

    def test_transformers_t5(self):
        self._compare("transformers_t5")

    def test_transformers_whisper(self):
        self._compare("transformers_whisper")

    def test_diffusers_dit(self):
        self._compare("diffusers_dit")

    def test_peft_lora_llama(self):
        self._compare("peft_lora_llama")

    def test_mmcv_conv_module(self):
        self._compare("mmcv_conv_module")

    def test_mmengine_base_module(self):
        self._compare("mmengine_base_module")

    def test_ms_swift_lora_llama(self):
        self._compare("ms_swift_lora_llama")


class OpenAIWhisperParity(EcosystemComparison):
    """Original OpenAI package on CPU; accelerator support is not declared."""

    def _require_runtime(self):
        for ready, reason in (
            (bool(REAL_TORCH_PYTHON), "REAL_TORCH_PYTHON is not configured"),
            (_torch_shim_is_active(), "this interpreter does not run torch as Jittor"),
        ):
            if not ready:
                if _enabled("JITTOR_REQUIRE_WHISPER"):
                    self.fail("JITTOR_REQUIRE_WHISPER=1: " + reason)
                self.skipTest(reason)

    def test_openai_whisper(self):
        self._require_runtime()
        self._compare("openai_whisper")

    def test_openai_whisper_log_mel(self):
        self._require_runtime()
        self.forward_tolerance = 2e-5
        self.backward_tolerance = 2e-3
        self._compare("openai_whisper_log_mel", expected_output_shape=(80, 100),
                      expected_output_dtype="float32")


@unittest.skipUnless(REAL_TORCH_PYTHON, "REAL_TORCH_PYTHON is not configured")
@unittest.skipUnless(_torch_shim_is_active(), "this interpreter does not run torch as Jittor")
@unittest.skipUnless(_cuda_is_available(), "CUDA is unavailable")
class EcosystemParityCUDA(EcosystemParity):
    """The same comparison with both runtimes executing on the GPU."""

    device = "cuda"
    # Accelerator kernels pick different accumulation orders than the CPU
    # reference implementations these libraries were written against.
    forward_tolerance = 5e-3
    backward_tolerance = 2e-2


@unittest.skipUnless(REAL_TORCH_PYTHON, "REAL_TORCH_PYTHON is not configured")
@unittest.skipUnless(_torch_shim_is_active(), "this interpreter does not run torch as Jittor")
@unittest.skipUnless(_npu_is_available(), "ACL is unavailable")
class EcosystemParityNPU(EcosystemComparison):
    """Downstream model parity on Jittor ACL and torch_npu."""

    device = "npu"
    forward_tolerance = 5e-3
    backward_tolerance = 2e-2

    def test_diffusers_unet2d(self):
        self._compare("diffusers_unet2d")

    def test_mmcv_conv_module(self):
        self._compare("mmcv_conv_module")

    def test_mmengine_base_module(self):
        self._compare("mmengine_base_module")

    def test_ms_swift_lora_llama(self):
        self._compare("ms_swift_lora_llama")


if __name__ == "__main__":
    unittest.main()

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
        self._compare("transformers_bert", required_gradients=(
            "grad::encoder.pooler.dense.weight", "grad::encoder.pooler.dense.bias"))

    def test_transformers_vit(self):
        self._compare("transformers_vit", required_gradients=(
            "grad::encoder.pooler.dense.weight", "grad::encoder.pooler.dense.bias",
            "ingrad::pixel_values"))

    def test_diffusers_unet2d(self):
        self._compare("diffusers_unet2d")

    def test_transformers_t5(self):
        self._compare("transformers_t5")

    def test_transformers_whisper(self):
        self._compare("transformers_whisper")

    def test_transformers_clip_multimodal(self):
        self._compare("transformers_clip_multimodal", required_gradients=(
            "grad::clip.visual_projection.weight",
            "grad::clip.text_projection.weight",
            "grad::clip.text_model.embeddings.token_embedding.weight",
            "grad::clip.logit_scale", "ingrad::pixel_values"))

    def test_transformers_mixtral_moe(self):
        # Both sparse blocks must route to all four experts in this fixture;
        # a gate with only a dense path or inactive experts is incomplete.
        experts = tuple(
            "grad::model.layers.%d.block_sparse_moe.experts.%d.%s.weight" % (layer, expert, weight)
            for layer in range(2) for expert in range(4) for weight in ("w1", "w2", "w3"))
        routers = tuple("grad::model.layers.%d.block_sparse_moe.gate.weight" % layer
                        for layer in range(2))
        self._compare("transformers_mixtral_moe", required_gradients=experts + routers)

    def test_transformers_llama_rope_2048(self):
        self._compare("transformers_llama_rope_2048")

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

    def test_torchmetrics_regression(self):
        self._compare("torchmetrics_regression", required_gradients=(
            "ingrad::prediction", "ingrad::target"))

    def test_tensordict_index_and_stack(self):
        self._compare("tensordict_index_and_stack", required_gradients=(
            "grad::projection.weight", "grad::projection.bias", "ingrad::sample"))


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

"""Opt-in vLLM runner policies verified with the Jittor adapter."""

import os

from jittor.compat.transaction import set_attr


def allow_experimental_qwen2_v2_runner(module):
    """Enable V2 only for the text Qwen2 architecture when explicitly asked.

    vLLM 0.24 does not include Qwen2ForCausalLM in its default V2 list. The
    global VLLM_USE_V2_MODEL_RUNNER switch also opts unrelated architectures
    in, so modify its model-specific architecture list for this experiment.
    """
    if os.environ.get("JITTOR_VLLM_EXPERIMENTAL_QWEN2_V2") != "1":
        return False
    architectures = getattr(module, "DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES", None)
    if not isinstance(architectures, frozenset):
        raise RuntimeError("vLLM V2 architecture policy changed; refusing Qwen2 override")
    if "Qwen2ForCausalLM" in architectures:
        return False
    set_attr(module, "DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES",
             architectures | {"Qwen2ForCausalLM"})
    return True


PATCHES = {"vllm.config.vllm": allow_experimental_qwen2_v2_runner}

# Copyright 2025 The HuggingFace Team. Apache License 2.0.
# Exact HF 5.9.0 function excerpts for host policy/lifecycle tests only.
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
import sys


class PreTrainedModel:
    def init_weights(self):
        self.initialized = True

    @classmethod
    def get_init_context(
        cls, dtype: torch.dtype, is_quantized: bool, _is_ds_init_called: bool, allow_all_kernels: bool | None
    ):
        # Need to instantiate with correct dtype
        init_contexts = [local_torch_dtype(dtype, cls.__name__), init.no_tie_weights(), apply_patches()]
        # Needed as we cannot forward the `allow_all_kernels` arg in the model's __init__
        if allow_all_kernels:
            init_contexts.append(allow_all_hub_kernels())
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # We cannot initialize the model on meta device with deepspeed when not quantized
            if not is_quantized and not _is_ds_init_called:
                logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
                init_contexts.extend(
                    [
                        init.no_init_weights(),
                        deepspeed.zero.Init(config_dict_or_path=deepspeed_config()),
                        set_zero3_state(),
                    ]
                )
            elif is_quantized:
                init_contexts.extend([torch.device("meta"), set_quantized_state()])
        else:
            # meta_device_safe_creation_ops patches torch.linspace to default to CPU
            # so that custom models calling .item() during __init__ (e.g. drop-path
            # schedules) don't crash on meta tensors.
            init_contexts.extend([torch.device("meta"), init.meta_device_safe_creation_ops()])

        return init_contexts


@contextmanager
def no_init_weights():
    """
    Disable weight initialization both at the torch-level, and at the transformers-level (`init_weights`).
    This is used to speed-up initializing an empty model with deepspeed, as we do not initialize the model on meta device
    with deepspeed, but we still don't need to run expensive weight initializations as we are loading params afterwards.
    """
    from .modeling_utils import PreTrainedModel

    def empty_func(*args, **kwargs):
        pass

    originals = defaultdict(dict)
    try:
        # Replace all torch funcs by empty ones
        for module_name in TORCH_MODULES_TO_PATCH:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                for func_name in TORCH_INIT_FUNCTIONS.keys():
                    if hasattr(module, func_name):
                        originals[module][func_name] = getattr(module, func_name)
                        setattr(module, func_name, empty_func)

        # Also patch our own `init_weights`
        original_init_weights = PreTrainedModel.init_weights
        PreTrainedModel.init_weights = empty_func

        yield
    finally:
        # Set back the original torch functions on all modules
        for module, functions in originals.items():
            for func_name, func in functions.items():
                setattr(module, func_name, func)
        # Set back `init_weights`
        PreTrainedModel.init_weights = original_init_weights

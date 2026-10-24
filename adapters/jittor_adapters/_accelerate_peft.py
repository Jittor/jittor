"""Pinned Accelerate/PEFT checkpoint routing through public FSDP state APIs.

PEFT owns adapter filtering and names; DCP owns shard metadata and storage.
This module performs no tensor arithmetic, sharding or framework mutation.
"""
from collections.abc import Mapping
from functools import wraps
import inspect
import os

from ._common import (
    UnsupportedAdapterVersion, require_version, required_patch, replace,
    replace_bound_aliases,
)

ACCELERATE_SUPPORTED_VERSIONS = frozenset(("1.10.1",))
PEFT_SUPPORTED_VERSIONS = frozenset(("0.17.1",))


def _adapter_state(model, canonical):
    from peft import PeftType, get_peft_model_state_dict

    require_version("peft", PEFT_SUPPORTED_VERSIONS)
    if tuple(inspect.signature(get_peft_model_state_dict).parameters) != (
            "model", "state_dict", "adapter_name", "unwrap_compiled", "save_embedding_layers"):
        raise UnsupportedAdapterVersion("PEFT adapter state getter contract changed")
    name = model.active_adapter
    if not isinstance(name, str) or not name or name not in model.peft_config:
        raise NotImplementedError("FSDP2 sharded adapter checkpoints require one named active adapter")
    config = model.peft_config[name]
    if (config.peft_type != PeftType.LORA or getattr(config, "use_dora", False)
            or getattr(config, "modules_to_save", None)
            or getattr(config, "trainable_token_indices", None)):
        raise NotImplementedError("FSDP2 sharded adapter checkpoints support plain LoRA only")

    # Unique inert values let PEFT apply its own exact filter/rename rules in
    # both directions. Reject transformations or ambiguous aliases explicitly.
    symbols = {key: object() for key in canonical}
    names = {id(value): key for key, value in symbols.items()}
    symbolic = get_peft_model_state_dict(model, state_dict=symbols, adapter_name=name)
    if not isinstance(symbolic, Mapping) or not symbolic:
        raise ValueError("PEFT selected no checkpoint adapter state")
    inverse = {}
    for key, value in symbolic.items():
        if not isinstance(key, str) or id(value) not in names:
            raise NotImplementedError("PEFT adapter filtering transformed checkpoint values")
        inverse[key] = names[id(value)]
    if len(set(inverse.values())) != len(inverse):
        raise ValueError("PEFT adapter checkpoint names are not one-to-one")
    selected = get_peft_model_state_dict(model, state_dict=canonical, adapter_name=name)
    if (not isinstance(selected, Mapping) or set(selected) != set(inverse)
            or any(selected[key] is not canonical[fqn] for key, fqn in inverse.items())):
        raise NotImplementedError("PEFT adapter filtering did not preserve checkpoint values")
    return dict(selected), inverse


@required_patch
def patch_accelerate_peft_fsdp_utils(module):
    """Fix only FSDP2 SHARDED adapter-only model I/O in pinned Accelerate."""
    require_version("accelerate", ACCELERATE_SUPPORTED_VERSIONS)
    save = vars(module).get("save_fsdp_model")
    load = vars(module).get("load_fsdp_model")
    options = vars(module).get("_prepare_sd_options")
    is_peft_model = vars(module).get("is_peft_model")
    common = ("fsdp_plugin", "accelerator", "model")
    if (not callable(save) or tuple(inspect.signature(save).parameters) !=
            common + ("output_dir", "model_index", "adapter_only")
            or not callable(load) or tuple(inspect.signature(load).parameters) !=
            common + ("input_dir", "model_index", "adapter_only")
            or not callable(options) or tuple(inspect.signature(options).parameters) != ("fsdp_plugin",)
            or not callable(is_peft_model)
            or vars(module).get("FSDP_MODEL_NAME") != "pytorch_model_fsdp"):
        raise UnsupportedAdapterVersion("Accelerate PEFT model checkpoint contract changed")
    if getattr(save, "_jittor_fsdp_peft_sharded", False):
        return False

    def applies(plugin, model, adapter_only):
        from torch.distributed.fsdp import StateDictType
        return (plugin.fsdp_version == 2
                and plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT
                and adapter_only and is_peft_model(model))

    @wraps(save)
    def save_fsdp_model(fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False):
        if not applies(fsdp_plugin, model, adapter_only):
            return save(fsdp_plugin, accelerator, model, output_dir,
                        model_index=model_index, adapter_only=adapter_only)
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        canonical = get_model_state_dict(model, options=options(fsdp_plugin))
        selected, _ = _adapter_state(model, canonical)
        checkpoint = os.path.join(output_dir, "pytorch_model_fsdp_%d" % model_index)
        dcp.save(state_dict={"model": selected}, storage_writer=dcp.FileSystemWriter(checkpoint),
                 planner=DefaultSavePlanner())

    @wraps(load)
    def load_fsdp_model(fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False):
        if not applies(fsdp_plugin, model, adapter_only):
            return load(fsdp_plugin, accelerator, model, input_dir,
                        model_index=model_index, adapter_only=adapter_only)
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

        accelerator.wait_for_everyone()
        sd_options = options(fsdp_plugin)
        canonical = get_model_state_dict(model, options=sd_options)
        selected, inverse = _adapter_state(model, canonical)
        checkpoint = (os.path.join(input_dir, "pytorch_model_fsdp_%d" % model_index)
                      if "pytorch_model_fsdp" not in os.fspath(input_dir) else input_dir)
        state = {"model": selected}
        dcp.load(state_dict=state, storage_reader=dcp.FileSystemReader(checkpoint),
                 planner=DefaultLoadPlanner())
        loaded = state["model"]
        if not isinstance(loaded, Mapping) or set(loaded) != set(inverse):
            raise ValueError("PEFT checkpoint adapter keys do not match the active model")
        # Keep every frozen/nonselected entry from the canonical snapshot so
        # the owner can rebuild flat shards under its normal strict contract.
        canonical.update({inverse[key]: value for key, value in loaded.items()})
        return set_model_state_dict(model, canonical, options=sd_options)

    for name, original, replacement in (("save_fsdp_model", save, save_fsdp_model),
                                         ("load_fsdp_model", load, load_fsdp_model)):
        replacement._jittor_fsdp_peft_sharded = True
        replacement._jittor_original = original
        replace(module, name, replacement, original)
        replace_bound_aliases("accelerate", name, original, replacement)
        replace_bound_aliases("transformers", name, original, replacement)
    return True

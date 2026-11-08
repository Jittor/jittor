"""Public DCP option semantics required by upstream FSDP2/PEFT checkpointing.

Run this same test in an independent binary Torch process and the Jittor Torch
frontend; no collective or FSDP result is inferred from this API contract.
"""
from dataclasses import asdict, fields, is_dataclass, replace

import pytest
from torch.distributed.checkpoint.state_dict import StateDictOptions


def test_state_dict_options_dataclass_replace_preserves_other_options():
    original = StateDictOptions(full_state_dict=True, cpu_offload=True,
                                broadcast_from_rank0=True)
    assert is_dataclass(original)
    saved = replace(original, ignore_frozen_params=True)
    restored = replace(original, strict=False)
    assert original.ignore_frozen_params is False and original.strict is True
    assert saved.ignore_frozen_params is True and saved.full_state_dict is True
    assert saved.cpu_offload is True and saved.broadcast_from_rank0 is True
    assert restored.strict is False and restored.full_state_dict is True
    assert asdict(saved)["dsd_fqn_modifiers"] == "_fqn_modifiers"
    assert [f.name for f in fields(original)] == [
        "full_state_dict", "cpu_offload", "ignore_frozen_params",
        "keep_submodule_prefixes", "strict", "broadcast_from_rank0",
        "flatten_optimizer_state_dict", "dsd_fqn_modifiers",
    ]
    assert StateDictOptions(True).full_state_dict is True
    assert StateDictOptions() == StateDictOptions()
    with pytest.raises(TypeError):
        replace(original, nonexistent=True)

"""Private full-state loading for true FSDP2 shards."""

import jittor as jt

from . import common, shard


def _load_full_state_dict(module, state_dict):
    """Broadcast a full state dict and write each rank-local FSDP shard."""
    module_names = {id(child): name for name, child in module.named_modules()}
    consumed = set()
    for fsdp_module in shard._iter_fsdp_modules(module, recurse=True):
        state = getattr(fsdp_module, "_fsdp_state", None)
        if state is None or not getattr(state, "true_fsdp_initialized", False):
            continue
        prefix = module_names.get(id(fsdp_module), "")
        full_values = []
        for entry in state.true_fsdp_params:
            key = "{}.{}".format(prefix, entry.name) if prefix else entry.name
            if key not in state_dict:
                raise KeyError(
                    "missing FSDP parameter in full state dict: {}".format(key))
            full = state_dict[key]
            if common._world_size() > 1:
                full = full.mpi_broadcast(0)
            full_values.append(full)
            consumed.add(key)

        if getattr(state, "true_fsdp_flat", False):
            if not full_values:
                continue
            flat = common._pad_flat(
                jt.concat(
                    [common._flatten_var(value) for value in full_values], dim=0),
                state.true_fsdp_flat_padded_numel,
            )
            local = common._slice_flat(
                flat,
                int(state.true_fsdp_rank) * int(state.true_fsdp_flat_shard_numel),
                int(state.true_fsdp_flat_shard_numel),
            ).stop_grad()
            state.true_fsdp_flat_shard.update(local)
            shard._refresh_flat_entry_shards(state)
        else:
            for entry, full in zip(state.true_fsdp_params, full_values):
                padded = common._pad_flat(
                    common._flatten_var(full), entry.padded_numel)
                local = common._slice_flat(
                    padded,
                    int(state.true_fsdp_rank) * int(entry.shard_numel),
                    int(entry.shard_numel),
                ).stop_grad()
                entry.shard.update(local)

        for entry in state.true_fsdp_params:
            if getattr(entry, "requires_grad", True) and entry.shard.is_stop_grad():
                entry.shard.start_grad()
            object.__setattr__(entry.owner, entry.attr, entry.shard)
            entry.full_param = None
        state.true_fsdp_unsharded = False
    return consumed

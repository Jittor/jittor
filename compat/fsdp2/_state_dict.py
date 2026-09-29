"""Private FULL checkpoint transforms for the existing FSDP2 shard owner."""

import numpy as np

import jittor as jt

from . import common, shard


def _checkpoint_entries(module):
    module_names = {id(child): name for name, child in module.named_modules()}
    for child in shard._iter_fsdp_modules(module, recurse=True):
        state = getattr(child, "_fsdp_state", None)
        if state is None or not getattr(state, "true_fsdp_initialized", False):
            continue
        prefix = module_names.get(id(child), "")
        for entry in state.true_fsdp_params:
            name = "{}.{}".format(prefix, entry.name) if prefix else entry.name
            yield name, state, entry


def _checkpoint_tensor(value, reference, *, preserve_dtype=False):
    from ..torch.context import get_install_context

    g = get_install_context(jt).target_namespace
    backend, device = jt.core.dispatch_context([reference])
    target = "{}:{}".format(backend, device) if backend == "cuda" else backend
    return g.tensor(value, dtype=value.dtype if preserve_dtype else reference.dtype, device=target).detach()


def _get_full_state_dict(module):
    result = {}
    flat_cache = {}
    for name, value in module.state_dict().items():
        state, entry = shard._fsdp_param_entry(value)
        if state is not None and entry is not None:
            with common._frontend_scope(state):
                if getattr(state, "true_fsdp_flat", False):
                    if id(state) not in flat_cache:
                        flat_cache[id(state)] = common._all_gather_shards(
                            state.true_fsdp_flat_shard, getattr(state, "shard_group", None))
                    value = common._slice_flat(flat_cache[id(state)], entry.flat_offset, entry.numel).reshape(entry.shape)
                else:
                    value = common._all_gather_shards(entry.shard, getattr(state, "shard_group", None))
                    value = common._slice_flat(value, 0, entry.numel).reshape(entry.shape)
        result[name] = value.detach().clone()
    return result


def _gather_optimizer_state_dict(module, state_dict, rank_metadata):
    from ..torch.context import get_install_context

    g = get_install_context(jt).target_namespace
    metadata = {}
    for rank in rank_metadata:
        for name, fields in rank.items():
            target = metadata.setdefault(name, {})
            for field, spec in fields.items():
                old = target.get(field)
                if spec[0] == "step" and old is not None:
                    spec = ("step", max(spec[1], old[1]))
                elif old is not None and old != spec:
                    raise ValueError("FSDP optimizer state metadata differs across ranks")
                target[field] = spec
    result = {name: dict(entry) for name, entry in state_dict["state"].items()}
    entries = list(_checkpoint_entries(module))
    grouped = {}
    for name, state, entry in entries:
        grouped.setdefault(id(state), (state, []))[1].append((name, entry))
        if name in metadata:
            target = result.setdefault(name, {})
            for field, spec in metadata[name].items():
                if spec[0] == "step":
                    target[field] = g.tensor(np.asarray(spec[1], dtype=np.float32), device="cpu")
                elif spec[0] == "value":
                    target[field] = spec[1]
    for state, named_entries in grouped.values():
        fields = sorted({field for name, _ in named_entries
                         for field, spec in metadata.get(name, {}).items() if spec[0] == "tensor"})
        with common._frontend_scope(state):
            for field in fields:
                local_values = []
                for name, entry in named_entries:
                    value = state_dict["state"].get(name, {}).get(field)
                    if value is None:
                        spec = next(metadata[current][field]
                                    for current, _ in named_entries if field in metadata.get(current, {}))
                        value = common._zeros_like_shape(entry.shard, entry.shard.shape, dtype=spec[1]).stop_grad()
                    if tuple(value.shape) != tuple(entry.shard.shape):
                        raise ValueError("FSDP optimizer state is not parameter-shaped: {}.{}".format(name, field))
                    local_values.append(common._flatten_var(value))
                if getattr(state, "true_fsdp_flat", False):
                    local = jt.concat(local_values, dim=0)
                    local = common._pad_flat(local, state.true_fsdp_flat_shard_numel)
                    full = common._all_gather_shards(local, getattr(state, "shard_group", None))
                    for name, entry in named_entries:
                        if field in metadata.get(name, {}):
                            result[name][field] = common._slice_flat(full, entry.flat_offset, entry.numel).reshape(entry.shape).detach().clone()
                else:
                    for (name, entry), local in zip(named_entries, local_values):
                        if field in metadata.get(name, {}):
                            full = common._all_gather_shards(local, getattr(state, "shard_group", None))
                            result[name][field] = common._slice_flat(full, 0, entry.numel).reshape(entry.shape).detach().clone()
    return {"state": result, "param_groups": state_dict["param_groups"]}


def _shard_optimizer_state_dict(module, state_dict):
    entries = {name: (state, entry) for name, state, entry in _checkpoint_entries(module)}
    result = {}
    for name, fields in state_dict["state"].items():
        if name not in entries:
            result[name] = dict(fields)
            continue
        state, entry = entries[name]
        result[name] = {}
        with common._frontend_scope(state):
            for field, value in fields.items():
                if field == "step" or not isinstance(value, (jt.Var, np.ndarray)):
                    result[name][field] = value
                    continue
                if tuple(value.shape) != tuple(entry.shape):
                    raise ValueError("FULL FSDP optimizer tensor has wrong shape: {}.{}".format(name, field))
                full = common._flatten_var(_checkpoint_tensor(value, entry.shard, preserve_dtype=True))
                if getattr(state, "true_fsdp_flat", False):
                    start = min(max(int(state.true_fsdp_rank) * int(state.true_fsdp_flat_shard_numel)
                                    - int(entry.flat_offset), 0), int(entry.numel))
                    local = common._slice_flat(full, start, entry.shard_numel)
                else:
                    full = common._pad_flat(full, entry.padded_numel)
                    local = common._slice_flat(full, int(state.true_fsdp_rank) * entry.shard_numel, entry.shard_numel)
                result[name][field] = local.detach().clone()
    return {"state": result, "param_groups": state_dict["param_groups"]}


def _load_full_state_dict(module, state_dict, strict=True):
    """Write a caller-broadcast FULL state dict to the existing local shards."""
    module_names = {id(child): name for name, child in module.named_modules()}
    current = module.state_dict()
    missing = set(current) - set(state_dict)
    unexpected = set(state_dict) - set(current)
    if strict and (missing or unexpected):
        raise RuntimeError("FSDP state dict keys mismatch: missing={} unexpected={}".format(sorted(missing), sorted(unexpected)))
    if missing and not strict:
        state_dict = dict(_get_full_state_dict(module), **state_dict)
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
            full = _checkpoint_tensor(state_dict[key], entry.shard)
            if tuple(full.shape) != tuple(entry.shape):
                raise RuntimeError("FSDP checkpoint parameter shape mismatch: {}".format(key))
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
            local = jt.Var.copy(common._slice_flat(
                flat,
                int(state.true_fsdp_rank) * int(state.true_fsdp_flat_shard_numel),
                int(state.true_fsdp_flat_shard_numel),
            )).stop_grad()
            state.true_fsdp_flat_shard.update(local)
            shard._refresh_flat_entry_shards(state)
        else:
            for entry, full in zip(state.true_fsdp_params, full_values):
                padded = common._pad_flat(
                    common._flatten_var(full), entry.padded_numel)
                local = jt.Var.copy(common._slice_flat(
                    padded,
                    int(state.true_fsdp_rank) * int(entry.shard_numel),
                    int(entry.shard_numel),
                )).stop_grad()
                entry.shard.update(local)

        for entry in state.true_fsdp_params:
            if getattr(entry, "requires_grad", True) and entry.shard.is_stop_grad():
                entry.shard.start_grad()
            object.__setattr__(entry.owner, entry.attr, entry.shard)
            entry.full_param = None
        state.true_fsdp_unsharded = False
    for key in set(current) & set(state_dict) - consumed:
        state, entry = shard._fsdp_param_entry(current[key])
        if state is not None and entry is not None:
            # Aliases of an owned parameter have already been loaded above.
            continue
        value = _checkpoint_tensor(state_dict[key], current[key])
        if tuple(value.shape) != tuple(current[key].shape):
            raise RuntimeError("FSDP checkpoint buffer shape mismatch: {}".format(key))
        current[key].assign(value)
    return consumed

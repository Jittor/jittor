"""Private FULL checkpoint transforms for the existing FSDP2 shard owner."""

import numpy as np

import jittor as jt

from . import common, shard, dtensor


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
    if backend == "acl":
        target = "npu:{}".format(device)
    else:
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


def _load_full_state_dict(module, state_dict, strict=True, *, already_replicated=False):
    """Load FULL state into local shards, accepting an already-broadcast payload."""
    current = module.state_dict()
    exported_keys = {}
    for key, value in current.items():
        state, entry = shard._fsdp_param_entry(value)
        if state is not None:
            exported_keys.setdefault(id(entry), []).append((state, key))
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
        full_values = []
        for entry in state.true_fsdp_params:
            # State-dict hooks define the portable namespace. Resolve the key
            # through the exported mapping and the managed parameter identity;
            # reconstructing it from named_modules() would reintroduce wrapper
            # internals such as CheckpointWrapper's private child name.
            key = next((name for owner, name in exported_keys.get(id(entry), ())
                        if owner is state), None)
            if key is None:
                raise KeyError(
                    "missing exported FSDP parameter for state-dict entry: {}".format(entry.name))
            if key not in state_dict:
                raise KeyError(
                    "missing FSDP parameter in full state dict: {}".format(key))
            full = _checkpoint_tensor(state_dict[key], entry.shard)
            if tuple(full.shape) != tuple(entry.shape):
                raise RuntimeError("FSDP checkpoint parameter shape mismatch: {}".format(key))
            # set_model_state_dict is authoritative on rank 0: torch's own
            # pattern fills the full dict on rank 0 only. The caller here does
            # not broadcast, so without this a rank-divergent full dict leaves
            # non-zero ranks writing their stale local values into their shard.
            if common._world_size() > 1 and not already_replicated:
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
            shard._storage._set_module_parameter(entry, entry.shard)
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


def _checkpoint_interval(state, entry):
    """Describe this parameter's valid elements without collective padding."""
    if getattr(state, "true_fsdp_flat", False):
        start = min(max(int(state.true_fsdp_rank) * int(state.true_fsdp_flat_shard_numel)
                        - int(entry.flat_offset), 0), int(entry.numel))
        _, count = shard._flat_local_overlap(state, entry)
    else:
        start = min(int(state.true_fsdp_rank) * int(entry.shard_numel), int(entry.numel))
        count = min(int(entry.shard_numel), int(entry.numel) - start)
    return start, count


def _checkpoint_sharded_tensor(value, state, entry):
    start, count = _checkpoint_interval(state, entry)
    local = common._flatten_var(value)
    if int(np.prod(local.shape, dtype=np.int64)) < count:
        raise ValueError("FSDP checkpoint tensor is shorter than its local interval")
    local = common._slice_flat(local, 0, count).detach().clone()
    return dtensor.DTensor(local, getattr(state, "mesh", None), (dtensor.Shard(0),),
                           checkpoint_shape=tuple(int(d) for d in entry.shape),
                           checkpoint_offset=start)


def _get_sharded_state_dict(module):
    entries = {name: (state, entry) for name, state, entry in _checkpoint_entries(module)}
    result = {}
    for name, value in module.state_dict().items():
        owner = entries.get(name)
        if owner is None:
            state, entry = shard._fsdp_param_entry(value)
            owner = (state, entry) if state is not None else None
        result[name] = (_checkpoint_sharded_tensor(owner[1].shard, *owner)
                        if owner is not None else value.detach().clone())
    return result


def _checkpoint_local_value(value, state, entry, name):
    metadata = getattr(value, "checkpoint_metadata", None)
    start, count = _checkpoint_interval(state, entry)
    expected = (tuple(int(d) for d in entry.shape), start)
    if metadata != expected:
        raise ValueError("FSDP checkpoint global shape/offset mismatch: " + name)
    local = value.to_local()
    if tuple(local.shape) != (count,):
        raise ValueError("FSDP checkpoint local interval shape mismatch: " + name)
    return local


def _load_sharded_state_dict(module, state_dict, strict=True):
    current = module.state_dict()
    missing, unexpected = set(current) - set(state_dict), set(state_dict) - set(current)
    if strict and (missing or unexpected):
        raise RuntimeError("FSDP checkpoint keys mismatch: missing={} unexpected={}".format(sorted(missing), sorted(unexpected)))
    entries = list(_checkpoint_entries(module))
    groups, ordinary = {}, []
    owned = set()
    for name, state, entry in entries:
        owned.add(name)
        if name not in state_dict:
            continue
        local = _checkpoint_local_value(state_dict[name], state, entry, name)
        local = _checkpoint_tensor(local, entry.shard, preserve_dtype=True)
        if str(local.dtype) != str(entry.shard.dtype):
            raise ValueError("FSDP checkpoint parameter dtype mismatch: " + name)
        groups.setdefault(id(state), (state, {}))[1][id(entry)] = local
    for name in set(current) & set(state_dict) - owned:
        state, entry = shard._fsdp_param_entry(current[name])
        if state is not None and entry is not None:
            continue  # Alias of an owned parameter.
        local = state_dict[name]
        if getattr(local, "checkpoint_metadata", None) is not None:
            raise ValueError("ordinary checkpoint buffer unexpectedly sharded: " + name)
        value = _checkpoint_tensor(local, current[name], preserve_dtype=True)
        if tuple(value.shape) != tuple(current[name].shape) or str(value.dtype) != str(current[name].dtype):
            raise ValueError("FSDP checkpoint buffer shape/dtype mismatch: " + name)
        ordinary.append((current[name], value))
    # Prepare every new local shard before changing any model owner.
    pending = []
    for state, values in groups.values():
        with common._frontend_scope(state):
            if getattr(state, "true_fsdp_flat", False):
                pieces = [values.get(id(entry), entry.shard) for entry in state.true_fsdp_params]
                value = jt.concat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
                pending.append((state, None, common._pad_flat(value, state.true_fsdp_flat_shard_numel)))
            else:
                for entry in state.true_fsdp_params:
                    if id(entry) in values:
                        pending.append((state, entry, common._pad_flat(values[id(entry)], entry.shard_numel)))
    for state, entry, value in pending:
        with common._frontend_scope(state):
            if entry is None:
                state.true_fsdp_flat_shard.update(value)
                shard._refresh_flat_entry_shards(state)
            else:
                entry.shard.update(value)
            for current_entry in state.true_fsdp_params:
                shard._storage._set_module_parameter(current_entry, current_entry.shard)
                current_entry.full_param = None
            state.true_fsdp_unsharded = False
    for target, value in ordinary:
        target.assign(value)
    return None


def _get_sharded_optimizer_state_dict(module, state_dict):
    from ..torch.context import get_install_context
    from jittor._core.dtypes import dtype_name
    g = get_install_context(jt).target_namespace
    entries = {name: (state, entry) for name, state, entry in _checkpoint_entries(module)}
    local_metadata = {}
    for name, fields in state_dict["state"].items():
        local_metadata[name] = {}
        for field, value in fields.items():
            if isinstance(value, jt.Var):
                local_metadata[name][field] = (("scalar_tensor", dtype_name(value.dtype), value.item())
                                               if len(value.shape) == 0 else ("tensor", dtype_name(value.dtype)))
            else:
                local_metadata[name][field] = ("value", value)
    dist = g.distributed
    ranks = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
    if len(ranks) > 1:
        dist.all_gather_object(ranks, local_metadata)
    else:
        ranks[0] = local_metadata
    metadata = {}
    for rank in ranks:
        for name, fields in rank.items():
            for field, spec in fields.items():
                previous = metadata.setdefault(name, {}).setdefault(field, spec)
                if previous != spec:
                    raise ValueError("FSDP optimizer checkpoint metadata/scalars differ across ranks: " + name + "." + field)
    result = {}
    for name in sorted(metadata):
        result[name] = {}
        for field, spec in sorted(metadata[name].items()):
            value = state_dict["state"].get(name, {}).get(field)
            if spec[0] == "tensor" and name in entries:
                state, entry = entries[name]
                _, count = _checkpoint_interval(state, entry)
                if value is None:
                    if count:
                        raise ValueError("FSDP optimizer checkpoint is missing a nonempty local state: " + name + "." + field)
                    value = common._zeros_like_shape(entry.shard, (0,), dtype=spec[1]).stop_grad()
                result[name][field] = _checkpoint_sharded_tensor(value, state, entry)
            elif value is not None:
                result[name][field] = value
            elif spec[0] == "scalar_tensor":
                result[name][field] = g.tensor(spec[2], dtype=spec[1], device="cpu")
            elif spec[0] == "value":
                result[name][field] = spec[1]
            else:
                raise ValueError("FSDP optimizer checkpoint is missing ordinary state: " + name + "." + field)
    return {"state": result, "param_groups": state_dict["param_groups"]}


def _local_sharded_optimizer_state_dict(module, state_dict):
    entries = {name: (state, entry) for name, state, entry in _checkpoint_entries(module)}
    result = {}
    for name, fields in state_dict["state"].items():
        result[name] = {}
        for field, value in fields.items():
            if getattr(value, "checkpoint_metadata", None) is not None:
                if name not in entries:
                    raise ValueError("optimizer checkpoint shard has no FSDP owner: " + name)
                state, entry = entries[name]
                local = _checkpoint_local_value(value, state, entry, name + "." + field)
                result[name][field] = common._pad_flat(local, entry.shard_numel)
            else:
                result[name][field] = value
    return {"state": result, "param_groups": state_dict["param_groups"]}

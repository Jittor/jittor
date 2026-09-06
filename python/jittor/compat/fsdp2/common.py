"""Low-level sharding and collective helpers for FSDP2 compatibility."""

import os

import numpy as np

import jittor as jt
from jittor import nn


def _prod(xs):
    out = 1
    for x in xs:
        try:
            out *= int(x)
        except Exception:
            pass
    return out


def _world_size():
    try:
        return int(getattr(jt, "world_size", 1))
    except Exception:
        return 1


def _rank():
    try:
        return int(getattr(jt, "rank", 0))
    except Exception:
        return 0


def _in_true_distributed():
    return _world_size() > 1 and (
        os.environ.get("JT_NCCL_WORLD_SIZE") is not None
        or os.environ.get("OMPI_COMM_WORLD_SIZE") is not None
        or getattr(jt, "in_mpi", False)
    )


def _nccl_ops():
    try:
        ops = getattr(jt.compile_extern, "nccl_ops", None)
        if ops is not None:
            return ops
        if os.environ.get("JT_NCCL_WORLD_SIZE") is not None:
            os.environ.setdefault("use_nccl", "1")
            setup = getattr(jt.compile_extern, "setup_nccl", None)
            if callable(setup):
                setup()
            return getattr(jt.compile_extern, "nccl_ops", None)
    except Exception:
        return None
    return None


def _flatten_var(v):
    return v.reshape((-1,))


def _ceil_div(a, b):
    return (int(a) + int(b) - 1) // int(b)


def _pad_flat(flat, padded_numel):
    n = int(flat.numel()) if callable(getattr(flat, "numel", None)) else int(np.prod(flat.shape))
    if n == int(padded_numel):
        return flat
    pad = jt.zeros((int(padded_numel) - n,), dtype=flat.dtype)
    return jt.concat([flat, pad], dim=0)


def _slice_flat(flat, start, length):
    start = int(start)
    length = int(length)
    return flat[start:start + length]


def _materialize_initial_shard(shard):
    """Detach a persistent shard from its full parameter's device storage."""
    shard = (shard + jt.zeros_like(shard)).stop_grad()
    shard.sync()
    return shard


def _value_requires_grad(value):
    if isinstance(value, jt.Var):
        try:
            return bool(value.requires_grad)
        except (AttributeError, TypeError):
            return not value.is_stop_grad()
    if isinstance(value, dict):
        return any(_value_requires_grad(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_value_requires_grad(item) for item in value)
    return False


def _primary_input_requires_grad(args, kwargs):
    """Conservatively identify whether a module must propagate input grads."""
    if args:
        return _value_requires_grad(args[0])
    for name in (
            "input", "inputs", "hidden_states", "inputs_embeds", "x",
            "input_ids", "pixel_values"):
        if name in kwargs:
            return _value_requires_grad(kwargs[name])
    return any(_value_requires_grad(value) for value in kwargs.values())


def _materialize_frozen_output(value):
    """Sever a completed frozen forward graph while preserving its structure."""
    if isinstance(value, jt.Var):
        return (value + jt.zeros_like(value)).stop_grad()
    if isinstance(value, tuple):
        values = tuple(_materialize_frozen_output(item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*values)
        if type(value) is tuple:
            return values
        try:
            return type(value)(values)
        except TypeError:
            return values
    if isinstance(value, list):
        values = [_materialize_frozen_output(item) for item in value]
        if type(value) is list:
            return values
        try:
            return type(value)(values)
        except TypeError:
            return values
    if isinstance(value, dict):
        values = {
            key: _materialize_frozen_output(item)
            for key, item in value.items()
        }
        if type(value) is dict:
            return values
        try:
            return type(value)(values)
        except TypeError:
            return values
    return value


def _all_gather_shards(local_shard):
    # On one rank the gather is the local shard itself. Say so before reaching
    # for a collective: ``fully_shard`` on a single process is a supported
    # configuration -- it is how the FSDP2 paths in ms-swift and verl run on
    # CPU -- and demanding NCCL there turns a no-op into a hard failure.
    if _world_size() <= 1:
        return local_shard
    ops = _nccl_ops()
    if ops is not None and callable(getattr(ops, "nccl_all_gather", None)):
        return ops.nccl_all_gather(local_shard)
    if callable(getattr(local_shard, "mpi_all_gather", None)):
        return local_shard.mpi_all_gather()
    raise RuntimeError("Jittor NCCL all_gather is not available; launch with jittor.distributed.launch and use_nccl=1")


def _full_gradient_from_shard(gradient, state, entry):
    """Reconstruct one public DTensor gradient from its rank-local shard."""
    if getattr(state, "true_fsdp_flat", False):
        stored = [
            getattr(current.shard, "_torch_grad", None)
            for current in state.true_fsdp_params
        ]
        if any(isinstance(value, jt.Var) for value in stored):
            parts = []
            real_numel = 0
            for current, value in zip(state.true_fsdp_params, stored):
                if current is entry:
                    value = gradient
                if not isinstance(value, jt.Var):
                    value = jt.zeros_like(current.shard)
                part_numel = _param_numel(value)
                if part_numel:
                    parts.append(_flatten_var(value))
                    real_numel += part_numel
            if real_numel < int(state.true_fsdp_flat_shard_numel):
                parts.append(jt.zeros(
                    (int(state.true_fsdp_flat_shard_numel) - real_numel,),
                    dtype=state.true_fsdp_flat_shard.dtype))
            local_flat = parts[0] if len(parts) == 1 else jt.concat(parts, dim=0)
        else:
            local_flat = state.true_fsdp_last_flat_grad
        full_flat = _all_gather_shards(local_flat)
        return _slice_flat(full_flat, entry.flat_offset, entry.numel).reshape(
            entry.shape)
    gathered = _all_gather_shards(_flatten_var(gradient))
    return _slice_flat(gathered, 0, entry.numel).reshape(entry.shape)


def _reduce_scatter_padded(full_grad):
    # Likewise the identity on one rank: nothing to reduce against, and rank 0's
    # shard is the whole padded gradient.
    if _world_size() <= 1:
        return full_grad
    ops = _nccl_ops()
    if ops is not None and callable(getattr(ops, "nccl_reduce_scatter", None)):
        return ops.nccl_reduce_scatter(full_grad)
    # Correct fallback for environments with all_reduce but without native
    # reduce_scatter.  It communicates more than needed, but preserves semantics.
    reduced = full_grad.mpi_all_reduce("sum")
    shard = int(reduced.shape[0]) // max(_world_size(), 1)
    return _slice_flat(reduced, _rank() * shard, shard)


def _param_numel(v):
    return int(np.prod(tuple(int(x) for x in v.shape)))


def _named_parameters_with_owner(module, recurse=True):
    out = []
    seen = set()

    def child_items(mod):
        try:
            items = mod.named_children()
            if items is not None:
                return list(items)
        except Exception:
            pass
        try:
            modules = getattr(mod, "_modules", None)
            if callable(modules):
                modules = modules()
            if isinstance(modules, dict):
                return list(modules.items())
        except Exception:
            pass
        return []

    def visit(mod, prefix=""):
        dc = getattr(mod, "__dict__", {})
        try:
            if isinstance(mod, nn.ParameterList):
                dc = mod.params
        except Exception:
            pass
        bufnames = getattr(mod, "__dict__", {}).get("_buffer_names", ())
        for name, value in list(dc.items()):
            if isinstance(name, str) and name.startswith("_"):
                continue
            if isinstance(value, jt.Var):
                if id(value) in seen:
                    continue
                if getattr(value, "is_buffer", False) or not getattr(value, "persistent", True) or name in bufnames:
                    continue
                seen.add(id(value))
                pname = f"{prefix}.{name}" if prefix else str(name)
                out.append((pname, mod, name, value))
        if recurse:
            for name, value in child_items(mod):
                if isinstance(value, nn.Module):
                    child_prefix = f"{prefix}.{name}" if prefix else str(name)
                    visit(value, child_prefix)

    visit(module)
    return out


def _fsdp2_flat_enabled(world_size, total_numel):
    value = os.environ.get("JITTOR_FSDP2_FLAT", "auto").lower()
    if value in ("0", "false", "no"):
        return False
    if value in ("1", "true", "yes"):
        return True
    # Flat sharding removes several tiny NCCL launches and is consistently
    # faster on 2 ranks. On 4 ranks it helps small models but the extra
    # flatten/slice work slows medium-size cases, so keep the legacy path there.
    return int(world_size) <= 2 or int(total_numel) <= 1_000_000


_EXPORTS = (
    "_prod",
    "_world_size",
    "_rank",
    "_in_true_distributed",
    "_nccl_ops",
    "_flatten_var",
    "_ceil_div",
    "_pad_flat",
    "_slice_flat",
    "_all_gather_shards",
    "_reduce_scatter_padded",
    "_param_numel",
    "_fsdp2_flat_enabled",
)

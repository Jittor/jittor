"""Low-level sharding and collective helpers for FSDP2 compatibility."""

import os
from contextlib import nullcontext
from functools import wraps

import jittor as jt
# Device-local factories and shape calculations are shared with native FSDP.
from jittor.distributed.fsdp.common import (  # noqa: F401 -- internal reexports
    StateRecord, _flatten_var, _ceil_div, _pad_flat, _zeros_like_shape,
    _param_numel, _value_requires_grad, _primary_input_requires_grad,
    _materialize_frozen_output, _tensor_values,
)
from ..diagnostics import EXPECTED, swallowed
# Rank/world queries and the two collectives moved down to jittor/compat/
# collectives.py: compat/torch/installers/distributed.py needs
# `_all_gather_shards` for plain `all_gather`, and borrowing it from here made
# the distributed installer depend on FSDP2. The original WORLD callables are
# re-exported under `_world_*`; the optional-group wrappers below additionally
# route FSDP mesh communicators without changing the shared collective owner.
from ..collectives import (          # noqa: F401
    _all_gather_shards as _world_all_gather_shards,
    _in_true_distributed,
    _nccl_ops,
    _rank,
    _reduce_scatter_padded as _world_reduce_scatter_padded,
    _slice_flat,
    _world_size,
)

def _frontend_scope(state):
    tensor_type = getattr(state, "frontend_type", None)
    if tensor_type is None:
        return nullcontext()
    from ..torch.frontend import tensor_frontend
    # FSDP lifecycle code manipulates already-placed parameters and collective
    # buffers. It must let native operations follow their inputs; applying the
    # Torch factory default (CPU) here makes a CUDA concat/setitem mix devices.
    return tensor_frontend(tensor_type, default_placement=False)

def _state_frontend(function):
    @wraps(function)
    def invoke(state, *args, **kwargs):
        with _frontend_scope(state):
            return function(state, *args, **kwargs)
    return invoke

def _all_gather_shards(value, group=None):
    if group is None or group.ranks is None:
        return _world_all_gather_shards(value)
    if group.rank() < 0:
        raise RuntimeError("FSDP collective called by a nonmember")
    if group.size() == 1:
        return value
    kind = group._get_backend_name()
    ops = getattr(jt.compile_extern, kind + "_ops", None)
    gather = getattr(ops, kind + "_all_gather", None)
    if gather is None or group._backend_handle is None:
        raise RuntimeError("FSDP mesh has no all_gather backend communicator")
    return gather(value, group._backend_handle)

def _reduce_scatter_padded(value, group=None):
    if group is None or group.ranks is None:
        return _world_reduce_scatter_padded(value)
    if group.rank() < 0:
        raise RuntimeError("FSDP collective called by a nonmember")
    if group.size() == 1:
        return value
    if group._get_backend_name() == "nccl":
        return _nccl_ops().nccl_reduce_scatter(value, group._backend_handle)
    reduced = group._all_reduce(value, "sum")
    size = int(reduced.shape[0]) // group.size()
    return _slice_flat(reduced, group.rank() * size, size)

def _prod(xs):
    out = 1
    for x in xs:
        try:
            out *= int(x)
        except EXPECTED as exc:
            swallowed("fsdp2/common.py _prod: out *= int(x)", exc)
    return out

def _full_gradient_from_shard(gradient, state, entry):
    """Reconstruct one public DTensor gradient from its rank-local shard."""
    group = getattr(state, "shard_group", None)
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
                parts.append(_zeros_like_shape(
                    state.true_fsdp_flat_shard,
                    (int(state.true_fsdp_flat_shard_numel) - real_numel,),
                    dtype=state.true_fsdp_flat_shard.dtype))
            local_flat = parts[0] if len(parts) == 1 else jt.concat(parts, dim=0)
        else:
            local_flat = state.true_fsdp_last_flat_grad
        full_flat = _all_gather_shards(local_flat, group)
        return _slice_flat(
            full_flat, entry.flat_offset, entry.numel).reshape(entry.shape)
    gathered = _all_gather_shards(_flatten_var(gradient), group)
    return _slice_flat(gathered, 0, entry.numel).reshape(entry.shape)

#: Where "auto" switches flat sharding off, and how to move it.
#:
#: Flat sharding removes several tiny NCCL launches and was consistently faster
#: on 2 ranks; on 4 ranks it helps small models but the extra flatten/slice
#: work slows medium-size cases. Both numbers come from one set of measurements
#: on one machine, so they are defaults rather than facts: a different
#: interconnect or a different model size moves them, and before this they were
#: literals in the middle of a boolean with no way to try another value short
#: of editing the source. Overriding either variable does not reach for
#: JITTOR_FSDP2_FLAT=1/0, which still forces the answer outright.
_FLAT_MAX_WORLD_SIZE = 2
_FLAT_MAX_NUMEL = 1_000_000

def _flat_threshold(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            "{}={!r} is not an integer. It is the parameter count (or rank "
            "count) at which FSDP2 stops using flat sharding; leave it unset "
            "for {}.".format(name, raw, default))

def _fsdp2_flat_enabled(world_size, total_numel):
    """Whether to shard this module flat, and why.

    ``JITTOR_FSDP2_FLAT`` forces the answer (1/0). Left at "auto", the two
    thresholds above decide, and each is itself overridable -- the point of
    8.11 is that a policy tuned on one machine should be reachable from the
    environment on another, not that these particular numbers are right.
    """
    value = os.environ.get("JITTOR_FSDP2_FLAT", "auto").lower()
    if value in ("0", "false", "no"):
        return False
    if value in ("1", "true", "yes"):
        return True
    max_world = _flat_threshold("JITTOR_FSDP2_FLAT_MAX_WORLD_SIZE",
                                _FLAT_MAX_WORLD_SIZE)
    max_numel = _flat_threshold("JITTOR_FSDP2_FLAT_MAX_NUMEL",
                                _FLAT_MAX_NUMEL)
    return int(world_size) <= max_world or int(total_numel) <= max_numel

# WORLD helpers remain owned below FSDP; the wrappers above add explicit
# process-group routing for mesh shards without changing those shared helpers.
_EXPORTS = (
    "_prod",
    "_flatten_var",
    "_ceil_div",
    "_pad_flat",
    "_param_numel",
    "_primary_input_requires_grad",
    "_materialize_frozen_output",
    "_tensor_values",
    "_full_gradient_from_shard",
    "_fsdp2_flat_enabled",
)

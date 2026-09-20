"""FSDP2 gradient synchronization and optimizer gradient publication."""

import numpy as np

import jittor as jt

from . import common, shard
from ..diagnostics import EXPECTED, swallowed


_EXPORTS = (
    "sync_sharded_grads",
    "_sync_sharded_grads_from_full_grads",
    "_globally_used_grads",
    "_visible_full_grads_from_shards",
    "_local_grad_from_visible_full",
    "_sync_visible_full_grads_to_optimizer",
    "refresh_visible_full_grads",
    "_fsdp_states_from_optimizers",
    "optimizer_has_fsdp_params",
    "optimizer_has_non_fsdp_params",
    "collect_fsdp_full_params_for_backward",
    "fill_fsdp_optimizer_grads_from_grad_map",
)


def sync_sharded_grads(module, loss=None, *, divide_by_world_size=True):
    """Return rank-local sharded gradients for a true-FSDP-managed module.

    The helper computes gradients against gathered full parameters, then
    reduce-scatters each flattened gradient so optimizers can update local shards.
    It is intentionally explicit: Jittor's generic optimizer integration does not
    yet know about FSDP2 sharded parameters.
    """
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        params = list(module.parameters())
        return jt.grad(loss, params) if loss is not None else []
    if loss is None:
        raise ValueError("sync_sharded_grads() requires a loss for true FSDP2")
    has_forward_params = all(getattr(entry, "full_param", None) is not None
                             for entry in state.true_fsdp_params
                             if getattr(entry, "requires_grad", True))
    if not getattr(state, "true_fsdp_unsharded", False) and not has_forward_params:
        shard._unshard_module_params(module)
    full_params = [entry.full_param for entry in state.true_fsdp_params
                   if getattr(entry, "requires_grad", True)]
    if not full_params:
        return [None] * len(state.true_fsdp_params)
    full_grads = jt.grad(loss, full_params)
    grads = iter(full_grads)
    full_grads = [next(grads) if getattr(entry, "requires_grad", True) else None
                  for entry in state.true_fsdp_params]
    sharded = _sync_sharded_grads_from_full_grads(
        state, full_grads, divide_by_world_size=divide_by_world_size)
    # The gathered parameters have served their purpose; holding them is what
    # made a sharded model cost more memory than an unsharded one.
    shard._release_full_params(state)
    return sharded


@common._state_frontend
def _sync_sharded_grads_from_full_grads(state, full_grads, *, divide_by_world_size=True):
    replicate = getattr(state, "replicate_group", None)
    if (getattr(state, "requires_gradient_sync", True)
            and not getattr(state, "requires_all_reduce", True)
            and replicate is not None and replicate.size() > 1):
        raise NotImplementedError(
            "FSDP2 set_requires_all_reduce(False) with a replicate mesh "
            "needs partial-gradient accumulation; reducing only shard "
            "gradients would silently produce incorrect optimizer updates")
    policy = getattr(state, "mp_policy", None)
    reduce_dtype = getattr(policy, "reduce_dtype", None)
    if reduce_dtype is None:
        reduce_dtype = getattr(policy, "param_dtype", None)
    if reduce_dtype is not None:
        full_grads = [grad.cast(reduce_dtype) if grad is not None and grad.dtype != reduce_dtype
                      else grad for grad in full_grads]
    pending = getattr(state, "true_fsdp_pending_full_grads", None)
    if pending is not None and len(pending) != len(full_grads):
        raise RuntimeError("FSDP2 parameter set changed during gradient accumulation")
    if not getattr(state, "requires_gradient_sync", True):
        if pending is None:
            pending = [None] * len(full_grads)
        accumulated = []
        for old, grad in zip(pending, full_grads):
            if grad is None:
                accumulated.append(old)
                continue
            # Materialize locally so pending grads do not retain every prior
            # microbatch's lazy forward/backward graph until final reduction.
            value = jt.Var.copy(old + grad if old is not None else grad).stop_grad()
            value.sync()
            accumulated.append(value)
        state.true_fsdp_pending_full_grads = accumulated
        return [None] * len(full_grads)
    if pending is not None:
        full_grads = [old + grad if old is not None and grad is not None
                      else (old if grad is None else grad)
                      for old, grad in zip(pending, full_grads)]
        state.true_fsdp_pending_full_grads = None
    if replicate is not None and replicate.size() > 1:
        full_grads = [replicate._all_reduce(grad, "mean" if divide_by_world_size
                                          else "sum") if grad is not None else None
                      for grad in full_grads]
    if getattr(state, "true_fsdp_flat", False):
        flat_grad = common._pad_flat(
            jt.concat([common._flatten_var(
                grad if grad is not None else common._zeros_like_shape(
                    entry.shard, entry.shape, dtype=reduce_dtype or entry.dtype))
                for entry, grad in zip(state.true_fsdp_params, full_grads)], dim=0),
            state.true_fsdp_flat_padded_numel,
        )
        flat_shard_grad = common._reduce_scatter_padded(flat_grad, getattr(state, "shard_group", None))
        if divide_by_world_size:
            flat_shard_grad = flat_shard_grad / max(int(state.true_fsdp_world_size), 1)
        if flat_shard_grad.dtype != state.true_fsdp_flat_shard.dtype:
            flat_shard_grad = flat_shard_grad.cast(state.true_fsdp_flat_shard.dtype)
        flat_shard_grad = flat_shard_grad.stop_grad()
        state.true_fsdp_last_flat_grad = flat_shard_grad
        sharded = [
            shard._mark_fsdp_param_var(
                grad.stop_grad(), state, entry, "grad_shard")
            for entry, grad in zip(
                state.true_fsdp_params,
                shard._flat_entry_slices(state, flat_shard_grad))
        ]
        for grad in sharded:
            object.__setattr__(grad, "_fsdp_norm_group", getattr(state, "shard_group", None))
        state.true_fsdp_last_grads = sharded
        return sharded
    sharded = []
    for entry, grad in zip(state.true_fsdp_params, full_grads):
        if grad is None:
            sharded.append(None)
            continue
        flat = common._pad_flat(common._flatten_var(grad), entry.padded_numel)
        shard_grad = common._reduce_scatter_padded(flat, getattr(state, "shard_group", None))
        if divide_by_world_size:
            shard_grad = shard_grad / max(int(state.true_fsdp_world_size), 1)
        if shard_grad.dtype != entry.shard.dtype:
            shard_grad = shard_grad.cast(entry.shard.dtype)
        shard_grad = shard._mark_fsdp_param_var(
            shard_grad.stop_grad(), state, entry, "grad_shard")
        sharded.append(shard_grad)
    for grad in sharded:
        if grad is not None:
            object.__setattr__(grad, "_fsdp_norm_group", getattr(state, "shard_group", None))
    state.true_fsdp_last_grads = sharded
    return sharded


def _globally_used_grads(local_used, state=None):
    if common._world_size() <= 1:
        return list(local_used)
    flags = jt.array(np.asarray(local_used, dtype=np.int32))
    if not callable(getattr(flags, "mpi_all_reduce", None)):
        raise RuntimeError("FSDP2 unused-gradient synchronization requires all_reduce")
    group = getattr(state, "shard_group", None)
    reduced = flags.mpi_all_reduce("sum") if group is None else group._all_reduce(flags, "sum")
    replicate = getattr(state, "replicate_group", None)
    if replicate is not None:
        reduced = replicate._all_reduce(reduced, "sum")
    return [bool(value) for value in np.asarray(reduced.numpy()).reshape(-1)]


def _visible_full_grads_from_shards(state):
    visible = []
    for entry in state.true_fsdp_params:
        full = getattr(entry, "full_param", None)
        visible.append(
            full is not None
            and getattr(entry.owner, entry.attr, None) is full
            and isinstance(getattr(entry.shard, "_torch_grad", None), jt.Var))
    if not any(visible):
        return [None] * len(visible)
    if getattr(state, "true_fsdp_flat", False):
        parts = []
        real_numel = 0
        for entry, used in zip(state.true_fsdp_params, visible):
            grad = getattr(entry.shard, "_torch_grad", None)
            part = grad if used else jt.zeros_like(entry.shard)
            part_numel = common._param_numel(part)
            if part_numel:
                parts.append(common._flatten_var(part))
                real_numel += part_numel
        if real_numel < int(state.true_fsdp_flat_shard_numel):
            parts.append(common._zeros_like_shape(
                state.true_fsdp_flat_shard,
                (int(state.true_fsdp_flat_shard_numel) - real_numel,),
                dtype=state.true_fsdp_flat_shard.dtype))
        local_flat = parts[0] if len(parts) == 1 else jt.concat(parts, dim=0)
        full_flat = local_flat if common._world_size() <= 1 else common._all_gather_shards(local_flat, getattr(state, "shard_group", None))
        return [
            common._slice_flat(full_flat, entry.flat_offset, entry.numel).reshape(entry.shape)
            if used else None
            for entry, used in zip(state.true_fsdp_params, visible)
        ]
    out = []
    for entry, used in zip(state.true_fsdp_params, visible):
        if not used:
            out.append(None)
            continue
        local = getattr(entry.shard, "_torch_grad")
        gathered = local if common._world_size() <= 1 else common._all_gather_shards(local, getattr(state, "shard_group", None))
        gathered = common._slice_flat(common._flatten_var(gathered), 0, entry.numel)
        out.append(gathered.reshape(entry.shape))
    return out


def _local_grad_from_visible_full(state, entry, full_grad):
    flat = common._flatten_var(full_grad)
    if getattr(state, "true_fsdp_flat", False):
        rank_start = int(state.true_fsdp_rank) * int(state.true_fsdp_flat_shard_numel)
        param_start = int(entry.flat_offset)
        param_end = param_start + int(entry.numel)
        overlap_start = max(rank_start, param_start)
        overlap_end = min(
            rank_start + int(state.true_fsdp_flat_shard_numel), param_end)
        start_in_param = max(overlap_start - param_start, 0)
        return jt.Var.copy(common._slice_flat(
            flat, start_in_param, max(overlap_end - overlap_start, 0)))
    padded = common._pad_flat(flat, entry.padded_numel)
    return jt.Var.copy(common._slice_flat(
        padded, int(state.true_fsdp_rank) * int(entry.shard_numel),
        int(entry.shard_numel)))


def _sync_visible_full_grads_to_optimizer(opt):
    for pg in getattr(opt, "param_groups", []):
        params = list(pg.get("params", []))
        grads = pg.get("grads")
        for i, param in enumerate(params):
            state, entry = shard._fsdp_param_entry(param)
            if state is None:
                continue
            full = getattr(entry, "full_param", None)
            if full is None or getattr(entry.owner, entry.attr, None) is not full:
                continue
            full_grad = getattr(full, "_torch_grad", None)
            if not isinstance(full_grad, jt.Var):
                continue
            if grads is None:
                grads = pg["grads"] = [None] * len(params)
            while len(grads) < len(params):
                grads.append(None)
            local = _local_grad_from_visible_full(
                state, entry, full_grad).stop_grad()
            existing = grads[i]
            if isinstance(existing, jt.Var) and list(existing.shape) == list(local.shape):
                existing.update(local)
                local = existing
            object.__setattr__(local, "_fsdp_norm_group", getattr(state, "shard_group", None))
            grads[i] = local
            entry.last_grad = local
            object.__setattr__(entry.shard, "_torch_grad", local)
            object.__setattr__(param, "_torch_grad", local)
            object.__setattr__(opt, "_Optimizer__zero_grad", False)
            try:
                opt._build_grad_map()
            except EXPECTED as exc:
                swallowed("fsdp2/grad_sync.py _sync_visible_full_grads_to_optimizer: opt._build_grad_map()", exc,
                          "the optimizer keeps the grad map from before this sync, so "
                          "step() may apply stale or missing gradients")


def _clear_pending_full_grads_for_optimizer(state, opt):
    pending = getattr(state, "true_fsdp_pending_full_grads", None)
    if pending is None:
        return
    owned = {
        id(entry) for pg in getattr(opt, "param_groups", ())
        for param in pg.get("params", ())
        for owner, entry in (shard._fsdp_param_entry(param),)
        if owner is state and entry is not None
    }
    if not owned:
        return
    pending = [None if id(entry) in owned else grad
               for entry, grad in zip(state.true_fsdp_params, pending)]
    state.true_fsdp_pending_full_grads = pending if any(
        grad is not None for grad in pending) else None


def refresh_visible_full_grads(opt):
    for state in _fsdp_states_from_optimizers([opt]):
        if getattr(opt, "_Optimizer__zero_grad", False):
            _clear_pending_full_grads_for_optimizer(state, opt)
        for entry, full_grad in zip(
                state.true_fsdp_params,
                _visible_full_grads_from_shards(state)):
            full = getattr(entry, "full_param", None)
            if full is not None and isinstance(full_grad, jt.Var):
                full_grad = full_grad.stop_grad()
                existing = getattr(entry, "full_public_grad", None)
                if not isinstance(existing, jt.Var):
                    existing = getattr(full, "_torch_grad", None)
                if isinstance(existing, jt.Var) \
                        and list(existing.shape) == list(full_grad.shape):
                    existing.update(full_grad)
                    full_grad = existing.stop_grad()
                # A visible full gradient is already replicated, even though
                # its parameter is FSDP-managed. Do not reduce its norm again.
                object.__setattr__(full_grad, "_fsdp_norm_group", None)
                object.__setattr__(full, "_torch_grad", full_grad)
                entry.full_public_grad = full_grad


def _fsdp_states_from_optimizers(optimizers):
    states = []
    seen = set()
    for opt in optimizers or ():
        for pg in getattr(opt, "param_groups", []):
            for param in pg.get("params", []):
                state, _ = shard._fsdp_param_entry(param)
                if state is None:
                    continue
                sid = id(state)
                if sid in seen:
                    continue
                seen.add(sid)
                states.append(state)
    return states


def optimizer_has_fsdp_params(opt):
    return bool(_fsdp_states_from_optimizers([opt]))


def optimizer_has_non_fsdp_params(opt):
    for pg in getattr(opt, "param_groups", []):
        for param in pg.get("params", []):
            if not shard.is_fsdp_managed_param(param):
                return True
    return False


def collect_fsdp_full_params_for_backward(optimizers):
    targets = []
    for state in _fsdp_states_from_optimizers(optimizers):
        has_forward_params = all(getattr(entry, "full_param", None) is not None
                                 for entry in state.true_fsdp_params
                                 if getattr(entry, "requires_grad", True))
        if not has_forward_params:
            module = getattr(state, "true_fsdp_module", None)
            if module is not None:
                shard._unshard_module_params(module)
        for entry in state.true_fsdp_params:
            if not getattr(entry, "requires_grad", True):
                continue
            full = getattr(entry, "full_param", None)
            if full is not None:
                targets.append(full)
    return targets


def fill_fsdp_optimizer_grads_from_grad_map(optimizers, grad_by_id, *,
                                            divide_by_world_size=True):
    states = _fsdp_states_from_optimizers(optimizers)
    if not states:
        return False
    entry_grad = {}
    for state in states:
        pending = getattr(state, "true_fsdp_pending_full_grads", None)
        if pending is not None:
            for opt in optimizers or ():
                if getattr(opt, "_Optimizer__zero_grad", True):
                    _clear_pending_full_grads_for_optimizer(state, opt)
            pending = getattr(state, "true_fsdp_pending_full_grads", None)
        # ``_release_full_params`` drops the gathered Var after the first
        # reduce-scatter.  Keep only its identity-to-entry association so a
        # repeated call with the same grad map can still recognize a shared
        # parameter without retaining the full-size parameter itself.
        full_grads = []
        local_used = []
        for entry in state.true_fsdp_params:
            if not getattr(entry, "requires_grad", True):
                local_used.append(False)
                full_grads.append(None)
                continue
            full = getattr(entry, "full_param", None)
            if full is not None:
                full_id = id(full)
                object.__setattr__(entry, "_jittor_fsdp_full_param_id", full_id)
            else:
                full_id = getattr(entry, "_jittor_fsdp_full_param_id", None)
            grad = grad_by_id.get(full_id) if full_id is not None else None
            pending_grad = (pending[len(full_grads)] if pending is not None
                            else None)
            local_used.append(grad is not None or pending_grad is not None)
            if grad is None and pending_grad is None and getattr(state, "requires_gradient_sync", True):
                reference = full if full is not None else entry.shard
                grad = common._zeros_like_shape(
                    reference, entry.shape, dtype=entry.dtype)
            full_grads.append(grad)
        if not getattr(state, "requires_gradient_sync", True):
            _sync_sharded_grads_from_full_grads(
                state, full_grads, divide_by_world_size=divide_by_world_size)
            shard._release_full_params(state)
            continue
        if not any(local_used) and common._world_size() <= 1:
            # This backward pass never reached the state's parameters -- a second
            # optimizer's loss, say, while a sharded model sits idle in the same
            # process. On one rank there is no peer waiting on a collective, so
            # skip it rather than reduce zeros over the shards and overwrite the
            # gradients a previous pass left in the state.
            continue
        globally_used = _globally_used_grads(local_used, state)
        sharded = _sync_sharded_grads_from_full_grads(
            state, full_grads, divide_by_world_size=divide_by_world_size)
        for entry, grad, used in zip(state.true_fsdp_params, sharded, globally_used):
            entry.last_grad = grad if used else None
            if used:
                entry_grad[(id(state), id(entry))] = grad
        # Same here: once this state's gradients are reduce-scattered onto the
        # shards, the full parameters and the full-size gradients hung off them
        # are dead weight until the next forward gathers them again. See
        # shard._release_full_params for the three buffers this frees.
        shard._release_full_params(state)

    filled_entries = set()
    for opt in optimizers or ():
        zero = getattr(opt, "_Optimizer__zero_grad", True)
        for pg in getattr(opt, "param_groups", []):
            grads_list = pg.get("grads")
            if grads_list is None:
                grads_list = pg["grads"] = [None] * len(pg.get("params", []))
            while len(grads_list) < len(pg.get("params", [])):
                grads_list.append(None)
            for i, param in enumerate(pg.get("params", [])):
                state, entry = shard._fsdp_param_entry(param)
                if state is None:
                    continue
                entry_key = (id(state), id(entry))
                if entry_key in filled_entries:
                    stored = getattr(entry.shard, "_torch_grad", None)
                    grads_list[i] = stored
                    object.__setattr__(param, "_torch_grad", stored)
                    continue
                grad = entry_grad.get((id(state), id(entry)))
                if grad is None:
                    continue
                existing = grads_list[i]
                if not isinstance(existing, jt.Var):
                    existing = getattr(param, "_torch_grad", None)
                if isinstance(existing, jt.Var) \
                        and list(existing.shape) == list(grad.shape):
                    if not zero:
                        grad = grad + existing
                    existing.update(grad)
                    stored = existing.stop_grad()
                else:
                    stored = grad.stop_grad()
                object.__setattr__(stored, "_fsdp_norm_group", getattr(state, "shard_group", None))
                grads_list[i] = stored
                object.__setattr__(param, "_torch_grad", stored)
                object.__setattr__(entry.shard, "_torch_grad", stored)
                filled_entries.add(entry_key)
        object.__setattr__(opt, "_Optimizer__zero_grad", False)
        try:
            opt._build_grad_map()
        except EXPECTED as exc:
            swallowed("fsdp2/grad_sync.py fill_fsdp_optimizer_grads_from_grad_map: opt._build_grad_map()", exc)
    for opt in optimizers or ():
        refresh_visible_full_grads(opt)
    return True

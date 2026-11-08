"""Shared full-gradient accumulation and reduce-scatter mathematics."""

import jittor as jt


def _sync_sharded_grads_from_full_grads(state, full_grads, *, divide_by_world_size=True, api):
    common = api.common
    shard = api
    replicate = getattr(state, "replicate_group", None)
    if (
        getattr(state, "requires_gradient_sync", True)
        and not getattr(state, "requires_all_reduce", True)
        and replicate is not None
        and replicate.size() > 1
    ):
        raise NotImplementedError(
            "FSDP2 set_requires_all_reduce(False) with a replicate mesh "
            "needs partial-gradient accumulation; reducing only shard "
            "gradients would silently produce incorrect optimizer updates"
        )
    policy = getattr(state, "mp_policy", None)
    reduce_dtype = getattr(policy, "reduce_dtype", None)
    if reduce_dtype is None:
        reduce_dtype = getattr(policy, "param_dtype", None)
    if reduce_dtype is not None:
        full_grads = [
            grad.cast(reduce_dtype) if grad is not None and grad.dtype != reduce_dtype else grad
            for grad in full_grads
        ]
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
            value = jt.Var.copy(old + grad if old is not None else grad).stop_grad()
            value.sync()
            accumulated.append(value)
        state.true_fsdp_pending_full_grads = accumulated
        return [None] * len(full_grads)
    if pending is not None:
        full_grads = [
            old + grad if old is not None and grad is not None else (old if grad is None else grad)
            for old, grad in zip(pending, full_grads)
        ]
        state.true_fsdp_pending_full_grads = None
    if replicate is not None and replicate.size() > 1:
        full_grads = [
            replicate._all_reduce(grad, "mean" if divide_by_world_size else "sum")
            if grad is not None
            else None
            for grad in full_grads
        ]
    if getattr(state, "true_fsdp_flat", False):
        flat_grad = common._pad_flat(
            jt.concat(
                [
                    common._flatten_var(
                        grad
                        if grad is not None
                        else common._zeros_like_shape(
                            entry.shard, entry.shape, dtype=reduce_dtype or entry.dtype
                        )
                    )
                    for entry, grad in zip(state.true_fsdp_params, full_grads)
                ],
                dim=0,
            ),
            state.true_fsdp_flat_padded_numel,
        )
        flat_shard_grad = common._reduce_scatter_padded(
            flat_grad, getattr(state, "shard_group", None)
        )
        if divide_by_world_size:
            flat_shard_grad = flat_shard_grad / max(int(state.true_fsdp_world_size), 1)
        if flat_shard_grad.dtype != state.true_fsdp_flat_shard.dtype:
            flat_shard_grad = flat_shard_grad.cast(state.true_fsdp_flat_shard.dtype)
        flat_shard_grad = flat_shard_grad.stop_grad()
        state.true_fsdp_last_flat_grad = flat_shard_grad
        sharded = [
            shard._mark_fsdp_param_var(grad.stop_grad(), state, entry, "grad_shard")
            for entry, grad in zip(
                state.true_fsdp_params, shard._flat_entry_slices(state, flat_shard_grad)
            )
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
        shard_grad = shard._mark_fsdp_param_var(shard_grad.stop_grad(), state, entry, "grad_shard")
        sharded.append(shard_grad)
    for grad in sharded:
        if grad is not None:
            object.__setattr__(grad, "_fsdp_norm_group", getattr(state, "shard_group", None))
    state.true_fsdp_last_grads = sharded
    return sharded

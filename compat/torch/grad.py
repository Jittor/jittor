"""Gradient mode and gradient clipping.

Automatic mixed precision used to live here too; it is now owned by
:mod:`jittor.compat.torch.amp` and :mod:`jittor.compat.torch.grad_scaler`, and
re-exported at the bottom of this module so every historical import path keeps
resolving to the same objects.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import numpy as np
from typing import Any, cast
import jittor as jt
from .. import collectives as _collectives


import functools as _functools


class _GradDecoratorCtx:
    """Mimics torch.no_grad/enable_grad: usable as a context manager, a bare
    decorator (@torch.no_grad), and a called decorator (@torch.no_grad())."""

    def __init__(self, scope_factory, func=None):
        self._scope_factory = scope_factory
        self._func = func if callable(func) else None

    def __call__(self, *args, **kwargs):
        # used as @torch.no_grad() returning a decorator, then applied to a func
        if self._func is None and len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            @_functools.wraps(func)
            def wrapped(*a, **k):
                with self._scope_factory():
                    return func(*a, **k)
            return wrapped
        # used as @torch.no_grad (bare): self._func was set at construction
        if self._func is not None:
            with self._scope_factory():
                return self._func(*args, **kwargs)
        raise TypeError("no_grad/enable_grad misuse")

    def __get__(self, obj, objtype=None):
        # Descriptor protocol: when @torch.no_grad wraps a *method*, this instance
        # replaces the method in the class dict. Without __get__, `inst.method`
        # returns this object unbound, so `self` is never passed and the first
        # real arg wrongly binds to the method's `self` (e.g. transformers'
        # @torch.no_grad ConversionOps.convert -> "missing 'input_dict'"). Bind
        # the instance like a normal function descriptor. Plain context-manager
        # instances (no wrapped func) are never class attributes -> return self.
        if self._func is None or obj is None:
            return self
        import types as _types
        return _types.MethodType(self, obj)

    def __enter__(self):
        self._scope = self._scope_factory()
        return self._scope.__enter__()

    def __exit__(self, *exc):
        return self._scope.__exit__(*exc)




def _reduce_norm_group(value, how, group):
    if group is None or group.size() == 1:
        return value
    # Match DTensor NormPartial: pack each gradient's p-power contribution
    # into a vector all-reduce, then root each entry locally. Materialize first
    # so every rank reduces the same completed shard values.
    value.sync()
    if group.rank() < 0:
        raise RuntimeError("gradient norm called by a process-group nonmember")
    if how == "sum":
        return group.all_reduce(value, "sum")
    if group.ranks is None:
        return _collectives._reduce_scalar(value, how)
    kind = group._get_backend_name()
    ops = getattr(jt.compile_extern, kind + "_ops", None)
    gather = getattr(ops, kind + "_all_gather", None)
    if gather is None or group._backend_handle is None:
        raise RuntimeError("gradient norm requires the mesh communicator")
    gathered = gather(value.reshape((1,)), group._backend_handle)
    return getattr(gathered, how)()


def _get_total_norm_device(grads, norm_type=2.0, error_if_nonfinite=False,
                           shard_reduce=False,
                           return_shard_clip_denominator=False):
    """Compute the total norm for gradient Vars on device.

    Match PyTorch's FSDP2 DTensor reduction order: compute each local
    gradient norm, take the p-norm of the rank-local vector, then reduce that
    rank-local total's p-th power once per shard group and take the root.

    FSDP may mix sharded and replicated gradients. Tagged gradients are grouped
    by their owning shard mesh; untagged gradients are counted once. Without
    tags, shard_reduce=True keeps its legacy meaning as a world reduction.
    """
    import math as _math

    grads = [g for g in grads if isinstance(g, jt.Var)]
    if not grads:
        return jt.array(0.0)

    p = float(norm_type)
    acc_dtype = "float64" if any(
        _jittor_dtype_name(g.dtype) == "float64" for g in grads
    ) else "float32"

    tagged_shards = shard_reduce is True and any(
        hasattr(g, "_fsdp_norm_group") for g in grads
    )

    def _reducer_for(grad):
        if tagged_shards:
            group = getattr(grad, "_fsdp_norm_group", None)
            return group if group is not None else False
        return shard_reduce

    def _reducer_key(reducer):
        # FSDP2 can create a distinct ProcessGroup wrapper for each fully_shard
        # unit. Gradients sharded over the same ranks still belong to one norm
        # bucket even when their wrapper and HCCL handle identities differ.
        if reducer is True:
            return ("world",)
        if reducer is False or reducer is None:
            return ("local",)
        if hasattr(reducer, "ranks"):
            try:
                ranks = getattr(reducer, "ranks")
                ranks = None if ranks is None else tuple(int(rank) for rank in ranks)
                if ranks is None:
                    world_size = int(getattr(jt, "world_size", 1))
                    if int(reducer.size()) != world_size:
                        return ("identity", id(reducer))
                    ranks = tuple(range(world_size))
            except (AttributeError, TypeError, ValueError):
                return ("identity", id(reducer))
            reducer_type = type(reducer)
            return ("group", reducer_type.__module__, reducer_type.__qualname__,
                    getattr(reducer, "_backend_kind", None), ranks)
        return ("identity", id(reducer))

    def _across(value, how, reducer):
        if not reducer:
            return value
        if reducer is not True:
            return _reduce_norm_group(value, how, reducer)
        return _collectives._reduce_scalar(value, how)

    def _vector_norm(values):
        values = jt.concat([value.reshape((1,)) for value in values]).cast(acc_dtype)
        if p == float("inf"):
            return values.max()
        if p == float("-inf"):
            return values.min()
        if p == 0.0 or p == 1.0:
            return values.sum()
        if p == 2.0:
            return jt.sqrt((values * values).sum())
        return (values ** p).sum() ** (1.0 / p)

    # Keep p=0 and infinity-norm semantics per logical tensor.
    if p == 0.0:
        tensor_norms = []
        for grad in grads:
            reducer = _reducer_for(grad)
            x = grad.abs() if "complex" in _jittor_dtype_name(grad.dtype) else grad
            local = (x.cast(acc_dtype).abs() != 0).sum()
            tensor_norms.append(
                (_across(local, "max", reducer) != 0).reshape((1,)).cast(acc_dtype)
            )
        total = jt.concat(tensor_norms).sum().cast(acc_dtype)
    elif p == float("inf") or p == float("-inf"):
        tensor_norms = []
        how = "max" if p == float("inf") else "min"
        empty_value = float("-inf") if p == float("inf") else float("inf")
        for grad in grads:
            reducer = _reducer_for(grad)
            x = grad.abs() if "complex" in _jittor_dtype_name(grad.dtype) else grad
            ax = x.cast(acc_dtype).abs()
            local = ((ax.max() if p == float("inf") else ax.min())
                     if int(grad.numel()) else
                     jt.array(empty_value).cast(acc_dtype))
            tensor_norms.append(
                _across(local, how, reducer).reshape((1,))
            )
        values = jt.concat(tensor_norms).cast(acc_dtype)
        total = values.max() if p == float("inf") else values.min()
    else:
        # DTensor keeps each local parameter norm partial until the outer
        # vector norm has run. Materializing that NormPartial then reduces the
        # p-th power of the rank-local total once per shard group. Reducing
        # each parameter's p-th power first is mathematically equivalent but
        # changes FP32 rounding and does not follow Torch's operation order.
        local_norms = [None] * len(grads)
        groups = []
        group_keys = []
        for index, grad in enumerate(grads):
            reducer = _reducer_for(grad)
            x = grad.abs() if "complex" in _jittor_dtype_name(grad.dtype) else grad
            ax = x.cast(acc_dtype).abs()
            if p == 2.0:
                local_norm = jt.sqrt((ax * ax).sum())
            else:
                local_norm = (ax ** p).sum() ** (1.0 / p)

            if not reducer:
                local_norms[index] = local_norm
                continue

            reducer_key = _reducer_key(reducer)
            try:
                group_index = group_keys.index(reducer_key)
                bucket = groups[group_index][1]
            except ValueError:
                bucket = []
                group_keys.append(reducer_key)
                groups.append((reducer, bucket))
            bucket.append(local_norm)

        group_norms = []
        for reducer, bucket in groups:
            local_components = list(bucket)
            # A replicated gradient converted to NormPartial is partitioned
            # by Torch as x / mesh_size**(1/p) before joining the local vector
            # norm. With one shard mesh, include such gradients in that same
            # local vector so the collective still happens only once.
            if len(groups) == 1:
                if reducer is True:
                    mesh_size = max(int(getattr(jt, "world_size", 1)), 1)
                else:
                    mesh_size = max(int(reducer.size()), 1)
                if local_norms and mesh_size > 1:
                    partition_scale = _math.pow(mesh_size, -1.0 / p)
                    local_components.extend(
                        local_norm * partition_scale
                        for local_norm in local_norms if local_norm is not None
                    )
            local_group_norm = _vector_norm(local_components)
            local_group_power = (
                local_group_norm ** p
            ).reshape((1,)).cast(acc_dtype)
            if reducer is True:
                global_group_power = _collectives._reduce_scalar(
                    local_group_power, "sum"
                )
            else:
                global_group_power = _reduce_norm_group(
                    local_group_power, "sum", reducer
                )
            group_norms.append(
                (global_group_power.reshape((-1,))[0] ** (1.0 / p)).reshape((1,))
            )

        total_components = list(group_norms)
        if len(groups) != 1:
            total_components.extend(
                local_norm.reshape((1,))
                for local_norm in local_norms if local_norm is not None
            )
        if len(total_components) == 1 and groups:
            total = total_components[0].reshape(())
        else:
            total = _vector_norm(total_components)

    if error_if_nonfinite:
        total_value = float(total.item())
        if not _math.isfinite(total_value):
            raise RuntimeError(
                "The total norm of order %s for gradients is non-finite, so it "
                "cannot be clipped. To disable this error set "
                "error_if_nonfinite=False." % norm_type
            )
    if return_shard_clip_denominator:
        epsilon = np.float64(1e-6) if acc_dtype == "float64" else np.float32(1e-6)
        return total, total + epsilon
    return total


def _clip_grads_with_norm_device(grads, max_norm, total_norm,
                                 add_epsilon=True):
    """Scale gradient Vars using an already-computed total norm."""
    grads = [g for g in grads if isinstance(g, jt.Var)]
    if not grads:
        return

    acc_dtype = "float64" if _jittor_dtype_name(total_norm.dtype) == "float64" else "float32"
    limit = float(max_norm)
    if limit == float("inf"):
        return
    scalar_type = np.float64 if _jittor_dtype_name(acc_dtype) == "float64" else np.float32
    denominator = (
        total_norm + scalar_type(1e-6) if add_epsilon else total_norm
    )
    raw_coef = scalar_type(limit) / denominator
    coef = jt.minimum(cast(Any, raw_coef), jt.array(float(scalar_type(1.0))).cast(acc_dtype))
    # CUDA fmin-style minimum may select the finite operand for NaN. Torch
    # propagates a NaN total norm into every gradient when errors are disabled.
    coef = jt.ternary(jt.isnan(raw_coef), raw_coef, coef)
    for g in grads:
        # The fast binary path on ACL does not infer a broadcast mask for a
        # typed 0-d Var. Materialize the scalar view at each local gradient
        # shape before multiplying (FSDP2 can expose 1-D parameter shards).
        scale = coef.cast(_jittor_dtype_name(g.dtype)).broadcast(g.shape)
        g.update(g * scale)


def _clip_grad_norm_device(grads, max_norm, norm_type=2.0,
                           error_if_nonfinite=False, shard_reduce=False,
                           norm_grads=None):
    """Clip gradient Vars with a device-resident clipping coefficient.

    The norm helper preserves PyTorch's DTensor reduction order. It reduces
    one rank-local vector norm per shard group, matching NormPartial without a
    collective per parameter tensor.
    The device coefficient avoids a per-step host read unless non-finite errors
    are explicitly requested.
    """
    grads = [g for g in grads if isinstance(g, jt.Var)]
    if not grads:
        return jt.array(0.0)
    norm_grads = grads if norm_grads is None else [
        g for g in norm_grads if isinstance(g, jt.Var)
    ]
    norm_result = _get_total_norm_device(
        norm_grads, norm_type, error_if_nonfinite,
        shard_reduce=shard_reduce,
        return_shard_clip_denominator=bool(shard_reduce),
    )
    if isinstance(norm_result, tuple):
        total, clip_denominator = norm_result
    else:
        total, clip_denominator = norm_result, None
    if clip_denominator is None:
        _clip_grads_with_norm_device(grads, max_norm, total)
    else:
        _clip_grads_with_norm_device(
            grads, max_norm, clip_denominator, add_epsilon=False
        )
    return total


#: The automatic-mixed-precision family moved to :mod:`jittor.compat.torch.amp`
#: when its stubs were replaced by real implementations (this module has an
#: 800-line budget and the family is ~800 lines on its own). Every historical
#: import path keeps working: these are the same objects, not copies.
from .amp import (                     # noqa: E402, F401
    OptState,
    autocast,
    autocast_cache_enabled,
    autocast_configured_dtype,
    autocast_decorator,
    autocast_decrement_nesting,
    autocast_dtype,
    autocast_increment_nesting,
    autocast_is_enabled,
    autocast_nesting,
    amp_definitely_not_available,
    clear_autocast_cache,
    cuda_custom_bwd,
    cuda_custom_fwd,
    custom_bwd,
    custom_fwd,
    is_autocast_available,
    set_autocast_cache_enabled_state,
    set_autocast_dtype_state,
    set_autocast_enabled_state,
    GradScaler,
    _amp_cast,
    _AutocastContext,
    _CpuAutocast,
    _CpuGradScaler,
    _CudaAutocast,
    _CudaGradScaler,
    _GradScaler,
)

"""Native fully sharded training using the shared storage/reduction owner."""

import hashlib
import itertools
import json
import types

import jittor as jt
from jittor import nn
from jittor._core.dtypes import dtype_name

from . import common, shard
from .policy import MixedPrecisionPolicy


_INSTANCES = itertools.count()


def _parameters_with_owner(module):
    result = []
    seen = {}

    def visit(current, prefix=""):
        for name, value, role in current._var_roles():
            if role != "parameter":
                continue
            path = prefix + name
            if id(value) in seen:
                raise ValueError(
                    "native FSDP does not support aliased parameters: {} and {}".format(
                        seen[id(value)], path
                    )
                )
            if isinstance(current, nn.ParameterList):
                raise NotImplementedError(
                    "native FSDP parameter containers need container-aware replacement"
                )
            if getattr(value, "_jittor_fsdp2_state", None) is not None:
                raise ValueError(
                    "native FSDP does not support nested or already-sharded parameters"
                )
            if (
                getattr(value, "_jittor_ddp_state", None) is not None
                or getattr(value, "_jittor_tp_spec", None) is not None
            ):
                raise ValueError("native FSDP cannot wrap DDP or tensor-parallel parameters")
            if getattr(type(value), "_frontend_backend", None) is not None:
                raise TypeError("native FSDP requires native parameters without a tensor frontend")
            seen[id(value)] = path
            result.append((path, current, name, value))
        for name, child in current.named_children():
            visit(child, prefix + name + ".")

    visit(module)
    return result


def _signature(parameters, module, policy):
    values = []
    for name, _, _, value in parameters:
        values.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": dtype_name(value.dtype),
                "requires_grad": not value.is_stop_grad(),
                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            }
        )
    buffers = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": dtype_name(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }
        for name, value in module.named_buffers()
    ]
    return json.dumps(
        {
            "parameters": values,
            "buffers": buffers,
            "precision": vars(policy) if policy is not None else None,
        },
        sort_keys=True,
    ).encode("utf8")


def _agree(state, payload, stage):
    if state.native_world == 1:
        return
    store = state.native_store
    prefix = "native-fsdp/{}/{}/".format(state.native_instance, stage)
    store.set(prefix + str(state.native_rank), payload)
    keys = [prefix + str(rank) for rank in range(state.native_world)]
    store.wait(keys, timeout=state.native_timeout)
    if any(store.get(key) != payload for key in keys):
        raise ValueError("native FSDP rank metadata or initial values differ")


def _wrapped_execute(self, *args, **kwargs):
    from .optimizer import _validate_parameter

    state = self._fsdp_state
    if getattr(state, "native_forward_pending", False):
        raise RuntimeError("native FSDP requires backward before the next forward")
    for entry in state.true_fsdp_params:
        _validate_parameter(entry.shard)
        if getattr(entry.owner, entry.attr) is not entry.shard:
            raise ValueError("native FSDP module parameter binding changed")
    completed = False
    try:
        result = shard._execute_with_true_fsdp(self, self._native_fsdp_execute, *args, **kwargs)
        completed = True
    finally:
        if not completed:
            # The shared lifecycle restored local module attributes. A failed
            # forward has no graph for a later backward to consume.
            shard._release_full_params(state)
    state.native_forward_pending = not jt.flags.no_grad and any(
        entry.requires_grad for entry in state.true_fsdp_params
    )
    if not state.native_forward_pending:
        shard._release_full_params(state)
    return result


def fully_shard(module, *, process_group=None, mixed_precision=None, timeout=60):
    """Shard each native parameter and all-gather it only for the forward.

    Build the optimizer afterwards, using the resulting local parameters.
    Multi-rank training currently supports CUDA/NCCL WORLD, FP32 parameters,
    per-parameter partitions and synchronized backward on every rank. Nested
    wrappers, parameter aliases, tensor parallelism and ZeRO composition are
    rejected. Checkpoint continuation requires the original world size/layout.
    """
    from jittor import distributed as dist

    if not isinstance(module, nn.Module):
        raise TypeError("fully_shard requires a native Module")
    if timeout <= 0:
        raise ValueError("native FSDP timeout must be positive")
    if process_group is not None and process_group is not dist.get_default_group():
        raise NotImplementedError("native FSDP currently requires the WORLD process group")
    if getattr(module, "_fsdp_state", None) is not None:
        raise ValueError("module is already fully sharded")
    parameters = _parameters_with_owner(module)
    if not parameters:
        raise ValueError("native FSDP requires at least one parameter")
    if any(dtype_name(value.dtype) != "float32" for _, _, _, value in parameters):
        raise NotImplementedError("native FSDP currently requires FP32 parameters")
    if mixed_precision is not None:
        if not isinstance(mixed_precision, MixedPrecisionPolicy):
            raise TypeError("mixed_precision must be a native MixedPrecisionPolicy")
        if "bfloat16" in vars(mixed_precision).values() and any(
            jt.core.dispatch_context([value])[0] != "cuda" for _, _, _, value in parameters
        ):
            raise NotImplementedError("native FSDP BF16 computation currently requires CUDA")
    world, rank = dist.get_world_size(), dist.get_rank()
    store = dist.get_default_store()
    if world > 1:
        if not dist.is_initialized() or store is None:
            raise ValueError("multi-rank native FSDP requires an initialized native WORLD Store")
        if dist.get_backend() != "nccl":
            raise NotImplementedError("multi-rank native FSDP currently requires CUDA/NCCL")
        if any(
            jt.core.dispatch_context([value]) != ("cuda", dist.get_device())
            for _, _, _, value in parameters
        ):
            raise ValueError("native FSDP parameters must be on the WORLD CUDA device")
    state = common.StateRecord(
        native_fsdp=True,
        native_world=world,
        native_rank=rank,
        native_device=dist.get_device(),
        native_store=store,
        native_instance=next(_INSTANCES),
        native_timeout=timeout,
        reshard_after_forward=True,
        requires_gradient_sync=True,
        requires_all_reduce=True,
        shard_group=process_group,
        mp_policy=mixed_precision,
        native_forward_pending=False,
    )
    _agree(state, _signature(parameters, module, mixed_precision), "construction")
    shard._init_state(module, state, parameters)
    for (_, _, _, original), entry in zip(parameters, state.true_fsdp_params):
        shard._mark_fsdp_param_var(original, state, entry, "retired")
    object.__setattr__(module, "_fsdp_state", state)
    object.__setattr__(module, "_native_fsdp_execute", module.execute)
    object.__setattr__(module, "execute", types.MethodType(_wrapped_execute, module))
    return module


class FullyShardedDataParallel(nn.Module):
    """Native Module wrapper around :func:`fully_shard` and its local storage."""

    def __init__(self, module, *, process_group=None, mixed_precision=None, timeout=60):
        super().__init__()
        self.module = fully_shard(
            module, process_group=process_group, mixed_precision=mixed_precision, timeout=timeout
        )

    def execute(self, *args, **kwargs):
        return self.module(*args, **kwargs)

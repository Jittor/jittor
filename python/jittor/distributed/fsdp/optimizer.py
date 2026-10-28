"""Native optimizer seams; sharding mathematics stays in the shared owner."""

import json
import weakref

import jittor as jt
from jittor._core.dtypes import dtype_name

from . import gradients, shard
from .api import _agree


def _state(parameter):
    return getattr(parameter, "_jittor_native_fsdp_state", None)


def _validate_parameter(parameter):
    from jittor import distributed as dist

    state = _state(parameter)
    if state is None:
        return
    entry = getattr(parameter, "_jittor_fsdp2_entry", None)
    if (
        entry is None
        or parameter is not entry.shard
        or getattr(parameter, "_jittor_fsdp2_role", None) != "shard"
    ):
        raise ValueError("native FSDP optimizer must be built after sharding from local parameters")
    if (dist.get_rank(), dist.get_world_size()) != (state.native_rank, state.native_world):
        raise ValueError("native FSDP process-group rank/world changed")
    if tuple(parameter.shape) != (entry.shard_numel,) or dtype_name(parameter.dtype) != dtype_name(
        entry.dtype
    ):
        raise ValueError("native FSDP local parameter shape/dtype changed")
    if bool(parameter.is_stop_grad()) == bool(entry.requires_grad):
        raise ValueError("native FSDP parameter trainability changed")
    if (
        getattr(parameter, "_jittor_ddp_state", None) is not None
        or getattr(parameter, "_jittor_tp_spec", None) is not None
    ):
        raise NotImplementedError("native FSDP cannot compose with DDP or tensor parallelism")
    if state.native_world > 1 and jt.core.dispatch_context([parameter]) != (
        "cuda",
        state.native_device,
    ):
        raise ValueError("native FSDP local parameter is not on the WORLD CUDA device")


def _states(parameters):
    states = []
    for parameter in parameters:
        state = _state(parameter)
        if state is not None:
            _validate_parameter(parameter)
            if all(state is not other for other in states):
                states.append(state)
    return states


def _prepare_backward(optimizer, parameters, targets, retain_graph):
    states = _states(parameters)
    if not states:
        return targets, set()
    from jittor.optim import Adam, AdamW, SGD

    if type(optimizer) not in (Adam, AdamW, SGD):
        raise NotImplementedError("native FSDP currently supports native SGD, Adam and AdamW")
    if retain_graph:
        raise NotImplementedError("native FSDP currently requires one backward per forward")
    identifiers = {id(value) for value in parameters}
    if len(identifiers) != len(parameters):
        raise ValueError("native FSDP optimizer parameters must occur once")
    for state in states:
        owner = getattr(state, "native_optimizer_ref", None)
        if owner is not None and owner() is not optimizer:
            raise ValueError("native FSDP parameters require one optimizer owner")
        for entry in state.true_fsdp_params:
            if entry.requires_grad and id(entry.shard) not in identifiers:
                raise NotImplementedError(
                    "native FSDP optimizer must own every trainable parameter in its module"
                )
        if not getattr(state, "native_forward_pending", False):
            raise RuntimeError("native FSDP backward requires a fresh forward")
        groups = []
        for group in optimizer.param_groups:
            names = []
            for parameter in group["params"]:
                entry = getattr(parameter, "_jittor_fsdp2_entry", None)
                names.append(entry.name if _state(parameter) is state else None)
            fields = ("lr", "momentum", "weight_decay", "dampening", "nesterov", "eps", "betas")
            controls = {name: group.get(name, getattr(optimizer, name, None)) for name in fields}
            groups.append(
                {"parameters": names, "controls": controls, "steps": group.get("n_step", 0)}
            )
        signature = {"optimizer": type(optimizer).__name__, "groups": groups}
        _agree(
            state,
            json.dumps(signature, sort_keys=True).encode("utf8"),
            "backward/{}".format(optimizer.n_step),
        )
        state.native_optimizer_ref = weakref.ref(optimizer)
    full_targets = []
    for target in targets:
        state = _state(target)
        if state is None:
            full_targets.append(target)
        else:
            entry = target._jittor_fsdp2_entry
            if entry.full_param is None:
                raise RuntimeError("native FSDP forward parameter is unavailable for backward")
            full_targets.append(entry.full_param)
    return full_targets, {id(value) for value in parameters if _state(value) is not None}


def _finish_backward(parameters, targets, gradients_by_target):
    states = _states(parameters)
    if not states:
        return gradients_by_target
    lookup = {id(target): gradient for target, gradient in zip(targets, gradients_by_target)}
    local = {}
    for state in states:
        full_grads = [lookup.get(id(entry.full_param)) for entry in state.true_fsdp_params]
        reduced = gradients._sync_sharded_grads_from_full_grads(state, full_grads, api=shard)
        roots = [value for value in reduced if value is not None]
        if roots:
            # Complete the exact collective roots before rank-local optimizer
            # graphs are constructed. No host readback or update math is used.
            jt.sync(roots, True, False)
        for entry, gradient in zip(state.true_fsdp_params, reduced):
            if entry.full_param is not None and gradient is not None:
                local[id(entry.full_param)] = gradient
        shard._release_full_params(state)
        state.native_forward_pending = False
    return [
        local.get(id(target), gradient) for target, gradient in zip(targets, gradients_by_target)
    ]


def _assert_step_ready(optimizer):
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    for state in _states(parameters):
        if getattr(state, "native_forward_pending", False):
            raise RuntimeError("native FSDP step requires backward for its pending forward")

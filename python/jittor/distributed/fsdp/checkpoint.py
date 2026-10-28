"""FSDP TensorShard descriptors for the existing native checkpoint owner.

Only valid rank-local intervals are serialized; collective padding is rebuilt
on restore. Buffers, optimizer scalars and layout metadata remain replicated.
"""

from jittor.serialization.sharded import TensorShard, _pointer

from . import common
from .optimizer import _validate_parameter


_FIELDS = ("grads", "m", "values")


def _optimizer_tensor_list(names, group, field):
    values = group[field]
    # Validate before zip can discard surplus entries. The shared checkpoint
    # owner checks restored tensor shapes after these descriptors are decoded.
    if not isinstance(values, (list, tuple)) or len(values) != len(names):
        raise ValueError("native checkpoint optimizer tensor-list length mismatch: " + field)
    return values


def _entries(model):
    result = {}
    for name, parameter in model.named_parameters():
        state = getattr(parameter, "_jittor_native_fsdp_state", None)
        if state is not None:
            _validate_parameter(parameter)
            if getattr(state, "native_forward_pending", False):
                raise RuntimeError("native FSDP checkpoint requires a completed forward/backward")
            result[name] = (state, parameter._jittor_fsdp2_entry)
    return result


def _layout(model):
    result = {}
    for name, (state, entry) in _entries(model).items():
        policy = getattr(state, "mp_policy", None)
        result[name] = {
            "shape": list(entry.shape),
            "dtype": str(entry.dtype),
            "world_size": state.native_world,
            "shard_numel": entry.shard_numel,
            "param_dtype": getattr(policy, "param_dtype", None),
            "reduce_dtype": getattr(policy, "reduce_dtype", None),
            "output_dtype": getattr(policy, "output_dtype", None),
        }
    return result


def _interval(state, entry):
    start = min(state.native_rank * entry.shard_numel, entry.numel)
    count = min(entry.shard_numel, entry.numel - start)
    return start, count


def _descriptor(value, state, entry):
    start, count = _interval(state, entry)
    return TensorShard(common._slice_flat(value, 0, count), entry.shape, start)


def _encode(state, model):
    entries = _entries(model)
    if not entries:
        return
    state["fsdp_layout"] = _layout(model)
    for name, (owner, entry) in entries.items():
        state["model"][name] = _descriptor(state["model"][name], owner, entry)
    saved = state["optimizer"]
    groups = saved["state"]["defaults"]["param_groups"]
    for names, group in zip(saved["param_names"], groups):
        for field in _FIELDS:
            if field not in group:
                continue
            values = _optimizer_tensor_list(names, group, field)
            group[field] = type(values)(
                _descriptor(value, *entries[name]) if name in entries else value
                for name, value in zip(names, values)
            )


def _requests(model, optimizer, manifest, rank, parameter_names):
    entries = _entries(model)
    tensors = manifest["records"][rank]["tensors"]
    requests = {}

    def add(path, name):
        if name not in entries or path not in tensors:
            return
        state, entry = entries[name]
        start, count = _interval(state, entry)
        value = common._zeros_like_shape(entry.shard, (count,)).stop_grad()
        requests[path] = TensorShard(value, entry.shape, start)

    for name in entries:
        add(_pointer("/model", name), name)
    for gi, names in enumerate(parameter_names):
        for field in _FIELDS:
            for pi, name in enumerate(names):
                add("/optimizer/state/defaults/param_groups/{}/{}/{}".format(gi, field, pi), name)
    return requests


def _decode(state, model):
    entries = _entries(model)
    if state.get("fsdp_layout", {}) != _layout(model):
        raise ValueError("native checkpoint FSDP ownership/layout mismatch")

    def local(value, name):
        if name not in entries:
            return value
        if not isinstance(value, TensorShard):
            raise ValueError("native checkpoint FSDP tensor is missing its shard descriptor")
        owner, entry = entries[name]
        start, count = _interval(owner, entry)
        if (
            tuple(value.global_shape) != entry.shape
            or value.offset != start
            or tuple(value.tensor.shape) != (count,)
        ):
            raise ValueError("native checkpoint FSDP interval mismatch")
        return common._pad_flat(value.tensor, entry.shard_numel).stop_grad()

    for name in entries:
        state["model"][name] = local(state["model"][name], name)
    saved = state["optimizer"]
    groups = saved["state"]["defaults"]["param_groups"]
    for names, group in zip(saved["param_names"], groups):
        for field in _FIELDS:
            if field in group:
                values = _optimizer_tensor_list(names, group, field)
                group[field] = type(values)(
                    local(value, name) for name, value in zip(names, values)
                )


def _restore_trainability(model):
    for _, parameter in model.named_parameters():
        state = getattr(parameter, "_jittor_native_fsdp_state", None)
        if state is not None:
            parameter._jittor_fsdp2_entry.requires_grad = not parameter.is_stop_grad()

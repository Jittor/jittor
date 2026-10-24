"""Strict full-state loading into WORLD FSDP2 shards.

Tensor payloads use real device copies. Object collectives carry metadata only.
"""
from collections import namedtuple

import numpy as np

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._core.carrier_protocol import copy_native_tensor

from . import common, shard
from ..torch.context import get_install_context


_LoadResult = namedtuple("_IncompatibleKeys", "missing_keys unexpected_keys")
_VALIDATION_ERRORS = (RuntimeError, ValueError, TypeError, KeyError, NotImplementedError)


def _agree(distributed, error):
    """Every rank raises the same validation error before the next collective."""
    errors = [None] * common._world_size()
    distributed.all_gather_object(errors, error)
    failed = [(rank, value) for rank, value in enumerate(errors) if value is not None]
    if failed:
        message = "; ".join("rank %s: %s" % (rank, value[1]) for rank, value in failed)
        exception = NotImplementedError if any(v[0] == "NotImplementedError" for _, v in failed) else RuntimeError
        raise exception("FSDP full-state load: " + message)


def _error(exc):
    return type(exc).__name__, str(exc)


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


def _optimizer_checkpoint_parameter_names(module):
    names = {}
    for name, state, entry in _checkpoint_entries(module):
        if entry.full_param is not None:
            names[id(entry.full_param)] = name
        if entry.shard is not None:
            names[id(entry.shard)] = name
    return names


def _checkpoint_tensor(value, reference, *, preserve_dtype=False):
    from ..torch.context import get_install_context

    g = get_install_context(jt).target_namespace
    backend, device = jt.core.dispatch_context([reference])
    target = "{}:{}".format(backend, device) if backend == "cuda" else backend
    return g.tensor(value, dtype=value.dtype if preserve_dtype else reference.dtype, device=target).detach()


def _get_full_state_dict(module):
    result = {}
    flat_cache = {}
    for name, value in module.state_dict(keep_vars=True).items():
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


def _load_options(options):
    # Historical internal callers supplied no options and a full rank-0 dict.
    if options is None:
        return True
    required = {"full_state_dict": True, "cpu_offload": False,
                "ignore_frozen_params": False, "keep_submodule_prefixes": True,
                "flatten_optimizer_state_dict": False}
    for name, expected in required.items():
        if getattr(options, name, expected) != expected:
            raise NotImplementedError("FSDP full-state loading does not support %s=%r" %
                                      (name, getattr(options, name)))
    return bool(getattr(options, "broadcast_from_rank0", False))


def _load_plan(module):
    modules = list(module.named_modules())
    names = {id(child): name for name, child in modules}
    targets, states, device_types = [], [], set()
    expected_ranks = tuple(range(common._world_size()))
    for child in shard._iter_fsdp_modules(module, recurse=True):
        state = getattr(child, "_fsdp_state", None)
        if state is None or not getattr(state, "true_fsdp_initialized", False):
            continue
        # Publishing new shards would discard the full Vars retained by a
        # live forward, or pair new weights with pending optimizer gradients.
        # Until that lifecycle is supported, reject during the initial plan
        # consensus, before any checkpoint copy, payload or model mutation.
        if getattr(state, "true_fsdp_unsharded", False) or any(
                getattr(entry, "full_param", None) is not None
                or getattr(entry.shard, "_torch_grad", None) is not None
                for entry in state.true_fsdp_params):
            raise NotImplementedError(
                "full-state loading requires quiescent, resharded parameters "
                "with no live forward parameters or pending gradients")
        mesh = getattr(state, "mesh", None)
        if mesh is None or mesh.ndim != 1 or tuple(int(v) for v in mesh.mesh.reshape(-1)) != expected_ranks:
            raise NotImplementedError("full-state loading requires an ordered WORLD one-dimensional mesh")
        group = getattr(state, "shard_group", None)
        if group is None or group.size() != common._world_size() or (group.ranks is not None and tuple(group.ranks) != expected_ranks):
            raise NotImplementedError("full-state loading requires WORLD sharding")
        if int(state.true_fsdp_rank) != common._rank() or int(state.true_fsdp_world_size) != common._world_size():
            raise RuntimeError("FSDP shard rank/world does not match the communication group")
        if getattr(state, "replicate_group", None) is not None:
            raise NotImplementedError("full-state loading does not support a replication axis")
        device_types.add(mesh.device_type)
        states.append(state)
        for entry in state.true_fsdp_params:
            keys = ["%s.%s" % (names[id(owner)], attr) if names[id(owner)] else attr
                    for owner, attr in shard._entry_owners(entry)]
            targets.append(dict(keys=keys, shape=tuple(entry.shape), dtype=entry.dtype,
                                entry=entry, state=state))
    if not states:
        raise NotImplementedError("full-state loading requires initialized FSDP2 shards")
    if len(device_types) != 1 or next(iter(device_types)) not in ("cuda", "npu", "cpu"):
        raise NotImplementedError("full-state loading requires one supported device type")
    device_type = next(iter(device_types))
    if device_type == "cpu" and common._world_size() > 1:
        raise NotImplementedError("CPU WORLD FSDP full-state transport is not implemented")
    buffers = {}
    for prefix, owner in modules:
        for name, buffer in owner.named_buffers(recurse=False, remove_duplicate=False):
            if name in getattr(owner, "_non_persistent_buffers_set", ()) or not getattr(buffer, "persistent", True):
                continue
            key = "%s.%s" % (prefix, name) if prefix else name
            if id(buffer) not in buffers:
                buffers[id(buffer)] = dict(keys=[], shape=tuple(buffer.shape), dtype=buffer.dtype,
                                           buffer=buffer)
            buffers[id(buffer)]["keys"].append(key)
    targets.extend(buffers.values())
    targets.sort(key=lambda target: target["keys"][0])
    if device_type == "cpu":
        device = "cpu"
    else:
        # A materialized CPU frontend may leave the ambient placement flag at
        # -1 (or a different ordinal). Resolve the actual current accelerator
        # through its canonical public facade, not the tensor factory flag.
        torch = get_install_context(jt).target_namespace
        ordinal = int(getattr(torch, device_type).current_device())
        if ordinal < 0:
            raise RuntimeError("FSDP full-state load has no current %s device" % device_type)
        device = "%s:%s" % (device_type, ordinal)
    return targets, states, device


def _manifest(state_dict):
    if not isinstance(state_dict, dict):
        raise TypeError("state_dict must be a dictionary")
    result = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not jt.core._is_tensor_carrier(value):
            raise TypeError("full state requires string keys and tensor values")
        result[key] = (tuple(int(v) for v in value.shape), dtype_name(value.dtype))
    return result


def _validate_manifest(targets, manifest, strict=True):
    expected = {key: target for target in targets for key in target["keys"]}
    missing, extra = sorted(set(expected) - set(manifest)), sorted(set(manifest) - set(expected))
    if strict and (missing or extra):
        raise RuntimeError("missing keys=%r, unexpected keys=%r" % (missing, extra))
    for key, target in expected.items():
        if key not in manifest:
            continue
        if tuple(manifest[key][0]) != target["shape"]:
            raise RuntimeError("shape mismatch for %s: %r != %r" %
                               (key, manifest[key][0], target["shape"]))
    return missing, extra


def _check_device(value, device):
    value.sync()
    if device == "cpu":
        if value.location() != "cpu":
            raise RuntimeError("full-state copy is not on CPU")
    else:
        kind, ordinal = device.split(":")
        backend, index = jt.core.dispatch_context([value])
        allowed = ("acl", "acl_legacy") if kind == "npu" else ("cuda",)
        if (value.location() != "device" or int(value.device_id) != int(ordinal)
                or backend not in allowed or int(index) != int(ordinal)):
            raise RuntimeError("full-state copy is not on requested device %s" % device)


def _slice_load_shard(flat, start, length):
    """Take one contiguous state-load shard without ACL SliceV2.

    CANN 9's ``aclnnSliceV2`` reports one element too few for sufficiently
    large tail slices (for example 269820162[134910081:269820162]).  Its static
    Jittor output still has the requested shape, so the workspace query fails
    instead of returning a short tensor.  The ACL backend's native
    SplitWithSize kernel has the exact contiguous-partition semantics needed
    here and handles that boundary correctly.  Keep the workaround local to
    state loading; general indexing and the CUDA/CPU paths retain SliceV2.
    """
    start, length = int(start), int(length)
    total = int(flat.shape[0])
    if start < 0 or length < 0 or start + length > total:
        raise RuntimeError(
            "invalid full-state shard [%s:%s] for flat length %s" %
            (start, start + length, total)
        )
    backend, _device_id = jt.core.dispatch_context([flat])
    if backend not in ("acl", "acl_legacy") or length == 0:
        return common._slice_flat(flat, start, length)
    if start == 0 and length == total:
        return flat
    sections = []
    if start:
        sections.append(start)
    selected = len(sections)
    sections.append(length)
    suffix = total - start - length
    if suffix:
        sections.append(suffix)
    # The ACL split kernel is intentionally inference/no-grad only. State
    # loading publishes detached parameter storage, so this is the correct
    # semantic boundary as well as the backend dispatch condition.
    with jt.flag_scope(no_grad=1):
        return jt.split(flat, sections, dim=0)[selected]


def _load_full_state_dict(module, state_dict, strict=True, *, options=None):
    """Load FULL tensors; retain upstream non-strict fill and local safety gates."""
    strict = bool(getattr(options, "strict", strict))
    torch = get_install_context(jt).target_namespace
    distributed = torch.distributed
    error = None
    try:
        broadcast = _load_options(options)
        targets, states, device = _load_plan(module)
    except _VALIDATION_ERRORS as exc:
        error = _error(exc)
    _agree(distributed, error)

    signature = [(target["keys"], target["shape"], dtype_name(target["dtype"])) for target in targets]
    signatures = [None] * common._world_size()
    distributed.all_gather_object(signatures, (signature, broadcast, strict))
    _agree(distributed, None if all(item == signatures[0] for item in signatures) else
           ("RuntimeError", "rank model schemas or loading modes differ"))

    envelope = [None]
    if not broadcast or common._rank() == 0:
        try:
            envelope[0] = (None, _manifest(state_dict))
        except _VALIDATION_ERRORS as exc:
            envelope[0] = (_error(exc), None)
    if broadcast:
        distributed.broadcast_object_list(envelope, src=0)
    error, manifest = envelope[0]
    if error is None:
        try:
            missing, unexpected = _validate_manifest(targets, manifest, strict)
        except _VALIDATION_ERRORS as exc:
            error = _error(exc)
    _agree(distributed, error)

    # Preserve the upstream non-strict contract: retain absent FULL values.
    # Every rank must enter the same gather sequence, even when local
    # broadcast=False dictionaries have different sets of missing keys.
    missing_by_rank = [None] * common._world_size()
    distributed.all_gather_object(missing_by_rank, bool(missing))
    retained = _get_full_state_dict(module) if any(missing_by_rank) else {}

    full_values = {}
    for target in targets:
        # Module traversal order (not dict insertion order) determines which
        # alias wins if a checkpoint gives different values for shared weights.
        provided = [key for key in target["keys"] if key in manifest]
        key = provided[-1] if provided else target["keys"][-1]
        error = None
        try:
            if not broadcast or common._rank() == 0:
                source = state_dict if provided else retained
                value = source[key].to(device=device, dtype=target["dtype"]).detach().clone()
            else:
                value = torch.empty(target["shape"], dtype=target["dtype"], device=device)
            _check_device(value, device)
            if dtype_name(value.dtype) != dtype_name(target["dtype"]):
                raise RuntimeError("checkpoint copy did not preserve the destination parameter dtype")
        except _VALIDATION_ERRORS as exc:
            error = _error(exc)
        _agree(distributed, error)
        if broadcast:
            distributed.broadcast(value, src=0)
        error = None
        try:
            _check_device(value, device)
        except _VALIDATION_ERRORS as exc:
            error = _error(exc)
        _agree(distributed, error)
        full_values[id(target)] = value

    # Hardware failure during mutation cannot be rolled back, but it must not
    # let another rank report a successful load. Validation itself was staged.
    error = None
    try:
        _apply_full_values(targets, states, full_values, device)
    except _VALIDATION_ERRORS as exc:
        error = _error(exc)
    _agree(distributed, error)
    return _LoadResult(missing, unexpected)


def _apply_full_values(targets, states, full_values, device):
    """Publish validated device tensors using the canonical shard lifecycle."""
    for state in states:
        state_targets = [target for target in targets if target.get("state") is state]
        by_entry = {id(target["entry"]): full_values[id(target)] for target in state_targets}
        # Torch-facing disabling is reversible and may leave native graph
        # edges live, so its public flag cannot be inferred from stop_grad.
        trainability = [shard._parameter_requires_grad(entry.shard)
                        for entry in state.true_fsdp_params]
        if getattr(state, "true_fsdp_flat", False):
            if not state.true_fsdp_params:
                continue
            flat = common._pad_flat(jt.concat([common._flatten_var(by_entry[id(entry)])
                                             for entry in state.true_fsdp_params], dim=0),
                                    state.true_fsdp_flat_padded_numel)
            local = copy_native_tensor(_slice_load_shard(
                flat, int(state.true_fsdp_rank) * int(state.true_fsdp_flat_shard_numel),
                int(state.true_fsdp_flat_shard_numel))).stop_grad()
            state.true_fsdp_flat_shard.update(local)
            _check_device(state.true_fsdp_flat_shard, device)
            for entry, value in zip(state.true_fsdp_params,
                                    shard._flat_entry_slices(state, state.true_fsdp_flat_shard)):
                # Loading must retain Parameter objects already held by an
                # optimizer. Rebind storage; do not assign through old views.
                entry.shard._update(value)
            (state.true_fsdp_flat_shard.start_grad() if any(trainability)
             else state.true_fsdp_flat_shard.stop_grad())
        else:
            for entry in state.true_fsdp_params:
                padded = common._pad_flat(common._flatten_var(by_entry[id(entry)]), entry.padded_numel)
                local = copy_native_tensor(_slice_load_shard(
                    padded, int(state.true_fsdp_rank) * int(entry.shard_numel),
                    int(entry.shard_numel))).stop_grad()
                # Singleton shards retain the original full Parameter holder
                # and public shape; only transport above uses flat storage.
                # Multi-rank local shards keep their existing flat layout.
                if int(state.true_fsdp_world_size) == 1:
                    local = local.reshape(entry.shape)
                entry.shard.update(local)
        for entry, trainable in zip(state.true_fsdp_params, trainability):
            entry.requires_grad = trainable
            entry.shard.start_grad() if trainable else entry.shard.stop_grad()
            entry.shard.requires_grad = trainable
            _check_device(entry.shard, device)
            shard._bind_entry(entry, entry.shard)
            entry.full_param = None
        state.true_fsdp_unsharded = False
    for target in targets:
        if "buffer" in target:
            target["buffer"].update(full_values[id(target)])
            _check_device(target["buffer"], device)

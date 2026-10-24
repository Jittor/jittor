"""Detached FSDP export aliases, not an owned/genuine DTensor implementation.

This preserves the existing legacy tagged-carrier protocol.  Actual Python
types/bases are unchanged and remain a separately failing DTensor contract.
Never retain a live FSDP StateRecord/entry or resolve its current parameters.
"""
from contextlib import nullcontext
from dataclasses import dataclass, replace
import types
import weakref

import jittor as jt

from . import common


_SNAPSHOT = "_jittor_fsdp2_detached_snapshot"


@dataclass(frozen=True, eq=False)
class _DetachedSnapshot:
    local: object
    gather_input: object
    group: object
    mesh: object
    placements: tuple
    shape: tuple
    offset: int
    numel: int
    gather: bool
    flat: bool
    frontend_type: object


def _scope(snapshot):
    if snapshot.frontend_type is None:
        return nullcontext()
    from ..torch.frontend import tensor_frontend
    return tensor_frontend(snapshot.frontend_type, default_placement=False)


class _DetachedTensorMethod:
    """No holder -> bound-method -> holder cycle; only a weak owner link."""
    __slots__ = ("function", "snapshot", "owner")

    def __init__(self, function, snapshot, owner):
        self.function = function
        self.snapshot = snapshot
        self.owner = weakref.ref(owner)

    def __call__(self, *args, **kwargs):
        if type(self.snapshot) is not _DetachedSnapshot:
            raise TypeError("unexpected detached FSDP snapshot metadata")
        independent = (self.function is _snapshot_to_local
                       or self.function is _snapshot_full_tensor)
        if not independent and self.function is not _snapshot_redistribute:
            raise TypeError("unexpected detached FSDP snapshot method")
        owner = self.owner()
        if owner is None:
            if not independent:
                raise ReferenceError("the detached FSDP tensor has been released")
        elif getattr(owner, _SNAPSHOT, None) is not self.snapshot:
            raise RuntimeError("detached FSDP owner snapshot metadata changed")
        return self.function(self.snapshot, owner, *args, **kwargs)


def _snapshot_to_local(snapshot, owner, *args, **kwargs):
    if args or kwargs:
        raise TypeError("detached FSDP to_local currently takes no arguments")
    return snapshot.local


def _snapshot_full_tensor(snapshot, owner, *args, **kwargs):
    if args or kwargs:
        raise TypeError("detached FSDP full_tensor currently takes no arguments")
    with _scope(snapshot), jt.flag_scope(no_grad=1):
        if not snapshot.gather:
            return snapshot.local
        gathered = common._all_gather_shards(snapshot.gather_input, snapshot.group)
        return common._slice_flat(
            gathered, snapshot.offset, snapshot.numel).reshape(snapshot.shape).stop_grad()


def _snapshot_redistribute(snapshot, owner, device_mesh=None, placements=None, **kwargs):
    if kwargs or device_mesh is not None and device_mesh is not snapshot.mesh:
        raise NotImplementedError("detached FSDP redistribution to another mesh")
    if placements is not None:
        requested = tuple(placements)
        same = len(requested) == len(snapshot.placements) and all(
            type(a) is type(b) and vars(a) == vars(b)
            for a, b in zip(requested, snapshot.placements))
        if not same:
            raise NotImplementedError("detached FSDP redistribution to another placement")
    return owner


def _detach_fsdp_tensor(source, out, native_detach):
    """Decorate a true native storage-sharing detach result, with fixed aliases.

    ``native_detach`` is the lower layer's captured native descriptor.  Calling
    it avoids recursing through this hook or changing the source's autograd
    state.  This helper performs no collective until ``full_tensor()``.
    """
    previous = getattr(source, _SNAPSHOT, None)
    if previous is not None:
        if type(previous) is not _DetachedSnapshot:
            raise TypeError("unexpected detached FSDP snapshot metadata")
        with _scope(previous):
            local = native_detach(out).stop_grad()
        snapshot = replace(previous, local=local,
                           gather_input=previous.gather_input if previous.flat else local)
    else:
        state = getattr(source, "_jittor_fsdp2_state", None)
        entry = getattr(source, "_jittor_fsdp2_entry", None)
        if state is None or not getattr(state, "true_fsdp_initialized", False):
            return out
        role = getattr(source, "_jittor_fsdp2_role", None)
        # Parameter/state_dict scope only. Gradient snapshot packing has a
        # distinct flat-gradient allocation contract and is not guessed here.
        if role not in ("shard", "full", "flat_shard"):
            return out
        if role != "flat_shard" and entry is None:
            raise RuntimeError("FSDP parameter detach lacks its initialized entry")
        flat = bool(getattr(state, "true_fsdp_flat", False)) and role != "full"
        gather = role != "full" and int(state.true_fsdp_world_size) > 1
        with common._frontend_scope(state):
            local = native_detach(out).stop_grad()
            # Hold the old allocation, not state.true_fsdp_flat_shard by name.
            # A flat parameter's local intersection can be empty and its
            # length differs across ranks: gathering that intersection is wrong.
            gather_input = (native_detach(state.true_fsdp_flat_shard).stop_grad()
                            if flat else local)
        if role == "flat_shard":
            offset, numel = 0, int(state.true_fsdp_flat_padded_numel)
            shape = (numel,)
        else:
            offset = int(entry.flat_offset) if flat else 0
            numel = int(entry.numel)
            shape = tuple(int(dim) for dim in entry.shape)
        snapshot = _DetachedSnapshot(
            local=local, gather_input=gather_input,
            group=getattr(state, "shard_group", None),
            mesh=getattr(source, "_dtensor_device_mesh", getattr(state, "mesh", None)),
            placements=tuple(getattr(source, "_dtensor_placements", ())),
            shape=shape, offset=offset, numel=numel, gather=gather, flat=flat,
            frontend_type=getattr(state, "frontend_type", None))
    return _publish_detached_snapshot(out, snapshot)


def _publish_detached_snapshot(out, snapshot):
    # Do not propagate state/entry/role/_is_param. Detached leaves are not
    # registered live FSDP parameters and must not enter optimizer routing.
    object.__setattr__(out, _SNAPSHOT, snapshot)
    object.__setattr__(out, "_dtensor_device_mesh", snapshot.mesh)
    object.__setattr__(out, "_dtensor_placements", snapshot.placements)
    object.__setattr__(out, "device_mesh", snapshot.mesh)
    object.__setattr__(out, "placements", snapshot.placements)
    object.__setattr__(out, "_spec", types.SimpleNamespace(
        mesh=snapshot.mesh, placements=snapshot.placements))
    object.__setattr__(out, "to_local", _DetachedTensorMethod(_snapshot_to_local, snapshot, out))
    object.__setattr__(out, "full_tensor", _DetachedTensorMethod(_snapshot_full_tensor, snapshot, out))
    object.__setattr__(out, "redistribute", _DetachedTensorMethod(_snapshot_redistribute, snapshot, out))
    return out


def _to_detached_fsdp_tensor(source, args, kwargs, plain_to):
    """Convert one detached export while preserving its real local/flat alias.

    ``plain_to`` is the original Tensor.to parser and copy implementation.
    Only detached snapshots enter here; native copying/placement stays owned
    by existing Tensor.to and the existing real native view implementation.
    """
    previous = getattr(source, _SNAPSHOT, None)
    if type(previous) is not _DetachedSnapshot:
        raise TypeError("unexpected detached FSDP snapshot metadata")
    if (getattr(source, "_jittor_fsdp2_state", None) is not None
            or getattr(source, "_jittor_fsdp2_entry", None) is not None
            or bool(source.requires_grad)):
        raise RuntimeError("detached FSDP conversion received a live parameter")
    from jittor._core.carrier_protocol import native_api
    with _scope(previous), jt.flag_scope(no_grad=1):
        if previous.flat and previous.gather:
            # Convert exactly ONE complete rank-local flat allocation. Copying
            # its entry-local intersection independently would break writes
            # through to_local(): full_tensor() reads this gather_input.
            original = previous.gather_input
            converted = plain_to(original, *args, **kwargs)
            if converted is original:
                return source
            shard_size = int(original.numel())
            rank = int(previous.group.rank()) if previous.group is not None else common._rank()
            first = rank * shard_size
            local_start = min(max(int(previous.offset) - first, 0), shard_size)
            local_stop = min(max(int(previous.offset) + int(previous.numel) - first, 0), shard_size)
            local_size = local_stop - local_start
            if local_size != int(previous.local.numel()):
                raise RuntimeError("detached flat snapshot has inconsistent local extent")
            local = common._slice_flat(converted, local_start, local_size).reshape(previous.local.shape)
            gather_input = converted
        else:
            original = previous.local
            local = plain_to(original, *args, **kwargs)
            if local is original:
                return source
            gather_input = local
        # A distinct real plain alias is the public result; snapshot.local
        # remains a plain tensor rather than acquiring its owner's DTensor tag.
        native_detach = native_api(type(local), local)["detach"]
        out = native_detach(local).stop_grad()
        snapshot = replace(previous, local=local, gather_input=gather_input)
    return _publish_detached_snapshot(out, snapshot)

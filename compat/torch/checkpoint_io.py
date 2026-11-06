"""Minimal filesystem ``torch.distributed.checkpoint`` save/load.

The job this serves is the official 3D-parallel example, which ends with
``dcp.save({"app": AppState(model, optimizer)})``.  Before this module those
entry points were explicit no-ops: ``save`` returned success without writing a
byte, so a completed training run silently lost its checkpoint.

What is implemented, and what each rank writes
----------------------------------------------
Each rank resolves its own ``Stateful`` objects, flattens the resulting nested
mapping to dotted keys (the same key space torch's planner uses), and writes
one file per key under ``<checkpoint_id>/<rank>/``.  Rank 0 writes the global
``<checkpoint_id>/.metadata`` index.  That is the shape of torch's default
filesystem checkpoint: a ``.metadata`` file plus one numeric directory per
rank.  The payload codec is ``torch.save``/``torch.load``; the metadata codec
is :mod:`pickle`.

DTensor values are stored as their rank-local shard together with the mesh and
placement metadata, so a load can rebuild them instead of pretending a shard
is a full tensor.  FSDP-managed parameter views are stored as plain tensors:
their local bytes are the complete state that rank owns.

Deliberately not implemented
----------------------------
Planner objects, caller-supplied storage readers/writers, asynchronous
staging, and deduplication are refused with ``NotImplementedError`` before any
file is touched.  Publishing an empty checkpoint is worse than failing, which
is exactly the behaviour this module replaces.
"""

from collections.abc import Mapping
import os
import pickle
import sys

import jittor as jt

from .context import get_install_context
from .. import fsdp_hooks

_FORMAT = "jittor-dcp"
_VERSION = 1
_METADATA_NAME = ".metadata"
_SUFFIX = ".distcp"
_SAVE_KWARGS = frozenset(("async_staging", "dedup_save"))
_LOAD_KWARGS = frozenset(("dedup_load",))


class Stateful:
    """Protocol for objects that checkpoint themselves.

    Mirrors ``torch.distributed.checkpoint.stateful.Stateful``: ``dcp.save``
    calls ``state_dict()`` on any nested instance and ``load`` hands the
    reconstructed nested mapping to ``load_state_dict``.
    """

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError


def _torch():
    return get_install_context(jt).target_namespace


def _dist():
    return _torch().distributed


def _context(no_dist, process_group):
    """Return ``(world_size, rank)`` for this rank.

    Only the default process group is supported: the on-disk layout is
    per-global-rank, so an arbitrary subgroup would need its own rank mapping
    that the default torch planner also does not provide.
    """
    if process_group is not None:
        raise NotImplementedError(
            "torch.distributed.checkpoint only supports the default process "
            "group; process_group= would need a subgroup-to-global-rank "
            "mapping that this implementation does not define")
    dist = _dist()
    if no_dist or not dist.is_initialized():
        return 1, 0
    return dist.get_world_size(), dist.get_rank()


def _reject(kwargs, storage, planner, operation, allowed):
    if planner is not None:
        raise NotImplementedError(
            "torch.distributed.checkpoint.%s: planner= is not implemented; "
            "this compat implements the default per-rank filesystem plan" % operation)
    if storage is not None:
        raise NotImplementedError(
            "torch.distributed.checkpoint.%s: caller-supplied storage is not "
            "implemented; use checkpoint_id= and the built-in filesystem "
            "layout" % operation)
    unknown = set(kwargs) - allowed
    if unknown:
        raise TypeError("torch.distributed.checkpoint.%s() got unexpected "
                        "keyword argument(s): %s"
                        % (operation, ", ".join(sorted(unknown))))
    unsupported = sorted(name for name, value in kwargs.items()
                         if value not in (None, False))
    if unsupported:
        raise NotImplementedError(
            "torch.distributed.checkpoint.%s: unsupported option(s): %s"
            % (operation, ", ".join(unsupported)))


def _is_stateful(value):
    """True for the deployed torch protocol *and* this module's fallback.

    The shim deploys torch's real ``checkpoint/stateful.py`` (a runtime
    ``Protocol``), so application classes subclass that one, not the class
    defined here.  Check both rather than changing what users import.
    """
    if isinstance(value, Stateful):
        return True
    module = sys.modules.get("torch.distributed.checkpoint.stateful")
    protocol = getattr(module, "Stateful", None) if module is not None else None
    return protocol is not None and isinstance(value, protocol)


def _placements(value):
    return getattr(value, "_dtensor_placements", None)


def _encode_placements(placements):
    encoded = []
    for placement in placements or ():
        if getattr(placement, "is_shard", None) and placement.is_shard():
            encoded.append({"type": "shard", "dim": int(placement.dim)})
        elif getattr(placement, "is_replicate", None) and placement.is_replicate():
            encoded.append({"type": "replicate"})
        else:
            raise NotImplementedError(
                "torch.distributed.checkpoint cannot store DTensor placement %r"
                % (placement,))
    return encoded


def _dtensor_api():
    provider = fsdp_hooks.provider()
    if provider is None:
        raise RuntimeError(
            "torch.distributed.checkpoint: this checkpoint stores DTensor "
            "mesh/placement metadata, which needs jittor.compat.fsdp2; import "
            "torch with the distributed compatibility surface enabled")
    return provider


def _decode_placements(encoded):
    fsdp2 = _dtensor_api()
    decoded = []
    for item in encoded:
        if item["type"] == "shard":
            decoded.append(fsdp2.Shard(int(item["dim"])))
        elif item["type"] == "replicate":
            decoded.append(fsdp2.Replicate())
        else:
            raise NotImplementedError(
                "unknown DTensor placement in checkpoint metadata: %r" % (item,))
    return tuple(decoded)


def _encode_mesh(mesh):
    if mesh is None:
        return None
    ranks = getattr(mesh, "mesh", None)
    if ranks is None:
        return None
    flat = [int(rank) for rank in list(ranks)]
    return {
        "device_type": getattr(mesh, "device_type", "cuda"),
        "ranks": flat,
        "mesh_dim_names": list(getattr(mesh, "mesh_dim_names", None) or ()),
    }


def _decode_mesh(encoded):
    if not encoded:
        return None
    fsdp2 = _dtensor_api()
    return fsdp2.DeviceMesh(
        encoded.get("device_type", "cuda"), encoded.get("ranks"),
        mesh_dim_names=tuple(encoded.get("mesh_dim_names") or ()) or None)


def _leaf_metadata(value):
    """Describe one flat state-dict leaf and return the object to save."""
    placements = _placements(value)
    if placements is not None and not isinstance(value, jt.Var):
        # A wrapper-style DTensor, not the marked-Var representation.
        local = value.to_local()
        return local, {
            "kind": "dtensor",
            "placements": _encode_placements(placements),
            "mesh": _encode_mesh(getattr(value, "device_mesh", None)),
            "global_shape": [int(size) for size in value.size()],
        }
    if isinstance(value, jt.Var):
        if getattr(value, "_jittor_fsdp2_entry", None) is not None:
            # FSDP2 hands out the shard view that this rank actually owns.
            # Recording it as a DTensor would claim a placement the bytes do
            # not have, so the local tensor is the honest representation.
            return value, {
                "kind": "tensor",
                "dtype": str(value.dtype),
                "shape": [int(size) for size in value.shape],
            }
        if placements is not None:
            global_shape = getattr(value, "_dtensor_global_shape", None)
            if global_shape is None:
                global_shape = tuple(value.shape)
            return value, {
                "kind": "dtensor",
                "dtype": str(value.dtype),
                "shape": [int(size) for size in global_shape],
                "global_shape": [int(size) for size in global_shape],
                "placements": _encode_placements(placements),
                "mesh": _encode_mesh(getattr(value, "device_mesh", None)),
            }
        return value, {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": [int(size) for size in value.shape],
        }
    return value, {"kind": "value"}


def _flatten(state_dict):
    """Resolve ``Stateful`` objects and flatten nested mappings to dotted keys."""
    flat = {}

    def visit(value, prefix):
        if _is_stateful(value):
            value = value.state_dict()
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, prefix + str(key) + ".")
        else:
            if not prefix:
                raise TypeError("checkpoint state_dict must be a mapping")
            flat[prefix[:-1]] = value

    visit(state_dict, "")
    return flat


def _check_key_sets(flat, world, rank):
    if world <= 1:
        return
    dist = _dist()
    known = set(flat)
    gathered = [None] * world
    dist.all_gather_object(gathered, sorted(known))
    reference = set(gathered[0] or ())
    for other_rank, keys in enumerate(gathered):
        if keys is None or set(keys) != reference:
            missing = sorted(reference - set(keys or ()))
            extra = sorted(set(keys or ()) - reference)
            raise RuntimeError(
                "torch.distributed.checkpoint.save: rank %d has a different "
                "state-dict key set (missing=%s, extra=%s); a sharded "
                "checkpoint needs the same keys on every rank"
                % (other_rank, missing[:5], extra[:5]))


def save(state_dict, *, checkpoint_id=None, storage_writer=None, planner=None,
         process_group=None, **kwargs):
    _reject(kwargs, storage_writer, planner, "save", _SAVE_KWARGS)
    if checkpoint_id is None:
        raise ValueError(
            "torch.distributed.checkpoint.save requires checkpoint_id= when "
            "no storage_writer is supplied")
    world, rank = _context(False, process_group)
    flat = _flatten(state_dict)
    _check_key_sets(flat, world, rank)

    checkpoint_dir = os.fspath(checkpoint_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    rank_dir = os.path.join(checkpoint_dir, str(rank))
    os.makedirs(rank_dir, exist_ok=True)
    torch = _torch()

    entries = {}
    for index, key in enumerate(sorted(flat)):
        filename = "__%d_0%s" % (index, _SUFFIX)
        payload, metadata = _leaf_metadata(flat[key])
        metadata["file"] = filename
        torch.save(payload, os.path.join(rank_dir, filename))
        entries[key] = metadata

    if world > 1:
        _dist().barrier()
    if rank == 0:
        metadata = {
            "format": _FORMAT,
            "version": _VERSION,
            "world_size": world,
            "keys": entries,
        }
        with open(os.path.join(checkpoint_dir, _METADATA_NAME), "wb") as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if world > 1:
        _dist().barrier()
    return None


def _read_metadata(checkpoint_dir):
    path = os.path.join(os.fspath(checkpoint_dir), _METADATA_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "torch.distributed.checkpoint: %s is not a checkpoint directory "
            "written by this compat (missing %s)"
            % (os.fspath(checkpoint_dir), _METADATA_NAME))
    with open(path, "rb") as handle:
        metadata = pickle.load(handle)
    if metadata.get("format") != _FORMAT or metadata.get("version") != _VERSION:
        raise RuntimeError(
            "torch.distributed.checkpoint: unsupported checkpoint metadata "
            "format/version %r/%r"
            % (metadata.get("format"), metadata.get("version")))
    return metadata


def _copy_into(target, loaded):
    if tuple(target.shape) != tuple(loaded.shape):
        raise RuntimeError(
            "torch.distributed.checkpoint: loaded tensor shape %s does not "
            "match the state-dict placeholder shape %s"
            % (tuple(loaded.shape), tuple(target.shape)))
    target.update(loaded)
    return target


def _assign_leaf(target, loaded, metadata):
    kind = metadata.get("kind")
    if kind == "dtensor":
        placements = metadata.get("placements") or ()
        if _placements(target) is not None:
            # The skeleton already owns a DTensor shard; copy this rank's bytes
            # through the same view so its mesh and placement stay untouched.
            local = (target.to_local()
                     if not isinstance(target, jt.Var) else target)
            if not isinstance(local, jt.Var):
                raise TypeError(
                    "torch.distributed.checkpoint: DTensor placeholder has no "
                    "tensor shard to load into")
            return _copy_into(local, loaded)
        mesh = _decode_mesh(metadata.get("mesh"))
        if mesh is None:
            raise RuntimeError(
                "torch.distributed.checkpoint: DTensor checkpoint has no "
                "device-mesh metadata")
        fsdp2 = _dtensor_api()
        rebuilt = fsdp2.DTensor.from_local(
            loaded, mesh, _decode_placements(placements),
            shape=tuple(metadata.get("global_shape") or ()))
        if isinstance(target, jt.Var):
            if all(item["type"] == "replicate" for item in placements):
                return _copy_into(target, loaded)
            raise RuntimeError(
                "torch.distributed.checkpoint: cannot load a sharded DTensor "
                "entry into a plain tensor placeholder without a device mesh")
        return rebuilt
    if kind == "tensor":
        if isinstance(target, jt.Var):
            return _copy_into(target, loaded)
        if target is not None:
            raise TypeError(
                "torch.distributed.checkpoint: tensor entry cannot be loaded "
                "into %s" % type(target).__name__)
        return loaded
    return loaded


def _load_tree(value, prefix, entries, rank_dir):
    if _is_stateful(value):
        nested = value.state_dict()
        loaded = _load_tree(nested, prefix, entries, rank_dir)
        value.load_state_dict(loaded)
        return value
    if isinstance(value, Mapping):
        loaded = {}
        for key, item in value.items():
            loaded[key] = _load_tree(item, prefix + str(key) + ".", entries, rank_dir)
        if isinstance(value, dict):
            value.clear()
            value.update(loaded)
            return value
        return loaded
    key = prefix[:-1] if prefix else ""
    metadata = entries.get(key)
    if metadata is None:
        raise KeyError(
            "torch.distributed.checkpoint: key %r is in the load skeleton but "
            "not in the checkpoint" % key)
    path = os.path.join(rank_dir, metadata["file"])
    loaded = _torch().load(path)
    return _assign_leaf(value, loaded, metadata)


def load(state_dict, *, checkpoint_id=None, storage_reader=None, planner=None,
         process_group=None, no_dist=False, **kwargs):
    _reject(kwargs, storage_reader, planner, "load", _LOAD_KWARGS)
    if checkpoint_id is None:
        raise ValueError(
            "torch.distributed.checkpoint.load requires checkpoint_id= when "
            "no storage_reader is supplied")
    world, rank = _context(no_dist, process_group)
    metadata = _read_metadata(checkpoint_id)
    stored_world = metadata.get("world_size")
    if stored_world != world:
        raise RuntimeError(
            "torch.distributed.checkpoint: checkpoint was written by %s ranks "
            "but is being loaded by %d; resharding is not implemented"
            % (stored_world, world))
    rank_dir = os.path.join(os.fspath(checkpoint_id), str(rank))
    return _load_tree(state_dict, "", metadata.get("keys") or {}, rank_dir)


#: Deprecated torch spellings keep working; they take the same arguments.
save_state_dict = save
load_state_dict = load


_EXPORTS = (
    "Stateful", "save", "load", "save_state_dict", "load_state_dict",
)

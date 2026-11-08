"""Default Torch DCP calls backed by the native sharded checkpoint owner.

The on-disk format is Jittor's validated manifest format. PyTorch .metadata /
.distcp files, custom planners and custom storage providers remain unsupported.
No tensor is gathered to manufacture a sharded checkpoint.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import itertools
import os
from pathlib import Path

import numpy as np


_SEQUENCES = {}


@dataclass(frozen=True)
class Metadata:
    state_dict_metadata: dict
    storage_data: dict


class FileSystemWriter:
    def __init__(self, path, *, single_file_per_rank=None, sync_files=True,
                 thread_count=1, overwrite=True):
        if single_file_per_rank is not None or sync_files is not True or thread_count != 1:
            raise NotImplementedError("native DCP uses per-leaf files; custom filesystem writer options are unsupported")
        if type(overwrite) is not bool:
            raise TypeError("overwrite must be bool")
        self.path = path
        self.checkpoint_id = self.path
        self.overwrite = overwrite


class FileSystemReader:
    def __init__(self, path):
        self.path = path
        self.checkpoint_id = self.path

    def read_metadata(self):
        native = _native()
        manifest = _read_manifest(native, self.path)
        return _metadata(manifest)


class DefaultSavePlanner:
    def __init__(self, flatten_state_dict=True, flatten_sharded_tensors=True,
                 dedup_replicated_tensors=True, storage_meta=None, *,
                 dedup_save_to_lowest_rank=False):
        if (flatten_state_dict is not True or flatten_sharded_tensors is not True
                or dedup_replicated_tensors not in (None, True) or dedup_save_to_lowest_rank is not False
                or storage_meta is not None):
            raise NotImplementedError("only default native DCP save planning is supported")
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.dedup_replicated_tensors = True
        self.storage_meta = storage_meta


class DefaultLoadPlanner:
    def __init__(self, flatten_state_dict=True, flatten_sharded_tensors=True,
                 allow_partial_load=False, dedup_replicated_tensors=True):
        if (flatten_state_dict is not True or flatten_sharded_tensors is not True
                or allow_partial_load is not False or dedup_replicated_tensors is not True):
            raise NotImplementedError("only complete default native DCP load planning is supported")
        self.flatten_state_dict = flatten_state_dict
        self.flatten_sharded_tensors = flatten_sharded_tensors
        self.allow_partial_load = allow_partial_load
        self.dedup_replicated_tensors = dedup_replicated_tensors


def _native():
    from jittor.distributed import checkpoint
    return checkpoint


def _pointer(path, key):
    from jittor.serialization.sharded import _pointer as native_pointer
    return native_pointer(path, key)


def _metadata(manifest):
    return Metadata(manifest["records"][0]["tensors"], manifest)


def _read_manifest(native, path):
    if (Path(path) / ".metadata").exists() and not (Path(path) / "manifest.json").exists():
        raise NotImplementedError("PyTorch .metadata/.distcp checkpoint format is unsupported; expected native Jittor manifest")
    return native.read_manifest(path)


def _io_context(native, process_group):
    from jittor import distributed as dist
    store = dist.get_default_store()
    if dist.get_world_size(process_group) > 1 and store is None:
        import jittor as jt
        from .context import get_install_context
        state = get_install_context(jt).state["distributed_api"]["state"]
        store = state.get("store")
        if store is None:
            # This registers only the existing Store protocol. The launcher
            # continues to own the active HCCL/NCCL communicator.
            from jittor.distributed.store import rendezvous
            store, rank, world = next(rendezvous("env://", rank=dist.get_rank(),
                                                 world_size=dist.get_world_size()))
            if rank != dist.get_rank() or world != dist.get_world_size():
                raise RuntimeError("checkpoint Store rank/world differs from the active native communicator")
            state["store"] = store
    rank, world, ranks, selected = native._context(process_group, store)
    return rank, world, ranks, selected


def _ready(native, context, path, operation, error):
    rank, world, ranks, store = context
    identity = (str(Path(path).resolve()), tuple(ranks), operation)
    sequence = next(_SEQUENCES.setdefault(identity, itertools.count()))
    key = "dcp/{}/{}/".format(hashlib.sha256(native.json_bytes(identity)).hexdigest(), sequence)
    status = {"error": "{}: {}".format(type(error).__name__, error)} if error else {"ready": True}
    statuses = native._exchange(store, key, rank, world, status, 60)
    failures = {i: item["error"] for i, item in enumerate(statuses) if "error" in item}
    if failures:
        raise native.CheckpointError("DCP {} validation failed: {}".format(operation, failures)) from error


def _local(value):
    metadata = getattr(value, "checkpoint_metadata", None)
    if metadata is not None:
        return value.to_local(), metadata
    return value, None


def _tensor(value):
    return isinstance(value, np.ndarray) or (hasattr(value, "shape") and hasattr(value, "dtype")
                                             and callable(getattr(value, "numpy", None)))


def _encode(value, native):
    local, metadata = _local(value)
    if metadata is not None:
        return native.TensorShard(local, tuple(metadata[0]), int(metadata[1]))
    if isinstance(value, Mapping):
        return {key: _encode(item, native) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_encode(item, native) for item in value)
    return value


def _targets(value, native, path="", result=None):
    result = {} if result is None else result
    local, metadata = _local(value)
    if _tensor(local):
        shape, offset = (tuple(metadata[0]), int(metadata[1])) if metadata is not None else (tuple(local.shape), 0)
        result[path] = native.TensorShard(local, shape, offset)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _targets(item, native, _pointer(path, key), result)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _targets(item, native, _pointer(path, index), result)
    return result


def _dtype(value):
    dtype = str(value.dtype)
    return dtype.removeprefix("torch.") if hasattr(dtype, "removeprefix") else dtype.replace("torch.", "", 1)


def _validate_template(target, node, record, path=""):
    """Validate the complete target schema before allocating decoded leaves."""
    kind = node["kind"]
    local, metadata = _local(target)
    if _tensor(local):
        if kind != "tensor":
            raise ValueError("checkpoint target is not a tensor at " + path)
        spec = record["tensors"][node["path"]]
        shape = tuple(metadata[0]) if metadata is not None else tuple(local.shape)
        if shape != tuple(spec["shape"]):
            raise ValueError("checkpoint target global shape mismatch at " + path)
        if _dtype(local) != spec["dtype"]:
            raise ValueError("checkpoint target dtype mismatch at " + path)
        return
    if isinstance(target, Mapping):
        if kind != "dict":
            raise ValueError("checkpoint target mapping differs at " + path)
        items = dict(node["items"])
        if set(target) != set(items):
            raise ValueError("checkpoint keys differ; partial/empty DCP templates are unsupported at " + path)
        for key, value in target.items():
            _validate_template(value, items[key], record, _pointer(path, key))
        return
    if isinstance(target, (tuple, list)):
        expected = "tuple" if isinstance(target, tuple) else "list"
        if kind != expected or len(node["items"]) != len(target):
            raise ValueError("checkpoint sequence type/length differs at " + path)
        for index, (value, saved) in enumerate(zip(target, node["items"])):
            _validate_template(value, saved, record, _pointer(path, index))
        return
    if kind not in ("value", "numpy_scalar"):
        raise ValueError("checkpoint scalar target differs from saved tensor/container at " + path)


def _plan(target, restored, native, copies, path=""):
    local, metadata = _local(target)
    if _tensor(local):
        if not isinstance(restored, native.TensorShard):
            raise ValueError("checkpoint target is not a tensor at " + path)
        source = restored.tensor
        if tuple(source.shape) != tuple(local.shape) or _dtype(source) != _dtype(local):
            raise ValueError("checkpoint tensor shape/dtype differs at " + path)
        copies.append((local, source))
        return target
    if isinstance(target, Mapping):
        if not isinstance(restored, Mapping) or set(restored) != set(target):
            raise ValueError("checkpoint keys differ; partial/empty DCP templates are unsupported at " + path)
        return {key: _plan(item, restored[key], native, copies, _pointer(path, key))
                for key, item in target.items()}
    if isinstance(target, (tuple, list)):
        if type(restored) is not type(target) or len(restored) != len(target):
            raise ValueError("checkpoint sequence type/length differs at " + path)
        return type(target)(_plan(item, saved, native, copies, _pointer(path, index))
                            for index, (item, saved) in enumerate(zip(target, restored)))
    if _tensor(restored) or isinstance(restored, native.TensorShard):
        raise ValueError("checkpoint scalar target differs from saved tensor at " + path)
    return restored


def _destination(checkpoint_id, storage, storage_type):
    if storage is not None and type(storage) is not storage_type:
        raise NotImplementedError("custom DCP storage providers are unsupported")
    if storage is None:
        if checkpoint_id is None:
            raise ValueError("checkpoint_id or a filesystem storage provider is required")
        storage = storage_type(checkpoint_id)
    if checkpoint_id is not None and Path(checkpoint_id).resolve() != Path(storage.path).resolve():
        raise ValueError("checkpoint_id and filesystem storage path differ")
    return storage


def save(state_dict, *, checkpoint_id=None, storage_writer=None, planner=None, process_group=None):
    if planner is not None and type(planner) is not DefaultSavePlanner:
        raise NotImplementedError("custom DCP save planners are unsupported")
    writer = _destination(checkpoint_id, storage_writer, FileSystemWriter)
    native = _native()
    context = _io_context(native, process_group)
    error, state = None, None
    try:
        if not isinstance(state_dict, dict):
            raise TypeError("DCP state_dict must be a dictionary")
        state = _encode(state_dict, native)
    except (TypeError, ValueError) as failure:
        error = failure
    _ready(native, context, writer.path, "save", error)
    native.save(state, writer.path, process_group=process_group, store=context[3], overwrite=writer.overwrite)
    return _metadata(_read_manifest(native, writer.path))


def load(state_dict, *, checkpoint_id=None, storage_reader=None, planner=None, process_group=None):
    if planner is not None and type(planner) is not DefaultLoadPlanner:
        raise NotImplementedError("custom DCP load planners are unsupported")
    reader = _destination(checkpoint_id, storage_reader, FileSystemReader)
    native = _native()
    context = _io_context(native, process_group)
    error, planned, copies = None, None, []
    try:
        if not isinstance(state_dict, dict):
            raise TypeError("DCP state_dict must be a dictionary")
        manifest = _read_manifest(native, reader.path)
        record = manifest["records"][0]
        _validate_template(state_dict, record["tree"], record)
        restored = native.load(reader.path, process_group=process_group, store=context[3],
                               tensor_shards=_targets(state_dict, native))
        planned = _plan(state_dict, restored, native, copies)
    except (OSError, TypeError, ValueError, KeyError, RuntimeError, NotImplementedError) as failure:
        error = failure
    _ready(native, context, reader.path, "load", error)
    # All payloads, all targets and all participating ranks were validated.
    # Target tensors remain the same objects, including their device placement.
    for target, source in copies:
        if isinstance(target, np.ndarray):
            np.copyto(target, source, casting="no")
        else:
            import jittor as jt
            from .installers.tensor.method_api import _copy_, _copy_detached_storage
            with jt.no_grad():
                # Checkpoint restore writes existing leaf storage. Native
                # Vars do not expose the frontend-only copy_ method.
                if not _copy_detached_storage(target, source):
                    _copy_(target, source)
    state_dict.clear()
    state_dict.update(planned)
    return None


def save_state_dict(state_dict, storage_writer, process_group=None, coordinator_rank=0,
                    no_dist=False, planner=None):
    if coordinator_rank != 0 or no_dist:
        raise NotImplementedError("custom coordinator/no_dist DCP writes are unsupported")
    return save(state_dict, storage_writer=storage_writer, process_group=process_group, planner=planner)


def load_state_dict(state_dict, storage_reader, process_group=None, coordinator_rank=0,
                    no_dist=False, planner=None):
    if coordinator_rank != 0 or no_dist:
        raise NotImplementedError("custom coordinator/no_dist DCP reads are unsupported")
    return load(state_dict, storage_reader=storage_reader, process_group=process_group, planner=planner)


def make_checkpoint_io(rank_local_load, rank_local_save):
    """Compose existing rank-local snapshots and native interval manifests."""
    def has_shards(value):
        _, metadata = _local(value)
        if metadata is not None:
            return True
        if isinstance(value, Mapping):
            return any(has_shards(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return any(has_shards(item) for item in value)
        return False

    def save_dispatch(state_dict, storage_writer=None, *args, checkpoint_id=None,
                      planner=None, process_group=None, **kwargs):
        writer = _destination(checkpoint_id, storage_writer, FileSystemWriter)
        if has_shards(state_dict):
            if args or kwargs:
                raise TypeError("unsupported native sharded DCP save arguments")
            return save(state_dict, storage_writer=writer, planner=planner,
                        process_group=process_group)
        if not writer.overwrite and Path(writer.path).exists():
            raise FileExistsError(writer.path)
        return rank_local_save(state_dict, writer, *args, checkpoint_id=checkpoint_id,
                               planner=planner, process_group=process_group, **kwargs)

    def load_dispatch(state_dict, storage_reader=None, *args, checkpoint_id=None,
                      planner=None, process_group=None, **kwargs):
        reader = _destination(checkpoint_id, storage_reader, FileSystemReader)
        native_format = ((Path(reader.path) / "manifest.json").exists()
                         or (Path(reader.path) / ".metadata").exists())
        if native_format or has_shards(state_dict):
            if args or kwargs:
                raise TypeError("unsupported native sharded DCP load arguments")
            return load(state_dict, storage_reader=reader, planner=planner,
                        process_group=process_group)
        return rank_local_load(state_dict, reader, *args, checkpoint_id=checkpoint_id,
                               planner=planner, process_group=process_group, **kwargs)
    return load_dispatch, save_dispatch

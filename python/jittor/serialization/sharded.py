"""Native, versioned tensor-shard files; no frontend or collective ownership.

The descriptors below describe checkpoint storage, not distributed computation.
JSON metadata and NumPy payloads do not execute pickle globals when loaded.
"""

from dataclasses import dataclass
from functools import reduce
import hashlib
import json
from operator import mul
import os
from pathlib import Path

import numpy as np
import jittor as jt


FORMAT = "jittor.sharded_checkpoint"
VERSION = 1


class CheckpointError(RuntimeError):
    """A checkpoint could not be completed or validated."""


@dataclass(frozen=True)
class TensorShard:
    """A dense local tensor's flat interval in a global tensor.

    ``offset`` counts elements, not bytes. This is an I/O descriptor with no
    tensor operations or parameter/gradient sharding behavior.
    """

    tensor: object
    global_shape: tuple
    offset: int = 0


@dataclass(frozen=True)
class RankLocal:
    """State that must be restored to its own global rank, such as RNG."""

    value: object


def _pointer(path, component):
    return path + "/" + str(component).replace("~", "~0").replace("/", "~1")


def _shape(shape):
    result = tuple(shape)
    if any(type(dimension) is not int or dimension < 0 for dimension in result):
        raise ValueError("checkpoint dimensions must be nonnegative integers")
    return result


def _numel(shape):
    return reduce(mul, shape, 1)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf8")


def _durable_bytes(path, contents):
    with Path(path).open("wb") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())


def write_rank_state(state, directory, rank, world_size, global_rank):
    """Snapshot a state tree and write only this rank's tensor intervals."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {}
    aliases = {}

    def encode(value, path="", local=False):
        if isinstance(value, RankLocal):
            return {"kind": "rank_local", "value": encode(value.value, path, True)}
        shard = value if isinstance(value, TensorShard) else None
        tensor = shard.tensor if shard is not None else value
        native = isinstance(tensor, jt.Var)
        if native or isinstance(tensor, np.ndarray):
            key = (id(tensor), local)
            if key in aliases:
                if shard is not None:
                    raise ValueError("a TensorShard cannot alias another checkpoint leaf")
                return {"kind": "tensor", "path": aliases[key]}
            aliases[key] = path
            dtype = str(tensor.dtype)
            if native:
                from jittor._core.dtypes import dtype_name
                dtype = dtype_name(tensor.dtype)
                backend, device = jt.core.dispatch_context([tensor])
                if backend not in ("cpu", "cuda"):
                    raise NotImplementedError("checkpoint tensor backend is not validated: " + backend)
                if dtype == "bfloat16":
                    # Public numpy() exports BF16 numerically as FP32. An
                    # explicit integer storage view preserves every raw bit,
                    # including signed zero and NaN payloads, with no cast.
                    dense = tensor.contiguous()
                    raw = jt.reinterpret_view(dense, dense.shape, "uint16")
                    array = np.array(raw.numpy(), copy=True)
                else:
                    array = np.array(tensor.numpy(), copy=True)
            else:
                backend, device = "numpy", None
                array = np.array(tensor, copy=True)
            expected_storage = "uint16" if native and dtype == "bfloat16" else dtype
            if array.dtype.kind not in "biufc" or str(array.dtype) != expected_storage:
                raise TypeError("unsupported checkpoint tensor dtype at " + path)
            flat = array.reshape(-1)
            shape = _shape(tuple(int(d) for d in array.shape))
            role = "rank_local" if local else "replicated"
            offset = 0
            if shard is not None:
                if local:
                    raise ValueError("TensorShard cannot be nested in RankLocal")
                shape = _shape(shard.global_shape)
                if type(shard.offset) is not int or shard.offset < 0:
                    raise ValueError("TensorShard offset must be a nonnegative integer")
                offset, role = shard.offset, "sharded"
                if offset + flat.size > _numel(shape):
                    raise ValueError("TensorShard exceeds its global shape")
                payload = flat
            elif local:
                payload = flat
            else:
                offset = flat.size * rank // world_size
                end = flat.size * (rank + 1) // world_size
                payload = flat[offset:end]
            filename = "tensor-{:06d}.npy".format(len(tensors))
            target = directory / filename
            with target.open("wb") as output:
                np.save(output, payload, allow_pickle=False)
                output.flush()
                os.fsync(output.fileno())
            spec = {
                "path": path, "shape": list(shape), "dtype": dtype,
                "storage_dtype": array.dtype.str, "native": native,
                "backend": backend, "source_device": device, "role": role,
                "offset": int(offset), "count": int(payload.size),
                "file": filename, "sha256": file_digest(target),
            }
            if role == "replicated":
                spec["replica_sha256"] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
            tensors[path] = spec
            return {"kind": "tensor", "path": path}
        if value is None or type(value) in (bool, int, float, str):
            json_bytes(value)
            return {"kind": "value", "value": value}
        if isinstance(value, np.generic):
            return {"kind": "numpy_scalar", "dtype": value.dtype.str,
                    "value": encode(value.item(), path, local)}
        if isinstance(value, dict):
            if any(type(key) is not str for key in value):
                raise TypeError("checkpoint dictionary keys must be strings")
            return {"kind": "dict", "items": [
                [key, encode(item, _pointer(path, key), local)]
                for key, item in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"kind": "tuple" if isinstance(value, tuple) else "list",
                    "items": [encode(item, _pointer(path, i), local)
                              for i, item in enumerate(value)]}
        raise TypeError("unsupported checkpoint value at {}: {}".format(path, type(value).__name__))

    tree = encode(state)
    record = {"rank": rank, "global_rank": global_rank, "world_size": world_size,
              "tree": tree, "tensors": tensors}
    _durable_bytes(directory / "metadata.json", json_bytes(record))
    return record


def _common_tree(node):
    if node["kind"] == "rank_local":
        return {"kind": "rank_local"}
    if node["kind"] == "dict":
        return {"kind": "dict", "items": [[key, _common_tree(value)]
                                            for key, value in node["items"]]}
    if node["kind"] in ("tuple", "list"):
        return {"kind": node["kind"], "items": [_common_tree(value) for value in node["items"]]}
    if node["kind"] == "numpy_scalar":
        return dict(node, value=_common_tree(node["value"]))
    return node


def validate_records(records, ranks):
    """Check tree identity, replica agreement and exact shard coverage."""
    if not records or len(records) != len(ranks) or len(set(ranks)) != len(ranks):
        raise CheckpointError("checkpoint rank set is incomplete or duplicated")
    common = _common_tree(records[0]["tree"])
    global_tensors = {}
    for rank, record in enumerate(records):
        if (record["rank"] != rank or record["global_rank"] != ranks[rank]
                or record["world_size"] != len(ranks)):
            raise CheckpointError("checkpoint rank/world metadata mismatch")
        if _common_tree(record["tree"]) != common:
            raise CheckpointError("checkpoint state tree or replicated metadata differs across ranks")
        referenced = set()

        def check_node(node, local=False):
            kind = node["kind"]
            if kind == "rank_local":
                check_node(node["value"], True)
            elif kind == "tensor":
                path = node["path"]
                if path not in record["tensors"]:
                    raise CheckpointError("checkpoint tree references a missing tensor")
                if local != (record["tensors"][path]["role"] == "rank_local"):
                    raise CheckpointError("checkpoint rank-local tensor role mismatch")
                referenced.add(path)
            elif kind == "dict":
                names = [item[0] for item in node["items"]]
                if any(type(name) is not str for name in names) or len(names) != len(set(names)):
                    raise CheckpointError("invalid checkpoint dictionary keys")
                for _, value in node["items"]:
                    check_node(value, local)
            elif kind in ("tuple", "list"):
                for value in node["items"]:
                    check_node(value, local)
            elif kind == "numpy_scalar":
                check_node(node["value"], local)
            elif kind != "value":
                raise CheckpointError("unknown checkpoint tree node")

        check_node(record["tree"])
        if referenced != set(record["tensors"]):
            raise CheckpointError("checkpoint contains an unreferenced tensor")
        for path, spec in record["tensors"].items():
            shape = _shape(spec["shape"])
            storage = "uint16" if spec["native"] and spec["dtype"] == "bfloat16" else spec["dtype"]
            if (spec["path"] != path or type(spec["native"]) is not bool
                    or np.dtype(storage).str != spec["storage_dtype"]):
                raise CheckpointError("checkpoint tensor dtype/path metadata mismatch")
            count, offset = spec["count"], spec["offset"]
            if (type(count) is not int or type(offset) is not int
                    or count < 0 or offset < 0 or count + offset > _numel(shape)):
                raise CheckpointError("invalid checkpoint tensor interval at " + path)
            if spec["role"] not in ("replicated", "sharded", "rank_local"):
                raise CheckpointError("invalid checkpoint tensor role")
            if spec["role"] != "rank_local":
                global_tensors.setdefault(path, []).append(spec)
    for path, entries in global_tensors.items():
        if len(entries) != len(records):
            raise CheckpointError("checkpoint tensor is missing a rank: " + path)
        reference = entries[0]
        fields = ("shape", "dtype", "storage_dtype", "native", "role", "backend")
        if reference["role"] == "replicated":
            fields += ("replica_sha256",)
        if any(any(spec[field] != reference[field] for field in fields) for spec in entries[1:]):
            raise CheckpointError("checkpoint tensor metadata or replicas differ at " + path)
        cursor = 0
        for spec in sorted(entries, key=lambda value: value["offset"]):
            if not spec["count"]:
                continue
            if spec["offset"] != cursor:
                raise CheckpointError("checkpoint shard overlap or gap at " + path)
            cursor += spec["count"]
        if cursor != _numel(reference["shape"]):
            raise CheckpointError("checkpoint shards do not cover the global tensor at " + path)


def validate_payloads(generation, records):
    generation = Path(generation)
    for rank, record in enumerate(records):
        rank_directory = generation / "rank-{:05d}".format(rank)
        metadata = rank_directory / "metadata.json"
        if (rank_directory.is_symlink() or metadata.is_symlink()
                or not metadata.is_file() or metadata.read_bytes() != json_bytes(record)):
            raise CheckpointError("checkpoint rank metadata is missing or differs from the manifest")
        for spec in record["tensors"].values():
            filename = spec["file"]
            if (not isinstance(filename, str) or not filename.startswith("tensor-")
                    or not filename.endswith(".npy") or Path(filename).name != filename):
                raise CheckpointError("invalid checkpoint payload filename")
            path = generation / "rank-{:05d}".format(rank) / filename
            if path.is_symlink() or not path.is_file():
                raise CheckpointError("missing checkpoint tensor payload: " + str(path))
            if file_digest(path) != spec["sha256"]:
                raise CheckpointError("checkpoint tensor checksum mismatch: " + str(path))
            array = np.load(str(path), allow_pickle=False, mmap_mode="r")
            if (array.shape != (spec["count"],) or array.dtype.str != spec["storage_dtype"]
                    or array.dtype.kind not in "biufc"):
                raise CheckpointError("checkpoint tensor payload shape/dtype mismatch")


def read_manifest(directory):
    directory = Path(directory)
    path = directory / "manifest.json"
    if not path.is_file() or path.is_symlink():
        raise CheckpointError("no complete checkpoint manifest: " + str(path))
    envelope = json.loads(path.read_text(encoding="utf8"))
    manifest = envelope["manifest"]
    if hashlib.sha256(json_bytes(manifest)).hexdigest() != envelope["sha256"]:
        raise CheckpointError("checkpoint manifest checksum mismatch")
    if manifest["format"] != FORMAT or manifest["version"] != VERSION:
        raise CheckpointError("unsupported checkpoint format/version")
    if type(manifest["world_size"]) is not int or manifest["world_size"] != len(manifest["ranks"]):
        raise CheckpointError("checkpoint manifest world size mismatch")
    generation = manifest["generation"]
    if (not isinstance(generation, str) or len(generation) != 32
            or any(char not in "0123456789abcdef" for char in generation)):
        raise CheckpointError("invalid checkpoint generation")
    validate_records(manifest["records"], manifest["ranks"])
    root = directory / generation
    if root.is_symlink():
        raise CheckpointError("checkpoint generation cannot be a symlink")
    validate_payloads(root, manifest["records"])
    return manifest


def restore_tree(directory, manifest, global_rank, tensor_shards=None):
    """Reassemble full leaves, or only explicitly requested local intervals."""
    records = manifest["records"]
    try:
        rank = manifest["ranks"].index(global_rank)
    except ValueError:
        rank = 0
    tree = records[rank]["tree"]
    root = Path(directory) / manifest["generation"]
    requested = dict(tensor_shards or {})
    restored = {}

    def decode(node):
        kind = node["kind"]
        if kind == "rank_local":
            if global_rank not in manifest["ranks"]:
                raise CheckpointError("rank-local checkpoint state has no matching global rank")
            return decode(node["value"])
        if kind == "value":
            return node["value"]
        if kind == "numpy_scalar":
            return np.asarray(decode(node["value"]), dtype=node["dtype"])[()]
        if kind == "dict":
            return {key: decode(value) for key, value in node["items"]}
        if kind in ("tuple", "list"):
            values = [decode(value) for value in node["items"]]
            return tuple(values) if kind == "tuple" else values
        if kind != "tensor":
            raise CheckpointError("unknown checkpoint tree node")
        path = node["path"]
        if path in restored:
            return restored[path]
        spec = records[rank]["tensors"][path]
        local = spec["role"] == "rank_local"
        selected = [(rank, spec)] if local else [
            (index, record["tensors"][path]) for index, record in enumerate(records)]
        target = requested.pop(path, None)
        shape = tuple(spec["shape"])
        start, count = 0, _numel(shape)
        if target is not None:
            if not isinstance(target, TensorShard) or tuple(target.global_shape) != shape:
                raise CheckpointError("requested TensorShard global shape mismatch at " + path)
            if isinstance(target.tensor, jt.Var) != spec["native"]:
                raise CheckpointError("requested TensorShard native/NumPy type mismatch at " + path)
            target_dtype = str(target.tensor.dtype)
            if target_dtype != spec["dtype"]:
                raise CheckpointError("requested TensorShard dtype mismatch at " + path)
            start = target.offset
            count = _numel(tuple(int(d) for d in target.tensor.shape))
            if type(start) is not int or start < 0 or start + count > _numel(shape) or local:
                raise CheckpointError("invalid requested checkpoint shard interval")
        output = np.empty(count, dtype=spec["storage_dtype"])
        for index, entry in selected:
            begin = max(start, entry["offset"])
            end = min(start + count, entry["offset"] + entry["count"])
            if begin < end:
                payload = np.load(str(root / "rank-{:05d}".format(index) / entry["file"]),
                                  allow_pickle=False, mmap_mode="r")
                output[begin - start:end - start] = payload[begin - entry["offset"]:end - entry["offset"]]
        output = output.reshape(tuple(target.tensor.shape) if target is not None else shape)
        if spec["native"]:
            def native_value():
                dtype = "uint16" if spec["dtype"] == "bfloat16" else spec["dtype"]
                tensor = jt.array(output, dtype=dtype).stop_grad()
                return (jt.reinterpret_view(tensor, tensor.shape, "bfloat16").stop_grad()
                        if spec["dtype"] == "bfloat16" else tensor)

            if target is not None:
                from jittor._core.var import placement_scope_like
                with placement_scope_like(target.tensor):
                    value = native_value()
            else:
                value = native_value()
        else:
            value = output
        if target is not None:
            value = TensorShard(value, shape, start)
        restored[path] = value
        return value

    result = decode(tree)
    if requested:
        raise CheckpointError("requested checkpoint tensor paths are missing: " + repr(sorted(requested)))
    return result

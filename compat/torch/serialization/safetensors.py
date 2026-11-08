"""SafeTensor codecs, readers and transactional optional frontend bindings."""
import json
import os
import struct
from types import MappingProxyType
import mmap as _mmap

import numpy as np
import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from ..context import get_install_context, registry_for
from .portable import _snapshot_tensors
from ..fidelity import Fidelity, register_api_bindings
from ...diagnostics import EXPECTED, swallowed
from ...transaction import runtime_hook, set_attr, _MISSING

_ST = {
    "F64": (np.float64, 8), "F32": (np.float32, 4), "F16": (np.float16, 2),
    "BF16": (None, 2), "I64": (np.int64, 8), "I32": (np.int32, 4),
    "I16": (np.int16, 2), "I8": (np.int8, 1), "U8": (np.uint8, 1),
    "U16": (np.uint16, 2), "U32": (np.uint32, 4), "U64": (np.uint64, 8),
    "BOOL": (np.bool_, 1), "F8_E4M3": (None, 1), "F8_E5M2": (None, 1),
}


_NP_TO_ST = {
    "float64": "F64", "float32": "F32", "float16": "F16",
    "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
    "uint8": "U8", "uint16": "U16", "uint32": "U32", "uint64": "U64",
    "bool": "BOOL", "bfloat16": "BF16",
}


def _bytes_to_np(raw, st_dtype, shape):
    if st_dtype.startswith("F8_"):
        raise NotImplementedError(
            "safetensors dtype %s has no supported Jittor decoder" % _jittor_dtype_name(st_dtype))
    npd, _itemsize = _ST[st_dtype]
    shape = tuple(shape)
    if _jittor_dtype_name(st_dtype) == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (u16 << 16).view(np.float32).reshape(shape)
    return np.frombuffer(raw, dtype=npd).reshape(shape)


def _to_tensor(array, st_dtype, device):
    g = get_install_context(jt).target_namespace
    array = np.asarray(array)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    dtype = "bfloat16" if _jittor_dtype_name(st_dtype) == "BF16" else array.dtype.name
    tensor = g.tensor(array, dtype=dtype, device="cpu", requires_grad=False)
    target_device = "cpu" if device is None else device
    if isinstance(target_device, (int, np.integer)) and not isinstance(target_device, bool):
        target_device = "cuda:%d" % int(target_device)
    return tensor.to(device=target_device)


def _read_range(filename, offset, length):
    with open(filename, "rb") as fh:
        fh.seek(offset)
        raw = fh.read(length)
    if len(raw) != length:
        raise EOFError("short safetensors payload: expected %d bytes, got %d"
                       % (length, len(raw)))
    return raw


def _contiguous_span(shape, idx):
    """Return (start element, element count, result shape) or None.

    ``None`` means the selection cannot be served by one contiguous byte
    range, so the caller must decode the whole tensor and let NumPy index it.
    """
    shape = tuple(int(value) for value in shape)
    if not shape:
        return None
    if idx is Ellipsis:
        return 0, int(np.prod(shape)), shape
    if not isinstance(idx, tuple):
        idx = (idx,)
    if Ellipsis in idx:
        position = idx.index(Ellipsis)
        fill = len(shape) - (len(idx) - 1)
        idx = idx[:position] + (slice(None),) * fill + idx[position + 1:]
    idx = idx + (slice(None),) * (len(shape) - len(idx))
    if any(item != slice(None) for item in idx[1:]):
        return None
    rows = shape[0]
    row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    first = idx[0]
    if isinstance(first, slice):
        if first.step not in (None, 1):
            return None
        start, stop, _ = first.indices(rows)
        if stop < start:
            return None
        return start * row, (stop - start) * row, (stop - start,) + shape[1:]
    if isinstance(first, (int, np.integer)) and not isinstance(first, bool):
        index = int(first)
        if index < 0:
            index += rows
        if not 0 <= index < rows:
            return None
        return index * row, row, shape[1:]
    return None


def _raw_to_tensor(raw, st_dtype, shape, device):
    """Build a tensor from file bytes; BF16 keeps its 16-bit payload."""
    if _jittor_dtype_name(st_dtype) != "BF16":
        return _to_tensor(_bytes_to_np(raw, st_dtype, shape), st_dtype, device)
    g = get_install_context(jt).target_namespace
    array = np.frombuffer(raw, dtype=np.uint16).reshape(tuple(shape))
    tensor = g.tensor(array, dtype="uint16", device="cpu", requires_grad=False)
    tensor = tensor.reinterpret_view(tuple(shape), "bfloat16")
    target_device = "cpu" if device is None else device
    if isinstance(target_device, (int, np.integer)) and not isinstance(target_device, bool):
        target_device = "cuda:%d" % int(target_device)
    return tensor.to(device=target_device)


class _PySafeSlice:
    def __init__(self, raw, st_dtype, shape, device="cpu", source=None):
        self._raw = raw
        self._dtype = st_dtype
        self._shape = shape
        self._device = device
        # ``source`` is (filename, byte offset, byte length). Bytes stay on
        # disk until a selection needs them, matching the native reader.
        self._source = source

    def get_shape(self):
        return list(self._shape)

    def get_dtype(self):
        return self._dtype

    def _read_all(self):
        if self._raw is None:
            filename, offset, length = self._source
            self._raw = _read_range(filename, offset, length)
        return self._raw

    def __getitem__(self, idx):
        span = None if self._raw is not None else _contiguous_span(self._shape, idx)
        if span is not None:
            start, count, out_shape = span
            itemsize = _ST[self._dtype][1]
            filename, offset, _length = self._source
            raw = _read_range(filename, offset + start * itemsize, count * itemsize)
            return _raw_to_tensor(raw, self._dtype, out_shape, self._device)
        raw = self._read_all()
        if idx is Ellipsis:
            return _raw_to_tensor(raw, self._dtype, self._shape, self._device)
        arr = _bytes_to_np(raw, self._dtype, self._shape)[idx]
        return _to_tensor(arr, self._dtype, self._device)


def _header_error(message):
    """The error safetensors' own reader raises for an unusable header."""
    try:
        from safetensors import SafetensorError as error_type
    except ImportError:
        error_type = ValueError
    return error_type("Error while deserializing header: " + message)


def _parse_header(prefix, read_header, total_size):
    """Decode and validate a header the way ``safe_open`` does before any read.

    A truncated or corrupt file used to surface as a ``struct.error``,
    ``UnicodeDecodeError``, ``JSONDecodeError`` or -- for a short payload -- a
    reshape error inside whichever tensor happened to be read first. Every one
    of those is reported up front now, as safetensors' ``SafetensorError``.
    Returns ``(header, header_length)``.
    """
    if len(prefix) != 8:
        raise _header_error("header too small")
    n = struct.unpack("<Q", prefix)[0]
    if n > total_size - 8:
        raise _header_error("header too large")
    try:
        header = json.loads(read_header(n).decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise _header_error("invalid UTF-8 in header: %s" % exc) from None
    except json.JSONDecodeError as exc:
        raise _header_error("invalid JSON in header: %s" % exc) from None
    if not isinstance(header, dict):
        raise _header_error("invalid JSON in header: expected an object, got %s"
                            % type(header).__name__)
    data_size = total_size - 8 - n
    for key, entry in header.items():
        if key == "__metadata__":
            continue
        offsets = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(type(value) is int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1]):
            raise _header_error("invalid tensor info for %r" % (key,))
        if offsets[1] > data_size:
            raise _header_error("incomplete metadata, file not fully covered")
    return header, n


class _PySafeOpen:
    def __init__(self, filename, framework="pt", device="cpu", backend="mmap", mmap_handle=None):
        self._filename = filename
        self._device = device
        self._mmap = mmap_handle
        with open(filename, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            self._header, n = _parse_header(fh.read(8), fh.read, size)
        # Tensors are read on demand, as safetensors' mmap reader does: holding
        # the whole payload doubled peak host memory while a checkpoint loaded.
        self._data_offset = 8 + n
        # Absence differs from an explicitly empty metadata dictionary.
        # Accelerate uses None to recognize an archive without a format tag.
        self._meta = self._header.pop("__metadata__", None)

    def keys(self):
        return list(self._header.keys())

    def metadata(self):
        return self._meta

    def _entry(self, key):
        entry = self._header[key]
        start, end = entry["data_offsets"]
        absolute = self._data_offset + start
        if self._mmap is not None:
            raw = memoryview(self._mmap)[absolute:absolute + end - start]
        else:
            with open(self._filename, "rb") as fh:
                fh.seek(absolute)
                raw = fh.read(end - start)
        if len(raw) != end - start:
            # The header was checked against the file at open; it shrank since.
            raise EOFError("short safetensors payload for %r: expected %d bytes, got %d"
                           % (key, end - start, len(raw)))
        return entry["dtype"], entry["shape"], raw

    def get_slice(self, key):
        entry = self._header[key]
        start, end = entry["data_offsets"]
        return _PySafeSlice(None, entry["dtype"], entry["shape"], self._device,
                            source=(self._filename, self._data_offset + start, end - start))

    def get_tensor(self, key):
        st_dtype, shape, raw = self._entry(key)
        return _raw_to_tensor(raw, st_dtype, shape, self._device)

    def get_dtype(self, key):
        return self._header[key]["dtype"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _load_bytes(data):
    header, n = _parse_header(data[:8], lambda length: data[8:8 + length], len(data))
    header.pop("__metadata__", None)
    base = 8 + n
    out = {}
    for key, entry in header.items():
        start, end = entry["data_offsets"]
        out[key] = _raw_to_tensor(data[base + start:base + end], entry["dtype"], entry["shape"], "cpu")
    return out


def _load_file(filename, device="cpu", *, backend="mmap"):
    # ``safetensors.torch.load_model`` calls ``load_file`` with ``backend=``;
    # accepting only ``device`` raised TypeError before any weight loaded.
    # The whole checkpoint is being read, so one read-only mapping avoids a
    # separate host copy per tensor while keeping peak RSS at the tensors.
    with open(filename, "rb") as fh:
        try:
            handle = _mmap.mmap(fh.fileno(), 0, access=_mmap.ACCESS_READ)
        except (OSError, ValueError):
            handle = None
        try:
            with _PySafeOpen(filename, device=device, backend=backend,
                             mmap_handle=handle) as safe:
                return {key: safe.get_tensor(key) for key in safe.keys()}
        finally:
            if handle is not None:
                handle.close()


def _reject_shared_storage(tensors):
    """Refuse tensors that share storage, as ``safetensors.torch`` does.

    The format has no aliasing: two names for one storage would be written as
    two copies and come back as two independent tensors, silently untying
    weights. safetensors raises instead and points at ``save_model``, and
    Transformers/PEFT remove the aliases before saving for exactly that reason.
    Storage identity is the frontend's ``untyped_storage().data_ptr()``, the
    same key those libraries deduplicate on.
    """
    allocations = {}
    for key, value in tensors.items():
        storage = getattr(value, "untyped_storage", None)
        if storage is not None and value.numel():
            pointer = storage().data_ptr()
            start = value.data_ptr()
            end = start + value.numel() * value.element_size()
            allocations.setdefault((str(value.device), pointer), []).append((start, end, key))
    shared = []
    for ranges in allocations.values():
        group, last_end = [], 0
        for start, end, key in sorted(ranges):
            if group and start >= last_end:
                if len(group) > 1:
                    shared.append(group)
                group = []
            group.append(key)
            last_end = max(last_end, end)
        if len(group) > 1:
            shared.append(group)
    if shared:
        raise RuntimeError(
            "Some tensors share memory, this will lead to duplicate memory on disk "
            "and potential differences when loading them again: %s.\n"
            "A potential way to correctly save your model is to use `save_model`.\n"
            "More information at https://huggingface.co/docs/safetensors/torch_shared_tensors"
            % [set(group) for group in shared])


def _serialize_parts(tensors, metadata=None):
    _reject_shared_storage(tensors)
    # A state_dict's entries alias live parameters, and `numpy()` would move
    # their sharing group to the host: fetch them as `torch.save` does.
    bf16_views = {}
    snapshot_input = {}
    for key, value in tensors.items():
        if isinstance(value, jt.Var) and _jittor_dtype_name(value.dtype) == "bfloat16":
            try:
                if value.is_contiguous():
                    bf16_views[key] = value.reinterpret_view(value.shape, "uint16")
                    snapshot_input[key] = bf16_views[key]
                    continue
            except EXPECTED:
                pass
        snapshot_input[key] = value
    snapshots = _snapshot_tensors(snapshot_input)
    header = {}
    parts = []
    offset = 0
    for key, value in tensors.items():
        tensor_dtype = _jittor_dtype_name(value.dtype) if isinstance(value, jt.Var) else None
        snapshot_object = bf16_views.get(key, value)
        arr = (snapshots[id(snapshot_object)] if isinstance(value, jt.Var) else
               value.numpy() if hasattr(value, "numpy") else np.asarray(value))
        arr = np.asarray(arr)
        shape = list(arr.shape)
        dtype = tensor_dtype or arr.dtype.name
        if dtype not in _NP_TO_ST:
            raise NotImplementedError("safetensors cannot save dtype %s" % _jittor_dtype_name(dtype))
        st_dtype = _NP_TO_ST[dtype]
        if _jittor_dtype_name(st_dtype) == "BF16":
            if key in bf16_views:
                part = np.ascontiguousarray(arr.astype(np.uint16, copy=False))
            else:
                # Native BF16 numpy() exposes exact values in float32. Encode
                # their upper 16 bits, not a float32 payload with a BF16 label.
                bits = arr.astype(np.float32, copy=False).view(np.uint32)
                part = np.ascontiguousarray((bits >> 16).astype(np.uint16))
        else:
            part = np.ascontiguousarray(arr.astype(_ST[st_dtype][0], copy=False))
        header[key] = {
            "dtype": st_dtype,
            "shape": shape,
            "data_offsets": [offset, offset + int(part.nbytes)],
        }
        parts.append(part)
        offset += int(part.nbytes)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # The format aligns the data buffer to eight bytes with JSON whitespace.
    raw_header += b" " * (-len(raw_header) % 8)
    return struct.pack("<Q", len(raw_header)) + raw_header, parts


def _save_dict(tensors, metadata=None):
    prefix, parts = _serialize_parts(tensors, metadata)
    return prefix + b"".join(memoryview(part) for part in parts)


def _save_file(tensors, filename, metadata=None):
    prefix, parts = _serialize_parts(tensors, metadata)
    with open(filename, "wb") as fh:
        fh.write(prefix)
        for part in parts:
            fh.write(memoryview(part))


def _safe_open(filename, framework="pt", device="cpu", backend="mmap"):
    if framework in ("pt", "pytorch"):
        return _PySafeOpen(filename, framework, device, backend)
    # NumPy and other frameworks keep their original loader and return
    # types. In particular, do not route safetensors.numpy through Tensor.
    return get_install_context(jt).state["safetensors_native_api"]["safe_open"](filename, framework=framework, device=device)


def _install_safetensors_shim(registry=None):
    """Bind optional SafeTensor APIs in one reversible runtime hook."""
    registry = registry_for(jt, registry)
    context = get_install_context(registry.target_namespace)
    try:
        import safetensors as safetensors_module
        import safetensors.torch as torch_module
    except EXPECTED as exc:
        swallowed("torch/serialization.py _install_safetensors_shim: import safetensors.torch", exc)
        return
    if getattr(safetensors_module, "_jittor_torch_compat", False):
        return
    with runtime_hook("torch.serialization.safetensors") as transaction:
        captured = MappingProxyType({"safe_open": safetensors_module.safe_open})
        transaction.record(context.state, "safetensors_native_api",
                           context.state.get("safetensors_native_api", _MISSING), captured)
        context.state["safetensors_native_api"] = captured
        set_attr(safetensors_module, "safe_open", _safe_open, context=context)
        set_attr(safetensors_module, "_jittor_torch_compat", True, context=context)
        published = registry.module_map.get("safetensors", safetensors_module)
        if published is not safetensors_module:
            set_attr(published, "safe_open", _safe_open, context=context)
        for name, implementation in (
            ("safe_open", _safe_open), ("load", _load_bytes),
            ("load_file", _load_file), ("save", _save_dict), ("save_file", _save_file),
        ):
            set_attr(torch_module, name, implementation, context=context)
        register_api_bindings(torch_module, "safetensors.torch",
            ("safe_open", "load", "load_file", "save", "save_file"),
            Fidelity.APPROXIMATE, "Supported dense dtype codecs preserve wide integers, "
            "BF16 and requested device; float8 is explicitly unsupported")
        register_api_bindings(safetensors_module, "safetensors", ("safe_open",),
            Fidelity.APPROXIMATE, "Torch reader uses the active frontend; other frameworks "
            "delegate to the original SafeTensor reader")

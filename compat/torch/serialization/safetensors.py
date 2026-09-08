"""SafeTensor codecs, readers and transactional optional frontend bindings."""
import json
import struct
from types import MappingProxyType
import numpy as np
import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from ..context import get_install_context, registry_for
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


class _PySafeSlice:
    def __init__(self, raw, st_dtype, shape, device="cpu"):
        self._raw = raw
        self._dtype = st_dtype
        self._shape = shape
        self._device = device

    def get_shape(self):
        return list(self._shape)

    def get_dtype(self):
        return self._dtype

    def __getitem__(self, idx):
        arr = _bytes_to_np(self._raw, self._dtype, self._shape)
        if idx is not Ellipsis:
            arr = arr[idx]
        return _to_tensor(arr, self._dtype, self._device)


class _PySafeOpen:
    def __init__(self, filename, framework="pt", device="cpu", backend="mmap"):
        self._device = device
        with open(filename, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            self._header = json.loads(fh.read(n).decode("utf-8"))
            self._data = fh.read()
        self._meta = self._header.pop("__metadata__", {})

    def keys(self):
        return list(self._header.keys())

    def metadata(self):
        return self._meta

    def _entry(self, key):
        entry = self._header[key]
        start, end = entry["data_offsets"]
        return entry["dtype"], entry["shape"], self._data[start:end]

    def get_slice(self, key):
        st_dtype, shape, raw = self._entry(key)
        return _PySafeSlice(raw, st_dtype, shape, self._device)

    def get_tensor(self, key):
        st_dtype, shape, raw = self._entry(key)
        return _to_tensor(_bytes_to_np(raw, st_dtype, shape), st_dtype, self._device)

    def get_dtype(self, key):
        return self._header[key]["dtype"]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _load_bytes(data):
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n].decode("utf-8"))
    header.pop("__metadata__", None)
    base = 8 + n
    out = {}
    for key, entry in header.items():
        start, end = entry["data_offsets"]
        arr = _bytes_to_np(data[base + start:base + end], entry["dtype"], entry["shape"])
        out[key] = _to_tensor(arr, entry["dtype"], "cpu")
    return out


def _load_file(filename, device="cpu"):
    with _PySafeOpen(filename, device=device) as safe:
        return {key: safe.get_tensor(key) for key in safe.keys()}


def _save_dict(tensors, metadata=None):
    header = {}
    blobs = []
    offset = 0
    for key, value in tensors.items():
        tensor_dtype = _jittor_dtype_name(value.dtype) if isinstance(value, jt.Var) else None
        arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        arr = np.asarray(arr)
        shape = list(arr.shape)
        dtype = tensor_dtype or arr.dtype.name
        if dtype not in _NP_TO_ST:
            raise NotImplementedError("safetensors cannot save dtype %s" % _jittor_dtype_name(dtype))
        st_dtype = _NP_TO_ST[dtype]
        if _jittor_dtype_name(st_dtype) == "BF16":
            # Native BF16 numpy() exposes exact values in float32. Encode
            # their upper 16 bits, not a float32 payload with a BF16 label.
            bits = arr.astype(np.float32, copy=False).view(np.uint32)
            blob = (bits >> 16).astype(np.uint16).tobytes(order="C")
        else:
            blob = arr.astype(_ST[st_dtype][0], copy=False).tobytes(order="C")
        header[key] = {
            "dtype": st_dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(raw_header)) + raw_header + b"".join(blobs)


def _save_file(tensors, filename, metadata=None):
    payload = _save_dict(tensors, metadata)
    with open(filename, "wb") as fh:
        fh.write(payload)


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

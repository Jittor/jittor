"""Serialization adapters used by the torch compatibility layer."""

import numpy as np
import jittor as jt


def _install_safetensors_shim():
    """Patch safetensors.torch to load tensors without real torch storage."""
    try:
        import json
        import struct
        import sys
        import safetensors as _st
    except Exception:
        return
    if getattr(_st, "_jittor_torch_compat", False):
        return

    _ST = {
        "F64": (np.float64, 8), "F32": (np.float32, 4), "F16": (np.float16, 2),
        "BF16": (None, 2), "I64": (np.int64, 8), "I32": (np.int32, 4),
        "I16": (np.int16, 2), "I8": (np.int8, 1), "U8": (np.uint8, 1),
        "U16": (np.uint16, 2), "U32": (np.uint32, 4), "U64": (np.uint64, 8),
        "BOOL": (np.bool_, 1), "F8_E4M3": (None, 1), "F8_E5M2": (None, 1),
    }

    def _bytes_to_np(raw, st_dtype, shape):
        npd, _itemsize = _ST[st_dtype]
        shape = tuple(shape)
        if st_dtype == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            return (u16 << 16).view(np.float32).reshape(shape)
        if st_dtype in ("F8_E4M3", "F8_E5M2"):
            return np.frombuffer(raw, dtype=np.uint8).astype(np.float32).reshape(shape)
        return np.frombuffer(raw, dtype=npd).reshape(shape)

    class _PySafeSlice:
        def __init__(self, raw, st_dtype, shape):
            self._raw = raw
            self._dtype = st_dtype
            self._shape = shape

        def get_shape(self):
            return list(self._shape)

        def get_dtype(self):
            return self._dtype

        def __getitem__(self, idx):
            arr = _bytes_to_np(self._raw, self._dtype, self._shape)
            if idx is not Ellipsis and idx != slice(None):
                arr = arr[idx]
            return jt.array(np.ascontiguousarray(arr))

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
            return _PySafeSlice(raw, st_dtype, shape)

        def get_tensor(self, key):
            st_dtype, shape, raw = self._entry(key)
            return jt.array(np.ascontiguousarray(_bytes_to_np(raw, st_dtype, shape)))

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
            out[key] = jt.array(np.ascontiguousarray(arr))
        return out

    def _load_file(filename, device="cpu"):
        with _PySafeOpen(filename, device=device) as safe:
            return {key: safe.get_tensor(key) for key in safe.keys()}

    _NP_TO_ST = {
        "float64": "F64", "float32": "F32", "float16": "F16",
        "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
        "uint8": "U8", "bool": "BOOL", "bfloat16": "BF16",
    }

    def _save_dict(tensors, metadata=None):
        header = {}
        blobs = []
        offset = 0
        for key, value in tensors.items():
            arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            arr = np.ascontiguousarray(arr)
            st_dtype = _NP_TO_ST.get(str(arr.dtype), "F32")
            if st_dtype not in _ST or _ST[st_dtype][0] is None:
                arr = arr.astype(np.float32)
                st_dtype = "F32"
            blob = arr.tobytes()
            header[key] = {
                "dtype": st_dtype,
                "shape": list(arr.shape),
                "data_offsets": [offset, offset + len(blob)],
            }
            blobs.append(blob)
            offset += len(blob)
        if metadata:
            header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
        raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        return struct.pack("<Q", len(raw_header)) + raw_header + b"".join(blobs)

    def _save_file(tensors, filename, metadata=None):
        with open(filename, "wb") as fh:
            fh.write(_save_dict(tensors, metadata))

    _st.safe_open = _PySafeOpen
    _st._jittor_torch_compat = True
    sys.modules["safetensors"].safe_open = _PySafeOpen
    try:
        import safetensors.torch as _stt
        _stt.safe_open = _PySafeOpen
        _stt.load = _load_bytes
        _stt.load_file = _load_file
        _save = lambda tensors, metadata=None: _save_dict(tensors, metadata)
        _stt.save = _save
        _stt.save_file = _save_file
    except Exception:
        pass
    try:
        import safetensors.numpy as _stn
        _stn.load_file = _load_file
        _stn.save_file = _save_file
    except Exception:
        pass

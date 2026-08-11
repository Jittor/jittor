"""Serialization adapters used by the torch compatibility layer."""

import numpy as np
import jittor as jt

from .context import registry_for


def _install_safetensors_shim(registry=None):
    """Patch safetensors.torch to load tensors without real torch storage."""
    _modules = registry_for(jt, registry).module_map
    try:
        import json
        import struct
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
    _modules["safetensors"].safe_open = _PySafeOpen
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


def install(ctx):
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    # ---- save / load ----
    # torch.save must handle BOTH tensor state-dicts AND arbitrary Python
    # objects (e.g. TrainingArguments). jittor's jt.save is numpy/pickle based
    # but chokes on live Vars and some objects, so we use standard pickle with
    # Vars converted to a portable (numpy) form and restored on load.
    import os as _os_pickle, pickle as _pickle
    _VAR_TAG = "__jt_var__"
    def _to_portable(obj, _seen=None):
        if isinstance(obj, jt.Var):
            # Var.numpy() can materialize a CUDA Var on CPU in-place. Checkpoint
            # serialization must not change the live module/optimizer tensors.
            return {_VAR_TAG: True, "data": obj.clone().numpy(), "dtype": str(obj.dtype)}
        # Drop non-picklable callables (e.g. an LR scheduler's local lr_lambda
        # closure in an extra/scheduler state_dict). torch's LambdaLR.state_dict
        # does the same -- the lambda is rebuilt on load, not restored.
        import types as _t
        if isinstance(obj, (_t.FunctionType, _t.LambdaType, _t.MethodType, _t.BuiltinFunctionType)):
            return None
        if isinstance(obj, dict):
            return {k: _to_portable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            items = [_to_portable(v) for v in obj]
            # Coerce list/tuple SUBCLASSES (e.g. the shim's local _ParamList in
            # an optimizer state_dict) to plain list/tuple -- local classes are
            # not picklable. Preserve namedtuples.
            if isinstance(obj, tuple):
                if hasattr(obj, "_fields"):
                    try:
                        return type(obj)(*items)
                    except Exception:
                        return tuple(items)
                return tuple(items)
            return list(items)
        return obj
    def _from_portable(obj):
        if isinstance(obj, dict):
            if obj.get(_VAR_TAG):
                # from_numpy preserves wide dtypes (float64/int64); jt.array narrows
                # them to float32/int32 -> torch.save/load silently downcast checkpoints.
                return g.from_numpy(obj["data"])
            return {k: _from_portable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = type(obj)
            return t(_from_portable(v) for v in obj)
        return obj
    def save(obj, f, *a, **k):
        try:
            jt.sync_all(True)
        except Exception:
            pass
        portable = _to_portable(obj)
        if hasattr(f, "write"):
            _pickle.dump(portable, f)
            return
        with open(f, "wb") as fh:
            _pickle.dump(portable, fh)

    # ---- load a REAL torch .pt checkpoint (zip archive w/ persistent-id storages) ----
    # torch.save writes a zip: <name>/data.pkl (object graph; tensors are
    # persistent_id refs to storages) + <name>/data/<key> (raw storage bytes).
    # Reconstruct tensors as jittor Vars without needing real torch.
    import io as _io, zipfile as _zipfile
    import numpy as _np_pt
    _TORCH_STORAGE_DTYPE = {
        "DoubleStorage": "float64", "FloatStorage": "float32", "HalfStorage": "float16",
        "BFloat16Storage": "bfloat16", "LongStorage": "int64", "IntStorage": "int32",
        "ShortStorage": "int16", "CharStorage": "int8", "ByteStorage": "uint8",
        "BoolStorage": "bool",
    }
    class _StorageMarker:
        def __init__(self, dtype_str): self.dtype_str = dtype_str
    def _np_from_storage(raw, dtype_str, numel):
        if dtype_str == "bfloat16":
            u16 = _np_pt.frombuffer(raw, dtype=_np_pt.uint16, count=numel).astype(_np_pt.uint32)
            return (u16 << 16).view(_np_pt.float32)   # widen bf16 -> f32 (ACL has no bf16 numpy)
        npd = {"float64": _np_pt.float64, "float32": _np_pt.float32, "float16": _np_pt.float16,
               "int64": _np_pt.int64, "int32": _np_pt.int32, "int16": _np_pt.int16,
               "int8": _np_pt.int8, "uint8": _np_pt.uint8, "bool": _np_pt.bool_}[dtype_str]
        return _np_pt.frombuffer(raw, dtype=npd, count=numel)
    def _load_torch_pt(path_or_file):
        zf = _zipfile.ZipFile(path_or_file, "r")
        names = zf.namelist()
        pkl_name = next(n for n in names if n.endswith("data.pkl"))
        data_dir = pkl_name[:-len("data.pkl")] + "data/"
        cache = {}
        def _persistent_load(pid):
            assert pid[0] == "storage", pid
            marker, key, numel = pid[1], str(pid[2]), int(pid[4])
            if key not in cache:
                cache[key] = (zf.read(data_dir + key), marker.dtype_str, numel)
            return cache[key]
        def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                               requires_grad=False, backward_hooks=None, metadata=None):
            raw, dtype_str, numel = storage
            arr = _np_from_storage(raw, dtype_str, numel)
            size = tuple(int(s) for s in size)
            n = 1
            for s in size: n *= s
            sub = arr[storage_offset:storage_offset + n]
            sub = _np_pt.ascontiguousarray(sub).reshape(size) if size else sub.reshape(())
            return jt.array(sub)
        def _rebuild_parameter(data, requires_grad=True, backward_hooks=None, *a, **k):
            return data
        class _Unpick(_pickle.Unpickler):
            def persistent_load(self, pid):
                return _persistent_load(pid)
            def find_class(self, module, name):
                if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                    return _rebuild_tensor_v2
                if module == "torch._utils" and name.startswith("_rebuild_parameter"):
                    return _rebuild_parameter
                if name.endswith("Storage") and module.startswith("torch"):
                    return _StorageMarker(_TORCH_STORAGE_DTYPE.get(name, "float32"))
                if module == "collections" and name == "OrderedDict":
                    from collections import OrderedDict
                    return OrderedDict
                if module == "torch" and name == "Size":
                    return tuple
                if module == "torch" and name == "device":
                    return lambda *a, **k: "cpu"
                try:
                    m = __import__(module, fromlist=[name]); return getattr(m, name)
                except Exception:
                    return type(name, (), {})
        return _Unpick(_io.BytesIO(zf.read(pkl_name))).load()

    def _is_zip(f):
        if hasattr(f, "read"):
            pos = f.tell(); head = f.read(2); f.seek(pos)
            return head[:2] == b"PK"
        with open(f, "rb") as fh:
            return fh.read(2)[:2] == b"PK"
    def _is_legacy_torch_pickle(path):
        # PyTorch's pre-zip serialization starts with three pickles:
        # MAGIC_NUMBER, PROTOCOL_VERSION and sys_info, then uses persistent
        # storage records. Plain torch-shim checkpoints are regular pickles and
        # must continue down the portable-pickle path.
        try:
            with open(path, "rb") as fh:
                magic = _pickle.load(fh, encoding="utf-8")
        except Exception:
            return False
        return magic == 0x1950a86a20f9469cfc6c
    def load(f, *a, **k):
        # accept map_location/weights_only/pickle_module kwargs (ignored).
        # Real torch .pt is a zip archive -> use the torch-format loader;
        # our own torch.save output is plain pickle -> _from_portable.
        path = None
        if not hasattr(f, "read"):
            path = _os_pickle.fspath(f)
            native_load = getattr(g, "_vj_native_load", None)
            if native_load is not None and path.startswith(("jittorhub://", "http://", "https://")):
                return native_load(path)
        try:
            if _is_zip(f):
                return _load_torch_pt(f)
        except Exception as _e:
            pass
        if path is not None and path.lower().endswith((".pth", ".pt", ".bin")) and _is_legacy_torch_pickle(path):
            from jittor_utils.load_pytorch import load_pytorch as _load_pytorch
            return _load_pytorch(path)
        try:
            if hasattr(f, "read"):
                obj = _pickle.load(f)
            else:
                with open(f, "rb") as fh:
                    obj = _pickle.load(fh)
        except Exception:
            native_load = getattr(g, "_vj_native_load", None)
            if native_load is not None and path is not None and path.lower().endswith(".pkl"):
                return native_load(path)
            raise
        return _from_portable(obj)
    g.save = save
    g.load = load
    # Stash the real pickle loader so adapters can restore it if torch.load gets
    # shadowed later (the torch_shim exposes cpp_extension.load(name, sources,...)
    # at the torch top level, which can mask this in some worker processes).
    g._vj_pickle_load = load
    g._vj_pickle_save = save

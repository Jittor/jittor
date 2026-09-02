"""Serialization adapters used by the torch compatibility layer."""

import numpy as np
import jittor as jt

from .context import registry_for
from .types import _make_cpu_resident, _make_cuda_resident
from ..diagnostics import EXPECTED, swallowed


def _install_safetensors_shim(registry=None):
    """Patch safetensors.torch to load tensors without real torch storage."""
    _modules = registry_for(jt, registry).module_map
    try:
        import json
        import struct
        import safetensors as _st
    except EXPECTED as exc:
        swallowed("torch/serialization.py _install_safetensors_shim: import json", exc)
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
    except EXPECTED as exc:
        swallowed("torch/serialization.py _install_safetensors_shim: import safetensors.torch as _stt", exc)
    try:
        import safetensors.numpy as _stn
        _stn.load_file = _load_file
        _stn.save_file = _save_file
    except (AttributeError, TypeError) as exc:
        swallowed("torch/serialization.py _install_safetensors_shim: import safetensors.numpy as _stn", exc)


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
            # `str(obj.dtype)` returns the torch-compat dtype OBJECT (a str
            # subclass), which pickles as a class reference. Store the bare
            # name so a checkpoint carries no importable global at all.
            return {_VAR_TAG: True, "data": obj.clone().numpy(),
                    "dtype": str.__str__(obj.dtype) if isinstance(obj.dtype, str)
                             else str(obj.dtype)}
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
                    except EXPECTED as exc:
                        swallowed("torch/serialization.py _to_portable: return type(obj)(*items)", exc)
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
        except EXPECTED as exc:
            swallowed("torch/serialization.py save: jt.sync_all(True)", exc)
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
    # ---- restricted unpickling (torch.load(weights_only=...)) -------------
    # This used to be a lie in two directions: `weights_only` was accepted and
    # dropped on the floor -- so the documented "no arbitrary code execution"
    # guarantee was simply absent -- and any class the unpickler could not
    # import was replaced by `type(name, (), {})`, an EMPTY class.  A checkpoint
    # referring to a class this process does not have therefore loaded
    # "successfully" into objects with no attributes and no data.
    _SAFE_NUMPY_NAMES = frozenset((
        "ndarray", "dtype", "_reconstruct", "scalar",
        "bool_", "int8", "int16", "int32", "int64", "intp", "longlong",
        "uint8", "uint16", "uint32", "uint64", "uintp", "ulonglong",
        "float16", "float32", "float64", "longdouble",
        "complex64", "complex128", "clongdouble", "bytes_", "str_",
    ))
    _SAFE_GLOBALS = frozenset((
        ("collections", "OrderedDict"), ("collections", "defaultdict"),
        ("collections", "Counter"), ("collections", "deque"),
        ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "list"),
        ("builtins", "dict"), ("builtins", "tuple"), ("builtins", "int"),
        ("builtins", "float"), ("builtins", "complex"), ("builtins", "bool"),
        ("builtins", "str"), ("builtins", "bytes"), ("builtins", "bytearray"),
        ("_codecs", "encode"),
        ("jittor.compat.torch.nested", "_rebuild_var_from_numpy"),
        ("jittor.compat.torch.nested", "_rebuild_nested_tensor"),
        # torch.dtype/torch.device are plain value objects here (a str subclass
        # and a name/index pair); older checkpoints reference them by name.
        ("jittor.compat.torch.types", "dtype"),
        ("jittor.compat.torch.types", "device"),
        ("torch", "dtype"), ("torch", "device"), ("torch", "Size"),
    ))

    def _is_safe_global(module, name):
        if (module, name) in _SAFE_GLOBALS:
            return True
        if module in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray",
                      "numpy.core.numeric", "numpy._core.numeric"):
            return name in _SAFE_NUMPY_NAMES
        return False

    def _resolve_global(module, name, weights_only):
        """Import module.name, or refuse -- never fabricate an empty class."""
        if weights_only and not _is_safe_global(module, name):
            raise _pickle.UnpicklingError(
                "Weights only load failed: %s.%s is not an allowed global. "
                "torch.load defaults to weights_only=True; re-run with "
                "torch.load(..., weights_only=False) only if you trust the "
                "file, because that lets the checkpoint execute arbitrary code."
                % (module, name))
        try:
            m = __import__(module, fromlist=[name])
            return getattr(m, name)
        except Exception as exc:
            raise _pickle.UnpicklingError(
                "checkpoint refers to %s.%s, which this interpreter cannot "
                "import (%s). Jittor's torch compatibility layer used to "
                "substitute an empty placeholder class here, which loaded the "
                "checkpoint successfully into objects that held none of the "
                "saved state. Install the package that defines it, or load the "
                "checkpoint with weights_only=True to keep only tensors."
                % (module, name, exc))

    class _PortableUnpickler(_pickle.Unpickler):
        """Plain-pickle loader for our own torch.save output."""
        weights_only = True

        def find_class(self, module, name):
            return _resolve_global(module, name, self.weights_only)

    def _portable_pickle_load(fh, weights_only):
        up = _PortableUnpickler(fh)
        up.weights_only = bool(weights_only)
        return up.load()

    # ---- map_location -----------------------------------------------------
    def _apply_map_location(obj, map_location, _depth=0):
        """Move every loaded Var to the requested device.

        `map_location` was documented as "(ignored)": a checkpoint saved from
        CUDA loaded onto whatever device happened to be current, so
        `torch.load(p, map_location="cpu")` -- the standard way to read a GPU
        checkpoint on a CPU-only box -- did nothing.
        """
        if map_location is None:
            return obj
        if isinstance(obj, dict):
            return {k: _apply_map_location(v, map_location, _depth + 1)
                    for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            built = [_apply_map_location(v, map_location, _depth + 1) for v in obj]
            if isinstance(obj, tuple):
                return type(obj)(*built) if hasattr(obj, "_fields") else tuple(built)
            return type(obj)(built) if type(obj) is not list else built
        if not isinstance(obj, jt.Var):
            return obj
        target = map_location
        if callable(target) and not isinstance(target, (str, dict)):
            moved = target(obj, "cpu")
            return moved if isinstance(moved, jt.Var) else obj
        if isinstance(target, dict):
            target = target.get("cpu", target.get("cuda:0"))
            if target is None:
                return obj
        name = getattr(target, "type", None) or str(target)
        name = str(name).split(":")[0]
        if name == "cpu":
            return _make_cpu_resident(obj)
        if name in ("cuda", "npu", "gpu"):
            if not jt.flags.use_cuda:
                raise RuntimeError(
                    "torch.load(map_location=%r) asks for an accelerator, but "
                    "no CUDA/NPU device is in use. Load with "
                    "map_location='cpu'." % (map_location,))
            return _make_cuda_resident(obj, force=True)
        from ..stub_policy import unimplemented
        return unimplemented(
            "torch.load(map_location=%r)" % (name,),
            "leave the tensors on whichever device happens to be current "
            "instead of the requested one",
            "Only 'cpu' and 'cuda' map_location targets are supported.",
            stub_result=obj)

    def _load_torch_pt(path_or_file, weights_only=True):
        zf = _zipfile.ZipFile(path_or_file, "r")
        names = zf.namelist()
        pkl_name = next(n for n in names if n.endswith("data.pkl"))
        data_dir = pkl_name[:-len("data.pkl")] + "data/"
        cache = {}
        def _persistent_load(pid):
            assert pid[0] == "storage", pid
            marker, key, numel = pid[1], str(pid[2]), int(pid[4])
            if key not in cache:
                # The key travels with the payload so a rebuild that cannot be
                # honoured can name the archive record it was reading.
                cache[key] = (zf.read(data_dir + key), marker.dtype_str,
                              numel, key)
            return cache[key]
        def _contiguous_stride(size):
            """The stride torch gives a freshly allocated tensor of this size."""
            stride = [1] * len(size)
            for i in range(len(size) - 2, -1, -1):
                stride[i] = stride[i + 1] * max(size[i + 1], 1)
            return tuple(stride)

        def _restore_strided(arr, offset, size, stride, key):
            """Read a tensor out of its storage the way torch described it.

            torch.save writes the whole storage and records, per tensor,
            (storage_offset, size, stride). A tensor that is a *view* -- a
            transpose, a slice, one head of a fused weight -- is therefore a
            non-contiguous description of a larger buffer. This used to slice
            `arr[offset:offset+numel]` and reshape, which reads a different set
            of elements for every such view and reports success: right shape,
            wrong numbers, no diagnostic.
            """
            numel = 1
            for s in size:
                numel *= s
            if stride is None:
                stride = _contiguous_stride(size)
            if len(stride) != len(size):
                raise _pickle.UnpicklingError(
                    "checkpoint storage %s describes a tensor of shape %s with "
                    "%d strides; the two must agree."
                    % (key, size, len(stride)))
            if numel == 0:
                return _np_pt.empty(size, dtype=arr.dtype)
            # Every element the description reaches must exist in the storage.
            last = offset
            for extent, step in zip(size, stride):
                if step < 0:
                    raise _pickle.UnpicklingError(
                        "checkpoint storage %s describes shape %s with negative "
                        "stride %s; torch does not produce negative strides, so "
                        "this file is not a tensor this loader can reconstruct."
                        % (key, size, tuple(stride)))
                last += (extent - 1) * step
            if offset < 0 or last >= arr.size:
                raise _pickle.UnpicklingError(
                    "checkpoint storage %s holds %d elements, but the tensor "
                    "saved from it (shape %s, stride %s, offset %d) reaches "
                    "element %d. The file is truncated or does not match this "
                    "reader; loading it would silently produce wrong weights."
                    % (key, arr.size, size, tuple(stride), offset, last))
            if not size:
                return arr[offset:offset + 1].reshape(())
            if tuple(stride) == _contiguous_stride(size):
                return _np_pt.ascontiguousarray(
                    arr[offset:offset + numel]).reshape(size)
            view = _np_pt.lib.stride_tricks.as_strided(
                arr[offset:], shape=size,
                strides=tuple(int(s) * arr.itemsize for s in stride))
            return _np_pt.ascontiguousarray(view)

        def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                               requires_grad=False, backward_hooks=None, metadata=None):
            raw, dtype_str, numel, key = storage
            arr = _np_from_storage(raw, dtype_str, numel)
            size = tuple(int(s) for s in size)
            stride = None if stride is None else tuple(int(s) for s in stride)
            sub = _restore_strided(arr, int(storage_offset), size, stride, key)
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
                return _resolve_global(module, name, weights_only)
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
        except EXPECTED as exc:
            swallowed("torch/serialization.py _is_legacy_torch_pickle: with open(path, 'rb') as fh:", exc)
            return False
        return magic == 0x1950a86a20f9469cfc6c
    def load(f, map_location=None, pickle_module=None, *, weights_only=None,
             mmap=None, **k):
        # torch >= 2.6 (this shim reports 2.11) defaults weights_only=True.
        # Both this and map_location used to be accepted and ignored.
        if weights_only is None:
            weights_only = True
        weights_only = bool(weights_only)
        path = None
        if not hasattr(f, "read"):
            path = _os_pickle.fspath(f)
            native_load = getattr(g, "_vj_native_load", None)
            if native_load is not None and path.startswith(("jittorhub://", "http://", "https://")):
                return _apply_map_location(native_load(path), map_location)
        _zip = False
        try:
            _zip = _is_zip(f)
        except EXPECTED as exc:
            swallowed("torch/serialization.py load: _zip = _is_zip(f)", exc)
            _zip = False
        if _zip:
            return _apply_map_location(
                _load_torch_pt(f, weights_only=weights_only), map_location)
        if path is not None and path.lower().endswith((".pth", ".pt", ".bin")) and _is_legacy_torch_pickle(path):
            from jittor_utils.load_pytorch import load_pytorch as _load_pytorch
            return _apply_map_location(_load_pytorch(path), map_location)
        try:
            if hasattr(f, "read"):
                obj = _portable_pickle_load(f, weights_only)
            else:
                with open(f, "rb") as fh:
                    obj = _portable_pickle_load(fh, weights_only)
        except _pickle.UnpicklingError:
            raise
        except EXPECTED as exc:
            swallowed("torch/serialization.py load: if hasattr(f, 'read'):", exc)
            native_load = getattr(g, "_vj_native_load", None)
            if native_load is not None and path is not None and path.lower().endswith(".pkl"):
                return _apply_map_location(native_load(path), map_location)
            raise
        return _apply_map_location(_from_portable(obj), map_location)
    g.save = save
    g.load = load
    # Stash the real pickle loader so adapters can restore it if torch.load gets
    # shadowed later (the torch_shim exposes cpp_extension.load(name, sources,...)
    # at the torch top level, which can mask this in some worker processes).
    g._vj_pickle_load = load
    g._vj_pickle_save = save

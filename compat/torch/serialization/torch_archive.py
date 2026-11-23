"""Torch zip storage reconstruction with per-reader ownership."""
import io as _io
import pickle as _pickle
import zipfile as _zipfile
import numpy as _np_pt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import jittor as jt
from ..context import get_install_context
from .security import _resolve_global

_TORCH_STORAGE_DTYPE = {
    "DoubleStorage": "float64", "FloatStorage": "float32", "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16", "LongStorage": "int64", "IntStorage": "int32",
    "ShortStorage": "int16", "CharStorage": "int8", "ByteStorage": "uint8",
    "BoolStorage": "bool",
}


class _StorageMarker:
    def __init__(self, dtype_str): self.dtype_str = dtype_str


def _np_from_storage(raw, dtype_str, numel):
    if _jittor_dtype_name(dtype_str) == "bfloat16":
        u16 = _np_pt.frombuffer(raw, dtype=_np_pt.uint16, count=numel).astype(_np_pt.uint32)
        return (u16 << 16).view(_np_pt.float32)   # widen bf16 -> f32 (ACL has no bf16 numpy)
    npd = {"float64": _np_pt.float64, "float32": _np_pt.float32, "float16": _np_pt.float16,
           "int64": _np_pt.int64, "int32": _np_pt.int32, "int16": _np_pt.int16,
           "int8": _np_pt.int8, "uint8": _np_pt.uint8, "bool": _np_pt.bool_}[dtype_str]
    return _np_pt.frombuffer(raw, dtype=npd, count=numel)


def _contiguous_stride(size):
    """The stride torch gives a freshly allocated tensor of this size."""
    stride = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        stride[i] = stride[i + 1] * max(size[i + 1], 1)
    return tuple(stride)


def _checked_layout(offset, size, stride, storage_size, key):
    """Validate the indices addressed by a checkpoint's storage view."""
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
        return numel, stride
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
    if offset < 0 or last >= storage_size:
        raise _pickle.UnpicklingError(
            "checkpoint storage %s holds %d elements, but the tensor "
            "saved from it (shape %s, stride %s, offset %d) reaches "
            "element %d. The file is truncated or does not match this "
            "reader; loading it would silently produce wrong weights."
            % (key, storage_size, size, tuple(stride), offset, last))
    return numel, stride


def _restore_strided(arr, offset, size, stride, key):
    """Read a tensor using the checkpoint's storage offset and strides."""
    numel, stride = _checked_layout(offset, size, stride, arr.size, key)
    if numel == 0:
        return _np_pt.empty(size, dtype=arr.dtype)
    if not size:
        return arr[offset:offset + 1].reshape(())
    if tuple(stride) == _contiguous_stride(size):
        return _np_pt.ascontiguousarray(
            arr[offset:offset + numel]).reshape(size)
    view = _np_pt.lib.stride_tricks.as_strided(
        arr[offset:], shape=size,
        strides=tuple(int(s) * arr.itemsize for s in stride))
    return _np_pt.ascontiguousarray(view)


def _cpu_device(*args, **kwargs):
    return "cpu"


def _meta_tensor(size, dtype, stride, offset, requires_grad):
    g = get_install_context(jt).target_namespace
    value = g.empty(size, dtype=dtype, device="meta", requires_grad=requires_grad)
    value._jittor_torch_meta_layout = (tuple(stride), int(offset))
    return value


def _load_parameter(data, requires_grad):
    g = get_install_context(jt).target_namespace
    value = g.nn.Parameter(data, requires_grad=requires_grad)
    if data.is_meta:
        value._jittor_torch_meta = True
        value._jittor_torch_meta_layout = (data.stride(), data.storage_offset())
    return value


class _ArchiveUnpickler(_pickle.Unpickler):
    def __init__(self, stream, archive, data_dir, weights_only, source_devices, map_location=None):
        super().__init__(stream)
        self.archive = archive
        self.data_dir = data_dir
        self.weights_only = bool(weights_only)
        self.source_devices = source_devices
        self.map_location = map_location
        self.storage_cache = {}

    def persistent_load(self, pid):
        assert pid[0] == "storage", pid
        marker, key, numel = pid[1], str(pid[2]), int(pid[4])
        if key not in self.storage_cache:
            source = str(pid[3])
            target = self.map_location
            if isinstance(target, dict):
                target = target.get(source, source)
            name = getattr(target, "type", None) or str(target).split(":", 1)[0]
            entry = self.data_dir + key
            if name == "meta":
                self.archive.getinfo(entry)
                raw = None
            else:
                raw = self.archive.read(entry)
            self.storage_cache[key] = (raw, marker.dtype_str, numel, key, source)
        return self.storage_cache[key]

    def _rebuild_tensor_v2(self, storage, storage_offset, size, stride,
                           requires_grad=False, backward_hooks=None, metadata=None):
        g = get_install_context(jt).target_namespace
        raw, dtype_str, numel, key, source_device = storage
        size = tuple(int(s) for s in size)
        stride = None if stride is None else tuple(int(s) for s in stride)
        offset = int(storage_offset)
        if raw is None:
            _, stride = _checked_layout(offset, size, stride, numel, key)
            value = _meta_tensor(size, dtype_str, stride, offset, bool(requires_grad))
        else:
            arr = _np_from_storage(raw, dtype_str, numel)
            sub = _restore_strided(arr, offset, size, stride, key)
            value = g.tensor(sub, dtype=dtype_str, device="cpu", requires_grad=bool(requires_grad))
        if self.source_devices is not None:
            self.source_devices[id(value)] = source_device
        return value

    def _rebuild_parameter(self, data, requires_grad=True, backward_hooks=None, *a, **k):
        parameter = _load_parameter(data, requires_grad)
        if self.source_devices is not None:
            self.source_devices[id(parameter)] = self.source_devices.get(id(data), "cpu")
        if a:
            if not isinstance(a[0], dict):
                raise _pickle.UnpicklingError("parameter state must be a dictionary")
            parameter.__dict__.update(a[0])
        return parameter

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return self._rebuild_tensor_v2
        if module == "torch._utils" and name.startswith("_rebuild_parameter"):
            return self._rebuild_parameter
        if name.endswith("Storage") and module.startswith("torch"):
            if name not in _TORCH_STORAGE_DTYPE:
                raise _pickle.UnpicklingError("unsupported checkpoint storage dtype: %s" % name)
            return _StorageMarker(_TORCH_STORAGE_DTYPE[name])
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module == "torch" and name == "Size":
            return tuple
        if module == "torch" and name == "device":
            return _cpu_device
        return _resolve_global(module, name, self.weights_only)


def _load_torch_pt(path_or_file, weights_only=True, source_devices=None, map_location=None):
    with _zipfile.ZipFile(path_or_file, "r") as archive:
        pkl_name = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        data_dir = pkl_name[:-len("data.pkl")] + "data/"
        stream = _io.BytesIO(archive.read(pkl_name))
        return _ArchiveUnpickler(stream, archive, data_dir, weights_only, source_devices,
                                 map_location).load()

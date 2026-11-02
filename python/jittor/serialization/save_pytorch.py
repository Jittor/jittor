"""PyTorch archive writer; imported only when serialization is requested."""
from contextlib import contextmanager
import io
import pickle
import sys
import threading

import jittor as jt
from jittor import nn
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import torch

class HalfStorage: pass
class BFloat16Storage: pass
class FloatStorage: pass
class LongStorage: pass
class IntStorage: pass
class ShortStorage: pass
class CharStorage: pass
class BoolStorage: pass
HalfStorage.__module__ = "torch"
BFloat16Storage.__module__ = "torch"
FloatStorage.__module__ = "torch"
LongStorage.__module__ = "torch"
IntStorage.__module__ = "torch"
ShortStorage.__module__ = "torch"
CharStorage.__module__ = "torch"
BoolStorage.__module__ = "torch"
def _rebuild_tensor_v2(*args): pass
_rebuild_tensor_v2.__module__ = "torch._utils"

targets = [HalfStorage, BFloat16Storage, FloatStorage, LongStorage, IntStorage, ShortStorage, CharStorage, BoolStorage, _rebuild_tensor_v2]

_TARGET_PATCH_LOCK = threading.RLock()


@contextmanager
def _temporary_torch_pickle_targets(targets):
    with _TARGET_PATCH_LOCK:
        bindings = []
        try:
            for target in targets:
                namespace = sys.modules[target.__module__].__dict__
                name = target.__name__
                existed = name in namespace
                original = namespace.get(name)
                bindings.append((namespace, name, existed, original))
                namespace[name] = target
            yield
        finally:
            for namespace, name, existed, original in reversed(bindings):
                if existed:
                    namespace[name] = original
                else:
                    namespace.pop(name, None)

class TensorStorage:
    def __init__(self, data):
        self.data = data

class TensorWrapper:
    def __init__(self, data):
        self.data = data
    def __reduce__(self):
        a = tuple(self.data.shape)
        # calc contiguous stride; a scalar has an empty shape and stride.
        stride = [1] * len(a)
        for i in range(len(a)-2, -1, -1):
            stride[i] = stride[i+1]*a[i+1]

        return (_rebuild_tensor_v2, (
            TensorStorage(self.data),
            0,
            tuple(self.data.shape),
            tuple(stride),
            False,
            {}
        ))

dtype_map = {
    "float16": HalfStorage,
    "bfloat16": BFloat16Storage,
    "float32": FloatStorage,
    "int64": LongStorage,
    "int32": IntStorage,
    "int16": ShortStorage,
    "int8": CharStorage,
    "bool": BoolStorage,
}

def save_pytorch(path, obj):

    serialized_storages = []
    # dfs and wrap jt.Var into TensorWrapper
    def dfs(obj):
        if isinstance(obj, jt.Var):
            return TensorWrapper(obj)
        elif isinstance(obj, dict):
            return {k: dfs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [dfs(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(dfs(x) for x in obj)
        else:
            return obj

    def persistent_id(obj):
        if isinstance(obj, TensorStorage):
            storage_type = dtype_map[_jittor_dtype_name(obj.data.dtype)]
            storage_key = len(serialized_storages)
            serialized_storages.append(obj.data)
            storage_numel = obj.data.numel()
            location = 'cpu'
            return ('storage',
                    storage_type,
                    storage_key,
                    location,
                    storage_numel)
        return None

    obj = dfs(obj)
    data_buf = io.BytesIO()
    pickle_protocol = 2
    pickler = pickle.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    with _temporary_torch_pickle_targets(tuple(targets)):
        pickler.dump(obj)
    data_value = data_buf.getvalue()

    # use previous pytorch code to save data
    # from torch.serialization import _open_zipfile_writer
    # with _open_zipfile_writer(path) as zip_file:
    #     print(data_value)
    #     zip_file.write_record('data.pkl', data_value, len(data_value))
    #     zip_file.write_record('byteorder', sys.byteorder, len(sys.byteorder))
    #     for i, v in enumerate(serialized_storages):
    #         b = v.numpy().tobytes()
    #         zip_file.write_record(f'data/{i}', b, len(b))

    import os
    path_base_name = os.path.basename(path).split(".")[0]
    contents = jt.ZipFile(path, "w")
    def write(name, data):
        if isinstance(data, str):
            write(name, data.encode())
        elif isinstance(data, bytes):
            import ctypes
            pointer = ctypes.cast(data, ctypes.c_void_p).value
            contents.write(path_base_name+'/'+name, pointer, len(data))
        elif isinstance(data, jt.Var):
            contents.write(path_base_name+'/'+name, data.raw_ptr, data.nbytes)
        else:
            raise TypeError(f"unsupported type {type(data)}")
    write("data.pkl", data_value)
    write("byteorder", sys.byteorder)
    for i, v in enumerate(serialized_storages):
        write(f"data/{i}", v)
    write("version", "3")
    del contents


if __name__ == "__main__":
    linear = nn.Linear(3, 3)
    save_pytorch("linear.bin", linear.state_dict())

    import torch
    res = torch.load("linear.bin")
    print(res)
    print(linear.state_dict())

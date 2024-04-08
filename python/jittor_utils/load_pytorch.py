import pickle
import os
import io
import shutil
from zipfile import ZipFile
import jittor as jt
import numpy as np
from typing import Any, BinaryIO, cast, Dict, Optional, Type, Tuple, Union, IO, List

loaded_storages = {}
deserialized_objects = {}

def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    if isinstance(bytes_str, bytes):
        return bytes_str.decode('ascii')
    return bytes_str

def load_tensor(contents, dtype, numel, key, location):
    if dtype == np.uint16: dtype = "bfloat16"
    name = os.path.join(prefix, "data", str(key))
    name = name.replace("\\", "/")
    loaded_storages[key] = contents.read_var(name, dtype)

def get_dtype_size(dtype):
    return jt.NanoString(dtype).dsize()

def persistent_load(saved_id):
    global contents
    assert isinstance(saved_id, tuple)
    typename = _maybe_decode_ascii(saved_id[0])
    data = saved_id[1:]
    assert typename == 'storage', \
        f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
    storage_type, key, location, numel = data
    dtype = storage_type.dtype
    if key not in loaded_storages:
        nbytes = numel
        load_tensor(contents, dtype, nbytes, key, _maybe_decode_ascii(location))
    return loaded_storages[key]

def _dtype_to_storage_type_map():
    return {
        np.float16: 'HalfStorage',
        # just fake np.uint16 as bfloat16
        np.uint16: 'BFloat16Storage',
        np.float32: 'FloatStorage',
        np.float64: 'DoubleStorage',
        np.int64: 'LongStorage',
        np.int32: 'IntStorage',
        np.int16: 'ShortStorage',
        np.int8: 'CharStorage',
        np.bool_: 'BoolStorage'
    }

def _storage_type_to_dtype_map():
    dtype_map = {
        val: key for key, val in _dtype_to_storage_type_map().items()}
    return dtype_map

def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    try:
        return _storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError:
        raise KeyError(
            f'pickle storage type "{pickle_storage_type}" is not recognized')

class StorageType():
    def __init__(self, name):
        self.dtype = _get_dtype_from_pickle_storage_type(name)

    def __str__(self):
        return f'StorageType(dtype={self.dtype})'

def jittor_rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if len(size) == 0:
        return jt.array(storage)
    record_size = np.prod(size)
    expect_stride = [1]
    for i in range(len(size)-1, 0, -1):
        expect_stride.append(expect_stride[-1]*size[i])
    expect_stride = tuple(expect_stride[::-1])
    if stride is not None and stride != expect_stride:
        if len(stride) > 1: # reshape the memory layout based on stride
            eval_list = []
            for idx in range(len(stride)):
                eval_list.append(f"@e0({idx}) * i{idx}")
            evals = "+".join(eval_list)
            return jt.array(storage[storage_offset:storage_offset+record_size]).reindex(size, [evals], extras=[jt.array(stride)])
    return jt.array(storage[storage_offset:storage_offset+record_size]).reshape(size)

def jittor_rebuild_var(data, requires_grad, backward_hooks):
    v = jt.array(data)
    v.requires_grad = requires_grad
    return v

class UnpicklerWrapper(pickle.Unpickler):  # type: ignore[name-defined]
    def find_class(self, mod_name, name):
        if mod_name.startswith("transformers"):
            return super().find_class("collections", "OrderedDict")
        if type(name) is str and 'Storage' in name:
            try:
                return StorageType(name)
            except KeyError:
                pass
        if type(name) is str and '_rebuild_tensor_v2' in name:
            return super().find_class("jittor_utils.load_pytorch", "jittor_rebuild")
        if type(name) is str and '_rebuild_parameter' in name:
            return super().find_class("jittor_utils.load_pytorch", "jittor_rebuild_var")
        
        return super().find_class(mod_name, name)

class ArrayWrapper:
    def __init__(self, storage, stride=None, size=None, requires_grad=None):
        self.requires_grad = requires_grad
        self.size = size
        self.storage = storage
        self.stride = stride

    def __str__(self):
        return self.storage.__str__()

def jittor_rebuild_direct(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if len(size) == 0:
        return ArrayWrapper(storage, stride=stride, size=size)
    storage.reshape(size)
    return ArrayWrapper(storage, stride=stride, size=size)

def jittor_rebuild_var_direct(data, requires_grad, backward_hooks):
    v = ArrayWrapper(storage, requires_grad=requires_grad)
    return v

def jittor_rebuild_direct_v0(storage, storage_offset, size, stride):
    if len(size) == 0:
        return ArrayWrapper(storage, stride=stride, size=size)
    storage.reshape(size)
    return ArrayWrapper(storage, stride=stride, size=size)

class DirectUnpicklerWrapper(pickle.Unpickler):  # type: ignore[name-defined]
    def find_class(self, mod_name, name):
        if mod_name.startswith("transformers"):
            return super().find_class("collections", "OrderedDict")

        if type(name) is str and 'Storage' in name:
            try:
                return StorageType(name)
            except KeyError:
                print("wrong type: ", name)
                pass
        if type(name) is str and '_rebuild_tensor_v2' in name:
            return super().find_class("jittor_utils.load_pytorch", "jittor_rebuild_direct")
        elif type(name) is str and '_rebuild_tensor' in name:
            return super().find_class("jittor_utils.load_pytorch", "jittor_rebuild_direct_v0")
        elif type(name) is str and '_rebuild_parameter' in name:
            return super().find_class("jittor_utils.load_pytorch", "jittor_rebuild_var_direct")
        return super().find_class(mod_name, name)

def _check_seekable(f) -> bool:
    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (str(e) + ". You can only load from a file that is seekable."
                                + " Please pre-load the data into a buffer like io.BytesIO and"
                                + " try to load from it instead.")
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)
    return False

def extract_zip(input_zip):
    input_zip = ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def _is_compressed_file(f):
    compress_modules = ['gzip']
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False

def _should_read_directly(f):
    if _is_compressed_file(f):
        return False
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False

def persistent_load_direct(saved_id):
    global deserialized_objects
    assert isinstance(saved_id, tuple)
    typename = _maybe_decode_ascii(saved_id[0])
    data = saved_id[1:]
    if typename == 'module':
        # Ignore containers that don't have any sources saved
        return data[0]
    elif typename == 'storage':
        data_type, root_key, location, size, view_metadata = data
        location = _maybe_decode_ascii(location)
        if root_key not in deserialized_objects:
            deserialized_objects[root_key] = np.zeros(size, dtype=data_type)
        storage = deserialized_objects[root_key]
        if view_metadata is not None:
            view_key, offset, view_size = view_metadata
            if view_key not in deserialized_objects:
                deserialized_objects[view_key] = storage[offset:offset + view_size]
            return deserialized_objects[view_key]
        else:
            return storage
    else:
        raise RuntimeError("Unknown saved id type: %s" % saved_id[0])

def clean_globals():
    global contents, deserialized_objects, loaded_storages, prefix
    loaded_storages = {}
    deserialized_objects = {}
    contents = None
    prefix = ""

def load_pytorch(fn_name):
    def dfs_results(result): # dfs the result dict in case of nested state dicts.
        if not isinstance(result, dict):
            return result
        for key, params in result.items():
            if isinstance(params, dict): # recursive
                result[key] = dfs_results(params)
            elif isinstance(params, ArrayWrapper): # process data
                requires_grad = params.requires_grad
                shape = params.size
                result[key] = jt.array(params.storage)
                if shape is not None and len(shape) > 0:
                    if len(params.stride) > 1: # reshape based on stride
                        eval_list = []
                        for idx in range(len(params.stride)):
                            eval_list.append(f"@e0({idx}) * i{idx}")
                        evals = "+".join(eval_list)
                        result[key] = result[key].reindex(params.size, [evals], extras=[jt.array(params.stride)])
                    else: # no need to reshape if only one dimension
                        result[key] = result[key].reshape(shape)
                if requires_grad is not None:
                    result[key].requires_grad = requires_grad
        return result
    import jittor as jt
    global contents, deserialized_objects, loaded_storages, prefix
    loaded_storages = {}
    deserialized_objects = {}
    if not (fn_name.endswith(".pth") or fn_name.endswith(".pt") or fn_name.endswith(".bin")):
        print("This function is designed to load pytorch pth format files.")
        return None
    else:
        contents = jt.ZipFile(fn_name)
        if contents.valid():
            loaded_storages = {}
            deserialized_objects = {}
            for name in contents.list():
                if "data.pkl" in name:
                    prefix = name[:-8]
                    break
            else:
                raise RuntimeError(f"zipfile <{fn_name}> format error, data.pkl not found")
                
            data_file = contents.read_var(prefix+"data.pkl")
           #import pdb; pdb.set_trace();
           #print(data_file)
            if data_file.dtype == "uint8":
                data_file = data_file.numpy().tobytes()
            else:
                data_file = data_file.data.tobytes()
            data_file = io.BytesIO(data_file)
            pickle_load_args = {'encoding': 'utf-8'}
            unpickler = UnpicklerWrapper(data_file,  **pickle_load_args)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
            result = dfs_results(result)
        else:
            deserialized_objects = {}
            f = open(fn_name, "rb")
            f_should_read_directly = _should_read_directly(f)
            MAGIC_NUMBER = 0x1950a86a20f9469cfc6c
            PROTOCOL_VERSION = 1001
            pickle_load_args = {'encoding': 'utf-8'}
            magic_number = pickle.load(f, **pickle_load_args)
            if magic_number != MAGIC_NUMBER:
                raise RuntimeError("Invalid magic number; corrupt file?")
            protocol_version = pickle.load(f, **pickle_load_args)
            if PROTOCOL_VERSION != protocol_version:
                raise RuntimeError("Invalid protocal version.")
            _sys_info = pickle.load(f, **pickle_load_args)
            unpickler = DirectUnpicklerWrapper(f, **pickle_load_args)
            unpickler.persistent_load = persistent_load_direct
            result = unpickler.load()
            offset = f.tell() if f_should_read_directly else None
            deserialized_storage_keys = pickle.load(f, **pickle_load_args)
            f.read(8)
            for key in deserialized_storage_keys:
                assert key in deserialized_objects
                dtype = deserialized_objects[key].dtype
                size = deserialized_objects[key].size * get_dtype_size(dtype)
                byte_data = f.read(size)
                deserialized_objects[key][:] = np.frombuffer(byte_data, dtype).copy()
                f.read(8)
                if offset is not None:
                    offset = f.tell()
            
            result = dfs_results(result)
        clean_globals()
        return result

if __name__ == "__main__":
    result = load_pytorch("van_base.pth")
    for key, val in result.items():
        print(key, val.shape)
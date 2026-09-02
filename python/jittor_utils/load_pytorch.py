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

def expected_stride(size):
    """The stride torch gives a freshly allocated (contiguous) tensor of this size."""
    stride = [1] * len(size)
    for i in range(len(size) - 2, -1, -1):
        stride[i] = stride[i + 1] * max(size[i + 1], 1)
    return tuple(stride)


def rebuild_strided(storage, storage_offset, size, stride):
    """Read one tensor out of its storage exactly as torch described it.

    torch.save writes the whole *storage* and records, per tensor,
    (storage_offset, size, stride).  A tensor that is a view -- a transpose, a
    slice, one head of a fused QKV weight -- is therefore a non-contiguous
    description of a larger buffer.

    This used to narrow the storage to ``[offset : offset + prod(size)]``
    *first* and then reindex the narrowed slice with the original strides.
    Those strides index the full storage, so every element the view reaches
    beyond the narrowed window fell outside the source: ``reindex`` fills
    out-of-range reads with 0.  A checkpoint saved from a non-contiguous view
    therefore loaded with the right shape, no warning, and zeros (or plain
    wrong values) in place of most of its weights.
    """
    size = tuple(int(s) for s in size)
    storage_offset = int(storage_offset)
    stride = expected_stride(size) if stride is None else tuple(int(s) for s in stride)
    if len(stride) != len(size):
        raise ValueError(
            f"checkpoint describes a tensor of shape {size} with {len(stride)} "
            f"strides; the two must agree")
    numel = 1
    for s in size:
        numel *= s
    # The zip format hands over a jt.Var, the legacy one a numpy array; both
    # are 1-D storages. Slice before converting so a contiguous tensor never
    # materializes the whole storage twice.
    total = int(storage.numel()) if isinstance(storage, jt.Var) else int(storage.size)
    if numel > 0:
        # Every element the description reaches has to exist in the storage.
        last = storage_offset
        for extent, step in zip(size, stride):
            if step < 0:
                raise ValueError(
                    f"checkpoint describes shape {size} with negative stride "
                    f"{stride}; torch does not produce negative strides, so "
                    f"this is not a tensor this loader can reconstruct")
            last += (extent - 1) * step
        if storage_offset < 0 or last >= total:
            raise ValueError(
                f"checkpoint storage holds {total} elements, but the tensor "
                f"saved from it (shape {size}, stride {stride}, offset "
                f"{storage_offset}) reaches element {last}; the file is "
                f"truncated or does not match this loader, and loading it "
                f"would silently produce wrong weights")
    if len(size) == 0:
        # jittor has no 0-d Var; a 1-element Var is what this loader has always
        # produced for a scalar. What matters here is that it is *this* element
        # of the storage, not the whole storage.
        return jt.array(storage[storage_offset:storage_offset + 1])
    if stride == expected_stride(size):
        return jt.array(
            storage[storage_offset:storage_offset + numel]).reshape(size)
    evals = " + ".join(f"@e0({idx}) * i{idx}" for idx in range(len(size)))
    if storage_offset:
        evals = f"{storage_offset} + " + evals
    source = storage if isinstance(storage, jt.Var) else jt.array(storage)
    return source.reindex(list(size), [evals], extras=[jt.array(stride)])


def jittor_rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    return rebuild_strided(storage, storage_offset, size, stride)

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
    """A tensor whose storage is not filled in yet.

    The legacy (non-zip) format unpickles the object graph first and writes the
    storage bytes afterwards, so a tensor cannot be materialized at rebuild
    time. Everything needed to materialize it later is kept here --
    ``storage_offset`` included: dropping it used to hand every view of a
    shared storage the *start* of that storage.
    """

    def __init__(self, storage, stride=None, size=None, requires_grad=None,
                 storage_offset=0):
        self.requires_grad = requires_grad
        self.size = size
        self.storage = storage
        self.stride = stride
        self.storage_offset = storage_offset

    def __str__(self):
        return self.storage.__str__()

def jittor_rebuild_direct(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    return ArrayWrapper(storage, stride=stride, size=size,
                        storage_offset=storage_offset)

def jittor_rebuild_var_direct(data, requires_grad, backward_hooks):
    # A Parameter wraps a tensor that jittor_rebuild_direct has already
    # described; keep that description and only record requires_grad. This
    # used to read a global named `storage` that does not exist, so every
    # legacy checkpoint holding a Parameter raised NameError here.
    if isinstance(data, ArrayWrapper):
        data.requires_grad = requires_grad
        return data
    return ArrayWrapper(data, requires_grad=requires_grad)

def jittor_rebuild_direct_v0(storage, storage_offset, size, stride):
    return ArrayWrapper(storage, stride=stride, size=size,
                        storage_offset=storage_offset)

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

def materialize_wrappers(result):
    """Turn every deferred ArrayWrapper into a Var, recursing into sub-dicts.

    Runs after the legacy format's storage bytes have been read, which is the
    earliest moment a tensor can be built.
    """
    if not isinstance(result, dict):
        return result
    for key, params in result.items():
        if isinstance(params, dict): # recursive
            result[key] = materialize_wrappers(params)
        elif isinstance(params, ArrayWrapper): # process data
            requires_grad = params.requires_grad
            if params.size is None:
                result[key] = jt.array(params.storage)
            else:
                # Same reconstruction as the zip path: honour offset and
                # stride, or say why the description cannot be honoured.
                result[key] = rebuild_strided(
                    params.storage, params.storage_offset,
                    params.size, params.stride)
            if requires_grad is not None:
                result[key].requires_grad = requires_grad
    return result


def load_pytorch(fn_name):
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
            result = materialize_wrappers(result)
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
            
            result = materialize_wrappers(result)
        clean_globals()
        return result

if __name__ == "__main__":
    result = load_pytorch("van_base.pth")
    for key, val in result.items():
        print(key, val.shape)
"""Public Torch save/load and portable tensor payload conversion."""
from collections import OrderedDict as _OrderedDict
import os as _os_pickle
import pickle as _pickle
import numpy as np
import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from ..context import get_install_context
from ..types import _make_cpu_resident, _make_cuda_resident, _move_to_cuda_index
from ...diagnostics import EXPECTED, swallowed
from .security import _portable_pickle_load
from .torch_archive import _load_torch_pt

_VAR_TAG = "__jt_var__"
_LIST_IN_PROGRESS = object()
_STORAGE_REFS = object()


def _loaded_storage_alias(storage, requires_grad, parameter):
    """Distinct loaded holders share bytes, not autograd version counters."""
    g = get_install_context(jt).target_namespace
    from ..frontend import tensor_frontend
    with tensor_frontend(g.Var):
        value = storage._storage_data_alias()
        value.requires_grad = bool(requires_grad)
        if parameter:
            value = g.nn.Parameter(value, requires_grad=value.requires_grad)
    return value


def _dense_snapshot_region(value):
    """Process-local physical identity, only after source materialization.

    General offset/strided/cross-dtype storage reconstruction is not covered.
    No address or allocator identity is serialized.
    """
    pointer = int(value._mem_ptr_now)
    shape = tuple(int(size) for size in value.shape)
    strides = tuple(int(step) for step in value._storage_strides())
    canonical = []
    step = 1
    for size in reversed(shape):
        canonical.insert(0, step)
        step *= size
    if (not pointer or not int(value.numel()) or not value._storage_is_contiguous()
            or int(value._storage_offset()) != 0 or strides != tuple(canonical)):
        return None
    return (value.location(), int(value.placement_backend), int(value.device_id),
            pointer, int(value.nbytes), _jittor_dtype_name(value.dtype), shape, strides)


def _validate_ordered_attrs(attrs, _seen=None):
    """Do not silently replace callable OrderedDict metadata with None."""
    import types
    if _seen is None:
        _seen = set()
    if id(attrs) in _seen:
        return
    _seen.add(id(attrs))
    if isinstance(attrs, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        raise TypeError("callable OrderedDict attributes are not portable")
    if isinstance(attrs, dict):
        for item in attrs.values():
            _validate_ordered_attrs(item, _seen)
        if isinstance(attrs, _OrderedDict):
            _validate_ordered_attrs(vars(attrs), _seen)
    elif isinstance(attrs, (list, tuple)):
        for item in attrs:
            _validate_ordered_attrs(item, _seen)


def _to_portable(obj, snapshots, _memo=None):
    if _memo is None:
        _memo = {}
    if id(obj) in _memo:
        if _memo[id(obj)] is _LIST_IN_PROGRESS:
            raise TypeError("recursive list subclass is not portable")
        return _memo[id(obj)]
    g = get_install_context(jt).target_namespace
    # Tensor.shape is backed by jittor_core.NanoVector. PyTorch exposes the
    # equivalent value as the picklable torch.Size tuple subclass.
    if (type(obj).__name__ == "NanoVector"
            and type(obj).__module__ == "jittor_core"):
        value = g.Size(tuple(int(value) for value in obj))
        _memo[id(obj)] = value
        return value
    if jt.core._is_tensor_carrier(obj):
        # Batched fetch supplies a host copy without moving live tensors.
        # Persist the native name, not the frontend object's torch-prefixed
        # display string or an importable Python dtype class reference.
        value = {_VAR_TAG: True, "data": snapshots[id(obj)],
                "dtype": _jittor_dtype_name(obj.dtype),
                "requires_grad": bool(obj.requires_grad),
                "parameter": isinstance(obj, g.nn.Parameter),
                "device": str(getattr(obj, "device", "cpu"))}
        _memo[id(obj)] = value
        return value
    # Drop non-picklable callables (e.g. an LR scheduler's local lr_lambda
    # closure in an extra/scheduler state_dict). torch's LambdaLR.state_dict
    # does the same -- the lambda is rebuilt on load, not restored.
    import types as _t
    if isinstance(obj, (_t.FunctionType, _t.LambdaType, _t.MethodType, _t.BuiltinFunctionType)):
        return None
    if isinstance(obj, dict):
        value = _OrderedDict() if isinstance(obj, _OrderedDict) else {}
        _memo[id(obj)] = value
        for key, item in obj.items():
            value[key] = _to_portable(item, snapshots, _memo)
        if isinstance(obj, _OrderedDict):
            _validate_ordered_attrs(vars(obj))
            value.__dict__ = _to_portable(vars(obj), snapshots, _memo)
        return value
    if isinstance(obj, list):
        value = []
        _memo[id(obj)] = value
        value.extend(_to_portable(v, snapshots, _memo) for v in obj)
        return value
    if isinstance(obj, tuple):
        items = [_to_portable(v, snapshots, _memo) for v in obj]
        # A recursive tuple/list graph may have memoized this tuple while
        # visiting its children, as in copy.deepcopy's immutable handling.
        if id(obj) in _memo:
            return _memo[id(obj)]
        if hasattr(obj, "_fields"):
            try:
                value = type(obj)(*items)
            except EXPECTED as exc:
                swallowed("torch/serialization.py _to_portable: return type(obj)(*items)", exc)
                value = tuple(items)
        else:
            value = tuple(items)
        _memo[id(obj)] = value
        return value
    return obj


def _from_portable(obj, source_devices, _memo=None):
    if _memo is None:
        _memo = {}
    if id(obj) in _memo:
        if _memo[id(obj)] is _LIST_IN_PROGRESS:
            raise TypeError("recursive list subclass is not portable")
        return _memo[id(obj)]
    g = get_install_context(jt).target_namespace
    if isinstance(obj, dict):
        if obj.get(_VAR_TAG):
            # from_numpy preserves wide dtypes (float64/int64); jt.array narrows
            # them to float32/int32 -> torch.save/load silently downcast checkpoints.
            # Pickle preserves the identity of each fetched ndarray. The save
            # side shares that object only for the same physical dense region,
            # never merely equal values. Keep old payloads readable unchanged.
            key = (id(obj["data"]), obj.get("dtype"), obj.get("device", "cpu"))
            storage_key = (_STORAGE_REFS, key)
            if storage_key not in _memo:
                # A backing storage has no user's autograd history. Restore
                # trainability only on fresh holders, including the first one.
                _memo[storage_key] = g.tensor(obj["data"], dtype=obj.get("dtype"),
                                            device="cpu", requires_grad=False)
            value = _loaded_storage_alias(
                _memo[storage_key], obj.get("requires_grad", False),
                obj.get("parameter", False))
            source_devices[id(value)] = obj.get("device", "cpu")
            source_devices.setdefault(_STORAGE_REFS, {})[id(value)] = key
            _memo[id(obj)] = value
            return value
        value = _OrderedDict() if isinstance(obj, _OrderedDict) else {}
        _memo[id(obj)] = value
        for key, item in obj.items():
            value[key] = _from_portable(item, source_devices, _memo)
        if isinstance(obj, _OrderedDict):
            value.__dict__ = _from_portable(vars(obj), source_devices, _memo)
        return value
    if type(obj) is list:
        value = []
        _memo[id(obj)] = value
        value.extend(_from_portable(v, source_devices, _memo) for v in obj)
        return value
    if isinstance(obj, (list, tuple)):
        if isinstance(obj, list):
            _memo[id(obj)] = _LIST_IN_PROGRESS
        t = type(obj)
        values = [_from_portable(v, source_devices, _memo) for v in obj]
        if id(obj) in _memo and _memo[id(obj)] is not _LIST_IN_PROGRESS:
            return _memo[id(obj)]
        value = t(*values) if hasattr(obj, "_fields") else t(values)
        _memo[id(obj)] = value
        return value
    return obj


class _TensorSnapshots:
    def __init__(self):
        self.tensors = []
        self.seen = set()
        self.values = {}
        self.fetch_tensors = []
        self.aliases = {}
        self.regions = {}

    def collect(self, value):
        if id(value) in self.seen:
            return
        self.seen.add(id(value))
        if jt.core._is_tensor_carrier(value):
            self.tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                self.collect(item)
            if isinstance(value, _OrderedDict):
                _validate_ordered_attrs(vars(value))
                self.collect(vars(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.collect(item)

    def prepare(self, preserve_storage):
        if not preserve_storage:
            self.fetch_tensors = self.tensors
            return
        # Finish lazy writes before observing actual addresses. Reading a
        # nonmaterialized holder id is not evidence of shared storage.
        jt.sync_all(True)
        regions = {}
        for value in self.tensors:
            key = _dense_snapshot_region(value)
            self.regions[id(value)] = key
            leader = regions.get(key) if key is not None else None
            if leader is None:
                leader = id(value)
                self.fetch_tensors.append(value)
                if key is not None:
                    regions[key] = leader
            self.aliases[id(value)] = leader

    def capture(self, *arrays):
        self.values.update((id(value), np.array(array, copy=True))
                           for value, array in zip(self.fetch_tensors, arrays))
        for identity, leader in self.aliases.items():
            self.values[identity] = self.values[leader]


def _snapshot_tensors(obj, preserve_storage=False):
    """Host values for codecs without changing source tensor residency."""
    snapshot = _TensorSnapshots()
    snapshot.collect(obj)
    snapshot.prepare(preserve_storage)
    if snapshot.tensors:
        # Fetch copies to host without moving the source sharing group.
        jt.fetch(*snapshot.fetch_tensors, snapshot.capture)
    jt.sync_all(True)
    if preserve_storage:
        for value in snapshot.tensors:
            before = snapshot.regions[id(value)]
            if before is not None and _dense_snapshot_region(value) != before:
                raise RuntimeError("checkpoint source storage changed during fetch")
    if len(snapshot.values) != len(snapshot.tensors):
        raise RuntimeError("checkpoint tensor fetch did not complete")
    return snapshot.values


def save(obj, f, *a, **k):
    portable = _to_portable(obj, _snapshot_tensors(obj, preserve_storage=True))
    if hasattr(f, "write"):
        _pickle.dump(portable, f)
        return
    with open(f, "wb") as fh:
        _pickle.dump(portable, fh)


def _preserve_parameter(obj, moved, g):
    if moved is not obj and isinstance(obj, g.nn.Parameter):
        on_cpu = moved.placement_backend == 0 if moved.placement_backend >= 0 else moved.location() == "cpu"
        obj.assign(moved.detach())
        if on_cpu:
            return _make_cpu_resident(obj, inplace=True)
        return obj
    return moved


def _apply_map_location(obj, map_location, _depth=0, source_devices=None, _memo=None):
    if _memo is None:
        _memo = {}
    if id(obj) in _memo:
        if _memo[id(obj)] is _LIST_IN_PROGRESS:
            raise TypeError("recursive list subclass is not portable")
        return _memo[id(obj)]
    storage = (source_devices or {}).get(_STORAGE_REFS, {}).get(id(obj))
    storage_key = (_STORAGE_REFS, storage)
    if storage is not None:
        g = get_install_context(jt).target_namespace
        if storage_key not in _memo:
            # Map an untracked backing once, then rebuild EVERY user holder.
            # Moving a trainable first Tensor would otherwise create a copy
            # grad_fn and make checkpoint leaf behavior depend on dict order.
            backing = _loaded_storage_alias(obj, False, False)
            devices = {id(backing): source_devices[id(obj)]}
            _memo[storage_key] = _apply_map_location_impl(
                backing, map_location, _depth, devices, _memo)
        value = _loaded_storage_alias(_memo[storage_key], obj.requires_grad,
                                      isinstance(obj, g.nn.Parameter))
    else:
        value = _apply_map_location_impl(obj, map_location, _depth, source_devices, _memo)
    _memo[id(obj)] = value
    return value


def _apply_map_location_impl(obj, map_location, _depth, source_devices, _memo):
    """Move every loaded Var to the requested device.

    `map_location` was documented as "(ignored)": a checkpoint saved from
    CUDA loaded onto whatever device happened to be current, so
    `torch.load(p, map_location="cpu")` -- the standard way to read a GPU
    checkpoint on a CPU-only box -- did nothing.
    """
    g = get_install_context(jt).target_namespace
    if map_location is None and not source_devices:
        return obj
    if isinstance(obj, dict):
        value = _OrderedDict() if isinstance(obj, _OrderedDict) else {}
        _memo[id(obj)] = value
        for key, item in obj.items():
            value[key] = _apply_map_location(item, map_location, _depth + 1, source_devices, _memo)
        if isinstance(obj, _OrderedDict):
            value.__dict__ = _apply_map_location(vars(obj), map_location, _depth + 1, source_devices, _memo)
        return value
    if type(obj) is list:
        value = []
        _memo[id(obj)] = value
        value.extend(_apply_map_location(v, map_location, _depth + 1, source_devices, _memo) for v in obj)
        return value
    if isinstance(obj, (list, tuple)):
        if isinstance(obj, list):
            _memo[id(obj)] = _LIST_IN_PROGRESS
        built = [_apply_map_location(v, map_location, _depth + 1, source_devices, _memo) for v in obj]
        if id(obj) in _memo and _memo[id(obj)] is not _LIST_IN_PROGRESS:
            return _memo[id(obj)]
        if isinstance(obj, tuple):
            if hasattr(obj, "_fields"):
                value = type(obj)(*built)
            else:
                value = tuple(built) if type(obj) is tuple else type(obj)(built)
        else:
            value = type(obj)(built)
        _memo[id(obj)] = value
        return value
    if not jt.core._is_tensor_carrier(obj):
        return obj
    source = (source_devices or {}).get(id(obj), "cpu")
    target = map_location if map_location is not None else source
    if callable(target) and not isinstance(target, (str, dict)):
        moved = target(obj, source)
        if jt.core._is_tensor_carrier(moved):
            return _preserve_parameter(obj, moved, g)
        target = source
    if isinstance(target, dict):
        target = target.get(source, source)
    name = getattr(target, "type", None) or str(target)
    name = str(name).split(":")[0]
    if name == "cpu":
        from ..frontend import tensor_frontend
        with tensor_frontend(g.Var):
            return _preserve_parameter(obj, _make_cpu_resident(
                obj, inplace=isinstance(obj, g.nn.Parameter)), g)
    if name in ("cuda", "npu", "gpu"):
        # `placement_backend` is 0 for an *explicitly CPU-placed* Var and > 0 for
        # an accelerator one (`types.py` reads it the same way: `== 0` is CPU).
        # Testing it as `>= 0` sent a CPU-placed Var down the direct-move path,
        # so the diagnosis below never ran: `torch.load(p, map_location="cuda")`
        # on a build with no accelerator raised the raw
        # "Invalid cuda device index 0; visible device count is 0" instead of
        # naming `map_location`. Only a Var that is already on an accelerator
        # belongs on that path.
        if obj.placement_backend > 0:
            if name == "gpu":
                target = "cuda" + str(target)[3:]
            moved = _make_cuda_resident(obj, force=True,
                                       inplace=isinstance(obj, g.nn.Parameter), device=target)
            return _preserve_parameter(obj, moved, g)
        if not jt.flags.use_cuda:
            if not jt.has_cuda and not getattr(jt.flags, "use_acl", False):
                raise RuntimeError(
                    "torch.load(map_location=%r) asks for an accelerator, but "
                    "no CUDA/NPU device is available. Load with "
                    "map_location='cpu'." % (map_location,))
            jt.flags.use_cuda = 1
        if name == "gpu":
            target = "cuda" + str(target)[3:]
        destination = g.device(target)
        from ..frontend import tensor_frontend
        with tensor_frontend(g.Var):
            if destination.index is not None:
                with jt.flag_scope(device_id=destination.index):
                    moved = _make_cuda_resident(obj, force=True,
                                                inplace=isinstance(obj, g.nn.Parameter))
                    moved = _move_to_cuda_index(moved, destination)
            else:
                moved = _make_cuda_resident(obj, force=True,
                                            inplace=isinstance(obj, g.nn.Parameter))
            return _preserve_parameter(obj, moved, g)
    from ...stub_policy import unimplemented
    return unimplemented(
        "torch.load(map_location=%r)" % (name,),
        "leave the tensors on whichever device happens to be current "
        "instead of the requested one",
        "Only 'cpu' and 'cuda' map_location targets are supported.",
        stub_result=obj)


def _is_zip(f):
    if hasattr(f, "read"):
        pos = f.tell()
        head = f.read(2)
        f.seek(pos)
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
            magic = _portable_pickle_load(fh, weights_only=True)
    except (_pickle.UnpicklingError,) + EXPECTED as exc:
        swallowed("torch/serialization.py _is_legacy_torch_pickle: with open(path, 'rb') as fh:", exc)
        return False
    return magic == 0x1950a86a20f9469cfc6c


def load(f, map_location=None, pickle_module=None, *, weights_only=None,
         mmap=None, **k):
    # torch >= 2.6 (this shim reports 2.11) defaults weights_only=True.
    # Both this and map_location used to be accepted and ignored.
    g = get_install_context(jt).target_namespace
    if weights_only is None:
        weights_only = True
    weights_only = bool(weights_only)
    path = None
    if not hasattr(f, "read"):
        path = _os_pickle.fspath(f)
        native_load = get_install_context(g).state["core_native_api"]["load"]
        if native_load is not None and path.startswith(("jittorhub://", "http://", "https://")):
            if weights_only:
                raise _pickle.UnpicklingError(
                    "weights_only=True cannot use the native URL loader; load a local "
                    "supported checkpoint or explicitly choose weights_only=False for trusted input")
            return _apply_map_location(native_load(path), map_location)
    _zip = False
    try:
        _zip = _is_zip(f)
    except EXPECTED as exc:
        swallowed("torch/serialization.py load: _zip = _is_zip(f)", exc)
        _zip = False
    if _zip:
        source_devices = {}
        return _apply_map_location(
            _load_torch_pt(f, weights_only=weights_only, source_devices=source_devices),
            map_location, source_devices=source_devices)
    if path is not None and path.lower().endswith((".pth", ".pt", ".bin")) and _is_legacy_torch_pickle(path):
        if weights_only:
            raise _pickle.UnpicklingError(
                "weights_only=True cannot safely load the legacy Torch format: the native "
                "reader has no restricted-Unpickler interface; use weights_only=False only "
                "for a trusted checkpoint")
        from jittor.serialization.load_pytorch import load_pytorch as _load_pytorch
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
        native_load = get_install_context(g).state["core_native_api"]["load"]
        if (not weights_only and native_load is not None and path is not None
                and path.lower().endswith(".pkl")):
            return _apply_map_location(native_load(path), map_location)
        raise
    source_devices = {}
    restored = _from_portable(obj, source_devices)
    return _apply_map_location(restored, map_location, source_devices=source_devices)

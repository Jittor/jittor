"""Public Torch save/load and portable tensor payload conversion."""
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


def _to_portable(obj, snapshots):
    g = get_install_context(jt).target_namespace
    if isinstance(obj, jt.Var):
        # Batched fetch supplies a host copy without moving live tensors.
        # Persist the native name, not the frontend object's torch-prefixed
        # display string or an importable Python dtype class reference.
        return {_VAR_TAG: True, "data": snapshots[id(obj)],
                "dtype": _jittor_dtype_name(obj.dtype),
                "requires_grad": bool(obj.requires_grad),
                "parameter": isinstance(obj, g.nn.Parameter),
                "device": str(getattr(obj, "device", "cpu"))}
    # Drop non-picklable callables (e.g. an LR scheduler's local lr_lambda
    # closure in an extra/scheduler state_dict). torch's LambdaLR.state_dict
    # does the same -- the lambda is rebuilt on load, not restored.
    import types as _t
    if isinstance(obj, (_t.FunctionType, _t.LambdaType, _t.MethodType, _t.BuiltinFunctionType)):
        return None
    if isinstance(obj, dict):
        return {k: _to_portable(v, snapshots) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = [_to_portable(v, snapshots) for v in obj]
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


def _from_portable(obj, source_devices):
    g = get_install_context(jt).target_namespace
    if isinstance(obj, dict):
        if obj.get(_VAR_TAG):
            # from_numpy preserves wide dtypes (float64/int64); jt.array narrows
            # them to float32/int32 -> torch.save/load silently downcast checkpoints.
            value = g.tensor(obj["data"], dtype=obj.get("dtype"),
                             device="cpu",
                             requires_grad=bool(obj.get("requires_grad", False)))
            if obj.get("parameter", False):
                value = g.nn.Parameter(value, requires_grad=value.requires_grad)
            source_devices[id(value)] = obj.get("device", "cpu")
            return value
        return {k: _from_portable(v, source_devices) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        values = [_from_portable(v, source_devices) for v in obj]
        return t(*values) if hasattr(obj, "_fields") else t(values)
    return obj


class _TensorSnapshots:
    def __init__(self):
        self.tensors = []
        self.seen = set()
        self.values = {}

    def collect(self, value):
        if id(value) in self.seen:
            return
        self.seen.add(id(value))
        if isinstance(value, jt.Var):
            self.tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                self.collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.collect(item)

    def capture(self, *arrays):
        self.values.update((id(value), np.array(array, copy=True))
                           for value, array in zip(self.tensors, arrays))


def save(obj, f, *a, **k):
    snapshot = _TensorSnapshots()
    snapshot.collect(obj)
    if snapshot.tensors:
        # Fetch copies to host without moving the source sharing group.
        jt.fetch(*snapshot.tensors, snapshot.capture)
    jt.sync_all(True)
    if len(snapshot.values) != len(snapshot.tensors):
        raise RuntimeError("checkpoint tensor fetch did not complete")
    portable = _to_portable(obj, snapshot.values)
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


def _apply_map_location(obj, map_location, _depth=0, source_devices=None):
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
        return {k: _apply_map_location(v, map_location, _depth + 1, source_devices)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        built = [_apply_map_location(v, map_location, _depth + 1, source_devices) for v in obj]
        if isinstance(obj, tuple):
            return type(obj)(*built) if hasattr(obj, "_fields") else tuple(built)
        return type(obj)(built) if type(obj) is not list else built
    if not isinstance(obj, jt.Var):
        return obj
    source = (source_devices or {}).get(id(obj), "cpu")
    target = map_location if map_location is not None else source
    if callable(target) and not isinstance(target, (str, dict)):
        moved = target(obj, source)
        if isinstance(moved, jt.Var):
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
        if obj.placement_backend >= 0:
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

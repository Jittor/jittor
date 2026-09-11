"""Stable Torch Tensor method implementations; install only binds these objects."""
from importlib import import_module
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from ...context import get_install_context
from ..core import _promote_pair

_owner = import_module(__package__)
_NativeVar = _owner.jt.Var

_DTYPE_BYTES = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
    "uint16": 2, "uint32": 4, "uint64": 8, "bool": 1,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
    "complex64": 8, "complex128": 16,
}


_FP_DTYPES = {"float16", "float32", "float64", "bfloat16",
              "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2",
              "float8_e5m2fnuz", "float8_e8m0fnu", "float4_e2m1fn_x2"}


_DTYPE_TO_TYPENAME = {
    "float32": "torch.FloatTensor", "float64": "torch.DoubleTensor",
    "float16": "torch.HalfTensor", "bfloat16": "torch.BFloat16Tensor",
    "int64": "torch.LongTensor", "int32": "torch.IntTensor",
    "int16": "torch.ShortTensor", "int8": "torch.CharTensor",
    "uint8": "torch.ByteTensor", "bool": "torch.BoolTensor",
}


_TYPENAME_TO_DTYPE = {v: k for k, v in _DTYPE_TO_TYPENAME.items()}


_TYPENAME_TO_DTYPE.update({v.replace("torch.", "torch.cuda."): k
                           for k, v in _DTYPE_TO_TYPENAME.items()})


def _dtype_get(self):
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _DTYPE_OBJS = _native['_DTYPE_OBJS']
    _d = _native['_native_desc']
    name = str(_d.__get__(self, type(self)))
    return _DTYPE_OBJS.get(name, name)


def _numpy_data_value(value):
    if isinstance(value, _NativeVar):
        return value.numpy()
    if isinstance(value, tuple):
        return tuple(_numpy_data_value(item) for item in value)
    if isinstance(value, list):
        return [_numpy_data_value(item) for item in value]
    return value


def _write_data_owner_numpy(view, value, slices):
    _context = get_install_context(_owner.jt)
    Var = _context.state["Var"]
    _native = _context.state["tensor_native_api"]
    _native_data_descriptor = _native['_native_data_descriptor']
    owner = getattr(view, "_torch_data_owner", None)
    if not isinstance(owner, _NativeVar) or _native_data_descriptor is None:
        return False
    target = _native_data_descriptor.__get__(owner, Var)
    for index in getattr(view, "_torch_data_path", ()):
        target = target[_numpy_data_value(index)]
    target[_numpy_data_value(slices)] = _numpy_data_value(value)
    return True


def _is_basic_data_index(index):
    if isinstance(index, tuple):
        return all(_is_basic_data_index(item) for item in index)
    if index is None or index is Ellipsis or isinstance(index, slice):
        return True
    return isinstance(index, _owner.numbers.Integral) and not isinstance(
        index, (bool, _owner.np.bool_)
    )


def _data_owner_uses_device(owner):
    if owner.placement_backend >= 0:
        return owner.placement_backend != 0
    try:
        location = owner.location()
    except _owner.EXPECTED as exc:
        _owner.swallowed("torch/installers/tensor.py _data_owner_uses_device: location = owner.location()", exc)
        location = None
    if location == "device":
        return True
    if location == "cpu":
        return False
    return bool(
        _owner.jt.flags.use_cuda or getattr(getattr(_owner.jt, "compiler", None), "has_acl", 0)
    )


def _restore_trainable_state(value, was_trainable):
    if was_trainable and value.is_stop_grad():
        value.start_grad()
    elif not was_trainable and not value.is_stop_grad():
        value.stop_grad()


def _assign_data_owner(view, value, extra_path=()):
    """Write a detached ``.data`` alias back without leaving the device."""
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _orig_getitem = _native['_orig_getitem']

    owner = getattr(view, "_torch_data_owner", None)
    if not isinstance(owner, _NativeVar):
        return False

    view_path = getattr(view, "_torch_data_path", ())
    path = view_path + tuple(extra_path)
    bases = []
    target = owner
    for index in path:
        bases.append((target, index))
        target = _orig_getitem(target, index)

    updated = value if isinstance(value, _NativeVar) else _owner.jt.array(value)
    if updated.numel() == 1 and target.numel() != 1:
        updated = updated.broadcast(target.shape)
    for base, index in reversed(bases):
        updated = base.setitem(index, updated)

    owner_was_trainable = not owner.is_stop_grad()
    owner.assign(updated)
    _restore_trainable_state(owner, owner_was_trainable)

    # A retained ``data = parameter.data`` alias observes its own mutation,
    # just like a Torch tensor sharing the parameter storage.
    view_updated = updated
    for index in view_path:
        view_updated = _orig_getitem(view_updated, index)
    view_was_trainable = not view.is_stop_grad()
    # The detached data alias must not share its requires-grad bit with
    # the newly assigned owner Var. Native assignment preserves that bit
    # by setting it on the RHS, so give the alias its own detached node.
    view.assign(view_updated.detach())
    _restore_trainable_state(view, view_was_trainable)
    return True


def _set_data_owner(view, slices, value):
    owner = getattr(view, "_torch_data_owner", None)
    if not isinstance(owner, _NativeVar):
        return False
    if _data_owner_uses_device(owner) and _is_basic_data_index(slices):
        return _assign_data_owner(view, value, (slices,))
    # Native Jittor exposes ``Var.data`` as a shared NumPy DataView. Keep
    # that exact CPU behavior so writes made after a sync remain visible to
    # already-materialized outputs which share the same storage.
    return _write_data_owner_numpy(view, value, slices)


def _torch_setitem(self, slices, value):
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _orig_setitem = _native['_orig_setitem']
    if _set_data_owner(self, slices, value):
        return self
    try:
        mask = slices
        if isinstance(mask, _NativeVar) and _jittor_dtype_name(mask.dtype) in ("bool", "uint8") \
                and isinstance(value, _NativeVar) \
                and len(mask.shape) < len(self.shape):
            # Region selected by a lower-rank bool mask has shape
            # (N, *self.shape[mask.ndim:]). Drop only provably redundant
            # leading singleton axes from value until ranks agree.
            region_rank = 1 + (len(self.shape) - len(mask.shape))
            while len(value.shape) > region_rank and value.shape[0] == 1:
                value = value.squeeze(0)
    except _owner.EXPECTED as exc:
        _owner.swallowed("torch/installers/tensor.py _torch_setitem: mask = slices", exc)
    result = _orig_setitem(self, slices, value)
    return result


def _ip(self, value):
    # In-place op x.OP_(...) -> x becomes `value` (which usually depends on x,
    # e.g. div_/mul_/add_). assign() ALREADY keeps x grad-connected when `value`
    # is grad-connected, so grad flows through the in-place op (torch parity).
    # But start_grad() RESETS x's grad node and SEVERS that just-built graph
    # (the same start_grad-severing bug behind the DPO/requires_grad fix), which
    # silently zeroed grads through x.div_()/etc (GRPO temperature scaling).
    # So only start_grad if assign actually left x stopped (a constant value like
    # fill_/zero_ on a previously-trainable leaf) -- never on an already-connected x.
    if _assign_data_owner(self, value):
        return self
    target = self
    was_trainable = not target.is_stop_grad()
    value_is_trainable = isinstance(value, _NativeVar) and not value.is_stop_grad()
    if not was_trainable and value_is_trainable:
        # assign() deliberately copies the old holder's stop-grad state onto
        # the new graph. Torch instead lets a constant destination become
        # differentiable when an in-place result depends on a trainable source.
        target._update(value)
        return self
    target.assign(value)
    if was_trainable and target.is_stop_grad():
        target.start_grad()
    elif not was_trainable and not target.is_stop_grad():
        target.stop_grad()
    return self


def _copy_(self, other, non_blocking=False):
    src = other if isinstance(other, _NativeVar) else _owner.jt.array(other)
    return _ip(self, src.cast(_jittor_dtype_name(self.dtype)) if hasattr(self, "dtype") else src)


def _norm_size(args):
    # torch allows new_ones(2,3), new_ones((2,3)), or new_ones(<NanoVector/Size>)
    # -- unwrap any single iterable that isn't itself a scalar int/Var.
    if len(args) == 1 and not isinstance(args[0], (int, _owner.jt.Var)) \
            and hasattr(args[0], "__len__"):   # tuple/list/NanoVector/Size
        args = tuple(args[0])
    # torch accepts 0-d int Vars / numpy ints as sizes (e.g. longformer computes
    # dims via torch.div); jittor's factories need plain ints -- coerce.
    return tuple(int(s.item()) if isinstance(s, _owner.jt.Var) else int(s) for s in args)


def _resolve_size(size, kw):
    # torch allows new_ones(2,3), new_ones((2,3)) AND the keyword form
    # new_ones(size=(2,3)) (used by longformer's new_ones(size=mask.size())).
    if not size and "size" in kw:
        return (kw["size"],)
    return size


def _new_finish(v, device=None, requires_grad=False):
    if _owner._device_is_cpu(device):
        v = _owner._make_cpu_resident(v)
    elif _owner._device_is_cuda(device):
        if v.placement_backend < 0:
            _owner._set_use_cuda()
        v = _owner._make_cuda_resident(v, force=True, device=device)
    if _owner._device_is_meta(device):
        _owner._set_meta_placeholder(v)
    if requires_grad:
        v.requires_grad_(True)
        _owner._torch_register_leaf(v)
    return v


def _new_scope(self, device):
    from ...frontend import tensor_frontend
    context = get_install_context(_owner.jt)
    return tensor_frontend(context.target_namespace.Var, device=device, like=self)


def _new_ones(self, *size, dtype=None, device=None, requires_grad=False, **kw):
    dt = _owner._dtype_to_str(dtype) if dtype is not None else _jittor_dtype_name(self.dtype)
    with _new_scope(self, device):
        return _new_finish(_owner.jt.ones(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)


def _new_zeros(self, *size, dtype=None, device=None, requires_grad=False, **kw):
    dt = _owner._dtype_to_str(dtype) if dtype is not None else _jittor_dtype_name(self.dtype)
    with _new_scope(self, device):
        return _new_finish(_owner.jt.zeros(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)


def _new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False, **kw):
    dt = _owner._dtype_to_str(dtype) if dtype is not None else _jittor_dtype_name(self.dtype)
    # size may be a tuple/list/torch.Size OR a jittor NanoVector (e.g. from
    # x.new_full(x.shape, v)); both are iterable with __len__.
    shp = tuple(int(s) for s in size) if hasattr(size, "__len__") else (int(size),)
    with _new_scope(self, device):
        return _new_finish(_owner.jt.full(shp, fill_value).cast(dt), device, requires_grad)


def _new_empty(self, *size, dtype=None, device=None, requires_grad=False, **kw):
    dt = _owner._dtype_to_str(dtype) if dtype is not None else _jittor_dtype_name(self.dtype)
    with _new_scope(self, device):
        return _new_finish(_owner.jt.empty(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)


def _new_tensor(self, data, dtype=None, device=None, requires_grad=False, **kw):
    dt = _owner._dtype_to_str(dtype) if dtype is not None else _jittor_dtype_name(self.dtype)
    # torch's new_tensor accepts a python list whose elements are 0-d tensors
    # (e.g. centernet_update_head builds start_coord_pre_level by accumulating
    # `_start = _start + batch * area_per_level[level]`, where the indexed term
    # is a scalar). jittor has no 0-d tensors, so those scalars are [1] Vars and
    # jt.array([int, Var, Var, ...]) raises "inhomogeneous shape". Coerce any
    # numel-1 Var element to a python number first so the list is homogeneous.
    if isinstance(data, (list, tuple)):
        def _coerce(v):
            if isinstance(v, _owner.jt.Var):
                return v.item() if v.numel() == 1 else v.tolist()
            if isinstance(v, (list, tuple)):
                return [_coerce(e) for e in v]
            return v
        data = [_coerce(v) for v in data]
    with _new_scope(self, device):
        return _new_finish(_owner.jt.array(data).cast(dt), device, requires_grad)


def _clamp(input, min=None, max=None, min_v=None, max_v=None):
    # accept BOTH torch (min/max) and jittor-native (min_v/max_v) kwarg names:
    # this override REPLACES jt.clamp, and jittor's own ops (e.g. nn.hardswish ->
    # jt.clamp(x+3, min_v=0, max_v=6)) call it with min_v/max_v.
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _native_clamp = _native['_native_clamp']
    return _native_clamp(input, min if min is not None else min_v,
                         max if max is not None else max_v)


def _torch_ne(input, other):
    a = input if isinstance(input, _NativeVar) else _owner.jt.array(input)
    b = other if isinstance(other, _NativeVar) else _owner.jt.array(other)
    if _jittor_dtype_name(a.dtype) == "bool":
        a = a.int32()
    if isinstance(b, _NativeVar) and _jittor_dtype_name(b.dtype) == "bool":
        b = b.int32()
    diff = (a - b).abs()
    out = diff > 0
    if "float" in _jittor_dtype_name(a.dtype) or (isinstance(b, _NativeVar) and "float" in _jittor_dtype_name(b.dtype)):
        try:
            out = out | _owner.jt.isnan(a) | _owner.jt.isnan(b)
        except _owner.EXPECTED as exc:
            _owner.swallowed("torch/installers/tensor.py _torch_ne: out = out | jt.isnan(a) | jt.isnan(b)", exc)
    return out


def _nonzero(self, as_tuple=False, **kw):
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _native_nonzero = _native['_native_nonzero']
    idx = _native_nonzero(self)
    if not as_tuple:
        return idx
    # idx is (N, ndim); split into one 1-D index Var per dimension. For a
    # 0/1-D input torch still returns a 1-tuple of the flat indices.
    ndim = idx.shape[1] if idx.ndim == 2 else 1
    if idx.ndim != 2:
        return (idx.reshape(-1),)
    return tuple(idx[:, d] for d in range(ndim))


def _element_size(self):
    _context = get_install_context(_owner.jt)
    return _DTYPE_BYTES.get(_jittor_dtype_name(self.dtype), 4)


class _Storage:
    def __init__(self, var):
        self._var = var

    def _owner(self):
        owner = getattr(self._var, "_torch_data_owner", None)
        return owner if isinstance(owner, _NativeVar) else self._var

    def _element_size(self):
        return _DTYPE_BYTES.get(_jittor_dtype_name(self._owner().dtype), 4)

    def data_ptr(self):
        first_element = int(self._var._storage_address)
        return first_element - int(self._var._storage_offset()) * self._element_size()
    def size(self):
        return int(self._owner().numel())
    def nbytes(self):
        return self.size() * self._element_size()


def _add(input, other, *, alpha=1, out=None):
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _native_add = _native['_native_add']
    if alpha != 1:
        other = other * alpha
    result = _native_add(input, other)
    if out is not None:
        return _owner._assign_out(out, result)
    return result


def _invert(self):
    if _jittor_dtype_name(self.dtype) == "bool":
        return self.logical_not()
    return _owner.jt.logical_not(self) if _jittor_dtype_name(self.dtype) == "bool" else (-self - 1)


def _device(self):
    if getattr(self, "_jittor_torch_meta", False):
        return _owner.device("meta")
    if self.placement_backend >= 0:
        if self.placement_backend == 0:
            return _owner.device("cpu")
        name = "npu" if self.placement_backend == 2 else "cuda"
        return _owner.device(name, int(self.device_id))
    # Report the Var's ACTUAL memory residency (matches jtorch's C++
    # is_cpu()/device()): a Var built/migrated to host -- e.g. via
    # torch.zeros(device='cpu') or .cpu() -- is "cpu" even while the
    # global use_cuda flag is 1. Only fall back to the global flag when
    # CUDA is on and the Var is genuinely device-resident.
    if (_owner.jt.flags.use_cuda or getattr(_owner.jt.compiler, "has_acl", 0)):
        if _owner._var_is_cpu_resident(self):
            return _owner.device("cpu")
        # The index is the Var's own, not a hardcoded 0: every Var carries
        # the device it lives on.
        idx = getattr(self, "device_id", 0)
        return _owner.device("cuda", int(idx) if idx is not None and idx >= 0 else 0)
    return _owner.device("cpu")


def _var_get_device(self):
    d = _device(self)
    if getattr(d, "type", "cpu") == "cpu":
        return -1
    return int(getattr(d, "index", 0) or 0)


def _is_basic_index(index):
    if isinstance(index, tuple):
        return all(_is_basic_index(item) for item in index)
    if index is None or index is Ellipsis or isinstance(index, slice):
        return True
    return isinstance(index, _owner.numbers.Integral) and not isinstance(index, (bool, _owner.np.bool_))


def _torch_getitem(self, slices):
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _orig_getitem = _native['_orig_getitem']
    out = _orig_getitem(self, slices)
    if isinstance(out, _NativeVar) and _owner._var_has_cpu_residency_hint(self):
        out = _owner._mark_cpu_like(out, self)
    # The core owns basic-index views and flattens them to their root.
    # Complete that record for Torch-only slice spellings; advanced
    # indexing stays a copy. No Python parent chain is needed.
    if isinstance(out, _NativeVar) and _is_basic_index(slices):
        if not out._is_view():
            out._set_view_of(self, slices)
        try:
            data_owner = getattr(self, "_torch_data_owner", None)
            if isinstance(data_owner, _NativeVar):
                out._torch_data_owner = data_owner
                out._torch_data_path = getattr(
                    self, "_torch_data_path", ()
                ) + (slices,)
        except _owner.EXPECTED as exc:
            _owner.swallowed("torch/installers/tensor.py _torch_getitem: out._torch_data_owner = data_owner", exc)
    return out


def _data_get(self):
    _context = get_install_context(_owner.jt)
    Var = _context.state["Var"]
    _native = _context.state["tensor_native_api"]
    _native_data_descriptor = _native['_native_data_descriptor']
    from ...tensor_state import compatibility_owner
    # Only when this interpreter is actually serving the torch
    # namespace. Composition runs either way, and a plain
    # ``import jittor`` must keep Jittor's own contract, where ``.data``
    # is a numpy view -- code like ``a.data[mask]`` with a numpy mask
    # depends on it, and 2.0 is meant to leave the native interface
    # as it was.
    preflight = getattr(_owner.jt, "_compat_preflight_result", None)
    torch_mode = (
        getattr(preflight, "active", False)
        or bool(vars(compatibility_owner(_owner.jt)).get(
            "_torch_compat_install_complete", False))
        or compatibility_owner(_owner.jt) is not _owner.jt
    )
    if (
        _native_data_descriptor is not None
        and not torch_mode
    ):
        return _native_data_descriptor.__get__(self, Var)
    view = self.detach().stop_grad()
    view._torch_data_owner = self
    view._torch_data_path = ()
    return view


def _data_set(self, value):
    src = value if isinstance(value, _NativeVar) else _owner.jt.array(value)
    was_trainable = not self.is_stop_grad()
    self.assign(src)
    if was_trainable:
        self.start_grad()


























def _to(self, *args, **kwargs):
    ds = None
    dev = None
    copy = bool(kwargs.get("copy", False))
    if kwargs.get("memory_format") not in (None, "preserve_format", "contiguous_format"):
        raise NotImplementedError("to supports preserve_format and contiguous_format")
    # device passed as a keyword (torch's .to(device=..., dtype=...))
    if "device" in kwargs:
        dev = kwargs["device"]
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, _owner.dtype):
            ds = a.name
        elif isinstance(a, _owner.device):
            dev = a
        elif isinstance(a, _NativeVar):
            # .to(other) copies other's dtype AND device.
            ds = _jittor_dtype_name(a.dtype)
            dev = a.device
        elif isinstance(a, str):
            bare = a.replace("torch.", "")
            if bare in _owner.dtype._registry:
                ds = bare
            elif bare.split(":")[0] in ("cpu", "cuda", "npu", "meta"):
                dev = bare
    if dev is None:
        dev = self.device
    out = self.clone() if copy else self
    if ds is not None:
        out = _cast_if_needed(out, ds)
    # Honor an explicit device= target by migrating residency. device=None
    # (the common .to(dtype) call) leaves placement on the global default.
    if _owner._device_is_cpu(dev):
        out = _owner._make_cpu_resident(out)
    elif _owner._device_is_cuda(dev):
        if out.placement_backend >= 0:
            out = _owner._make_cuda_resident(out, force=True, device=dev)
        else:
            src_index = getattr(self, "device_id", -1)
            out = _owner._make_cuda_resident(out, force=True)
            # .to("cuda:N") copies across devices when N is not where the Var
            # already is; a bare .to("cuda") leaves the tensor on its own
            # device, as in torch.
            out = _owner._move_to_cuda_index(out, dev, src_index)
    elif _owner._device_is_meta(dev):
        if out is self and not getattr(self, "_jittor_torch_meta", False):
            out = self.clone()
        _owner._set_meta_placeholder(out)
    if getattr(self, "_torch_0d", False):
        out._torch_0d = True
    return out


def _type_as(self, other):
    """Match ``Tensor.type_as`` by inheriting dtype and device from ``other``."""
    if not isinstance(other, _NativeVar):
        raise TypeError("type_as expects a Tensor argument")
    # Passing the tensor itself through ``_to`` applies both its dtype and
    # device.  Jittor's native ``type_as`` only changes dtype, which leaves
    # CUDA constants created by Transformers on the host.
    return _to(self, other)


def _var_detach(self):
    _context = get_install_context(_owner.jt)
    Var = _context.state["Var"]
    _native = _context.state["tensor_native_api"]
    _native_detach = _native['_native_detach']
    out = _native_detach(self)
    # Jittor's native detach marks the producing clone op as stopped while
    # leaving the returned Var's requires_grad bit set.  Torch's detached
    # tensor is a stopped leaf.  The torch-facing runtime selects
    # EXPLICIT_REQUIRES_GRAD, so apply that policy at this API boundary;
    # native Jittor callers retain the native behavior.
    policy = getattr(getattr(_owner.jt, "autograd", None), "get_policy", None)
    if Var is not _NativeVar or (
            policy is not None and policy().stop_outputs_when_inputs_stopped):
        # This Python method runs after the native binding's policy scope
        # has restored its caller. Its frontend owner determines detach's
        # contract even when the surrounding native policy is unchanged.
        out = out.stop_grad()
    if getattr(self, "_torch_0d", False):
        out._torch_0d = True
    if getattr(self, "_jittor_torch_meta", False):
        _owner._set_meta_placeholder(out)
    return out


def _var_numpy(self, *args, **kwargs):
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _native_numpy = _native['_native_numpy']
    out = _native_numpy(self, *args, **kwargs)
    if getattr(self, "_torch_0d", False) and getattr(out, "size", 0) == 1:
        return out.reshape(())
    return out


def _var_cpu(self, *a, **k):
    out = _owner._make_cpu_resident(self)
    if getattr(self, "_torch_0d", False):
        out._torch_0d = True
    if out.placement_backend >= 0:
        return out
    try:
        out._jittor_torch_force_cpu = True
    except (AttributeError, TypeError) as exc:
        _owner.swallowed("torch/installers/tensor.py _var_cpu: out._jittor_torch_force_cpu = True", exc)
    return out


def _var_cuda(self, device=None, *a, **k):
    if self.placement_backend >= 0:
        out = _owner._make_cuda_resident(self, force=True, device=device)
    else:
        _owner._set_use_cuda()
        src_index = getattr(self, "device_id", -1)
        out = _owner._make_cuda_resident(self, force=True)
        # .cuda(N) is .to("cuda:N"); .cuda() keeps the tensor where it is.
        out = _owner._move_to_cuda_index(out, device, src_index)
    if getattr(self, "_torch_0d", False):
        out._torch_0d = True
    return out


def _cast_if_needed(tensor, dtype):
    return tensor if _jittor_dtype_name(tensor.dtype) == dtype else tensor.cast(dtype)


def _var_type(self, dst_type=None, non_blocking=False, **kw):
    _context = get_install_context(_owner.jt)
    if dst_type is None:
        return _DTYPE_TO_TYPENAME.get(_jittor_dtype_name(self.dtype), "torch.FloatTensor")
    if isinstance(dst_type, str) and dst_type in _jittor_dtype_name(_TYPENAME_TO_DTYPE):
        return _cast_if_needed(self, _TYPENAME_TO_DTYPE[dst_type])
    ds = _owner._dtype_to_str(dst_type)
    return _cast_if_needed(self, ds) if ds is not None else self


def _complex_scalar_var(value):
    # Python/NumPy complex scalars are not accepted by Jittor's automatic
    # Var converter. Materialize the torch-default complex64 scalar first;
    # the actual arithmetic remains a normal device op.
    return _owner.jt.array(_owner.np.asarray([value], dtype=_owner.np.complex64))


def _truediv_target(da, db):
    _context = get_install_context(_owner.jt)
    g = _context.target_namespace
    r = _promote_pair(da, db)
    if r.startswith(("float", "bfloat", "complex")):
        return r
    return _owner._dtype_to_str(g.get_default_dtype()) or "float32"


def _scalar_dtype_name(x):
    _context = get_install_context(_owner.jt)
    g = _context.target_namespace
    if isinstance(x, bool):
        return "bool"
    if isinstance(x, int):
        return "int64"
    if isinstance(x, float):
        return _owner._dtype_to_str(g.get_default_dtype()) or "float32"
    if isinstance(x, complex):
        return "complex64"
    return None


def _is_cuda(self):
    if getattr(self, "_jittor_torch_meta", False):
        return False
    if self.placement_backend >= 0:
        return self.placement_backend != 0
    if not (_owner.jt.flags.use_cuda or getattr(_owner.jt.compiler, "has_acl", 0)):
        return False
    return not _owner._var_is_cpu_resident(self)


def _narrow(self, dim, start, length):
    nd = self.ndim
    d = dim if dim >= 0 else dim + nd
    if start < 0:
        start += self.shape[d]
    sl = [slice(None)] * nd
    sl[d] = slice(start, start + length)
    return self[tuple(sl)]


def _stride(self, dim=None):
    st = tuple(self._storage_strides())
    if dim is None:
        return tuple(st)
    return st[dim]


def _as_strided(self, size, stride, storage_offset=0):
    size = [int(s) for s in size]
    stride = [int(s) for s in stride]
    flat = self.reshape(-1)
    idx = None
    for d in range(len(size)):
        ar = _owner.jt.arange(size[d], dtype="int64") * stride[d]
        shp = [1] * len(size)
        shp[d] = size[d]
        ar = ar.reshape(shp)
        idx = ar if idx is None else idx + ar
    if storage_offset:
        idx = idx + int(storage_offset)
    return flat[idx.reshape(-1)].reshape(size)


def _torch_where(self, *args):
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _jt_var_where = _native['_jt_var_where']
    if len(args) == 2:
        condition, other = args
        return _owner._torch_where_select(condition, self, other)
    return _jt_var_where(self, *args)


def _tile(self, *dims):
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    return self.repeat(*dims)


def _squeeze(self, dim=None):
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _native_squeeze = _native['_native_squeeze']
    if dim is None:
        out = _native_squeeze(self)
        logical_0d = all(int(s) == 1 for s in self.shape)
        if logical_0d:
            out._torch_0d = True
        return out
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    nd = self.ndim
    # normalize negatives and keep only the dims whose size is 1 (torch
    # silently ignores the rest). Remove from highest index to lowest so
    # earlier removals don't shift the indices of later ones.
    norm = sorted({(d if d >= 0 else d + nd) for d in dims}, reverse=True)
    out = self
    for d in norm:
        if 0 <= d < out.ndim and out.shape[d] == 1:
            out = _native_squeeze(out, d)
    removed = {d for d in norm if 0 <= d < nd and self.shape[d] == 1}
    if len(removed) == nd:
        out._torch_0d = True
    return out


def _baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
    res = _owner.jt.matmul(batch1, batch2)
    if alpha != 1:
        res = res * alpha
    if beta == 0:
        return res
    return beta * self + res


def _addmm_method(self, mat1, mat2, *, beta=1, alpha=1):
    res = _owner.jt.matmul(mat1, mat2)
    if alpha != 1:
        res = res * alpha
    if beta == 0:
        return res
    return beta * self + res


def _T(self):
    nd = self.ndim
    if nd < 2:
        return self
    return self.permute(*range(nd - 1, -1, -1))


def _mT(self):
    return self.transpose(-1, -2)


def _var_norm(self, p="fro", dim=None, keepdims=None, *rest,
              keepdim=False, dtype=None, eps=None, **kw):
    # jittor's internal convention is norm(p, dim, keepdims, eps): when a
    # 4th positional eps (a non-bool number) or an explicit eps= is present,
    # this is an internal call -- delegate verbatim to the native op so its
    # eps-floor (used by misc.normalize/weightnorm to avoid div-by-zero) is
    # preserved exactly.
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _native_norm = _native['_native_norm']
    _norm_via = _native['_norm_via']
    fourth = rest[0] if rest else None
    is_internal = eps is not None or (
        isinstance(fourth, (int, float)) and not isinstance(fourth, bool))
    if is_internal:
        kdv = bool(keepdims) if keepdims is not None else keepdim
        ev = eps if eps is not None else (fourth if fourth is not None else 1e-30)
        d = -1 if dim is None else dim
        return _native_norm(self, p if p != "fro" else 2, d, kdv, ev)
    # torch convention: norm(p='fro', dim=None, keepdim=False, dtype=None)
    kd = bool(keepdims) if keepdims is not None else keepdim
    if fourth is not None:
        dtype = fourth
    return _norm_via(self, p=p, dim=dim, keepdim=kd, dtype=dtype)



def _binary_native(opname, left, right):
    native = get_install_context(_owner.jt).state["tensor_native_api"]["operators"][opname]
    result = native(left, right)
    return _owner._mark_cpu_like(result, left, right)

def _promoting_binary(self, other, opname, reflected):
    g = get_install_context(_owner.jt).target_namespace
    if isinstance(other, (complex, _owner.np.complexfloating)):
        other = _complex_scalar_var(other)
    if isinstance(other, _NativeVar):
        da, db = _jittor_dtype_name(self.dtype), _jittor_dtype_name(other.dtype)
        if da == db and not da.startswith("uint"):
            return _binary_native(opname, self, other)
        res = _promote_pair(da, db)
        a = self if da == res else self.cast(res)
        b = other if db == res else other.cast(res)
        out = _binary_native(opname, a, b)
        # native may still mis-infer (unsigned -> signed); fix it up.
        if isinstance(out, _NativeVar) and _jittor_dtype_name(out.dtype) != res:
            out = out.cast(res)
        return out
    # torch defers numeric ops against a Python sequence to the sequence's
    # own protocol: `Tensor.__mul__([x])` / `__rmul__([x])` return
    # NotImplemented, so `[x] * t` becomes list-repeat (via Tensor.__index__)
    # and `t * [x]` raises. jittor's native op would instead broadcast the
    # list into a Var (e.g. `[tok] * grid.prod()` -> Var, breaking ms-swift's
    # `_extend_tokens` list concatenation). Match torch: defer to the sequence.
    if isinstance(other, (list, tuple)):
        return NotImplemented
    if reflected and isinstance(other, (bool, int, float)):
        # Jittor may materialize a Python scalar on CPU for reflected ops
        # (notably ``base ** cuda_tensor``). Keep the scalar on the Var's
        # backend so the native operation cannot create a mixed-device graph.
        scalar = _owner.jt.array(other, dtype=_jittor_dtype_name(self.dtype))
        if bool(getattr(self, "is_cuda", False)):
            scalar = scalar.cuda()
        elif bool(getattr(self, "is_cpu", False)):
            # A frontend CPU tensor remains explicitly host-resident even
            # when the process-wide Jittor default is CUDA.  Without this
            # branch ``0 + cpu_tensor`` can combine a CUDA-default scalar
            # with an explicit CPU Var inside a module frontend scope.
            scalar = scalar.cpu()
        other = scalar
    out = _binary_native(opname, self, other)
    if isinstance(other, (bool, int, float)) and isinstance(out, _NativeVar):
        expected = _owner._dtype_to_str(g.result_type(self, other))
        if expected is not None and _jittor_dtype_name(out.dtype) != expected:
            out = out.cast(expected)
    return out


def _true_division(self, other, opname):
    if isinstance(other, (complex, _owner.np.complexfloating)):
        other = _complex_scalar_var(other)
    if isinstance(other, _NativeVar):
        da, db = _jittor_dtype_name(self.dtype), _jittor_dtype_name(other.dtype)
        if da == db and da.startswith(("float", "bfloat", "complex")):
            return _binary_native(opname, self, other)
        tgt = _truediv_target(da, db)
        a = self if da == tgt else self.cast(tgt)
        b = other if db == tgt else other.cast(tgt)
        out = _binary_native(opname, a, b)
        if isinstance(out, _NativeVar) and _jittor_dtype_name(out.dtype) != tgt:
            out = out.cast(tgt)
        return out
    # python sequence: defer to it (torch returns NotImplemented), matching
    # the integer-op behaviour above.
    if isinstance(other, (list, tuple)):
        return NotImplemented
    sd = _scalar_dtype_name(other)
    if sd is not None:
        tgt = _truediv_target(_jittor_dtype_name(self.dtype), sd)
        src_dt = _jittor_dtype_name(self.dtype)
        # CPU/CUDA widen Python floats for PyTorch 1-ulp parity; torch_npu
        # stays in the tensor dtype because ACL has no float64 arithmetic.
        acl_active = bool(getattr(_owner.jt.compiler, "has_acl", 0)) and (
            bool(getattr(_owner.jt.flags, "use_acl", 0)) and bool(_owner.jt.flags.use_cuda))
        use_wide = sd.startswith("float") and src_dt != "float64" and not acl_active
        calc_dt = "float64" if use_wide else tgt
        a = self if src_dt == calc_dt else self.cast(calc_dt)
        # Reflected scalar division (``1 / cuda_tensor``) otherwise hands a
        # Python scalar to Jittor's native op, which may materialize it on the
        # host even though the tensor operand is CUDA-resident.
        reflected = opname.startswith("__r")
        if reflected or use_wide:
            b = _owner.jt.array(other, dtype=calc_dt)
            if bool(getattr(self, "is_cuda", False)):
                b = b.cuda()
            elif bool(getattr(self, "is_cpu", False)):
                b = b.cpu()
        else:
            b = other
        out = _binary_native(opname, a, b)
        if isinstance(out, _NativeVar) and _jittor_dtype_name(out.dtype) != tgt:
            out = out.cast(tgt)
        return out
    return _binary_native(opname, self, other)


def _tensor_add(self, other):
    return _promoting_binary(self, other, '__add__', False)


def _tensor_radd(self, other):
    return _promoting_binary(self, other, '__radd__', True)


def _tensor_sub(self, other):
    return _promoting_binary(self, other, '__sub__', False)


def _tensor_rsub(self, other):
    return _promoting_binary(self, other, '__rsub__', True)


def _tensor_mul(self, other):
    return _promoting_binary(self, other, '__mul__', False)


def _tensor_rmul(self, other):
    return _promoting_binary(self, other, '__rmul__', True)


def _tensor_floordiv(self, other):
    return _promoting_binary(self, other, '__floordiv__', False)


def _tensor_rfloordiv(self, other):
    return _promoting_binary(self, other, '__rfloordiv__', True)


def _tensor_mod(self, other):
    return _promoting_binary(self, other, '__mod__', False)


def _tensor_rmod(self, other):
    return _promoting_binary(self, other, '__rmod__', True)


def _tensor_pow(self, other):
    return _promoting_binary(self, other, '__pow__', False)


def _tensor_rpow(self, other):
    return _promoting_binary(self, other, '__rpow__', True)


def _tensor_truediv(self, other):
    return _true_division(self, other, '__truediv__')


def _tensor_rtruediv(self, other):
    return _true_division(self, other, '__rtruediv__')


_BINARY_APIS = {
    '__add__': _tensor_add,
    '__radd__': _tensor_radd,
    '__sub__': _tensor_sub,
    '__rsub__': _tensor_rsub,
    '__mul__': _tensor_mul,
    '__rmul__': _tensor_rmul,
    '__floordiv__': _tensor_floordiv,
    '__rfloordiv__': _tensor_rfloordiv,
    '__mod__': _tensor_mod,
    '__rmod__': _tensor_rmod,
    '__pow__': _tensor_pow,
    '__rpow__': _tensor_rpow,
    '__truediv__': _tensor_truediv,
    '__rtruediv__': _tensor_rtruediv,
}


def _api_fill(self, val):
    return _ip(self, _owner.jt.ones(self.shape, self.dtype) * val)


def _api_zero(self):
    return _ip(self, _owner.jt.zeros(self.shape, self.dtype))


def _api_add(self, o, alpha=1):
    return _ip(self, self + o * alpha)


def _api_sub(self, o, alpha=1):
    return _ip(self, self - o * alpha)


def _api_mul(self, o):
    return _ip(self, self * o)


def _api_div(self, o):
    return _ip(self, self / o)


def _api_clamp_min(input, v):
    return _clamp(input, min=v)


def _api_clamp_max(input, v):
    return _clamp(input, max=v)


def _api_clamp(self, min=None, max=None, min_v=None, max_v=None):
    return _clamp(self, min, max, min_v, max_v)


def _api_clamp_alias(self, min=None, max=None, min_v=None, max_v=None):
    return _ip(self, _clamp(self, min, max, min_v, max_v))


def _api_ne(self, other):
    return _torch_ne(self, other)


def _api_ne_alias(self, other):
    return _torch_ne(self, other)


def _api_nonzero(input, as_tuple=False, **kw):
    return _nonzero(input, as_tuple=as_tuple)


def _api_normal(self, mean=0.0, std=1.0, generator=None):
    return _ip(self, _owner.jt.normal(float(mean), float(std), self.shape).cast(_jittor_dtype_name(self.dtype)))


def _api_uniform(self, a=0.0, b=1.0, generator=None):
    return _ip(self, (_owner.jt.rand(self.shape) * (b - a) + a).cast(_jittor_dtype_name(self.dtype)))


def _api_tolist(self):
    return self.item() if getattr(self, '_torch_0d', False) else self.numpy().tolist()


def _api_contiguous(self):
    if self._storage_is_contiguous():
        return self
    from ...frontend import tensor_frontend
    context = get_install_context(_owner.jt)
    with tensor_frontend(context.state["Var"], like=self):
        out = _owner.jt.ops.contiguous(self)
    if getattr(self, "_torch_0d", False):
        out._torch_0d = True
    if getattr(self, "_jittor_torch_meta", False):
        _owner._set_meta_placeholder(out)
    return out


def _api_argwhere(input):
    return _nonzero(input, as_tuple=False)


def _api_argwhere_alias(self):
    return _nonzero(self, as_tuple=False)


def _api_hash(self):
    return id(self)


def _api_nelement(self):
    return int(self.numel())


def _api_is_floating_point(self):
    return _jittor_dtype_name(self.dtype) in _FP_DTYPES


def _api_is_complex(self):
    return _jittor_dtype_name(self.dtype) in ('complex64', 'complex128')


def _api_is_signed(self):
    return _jittor_dtype_name(self.dtype) not in ('bool', 'uint8', 'uint16', 'uint32', 'uint64')


def _api_storage(self):
    return _Storage(self)


def _api_untyped_storage(self):
    return _Storage(self)


def _api_data_ptr(self):
    return int(self._storage_address)


def _api_is_contiguous(self, *a, **k):
    return self._storage_is_contiguous()


def _api_is_leaf(self):
    return bool(self.is_backward_leaf)


def _api_retains_grad(self):
    return bool(getattr(self, '_torch_retains_grad', False))


def _api_is_cpu(self):
    return not getattr(self, "_jittor_torch_meta", False) and not _is_cuda(self)


def _api_is_mps(self):
    return False


def _api_is_xpu(self):
    return False


def _api_is_meta(self):
    return getattr(self.device, 'type', None) == 'meta'


def _api_get_device(self):
    return _var_get_device(self)


def _api_storage_offset(self):
    return int(self._storage_offset())


def _api_reduce_ex(self, protocol):
    from ...frontend import reduce_tensor
    return reduce_tensor(self)


def _api_reduce(self):
    return (_owner._rebuild_var_from_numpy, (self.numpy(), _jittor_dtype_name(self.dtype)))


def _api_is_nested(self):
    return False


def _cast_byte(self):
    return _cast_if_needed(self, 'uint8')


def _cast_char(self):
    return _cast_if_needed(self, 'int8')


def _cast_short(self):
    return _cast_if_needed(self, 'int16')


def _cast_int(self):
    return _cast_if_needed(self, 'int32')


def _cast_long(self):
    return _cast_if_needed(self, 'int64')


def _cast_half(self):
    return _cast_if_needed(self, 'float16')


def _cast_float(self):
    return _cast_if_needed(self, 'float32')


def _cast_double(self):
    return _cast_if_needed(self, 'float64')


def _cast_bfloat16(self):
    return _cast_if_needed(self, 'bfloat16')


def _cast_bool(self):
    return _cast_if_needed(self, 'bool')


_CAST_APIS = {
    'byte': _cast_byte,
    'char': _cast_char,
    'short': _cast_short,
    'int': _cast_int,
    'long': _cast_long,
    'half': _cast_half,
    'float': _cast_float,
    'double': _cast_double,
    'bfloat16': _cast_bfloat16,
    'bool': _cast_bool,
}

def _api_neg_(self):
    return _ip(self, -self)


def _api_reciprocal_(self):
    return _ip(self, 1.0 / self)


def _api_rsqrt_(self):
    return _ip(self, 1.0 / _owner.jt.sqrt(self))


_UNARY_INPLACE_APIS = {
    'neg_': _api_neg_,
    'reciprocal_': _api_reciprocal_,
    'rsqrt_': _api_rsqrt_,
}

from .autograd_api import (
    _TorchGradFn,
    _backward,
    _fill_opt_grads,
    _grad_fn,
    _grad_get,
    _grad_set,
    _optimizer_maybe_has_fsdp_params,
    _register_leaf,
    _retain_grad,
    _rg_get,
    _rg_set,
    requires_grad_,
)

def _api_log_(self):
    return _ip(self, _owner.jt.log(self))

def _api_exp_(self):
    return _ip(self, _owner.jt.exp(self))

def _api_sqrt_(self):
    return _ip(self, _owner.jt.sqrt(self))

def _api_abs_(self):
    return _ip(self, _owner.jt.abs(self))

def _api_sigmoid_(self):
    return _ip(self, _owner.jt.sigmoid(self))

def _api_tanh_(self):
    return _ip(self, _owner.jt.tanh(self))

_UNARY_INPLACE_APIS.update({
    'log_': _api_log_,
    'exp_': _api_exp_,
    'sqrt_': _api_sqrt_,
    'abs_': _api_abs_,
    'sigmoid_': _api_sigmoid_,
    'tanh_': _api_tanh_,
})

"""Torch tensor methods ownership."""

def _install_tensor_methods(g, Var, _DTYPE_OBJS=None):
    # Var.dtype natively returns jittor's NanoString, which is unhashable and
    # not == to torch dtype objects. Wrap it to return our hashable `dtype`
    # (str subclass), so `t.dtype in {torch.float16, ...}` and dict keys work.
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    _NativeVar = _owner.jt.Var

    def _type_attribute(name):
        # A frontend subclass inherits C descriptors and numeric slots; reading
        # only its own __dict__ silently misses the native implementation.
        for base in Var.__mro__:
            if name in vars(base):
                return vars(base)[name]
        return None
    if _DTYPE_OBJS is not None and not getattr(Var, "_dtype_wrapped", False):
        try:
            _native_desc = _type_attribute("dtype")  # C getset_descriptor
            if _native_desc is not None:
                def _dtype_get(self, _d=_native_desc):
                    name = str(_d.__get__(self, type(self)))
                    return _DTYPE_OBJS.get(name, name)
                Var.dtype = property(_dtype_get)
                Var._dtype_wrapped = True
        except _owner.EXPECTED as exc:
            _owner.swallowed("torch/installers/tensor.py _install_tensor_methods: _native_desc = Var.__dict__.get('dtype') # C getset_des...", exc)

    if not hasattr(Var, "_vj_native_data_descriptor"):
        native_data_descriptor = _type_attribute("data")
        if native_data_descriptor is not None:
            Var._vj_native_data_descriptor = native_data_descriptor
    _native_data_descriptor = getattr(Var, "_vj_native_data_descriptor", None)

    def _numpy_data_value(value):
        if isinstance(value, _NativeVar):
            return value.numpy()
        if isinstance(value, tuple):
            return tuple(_numpy_data_value(item) for item in value)
        if isinstance(value, list):
            return [_numpy_data_value(item) for item in value]
        return value

    def _write_data_owner_numpy(view, value, slices):
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

    # torch parity for `x[bool_mask] = value` when the mask has lower rank than
    # `x` and `value` carries redundant leading size-1 batch axes. Torch assigns
    # a RHS shaped like (1, N, C) into the selected region shaped (N, C); jittor's
    # native setitem rejects the extra leading axis. This path is used by
    # TRELLIS/o_voxel texture baking (`attrs[mask] = grid_sample_3d(...)`).
    _orig_setitem = Var.__setitem__
    if not getattr(_orig_setitem, "_torch_mask_bcast", False):
        def _torch_setitem(self, slices, value):
            if _set_data_owner(self, slices, value):
                return self
            try:
                mask = slices
                if isinstance(mask, _NativeVar) and mask.dtype in ("bool", "uint8") \
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
        _torch_setitem._torch_mask_bcast = True
        Var.__setitem__ = _torch_setitem

    # in-place tensor ops torch code uses heavily (jittor exposes assign()).
    # _ip() preserves grad-tracking: jittor's assign() adopts the source's
    # stop_grad flag, which would freeze a trainable parameter.
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
        target.assign(value)
        if was_trainable and target.is_stop_grad():
            target.start_grad()
        elif not was_trainable and not target.is_stop_grad():
            target.stop_grad()
        return self
    def _copy_(self, other, non_blocking=False):
        src = other if isinstance(other, _NativeVar) else _owner.jt.array(other)
        return _ip(self, src.cast(str(self.dtype)) if hasattr(self, "dtype") else src)
    if not hasattr(Var, "copy_"):
        Var.copy_ = _copy_

    # torch's new_*(size, *, dtype=, device=, requires_grad=) factory methods.
    # jittor's native new_ones/new_zeros only take a size, so override to accept
    # torch kwargs (dtype defaults to self's dtype, like torch).
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
            _owner._set_use_cuda()
            v = _owner._make_cuda_resident(v, force=True)
        if requires_grad:
            v.requires_grad_(True)
            _owner._torch_register_leaf(v)
        return v
    def _new_ones(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _owner._dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return _new_finish(_owner.jt.ones(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)
    def _new_zeros(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _owner._dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return _new_finish(_owner.jt.zeros(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)
    def _new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False, **kw):
        dt = _owner._dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        # size may be a tuple/list/torch.Size OR a jittor NanoVector (e.g. from
        # x.new_full(x.shape, v)); both are iterable with __len__.
        shp = tuple(int(s) for s in size) if hasattr(size, "__len__") else (int(size),)
        return _new_finish(_owner.jt.full(shp, fill_value).cast(dt), device, requires_grad)
    def _new_empty(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _owner._dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return _new_finish(_owner.jt.empty(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)
    def _new_tensor(self, data, dtype=None, device=None, requires_grad=False, **kw):
        dt = _owner._dtype_to_str(dtype) if dtype is not None else str(self.dtype)
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
        return _new_finish(_owner.jt.array(data).cast(dt), device, requires_grad)
    Var.new_ones = _new_ones
    Var.new_zeros = _new_zeros
    Var.new_full = _new_full
    Var.new_empty = _new_empty
    Var.new_tensor = _new_tensor
    # Override the native methods even when they already exist. Transformers
    # initializes parameters through ``param.data.normal_()/zero_()/fill_()``
    # inside @torch.no_grad(); Jittor's native bound initializers adopt the
    # constant source's stop-grad flag and permanently freeze the parameter.
    Var.fill_ = lambda self, val: _ip(self, _owner.jt.ones(self.shape, self.dtype) * val)
    Var.zero_ = lambda self: _ip(self, _owner.jt.zeros(self.shape, self.dtype))
    Var.add_ = lambda self, o, alpha=1: _ip(self, self + (o * alpha))
    Var.sub_ = lambda self, o, alpha=1: _ip(self, self - (o * alpha))
    Var.mul_ = lambda self, o: _ip(self, self * o)
    Var.div_ = lambda self, o: _ip(self, self / o)
    # in-place unary math ops (recurrent_gemma uses x.log_(); common torch idioms)
    for _name, _fn in (("log_", _owner.jt.log), ("exp_", _owner.jt.exp), ("sqrt_", _owner.jt.sqrt),
                       ("neg_", lambda x: -x), ("abs_", _owner.jt.abs), ("sigmoid_", _owner.jt.sigmoid),
                       ("tanh_", _owner.jt.tanh), ("reciprocal_", lambda x: 1.0 / x),
                       ("rsqrt_", lambda x: 1.0 / _owner.jt.sqrt(x))):
        if not hasattr(Var, _name):
            setattr(Var, _name, (lambda fn: lambda self: _ip(self, fn(self)))(_fn))
    # torch.clamp(input, min=None, max=None) and Tensor.clamp(min=, max=)
    # accept min/max as keyword args, either of which may be None. jittor's
    # native clamp only takes them positionally and rejects the keywords (it
    # also exposes `low`/`high` names, not `min`/`max`). Wrap both the
    # top-level op and the method so torch's keyword form works, while plain
    # positional calls (jittor's own usage) pass straight through unchanged.
    _native_clamp = _owner.jt.clamp
    def _clamp(input, min=None, max=None, min_v=None, max_v=None):
        # accept BOTH torch (min/max) and jittor-native (min_v/max_v) kwarg names:
        # this override REPLACES jt.clamp, and jittor's own ops (e.g. nn.hardswish ->
        # jt.clamp(x+3, min_v=0, max_v=6)) call it with min_v/max_v.
        return _native_clamp(input, min if min is not None else min_v,
                             max if max is not None else max_v)
    g.clamp = _clamp
    g.clip = _clamp                      # torch.clip is an alias of torch.clamp
    # torch.clamp_min / clamp_max free functions (3DGS gm:159 clamps distCUDA2)
    g.clamp_min = lambda input, v: _clamp(input, min=v)
    g.clamp_max = lambda input, v: _clamp(input, max=v)
    Var.clamp = lambda self, min=None, max=None, min_v=None, max_v=None: _clamp(self, min, max, min_v, max_v)
    Var.clip = Var.clamp
    Var.clamp_ = lambda self, min=None, max=None, min_v=None, max_v=None: _ip(self, _clamp(self, min, max, min_v, max_v))
    Var.clip_ = Var.clamp_

    def _torch_ne(input, other):
        a = input if isinstance(input, _NativeVar) else _owner.jt.array(input)
        b = other if isinstance(other, _NativeVar) else _owner.jt.array(other)
        if str(a.dtype) == "bool":
            a = a.int32()
        if isinstance(b, _NativeVar) and str(b.dtype) == "bool":
            b = b.int32()
        diff = (a - b).abs()
        out = diff > 0
        if "float" in str(a.dtype) or (isinstance(b, _NativeVar) and "float" in str(b.dtype)):
            try:
                out = out | _owner.jt.isnan(a) | _owner.jt.isnan(b)
            except _owner.EXPECTED as exc:
                _owner.swallowed("torch/installers/tensor.py _torch_ne: out = out | jt.isnan(a) | jt.isnan(b)", exc)
        return out

    g.ne = _torch_ne
    g.not_equal = _torch_ne
    Var.ne = lambda self, other: _torch_ne(self, other)
    Var.__ne__ = lambda self, other: _torch_ne(self, other)

    # torch's Tensor.nonzero(as_tuple=False) returns an (N, ndim) index matrix;
    # nonzero(as_tuple=True) instead returns a tuple of ndim 1-D index Vars (one
    # per dimension) -- transformers/diffusers use the tuple form for advanced
    # indexing. jittor's nonzero only returns the matrix and rejects as_tuple.
    _native_nonzero = getattr(_owner.jt, "_vj_native_nonzero", _owner.jt.nonzero)
    def _nonzero(self, as_tuple=False, **kw):
        idx = _native_nonzero(self)
        if not as_tuple:
            return idx
        # idx is (N, ndim); split into one 1-D index Var per dimension. For a
        # 0/1-D input torch still returns a 1-tuple of the flat indices.
        ndim = idx.shape[1] if idx.ndim == 2 else 1
        if idx.ndim != 2:
            return (idx.reshape(-1),)
        return tuple(idx[:, d] for d in range(ndim))
    Var.nonzero = _nonzero
    g.nonzero = lambda input, as_tuple=False, **kw: _nonzero(input, as_tuple=as_tuple)
    # torch-compat: torch.argwhere(input) / Tensor.argwhere() -> the indices of the
    # nonzero elements as an (N, ndim) matrix (identical to nonzero(as_tuple=False)).
    if not hasattr(g, "argwhere"):
        g.argwhere = lambda input: _nonzero(input, as_tuple=False)
    if not hasattr(Var, "argwhere"):
        Var.argwhere = lambda self: _nonzero(self, as_tuple=False)
    Var.normal_ = lambda self, mean=0.0, std=1.0, generator=None: _ip(self, _owner.jt.normal(float(mean), float(std), self.shape).cast(str(self.dtype)))
    Var.uniform_ = lambda self, a=0.0, b=1.0, generator=None: _ip(self, (_owner.jt.rand(self.shape)*(b-a)+a).cast(str(self.dtype)))

    # torch tensors are hashable by identity (they define __eq__ elementwise but
    # keep an id-based __hash__). jittor's Var defines __eq__ and so becomes
    # unhashable, breaking `var in set_of_vars` / dict keys in peft. Restore an
    # identity hash. Membership tests use hash first, then `is`, so this matches
    # torch semantics without invoking elementwise __eq__.
    if Var.__hash__ is None:
        Var.__hash__ = lambda self: id(self)

    # element_size / nelement (torch byte-accounting helpers)
    _DTYPE_BYTES = {
        "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
        "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
        "uint16": 2, "uint32": 4, "uint64": 8, "bool": 1,
        "float8_e4m3fn": 1, "float8_e5m2": 1,
        "complex64": 8, "complex128": 16,
    }
    if not hasattr(Var, "element_size"):
        def _element_size(self):
            return _DTYPE_BYTES.get(str(self.dtype), 4)
        Var.element_size = _element_size
    if not hasattr(Var, "nelement"):
        Var.nelement = lambda self: int(self.numel())

    # torch dtype predicates on the tensor itself. transformers computes
    # model.dtype via `next(p.dtype for p in params if p.is_floating_point())`,
    # so save_pretrained needs these. jittor has no native complex, so
    # is_complex is always False here.
    _FP_DTYPES = {"float16", "float32", "float64", "bfloat16",
                  "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2",
                  "float8_e5m2fnuz", "float8_e8m0fnu", "float4_e2m1fn_x2"}
    if not hasattr(Var, "is_floating_point"):
        Var.is_floating_point = lambda self: str(self.dtype) in _FP_DTYPES
    if not hasattr(Var, "is_complex"):
        Var.is_complex = lambda self: str(self.dtype) in ("complex64", "complex128")
    if not hasattr(Var, "is_signed"):
        Var.is_signed = lambda self: str(self.dtype) not in (
            "bool", "uint8", "uint16", "uint32", "uint64")

    # torch storage introspection: peft/safetensors call tensor.storage()
    # .data_ptr() / .untyped_storage().nbytes() to detect shared/tied weights.
    # jittor has no exposed storage object; expose identity-based stand-ins so
    # save_pretrained's tied-weight detection works (each Var is its own storage).
    class _Storage:
        def __init__(self, var):
            self._var = var
        def data_ptr(self):
            return id(self._var)
        def size(self):
            return int(self._var.numel())
        def nbytes(self):
            return int(self._var.numel()) * _DTYPE_BYTES.get(str(self._var.dtype), 4)
    if not hasattr(Var, "storage"):
        Var.storage = lambda self: _Storage(self)
    if not hasattr(Var, "untyped_storage"):
        Var.untyped_storage = lambda self: _Storage(self)
    if not hasattr(Var, "data_ptr"):
        Var.data_ptr = lambda self: id(self)
    # torch tensors expose is_contiguous()/contiguous(); jittor Vars are always
    # contiguous in the sense safetensors cares about.
    if not hasattr(Var, "is_contiguous"):
        Var.is_contiguous = lambda self, *a, **k: True

    _native_add = g.add
    def _add(input, other, *, alpha=1, out=None):
        if alpha != 1:
            other = other * alpha
        result = _native_add(input, other)
        if out is not None:
            return _owner._assign_out(out, result)
        return result
    g.add = _add

    g.cumsum = _owner.cumsum
    Var.cumsum = _owner.cumsum
    # cumprod has the same ACL fragility; keep the presence guard.
    if _owner._NATIVE_CUMPROD is not None:
        g.cumprod = _owner.cumprod
        Var.cumprod = _owner.cumprod

    # bitwise/logical operators torch supports on tensors
    if not hasattr(Var, "__invert__"):
        def _invert(self):
            if str(self.dtype) == "bool":
                return self.logical_not()
            return _owner.jt.logical_not(self) if str(self.dtype) == "bool" else (-self - 1)
        Var.__invert__ = _invert

    def _device(self):
        # Inside a `with torch.device("meta")` block (transformers'
        # from_pretrained), report "meta" so its meta-context detection
        # fires and eager weight init is skipped. See device.__enter__.
        if _owner._DEVICE_CTX_STACK:
            return _owner._DEVICE_CTX_STACK[-1]
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
    Var.device = property(_device)

    # torch's Tensor.get_device(): the device index, -1 for a CPU tensor.
    def _var_get_device(self):
        d = _device(self)
        if getattr(d, "type", "cpu") == "cpu":
            return -1
        return int(getattr(d, "index", 0) or 0)
    Var.get_device = _var_get_device

    _orig_getitem = getattr(Var, "__getitem__", None)
    if _orig_getitem is not None and not getattr(_orig_getitem, "_torch_cpu_residency", False):
        def _is_basic_index(index):
            if isinstance(index, tuple):
                return all(_is_basic_index(item) for item in index)
            if index is None or index is Ellipsis or isinstance(index, slice):
                return True
            return isinstance(index, _owner.numbers.Integral) and not isinstance(index, (bool, _owner.np.bool_))

        def _torch_getitem(self, slices):
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
        _torch_getitem._torch_cpu_residency = True
        Var.__getitem__ = _torch_getitem

    for _op_name in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
                     "__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__"):
        _orig_op = getattr(Var, _op_name, None)
        if _orig_op is None or getattr(_orig_op, "_torch_cpu_residency", False):
            continue
        def _make_cpu_binary_wrapper(orig):
            def _wrapped(self, other):
                out = orig(self, other)
                return _owner._mark_cpu_like(out, self, other)
            _wrapped._torch_cpu_residency = True
            return _wrapped
        setattr(Var, _op_name, _make_cpu_binary_wrapper(_orig_op))

    # torch's Tensor.data returns a detached *tensor* (and is assignable:
    # `param.data = new_tensor`). jittor's native Var.data returns a numpy
    # ndarray, breaking `param.data.to(...)`. Override to torch semantics.
    if not getattr(Var, "_data_wrapped", False):
        def _data_get(self):
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
        Var.data = property(_data_get, _data_set)
        Var._data_wrapped = True

    # jittor's native Var.__reduce__ is `(Var, (self.data,))`, which assumes
    # .data is a numpy ndarray. The shim above redefines .data to return a Var,
    # so the stock reduce recurses forever (pickle re-reduces the Var arg). Make
    # Vars picklable by serializing through numpy + dtype (needed for Ray to
    # ship token tensors to reward actors, torch.multiprocessing, etc.).
    if not getattr(Var, "_reduce_wrapped", False):
        if Var is not _NativeVar:
            from ...frontend import deepcopy_tensor, reduce_tensor
            Var.__reduce__ = reduce_tensor
            Var.__reduce_ex__ = lambda self, protocol: reduce_tensor(self)
            Var.__deepcopy__ = deepcopy_tensor
        else:
            Var.__reduce__ = lambda self: (
                _owner._rebuild_var_from_numpy, (self.numpy(), str(self.dtype)))
        Var._reduce_wrapped = True

    # Leaf registry for the no-optimizer backward() path (below): torch's
    # loss.backward() accumulates grads into the .grad of every leaf that
    # requires grad, but jittor has no graph-walk to recover those leaves. So
    # track Vars whose grad was explicitly enabled through the torch-facing
    # API (requires_grad=True / requires_grad_()). Keyed by id() to dedupe;
    # jittor Vars are not weak-referenceable, so we hold strong refs (leaf
    # params are long-lived anyway) and prune entries that drop stop-grad.
    def _register_leaf(v):
        _owner._torch_register_leaf(v)

    # Override requires_grad with a Python property even though jittor exposes a
    # native getset descriptor: the native setter maps directly to start_grad/
    # stop_grad (identical semantics), but we additionally register the Var as a
    # leaf so the no-optimizer loss.backward() path (below) can find it. This is
    # behavior-preserving for the getter/setter; it only adds leaf bookkeeping.
    if not isinstance(_type_attribute("requires_grad"), property):
        _native_requires_grad = _type_attribute("requires_grad")
        def _rg_get(self):
            return bool(_native_requires_grad.__get__(self, Var))
        def _rg_set(self, v):
            # The native descriptor owns the reversible-vs-permanent distinction:
            # requires_grad_(False) preserves old edges, while stop_grad() does not.
            v = bool(v)
            fsdp_entry = getattr(self, "_jittor_fsdp2_entry", None)
            fsdp_state = getattr(self, "_jittor_fsdp2_state", None)
            if fsdp_entry is not None and fsdp_state is not None:
                fsdp_entry.requires_grad = v
                for peer in (getattr(fsdp_entry, "shard", None),
                             getattr(fsdp_entry, "full_param", None)):
                    if not isinstance(peer, _NativeVar) or peer is self:
                        continue
                    _native_requires_grad.__set__(peer, v)
                    if v:
                        _register_leaf(peer)
                if getattr(fsdp_state, "true_fsdp_flat", False):
                    flat = getattr(fsdp_state, "true_fsdp_flat_shard", None)
                    any_trainable = any(getattr(entry, "requires_grad", True)
                                        for entry in fsdp_state.true_fsdp_params)
                    if isinstance(flat, _NativeVar):
                        _native_requires_grad.__set__(flat, any_trainable)
                        if any_trainable:
                            _register_leaf(flat)
            _native_requires_grad.__set__(self, v)
            if v:
                _register_leaf(self)
        Var.requires_grad = property(_rg_get, _rg_set)

    def requires_grad_(self, v=True):
        self.requires_grad = v
        if v:
            _register_leaf(self)
        return self
    Var.requires_grad_ = requires_grad_

    # ------------------------------------------------------------------
    # torch-style autograd bridge: loss.backward() / param.grad
    # ------------------------------------------------------------------
    # jittor has no tensor-level backward(); gradients flow through
    # `optimizer.backward(loss)` then `optimizer.step()`. torch/accelerate
    # instead call `loss.backward()`, read/modify `param.grad` (grad clipping),
    # then call `optimizer.step()` with no loss. We bridge the two:
    #   * loss.backward(): route to the active optimizer's backward(loss),
    #     which fills pg["grads"]; then expose those grad Vars on each param.
    #   * param.grad: getter returns the optimizer-held grad Var (so in-place
    #     clipping mutates the very Var that step() consumes); setter stores it.
    def _fill_opt_grads(opt, grad_by_id, filled_param_ids=None):
        # Replicate the grad-storage half of jittor's Optimizer.backward() but
        # from an already-computed {id(param): grad} map (so a SINGLE jt.grad
        # pass feeds every optimizer + every leaf — no N-times-repeated backward).
        # Honors the per-optimizer __zero_grad flag (post_step zeros it, so the
        # next backward overwrites rather than accumulates) and tolerates a param
        # whose shape changed (3DGS densify replaces params) by replacing — not
        # .update()-ing — the stored grad Var.
        zero = getattr(opt, "_Optimizer__zero_grad", True)
        if filled_param_ids is None:
            filled_param_ids = set()
        for pg in opt.param_groups:
            grads_list = pg.get("grads")
            if grads_list is None:
                grads_list = pg["grads"] = [None] * len(pg["params"])
            for i, p in enumerate(pg["params"]):
                if not isinstance(p, _NativeVar) or not p.requires_grad:
                    continue
                g = grad_by_id.get(id(p))
                if g is None:
                    continue
                if id(p) in filled_param_ids:
                    while len(grads_list) <= i:
                        grads_list.append(None)
                    grads_list[i] = getattr(p, "_torch_grad", None)
                    continue
                g = g.stop_grad()
                existing = grads_list[i] if i < len(grads_list) else None
                if not isinstance(existing, _NativeVar):
                    existing = getattr(p, "_torch_grad", None)
                if isinstance(existing, _NativeVar) and list(existing.shape) == list(g.shape):
                    if not zero:
                        g = g + existing
                    existing.update(g)
                    stored = existing
                else:
                    stored = g
                while len(grads_list) <= i:
                    grads_list.append(None)
                grads_list[i] = stored
                object.__setattr__(p, "_torch_grad", stored)
                filled_param_ids.add(id(p))
        object.__setattr__(opt, "_Optimizer__zero_grad", False)
        try:
            opt._build_grad_map()
        except _owner.EXPECTED as exc:
            _owner.swallowed("torch/installers/tensor.py _fill_opt_grads: opt._build_grad_map()", exc,
                      "the optimizer keeps the grad map from before this backward, so "
                      "step() may apply stale or missing gradients")

    def _optimizer_maybe_has_fsdp_params(opt):
        for _pg in getattr(opt, "param_groups", []):
            for _p in _pg.get("params", []):
                if getattr(_p, "_jittor_fsdp2_state", None) is not None:
                    return True
        return False

    def _backward(self, gradient=None, retain_graph=None, create_graph=False, **kw):
        # torch's signature is (gradient=None, retain_graph=None,
        # create_graph=False, inputs=None) and retain_graph defaults to
        # create_graph. The default here was False, not None, so the line below
        # could never see None: `loss.backward(create_graph=True)` freed the
        # graph anyway and the second-order backward it was asked for then
        # failed. In the common loss.backward() case both are false, so the
        # graph is still freed.
        retain_graph = bool(create_graph) if retain_graph is None else bool(retain_graph)
        # torch's `gradient` is the vector of the vector-Jacobian product:
        # y.backward(v) computes d(sum(y*v))/dx. It used to be accepted and
        # dropped, so every weighted backward -- per-sample loss weights, a
        # manual chain rule from a custom head -- silently computed the
        # UNWEIGHTED gradient d(sum(y))/dx and trained on the wrong numbers.
        if gradient is not None:
            grad_var = gradient if isinstance(gradient, _NativeVar) else _owner.jt.array(gradient)
            if tuple(grad_var.shape) != tuple(self.shape):
                try:
                    grad_var = grad_var.broadcast(self.shape)
                except Exception:
                    raise RuntimeError(
                        "Tensor.backward(gradient=...) expects a gradient with "
                        "the same shape as the tensor, got %s for a tensor of "
                        "shape %s" % (tuple(grad_var.shape), tuple(self.shape)))
            self = (self * grad_var.cast(self.dtype)).sum()
        # Materialize the loss's FORWARD graph before computing gradients. A custom
        # CUDA-ext Function (3DGS rasterizer / fused-ssim) writes its outputs
        # out-of-band; if the forward is left lazy, jt.grad recomputes that
        # subgraph during the backward pass and the ext's lazy "empty/full"
        # factory op re-runs WITHOUT the kernel's writes -> garbage/NaN loss
        # (proven: a plain float(loss) before backward makes train.py finite).
        # Forcing the forward to settle once here decouples it from the grad pass.
        try:
            self.sync()
        except _owner.EXPECTED as exc:
            _owner.swallowed("torch/installers/tensor.py _backward: self.sync()", exc)
        # Collect EVERY live optimizer (torch allows several at once — 3DGS uses a
        # Gaussian Adam + an exposure Adam; routing to just _current_optimizer
        # left the other's params with .grad=None -> KeyError 'grads' in step()).
        reg = _owner.get_tensor_state(_owner.jt).active_optimizers
        opts = []
        if reg:
            alive = []
            for r in reg:
                o = r() if callable(r) else r
                if o is not None:
                    alive.append(r)
                    opts.append(o)
            reg[:] = alive
        # The union of grad targets: every optimizer's trainable params, plus
        # retain_grad'd non-leaves (3DGS's screenspace `means2D`, read by
        # densification as .grad). Without optimizers, fall back to the global
        # leaf registry so standalone Tensor.backward() still works.
        #
        # When optimizers are live, their current param_groups are authoritative:
        # torch code such as 3DGS replaces parameters during densification, and
        # stale strong refs in the registry would otherwise keep old params and
        # their Jittor graphs alive until OOM.
        fsdp_opts = [o for o in opts if _optimizer_maybe_has_fsdp_params(o)]
        # Ask the seam rather than importing fsdp2: this file is *below* fsdp2
        # in the dependency order (see jittor/compat/fsdp_hooks.py). The guard
        # above already proves the answer cannot be None when it matters --
        # `_optimizer_maybe_has_fsdp_params` looks for `_jittor_fsdp2_state`,
        # a marker only fsdp2 sets, and fsdp2 registers when it is imported.
        _fsdp2_backward = _owner._fsdp_hooks.provider() if fsdp_opts else None
        fsdp_opt_ids = {id(o) for o in fsdp_opts} if _fsdp2_backward is not None else set()
        leaf_map = {}
        opt_ids = set()
        filled_param_ids = set()
        for o in opts:
            for pg in getattr(o, "param_groups", []):
                for p in pg.get("params", []):
                    if not isinstance(p, _NativeVar) or not p.requires_grad:
                        continue
                    if _fsdp2_backward is not None and _fsdp2_backward.is_fsdp_managed_param(p):
                        opt_ids.add(id(p))
                        continue
                    leaf_map.setdefault(id(p), p)
                    opt_ids.add(id(p))
        if _fsdp2_backward is not None and fsdp_opts:
            for p in _fsdp2_backward.collect_fsdp_full_params_for_backward(fsdp_opts):
                if isinstance(p, _NativeVar) and p.requires_grad:
                    leaf_map.setdefault(id(p), p)
                    opt_ids.add(id(p))
        tensor_state = _owner.get_tensor_state(_owner.jt)
        retained = tensor_state.retained
        retained_ids = set()
        if retained:
            for v in list(retained.values()):
                if isinstance(v, _NativeVar) and v.requires_grad:
                    leaf_map.setdefault(id(v), v)
                    retained_ids.add(id(v))
        if opts:
            # Optimizer parameter groups supersede stale Parameter objects after
            # parameter replacement, but unrelated input leaves must still receive
            # gradients just as they do in Torch.
            _owner._torch_prune_leaf_registry(
                opt_ids | retained_ids,
                keep_non_parameters=True,
            )
            for v in list(tensor_state.leaf_params.values()):
                if isinstance(v, _NativeVar) and v.requires_grad:
                    leaf_map.setdefault(id(v), v)
        else:
            _owner._torch_prune_leaf_registry()
            for v in list(tensor_state.leaf_params.values()):
                if isinstance(v, _NativeVar) and v.requires_grad:
                    leaf_map.setdefault(id(v), v)
        if not leaf_map:
            return None
        leaves = list(leaf_map.values())
        # torch leaves a disconnected target at grad=None. Keep jt.grad's
        # historical zero-materialization untouched and use the compatibility
        # core entry point that preserves missing gradients explicitly.
        grads = _owner.jt.core.grad_optional(self, leaves, retain_graph)
        grad_by_id = {}
        for p, gr in zip(leaves, grads):
            if gr is None:
                if id(p) not in opt_ids and id(p) not in retained_ids:
                    tensor_state.leaf_params.pop(id(p), None)
                continue
            grad_by_id[id(p)] = gr
            if id(p) not in opt_ids:
                # non-optimizer leaf (retain_grad screenspace etc.): accumulate
                # onto .grad like torch (zeroed externally / per render).
                prev = getattr(p, "_torch_grad", None)
                object.__setattr__(p, "_torch_grad",
                                   gr if prev is None else (prev + gr))
        # fill each optimizer's pg["grads"] so its step(loss=None) consumes them
        if _fsdp2_backward is not None and fsdp_opts:
            _fsdp2_backward.fill_fsdp_optimizer_grads_from_grad_map(fsdp_opts, grad_by_id)
        for o in opts:
            if _fsdp2_backward is not None and id(o) in fsdp_opt_ids \
                    and not _fsdp2_backward.optimizer_has_non_fsdp_params(o):
                continue
            _fill_opt_grads(o, grad_by_id, filled_param_ids)
        # DDP's synchronisation point, deliberately here rather than next to
        # grad_optional above: it has to average the *accumulated* gradient.
        # `no_sync()` exists so several micro-batches accumulate locally and
        # only the closing backward pays for one collective -- averaging each
        # backward's own contribution instead would leave everything gathered
        # under no_sync() unsynchronised for good. By this line `p._torch_grad`
        # is the accumulated Var and, for optimizer parameters, is the very Var
        # in `pg["grads"]`, so one in-place assign updates `p.grad` and what
        # step() consumes together. Still before backward() returns, which is
        # what torch's autograd hooks guarantee: clipping and norm logging in
        # between must see the synchronised gradient.
        _owner._ddp_all_reduce_grads(leaves)
        # retain_grad is per-forward in torch; clear so the next iteration's fresh
        # screenspace tensor doesn't leak (jittor Vars aren't weak-referenceable).
        if retained:
            retained.clear()
        return None
    Var.backward = _backward

    def _grad_get(self):
        # _backward publishes _torch_grad on every leaf (for optimizer params it
        # points AT pg["grads"][i], so in-place grad clipping mutates the very Var
        # step() consumes). Fall back to any live optimizer's grad map if a param
        # hasn't gone through _backward yet.
        g = getattr(self, "_torch_grad", None)
        if g is not None:
            return g
        for r in _owner.get_tensor_state(_owner.jt).active_optimizers:
            o = r() if callable(r) else r
            if o is None:
                continue
            try:
                return o.find_grad(self)
            except _owner.EXPECTED as exc:
                _owner.swallowed("torch/installers/tensor.py _grad_get: return o.find_grad(self)", exc)
        return None
    def _grad_set(self, value):
        object.__setattr__(self, "_torch_grad", value)
        fsdp_entry = getattr(self, "_jittor_fsdp2_entry", None)
        fsdp_role = getattr(self, "_jittor_fsdp2_role", None)
        if fsdp_entry is not None:
            try:
                if value is None:
                    fsdp_entry.last_grad = None
                    fsdp_entry.full_public_grad = None
                    object.__setattr__(fsdp_entry.shard, "_torch_grad", None)
                    full = getattr(fsdp_entry, "full_param", None)
                    if full is not None and full is not self:
                        object.__setattr__(full, "_torch_grad", None)
                elif fsdp_role != "full":
                    fsdp_entry.last_grad = value
                    full = getattr(fsdp_entry, "full_param", None)
                    if full is not None and full is not self:
                        object.__setattr__(full, "_torch_grad", None)
            except (AttributeError, TypeError) as exc:
                _owner.swallowed("torch/installers/tensor.py _grad_set: if value is None:", exc)
        # Write through by identity so step() sees manual grad assignment and,
        # critically, p.grad=None cannot leave an old optimizer slot behind.
        for r in _owner.get_tensor_state(_owner.jt).active_optimizers:
            o = r() if callable(r) else r
            if o is None:
                continue
            changed = False
            for pg in getattr(o, "param_groups", []):
                params = list(pg.get("params", []))
                for i, p in enumerate(params):
                    same_fsdp_entry = fsdp_entry is not None and getattr(
                        p, "_jittor_fsdp2_entry", None) is fsdp_entry
                    if p is not self and not same_fsdp_entry:
                        continue
                    if fsdp_role == "full" and value is not None and p is not self:
                        continue
                    if value is None:
                        grads = pg.get("grads")
                        if grads is not None and i < len(grads):
                            grads[i] = None
                    else:
                        grads = pg.get("grads")
                        if grads is None:
                            grads = pg["grads"] = [None] * len(params)
                        while len(grads) < len(params):
                            grads.append(None)
                        grads[i] = value
                    changed = True
            if changed:
                try:
                    object.__setattr__(o, "_grad_map", {})
                    if value is None:
                        object.__setattr__(o, "_torch_backward_advanced_n_step", False)
                    if value is not None:
                        object.__setattr__(o, "_Optimizer__zero_grad", False)
                except (AttributeError, TypeError) as exc:
                    _owner.swallowed("torch/installers/tensor.py _grad_set: object.__setattr__(o, '_grad_map', {})", exc)
    Var.grad = property(_grad_get, _grad_set)

    # The core query is the source of truth for torch's backward-graph view.
    # Keep the compatibility spelling on Var so it follows the graph instead
    # of silently calling every intermediate a leaf.
    Var.is_leaf = property(lambda self: bool(self.is_backward_leaf))
    # torch's nested-tensor flag; jittor has no nested tensors -> always False.
    if not hasattr(Var, "is_nested"):
        Var.is_nested = property(lambda self: False)
    # torch exposes an opaque node object here.  The shim cannot expose a
    # torch autograd Node, but the core supplies a stable node id and a
    # diagnostic name.  Equality/hash use the node id so repeated reads on
    # tensors from the same producing op have the expected identity semantics.
    class _TorchGradFn:
        __slots__ = ("node_id", "op_id", "name")

        def __init__(self, node_id, op_id, name):
            self.node_id = int(node_id)
            self.op_id = int(op_id)
            self.name = str(name)

        def __repr__(self):
            return self.name or "<grad_fn>"

        def __eq__(self, other):
            return (isinstance(other, _TorchGradFn)
                    and self.node_id == other.node_id)

        def __hash__(self):
            return hash(self.node_id)

    def _grad_fn(self):
        node_id = int(self.grad_fn_node_id)
        if node_id == -1:
            return None
        return _TorchGradFn(node_id, self.grad_fn_op_id, self.grad_fn_name)

    Var.grad_fn = property(_grad_fn)
    # torch's retain_grad() marks a NON-leaf tensor so its .grad is populated
    # after backward (normally only leaves keep .grad). 3DGS relies on this for
    # the screenspace `means2D` tensor (`zeros_like(xyz)+0` then retain_grad()),
    # whose .grad drives densification. Register into a per-forward set the
    # _backward pass includes as a grad target; cleared each backward so the
    # next iteration's fresh tensor doesn't accumulate (jittor Vars can't be
    # weak-ref'd, so a persistent dict would leak one Var per iteration).
    def _retain_grad(self):
        try:
            _owner.get_tensor_state(_owner.jt).retained[id(self)] = self
        except _owner.EXPECTED as exc:
            _owner.swallowed("torch/installers/tensor.py _retain_grad: jt._torch_retained[id(self)] = self", exc)
        return self
    Var.retain_grad = _retain_grad

    def _to(self, *args, **kwargs):
        ds = None
        dev = None
        copy = bool(kwargs.get("copy", False))
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
                ds = str(a.dtype)
                dev = a.device
            elif isinstance(a, str):
                bare = a.replace("torch.", "")
                if bare in _owner.dtype._registry:
                    ds = bare
                elif bare.split(":")[0] in ("cpu", "cuda", "npu"):
                    dev = bare
        if ds is not None:
            out = self.cast(ds) if copy else _cast_if_needed(self, ds)
        else:
            out = self.clone() if copy else self
        # Honor an explicit device= target by migrating residency. device=None
        # (the common .to(dtype) call) leaves placement on the global default.
        if _owner._device_is_cpu(dev):
            out = _owner._make_cpu_resident(out)
        elif _owner._device_is_cuda(dev):
            src_index = getattr(self, "device_id", -1)
            out = _owner._make_cuda_resident(out, force=True)
            # .to("cuda:N") copies across devices when N is not where the Var
            # already is; a bare .to("cuda") leaves the tensor on its own
            # device, as in torch.
            moved = _owner._move_to_cuda_index(out, dev, src_index)
            if moved is not out and getattr(out, "_torch_0d", False):
                moved._torch_0d = True
            out = moved
        if getattr(self, "_torch_0d", False):
            out._torch_0d = True
        return out
    Var.to = _to

    # Jittor stores torch 0-D scalars as one-element Vars. Preserve a lightweight
    # provenance marker through the copy-like methods used before host export,
    # then expose the scalar shape only at the Python/NumPy boundary.
    _native_detach = Var.detach
    def _var_detach(self):
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
        return out
    Var.detach = _var_detach

    _native_numpy = Var.numpy
    def _var_numpy(self, *args, **kwargs):
        out = _native_numpy(self, *args, **kwargs)
        if getattr(self, "_torch_0d", False) and getattr(out, "size", 0) == 1:
            return out.reshape(())
        return out
    Var.numpy = _var_numpy
    Var.tolist = lambda self: (self.item() if getattr(self, "_torch_0d", False)
                               else self.numpy().tolist())

    # torch's Tensor.cpu()/.cuda() MIGRATE the tensor's residency (native exts
    # check tensor.is_cpu()). jittor's base Var.cpu just clones (stays on GPU)
    # and Var.cuda only flips the global flag, so override both to actually move
    # the data: .cpu() rebuilds the Var under the host allocator, .cuda() under
    # the device allocator. Var.location()/jtorch's C++ is_cpu() then agree.
    def _var_cpu(self, *a, **k):
        out = _owner._make_cpu_resident(self)
        try:
            out._jittor_torch_force_cpu = True
            if getattr(self, "_torch_0d", False):
                out._torch_0d = True
        except (AttributeError, TypeError) as exc:
            _owner.swallowed("torch/installers/tensor.py _var_cpu: out._jittor_torch_force_cpu = True", exc)
        return out
    Var.cpu = _var_cpu
    def _var_cuda(self, device=None, *a, **k):
        _owner._set_use_cuda()
        src_index = getattr(self, "device_id", -1)
        out = _owner._make_cuda_resident(self, force=True)
        # .cuda(N) is .to("cuda:N"); .cuda() keeps the tensor where it is.
        out = _owner._move_to_cuda_index(out, device, src_index)
        if getattr(self, "_torch_0d", False):
            out._torch_0d = True
        return out
    Var.cuda = _var_cuda

    # ---- integer/float dtype cast methods (torch parity) ----
    # jittor aliases Var.long = Var.int32 and Var.int = Var.int32, so BOTH
    # .long() and (from a non-int32 input) the torch dtype is wrong: torch's
    # .long() is int64, .int() is int32. It also lacks .short()/.byte()/.char().
    # Pin every cast method to torch's EXACT dtype. (.bool()/.half()/.double()/
    # .float()/.float32()/.int64()/... were already correct, but reassigning
    # them through .cast is behavior-identical and keeps the mapping in one place.)
    _CAST_METHOD_DTYPE = {
        "byte": "uint8", "char": "int8", "short": "int16", "int": "int32",
        "long": "int64", "half": "float16", "float": "float32",
        "double": "float64", "bfloat16": "bfloat16", "bool": "bool",
    }
    def _cast_if_needed(tensor, dtype):
        return tensor if str(tensor.dtype) == dtype else tensor.cast(dtype)

    for _mname, _mdt in _CAST_METHOD_DTYPE.items():
        setattr(Var, _mname, (lambda dt: lambda self: _cast_if_needed(self, dt))(_mdt))

    # torch's Tensor.type(): with a dtype/typed-tensor-name it casts; with no
    # argument it returns the torch type-NAME string ('torch.FloatTensor' ...).
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
    def _var_type(self, dst_type=None, non_blocking=False, **kw):
        if dst_type is None:
            return _DTYPE_TO_TYPENAME.get(str(self.dtype), "torch.FloatTensor")
        if isinstance(dst_type, str) and dst_type in _TYPENAME_TO_DTYPE:
            return _cast_if_needed(self, _TYPENAME_TO_DTYPE[dst_type])
        ds = _owner._dtype_to_str(dst_type)
        return _cast_if_needed(self, ds) if ds is not None else self
    Var.type = _var_type

    # ---- torch-parity binary-op type promotion ----
    # jittor's native arithmetic operators keep the LEFT/narrower operand's dtype
    # for mixed-dtype Var op Var (int32+int64 -> int32, float32+float64 -> float32,
    # float16+int64 -> float32, uint8+int8 -> int8), silently losing range/precision
    # vs torch. torch instead promotes BOTH operands to result_type, then computes.
    # Wrap the affected operators to do exactly that: when the other operand is a Var
    # of a DIFFERENT dtype, cast both to the promoted dtype and call the original
    # native op (now same-dtype -> jittor returns the promoted dtype). All other
    # paths -- matching dtypes, or a Python scalar (jittor already matches torch:
    # int scalar keeps the int dtype, float scalar lifts int->float32) -- pass
    # straight through to the native op, so nothing else changes.
    # True division ('/') has its OWN rule (always float) and is wrapped separately
    # just below; the operators wrapped here follow the plain promotion lattice.
    # jittor's native binary ops ALSO corrupt unsigned dtypes even when both
    # operands match (uint8+uint8 -> int8, uint16+uint16 -> int16) -- a C++
    # binary_dtype_infer quirk we cannot touch. So the wrapper post-corrects the
    # native result to the torch-expected dtype whenever they differ, which both
    # restores unsigned results and double-guards the mixed-dtype promotion.
    def _complex_scalar_var(value):
        # Python/NumPy complex scalars are not accepted by Jittor's automatic
        # Var converter. Materialize the torch-default complex64 scalar first;
        # the actual arithmetic remains a normal device op.
        return _owner.jt.array(_owner.np.asarray([value], dtype=_owner.np.complex64))

    def _make_promoting_op(opname, reflected):
        native = _type_attribute(opname)
        if native is None:
            return None
        def _op(self, other):
            if isinstance(other, (complex, _owner.np.complexfloating)):
                other = _complex_scalar_var(other)
            if isinstance(other, _NativeVar):
                da, db = str(self.dtype), str(other.dtype)
                if da == db and not da.startswith("uint"):
                    return native(self, other)
                res = g._torch_promote_pair(da, db)
                a = self if da == res else self.cast(res)
                b = other if db == res else other.cast(res)
                out = native(a, b)
                # native may still mis-infer (unsigned -> signed); fix it up.
                if isinstance(out, _NativeVar) and str(out.dtype) != res:
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
            out = native(self, other)
            if isinstance(other, (bool, int, float)) and isinstance(out, _NativeVar):
                expected = _owner._dtype_to_str(g.result_type(self, other))
                if expected is not None and str(out.dtype) != expected:
                    out = out.cast(expected)
            return out
        _op.__name__ = opname
        return _op
    # (opname, reflected?) -- reflected ops receive the *other* operand as the left
    # value, but promotion is symmetric so the same body is correct.
    for _opn, _refl in [("__add__", False), ("__radd__", True),
                        ("__sub__", False), ("__rsub__", True),
                        ("__mul__", False), ("__rmul__", True),
                        ("__floordiv__", False), ("__rfloordiv__", True),
                        ("__mod__", False), ("__rmod__", True),
                        ("__pow__", False), ("__rpow__", True)]:
        _wrapped = _make_promoting_op(_opn, _refl)
        if _wrapped is not None:
            setattr(Var, _opn, _wrapped)

    # True division ('/') is the documented special case: torch ALWAYS yields a
    # float. The result dtype is result_type(a, b) when that is already floating
    # (so float16/int64 -> float16, float32/float64 -> float64), otherwise the
    # default float dtype (so every integral pair, incl. int64/int32 and int8/int8,
    # -> float32). jittor instead follows numpy's "int -> float of matching width"
    # (int64/int32 -> float64, int8/int8 -> float16, float16/int64 -> float64),
    # which loses torch parity. Cast operands to the torch target float, then div.
    def _truediv_target(da, db):
        r = g._torch_promote_pair(da, db)
        if r.startswith(("float", "bfloat", "complex")):
            return r
        return _owner._dtype_to_str(g.get_default_dtype()) or "float32"
    def _scalar_dtype_name(x):
        if isinstance(x, bool):
            return "bool"
        if isinstance(x, int):
            return "int64"
        if isinstance(x, float):
            return _owner._dtype_to_str(g.get_default_dtype()) or "float32"
        if isinstance(x, complex):
            return "complex64"
        return None
    def _make_truediv(opname):
        native = _type_attribute(opname)
        if native is None:
            return None
        def _op(self, other):
            if isinstance(other, (complex, _owner.np.complexfloating)):
                other = _complex_scalar_var(other)
            if isinstance(other, _NativeVar):
                da, db = str(self.dtype), str(other.dtype)
                if da == db and da.startswith(("float", "bfloat", "complex")):
                    return native(self, other)
                tgt = _truediv_target(da, db)
                a = self if da == tgt else self.cast(tgt)
                b = other if db == tgt else other.cast(tgt)
                out = native(a, b)
                if isinstance(out, _NativeVar) and str(out.dtype) != tgt:
                    out = out.cast(tgt)
                return out
            # python sequence: defer to it (torch returns NotImplemented), matching
            # the integer-op behaviour above.
            if isinstance(other, (list, tuple)):
                return NotImplemented
            sd = _scalar_dtype_name(other)
            if sd is not None:
                tgt = _truediv_target(str(self.dtype), sd)
                src_dt = str(self.dtype)
                # CPU/CUDA widen Python floats for PyTorch 1-ulp parity; torch_npu
                # stays in the tensor dtype because ACL has no float64 arithmetic.
                acl_active = bool(getattr(_owner.jt.compiler, "has_acl", 0)) and (
                    bool(getattr(_owner.jt.flags, "use_acl", 0)) and bool(_owner.jt.flags.use_cuda))
                use_wide = sd.startswith("float") and src_dt != "float64" and not acl_active
                calc_dt = "float64" if use_wide else tgt
                a = self if src_dt == calc_dt else self.cast(calc_dt)
                b = _owner.jt.array(other, dtype=calc_dt) if use_wide else other
                out = native(a, b)
                if isinstance(out, _NativeVar) and str(out.dtype) != tgt:
                    out = out.cast(tgt)
                return out
            return native(self, other)
        _op.__name__ = opname
        return _op
    for _opn in ("__truediv__", "__rtruediv__"):
        _w = _make_truediv(_opn)
        if _w is not None:
            setattr(Var, _opn, _w)

    # Jittor Vars do not expose PyTorch-style strided non-contiguous storage;
    # materialized op outputs are already laid out for their logical shape. The
    # The old jittor.misc.tensor_ops.contiguous hook returned clone(), which
    # adds avoidable graph nodes and copies in PyTorch code that calls
    # transpose(...).contiguous() before export or parameter construction.
    Var.contiguous = lambda self: self
    # torch's Tensor.is_cuda / .is_cpu report the tensor's ACTUAL residency.
    # A Var built/migrated to host (torch.zeros(device='cpu'), .cpu()) is on the
    # CPU even under global use_cuda=1, so read Var.location() rather than the
    # global flag (matches jtorch's C++ is_cuda()/is_cpu()). When CUDA is off
    # everything is host-resident.
    def _is_cuda(self):
        if not (_owner.jt.flags.use_cuda or getattr(_owner.jt.compiler, "has_acl", 0)):
            return False
        return not _owner._var_is_cpu_resident(self)
    Var.is_cuda = property(_is_cuda)
    Var.is_cpu = property(lambda self: not _is_cuda(self))
    Var.is_mps = property(lambda self: False)
    Var.is_xpu = property(lambda self: False)
    Var.is_meta = property(lambda self: getattr(self.device, "type", None) == "meta")
    # torch's Tensor.get_device(): CUDA device index, or -1 for CPU tensors.
    # 3DGS's fallback ssim (utils/loss_utils.py) does window.cuda(img.get_device()).
    if not hasattr(Var, "get_device"):
        Var.get_device = lambda self: (0 if _is_cuda(self) else -1)

    # torch's Tensor.narrow(dim, start, length): a view of `length` elements
    # starting at `start` along `dim` (jittor has no narrow; use a slice).
    if not hasattr(Var, "narrow"):
        def _narrow(self, dim, start, length):
            nd = self.ndim
            d = dim if dim >= 0 else dim + nd
            if start < 0:
                start += self.shape[d]
            sl = [slice(None)] * nd
            sl[d] = slice(start, start + length)
            return self[tuple(sl)]
        Var.narrow = _narrow

    # torch's Tensor.stride()/.as_strided(): jittor Vars are always materialized
    # contiguous (row-major) -- `.contiguous` above is a no-op -- so a Var's strides
    # are exactly the row-major strides of its shape (this matches torch's strides
    # right after a `.view()`/`.reshape()`, which is where this is used, e.g.
    # longformer's `_chunk` sliding-window attention).
    if not hasattr(Var, "stride"):
        def _stride(self, dim=None):
            shape = self.shape
            st = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                st[i] = st[i + 1] * shape[i + 1]
            if dim is None:
                return tuple(st)
            return st[dim if dim >= 0 else dim + len(shape)]
        Var.stride = _stride
    if not hasattr(Var, "storage_offset"):
        Var.storage_offset = lambda self: 0
    # as_strided over a contiguous buffer == gather at linear offsets
    #   out[i0,i1,...] = flat[storage_offset + sum_d i_d * stride[d]]
    # Built with broadcast arange grids; routed through jittor advanced-indexing so
    # the backward is the correct scatter-add (overlapping windows read shared inputs).
    if not hasattr(Var, "as_strided"):
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
        Var.as_strided = _as_strided

    # torch's Tensor.where(condition, other): elements of *self* where condition is
    # True, else from `other`. jittor's native Var.where treats *self* as the condition
    # (ternary(self, a, b)) -- the opposite role -- so `t.where(cond, other)` silently
    # returned `cond` cast to t's dtype (breaks e.g. longformer's _mask_invalid_locations
    # edge masking). Add the torch 2-arg method semantics while preserving jittor's
    # native 0/1-arg form (nonzero indices), used by contrib.py. No jittor-core caller
    # uses the 2-arg method form, so this only fixes, never regresses.
    if not getattr(Var.where, "_torch_where_compat", False):
        _jt_var_where = Var.where
        def _torch_where(self, *args):
            if len(args) == 2:
                condition, other = args
                return _owner._torch_where_select(condition, self, other)
            return _jt_var_where(self, *args)
        _torch_where._torch_where_compat = True
        Var.where = _torch_where

    # torch's Tensor.tile(*dims): like numpy.tile -- when fewer dims than the
    # tensor rank are given, dims are left-padded with 1. jittor's repeat
    # already implements exactly this padding, so route tile through it.
    if not hasattr(Var, "tile"):
        def _tile(self, *dims):
            if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
                dims = tuple(dims[0])
            return self.repeat(*dims)
        Var.tile = _tile

    # torch's Tensor.squeeze(dim=None): differs from jittor's in two ways --
    #   * squeeze(dim) where that dim's size != 1 is a NO-OP in torch, but
    #     jittor asserts (AssertionError). Models call x.squeeze(d) defensively.
    #   * torch 2.0+ accepts a tuple/list of dims (squeeze((0,2))); jittor's
    #     native squeeze only takes a single int (raises TypeError on a tuple).
    # Wrap to match torch while delegating the actual op to jittor's squeeze.
    _native_squeeze = Var.squeeze
    def _squeeze(self, dim=None):
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
    Var.squeeze = _squeeze

    # torch's Tensor.baddbmm(batch1, batch2, *, beta=1, alpha=1):
    #   out = beta * self + alpha * (batch1 @ batch2)   (batched matmul)
    # jittor exposes a module-level baddbmm but no Var method (bloom calls
    # the method form). Mirror torch's keyword-only beta/alpha here.
    if not hasattr(Var, "baddbmm"):
        def _baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
            res = _owner.jt.matmul(batch1, batch2)
            if alpha != 1:
                res = res * alpha
            if beta == 0:
                return res
            return beta * self + res
        Var.baddbmm = _baddbmm
    # torch's Tensor.addmm(mat1, mat2, *, beta=1, alpha=1):
    #   out = beta * self + alpha * (mat1 @ mat2)   (2-D matmul)
    if not hasattr(Var, "addmm"):
        def _addmm_method(self, mat1, mat2, *, beta=1, alpha=1):
            res = _owner.jt.matmul(mat1, mat2)
            if alpha != 1:
                res = res * alpha
            if beta == 0:
                return res
            return beta * self + res
        Var.addmm = _addmm_method

    # torch's Tensor.T: reverse ALL dims (a deprecated-but-ubiquitous alias for
    # x.permute(reversed(range(ndim)))); a no-op for ndim < 2. jittor lacks it.
    if not isinstance(getattr(Var, "T", None), property):
        def _T(self):
            nd = self.ndim
            if nd < 2:
                return self
            return self.permute(*range(nd - 1, -1, -1))
        Var.T = property(_T)
    # torch's Tensor.mT: swap the last two dims (batched matrix transpose);
    # requires ndim >= 2. Used by modern attention code (q.mT @ k etc.).
    if not isinstance(getattr(Var, "mT", None), property):
        def _mT(self):
            return self.transpose(-1, -2)
        Var.mT = property(_mT)

    # torch's Tensor.norm(p='fro', dim=None, keepdim=False, dtype=None):
    # default (dim=None) reduces over ALL dims to a 0-dim scalar -- but jittor's
    # native Var.norm defaults to dim=-1 (per-row). Override to torch semantics
    # while STAYING compatible with jittor's internal positional convention
    #   jt.norm(x, p=2, dim=-1, keepdims=False, eps=1e-30, keepdim=False)
    # which callers like misc.normalize use as input.norm(p, dim, True, eps).
    # The collision is the 4th positional: torch=dtype, jittor=eps. Disambiguate
    # by type (a number -> jittor eps; a dtype/str/None -> torch dtype). When dim
    # is given explicitly (the only way internal callers reach here) behavior is
    # identical to before; only the dim=None default changes to a full reduce.
    _norm_via = _owner._torch_norm_impl
    _native_norm = Var.norm  # jittor's native Var.norm (eps-floored, dim=-1)
    def _var_norm(self, p="fro", dim=None, keepdims=None, *rest,
                  keepdim=False, dtype=None, eps=None, **kw):
        # jittor's internal convention is norm(p, dim, keepdims, eps): when a
        # 4th positional eps (a non-bool number) or an explicit eps= is present,
        # this is an internal call -- delegate verbatim to the native op so its
        # eps-floor (used by misc.normalize/weightnorm to avoid div-by-zero) is
        # preserved exactly.
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
    Var.norm = _var_norm

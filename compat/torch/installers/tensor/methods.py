"""Torch tensor methods ownership."""
from ...fidelity import Fidelity, register_api_bindings
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from .method_api import (
    _CAST_APIS, _UNARY_INPLACE_APIS,
    _api_fill,
    _api_zero,
    _api_add,
    _api_sub,
    _api_mul,
    _api_div,
    _api_clamp_min,
    _api_clamp_max,
    _api_clamp,
    _api_clamp_alias,
    _api_ne,
    _api_ne_alias,
    _api_nonzero,
    _api_normal,
    _api_uniform,
    _api_tolist,
    _api_contiguous,
    _api_argwhere,
    _api_argwhere_alias,
    _api_hash,
    _api_nelement,
    _api_is_floating_point,
    _api_is_complex,
    _api_is_signed,
    _api_storage,
    _api_untyped_storage,
    _api_data_ptr,
    _api_is_contiguous,
    _api_is_leaf,
    _api_retains_grad,
    _api_is_cpu,
    _api_is_mps,
    _api_is_xpu,
    _api_is_meta,
    _api_get_device,
    _api_storage_offset,
    _api_reduce_ex,
    _api_reduce,
    _api_is_nested,

    _BINARY_APIS,
    _DTYPE_BYTES,
    _DTYPE_TO_TYPENAME,
    _FP_DTYPES,
    _Storage,
    _T,
    _TYPENAME_TO_DTYPE,
    _TorchGradFn,
    _add,
    _addmm_method,
    _as_strided,
    _assign_data_owner,
    _backward,
    _baddbmm,
    _cast_if_needed,
    _clamp,
    _complex_scalar_var,
    _copy_,
    _data_get,
    _data_owner_uses_device,
    _data_set,
    _device,
    _dtype_get,
    _element_size,
    _fill_opt_grads,
    _grad_fn,
    _grad_get,
    _grad_set,
    _invert,
    _ip,
    _is_basic_data_index,
    _is_basic_index,
    _is_cuda,
    _mT,
    _narrow,
    _new_empty,
    _new_finish,
    _new_full,
    _new_ones,
    _new_tensor,
    _new_zeros,
    _nonzero,
    _norm_size,
    _numpy_data_value,
    _optimizer_maybe_has_fsdp_params,
    _register_leaf,
    _resolve_size,
    _restore_trainable_state,
    _retain_grad,
    _rg_get,
    _rg_set,
    _scalar_dtype_name,
    _set_data_owner,
    _squeeze,
    _stride,
    _tile,
    _to,
    _type_as,
    _torch_getitem,
    _torch_ne,
    _torch_setitem,
    _torch_where,
    _truediv_target,
    _var_cpu,
    _var_cuda,
    _var_detach,
    _var_get_device,
    _var_norm,
    _var_numpy,
    _var_type,
    _write_data_owner_numpy,
    requires_grad_,
)
from ...context import get_install_context
from types import MappingProxyType

def _type_attribute(Var, name):
    # A frontend subclass inherits C descriptors and numeric slots; reading
    # only its own __dict__ silently misses the native implementation.
    for base in Var.__mro__:
        if name in vars(base):
            return vars(base)[name]
    return None


def _install_tensor_methods(g, Var, _DTYPE_OBJS=None):
    # Var.dtype natively returns jittor's NanoString, which is unhashable and
    # not == to torch dtype objects. Return the canonical immutable frontend
    # dtype so `t.dtype in {torch.float16, ...}` and dictionary keys work.
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    _NativeVar = _owner.jt.Var
    _native_operators = {name: getattr(Var, name, None) for name in _BINARY_APIS}

    if _jittor_dtype_name(_DTYPE_OBJS) is not None and not getattr(Var, "_dtype_wrapped", False):
        try:
            _native_desc = _type_attribute(Var, "dtype")  # C getset_descriptor
            if _native_desc is not None:
                Var.dtype = property(_dtype_get)
                Var._dtype_wrapped = True
        except _owner.EXPECTED as exc:
            _owner.swallowed("torch/installers/tensor.py _install_tensor_methods: _native_desc = Var.__dict__.get('dtype') # C getset_des...", exc)

    if not hasattr(Var, "_vj_native_data_descriptor"):
        native_data_descriptor = _type_attribute(Var, "data")
        if native_data_descriptor is not None:
            Var._vj_native_data_descriptor = native_data_descriptor
    _native_data_descriptor = getattr(Var, "_vj_native_data_descriptor", None)

    # torch parity for `x[bool_mask] = value` when the mask has lower rank than
    # `x` and `value` carries redundant leading size-1 batch axes. Torch assigns
    # a RHS shaped like (1, N, C) into the selected region shaped (N, C); jittor's
    # native setitem rejects the extra leading axis. This path is used by
    # TRELLIS/o_voxel texture baking (`attrs[mask] = grid_sample_3d(...)`).
    _orig_setitem = Var.__setitem__
    if not getattr(_orig_setitem, "_torch_mask_bcast", False):
        _torch_setitem._torch_mask_bcast = True
        Var.__setitem__ = _torch_setitem

    # in-place tensor ops torch code uses heavily (jittor exposes assign()).
    # _ip() preserves grad-tracking: jittor's assign() adopts the source's
    # stop_grad flag, which would freeze a trainable parameter.
    if not hasattr(Var, "copy_"):
        Var.copy_ = _copy_

    # torch's new_*(size, *, dtype=, device=, requires_grad=) factory methods.
    # jittor's native new_ones/new_zeros only take a size, so override to accept
    # torch kwargs (dtype defaults to self's dtype, like torch).
    Var.new_ones = _new_ones
    Var.new_zeros = _new_zeros
    Var.new_full = _new_full
    Var.new_empty = _new_empty
    Var.new_tensor = _new_tensor
    # Override the native methods even when they already exist. Transformers
    # initializes parameters through ``param.data.normal_()/zero_()/fill_()``
    # inside @torch.no_grad(); Jittor's native bound initializers adopt the
    # constant source's stop-grad flag and permanently freeze the parameter.
    Var.fill_ = _api_fill
    Var.zero_ = _api_zero
    Var.add_ = _api_add
    Var.sub_ = _api_sub
    Var.mul_ = _api_mul
    Var.div_ = _api_div
    # in-place unary math ops (recurrent_gemma uses x.log_(); common torch idioms)
    for name, implementation in _UNARY_INPLACE_APIS.items():
        if not hasattr(Var, name):
            setattr(Var, name, implementation)
    # torch.clamp(input, min=None, max=None) and Tensor.clamp(min=, max=)
    # accept min/max as keyword args, either of which may be None. jittor's
    # native clamp only takes them positionally and rejects the keywords (it
    # also exposes `low`/`high` names, not `min`/`max`). Wrap both the
    # top-level op and the method so torch's keyword form works, while plain
    # positional calls (jittor's own usage) pass straight through unchanged.
    _native_clamp = _owner.jt.clamp
    g.clamp = _clamp
    g.clip = _clamp                      # torch.clip is an alias of torch.clamp
    # torch.clamp_min / clamp_max free functions (3DGS gm:159 clamps distCUDA2)
    g.clamp_min = _api_clamp_min
    g.clamp_max = _api_clamp_max
    Var.clamp = _api_clamp
    Var.clip = Var.clamp
    Var.clamp_ = _api_clamp_alias
    Var.clip_ = Var.clamp_


    g.ne = _torch_ne
    g.not_equal = _torch_ne
    Var.ne = _api_ne
    Var.__ne__ = _api_ne_alias

    # torch's Tensor.nonzero(as_tuple=False) returns an (N, ndim) index matrix;
    # nonzero(as_tuple=True) instead returns a tuple of ndim 1-D index Vars (one
    # per dimension) -- transformers/diffusers use the tuple form for advanced
    # indexing. jittor's nonzero only returns the matrix and rejects as_tuple.
    _native_nonzero = get_install_context(g).state["core_native_api"]["nonzero"]
    Var.nonzero = _nonzero
    g.nonzero = _api_nonzero
    # torch-compat: torch.argwhere(input) / Tensor.argwhere() -> the indices of the
    # nonzero elements as an (N, ndim) matrix (identical to nonzero(as_tuple=False)).
    if not hasattr(g, "argwhere"):
        g.argwhere = _api_argwhere
    if not hasattr(Var, "argwhere"):
        Var.argwhere = _api_argwhere_alias
    Var.normal_ = _api_normal
    Var.uniform_ = _api_uniform

    # torch tensors are hashable by identity (they define __eq__ elementwise but
    # keep an id-based __hash__). jittor's Var defines __eq__ and so becomes
    # unhashable, breaking `var in set_of_vars` / dict keys in peft. Restore an
    # identity hash. Membership tests use hash first, then `is`, so this matches
    # torch semantics without invoking elementwise __eq__.
    if Var.__hash__ is None:
        Var.__hash__ = _api_hash

    # element_size / nelement (torch byte-accounting helpers)
    if not hasattr(Var, "element_size"):
        Var.element_size = _element_size
    if not hasattr(Var, "nelement"):
        Var.nelement = _api_nelement

    # torch dtype predicates on the tensor itself. transformers computes
    # model.dtype via `next(p.dtype for p in params if p.is_floating_point())`,
    # so save_pretrained needs these. jittor has no native complex, so
    # is_complex is always False here.
    if not hasattr(Var, "is_floating_point"):
        Var.is_floating_point = _api_is_floating_point
    if not hasattr(Var, "is_complex"):
        Var.is_complex = _api_is_complex
    if not hasattr(Var, "is_signed"):
        Var.is_signed = _api_is_signed

    # torch storage introspection: peft/safetensors call tensor.storage()
    # .data_ptr() / .untyped_storage().nbytes() to detect shared/tied weights.
    # jittor has no exposed storage object; expose identity-based stand-ins so
    # save_pretrained's tied-weight detection works (each Var is its own storage).
    if not hasattr(Var, "storage"):
        Var.storage = _api_storage
    if not hasattr(Var, "untyped_storage"):
        Var.untyped_storage = _api_untyped_storage
    if not hasattr(Var, "data_ptr"):
        # Tensor.data_ptr is the first element address, not Python object
        # identity. The native accessor synchronizes without migrating it.
        Var.data_ptr = _api_data_ptr
    # Query the physical layout, including stride-zero expanded storage.
    if not hasattr(Var, "is_contiguous"):
        Var.is_contiguous = _api_is_contiguous

    _native_add = g.add
    g.add = _add

    g.cumsum = _owner.cumsum
    Var.cumsum = _owner.cumsum
    # cumprod has the same ACL fragility; keep the presence guard.
    if _owner._NATIVE_CUMPROD is not None:
        g.cumprod = _owner.cumprod
        Var.cumprod = _owner.cumprod

    # bitwise/logical operators torch supports on tensors
    if not hasattr(Var, "__invert__"):
        Var.__invert__ = _invert

    Var.device = property(_device)

    # torch's Tensor.get_device(): the device index, -1 for a CPU tensor.
    Var.get_device = _var_get_device

    _orig_getitem = getattr(Var, "__getitem__", None)
    if _orig_getitem is not None and not getattr(_orig_getitem, "_torch_cpu_residency", False):

        _torch_getitem._torch_cpu_residency = True
        Var.__getitem__ = _torch_getitem


    # torch's Tensor.data returns a detached *tensor* (and is assignable:
    # `param.data = new_tensor`). jittor's native Var.data returns a numpy
    # ndarray, breaking `param.data.to(...)`. Override to torch semantics.
    if not getattr(Var, "_data_wrapped", False):
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
            Var.__reduce_ex__ = _api_reduce_ex
            Var.__deepcopy__ = deepcopy_tensor
        else:
            Var.__reduce__ = _api_reduce
        Var._reduce_wrapped = True

    # Leaf registry for the no-optimizer backward() path (below): torch's
    # loss.backward() accumulates grads into the .grad of every leaf that
    # requires grad, but jittor has no graph-walk to recover those leaves. So
    # track Vars whose grad was explicitly enabled through the torch-facing
    # API (requires_grad=True / requires_grad_()). The identity index weakly
    # references independent Tensor holders; native leaf identity decides
    # whether a registered holder participates. Legacy Vars keep their pruning.

    # Override requires_grad with a Python property even though jittor exposes a
    # native getset descriptor: the native setter maps directly to start_grad/
    # stop_grad (identical semantics), but we additionally register the Var as a
    # leaf so the no-optimizer loss.backward() path (below) can find it. This is
    # behavior-preserving for the getter/setter; it only adds leaf bookkeeping.
    if not isinstance(_type_attribute(Var, "requires_grad"), property):
        _native_requires_grad = _type_attribute(Var, "requires_grad")
        Var.requires_grad = property(_rg_get, _rg_set)

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


    Var.backward = _backward

    Var.grad = property(_grad_get, _grad_set)

    # The core query is the source of truth for torch's backward-graph view.
    # Keep the compatibility spelling on Var so it follows the graph instead
    # of silently calling every intermediate a leaf.
    Var.is_leaf = property(_api_is_leaf)
    # torch's nested-tensor flag; jittor has no nested tensors -> always False.
    if not hasattr(Var, "is_nested"):
        Var.is_nested = property(_api_is_nested)
    # torch exposes an opaque node object here.  The shim cannot expose a
    # torch autograd Node, but the core supplies a stable node id and a
    # diagnostic name.  Equality/hash use the node id so repeated reads on
    # tensors from the same producing op have the expected identity semantics.


    Var.grad_fn = property(_grad_fn)
    # torch's retain_grad() marks a NON-leaf tensor so its .grad is populated
    # after backward (normally only leaves keep .grad). Registration follows
    # the holder's lifetime, including repeated or interleaved backward graphs.
    Var.retain_grad = _retain_grad
    Var.retains_grad = property(_api_retains_grad)

    Var.to = _to
    Var.type_as = _type_as

    # Jittor stores torch 0-D scalars as one-element Vars. Preserve a lightweight
    # provenance marker through the copy-like methods used before host export,
    # then expose the scalar shape only at the Python/NumPy boundary.
    _native_detach = Var.detach
    Var.detach = _var_detach

    _native_numpy = Var.numpy
    Var.numpy = _var_numpy
    Var.tolist = _api_tolist

    # torch's Tensor.cpu()/.cuda() MIGRATE the tensor's residency (native exts
    # check tensor.is_cpu()). jittor's base Var.cpu just clones (stays on GPU)
    # and Var.cuda only flips the global flag, so override both to actually move
    # the data: .cpu() rebuilds the Var under the host allocator, .cuda() under
    # the device allocator. Var.location()/jtorch's C++ is_cpu() then agree.
    Var.cpu = _var_cpu
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

    for name, implementation in _CAST_APIS.items():
        setattr(Var, name, implementation)

    # torch's Tensor.type(): with a dtype/typed-tensor-name it casts; with no
    # argument it returns the torch type-NAME string ('torch.FloatTensor' ...).
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

    # (opname, reflected?) -- reflected ops receive the *other* operand as the left
    # value, but promotion is symmetric so the same body is correct.

    # True division ('/') is the documented special case: torch ALWAYS yields a
    # float. The result dtype is result_type(a, b) when that is already floating
    # (so float16/int64 -> float16, float32/float64 -> float64), otherwise the
    # default float dtype (so every integral pair, incl. int64/int32 and int8/int8,
    # -> float32). jittor instead follows numpy's "int -> float of matching width"
    # (int64/int32 -> float64, int8/int8 -> float16, float16/int64 -> float64),
    # which loses torch parity. Cast operands to the torch target float, then div.

    # The refactored core exposes true strided storage. Preserve identity for
    # dense tensors and materialize non-contiguous views before Torch callers
    # reshape them.
    Var.contiguous = _api_contiguous
    # torch's Tensor.is_cuda / .is_cpu report the tensor's ACTUAL residency.
    # A Var built/migrated to host (torch.zeros(device='cpu'), .cpu()) is on the
    # CPU even under global use_cuda=1, so read Var.location() rather than the
    # global flag (matches jtorch's C++ is_cuda()/is_cpu()). When CUDA is off
    # everything is host-resident.
    Var.is_cuda = property(_is_cuda)
    Var.is_cpu = property(_api_is_cpu)
    Var.is_mps = property(_api_is_mps)
    Var.is_xpu = property(_api_is_xpu)
    Var.is_meta = property(_api_is_meta)
    # torch's Tensor.get_device(): CUDA device index, or -1 for CPU tensors.
    # 3DGS's fallback ssim (utils/loss_utils.py) does window.cuda(img.get_device()).
    if not hasattr(Var, "get_device"):
        Var.get_device = _api_get_device

    # torch's Tensor.narrow(dim, start, length): a view of `length` elements
    # starting at `start` along `dim` (jittor has no narrow; use a slice).
    if not hasattr(Var, "narrow"):
        Var.narrow = _narrow

    # Tensor strides come from the native storage descriptor, not shape math.
    if not hasattr(Var, "stride"):
        Var.stride = _stride
    if not hasattr(Var, "storage_offset"):
        Var.storage_offset = _api_storage_offset
    # as_strided over a contiguous buffer == gather at linear offsets
    #   out[i0,i1,...] = flat[storage_offset + sum_d i_d * stride[d]]
    # Built with broadcast arange grids; routed through jittor advanced-indexing so
    # the backward is the correct scatter-add (overlapping windows read shared inputs).
    if not hasattr(Var, "as_strided"):
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
        _torch_where._torch_where_compat = True
        Var.where = _torch_where

    # torch's Tensor.tile(*dims): like numpy.tile -- when fewer dims than the
    # tensor rank are given, dims are left-padded with 1. jittor's repeat
    # already implements exactly this padding, so route tile through it.
    if not hasattr(Var, "tile"):
        Var.tile = _tile

    # torch's Tensor.squeeze(dim=None): differs from jittor's in two ways --
    #   * squeeze(dim) where that dim's size != 1 is a NO-OP in torch, but
    #     jittor asserts (AssertionError). Models call x.squeeze(d) defensively.
    #   * torch 2.0+ accepts a tuple/list of dims (squeeze((0,2))); jittor's
    #     native squeeze only takes a single int (raises TypeError on a tuple).
    # Wrap to match torch while delegating the actual op to jittor's squeeze.
    _native_squeeze = Var.squeeze
    Var.squeeze = _squeeze

    # torch's Tensor.baddbmm(batch1, batch2, *, beta=1, alpha=1):
    #   out = beta * self + alpha * (batch1 @ batch2)   (batched matmul)
    # jittor exposes a module-level baddbmm but no Var method (bloom calls
    # the method form). Mirror torch's keyword-only beta/alpha here.
    if not hasattr(Var, "baddbmm"):
        Var.baddbmm = _baddbmm
    # torch's Tensor.addmm(mat1, mat2, *, beta=1, alpha=1):
    #   out = beta * self + alpha * (mat1 @ mat2)   (2-D matmul)
    if not hasattr(Var, "addmm"):
        Var.addmm = _addmm_method

    # torch's Tensor.T: reverse ALL dims (a deprecated-but-ubiquitous alias for
    # x.permute(reversed(range(ndim)))); a no-op for ndim < 2. jittor lacks it.
    if not isinstance(getattr(Var, "T", None), property):
        Var.T = property(_T)
    # torch's Tensor.mT: swap the last two dims (batched matrix transpose);
    # requires ndim >= 2. Used by modern attention code (q.mT @ k etc.).
    if not isinstance(getattr(Var, "mT", None), property):
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
    Var.norm = _var_norm

    _context = get_install_context(g)
    _context.state["tensor_native_api"] = MappingProxyType({
        'operators': MappingProxyType(_native_operators),
        '_DTYPE_OBJS': locals().get('_DTYPE_OBJS'),
        '_jt_var_where': locals().get('_jt_var_where'),
        '_native_add': locals().get('_native_add'),
        '_native_clamp': locals().get('_native_clamp'),
        '_native_data_descriptor': locals().get('_native_data_descriptor'),
        '_native_desc': locals().get('_native_desc'),
        '_native_detach': locals().get('_native_detach'),
        '_native_nonzero': locals().get('_native_nonzero'),
        '_native_norm': locals().get('_native_norm'),
        '_native_numpy': locals().get('_native_numpy'),
        '_native_requires_grad': locals().get('_native_requires_grad'),
        '_native_squeeze': locals().get('_native_squeeze'),
        '_norm_via': locals().get('_norm_via'),
        '_orig_getitem': locals().get('_orig_getitem'),
        '_orig_setitem': locals().get('_orig_setitem'),
    })

    for name, implementation in _BINARY_APIS.items():
        if _native_operators[name] is not None:
            setattr(Var, name, implementation)

    register_api_bindings(Var, 'torch.Tensor',
        ('T', '__deepcopy__', '__getitem__', '__hash__', '__invert__', '__ne__', '__reduce__', '__reduce_ex__', '__setitem__', 'add_', 'addmm', 'argwhere', 'as_strided', 'backward', 'baddbmm', 'clamp', 'clamp_', 'clip', 'clip_', 'contiguous', 'copy_', 'cpu', 'cuda', 'cumprod', 'cumsum', 'data', 'data_ptr', 'detach', 'device', 'div_', 'dtype', 'element_size', 'fill_', 'get_device', 'grad', 'grad_fn', 'is_complex', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_floating_point', 'is_leaf', 'is_meta', 'is_mps', 'is_nested', 'is_signed', 'is_xpu', 'mT', 'mul_', 'narrow', 'ne', 'nelement', 'new_empty', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'nonzero', 'norm', 'normal_', 'numpy', 'requires_grad', 'requires_grad_', 'retain_grad', 'retains_grad', 'squeeze', 'storage', 'storage_offset', 'stride', 'sub_', 'tile', 'to', 'tolist', 'type', 'type_as', 'uniform_', 'untyped_storage', 'where', 'zero_') + tuple(_BINARY_APIS.keys() | _CAST_APIS.keys() | _UNARY_INPLACE_APIS.keys()),
        Fidelity.APPROXIMATE, 'Tensor operations share the native Var/Op graph and explicit frontend state; unsupported layouts, device capabilities, and retained compatibility approximations remain restricted')

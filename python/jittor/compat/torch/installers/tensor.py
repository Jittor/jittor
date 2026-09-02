"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt
from jittor import nn
import numbers
import numpy as np

from ..functional import (
    _diff, _torch_norm_impl,
    _torch_where_select, _trapz,
)
from ..grad import (
    _GradDecoratorCtx,
)
from ..nested import (
    _NestedTensor, _rebuild_var_from_numpy,
    _TorchSize, _torch_prune_leaf_registry,
    _torch_register_leaf,
)
from .factories import _install_random_and_linspace, _wrap_constructors
from ..types import (
    _DEVICE_CTX_STACK, _device_is_cpu, _device_is_cuda, _dtype_to_str,
    _make_cpu_resident, _make_cuda_resident, _mark_cpu_like,
    _var_has_cpu_residency_hint, _var_is_cpu_resident, device, dtype,
    _cuda_index_of, _move_to_cuda_index,
)

import collections as _collections
from ...diagnostics import EXPECTED, swallowed
from ... import fsdp_hooks as _fsdp_hooks
from ... import collectives as _collectives


def _ddp_all_reduce_grads(leaves):
    """Average DDP-managed gradients across ranks, in a rank-stable order.

    ``_jittor_ddp_state`` is set only by ``DistributedDataParallel`` (see
    installers/nn.py), which sits *above* this file -- the marker carries the
    state so nothing here has to import it, the same inversion FSDP2 uses.

    Operates on the accumulated ``_torch_grad`` and assigns in place, because
    that Var is also the one in the optimizer's ``pg["grads"]``: one write
    updates ``p.grad`` and what ``step()`` consumes.

    The ordering matters and is not incidental. Jittor's collectives are graph
    ops, and every rank must issue them in the same sequence or they pair up
    wrongly and the run deadlocks or mixes gradients between parameters. The
    backward's own leaf collection is keyed by ``id()`` and differs between
    processes; DDP stamps ``_jittor_ddp_order`` in ``module.parameters()``
    order, identical on every rank, and that is what this sorts by. The
    dependency chain then stops the scheduler reordering them again -- the same
    guard jittor's own ``optim/base.py`` puts around its all-reduce.
    """
    if _collectives._world_size() <= 1:
        return
    pending = []
    for leaf in leaves:
        state = getattr(leaf, "_jittor_ddp_state", None)
        if state is None or not getattr(state, "sync_enabled", False):
            continue
        order = getattr(leaf, "_jittor_ddp_order", None)
        grad = getattr(leaf, "_torch_grad", None)
        if order is None or not isinstance(grad, jt.Var):
            continue
        pending.append((order, grad))
    if not pending:
        return
    pending.sort(key=lambda item: item[0])
    dep = []
    for _order, grad in pending:
        grad.assign(_collectives._all_reduce_mean(grad))
        try:
            producer = grad._input(0)
        except EXPECTED as exc:
            swallowed("torch/installers/tensor.py _ddp_all_reduce_grads: "
                      "grad._input(0) for the collective ordering chain", exc,
                      "the all-reduces may be scheduled in a different order "
                      "on different ranks")
        else:
            producer._add_dependency(dep)
            dep = [producer]


_MinMax = _collections.namedtuple("torch_return_types", ["values", "indices"])
_TopK = _collections.namedtuple("topk", ["values", "indices"])
_Sort = _collections.namedtuple("sort", ["values", "indices"])
_Median = _collections.namedtuple("median", ["values", "indices"])


def _install_reductions(g):
    """torch-correct argmax/argmin/max/min/sort/topk (jittor's differ:
    jittor argmax->(idx,val), jittor max(dim)->values only).
    NB: g IS the jittor module, so capture the ORIGINAL jittor ops before
    overwriting (else infinite recursion)."""
    import jittor as _jt
    _argmax = _jt.argmax
    _argmin = _jt.argmin
    _argsort = _jt.argsort
    _maximum = _jt.maximum
    _minimum = _jt.minimum
    _jt_max = _jt.max          # jittor-native reductions (values only)
    _jt_min = _jt.min
    _jt_var_max = _jt.Var.max  # native METHODS (0-dim scalar for full reduction)
    _jt_var_min = _jt.Var.min
    _topk = getattr(_jt, "topk", None)
    _gather = _jt.gather
    _median = _jt.median

    def _reduce_index(result):
        if isinstance(result, (tuple, list)):
            result = result[0]
        return result.int64()

    def argmax(x, dim=None, keepdim=False, keepdims=None):
        if keepdims is not None:
            keepdim = keepdims
        if dim is None:
            return _reduce_index(_argmax(x.reshape(-1), 0))
        try:
            res = _argmax(x, dim, keepdims=keepdim)
        except TypeError:
            res = _argmax(x, dim, keepdim=keepdim)
        return _reduce_index(res)
    def argmin(x, dim=None, keepdim=False, keepdims=None):
        if keepdims is not None:
            keepdim = keepdims
        if dim is None:
            return _reduce_index(_argmin(x.reshape(-1), 0))
        try:
            res = _argmin(x, dim, keepdims=keepdim)
        except TypeError:
            res = _argmin(x, dim, keepdim=keepdim)
        return _reduce_index(res)
    g.argmax = argmax
    g.argmin = argmin

    def _maxmin(which, x, *args, **kwargs):
        # jittor-internal callers use the `keepdims` kwarg (with an 's') and
        # expect values-only semantics; delegate straight to the native op so
        # we don't break jittor's own softmax/layernorm/etc.
        if "keepdims" in kwargs:
            native = _jt_max if which == "max" else _jt_min
            return native(x, *args, **kwargs)
        dim = kwargs.get("dim", None)
        keepdim = kwargs.get("keepdim", False)
        other = kwargs.get("other", None)
        pos = list(args)
        if pos:
            if isinstance(pos[0], _jt.Var):
                other = pos[0]
            else:
                dim = pos[0]
                if len(pos) > 1:
                    keepdim = pos[1]
        if other is not None:
            return _maximum(x, other) if which == "max" else _minimum(x, other)
        if dim is None:
            # native scalar reduction via the captured METHOD (0-dim scalar);
            # NOT x.max(), which now routes back into this wrapper (recursion).
            return _jt_var_max(x) if which == "max" else _jt_var_min(x)
        af = argmax if which == "max" else argmin
        idx = af(x, dim=dim, keepdim=keepdim)
        if getattr(_jt.compiler, "has_acl", 0):
            native = _jt_max if which == "max" else _jt_min
            val = native(x, dim, keepdims=keepdim)
        elif keepdim:
            val = _jt.gather(x, dim, idx)
        elif x.ndim == 1:
            val = x[idx]
        else:
            val = _jt.gather(x, dim, idx.unsqueeze(dim)).squeeze(dim)
        return _MinMax(val, idx.int64())
    g.max = lambda x, *a, **k: _maxmin("max", x, *a, **k)
    g.min = lambda x, *a, **k: _maxmin("min", x, *a, **k)

    def topk(x, k, dim=-1, largest=True, sorted=True):
        # jittor's native topk is unreliable on the ACL backend (internal
        # getitem "too many slices"); use an argsort-based gather instead.
        idx, _ = _argsort(x, dim=dim, descending=largest)
        nd = x.ndim
        d = dim if dim >= 0 else dim + nd
        sl = [slice(None)] * nd
        sl[d] = slice(0, k)
        idx = idx[tuple(sl)]
        val = _gather(x, d, idx)
        return _TopK(val, idx.int64())
    g.topk = topk

    def sort(x, dim=-1, descending=False, **kw):
        idx, val = _argsort(x, dim=dim, descending=descending)
        return _Sort(val, idx.int64())
    g.sort = sort
    g.argsort = lambda x, dim=-1, descending=False, **kw: _argsort(x, dim=dim, descending=descending)[0].int64()

    def median(x, dim=None, keepdim=False):
        if dim is None:
            return _median(x, keepdim=keepdim)
        d = dim if dim >= 0 else dim + x.ndim
        if d < 0 or d >= x.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[-{x.ndim}, {x.ndim - 1}], but got {dim})"
            )
        idx, values = _argsort(x, dim=d)
        k = (x.shape[d] - 1) // 2
        slices = [slice(None)] * x.ndim
        slices[d] = slice(k, k + 1) if keepdim else k
        slices = tuple(slices)
        return _Median(values[slices], idx[slices].int64())

    g.median = median

    # --- Tensor METHOD forms. jittor-core uses none of these as Var methods (only
    # the python list.sort builtin), so installing torch semantics here is safe;
    # it was verified that .max/.min methods ARE used internally, so those stay
    # native (values-only) and are intentionally NOT overridden. ---
    Var = _jt.Var
    Var.sort = lambda self, dim=-1, descending=False, **kw: sort(self, dim=dim, descending=descending)
    Var.argsort = lambda self, dim=-1, descending=False, **kw: g.argsort(self, dim=dim, descending=descending)
    Var.topk = lambda self, k, dim=-1, largest=True, sorted=True: topk(self, k, dim=dim, largest=largest, sorted=sorted)
    Var.median = lambda self, dim=None, keepdim=False: median(
        self, dim=dim, keepdim=keepdim
    )
    # Tensor.softmax/log_softmax accept a `dtype=` (cast before the op) which
    # jittor's native method rejects (vLLM's sampler: logits.softmax(dim=-1,
    # dtype=torch.float32)).
    def _var_softmax(self, dim=-1, dtype=None, **kw):
        x = self.cast(_dtype_to_str(dtype)) if dtype is not None else self
        return _jt.nn.softmax(x, dim=dim)
    Var.softmax = _var_softmax
    def _var_log_softmax(self, dim=-1, dtype=None, **kw):
        x = self.cast(_dtype_to_str(dtype)) if dtype is not None else self
        return _jt.nn.log_softmax(x, dim=dim)
    Var.log_softmax = _var_log_softmax
    # torch's Tensor.max(dim)/min(dim) returns the (values, indices) namedtuple --
    # mmdetection relies on this pervasively (`v, i = overlaps.max(dim=0)`). jittor's
    # native method returns values-only and is used by core/linalg/einops with the
    # `keepdims=` spelling (handled natively inside _maxmin) or a bare dim. Route
    # everything through _maxmin: keepdims= -> native values; a bare/torch dim ->
    # namedtuple; no dim -> native scalar. The few jittor-internal callers that pass
    # a BARE dim and want values-only extract `.values` at their call site.
    Var.max = lambda self, *a, **k: _maxmin("max", self, *a, **k)
    Var.min = lambda self, *a, **k: _maxmin("min", self, *a, **k)

    # torch's var/std default to UNBIASED (Bessel, correction=1); jittor's native var
    # defaults to biased (numpy-aligned) -- a silent-wrong divergence for torch code.
    # Fix in the torch layer only (native jt.var stays numpy-aligned). Support both
    # the legacy `unbiased=` and modern `correction=` kwargs.
    _jt_var = Var.var
    def _correction_to_unbiased(unbiased, correction):
        if correction is not None:
            return correction != 0
        if unbiased is not None:
            return bool(unbiased)
        return True                       # torch default
    def _multidim_var(self, dims, unbiased, keepdim):
        # torch-compat: var over a LIST/TUPLE of axes. jittor's native var() `dim=`
        # slot is scalar-only (a list crashes with `is_type<int64>(oi)`), and its
        # separate `dims=` path returns a WRONG-shaped/value result for partial
        # multi-axis reductions. Compute directly from mean/sum (which DO accept a
        # tuple) so every axis subset matches torch exactly, preserving unbiased
        # (Bessel) + keepdim semantics.
        dims = [int(d) % self.ndim for d in dims]
        mean = _jt.mean(self, dims, keepdims=True)
        sqr = (self - mean) ** 2
        out = _jt.sum(sqr, dims=dims, keepdims=keepdim)
        n = 1
        for d in dims:
            n *= self.shape[d]
        if unbiased:
            n = n - 1
        return out / n
    def _torch_var(self, dim=None, unbiased=None, keepdim=False, keepdims=None,
                   correction=None, **kw):
        ub = _correction_to_unbiased(unbiased, correction)
        kd = bool(keepdim) or bool(keepdims)
        if isinstance(dim, (list, tuple)):
            return _multidim_var(self, dim, ub, kd)
        return _jt_var(self, dim=dim, unbiased=ub, keepdims=kd)
    def _torch_std(self, dim=None, unbiased=None, keepdim=False, keepdims=None,
                   correction=None, **kw):
        # std == sqrt(var) with the correct bias. jittor's native std is hardcoded
        # unbiased AND floors at maximum(1e-6) (torch doesn't), so derive from var.
        return _torch_var(self, dim=dim, unbiased=unbiased, keepdim=keepdim,
                          keepdims=keepdims, correction=correction).sqrt()
    Var.var = _torch_var
    Var.std = _torch_std
    g.var = lambda x, *a, **k: _torch_var(x, *a, **k)
    g.std = lambda x, *a, **k: _torch_std(x, *a, **k)

    # missing methods (truly absent on Var -> pure additive)
    Var.masked_select = lambda self, mask: self[mask]      # torch: 1-D of selected

    def _masked_scatter(self, mask, source):
        # torch.Tensor.masked_scatter(mask, source): copy elements of `source`
        # (consumed in row-major order) into the positions of `self` where `mask`
        # is True; `mask` broadcasts to self.shape. Out-of-place, and DIFFERENTIABLE
        # w.r.t. both self and source -- the Qwen-VL path scatters vision-tower
        # image_embeds into the text inputs_embeds, and grads must reach the ViT.
        # Implemented as gather(source, running-count-of-True) then where(mask),
        # avoiding any sliced in-place write (a jittor no-view no-op).
        m = mask
        if tuple(m.shape) != tuple(self.shape):
            m = m.broadcast(self.shape)
        mb = m.bool()
        flat_mask = mb.reshape(-1)
        # index into source.flatten() for each position = (#True strictly before it)
        sel_idx = flat_mask.int32().cumsum(0) - 1
        sel_idx = sel_idx.maximum(0).minimum(source.numel() - 1)  # clamp (unused where mask False)
        src_flat = source.reshape(-1)
        gathered = src_flat[sel_idx].reshape(self.shape)
        if str(gathered.dtype) != str(self.dtype):
            gathered = gathered.cast(str(self.dtype))
        return jt.ternary(mb, gathered, self)
    Var.masked_scatter = _masked_scatter

    def _masked_scatter_(self, mask, source):
        # in-place variant: write the result back through assign() so the same Var
        # (and any module attribute holding it) reflects the update.
        out = _masked_scatter(self, mask, source)
        self.assign(out)
        return self
    Var.masked_scatter_ = _masked_scatter_

    def _unfold(self, dimension, size, step):
        # torch's Tensor.unfold(dim, size, step): sliding windows along `dim`,
        # appending a new last dim of length `size`. out[...,i,...,j]=x[...,i*step+j,...]
        nd = self.ndim
        d = dimension if dimension >= 0 else dimension + nd
        n = (self.shape[d] - size) // step + 1
        out_shape = list(self.shape); out_shape[d] = n; out_shape.append(size)
        src = [f"i{k}" for k in range(nd)]
        src[d] = f"i{d}*{step}+i{nd}"                       # window pos + within-window
        return self.reindex(out_shape, src)
    Var.unfold = _unfold

    def _diagonal(self, offset=0, dim1=0, dim2=1):
        # torch's Tensor.diagonal: drop dim1,dim2 and append a diagonal dim.
        nd = self.ndim
        d1 = dim1 if dim1 >= 0 else dim1 + nd
        d2 = dim2 if dim2 >= 0 else dim2 + nd
        s1, s2 = self.shape[d1], self.shape[d2]
        dl = max(0, min(s1, s2 - offset)) if offset >= 0 else max(0, min(s1 + offset, s2))
        keep = [k for k in range(nd) if k != d1 and k != d2]
        out_shape = [self.shape[k] for k in keep] + [dl]
        last = len(keep)
        src = [None] * nd
        for outpos, k in enumerate(keep):
            src[k] = f"i{outpos}"
        src[d1] = f"i{last}+{max(0, -offset)}"
        src[d2] = f"i{last}+{max(0, offset)}"
        return self.reindex(out_shape, src)
    Var.diagonal = _diagonal

    # --- elementwise / reduction ops missing as torch methods (all additive) ---
    if not hasattr(Var, "sign"):
        # torch sign: -1/0/+1 (nan->nan in torch; this gives 0 for nan, an accepted edge)
        Var.sign = lambda self: (self > 0).cast(self.dtype) - (self < 0).cast(self.dtype)
    if not hasattr(Var, "trunc"):
        Var.trunc = lambda self: _jt.ternary(self >= 0, _jt.floor(self), _jt.ceil(self))
    if not hasattr(Var, "frac"):
        Var.frac = lambda self: self - _jt.ternary(self >= 0, _jt.floor(self), _jt.ceil(self))
    if not hasattr(Var, "nan_to_num"):
        def _nan_to_num(self, nan=0.0, posinf=None, neginf=None):
            # Replace nan with one ternary, then clamp to the ±inf replacement bounds.
            # NB: a jittor JIT codegen bug SEGFAULTS on chained isinf+ternary over a
            # tensor holding inf/nan (tracked, #11), so we deliberately avoid that and
            # use a clamp. This is EXACT for the default (float32-max) bounds -- finite
            # values are untouched and ±inf map to ±max. For *narrow custom* posinf/
            # neginf it also clamps finite values past them (a rare, documented
            # deviation accepted to avoid the core segfault).
            pi = 3.4028234663852886e38 if posinf is None else posinf   # exact float32 max
            ni = -3.4028234663852886e38 if neginf is None else neginf
            out = _jt.ternary(_jt.isnan(self), _jt.full_like(self, nan), self)
            return out.minimum(pi).maximum(ni)
        Var.nan_to_num = _nan_to_num
        g.nan_to_num = lambda x, nan=0.0, posinf=None, neginf=None: _nan_to_num(x, nan, posinf, neginf)
    if not hasattr(Var, "amax"):
        def _amax(self, dim=None, keepdim=False):
            d = list(dim) if isinstance(dim, (tuple, list)) else dim
            return _jt_max(self, d, keepdims=keepdim) if d is not None else self.max()
        def _amin(self, dim=None, keepdim=False):
            d = list(dim) if isinstance(dim, (tuple, list)) else dim
            return _jt_min(self, d, keepdims=keepdim) if d is not None else self.min()
        Var.amax = _amax
        Var.amin = _amin
        g.amax = lambda x, dim=None, keepdim=False: _amax(x, dim, keepdim)
        g.amin = lambda x, dim=None, keepdim=False: _amin(x, dim, keepdim)
    if not hasattr(Var, "count_nonzero"):
        def _count_nonzero(self, dim=None):
            nz = (self != 0).int32()
            return nz.sum(dim) if dim is not None else nz.sum()
        Var.count_nonzero = _count_nonzero
        g.count_nonzero = lambda x, dim=None: _count_nonzero(x, dim)
    if not hasattr(g, "logaddexp"):
        def _logaddexp(a, b):
            m = _jt.maximum(a, b)                       # numerically stable
            return m + _jt.log(_jt.exp(a - m) + _jt.exp(b - m))
        g.logaddexp = _logaddexp
        Var.logaddexp = _logaddexp

    # argmax/argmin METHOD forms: torch returns just the indices; jittor's native
    # Var.argmax returns (idx, val). Core uses these only in docstrings, so override.
    Var.argmax = lambda self, dim=None, keepdim=False: argmax(self, dim, keepdim)
    Var.argmin = lambda self, dim=None, keepdim=False: argmin(self, dim, keepdim)
    # addcmul/addcdiv: self + value * (t1 (*|/) t2)
    Var.addcmul = lambda self, t1, t2, value=1: self + value * (t1 * t2)
    Var.addcdiv = lambda self, t1, t2, value=1: self + value * (t1 / t2)
    if not hasattr(Var, "broadcast_to"):
        Var.broadcast_to = lambda self, shape: self.broadcast(shape)
    # torch-compat: module-level torch.broadcast_to(input, shape) (some code calls
    # the functional form, not the method). Expands `input` to `shape` without copy.
    if not hasattr(g, "broadcast_to"):
        g.broadcast_to = lambda input, shape: input.broadcast(shape)


def _install_tensor_methods(g, Var, _DTYPE_OBJS=None):
    # Var.dtype natively returns jittor's NanoString, which is unhashable and
    # not == to torch dtype objects. Wrap it to return our hashable `dtype`
    # (str subclass), so `t.dtype in {torch.float16, ...}` and dict keys work.
    if _DTYPE_OBJS is not None and not getattr(Var, "_dtype_wrapped", False):
        try:
            _native_desc = Var.__dict__.get("dtype")  # C getset_descriptor
            if _native_desc is not None:
                def _dtype_get(self, _d=_native_desc):
                    name = str(_d.__get__(self, type(self)))
                    return _DTYPE_OBJS.get(name, name)
                Var.dtype = property(_dtype_get)
                Var._dtype_wrapped = True
        except EXPECTED as exc:
            swallowed("torch/installers/tensor.py _install_tensor_methods: _native_desc = Var.__dict__.get('dtype') # C getset_des...", exc)

    if not hasattr(Var, "_vj_native_data_descriptor"):
        native_data_descriptor = Var.__dict__.get("data")
        if native_data_descriptor is not None:
            Var._vj_native_data_descriptor = native_data_descriptor
    _native_data_descriptor = getattr(Var, "_vj_native_data_descriptor", None)

    def _numpy_data_value(value):
        if isinstance(value, Var):
            return value.numpy()
        if isinstance(value, tuple):
            return tuple(_numpy_data_value(item) for item in value)
        if isinstance(value, list):
            return [_numpy_data_value(item) for item in value]
        return value

    def _write_data_owner_numpy(view, value, slices):
        owner = getattr(view, "_torch_data_owner", None)
        if not isinstance(owner, Var) or _native_data_descriptor is None:
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
        return isinstance(index, numbers.Integral) and not isinstance(
            index, (bool, np.bool_)
        )

    def _data_owner_uses_device(owner):
        try:
            location = owner.location()
        except EXPECTED as exc:
            swallowed("torch/installers/tensor.py _data_owner_uses_device: location = owner.location()", exc)
            location = None
        if location == "device":
            return True
        if location == "cpu":
            return False
        return bool(
            jt.flags.use_cuda or getattr(getattr(jt, "compiler", None), "has_acl", 0)
        )

    def _restore_trainable_state(value, was_trainable):
        if was_trainable and value.is_stop_grad():
            value.start_grad()
        elif not was_trainable and not value.is_stop_grad():
            value.stop_grad()

    def _assign_data_owner(view, value, extra_path=()):
        """Write a detached ``.data`` alias back without leaving the device."""

        owner = getattr(view, "_torch_data_owner", None)
        if not isinstance(owner, Var):
            return False

        view_path = getattr(view, "_torch_data_path", ())
        path = view_path + tuple(extra_path)
        bases = []
        target = owner
        for index in path:
            bases.append((target, index))
            target = _orig_getitem(target, index)

        updated = value if isinstance(value, Var) else jt.array(value)
        if updated.numel() == 1 and target.numel() != 1:
            updated = updated.broadcast(target.shape)
        for base, index in reversed(bases):
            updated = base.setitem(index, updated)

        owner_was_trainable = not owner.is_stop_grad()
        owner.assign(updated)
        _restore_trainable_state(owner, owner_was_trainable)

        # The owner may itself be a basic-index view of something else, which
        # is how a sharded weight loader writes: ``param[:rows].data.copy_(w)``.
        # Torch's slice shares the parameter's storage, so carry the write the
        # rest of the way up; without this the parameter keeps its old values
        # and the weight is silently lost.
        _write_index_parent(owner, updated)

        # A retained ``data = parameter.data`` alias observes its own mutation,
        # just like a Torch tensor sharing the parameter storage.
        view_updated = updated
        for index in view_path:
            view_updated = _orig_getitem(view_updated, index)
        view_was_trainable = not view.is_stop_grad()
        view.assign(view_updated)
        _restore_trainable_state(view, view_was_trainable)
        _write_index_parent(view, view_updated)
        return True

    def _set_data_owner(view, slices, value):
        owner = getattr(view, "_torch_data_owner", None)
        if not isinstance(owner, Var):
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
                if isinstance(mask, Var) and mask.dtype in ("bool", "uint8") \
                        and isinstance(value, Var) \
                        and len(mask.shape) < len(self.shape):
                    # Region selected by a lower-rank bool mask has shape
                    # (N, *self.shape[mask.ndim:]). Drop only provably redundant
                    # leading singleton axes from value until ranks agree.
                    region_rank = 1 + (len(self.shape) - len(mask.shape))
                    while len(value.shape) > region_rank and value.shape[0] == 1:
                        value = value.squeeze(0)
            except EXPECTED as exc:
                swallowed("torch/installers/tensor.py _torch_setitem: mask = slices", exc)
            result = _orig_setitem(self, slices, value)
            _write_index_parent(self, self)
            return result
        _torch_setitem._torch_mask_bcast = True
        Var.__setitem__ = _torch_setitem

    def _write_index_parent(view, value):
        parent = getattr(view, "_torch_index_parent", None)
        parent_slices = getattr(view, "_torch_index_slices", None)
        if not isinstance(parent, Var):
            return
        parent_was_trainable = not parent.is_stop_grad()
        # Bypass the compatibility wrapper here; this helper owns the one
        # explicit ancestor walk. Calling patched __setitem__ would recurse once
        # implicitly and once below, duplicating graph nodes at every depth.
        _orig_setitem(parent, parent_slices, value)
        if parent_was_trainable and parent.is_stop_grad():
            parent.start_grad()
        elif not parent_was_trainable and not parent.is_stop_grad():
            parent.stop_grad()
        # Jittor basic indexing materializes a separate Var. Propagate a
        # mutation through every retained view so x[0][1].zero_() reaches x.
        _write_index_parent(parent, parent)

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
        _write_index_parent(target, value)
        target.assign(value)
        if was_trainable and target.is_stop_grad():
            target.start_grad()
        elif not was_trainable and not target.is_stop_grad():
            target.stop_grad()
        return self
    def _copy_(self, other, non_blocking=False):
        src = other if isinstance(other, Var) else jt.array(other)
        return _ip(self, src.cast(str(self.dtype)) if hasattr(self, "dtype") else src)
    if not hasattr(Var, "copy_"):
        Var.copy_ = _copy_

    # torch's new_*(size, *, dtype=, device=, requires_grad=) factory methods.
    # jittor's native new_ones/new_zeros only take a size, so override to accept
    # torch kwargs (dtype defaults to self's dtype, like torch).
    def _norm_size(args):
        # torch allows new_ones(2,3), new_ones((2,3)), or new_ones(<NanoVector/Size>)
        # -- unwrap any single iterable that isn't itself a scalar int/Var.
        if len(args) == 1 and not isinstance(args[0], (int, jt.Var)) \
                and hasattr(args[0], "__len__"):   # tuple/list/NanoVector/Size
            args = tuple(args[0])
        # torch accepts 0-d int Vars / numpy ints as sizes (e.g. longformer computes
        # dims via torch.div); jittor's factories need plain ints -- coerce.
        return tuple(int(s.item()) if isinstance(s, jt.Var) else int(s) for s in args)
    def _resolve_size(size, kw):
        # torch allows new_ones(2,3), new_ones((2,3)) AND the keyword form
        # new_ones(size=(2,3)) (used by longformer's new_ones(size=mask.size())).
        if not size and "size" in kw:
            return (kw["size"],)
        return size
    def _new_finish(v, device=None, requires_grad=False):
        if _device_is_cpu(device):
            v = _make_cpu_resident(v)
        elif _device_is_cuda(device):
            jt.flags.use_cuda = 1
            v = _make_cuda_resident(v, force=True)
        if requires_grad:
            v.requires_grad_(True)
            _torch_register_leaf(v)
        return v
    def _new_ones(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return _new_finish(jt.ones(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)
    def _new_zeros(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return _new_finish(jt.zeros(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)
    def _new_full(self, size, fill_value, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        # size may be a tuple/list/torch.Size OR a jittor NanoVector (e.g. from
        # x.new_full(x.shape, v)); both are iterable with __len__.
        shp = tuple(int(s) for s in size) if hasattr(size, "__len__") else (int(size),)
        return _new_finish(jt.full(shp, fill_value).cast(dt), device, requires_grad)
    def _new_empty(self, *size, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        return _new_finish(jt.empty(_norm_size(_resolve_size(size, kw)), dt), device, requires_grad)
    def _new_tensor(self, data, dtype=None, device=None, requires_grad=False, **kw):
        dt = _dtype_to_str(dtype) if dtype is not None else str(self.dtype)
        # torch's new_tensor accepts a python list whose elements are 0-d tensors
        # (e.g. centernet_update_head builds start_coord_pre_level by accumulating
        # `_start = _start + batch * area_per_level[level]`, where the indexed term
        # is a scalar). jittor has no 0-d tensors, so those scalars are [1] Vars and
        # jt.array([int, Var, Var, ...]) raises "inhomogeneous shape". Coerce any
        # numel-1 Var element to a python number first so the list is homogeneous.
        if isinstance(data, (list, tuple)):
            def _coerce(v):
                if isinstance(v, jt.Var):
                    return v.item() if v.numel() == 1 else v.tolist()
                if isinstance(v, (list, tuple)):
                    return [_coerce(e) for e in v]
                return v
            data = [_coerce(v) for v in data]
        return _new_finish(jt.array(data).cast(dt), device, requires_grad)
    Var.new_ones = _new_ones
    Var.new_zeros = _new_zeros
    Var.new_full = _new_full
    Var.new_empty = _new_empty
    Var.new_tensor = _new_tensor
    # Override the native methods even when they already exist. Transformers
    # initializes parameters through ``param.data.normal_()/zero_()/fill_()``
    # inside @torch.no_grad(); Jittor's native bound initializers adopt the
    # constant source's stop-grad flag and permanently freeze the parameter.
    Var.fill_ = lambda self, val: _ip(self, jt.ones(self.shape, self.dtype) * val)
    Var.zero_ = lambda self: _ip(self, jt.zeros(self.shape, self.dtype))
    Var.add_ = lambda self, o, alpha=1: _ip(self, self + (o * alpha))
    Var.sub_ = lambda self, o, alpha=1: _ip(self, self - (o * alpha))
    Var.mul_ = lambda self, o: _ip(self, self * o)
    Var.div_ = lambda self, o: _ip(self, self / o)
    # in-place unary math ops (recurrent_gemma uses x.log_(); common torch idioms)
    for _name, _fn in (("log_", jt.log), ("exp_", jt.exp), ("sqrt_", jt.sqrt),
                       ("neg_", lambda x: -x), ("abs_", jt.abs), ("sigmoid_", jt.sigmoid),
                       ("tanh_", jt.tanh), ("reciprocal_", lambda x: 1.0 / x),
                       ("rsqrt_", lambda x: 1.0 / jt.sqrt(x))):
        if not hasattr(Var, _name):
            setattr(Var, _name, (lambda fn: lambda self: _ip(self, fn(self)))(_fn))
    # torch.clamp(input, min=None, max=None) and Tensor.clamp(min=, max=)
    # accept min/max as keyword args, either of which may be None. jittor's
    # native clamp only takes them positionally and rejects the keywords (it
    # also exposes `low`/`high` names, not `min`/`max`). Wrap both the
    # top-level op and the method so torch's keyword form works, while plain
    # positional calls (jittor's own usage) pass straight through unchanged.
    _native_clamp = jt.clamp
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
        a = input if isinstance(input, Var) else jt.array(input)
        b = other if isinstance(other, Var) else jt.array(other)
        if str(a.dtype) == "bool":
            a = a.int32()
        if isinstance(b, Var) and str(b.dtype) == "bool":
            b = b.int32()
        diff = (a - b).abs()
        out = diff > 0
        if "float" in str(a.dtype) or (isinstance(b, Var) and "float" in str(b.dtype)):
            try:
                out = out | jt.isnan(a) | jt.isnan(b)
            except EXPECTED as exc:
                swallowed("torch/installers/tensor.py _torch_ne: out = out | jt.isnan(a) | jt.isnan(b)", exc)
        return out

    g.ne = _torch_ne
    g.not_equal = _torch_ne
    Var.ne = lambda self, other: _torch_ne(self, other)
    Var.__ne__ = lambda self, other: _torch_ne(self, other)

    # torch's Tensor.nonzero(as_tuple=False) returns an (N, ndim) index matrix;
    # nonzero(as_tuple=True) instead returns a tuple of ndim 1-D index Vars (one
    # per dimension) -- transformers/diffusers use the tuple form for advanced
    # indexing. jittor's nonzero only returns the matrix and rejects as_tuple.
    _native_nonzero = getattr(jt, "_vj_native_nonzero", jt.nonzero)
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
    Var.normal_ = lambda self, mean=0.0, std=1.0, generator=None: _ip(self, jt.normal(float(mean), float(std), self.shape).cast(str(self.dtype)))
    Var.uniform_ = lambda self, a=0.0, b=1.0, generator=None: _ip(self, (jt.rand(self.shape)*(b-a)+a).cast(str(self.dtype)))

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

    # cumsum: ACL's aclnnCumsum SEGFAULTS on bool input (transformers builds
    # position_ids via mask.cumsum(-1)). torch.cumsum promotes bool/uint8 to
    # int64 anyway, so cast before the native op to match torch AND avoid the
    # crash. Override both torch.cumsum and Var.cumsum (g IS the jittor module).
    _native_cumsum = jt.cumsum
    def _assign_out(out, value):
        out.assign(value)
        _write_index_parent(out, out)
        return out

    _native_add = g.add
    def _add(input, other, *, alpha=1, out=None):
        if alpha != 1:
            other = other * alpha
        result = _native_add(input, other)
        if out is not None:
            return _assign_out(out, result)
        return result
    g.add = _add

    def _cumsum(x, dim=-1, dtype=None, out=None, **kw):
        if isinstance(x, jt.Var) and str(x.dtype) in ("bool", "uint8"):
            x = x.cast("int64")
        r = _native_cumsum(x, dim)
        if dtype is not None:
            r = r.cast(_dtype_to_str(dtype))
        if out is not None:
            return _assign_out(out, r)
        return r
    g.cumsum = _cumsum
    Var.cumsum = lambda self, dim=-1, dtype=None, out=None, **kw: _cumsum(self, dim, dtype, out=out)
    # cumprod has the same ACL fragility; guard it the same way if present.
    if hasattr(jt, "cumprod"):
        _native_cumprod = jt.cumprod
        def _cumprod(x, dim=-1, dtype=None, out=None, **kw):
            if isinstance(x, jt.Var) and str(x.dtype) in ("bool", "uint8"):
                x = x.cast("int64")
            r = _native_cumprod(x, dim)
            if dtype is not None:
                r = r.cast(_dtype_to_str(dtype))
            if out is not None:
                return _assign_out(out, r)
            return r
        g.cumprod = _cumprod
        Var.cumprod = lambda self, dim=-1, dtype=None, out=None, **kw: _cumprod(self, dim, dtype, out=out)

    # bitwise/logical operators torch supports on tensors
    if not hasattr(Var, "__invert__"):
        def _invert(self):
            if str(self.dtype) == "bool":
                return self.logical_not()
            return jt.logical_not(self) if str(self.dtype) == "bool" else (-self - 1)
        Var.__invert__ = _invert

    def _device(self):
        # Inside a `with torch.device("meta")` block (transformers'
        # from_pretrained), report "meta" so its meta-context detection
        # fires and eager weight init is skipped. See device.__enter__.
        if _DEVICE_CTX_STACK:
            return _DEVICE_CTX_STACK[-1]
        # Report the Var's ACTUAL memory residency (matches jtorch's C++
        # is_cpu()/device()): a Var built/migrated to host -- e.g. via
        # torch.zeros(device='cpu') or .cpu() -- is "cpu" even while the
        # global use_cuda flag is 1. Only fall back to the global flag when
        # CUDA is on and the Var is genuinely device-resident.
        if (jt.flags.use_cuda or getattr(jt.compiler, "has_acl", 0)):
            if _var_is_cpu_resident(self):
                return device("cpu")
            # The index is the Var's own, not a hardcoded 0: every Var carries
            # the device it lives on.
            idx = getattr(self, "device_id", 0)
            return device("cuda", int(idx) if idx is not None and idx >= 0 else 0)
        return device("cpu")
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
            return isinstance(index, numbers.Integral) and not isinstance(index, (bool, np.bool_))

        def _torch_getitem(self, slices):
            out = _orig_getitem(self, slices)
            if isinstance(out, Var) and _var_has_cpu_residency_hint(self):
                out = _mark_cpu_like(out, self)
            # Only basic indexing returns a view in PyTorch. Retaining the
            # parent for advanced-index copies creates optimizer-state chains
            # across Gaussian Splatting densification generations.
            if isinstance(out, Var) and _is_basic_index(slices):
                try:
                    out._torch_index_parent = self
                    out._torch_index_slices = slices
                    data_owner = getattr(self, "_torch_data_owner", None)
                    if isinstance(data_owner, Var):
                        out._torch_data_owner = data_owner
                        out._torch_data_path = getattr(
                            self, "_torch_data_path", ()
                        ) + (slices,)
                except EXPECTED as exc:
                    swallowed("torch/installers/tensor.py _torch_getitem: out._torch_index_parent = self", exc)
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
                return _mark_cpu_like(out, self, other)
            _wrapped._torch_cpu_residency = True
            return _wrapped
        setattr(Var, _op_name, _make_cpu_binary_wrapper(_orig_op))

    # torch's Tensor.data returns a detached *tensor* (and is assignable:
    # `param.data = new_tensor`). jittor's native Var.data returns a numpy
    # ndarray, breaking `param.data.to(...)`. Override to torch semantics.
    if not getattr(Var, "_data_wrapped", False):
        def _data_get(self):
            # Only when this interpreter is actually serving the torch
            # namespace. Composition runs either way, and a plain
            # ``import jittor`` must keep Jittor's own contract, where ``.data``
            # is a numpy view -- code like ``a.data[mask]`` with a numpy mask
            # depends on it, and 2.0 is meant to leave the native interface
            # as it was.
            preflight = getattr(jt, "_compat_preflight_result", None)
            torch_mode = (
                getattr(preflight, "active", False)
                or bool(getattr(jt, "_torch_compat_install_complete", False))
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
            src = value if isinstance(value, Var) else jt.array(value)
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
        Var.__reduce__ = lambda self: (
            _rebuild_var_from_numpy, (self.numpy(), str(self.dtype)))
        Var._reduce_wrapped = True

    # Leaf registry for the no-optimizer backward() path (below): torch's
    # loss.backward() accumulates grads into the .grad of every leaf that
    # requires grad, but jittor has no graph-walk to recover those leaves. So
    # track Vars whose grad was explicitly enabled through the torch-facing
    # API (requires_grad=True / requires_grad_()). Keyed by id() to dedupe;
    # jittor Vars are not weak-referenceable, so we hold strong refs (leaf
    # params are long-lived anyway) and prune entries that drop stop-grad.
    if not hasattr(jt, "_torch_leaf_params"):
        jt._torch_leaf_params = {}
    def _register_leaf(v):
        _torch_register_leaf(v)

    # Override requires_grad with a Python property even though jittor exposes a
    # native getset descriptor: the native setter maps directly to start_grad/
    # stop_grad (identical semantics), but we additionally register the Var as a
    # leaf so the no-optimizer loss.backward() path (below) can find it. This is
    # behavior-preserving for the getter/setter; it only adds leaf bookkeeping.
    if not isinstance(Var.__dict__.get("requires_grad"), property):
        _native_requires_grad = Var.__dict__["requires_grad"]
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
                    if not isinstance(peer, Var) or peer is self:
                        continue
                    _native_requires_grad.__set__(peer, v)
                    if v:
                        _register_leaf(peer)
                if getattr(fsdp_state, "true_fsdp_flat", False):
                    flat = getattr(fsdp_state, "true_fsdp_flat_shard", None)
                    any_trainable = any(getattr(entry, "requires_grad", True)
                                        for entry in fsdp_state.true_fsdp_params)
                    if isinstance(flat, Var):
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
                if not isinstance(p, Var) or not p.requires_grad:
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
                if not isinstance(existing, Var):
                    existing = getattr(p, "_torch_grad", None)
                if isinstance(existing, Var) and list(existing.shape) == list(g.shape):
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
        except EXPECTED as exc:
            swallowed("torch/installers/tensor.py _fill_opt_grads: opt._build_grad_map()", exc,
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
            grad_var = gradient if isinstance(gradient, Var) else jt.array(gradient)
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
        except EXPECTED as exc:
            swallowed("torch/installers/tensor.py _backward: self.sync()", exc)
        # Collect EVERY live optimizer (torch allows several at once — 3DGS uses a
        # Gaussian Adam + an exposure Adam; routing to just _current_optimizer
        # left the other's params with .grad=None -> KeyError 'grads' in step()).
        reg = getattr(jt, "_active_optimizers", None)
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
        _fsdp2_backward = _fsdp_hooks.provider() if fsdp_opts else None
        fsdp_opt_ids = {id(o) for o in fsdp_opts} if _fsdp2_backward is not None else set()
        leaf_map = {}
        opt_ids = set()
        filled_param_ids = set()
        for o in opts:
            for pg in getattr(o, "param_groups", []):
                for p in pg.get("params", []):
                    if not isinstance(p, Var) or not p.requires_grad:
                        continue
                    if _fsdp2_backward is not None and _fsdp2_backward.is_fsdp_managed_param(p):
                        opt_ids.add(id(p))
                        continue
                    leaf_map.setdefault(id(p), p)
                    opt_ids.add(id(p))
        if _fsdp2_backward is not None and fsdp_opts:
            for p in _fsdp2_backward.collect_fsdp_full_params_for_backward(fsdp_opts):
                if isinstance(p, Var) and p.requires_grad:
                    leaf_map.setdefault(id(p), p)
                    opt_ids.add(id(p))
        retained = getattr(jt, "_torch_retained", None)
        retained_ids = set()
        if retained:
            for v in list(retained.values()):
                if isinstance(v, Var) and v.requires_grad:
                    leaf_map.setdefault(id(v), v)
                    retained_ids.add(id(v))
        if opts:
            # Optimizer parameter groups supersede stale Parameter objects after
            # parameter replacement, but unrelated input leaves must still receive
            # gradients just as they do in Torch.
            _torch_prune_leaf_registry(
                opt_ids | retained_ids,
                keep_non_parameters=True,
            )
            for v in list(jt._torch_leaf_params.values()):
                if isinstance(v, Var) and v.requires_grad:
                    leaf_map.setdefault(id(v), v)
        else:
            _torch_prune_leaf_registry()
            for v in list(jt._torch_leaf_params.values()):
                if isinstance(v, Var) and v.requires_grad:
                    leaf_map.setdefault(id(v), v)
        if not leaf_map:
            return None
        leaves = list(leaf_map.values())
        # torch leaves a disconnected target at grad=None. Keep jt.grad's
        # historical zero-materialization untouched and use the compatibility
        # core entry point that preserves missing gradients explicitly.
        grads = jt.core.grad_optional(self, leaves, retain_graph)
        grad_by_id = {}
        for p, gr in zip(leaves, grads):
            if gr is None:
                if id(p) not in opt_ids and id(p) not in retained_ids:
                    jt._torch_leaf_params.pop(id(p), None)
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
        _ddp_all_reduce_grads(leaves)
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
        for r in getattr(jt, "_active_optimizers", None) or []:
            o = r() if callable(r) else r
            if o is None:
                continue
            try:
                return o.find_grad(self)
            except EXPECTED as exc:
                swallowed("torch/installers/tensor.py _grad_get: return o.find_grad(self)", exc)
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
                swallowed("torch/installers/tensor.py _grad_set: if value is None:", exc)
        # Write through by identity so step() sees manual grad assignment and,
        # critically, p.grad=None cannot leave an old optimizer slot behind.
        for r in getattr(jt, "_active_optimizers", None) or []:
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
                    swallowed("torch/installers/tensor.py _grad_set: object.__setattr__(o, '_grad_map', {})", exc)
    Var.grad = property(_grad_get, _grad_set)

    # torch's `is_leaf`: True for tensors not produced by a grad-tracked op
    # (user-created params/inputs). jittor has no autograd-graph leaf concept;
    # treat every Var as a leaf so peft's `if param.is_leaf:` guards pass.
    if not hasattr(Var, "is_leaf"):
        Var.is_leaf = property(lambda self: True)
    # torch's nested-tensor flag; jittor has no nested tensors -> always False.
    if not hasattr(Var, "is_nested"):
        Var.is_nested = property(lambda self: False)
    # torch's `grad_fn` is None for leaves; libs check `t.grad_fn is None`.
    if not hasattr(Var, "grad_fn"):
        Var.grad_fn = property(lambda self: None)
    # torch's retain_grad() marks a NON-leaf tensor so its .grad is populated
    # after backward (normally only leaves keep .grad). 3DGS relies on this for
    # the screenspace `means2D` tensor (`zeros_like(xyz)+0` then retain_grad()),
    # whose .grad drives densification. Register into a per-forward set the
    # _backward pass includes as a grad target; cleared each backward so the
    # next iteration's fresh tensor doesn't accumulate (jittor Vars can't be
    # weak-ref'd, so a persistent dict would leak one Var per iteration).
    if not hasattr(jt, "_torch_retained"):
        jt._torch_retained = {}
    def _retain_grad(self):
        try:
            jt._torch_retained[id(self)] = self
        except EXPECTED as exc:
            swallowed("torch/installers/tensor.py _retain_grad: jt._torch_retained[id(self)] = self", exc)
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
            if isinstance(a, dtype):
                ds = a.name
            elif isinstance(a, device):
                dev = a
            elif isinstance(a, Var):
                # .to(other) copies other's dtype AND device.
                ds = str(a.dtype)
                dev = a.device
            elif isinstance(a, str):
                bare = a.replace("torch.", "")
                if bare in dtype._registry:
                    ds = bare
                elif bare.split(":")[0] in ("cpu", "cuda", "npu"):
                    dev = bare
        if ds is not None:
            out = self.cast(ds) if copy else _cast_if_needed(self, ds)
        else:
            out = self.clone() if copy else self
        # Honor an explicit device= target by migrating residency. device=None
        # (the common .to(dtype) call) leaves placement on the global default.
        if _device_is_cpu(dev):
            out = _make_cpu_resident(out)
        elif _device_is_cuda(dev):
            src_index = getattr(self, "device_id", -1)
            out = _make_cuda_resident(out, force=True)
            # .to("cuda:N") copies across devices when N is not where the Var
            # already is; a bare .to("cuda") leaves the tensor on its own
            # device, as in torch.
            moved = _move_to_cuda_index(out, dev, src_index)
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
        out = _make_cpu_resident(self)
        try:
            out._jittor_torch_force_cpu = True
            if getattr(self, "_torch_0d", False):
                out._torch_0d = True
        except (AttributeError, TypeError) as exc:
            swallowed("torch/installers/tensor.py _var_cpu: out._jittor_torch_force_cpu = True", exc)
        return out
    Var.cpu = _var_cpu
    def _var_cuda(self, device=None, *a, **k):
        jt.flags.use_cuda = 1
        src_index = getattr(self, "device_id", -1)
        out = _make_cuda_resident(self, force=True)
        # .cuda(N) is .to("cuda:N"); .cuda() keeps the tensor where it is.
        out = _move_to_cuda_index(out, device, src_index)
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
        ds = _dtype_to_str(dst_type)
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
        return jt.array(np.asarray([value], dtype=np.complex64))

    def _make_promoting_op(opname, reflected):
        native = Var.__dict__.get(opname)
        if native is None:
            return None
        def _op(self, other):
            if isinstance(other, (complex, np.complexfloating)):
                other = _complex_scalar_var(other)
            if isinstance(other, Var):
                da, db = str(self.dtype), str(other.dtype)
                if da == db and not da.startswith("uint"):
                    return native(self, other)
                res = g._torch_promote_pair(da, db)
                a = self if da == res else self.cast(res)
                b = other if db == res else other.cast(res)
                out = native(a, b)
                # native may still mis-infer (unsigned -> signed); fix it up.
                if isinstance(out, Var) and str(out.dtype) != res:
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
            if isinstance(other, (bool, int, float)) and isinstance(out, Var):
                expected = _dtype_to_str(g.result_type(self, other))
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
        return _dtype_to_str(g.get_default_dtype()) or "float32"
    def _scalar_dtype_name(x):
        if isinstance(x, bool):
            return "bool"
        if isinstance(x, int):
            return "int64"
        if isinstance(x, float):
            return _dtype_to_str(g.get_default_dtype()) or "float32"
        if isinstance(x, complex):
            return "complex64"
        return None
    def _make_truediv(opname):
        native = Var.__dict__.get(opname)
        if native is None:
            return None
        def _op(self, other):
            if isinstance(other, (complex, np.complexfloating)):
                other = _complex_scalar_var(other)
            if isinstance(other, Var):
                da, db = str(self.dtype), str(other.dtype)
                if da == db and da.startswith(("float", "bfloat", "complex")):
                    return native(self, other)
                tgt = _truediv_target(da, db)
                a = self if da == tgt else self.cast(tgt)
                b = other if db == tgt else other.cast(tgt)
                out = native(a, b)
                if isinstance(out, Var) and str(out.dtype) != tgt:
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
                acl_active = bool(getattr(jt.compiler, "has_acl", 0)) and (
                    bool(getattr(jt.flags, "use_acl", 0)) and bool(jt.flags.use_cuda))
                use_wide = sd.startswith("float") and src_dt != "float64" and not acl_active
                calc_dt = "float64" if use_wide else tgt
                a = self if src_dt == calc_dt else self.cast(calc_dt)
                b = jt.array(other, dtype=calc_dt) if use_wide else other
                out = native(a, b)
                if isinstance(out, Var) and str(out.dtype) != tgt:
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
        if not (jt.flags.use_cuda or getattr(jt.compiler, "has_acl", 0)):
            return False
        return not _var_is_cpu_resident(self)
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
                ar = jt.arange(size[d], dtype="int64") * stride[d]
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
                return _torch_where_select(condition, self, other)
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
            res = jt.matmul(batch1, batch2)
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
            res = jt.matmul(mat1, mat2)
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
    _norm_via = _torch_norm_impl
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


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    _DTYPE_OBJS = ctx.state["dtypes"]
    # jt.grad's C-binding only accepts a *plain* list of targets, so passing the
    # torch-style parameters() iterator/_ParamList (a list subclass) or a single
    # Var raises a cryptic "Wrong inputs arguments". Coerce to a plain list (and
    # accept a lone Var, like torch.autograd.grad). Internal jittor callers pass a
    # plain list -> passthrough, so this never changes their behavior.
    _native_grad = g.grad
    def _grad_compat(loss, targets, *a, **k):
        # A lone Var target must return a lone grad (native jt.grad unwraps it via
        # core.grad(...)[0]). Wrapping it into [targets] here made jt.grad(loss, var)
        # return a 1-element LIST instead of a Var, breaking single-target callers
        # (e.g. softmax/ctc backward in test_misc_op). Remember the single-Var case and
        # unwrap the result to restore native behavior; list/iterable targets pass through.
        single = isinstance(targets, jt.Var)
        if type(targets) is not list:
            if single:
                targets = [targets]
            else:
                try:
                    targets = list(targets)
                except EXPECTED as exc:
                    swallowed("torch/installers/tensor.py _grad_compat: targets = list(targets)", exc)
        res = _native_grad(loss, targets, *a, **k)
        if single and isinstance(res, (list, tuple)) and len(res) == 1:
            return res[0]
        return res
    g.grad = _grad_compat

    # torch.no_grad / enable_grad work as bare decorator (@torch.no_grad),
    # called decorator (@torch.no_grad()), and context manager.
    # NB: g IS the jittor module, so capture the originals before overwriting.
    _orig_no_grad = jt.no_grad
    _orig_enable_grad = jt.enable_grad
    g.no_grad = lambda func=None: _GradDecoratorCtx(_orig_no_grad, func)
    g.enable_grad = lambda func=None: _GradDecoratorCtx(_orig_enable_grad, func)
    g.inference_mode = lambda func=None: _GradDecoratorCtx(_orig_no_grad, func)

    Var = jt.Var
    _native_index_select = g.index_select
    def _index_select(input, dim, index, *, out=None):
        result = _native_index_select(input, dim, index)
        if out is not None:
            out[...] = result
            return out
        return result
    g.index_select = _index_select
    # torch.Tensor is both (a) the isinstance target and (b) a legacy constructor:
    # torch.Tensor(d0, d1, ...) makes an UNINITIALISED tensor of that shape (DETR's
    # _init_layers: torch.Tensor(num_levels, embed_dims)), while torch.Tensor(data)
    # builds from data. A metaclass gives us both without breaking isinstance(x, Var).
    class _TensorMeta(type):
        def __instancecheck__(cls, inst):
            return isinstance(inst, (Var, _NestedTensor))
        def __subclasscheck__(cls, sub):
            return issubclass(sub, (Var, _NestedTensor))
        def __call__(cls, *args, **kw):
            if len(args) == 0:
                return jt.empty((0,))
            if all(isinstance(a, int) for a in args):   # torch.Tensor(*sizes)
                return jt.empty(tuple(args))
            # torch.Tensor(size) with a shape object (torch.Size / our Size / a
            # jittor NanoVector, e.g. weight.size()) -> an uninitialized tensor of
            # that shape, NOT data (mmdet SAConv2d: torch.Tensor(self.weight.size())).
            if len(args) == 1 and isinstance(args[0], (jt.NanoVector, Size)):
                return jt.empty(tuple(int(x) for x in args[0]))
            data = args[0]
            if isinstance(data, Var):
                return data.float32()
            return jt.array(data).float32()
    class Tensor(metaclass=_TensorMeta):
        pass
    g.Tensor = Tensor
    # torch's typed tensor classes (FloatTensor/LongTensor/...). jittor is dtype-typed
    # at the data level (no tensor subclasses), but we must NOT just alias them all to
    # Var: that makes isinstance(any_var, torch.LongTensor) always True, so libraries
    # that detect integer tensors via isinstance break with silent-wrong results
    # (e.g. diffusers EulerDiscreteScheduler.step rejects every float timestep with
    # "Passing integer indices ... is not supported"). Instead give each a metaclass
    # whose isinstance check matches the Var's actual dtype, and whose construction
    # casts to that dtype (torch.FloatTensor(2,3) / torch.LongTensor([1,2])).
    _TYPED_TENSOR_DTYPE = {
        "FloatTensor": "float32", "DoubleTensor": "float64", "HalfTensor": "float16",
        "BFloat16Tensor": "bfloat16", "LongTensor": "int64", "IntTensor": "int32",
        "ShortTensor": "int16", "CharTensor": "int8", "ByteTensor": "uint8",
        "BoolTensor": "bool",
    }
    class _TypedTensorMeta(type):
        def __instancecheck__(cls, obj):
            return isinstance(obj, Var) and str(obj.dtype) == cls._jdtype
        def __call__(cls, *args, **kw):
            if len(args) == 1 and isinstance(args[0], Var):
                v = args[0]
            elif len(args) == 1 and not isinstance(args[0], int):
                v = jt.array(args[0])           # from list/ndarray
            elif len(args) == 0:
                v = jt.zeros((0,))
            else:
                v = jt.zeros(tuple(int(a) for a in args))  # from sizes
            return v.cast(cls._jdtype)
    for _tn, _dt in _TYPED_TENSOR_DTYPE.items():
        setattr(g, _tn, _TypedTensorMeta(_tn, (), {"_jdtype": _dt}))

    def _array_keep_dtype(data):
        # jittor's jt.array downcasts numpy int64 -> int32; torch keeps int64.
        # Preserve the source dtype for (u)int64/float64 so dtypes match torch.
        import numpy as _np
        # jt.array rejects ndarray SUBCLASSES (e.g. the adapter's numpy-backed
        # buffer tensors) -> coerce to a base ndarray (same data, no copy).
        if isinstance(data, _np.ndarray) and type(data) is not _np.ndarray:
            data = _np.asarray(data)
        if isinstance(data, _np.ndarray):
            dn = data.dtype.name
            # jt.array(numpy_int64) silently downcasts to int32, OVERFLOWING values
            # that don't fit in 32 bits (e.g. byte counts ~1e10) BEFORE any later
            # .int64() cast can recover them. Build the wide-dtype Var directly.
            if dn in ("int64", "uint64"):
                return jt.array(data, dtype="int64")
            if dn == "float64":
                return jt.array(data, dtype="float64")
        return jt.array(data)

    def tensor(data, dtype=None, device=None, requires_grad=False, **kw):
        import numpy as _np
        ds = _dtype_to_str(dtype)
        if isinstance(data, Var):
            v = data.clone()
        elif isinstance(data, _np.ndarray):
            # Respect an explicit complex64 request before constructing the Var.
            # NumPy otherwise keeps complex literals as unsupported complex128,
            # so casting only after jt.array() is too late.
            if ds == "complex64" and data.dtype.name != "complex64":
                data = _np.asarray(data, dtype=_np.complex64)
            v = _array_keep_dtype(data)          # explicit numpy: preserve dtype (torch does too)
        else:
            # torch's tensor/as_tensor([t1, t2, ...]) flattens SCALAR tensors into a
            # 1-D tensor; jittor has no 0-d scalars (a "scalar" Var is shape (1,)), so
            # numpy.asarray of a list-of-Vars adds a spurious dim ((1,)->(1,1)). Coerce
            # contained scalar Vars to Python numbers first (e.g. tapas builds shapes
            # via torch.as_tensor([index.num_segments])).
            if isinstance(data, (list, tuple)) and any(isinstance(d, Var) for d in data):
                data = [(d.item() if isinstance(d, Var) and d.numel() == 1 else d)
                        for d in data]
            # Python scalar/list/tuple: numpy infers float64 from Python floats, but
            # torch's default float dtype is float32. Match torch (and avoid float64,
            # which Ascend/ACL does not support) by downcasting inferred float64.
            arr = _np.asarray(data, dtype=_np.complex64 if ds == "complex64" else None)
            if arr.dtype == _np.float64:
                arr = arr.astype(_np.float32)
            elif arr.dtype == _np.complex128 and ds != "complex128":
                # torch's default complex dtype follows its default float dtype,
                # so Python complex literals default to complex64.
                arr = arr.astype(_np.complex64)
            v = _array_keep_dtype(arr)
        if ds is not None:
            v = v.cast(ds)
        # torch.tensor(..., device='cpu') must land in host memory so native
        # extensions' tensor.is_cpu() checks pass.
        if _device_is_cpu(device):
            v = _make_cpu_resident(v)
        elif _device_is_cuda(device):
            jt.flags.use_cuda = 1
            v = _make_cuda_resident(v, force=True)
        if requires_grad:
            v.requires_grad_(True)
            _torch_register_leaf(v)
        v._jt_plain_tensor = True   # see _torch_style_registration (core_api)
        return v
    g.tensor = tensor

    def as_tensor(data, dtype=None, device=None):
        if isinstance(data, Var):
            r = data if dtype is None else data.cast(_dtype_to_str(dtype))
            if _device_is_cpu(device):
                return _make_cpu_resident(r)
            if _device_is_cuda(device):
                jt.flags.use_cuda = 1
                return _make_cuda_resident(r, force=True)
            return r
        return tensor(data, dtype=dtype, device=device)
    g.as_tensor = as_tensor

    def from_numpy(arr, *, device=None):
        v = _array_keep_dtype(arr)
        if _device_is_cpu(device):
            return _make_cpu_resident(v)
        if _device_is_cuda(device):
            jt.flags.use_cuda = 1
            return _make_cuda_resident(v, force=True)
        return v
    g.from_numpy = from_numpy

    def frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False):
        import numpy as _np
        ds = _dtype_to_str(dtype)
        np_dtype = {
            "bool": _np.bool_, "uint8": _np.uint8, "int8": _np.int8,
            "uint16": _np.uint16, "int16": _np.int16,
            "uint32": _np.uint32, "int32": _np.int32,
            "uint64": _np.uint64, "int64": _np.int64,
            "float16": _np.float16, "float32": _np.float32,
            "float64": _np.float64,
        }.get(ds)
        if ds == "bfloat16":
            raw = _np.frombuffer(buffer, dtype=_np.uint16, count=count, offset=offset)
            arr = (raw.astype(_np.uint32) << 16).view(_np.float32)
            v = from_numpy(_np.ascontiguousarray(arr))
        else:
            if np_dtype is None:
                raise TypeError(f"torch.frombuffer unsupported dtype: {dtype}")
            arr = _np.frombuffer(buffer, dtype=np_dtype, count=count, offset=offset)
            v = from_numpy(_np.ascontiguousarray(arr))
        if requires_grad:
            v.requires_grad_(True)
            _torch_register_leaf(v)
        return v
    g.frombuffer = frombuffer

    Size = _TorchSize
    g.Size = Size

    # torch.broadcast_shapes(*shapes) -> Size : broadcasted shape of the inputs
    # (used by verl's advantage/reward broadcasting). numpy implements the same rule.
    def broadcast_shapes(*shapes):
        import numpy as _npb
        norm = [(int(s),) if isinstance(s, (int, np.integer)) else tuple(int(d) for d in s) for s in shapes]
        return Size(_npb.broadcast_shapes(*norm)) if norm else Size(())
    g.broadcast_shapes = broadcast_shapes

    # torch.corrcoef(input) -> correlation-coefficient matrix (verl logs the
    # rollout-vs-recompute logprob correlation as a diagnostic). numpy matches.
    def corrcoef(x, *a, **k):
        import numpy as _npc
        r = _npc.corrcoef(x.float32().numpy())
        return jt.array(_npc.ascontiguousarray(r))
    g.corrcoef = corrcoef

    # torch.Generator (RNG handle) -- jittor uses a global seed; provide a
    # lightweight stand-in that supports manual_seed and is accepted where a
    # generator is passed (it is otherwise ignored).
    class Generator:
        def __init__(self, device=None):
            self.device = globals()["device"](device or "cpu")
            self._seed = 0
        def manual_seed(self, s):
            self._seed = int(s)
            return self
        def get_state(self):
            return jt.array([self._seed])
        def set_state(self, s):
            return self
        def seed(self):
            return self._seed
        def initial_seed(self):
            return self._seed
    g.Generator = Generator

    # numeric / misc top-level constants and small types
    import math as _math
    g.inf = _math.inf
    g.nan = _math.nan
    g.pi = _math.pi
    g.e = _math.e
    g.strided = "strided"
    g.jagged = "jagged"
    g.contiguous_format = "contiguous_format"
    g.preserve_format = "preserve_format"
    g.channels_last = "channels_last"
    class layout:  # torch.layout placeholder
        pass
    g.layout = layout
    class memory_format:
        pass
    g.memory_format = memory_format

    import types as _types_nested
    nested_mod = _types_nested.ModuleType("torch.nested")
    nested_mod.__path__ = []
    def _nested_from_tensors(tensors, *a, layout=None, **k):
        return _NestedTensor.from_tensors(tensors, ragged_idx=k.pop("ragged_idx", 1))
    def _nested_from_jagged(values, offsets, *a, **k):
        return _NestedTensor.from_jagged(values, offsets, ragged_idx=k.pop("ragged_idx", None))
    nested_mod.as_nested_tensor = _nested_from_tensors
    nested_mod.nested_tensor = _nested_from_tensors
    nested_mod.nested_tensor_from_jagged = _nested_from_jagged
    g.nested = nested_mod
    _modules["torch.nested"] = nested_mod
    nested_internal_mod = _types_nested.ModuleType("torch.nested._internal")
    nested_internal_mod.__path__ = []
    nested_tensor_mod = _types_nested.ModuleType("torch.nested._internal.nested_tensor")
    nested_tensor_mod.NestedTensor = _NestedTensor
    nested_internal_mod.nested_tensor = nested_tensor_mod
    nested_mod._internal = nested_internal_mod
    _modules["torch.nested._internal"] = nested_internal_mod
    _modules["torch.nested._internal.nested_tensor"] = nested_tensor_mod

    # torch._check family: assertion helpers used by dynamo / TorchScript-friendly
    # code (e.g. vLLM's sampler does `torch._check(x.shape[0] >= 1)`). The message
    # may be a zero-arg callable that torch invokes lazily only on failure. The
    # condition is usually a python bool but can be a bool tensor (_check_tensor_all).
    def _check_to_pybool(cond):
        if hasattr(cond, "all") and not isinstance(cond, (bool, int, float)):
            try:
                return bool(cond.all().item())
            except EXPECTED as exc:
                swallowed("torch/installers/tensor.py _check_to_pybool: return bool(cond.all().item())", exc)
                return bool(cond)
        return bool(cond)
    def _check_with(_exc):
        def _chk(cond, message=None):
            if not _check_to_pybool(cond):
                msg = message() if callable(message) else message
                raise _exc(msg if msg is not None else "Expected cond to be True, but got False")
        return _chk
    g._check = _check_with(RuntimeError)
    g._check_is_size = lambda i, message=None, **k: g._check(int(i) >= 0, message)
    g._check_index = _check_with(IndexError)
    g._check_value = _check_with(ValueError)
    g._check_type = _check_with(TypeError)
    g._check_not_implemented = _check_with(NotImplementedError)
    g._check_tensor_all = _check_with(RuntimeError)
    g._assert_async = lambda t, *a, **k: g._check(_check_to_pybool(t), "torch._assert_async failed")

    # torch.cat: tolerate empty tensors (skip zero-numel inputs) like torch,
    # accept `dim=`/`out=`. jittor's concat trips on an empty leading tensor.
    _jt_concat = jt.concat
    def cat(tensors, dim=0, out=None, axis=None):
        if axis is not None: dim = axis      # torch accepts axis= (mmrotate PSC head)
        # Honor the __torch_function__ protocol: tensordict (and other tensor-likes)
        # override torch.cat to handle their own structure -- e.g. cat a list of
        # TensorDicts field-by-field. Without this, jittor's concat treats each
        # TensorDict as a Var (dtype None) and aborts. Delegate to the first arg
        # whose type overrides __torch_function__ (Vars are handled normally below).
        try:
            _seq = list(tensors)
        except TypeError:
            _seq = None
        if _seq is not None:
            if any(isinstance(_t, _NestedTensor) for _t in _seq):
                assert all(isinstance(_t, _NestedTensor) for _t in _seq), "cannot cat nested and dense tensors together"
                if dim == 0:
                    parts = []
                    for _t in _seq:
                        parts.extend(list(_t.unbind(0)))
                    return _NestedTensor.from_tensors(
                        parts,
                        ragged_idx=getattr(_seq[0], "_ragged_idx", _seq[0].dim() - 1),
                    )
                assert all(len(_t) == len(_seq[0]) for _t in _seq), "nested cat with dim!=0 requires same batch size"
                return _NestedTensor.from_tensors(
                    [_jt_concat([_t.unbind(0)[i] for _t in _seq], dim=dim - 1) for i in range(len(_seq[0]))],
                    ragged_idx=getattr(_seq[0], "_ragged_idx", _seq[0].dim() - 1),
                )
            for _t in _seq:
                _tf = getattr(type(_t), "__torch_function__", None)
                if _tf is not None and not isinstance(_t, jt.Var):
                    _kw = {}
                    if dim != 0: _kw["dim"] = dim
                    if out is not None: _kw["out"] = out
                    return _tf(g.cat, (type(_t),), (_seq,), _kw)
        tensors = [t for t in tensors if t is not None]
        nonempty = [t for t in tensors if t.numel() > 0]
        if len(nonempty) == 0:
            return tensors[0]
        if len(nonempty) == 1:
            return nonempty[0]
        # torch requires all tensors to share ndim. jittor has no 0-d scalars, so
        # a torch-scalar `s` (0-d) becomes a [1] Var and `s.unsqueeze(0)` yields
        # [1,1] instead of torch's [1] -- mixing 2-D and 1-D entries that torch
        # would see as uniformly 1-D (e.g. SOLO's per-image dice losses). Strip
        # the spurious LEADING size-1 dims off any over-ranked entry so the ndims
        # line up the way torch sees them. Only size-1 leading dims are removed;
        # a genuine ndim/shape mismatch is left for jittor's concat to reject.
        min_nd = min(t.ndim for t in nonempty)
        fixed = []
        for t in nonempty:
            while t.ndim > min_nd and t.shape[0] == 1:
                t = t.squeeze(0)
            fixed.append(t)
        out_var = _jt_concat(fixed, dim)
        # jittor's concat downcasts a uniform uint8 input to int8 (e.g. mask-rcnn-c4
        # builds a uint8 pos_inds mask via torch.cat of uint8 ones/zeros). torch keeps
        # the common input dtype; restore it so downstream byte-mask indexing works.
        in_dtypes = {str(t.dtype) for t in fixed}
        if len(in_dtypes) == 1:
            d = in_dtypes.pop()
            if str(out_var.dtype) != d:
                out_var = out_var.cast(d)
        return out_var
    g.cat = cat
    g.concat = cat
    g.concatenate = cat

    # torch.stack accepts a numpy-style `axis=` alias for `dim=` (and `out=`); jittor's
    # jt.stack is `stack(x, dim=0)` only, so trl's PPO advantage stacking
    # `torch.stack(advantages_reversed[::-1], axis=1)` dies on the unexpected kwarg.
    _jt_stack = jt.stack
    def stack(tensors, dim=0, *, axis=None, out=None):
        if axis is not None: dim = axis
        res = _jt_stack(list(tensors), dim)
        if out is not None:
            out.assign(res)
            return out
        return res
    g.stack = stack

    # Wrap tensor constructors to tolerate torch's device=/requires_grad=/
    # layout=/pin_memory= kwargs and torch dtype objects. jittor's versions
    # don't accept device=, which torch code passes everywhere.
    _wrap_constructors(g)
    _install_random_and_linspace(g)

    _install_reductions(g)


def install_methods(ctx):
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    _install_tensor_methods(g, Var, _DTYPE_OBJS)
    # torch's Tensor.size() returns a torch.Size (tuple subclass) when called with
    # no arg, and an int for size(dim); jittor's native size() returns a NanoVector,
    # which breaks torch idioms like `(n,) + data.size()[1:]` (mmdet's unmap()).
    _Size = getattr(g, "Size", tuple)
    def _torch_size(self, dim=None):
        return self.shape[dim] if dim is not None else _Size(self.shape)
    Var.size = _torch_size

    # jittor's core reshape/view reject a torch.Size (a tuple SUBCLASS) -> normalize
    # a single Size/tuple-subclass arg to a plain tuple so `x.reshape(other.size())`
    # / `x.view(t.size())` works (mmdet queryinst). Only intervene for that case to
    # keep the (very hot) reshape path otherwise untouched.
    _orig_reshape = Var.reshape
    _np_view_of = None
    def _dtype_itemsize_name(ds):
        d = dtype._registry.get(ds)
        if d is not None:
            return d.itemsize
        return dtype(ds).itemsize
    def _bitcast(self, dt):
        import numpy as _np
        nonlocal _np_view_of
        if _np_view_of is None:
            _np_view_of = {"bool": _np.bool_, "uint8": _np.uint8, "int8": _np.int8, "uint16": _np.uint16,
                           "int16": _np.int16, "int32": _np.int32, "int64": _np.int64,
                           "float16": _np.float16, "bfloat16": _np.uint16,
                           "float32": _np.float32, "float64": _np.float64}
        ds = getattr(dt, "name", str(dt)).replace("torch.", "")
        itemsize = getattr(dt, "itemsize", None)
        itemsize = itemsize if isinstance(itemsize, int) else _dtype_itemsize_name(ds)
        old_itemsize = getattr(getattr(self, "dtype", None), "itemsize", None)
        if old_itemsize is None:
            old_itemsize = _dtype_itemsize_name(str(self.dtype))
        shape = list(self.shape)
        if len(shape) == 0:
            if old_itemsize != itemsize:
                raise RuntimeError("view(dtype) cannot change itemsize on a scalar tensor")
        else:
            last_bytes = int(shape[-1]) * int(old_itemsize)
            if itemsize <= 0 or last_bytes % int(itemsize) != 0:
                raise RuntimeError("view(dtype) requires the last dimension to be byte-compatible")
            shape[-1] = last_bytes // int(itemsize)
        reinterpret_view = getattr(jt, "reinterpret_view", None)
        npd = _np_view_of.get(ds, _np.uint8)
        if reinterpret_view is not None and ds in _np_view_of:
            return reinterpret_view(self, shape, ds)
        return jt.array(_np.ascontiguousarray(self.numpy()).view(npd))
    def _torch_reshape(self, *shape, **_kw):
        # torch's `.view(dtype)` / `.view(dtype=...)` REINTERPRETS the bytes as
        # another dtype (bitcast), e.g. weight.view(torch.uint8) for byte-packing
        # in vLLM weight transfer. jittor has no dtype-view; bitcast via numpy.
        # (NB: 'dtype' the kwarg must not shadow the `dtype` class used below.)
        _dt = _kw.get("dtype", None)
        if _dt is not None:
            return _bitcast(self, _dt)
        if not shape:
            # torch spells the target shape as a keyword too: `reshape(shape=...)`
            # (diffusers' DiT unpatchify) and `view(size=...)`. Dropping it left
            # an empty positional tuple and a "shape can't be empty" core error.
            _named = _kw.get("shape", _kw.get("size", None))
            if _named is not None:
                shape = _named if isinstance(_named, (tuple, list)) else (_named,)
                shape = (tuple(int(s) for s in shape),)
        if len(shape) == 1 and isinstance(shape[0], dtype):
            return _bitcast(self, shape[0])
        if len(shape) == 1 and isinstance(shape[0], tuple) and type(shape[0]) is not tuple:
            shape = (tuple(int(s) for s in shape[0]),)
        return _orig_reshape(self, *shape)
    Var.reshape = _torch_reshape
    Var.view = _torch_reshape

    # Keep the existing Torch-facing promotion for narrow integer sums
    # (yolox/rtmdet SimOTA assigners do mask.sum() on a uint8 match matrix).
    # Native CUDA reductions now support these dtypes directly, but exposing the
    # narrow native output here would change the compatibility-layer dtype policy.
    # torch reductions accept a *tuple* of dims (e.g. loss.mean(dim=(1, 2)) in
    # yolact_head, x.sum(dim=(2, 3))). jittor splits these into a scalar overload
    # (kwarg `dim`, single int) and a tuple overload (kwarg `dims`); passing a tuple
    # under `dim` raises "Not a valid keyword: dim". Normalize: route a tuple/list of
    # dims to `dims`, a scalar to `dim`, accepting it via `axis`, `dim`, or as the
    # first positional arg (torch also allows axis as a dim alias).
    def _norm_reduce_kw(a, k):
        d = None
        if "axis" in k:
            d = k.pop("axis")
        if "dim" in k:
            d = k.pop("dim")
        if "dims" in k:
            d = k.pop("dims")
        if d is None and len(a) >= 1:
            if isinstance(a[0], (tuple, list)):
                d = a[0]; a = a[1:]            # consume positional tuple-of-dims
            elif isinstance(a[0], (int, np.integer)) and not isinstance(a[0], bool):
                d = a[0]; a = a[1:]            # consume positional scalar dim
        # torch spells it keepdim; jittor's tuple overload spells it keepdims.
        keep = k.pop("keepdim", k.pop("keepdims", None))
        if keep is None and d is not None and len(a) >= 1 and isinstance(a[0], bool):
            keep = a[0]; a = a[1:]             # consume positional keepdim
        if d is not None:
            # jittor's scalar `dim` overload rejects keepdims, while its tuple
            # `dims` overload supports it -> always route through `dims` when a
            # keepdim was requested (wrap a scalar dim into a 1-tuple).
            if isinstance(d, (tuple, list)):
                k["dims"] = tuple(int(x) for x in d)
            elif keep is not None:
                k["dims"] = (int(d),)
            else:
                k["dim"] = int(d)
        if keep is not None:
            k["keepdims"] = bool(keep)
        return a, k

    def _looks_like_dtype(x):
        return isinstance(x, dtype) or (isinstance(x, str) and x.replace("torch.", "") in dtype._registry)

    _orig_var_sum = Var.sum
    _orig_module_sum = getattr(g, "sum", None)
    def _torch_var_sum(self, *a, **k):
        out = k.pop("out", None)
        dt = k.pop("dtype", None)
        a, k = _norm_reduce_kw(a, k)
        if dt is None and len(a) >= 1 and _looks_like_dtype(a[0]):
            dt = a[0]
            a = a[1:]
        if dt is not None:
            self = self.cast(_dtype_to_str(dt))
        elif str(self.dtype) in ("uint8", "int8", "uint16"):
            self = self.int32()
        result = _orig_var_sum(self, *a, **k)
        if out is not None:
            out.assign(result)
            return out
        return result
    Var.sum = _torch_var_sum
    if _orig_module_sum is not None:
        def _torch_sum(input, *a, **k):
            if isinstance(input, Var):
                return _torch_var_sum(input, *a, **k)
            return _orig_module_sum(input, *a, **k)
        g.sum = _torch_sum
    # Full dim/dims/keepdim normalization for the plain reductions that map onto
    # jittor's scalar-`dim` / tuple-`dims` overload pair (mean/prod/any/all). mmdet
    # exercises tuple dims here, e.g. yolact_head's loss.mean(dim=(1, 2)).
    def _reduce_wrap(orig):
        def _w(self, *a, **k):
            a, k = _norm_reduce_kw(a, k)
            return orig(self, *a, **k)
        return _w
    for _rn in ("mean", "prod"):
        _ro = getattr(Var, _rn, None)
        if _ro is not None:
            setattr(Var, _rn, _reduce_wrap(_ro))
    # any/all: jittor's only accept a scalar `dim` (no `dims` tuple, no keepdims).
    # Support torch's tuple-of-dims and keepdim by reducing one dim at a time
    # (descending so earlier dim indices stay valid), keeping a length-1 axis when
    # keepdim is set. Plain scalar/axis use falls through to the native op.
    def _anyall_wrap(orig, name):
        def _w(self, *a, **k):
            d = None
            if "axis" in k: d = k.pop("axis")
            if "dim" in k:  d = k.pop("dim")
            if "dims" in k: d = k.pop("dims")
            if d is None and len(a) >= 1 and isinstance(a[0], (tuple, list)):
                d = a[0]; a = a[1:]
            keep = k.pop("keepdim", k.pop("keepdims", None))
            if d is None:
                return orig(self, *a, **k)
            dims = [int(x) for x in d] if isinstance(d, (tuple, list)) else [int(d)]
            ndim = self.ndim
            dims = sorted((x % ndim for x in dims), reverse=True)
            out = self
            for ax in dims:
                out = orig(out, dim=ax)
                if keep:
                    out = out.unsqueeze(ax)
            return out
        return _w
    for _rn in ("any", "all"):
        _ro = getattr(Var, _rn, None)
        if _ro is not None:
            setattr(Var, _rn, _anyall_wrap(_ro, _rn))
    # max/min/argmax/argmin/amax/amin/cumsum/norm/std/var are already wrapped above
    # with custom torch-return semantics (value+index tuples, etc.); only translate
    # torch's `axis` alias for them so we don't disturb that handling.
    def _axis_to_dim(orig):
        def _w(self, *a, **k):
            if "axis" in k:
                k["dim"] = k.pop("axis")
            return orig(self, *a, **k)
        return _w
    for _rn in ("max", "min", "argmax", "argmin", "amax", "amin", "cumsum",
                "norm", "std", "var"):
        _ro = getattr(Var, _rn, None)
        if _ro is not None:
            setattr(Var, _rn, _axis_to_dim(_ro))

    # ---- Tensor methods used by mmdetection + cheap torch-standard completeness ----
    # (.relu 86x, .eq 11x, .gt 12x, .diff, .fliplr are exercised by mmdet; the rest
    #  are one-line torch standards added to reduce downstream surprises.)
    if not hasattr(Var, "relu"):        Var.relu = lambda self: nn.relu(self)
    if not hasattr(Var, "relu_"):       Var.relu_ = lambda self: nn.relu(self)
    if not hasattr(Var, "eq"):          Var.eq = lambda self, other: self == other
    if not hasattr(Var, "ne"):          Var.ne = lambda self, other: self != other
    if not hasattr(Var, "gt"):          Var.gt = lambda self, other: self > other
    if not hasattr(Var, "ge"):          Var.ge = lambda self, other: self >= other
    if not hasattr(Var, "lt"):          Var.lt = lambda self, other: self < other
    if not hasattr(Var, "le"):          Var.le = lambda self, other: self <= other
    if not hasattr(Var, "neg"):         Var.neg = lambda self: -self
    if not hasattr(Var, "reciprocal"):  Var.reciprocal = lambda self: 1.0 / self
    if not hasattr(Var, "expm1"):       Var.expm1 = lambda self: jt.exp(self) - 1
    if not hasattr(Var, "log1p"):       Var.log1p = lambda self: jt.log(self + 1)
    if not hasattr(Var, "square"):      Var.square = lambda self: self * self
    if not hasattr(Var, "square_"):     Var.square_ = lambda self: self.assign(self * self)
    if not hasattr(Var, "clamp_min"):   Var.clamp_min = lambda self, v: jt.maximum(self, v)
    if not hasattr(Var, "clamp_max"):   Var.clamp_max = lambda self, v: jt.minimum(self, v)
    _orig_index_add_inplace = getattr(Var, "index_add_", None)
    if _orig_index_add_inplace is not None and not getattr(_orig_index_add_inplace, "_torch_returns_self", False):
        def _index_add_inplace(self, dim, index, source, *, alpha=1):
            if alpha != 1:
                source = source * alpha
            _orig_index_add_inplace(self, dim, index, source)
            return self
        _index_add_inplace._torch_returns_self = True
        Var.index_add_ = _index_add_inplace
    if not hasattr(Var, "bmm"):         Var.bmm = lambda self, other: jt.matmul(self, other)
    if not hasattr(Var, "mm"):          Var.mm = lambda self, other: jt.matmul(self, other)
    if not hasattr(Var, "mv"):          Var.mv = lambda self, vec: g.mv(self, vec)
    if not hasattr(Var, "fliplr"):      Var.fliplr = lambda self: jt.flip(self, 1)
    if not hasattr(Var, "flipud"):      Var.flipud = lambda self: jt.flip(self, 0)
    if not hasattr(Var, "diff"):
        Var.diff = lambda self, n=1, dim=-1, prepend=None, append=None: _diff(self, n, dim, prepend, append)
    if not hasattr(Var, "trapz"):
        Var.trapz = lambda self, x=None, dx=1, dim=-1: _trapz(self, x=x, dx=dx, dim=dim)
    if not hasattr(Var, "trapezoid"):
        Var.trapezoid = lambda self, x=None, dx=1, dim=-1: _trapz(self, x=x, dx=dx, dim=dim)
    if not hasattr(Var, "fmod"):        # truncated remainder, sign of dividend
        Var.fmod = lambda self, other: self - jt.trunc(self / other) * other
    if not hasattr(Var, "remainder"):   # floored remainder, sign of divisor
        Var.remainder = lambda self, other: self - jt.floor(self / other) * other
    if not hasattr(Var, "softplus"):    Var.softplus = lambda self, beta=1, threshold=20: nn.softplus(self)

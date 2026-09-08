"""Stable Tensor gradient APIs using the shared native Var/Op graph."""
from importlib import import_module
from ...context import get_install_context
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
_owner = import_module(__package__)
_NativeVar = _owner.jt.Var

def _register_leaf(v):
    _owner._torch_register_leaf(v)


def _rg_get(self):
    _context = get_install_context(_owner.jt)
    Var = _context.state["Var"]
    _native = _context.state["tensor_native_api"]
    _native_requires_grad = _native['_native_requires_grad']
    return bool(_native_requires_grad.__get__(self, Var))


def _rg_set(self, v):
    # The native descriptor owns the reversible-vs-permanent distinction:
    # requires_grad_(False) preserves old edges, while stop_grad() does not.
    _context = get_install_context(_owner.jt)
    _native = _context.state["tensor_native_api"]
    _native_requires_grad = _native['_native_requires_grad']
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


def requires_grad_(self, v=True):
    self.requires_grad = v
    if v:
        _register_leaf(self)
    return self


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
            if isinstance(v, _NativeVar) and v.requires_grad and v.is_backward_leaf:
                leaf_map.setdefault(id(v), v)
    else:
        _owner._torch_prune_leaf_registry()
        for v in list(tensor_state.leaf_params.values()):
            if isinstance(v, _NativeVar) and v.requires_grad and v.is_backward_leaf:
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
            if (id(p) not in opt_ids and id(p) not in retained_ids
                    and not tensor_state.leaf_params.is_weak(id(p))):
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
    # Independent retain_grad lasts as long as its weakly indexed holder.
    # Preserve bounded cleanup only for legacy non-weak-referenceable Vars.
    if retained:
        retained.clear_legacy()
    return None


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


def _retain_grad(self):
    if not self.requires_grad:
        raise RuntimeError("cannot retain_grad on a Tensor with requires_grad=False")
    if self.is_backward_leaf:
        return None
    self._torch_retains_grad = True
    _owner.get_tensor_state(_owner.jt).retained[id(self)] = self
    return None

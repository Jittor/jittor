"""Stable Tensor gradient APIs using the shared native Var/Op graph."""
from contextlib import contextmanager
from importlib import import_module
import os
from ...context import get_install_context
from jittor._core.var import placement_scope_like as _placement_scope_like
from jittor._runtime import step_capture as _step_capture
_owner = import_module(__package__)
# Both native Var and the independent TensorBase own the same native graph.
# Admission must test that real initialized carrier, not legacy inheritance.
_is_tensor_carrier = _owner.jt.core._is_tensor_carrier


@contextmanager
def _fsdp_grad_placement_scope(reference):
    """Allocate FSDP autograd results on the full parameter's accelerator.

    Native Jittor tensors created under the ambient accelerator can carry a
    device_id without an explicit placement_backend.  placement_scope_like()
    deliberately does nothing for those tensors, so cover the ACL spelling
    explicitly while retaining its normal path for published placements.
    """
    if reference is None:
        yield
        return
    backend = int(getattr(reference, "placement_backend", -1))
    device_id = int(getattr(reference, "device_id", -1))
    if os.environ.get("JITTOR_VERL_FATAL_TRACE") == "1":
        print("JITTOR_VERL_FSDP_GRAD_SCOPE backend=%d device=%d use_acl=%d" % (
            backend, device_id, int(bool(getattr(_owner.jt.flags, "use_acl", 0)))),
            flush=True)
    if backend > 0:
        with _placement_scope_like(reference, override=True):
            yield
        return
    if bool(getattr(_owner.jt.flags, "use_acl", 0)) and device_id >= 0:
        token = _owner.jt.core._set_tensor_placement(2, device_id)
        try:
            yield
        finally:
            _owner.jt.core._reset_tensor_placement(token)
        return
    yield

def _register_leaf(v):
    _owner._torch_register_leaf(v)


def _owned_leaf_gradient(parameter, gradient):
    wrap = getattr(type(parameter), "_frontend_gradient", None)
    return wrap(parameter, gradient) if wrap is not None else gradient


def _initial_leaf_gradient(parameter, gradient):
    """Apply AccumulateGrad's dense layout contract without changing AD edges."""
    sizes, strides = tuple(parameter.shape), parameter.stride()
    extent = 1
    dense = True
    for axis in sorted(range(len(sizes)), key=lambda i: strides[i]):
        if sizes[axis] > 1:
            dense = dense and strides[axis] == extent
            extent *= sizes[axis]
    if not dense:
        return _owned_leaf_gradient(parameter, gradient.contiguous())
    grad_sizes, grad_strides = tuple(gradient.shape), gradient.stride()
    # .data may have reduced the visible rank after this gradient's contract
    # was recorded. Check only existing stride axes, never index beyond the
    # current layout; the native graph separately validates recorded shape.
    obeys = all(
        (size == 1 and grad_strides[i] != 0)
        or (size != 1 and (i >= len(strides) or grad_strides[i] == strides[i]))
        for i, size in enumerate(grad_sizes)
    )
    if obeys:
        return _owned_leaf_gradient(parameter, gradient)
    # Existing as_strided implements dense permutations as real native views.
    # Avoid empty_strided's documented contiguous-only allocation restriction.
    target = get_install_context(_owner.jt).target_namespace
    result = target.empty(sizes, dtype=gradient.dtype, device=parameter.device)
    with _owner.jt.flag_scope(transpose_storage_view=1):
        if not parameter.is_contiguous():
            result = result.reshape(-1).as_strided(sizes, strides)
        result.copy_(gradient)
    return _owned_leaf_gradient(parameter, result)


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
            if not _is_tensor_carrier(peer) or peer is self:
                continue
            _native_requires_grad.__set__(peer, v)
            if v:
                _register_leaf(peer)
        if getattr(fsdp_state, "true_fsdp_flat", False):
            flat = getattr(fsdp_state, "true_fsdp_flat_shard", None)
            any_trainable = any(getattr(entry, "requires_grad", True)
                                for entry in fsdp_state.true_fsdp_params)
            if _is_tensor_carrier(flat):
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


def _register_post_accumulate_grad_hook(self, hook):
    if not self.requires_grad:
        raise RuntimeError("cannot register a hook on a tensor that does not require gradient")
    if not self.is_backward_leaf:
        raise RuntimeError("post accumulate grad hooks cannot be registered on non-leaf tensors")
    if not callable(hook):
        raise TypeError("post accumulate grad hook must be callable")
    from jittor._core.hooks import _RemovableHandle
    hooks = self.__dict__.setdefault("_torch_post_accumulate_hooks", {})
    key = self.__dict__.get("_torch_post_accumulate_next_id", 0)
    self.__dict__["_torch_post_accumulate_next_id"] = key + 1
    hooks[key] = hook
    _register_leaf(self)
    return _RemovableHandle(lambda: hooks.pop(key, None))


def _run_post_accumulate_hooks(leaves, grad_by_id):
    # Differentiate once with the native engine, then notify observers of the
    # accumulated storage. Optimizer slots are already published, so a hook's
    # p.grad mutation/clearing writes through the existing gradient owner.
    from ...gradient_task import _participation_scope
    with _owner.jt.no_grad(), _participation_scope(leaves, grad_by_id):
        for parameter in leaves:
            if id(parameter) not in grad_by_id:
                continue
            hooks = parameter.__dict__.get("_torch_post_accumulate_hooks")
            if not hooks:
                continue
            for key, hook in tuple(hooks.items()):
                if key in hooks and hook(parameter) is not None:
                    raise RuntimeError("Tensor post accumulate grad hook must return None")


def _fill_opt_grads(opt, grad_by_id, filled_param_ids=None, create_graph=False):
    # Replicate the grad-storage half of jittor's Optimizer.backward() but
    # from an already-computed {id(param): grad} map (so a SINGLE jt.grad
    # pass feeds every optimizer + every leaf — no N-times-repeated backward).
    # Honors the per-optimizer __zero_grad flag (post_step zeros it, so the
    # next backward overwrites rather than accumulates) and tolerates a param
    # whose shape changed (3DGS densify replaces params) by replacing — not
    # .update()-ing — the stored grad Var.
    from jittor.optim.base import optimizer_execution_groups
    zero = getattr(opt, "_Optimizer__zero_grad", True)
    if filled_param_ids is None:
        filled_param_ids = set()
    for pg in optimizer_execution_groups(opt):
        grads_list = pg.get("grads")
        if grads_list is None:
            grads_list = pg["grads"] = [None] * len(pg["params"])
        for i, p in enumerate(pg["params"]):
            if not _is_tensor_carrier(p) or not p.requires_grad:
                continue
            g = grad_by_id.get(id(p))
            if g is None:
                continue
            if id(p) in filled_param_ids:
                while len(grads_list) <= i:
                    grads_list.append(None)
                grads_list[i] = getattr(p, "_torch_grad", None)
                continue
            if not create_graph:
                # _backward already supplied a fresh stopped publication copy.
                g = g.stop_grad()
            existing = grads_list[i] if i < len(grads_list) else None
            if not _is_tensor_carrier(existing):
                existing = getattr(p, "_torch_grad", None)
            if _is_tensor_carrier(existing) and list(existing.shape) == list(g.shape):
                if not zero:
                    g = g + existing
                if create_graph:
                    # A differentiable accumulation must preserve the old
                    # gradient object and both contributions' graph edges.
                    stored = g
                else:
                    # The public gradient and optimizer slot share this
                    # storage; accumulation must not rebind it away from
                    # detached aliases or replace an explicitly set layout.
                    from .method_api import _copy_
                    with _owner.jt.no_grad():
                        _copy_(existing, g)
                    stored = existing
            else:
                stored = g if create_graph else _initial_leaf_gradient(p, g)
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
    from ...autograd import _backward_callbacks
    from contextlib import nullcontext
    provider = _owner._fsdp_hooks.provider()
    scope = (provider._resident_backward_task_scope(
        kw.get("inputs"), create_graph, retain_graph, set(kw) - {"inputs"})
        if provider is not None else nullcontext(None))
    with scope as resident_task, _backward_callbacks():
        result = _backward_impl(self, gradient, retain_graph, create_graph, **kw)
        if resident_task is not None:
            resident_task.queue_finish()
        return result


def _backward_impl(self, gradient=None, retain_graph=None, create_graph=False, **kw):
    # torch's signature is (gradient=None, retain_graph=None,
    # create_graph=False, inputs=None) and retain_graph defaults to
    # create_graph. The default here was False, not None, so the line below
    # could never see None: `loss.backward(create_graph=True)` freed the
    # graph anyway and the second-order backward it was asked for then
    # failed. In the common loss.backward() case both are false, so the
    # graph is still freed.
    if self.requires_grad and not self._has_gradient_edge():
        raise RuntimeError("Tensor does not require grad and does not have a grad_fn")
    retain_graph = bool(create_graph) if retain_graph is None else bool(retain_graph)
    # torch's `gradient` is the vector of the vector-Jacobian product:
    # y.backward(v) computes d(sum(y*v))/dx. It used to be accepted and
    # dropped, so every weighted backward -- per-sample loss weights, a
    # manual chain rule from a custom head -- silently computed the
    # UNWEIGHTED gradient d(sum(y))/dx and trained on the wrong numbers.
    if gradient is not None:
        grad_var = gradient if _is_tensor_carrier(gradient) else _owner.jt.array(gradient)
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
    #
    # Not while a step is being captured: a capture builds the step whole and
    # runs it once, and a forward settled in the middle of it had its buffers
    # freed after their last use and was computed again by that run -- a
    # Qwen3 training step's forward ran twice to capture it. Nor can a
    # capture replay an extension that writes out of band anyway.
    if not _step_capture.active():
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
    provider = _owner._fsdp_hooks.provider()
    resident_task = provider._resident_backward_task_current() if provider is not None else None
    if resident_task is not None:
        resident_task.validate_optimizers(opts)
    fsdp_opts = ([] if resident_task is not None else
                 [o for o in opts if _optimizer_maybe_has_fsdp_params(o)])
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
    fsdp_grad_reference = None
    for o in opts:
        for pg in getattr(o, "param_groups", []):
            for p in pg.get("params", []):
                if not _is_tensor_carrier(p) or not p.requires_grad:
                    continue
                if resident_task is not None and id(p) in resident_task.canonical_ids:
                    opt_ids.add(id(p))
                    continue
                if _fsdp2_backward is not None and _fsdp2_backward.is_fsdp_managed_param(p):
                    opt_ids.add(id(p))
                    continue
                leaf_map.setdefault(id(p), p)
                opt_ids.add(id(p))
    if _fsdp2_backward is not None and fsdp_opts:
        for p in _fsdp2_backward.collect_fsdp_full_params_for_backward(fsdp_opts):
            if _is_tensor_carrier(p) and p.requires_grad:
                leaf_map.setdefault(id(p), p)
                opt_ids.add(id(p))
                # FSDP all-gather publishes the accelerator placement on the
                # full parameter. Keep one such parameter as the allocation
                # owner for the gradients created by grad_optional below.
                # Without this scope the torch frontend's CPU factory default
                # marks a lazy BF16 full gradient as explicitly CPU; the later
                # HCCL reduce-scatter then inherits CPU and has no valid kernel.
                if (fsdp_grad_reference is None
                        or (int(getattr(fsdp_grad_reference,
                                        "placement_backend", -1)) < 0
                            and int(getattr(p, "placement_backend", -1)) > 0)):
                    fsdp_grad_reference = p
    if resident_task is not None:
        opt_ids.update(resident_task.owned_ids)
        for p in resident_task.targets:
            leaf_map.setdefault(id(p), p)
        if fsdp_grad_reference is None and resident_task.targets:
            fsdp_grad_reference = resident_task.targets[0]
    tensor_state = _owner.get_tensor_state(_owner.jt)
    retained = tensor_state.retained
    retained_ids = set()
    if retained:
        for v in list(retained.values()):
            if _is_tensor_carrier(v) and v.requires_grad:
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
            if _is_tensor_carrier(v) and v.requires_grad and v.is_backward_leaf:
                leaf_map.setdefault(id(v), v)
    else:
        _owner._torch_prune_leaf_registry()
        for v in list(tensor_state.leaf_params.values()):
            if _is_tensor_carrier(v) and v.requires_grad and v.is_backward_leaf:
                leaf_map.setdefault(id(v), v)
    if resident_task is not None:
        # C is only a publication target, never a differentiation target.
        for key in resident_task.canonical_ids:
            leaf_map.pop(key, None)
    if not leaf_map:
        return None
    leaves = list(leaf_map.values())
    # torch leaves a disconnected target at grad=None. Keep jt.grad's
    # historical zero-materialization untouched and use the compatibility
    # core entry point that preserves missing gradients explicitly.
    with _fsdp_grad_placement_scope(fsdp_grad_reference):
        # Leave VJP scalarization and public gradient publication outside this scope.
        with (_owner.jt.enable_grad() if create_graph else _owner.jt.no_grad()):
            grads = _owner.jt.core.grad_optional(self, leaves, retain_graph)
    grad_by_id = {}
    for p, gr in zip(leaves, grads):
        if gr is None:
            if (id(p) not in opt_ids and id(p) not in retained_ids
                    and not tensor_state.leaf_params.is_weak(id(p))):
                tensor_state.leaf_params.pop(id(p), None)
            continue
        # torch's AccumulateGrad gives a leaf a grad of the leaf's own dtype.
        # jt.grad does not: under autocast a float32 weight read by a
        # float16 op comes back with a float16 grad, and AdamW's
        # (1 - beta2) * g * g then underflows in float16 -- an lr / eps step,
        # NaN on the next forward (test_torch_amp_training_loop.py).
        if gr.dtype != p.dtype:
            gr = gr.cast(p.dtype)
        if not create_graph:
            # Match autograd.grad(False): stop only a fresh returned copy,
            # never the raw gradient or a source graph retained by the caller.
            gr = gr.detach()
            gr.stop_grad()
        grad_by_id[id(p)] = gr
        if id(p) not in opt_ids:
            # non-optimizer leaf (retain_grad screenspace etc.): accumulate
            # onto .grad like torch (zeroed externally / per render).
            #
            # In place, the way torch's AccumulateGrad does it: when .grad
            # already exists, torch adds into that very tensor. Rebuilding it
            # with ``prev + gr`` is numerically identical but hands back a new
            # object, and a caller that already holds the previous ``.grad``
            # (a training loop keeping ``[p.grad for p in model.parameters()]``,
            # for instance) then keeps a full gradient set alive per step. That
            # is a memory regression against torch, not a rounding detail.
            prev = getattr(p, "_torch_grad", None)
            if prev is None:
                object.__setattr__(p, "_torch_grad", _initial_leaf_gradient(p, gr))
            elif create_graph:
                object.__setattr__(p, "_torch_grad", prev + gr)
            else:
                prev.add_(gr)
    if resident_task is not None:
        resident_task.publish(grad_by_id, _initial_leaf_gradient, _run_post_accumulate_hooks)
    # fill each optimizer's pg["grads"] so its step(loss=None) consumes them
    if _fsdp2_backward is not None and fsdp_opts:
        _fsdp2_backward.fill_fsdp_optimizer_grads_from_grad_map(fsdp_opts, grad_by_id)
    for o in opts:
        if _fsdp2_backward is not None and id(o) in fsdp_opt_ids \
                and not _fsdp2_backward.optimizer_has_non_fsdp_params(o):
            continue
        _fill_opt_grads(o, grad_by_id, filled_param_ids, create_graph=create_graph)
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
    ordinary_leaves = ([p for p in leaves if id(p) not in resident_task.owned_ids]
                       if resident_task is not None else leaves)
    _run_post_accumulate_hooks(ordinary_leaves, grad_by_id)
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
        slots = _param_slots(o, self)
        if not any(i < len(pg.get("grads", ())) and pg["grads"][i] is not None
                   for pg, i in slots if pg.get("grads") is not None):
            continue
        try:
            return o.find_grad(self)
        except _owner.EXPECTED as exc:
            _owner.swallowed("torch/installers/tensor.py _grad_get: return o.find_grad(self)", exc)
    return None


def _param_slots(opt, param):
    """Every (group, index) at which ``opt`` holds ``param``.

    Answered from an index kept on the optimizer. Scanning every parameter of
    every group on each assignment made ``zero_grad(set_to_none=True)``, which
    assigns ``p.grad = None`` for each parameter, quadratic in the parameter
    count: 21 ms of a 90 ms DDPM training step at 450 parameters. A hit is
    checked against the group it names before it is used and a miss rescans,
    so an index made stale by a parameter replaced in place (3DGS
    densification does ``group["params"][0] = new``) or a group added later
    is rebuilt, never trusted.
    """
    from jittor.optim.base import optimizer_execution_groups
    index = opt.__dict__.get("_torch_param_slots")
    if index is not None:
        slots = index.get(id(param))
        if slots is not None:
            for pg, i in slots:
                params = pg.get("params", ())
                if i >= len(params) or params[i] is not param:
                    break
            else:
                return slots
    index = {}
    for pg in optimizer_execution_groups(opt):
        for i, p in enumerate(pg.get("params", [])):
            index.setdefault(id(p), []).append((pg, i))
    object.__setattr__(opt, "_torch_param_slots", index)
    return index.get(id(param), ())


def _grad_set(self, value):
    from jittor.optim.base import optimizer_execution_groups
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
        if fsdp_entry is None:
            slots = _param_slots(o, self)
        else:
            # An FSDP shard answers for its full parameter too, which only a
            # scan comparing entries finds.
            slots = []
            for pg in optimizer_execution_groups(o):
                for i, p in enumerate(pg.get("params", [])):
                    same_fsdp_entry = getattr(p, "_jittor_fsdp2_entry", None) is fsdp_entry
                    if p is not self and not same_fsdp_entry:
                        continue
                    if fsdp_role == "full" and value is not None and p is not self:
                        continue
                    slots.append((pg, i))
        for pg, i in slots:
            count = len(pg.get("params", []))
            if value is None:
                grads = pg.get("grads")
                if grads is not None and i < len(grads):
                    grads[i] = None
            else:
                grads = pg.get("grads")
                if grads is None:
                    grads = pg["grads"] = [None] * count
                while len(grads) < count:
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
    __slots__ = ("node_id", "op_id", "name", "__weakref__")

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
    try:
        node_id = self.grad_fn_node_id
    except _owner.jt.JittorError as error:
        # Only the generic native metadata-query error becomes Torch's public
        # RuntimeError. Typed/internal errors and later graph/intern failures
        # remain their original exceptions; the original args/cause survive.
        if type(error) is not _owner.jt.JittorError:
            raise
        raise RuntimeError(*error.args) from error
    node_id = int(node_id)
    if node_id == -1:
        return None
    # Native metadata alone decides whether a producer exists. Interning must
    # never turn a native None into a cached historical/executable producer.
    state = _owner.get_tensor_state(_owner.jt)
    return state.intern_grad_fn(node_id, self.grad_fn_op_id,
                               self.grad_fn_name, _TorchGradFn)


def _retain_grad(self):
    if not self.requires_grad:
        raise RuntimeError("cannot retain_grad on a Tensor with requires_grad=False")
    if self.is_backward_leaf:
        return None
    self._torch_retains_grad = True
    _owner.get_tensor_state(_owner.jt).retained[id(self)] = self
    return None

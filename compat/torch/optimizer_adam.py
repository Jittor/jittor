"""Torch-semantics Adam/AdamW step for the stable optimizer adapters.

Split out of ``optimizer_api`` by size. Helpers that stay there are reached
through a call-time import of that module: ``optimizer_api`` imports this one,
and an import at module level here would put both on an import-time cycle.
"""
import jittor as jt
from jittor.optim.algorithms.adam import adam_update
from jittor.optim.base import _update_preserve_dtype
from .context import get_install_context
from .installers.tensor.method_api import _optimizer_write_scope, _optimizer_update
from ..diagnostics import EXPECTED, swallowed


class _AdamStep:
    """Keep native bookkeeping until a caller asks for the tensor state.

    An exposed tensor is authoritative, including writes through aliases or
    NumPy, which need not advance an autograd version. Do not cache its value.
    Unobserved counters keep ordinary optimizer capture free of host reads.
    """
    def __init__(self, value, *, device=None, dtype=None):
        self.counter = value if isinstance(value, jt.Var) else None
        self.value = value if self.counter is None else None
        self.device = device
        self.dtype = dtype

    def tensor(self):
        if self.counter is None:
            g = get_install_context(jt).target_namespace
            self.counter = g.tensor(float(self.value), dtype=self.dtype, device=self.device)
        return self.counter

    def __int__(self):
        return int(self.value if self.counter is None else self.counter.item())

    def __float__(self):
        return float(self.value if self.counter is None else self.counter.item())

    def __eq__(self, other):
        return int(self) == int(other)

    def set(self, value):
        self.value = value
        if self.counter is not None:
            with jt.no_grad():
                self.counter.fill_(value)


class _AdamSteps(list):
    """Native step bookkeeping backed by Torch's mutable scalar state.

    Uninitialized slots stay integer zero. Once initialized, numeric writes
    update the existing tensor so references from ``state``/``state_dict``
    remain live; explicit tensor assignments replace it, just like a dict.
    Native update and replay code still read the entries with int()/float().
    """
    def __init__(self, group, values):
        super().__init__(values)
        self.configure(group)

    def configure(self, group):
        # Do not retain the owning dict: it owns this list, and a cycle would
        # keep discarded optimizers' parameters and lazy graphs alive.
        self.fused = group.get("fused", False)
        # Only fused counters live on the parameter device. Read current
        # placements when that policy is selected, including after a move.
        self.devices = (tuple(parameter.device for parameter in group["params"])
                        if self.fused else ())

    def __setitem__(self, index, value):
        old = self[index]
        if isinstance(value, jt.Var):
            return super().__setitem__(index, _AdamStep(value))
        if isinstance(old, _AdamStep):
            old.set(value)
            return
        if value != 0:
            g = get_install_context(jt).target_namespace
            fused = self.fused
            device = self.devices[index] if fused else "cpu"
            dtype = g.float64 if not fused and g.get_default_dtype() == g.float64 else g.float32
            value = _AdamStep(value, dtype=dtype, device=device)
        super().__setitem__(index, value)


def _update_in_target_dtype(target, value):
    _update_preserve_dtype(target, value)


def _try_batch_update(active, updates):
    """Batch canonical writes, preserving custom writers and capture paths."""
    from jittor._runtime import step_capture
    if step_capture.active() or step_capture.tracing():
        return False
    # Providers may return iterators. Leave those intact for the ordinary
    # writer instead of consuming their updates during a speculative check.
    if type(updates) not in (list, tuple) or len(updates) != len(active):
        return False
    if any(type(triple) not in (list, tuple) or len(triple) != 3 for triple in updates):
        return False
    frontend = get_install_context(jt).target_namespace
    allowed = (jt.Var, frontend.Tensor, frontend.nn.Parameter)
    targets, values = [], []
    for (p, m, v, _, _), triple in zip(active, updates):
        for target, value in zip((p, m, v), triple):
            if (type(target) not in allowed or type(value) not in allowed
                    or getattr(type(target), '_optimizer_update', None) is not _optimizer_update):
                return False
            targets.append(target)
            values.append(value)
    with jt.no_grad():
        return jt.core._try_optimizer_copies(targets, values)


def _adam_step(self, loss, retain_graph, closure, kwargs, decoupled_weight_decay):
    from . import optimizer_api as _api
    for group in self.param_groups:
        _api._validate_adam_execution_options(group)
        _api._sync_adam_group_params(group, validate_shapes=True)
        if group.get('maximize') and _api._optimizer_maybe_has_fsdp_params(self):
            raise NotImplementedError('FSDP Adam/AdamW maximize=True is not supported')
        group.setdefault('fused', getattr(self, 'fused', None))
    native_fsdp_loss = None
    if closure is None and callable(loss):
        closure = loss
        loss = None
    if closure is not None:
        loss = closure()
    _fsdp2_step = _api._load_fsdp2_for_optimizer(self)
    if _fsdp2_step is not None and _fsdp2_step.optimizer_has_fsdp_params(self):
        if loss is not None and closure is None:
            native_fsdp_loss = loss
            loss.backward(retain_graph=retain_graph)
            loss = None
        if not _fsdp2_step.optimizer_step(
                self, None, retain_graph=retain_graph,
                native_kind="adamw" if decoupled_weight_decay else "adam"):
            raise NotImplementedError(
                f"FSDP2 optimizer step is not implemented for {type(self).__name__}")
        if not _fsdp2_step.optimizer_has_non_fsdp_params(self):
            if native_fsdp_loss is not None:
                try:
                    self.post_step()
                except EXPECTED as exc:
                    swallowed("torch/optimizers.py _adam_step_torch: self.post_step()", exc)
            else:
                jt.flags.node_order = 0
                object.__setattr__(
                    self, "_torch_backward_advanced_n_step", False)
            return native_fsdp_loss if native_fsdp_loss is not None else loss
        _fsdp2_step.clear_fsdp_optimizer_grads(self)
    self.pre_step(None if closure is not None and _api._optimizer_has_ready_grads(self) else loss, retain_graph)
    if loss is None or closure is not None:
        # ZeRO-3 sets a flat parameter's grad before temporarily rebinding its
        # optimizer group. The tensor owns the current grad; an old group slot
        # can be empty or refer to the preceding subgroup/step.
        for pg in self.param_groups:
            _api._validate_adam_execution_options(pg)
            _api._sync_adam_group_params(pg, validate_shapes=True)
            pg["grads"] = [
                None if (_fsdp2_step is not None
                         and _fsdp2_step.is_fsdp_managed_param(param))
                else getattr(param, "_torch_grad", None)
                for param in pg.get("params", ())
            ]
    if not _api._optimizer_has_ready_grads(self):
        if native_fsdp_loss is not None:
            self.post_step()
        else:
            jt.flags.node_order = 0
            object.__setattr__(
                self, "_torch_backward_advanced_n_step", False)
        return native_fsdp_loss if native_fsdp_loss is not None else loss
    from jittor._runtime import step_capture
    if not getattr(self, "_torch_backward_advanced_n_step", False):
        self.n_step = int(getattr(self, "n_step", 0)) + 1
        step_capture.on_replay(lambda: setattr(self, "n_step", int(self.n_step) + 1))
    jt.flags.node_order = 1
    for pg in self.param_groups:
        lr = pg.get("lr", self.lr)
        eps = pg.get("eps", self.eps)
        weight_decay = pg.get("weight_decay", self.weight_decay)
        b0, b1 = pg.get("betas", self.betas)
        param_steps = _api._torch_param_steps(pg)
        # torch permits optimizers containing frozen or otherwise
        # unused parameters. loss.backward() then leaves the group
        # without gradients and step() must be a no-op, not KeyError.
        grads = pg.get("grads") or [None] * len(pg["params"])
        # `use_acl` is an alias of `use_cuda` (see FLAG_ALIASES), so it is true
        # on a CUDA build as well and cannot say which backend is actually in
        # use. Asking the dispatcher does: `optim.adamw_fused` is registered
        # for "acl" only, so it answers None everywhere else. Keying off the
        # alias sent CUDA into the Ascend-only path, where the native
        # `fused_adamw` operator aborts with "only available through a mapped
        # backend" -- reachable from plain `torch.optim.AdamW(..., fused=True)`.
        fused_impl = None
        # torch's AdamW is a multi-tensor kernel unless told otherwise: its
        # default (fused=None, foreach=None) takes the foreach path on CUDA.
        # So the fused list update is taken unless the caller turned both
        # off, or asked for a variant it does not compute.
        fused = pg.get("fused", getattr(self, "fused", None))
        want_fused = (decoupled_weight_decay and fused is not False
                      and (fused is True or pg.get("foreach") is not False)
                      and not pg.get("amsgrad") and not pg.get("maximize"))
        active = []
        if want_fused:
            for i, (p, g, v, m) in enumerate(zip(
                    pg["params"], grads, pg["values"], pg["m"])):
                if not p.requires_grad or not isinstance(g, jt.Var) \
                        or list(g.shape) != list(p.shape):
                    continue
                active.append((p, m, v, g, int(param_steps[i])))
            if active:
                from jittor._runtime.dispatch import select_kernel
                fused_impl = select_kernel("optim.adamw_fused", active)
        if fused_impl is not None:
            stepped = []
            for i, (p, g, v, m) in enumerate(zip(
                    pg["params"], grads, pg["values"], pg["m"])):
                if not p.requires_grad or not isinstance(g, jt.Var) \
                        or list(g.shape) != list(p.shape):
                    continue
                param_steps[i] = int(param_steps[i]) + 1
                stepped.append(i)
            lr_arg = lr
            if step_capture.active():   # a replay advances these without step()
                lr_arg = step_capture.live_fused_update(
                    fused_impl, lr, param_steps, stepped, lambda g=pg: g.get("lr", self.lr),
                    lambda g=pg: (tuple(g.get("betas", self.betas)),
                                  g.get("weight_decay", self.weight_decay), g.get("eps", self.eps)),
                    self.__dict__.get("_amp_found_inf"), self)
            updates = fused_impl(
                active, lr_arg, b0, b1, weight_decay, eps)
            with _optimizer_write_scope():
                copied = _try_batch_update(active, updates)
                for (p, m, v, _, _), (new_p, new_m, new_v) in zip(
                        active, updates):  # fused_adamw keeps each input's dtype
                    if not copied:
                        _update_preserve_dtype(p, new_p)
                        _update_preserve_dtype(m, new_m)
                        _update_preserve_dtype(v, new_v)
                    if p.is_stop_grad():
                        p.start_grad()
            continue
        step_capture.refuse("the per-parameter Adam update bakes its step count in")
        with _optimizer_write_scope():
            for i, (p, g, v, m) in enumerate(zip(
                    pg["params"], grads, pg["values"], pg["m"])):
                was_trainable = bool(p.requires_grad)
                if not was_trainable or not isinstance(g, jt.Var) or list(g.shape) != list(p.shape):
                    continue
                param_steps[i] = int(param_steps[i]) + 1
                _update_in_target_dtype(p, adam_update(
                    p, -g if pg.get("maximize", False) else g, v, m,
                    lr=lr, eps=eps, weight_decay=weight_decay,
                    betas=(b0, b1), step=param_steps[i],
                    decoupled_weight_decay=decoupled_weight_decay,
                    torch_math=True,
                ))
                try:
                    if was_trainable and p.is_stop_grad():
                        p.start_grad()
                except EXPECTED as exc:
                    swallowed("torch/optimizers.py _adam_step_torch: if was_trainable and p.is_stop_grad():", exc)
    # The torch loss.backward(); optimizer.step() spelling keeps grads
    # until an explicit zero_grad(). Preserve Jittor's historical
    # step(loss) behavior for callers using that native shorthand.
    if (loss is not None and closure is None) or native_fsdp_loss is not None:
        self.post_step()
    else:
        jt.flags.node_order = 0
    object.__setattr__(self, "_torch_backward_advanced_n_step", False)
    return native_fsdp_loss if native_fsdp_loss is not None else loss


def _validate_adam_execution_options(options):
    for name in ("amsgrad", "differentiable", "capturable"):
        if options.get(name, False):
            raise NotImplementedError(
                "Adam/AdamW %s=True is not supported by this frontend" % name)


def _initialize_adam(self, params, lr, args, kwargs, name):
    from . import optimizer_api as _api
    # These are execution-policy options in Torch. Native Jittor's initializer
    # does not accept them, but downstream optimizers inspect the group values.
    foreach = kwargs.pop("foreach", None)
    options = {
        "amsgrad": kwargs.pop("amsgrad", False),
        "maximize": kwargs.pop("maximize", False),
        "capturable": kwargs.pop("capturable", False),
        "differentiable": kwargs.pop("differentiable", False),
    }
    _validate_adam_execution_options(options)
    if name == "Adam":
        fused = kwargs.pop("fused", None)
    else:
        fused = kwargs.get("fused", None)
    result = _api._initialize_default(self, params, lr, args, kwargs, name)
    options.update(
        foreach=foreach,
        fused=fused,
        decoupled_weight_decay=(name == "AdamW"),
    )
    for key, value in options.items():
        setattr(self, key, value)
    defaults = {
        "lr": self.lr,
        "betas": self.betas,
        "eps": self.eps,
        "weight_decay": self.weight_decay,
        **options,
    }
    for group in self.param_groups:
        for key, value in defaults.items():
            group.setdefault(key, value)
        _validate_adam_execution_options(group)
        group["_torch_adam_params"] = tuple(group["params"])
    return result

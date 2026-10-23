"""Torch-semantics Adam/AdamW step for the stable optimizer adapters.

Split out of ``optimizer_api`` by size. Helpers that stay there are reached
through a call-time import of that module: ``optimizer_api`` imports this one,
and an import at module level here would put both on an import-time cycle.
"""
from jittor._core.dtypes import var_dtype_name
import jittor as jt
from jittor.optim.algorithms.adam import adam_update
from ..diagnostics import EXPECTED, swallowed


def _update_in_target_dtype(target, value):
    if var_dtype_name(value) != var_dtype_name(target):  # native, not `.dtype`
        value = value.cast(var_dtype_name(target))
    target.update(value)


def _adam_step(self, loss, retain_graph, closure, kwargs, decoupled_weight_decay):
    from . import optimizer_api as _api
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
            _api._sync_adam_group_params(pg)
            pg["grads"] = [
                # FSDP2 already applied this shard above. Its public .grad stays
                # visible until zero_grad(), but must not feed the ordinary Adam
                # phase of a mixed FSDP/plain group a second time.
                None
                if _fsdp2_step is not None
                and _fsdp2_step.is_fsdp_managed_param(param)
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
            for (p, m, v, _, _), (new_p, new_m, new_v) in zip(
                    active, updates):  # fused_adamw keeps each input's dtype
                p.update(new_p)
                m.update(new_m)
                v.update(new_v)
                if p.is_stop_grad():
                    p.start_grad()
            continue
        step_capture.refuse("the per-parameter Adam update bakes its step count in")
        for i, (p, g, v, m) in enumerate(zip(
                pg["params"], grads, pg["values"], pg["m"])):
            was_trainable = bool(p.requires_grad)
            if not was_trainable or not isinstance(g, jt.Var) or list(g.shape) != list(p.shape):
                continue
            param_steps[i] = int(param_steps[i]) + 1
            _update_in_target_dtype(p, adam_update(
                p, g, v, m, lr=lr, eps=eps, weight_decay=weight_decay,
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

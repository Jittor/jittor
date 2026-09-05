"""FSDP2 optimizer updates and local sharded state helpers."""

import numpy as np

import jittor as jt

from . import common, grad_sync, shard
from .. import optimizer_kinds
from ..diagnostics import EXPECTED, swallowed
from ..torch.tensor_state import get_tensor_state


_EXPORTS = (
    "clear_fsdp_optimizer_grads",
    "_optimizer_param_steps",
    "_assign_preserve_trainability",
    "refresh_optimizer_fsdp_params",
    "_refresh_all_optimizer_fsdp_params",
    "_sgd_hparams",
    "_sgd_update_for_param",
    "_adam_hparams",
    "_adam_update_for_param",
    "_optimizer_kind",
    "optimizer_step",
    "sharded_sgd_step",
    "local_sharded_state_dict",
)


def clear_fsdp_optimizer_grads(opt):
    for pg in getattr(opt, "param_groups", []):
        grads = pg.get("grads")
        if grads is None:
            continue
        for i, param in enumerate(pg.get("params", [])):
            if i < len(grads) and shard.is_fsdp_managed_param(param):
                grads[i] = None


def _optimizer_param_steps(pg):
    params = list(pg.get("params", []))
    steps = pg.get("_torch_steps")
    if not isinstance(steps, list):
        steps = pg["_torch_steps"] = [0] * len(params)
    while len(steps) < len(params):
        steps.append(0)
    if len(steps) > len(params):
        del steps[len(params):]
    return steps


def _assign_preserve_trainability(target, value, was_trainable=None):
    if was_trainable is None:
        was_trainable = not target.is_stop_grad()
    target.assign(value)
    if was_trainable and target.is_stop_grad():
        target.start_grad()
    elif not was_trainable and not target.is_stop_grad():
        target.stop_grad()
    return target


def refresh_optimizer_fsdp_params(opt, state_ids=None):
    for pg in getattr(opt, "param_groups", []):
        params = pg.get("params", [])
        for i, param in enumerate(list(params)):
            state, entry = shard._fsdp_param_entry(param)
            if state is None or state_ids is not None and id(state) not in state_ids:
                continue
            params[i] = entry.shard


def _refresh_all_optimizer_fsdp_params(states, current=None):
    state_ids = {id(state) for state in states}
    seen = set()
    for ref in get_tensor_state(jt).active_optimizers:
        opt = ref() if callable(ref) else ref
        if opt is None or id(opt) in seen:
            continue
        seen.add(id(opt))
        refresh_optimizer_fsdp_params(opt, state_ids)
    if current is not None and id(current) not in seen:
        refresh_optimizer_fsdp_params(current, state_ids)


def _sgd_hparams(opt, pg):
    return (
        pg.get("lr", getattr(opt, "lr", 0.0)),
        pg.get("momentum", getattr(opt, "momentum", 0.0)),
        pg.get("weight_decay", getattr(opt, "weight_decay", 0.0)),
        pg.get("dampening", getattr(opt, "dampening", 0.0)),
        pg.get("nesterov", getattr(opt, "nesterov", False)),
    )


def _sgd_update_for_param(opt, pg, state, entry, param, grad, value):
    lr, momentum, weight_decay, dampening, nesterov = _sgd_hparams(opt, pg)
    dp = grad
    if weight_decay != 0:
        dp = dp + param * weight_decay
    if momentum != 0:
        if not isinstance(value, jt.Var) or list(value.shape) != list(dp.shape):
            value = jt.zeros(dp.shape, dp.dtype).stop_grad()
        value.update(momentum * value + dp * (1 - dampening))
        dp = dp + momentum * value if nesterov else value
    return (param - dp * lr).stop_grad(), value


def _adam_hparams(opt, pg):
    return (
        pg.get("lr", getattr(opt, "lr", 0.0)),
        pg.get("eps", getattr(opt, "eps", 1e-8)),
        pg.get("weight_decay", getattr(opt, "weight_decay", 0.0)),
        pg.get("betas", getattr(opt, "betas", (0.9, 0.999))),
    )


def _adam_update_for_param(opt, pg, param, grad, value, momentum, *,
                           decoupled_weight_decay, n_step):
    lr, eps, weight_decay, betas = _adam_hparams(opt, pg)
    b0, b1 = betas
    if not isinstance(value, jt.Var) or list(value.shape) != list(param.shape):
        value = jt.zeros(param.shape, param.dtype).stop_grad()
    if not isinstance(momentum, jt.Var) or list(momentum.shape) != list(param.shape):
        momentum = jt.zeros(param.shape, param.dtype).stop_grad()
    if weight_decay != 0 and decoupled_weight_decay:
        param = param * (1 - lr * weight_decay)
    elif weight_decay != 0:
        grad = grad + param * weight_decay
    momentum.update(b0 * momentum + (1 - b0) * grad)
    value.update(b1 * value + (1 - b1) * grad * grad)
    bias_correction1 = 1 - b0 ** float(n_step)
    bias_correction2 = 1 - b1 ** float(n_step)
    step_size = lr / bias_correction1
    denom = jt.sqrt(value) / np.sqrt(bias_correction2) + eps
    return (param - momentum * step_size / denom).stop_grad(), value, momentum


#: The update rules this module implements against a shard. Anything else has
#: to refuse: applying one of these three to an optimizer that implements a
#: different rule would train the model with arithmetic nobody asked for.
_SHARDED_KINDS = ("sgd", "adam", "adamw")


def _unsupported_optimizer_message(opt, kind):
    """Say which of the three reasons this optimizer cannot be stepped.

    The old message was `FSDP2 optimizer step is not implemented for <name>`
    for all of them, which does not distinguish "write one" from "your
    subclass was being ignored until now".
    """
    name = type(opt).__name__
    loose = optimizer_kinds.kind_of(opt)
    if loose is None:
        return (
            "FSDP2 cannot step %s: it does not inherit any update rule FSDP2 "
            "can apply to a shard (%s). Wrap it so the sharded parameters are "
            "stepped by an optimizer that does, or run without fully_shard."
            % (name, ", ".join(_SHARDED_KINDS)))
    if kind is None:
        return (
            "FSDP2 cannot step %s: it inherits from the %s optimizer but "
            "overrides step(), so it defines its own update rule. FSDP2 would "
            "have to apply the base %s update to the shards instead, which "
            "would silently train the model with arithmetic %s does not "
            "implement. (Until this check existed, that is exactly what "
            "happened.)" % (name, loose, loose, name))
    return (
        "FSDP2 cannot step %s: the %s update rule is not implemented against a "
        "shard, only %s are." % (name, kind, ", ".join(_SHARDED_KINDS)))


def _optimizer_kind(opt):
    """Which sharded update to run, or ``None`` when that is not knowable.

    This selects *arithmetic to apply to the user's weights*, so it asks for
    the strict answer: a subclass that replaces ``step()`` is not treated as
    its base class. It used to substring-match the class name, which meant a
    subclass named ``...Adam...`` with its own update rule silently got the
    base AdamW update instead of its own. See jittor/compat/optimizer_kinds.py.
    """
    return optimizer_kinds.kind_of(opt, require_unmodified_step=True)


def optimizer_step(opt, loss=None, retain_graph=False):
    """Apply one torch-style optimizer step for FSDP-managed parameters.

    Returns True when the optimizer contained FSDP parameters.  FSDP gradients are
    consumed from ``loss.backward()`` state when ``loss`` is None; passing a loss
    keeps Jittor's ``optimizer.step(loss)`` shortcut working for simple scripts.
    Non-FSDP parameters are intentionally left untouched so the caller can run the
    original optimizer step for them.
    """
    states = grad_sync._fsdp_states_from_optimizers([opt])
    if not states:
        return False
    if loss is not None:
        grad_by_id = {}
        targets = grad_sync.collect_fsdp_full_params_for_backward([opt])
        if targets:
            grads = jt.core.grad_optional(loss, targets, retain_graph)
            grad_by_id.update({id(p): g for p, g in zip(targets, grads)
                               if g is not None})
        grad_sync.fill_fsdp_optimizer_grads_from_grad_map([opt], grad_by_id)

    grad_sync._sync_visible_full_grads_to_optimizer(opt)

    kind = _optimizer_kind(opt)
    if kind not in _SHARDED_KINDS:
        raise NotImplementedError(_unsupported_optimizer_message(opt, kind))
    has_fsdp_grad = False
    for pg in getattr(opt, "param_groups", []):
        grads = pg.get("grads") or []
        for i, param in enumerate(pg.get("params", [])):
            if not shard.is_fsdp_managed_param(param):
                continue
            grad = grads[i] if i < len(grads) else None
            if not isinstance(grad, jt.Var):
                grad = getattr(param, "_torch_grad", None)
            if isinstance(grad, jt.Var):
                has_fsdp_grad = True
                break
        if has_fsdp_grad:
            break
    if has_fsdp_grad:
        if not getattr(opt, "_torch_backward_advanced_n_step", False):
            opt.n_step = int(getattr(opt, "n_step", 0)) + 1
        object.__setattr__(opt, "_torch_backward_advanced_n_step", True)

    jt.flags.node_order = 1
    flat_updates = {}
    flat_public_grads = {}
    entry_trainable = {
        (id(state), id(entry)): not entry.shard.is_stop_grad()
        for state in states for entry in state.true_fsdp_params
    }
    flat_trainable = {
        id(state): not state.true_fsdp_flat_shard.is_stop_grad()
        for state in states if getattr(state, "true_fsdp_flat", False)
    }
    for pg in getattr(opt, "param_groups", []):
        grads = pg.get("grads") or []
        param_steps = _optimizer_param_steps(pg)
        values = pg.get("values")
        if values is None:
            values = pg["values"] = [None] * len(pg.get("params", []))
        while len(values) < len(pg.get("params", [])):
            values.append(None)
        momentums = pg.get("m")
        if momentums is None and kind in ("adam", "adamw"):
            momentums = pg["m"] = [None] * len(pg.get("params", []))
        if momentums is not None:
            while len(momentums) < len(pg.get("params", [])):
                momentums.append(None)
        for i, param in enumerate(pg.get("params", [])):
            state, entry = shard._fsdp_param_entry(param)
            if state is None:
                continue
            grad = grads[i] if i < len(grads) else None
            if not isinstance(grad, jt.Var):
                grad = getattr(param, "_torch_grad", None)
            if not isinstance(grad, jt.Var):
                continue
            param_steps[i] = int(param_steps[i]) + 1
            if kind == "sgd":
                new_param, new_value = _sgd_update_for_param(
                    opt, pg, state, entry, entry.shard, grad, values[i])
                values[i] = new_value
            else:
                new_param, new_value, new_momentum = _adam_update_for_param(
                    opt, pg, entry.shard, grad, values[i], momentums[i],
                    decoupled_weight_decay=(kind == "adamw"),
                    n_step=param_steps[i])
                values[i] = new_value
                momentums[i] = new_momentum
            if getattr(state, "true_fsdp_flat", False):
                key = (id(state), id(entry))
                flat_updates[key] = new_param
                flat_public_grads[key] = grad
            else:
                _assign_preserve_trainability(
                    entry.shard, new_param,
                    entry_trainable[(id(state), id(entry))])
                object.__setattr__(entry.shard, "_torch_grad", grad)
                object.__setattr__(entry.owner, entry.attr, entry.shard)
                entry.full_param = None

    for state in states:
        if not getattr(state, "true_fsdp_flat", False):
            continue
        parts = []
        real_numel = 0
        for entry in state.true_fsdp_params:
            key = (id(state), id(entry))
            part = flat_updates.get(key, entry.shard)
            part_numel = common._param_numel(part)
            if part_numel:
                parts.append(common._flatten_var(part))
                real_numel += part_numel
        if real_numel < int(state.true_fsdp_flat_shard_numel):
            parts.append(state.true_fsdp_flat_shard[real_numel:])
        if parts and flat_updates:
            new_flat = parts[0] if len(parts) == 1 else jt.concat(parts, dim=0)
            _assign_preserve_trainability(
                state.true_fsdp_flat_shard, new_flat.stop_grad(),
                flat_trainable[id(state)])
            # Public parameters are refreshed as views of the flat shard below.
            # Materialize the update first so a following model.cpu() does not
            # execute the whole optimizer graph while migrating the first view.
            state.true_fsdp_flat_shard.sync()
            shard._refresh_flat_entry_shards(state)
        for entry in state.true_fsdp_params:
            key = (id(state), id(entry))
            was_trainable = entry_trainable[key]
            if was_trainable and entry.shard.is_stop_grad():
                entry.shard.start_grad()
            elif not was_trainable and not entry.shard.is_stop_grad():
                entry.shard.stop_grad()
            grad = flat_public_grads.get(key)
            if isinstance(grad, jt.Var):
                object.__setattr__(entry.shard, "_torch_grad", grad)
            object.__setattr__(entry.owner, entry.attr, entry.shard)
            entry.full_param = None
        state.true_fsdp_unsharded = False

    for state in states:
        state.true_fsdp_unsharded = False
    _refresh_all_optimizer_fsdp_params(states, opt)
    try:
        opt._build_grad_map()
    except EXPECTED as exc:
        swallowed("fsdp2/optimizer.py optimizer_step: opt._build_grad_map()", exc)
    return True


def sharded_sgd_step(module, loss, lr=1e-4, *, divide_by_world_size=True):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        params = list(module.parameters())
        grads = jt.grad(loss, params)
        for p, g in zip(params, grads):
            p.assign(p - g * lr)
        return grads
    grads = grad_sync.sync_sharded_grads(
        module, loss, divide_by_world_size=divide_by_world_size)
    if getattr(state, "true_fsdp_flat", False):
        flat_grad = getattr(state, "true_fsdp_last_flat_grad", None)
        if flat_grad is None:
            raise RuntimeError("true FSDP2 flat gradient was not produced")
        flat_was_trainable = not state.true_fsdp_flat_shard.is_stop_grad()
        entry_trainable = [not entry.shard.is_stop_grad()
                           for entry in state.true_fsdp_params]
        _assign_preserve_trainability(
            state.true_fsdp_flat_shard,
            (state.true_fsdp_flat_shard - flat_grad * lr).stop_grad(),
            flat_was_trainable)
        shard._refresh_flat_entry_shards(state)
        for entry, was_trainable in zip(state.true_fsdp_params, entry_trainable):
            if was_trainable and entry.shard.is_stop_grad():
                entry.shard.start_grad()
            object.__setattr__(entry.owner, entry.attr, entry.shard)
            entry.full_param = None
        state.true_fsdp_unsharded = False
        return grads
    for entry, grad in zip(state.true_fsdp_params, grads):
        _assign_preserve_trainability(
            entry.shard, (entry.shard - grad * lr).stop_grad())
        object.__setattr__(entry.owner, entry.attr, entry.shard)
        entry.full_param = None
    state.true_fsdp_unsharded = False
    return grads


def local_sharded_state_dict(module):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return module.state_dict()
    return {entry.name: entry.shard for entry in state.true_fsdp_params}

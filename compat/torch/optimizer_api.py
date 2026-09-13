"""Stable optimizer state and update adapters, sharing native mathematics."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from collections.abc import Mapping
import weakref as _weakref
import jittor as jt
import numpy as np
from .context import get_install_context
from .types import _dtype_to_str
from ..diagnostics import EXPECTED, swallowed
from typing import Any, Dict, List
from .. import fsdp_hooks as _fsdp_hooks
from .. import optimizer_kinds as _optimizer_kinds
from .tensor_state import get_tensor_state
from .installers.tensor.autograd_api import _optimizer_maybe_has_fsdp_params
from jittor.optim.algorithms.adam import adam_update

def _init(self, *a, **k):
    _context = get_install_context(jt)
    _orig_init = _context.state["optimizer_native_api"]['_orig_init']
    _orig_init(self, *a, **k)
    # Maintain a registry of ALL live optimizers (not just the last). torch
    # supports several optimizers active at once (3DGS has a Gaussian Adam +
    # an exposure Adam); loss.backward() must fill grads for every one. Hold
    # weakrefs, pruned of dead entries + this same object.
    reg = get_tensor_state(jt).active_optimizers
    reg[:] = [r for r in reg if r() is not None and r() is not self]
    reg.append(_weakref.ref(self))
    try:
        for pg in self.param_groups:
            pg.setdefault("lr", self.lr)
    except EXPECTED as exc:
        swallowed("torch/optimizers.py _init: for pg in self.param_groups:", exc)


def _torch_param_steps(pg):
    params = list(pg.get("params", []))
    steps = pg.get("_torch_steps")
    if not isinstance(steps, list):
        steps = pg["_torch_steps"] = [0] * len(params)
    while len(steps) < len(params):
        steps.append(0)
    if len(steps) > len(params):
        del steps[len(params):]
    return steps


def _torch_optimizer_kind(opt):
    """Which optimizer's state layout `opt` has.

    Identity through the MRO, not a substring of the class name -- `SGDW`
    and `MyAdamWrapper` used to match rules they do not implement. This
    answer only describes *state layout* (which keys `state` and
    `state_dict()` expose), so unlike the FSDP2 one it does not refuse a
    subclass that overrides step(): such a subclass still keeps the base
    class's state arrays. It falls back to the lowercased class name so an
    unrecognised optimizer keeps its previous, harmless behaviour here.

    See jittor/compat/optimizer_kinds.py.
    """
    return (_optimizer_kinds.kind_of(opt)
            or type(opt).__name__.lower())


class _ParamState(dict):
    def __init__(self, owner, param, values):
        dict.__init__(self, values)
        self._owner = owner
        self._param = param
    def __setitem__(self, key, value):
        self._owner._set_field(self._param, key, value)
        dict.__setitem__(self, key, value)
    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = value


class _OptState:
    def __init__(self, opt):
        self._opt = opt
    def _find(self, param):
        for pg in self._opt.param_groups:
            for i, p in enumerate(pg.get("params", [])):
                if p is param:
                    return pg, i
        return None, None
    def _params(self):
        for pg in self._opt.param_groups:
            for p in pg.get("params", []):
                marker = object()
                if self.get(p, marker) is not marker:
                    yield p
    def _reset_slot(self, pg, i):
        _torch_param_steps(pg)[i] = 0
        for key in ("m", "values", "v", "d", "pre_grad"):
            buffers = pg.get(key)
            if not isinstance(buffers, list) or i >= len(buffers):
                continue
            buffer = buffers[i]
            buffers[i] = (jt.zeros_like(buffer).stop_grad()
                          if isinstance(buffer, jt.Var) else None)
    def _sync_n_step(self):
        self._opt.n_step = max(
            (int(step) for pg in self._opt.param_groups
             for step in _torch_param_steps(pg)), default=0)
    def _set_field(self, param, key, value):
        pg, i = self._find(param)
        if pg is None:
            raise KeyError(param)
        kind = _torch_optimizer_kind(self._opt)
        if key == "step":
            if isinstance(value, jt.Var):
                value = value.item()
            _torch_param_steps(pg)[i] = int(value)
            self._sync_n_step()
            return
        mappings = {
            "adam": {"exp_avg": "m", "exp_avg_sq": "values"},
            "adamw": {"exp_avg": "m", "exp_avg_sq": "values"},
            "sgd": {"momentum_buffer": "values"},
            "rmsprop": {"square_avg": "values"},
            "adan": {"exp_avg": "m", "exp_avg_sq": "v",
                     "exp_avg_diff": "d", "pre_grad": "pre_grad"},
        }
        target = mappings.get(kind, {}).get(key)
        buffers = pg.get(target) if target is not None else None
        if isinstance(buffers, list) and i < len(buffers):
            buffers[i] = value
    def get(self, param, default=None):
        pg, i = self._find(param)
        if pg is None:
            return default
        steps = _torch_param_steps(pg)
        if int(steps[i]) <= 0:
            return default
        kind = _torch_optimizer_kind(self._opt)
        if kind in ("adam", "adamw") and "m" in pg and "values" in pg:
            return _ParamState(self, param, {
                "exp_avg": pg["m"][i],
                "exp_avg_sq": pg["values"][i],
                "step": float(steps[i])})
        if kind == "sgd" and "values" in pg and pg.get(
                "momentum", getattr(self._opt, "momentum", 0)):
            return _ParamState(self, param, {
                "momentum_buffer": pg["values"][i]})
        if kind == "rmsprop" and "values" in pg:
            return _ParamState(self, param, {
                "square_avg": pg["values"][i],
                "step": float(steps[i])})
        if kind == "adan":
            out = {"step": float(steps[i])}
            for source, target in (
                    ("m", "exp_avg"), ("v", "exp_avg_sq"),
                    ("d", "exp_avg_diff"),
                    ("pre_grad", "pre_grad")):
                if source in pg and i < len(pg[source]):
                    out[target] = pg[source][i]
            return _ParamState(self, param, out)
        return default
    def __getitem__(self, param):
        r = self.get(param, None)
        if r is None:
            raise KeyError(param)
        return r
    def __setitem__(self, param, d):
        pg, i = self._find(param)
        if pg is None:
            raise KeyError(param)
        if not isinstance(d, Mapping):
            raise TypeError("optimizer state must be a mapping")
        values = dict(d)
        self._reset_slot(pg, i)
        for key, value in values.items():
            if key != "step":
                self._set_field(param, key, value)
        self._set_field(param, "step", values.get("step", 1 if values else 0))
    def __delitem__(self, param):
        pg, i = self._find(param)
        marker = object()
        if pg is None or self.get(param, marker) is marker:
            raise KeyError(param)
        self._reset_slot(pg, i)
        self._sync_n_step()
    def __contains__(self, param):
        marker = object()
        return self.get(param, marker) is not marker
    def __iter__(self):
        return self._params()
    def __len__(self):
        return sum(1 for _ in self._params())
    def keys(self):
        return list(self._params())
    def values(self):
        return [self.get(p, {}) for p in self._params()]
    def items(self):
        return [(p, self.get(p, {})) for p in self._params()]
    def get_state_dict_key(self, param):
        return self._find(param)


def _state_dict_torch(self):
    _context = get_install_context(jt)
    g = _context.target_namespace
    kind = _torch_optimizer_kind(self)
    param_ids: Dict[int, int] = {}
    param_groups = []
    next_id = 0
    for pg in self.param_groups:
        group = {}
        params = []
        for p in pg.get("params", []):
            pid = param_ids.get(id(p))
            if pid is None:
                pid = next_id
                next_id += 1
                param_ids[id(p)] = pid
            params.append(pid)
        for k, v in pg.items():
            if k in ("params", "grads", "m", "values", "v", "d",
                     "pre_grad", "_torch_steps"):
                continue
            group[k] = v
        group.setdefault("lr", pg.get("lr", getattr(self, "lr", 0.0)))
        if kind in ("adam", "adamw"):
            group.setdefault("betas", pg.get(
                "betas", getattr(self, "betas", (0.9, 0.999))))
            group.setdefault("eps", pg.get(
                "eps", getattr(self, "eps", 1e-8)))
            group.setdefault("weight_decay", pg.get(
                "weight_decay", getattr(self, "weight_decay", 0)))
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", getattr(self, "fused", None))
        elif kind == "sgd":
            for key, default in (
                    ("momentum", 0), ("dampening", 0),
                    ("weight_decay", 0), ("nesterov", False)):
                group.setdefault(key, pg.get(
                    key, getattr(self, key, default)))
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)
        elif kind == "rmsprop":
            group.setdefault("alpha", pg.get(
                "alpha", getattr(self, "alpha", 0.99)))
            group.setdefault("eps", pg.get(
                "eps", getattr(self, "eps", 1e-8)))
            group.setdefault("weight_decay", 0)
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)
            group.setdefault("capturable", False)
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
        group["params"] = params
        param_groups.append(group)
    state = {}
    for pg in self.param_groups:
        steps = _torch_param_steps(pg)
        for i, p in enumerate(pg.get("params", [])):
            pid = param_ids.get(id(p))
            if pid is None or int(steps[i]) <= 0:
                continue
            entry = {}
            if kind in ("adam", "adamw"):
                if "m" in pg and i < len(pg["m"]):
                    entry["exp_avg"] = pg["m"][i]
                if "values" in pg and i < len(pg["values"]):
                    entry["exp_avg_sq"] = pg["values"][i]
            elif kind == "sgd":
                momentum = pg.get(
                    "momentum", getattr(self, "momentum", 0))
                if momentum and "values" in pg and i < len(pg["values"]):
                    entry["momentum_buffer"] = pg["values"][i]
            elif kind == "rmsprop":
                if "values" in pg and i < len(pg["values"]):
                    entry["square_avg"] = pg["values"][i]
            elif kind == "adan":
                for source, target in (
                        ("m", "exp_avg"), ("v", "exp_avg_sq"),
                        ("d", "exp_avg_diff"),
                        ("pre_grad", "pre_grad")):
                    values = pg.get(source)
                    if values is not None and i < len(values):
                        entry[target] = values[i]
            if entry:
                if kind != "sgd":
                    entry["step"] = g.tensor(float(steps[i]), dtype=g.float32)
                state[pid] = entry
    return {"state": state, "param_groups": param_groups}


def _load_state_dict_torch(self, state_dict):
    _context = get_install_context(jt)
    _native_load_state_dict = _context.state["optimizer_native_api"]['_native_load_state_dict']
    if not isinstance(state_dict, Mapping) or "param_groups" not in state_dict:
        return _native_load_state_dict(self, state_dict)
    saved_groups = state_dict.get("param_groups", [])
    saved_state = state_dict.get("state", {})
    kind = _torch_optimizer_kind(self)
    if not isinstance(saved_groups, (list, tuple)):
        raise TypeError("loaded optimizer param_groups must be a sequence")
    if not isinstance(saved_state, Mapping):
        raise TypeError("loaded optimizer state must be a mapping")
    if len(saved_groups) != len(self.param_groups):
        raise ValueError(
            "loaded state dict has a different number of parameter groups")
    load_plan = []
    max_step = 0
    for saved_pg, current_pg in zip(saved_groups, self.param_groups):
        if not isinstance(saved_pg, Mapping):
            raise TypeError("loaded optimizer parameter group must be a mapping")
        saved_params = saved_pg.get("params", [])
        if not isinstance(saved_params, (list, tuple)):
            raise TypeError("loaded optimizer group params must be a sequence")
        if len(saved_params) != len(current_pg.get("params", [])):
            raise ValueError(
                "loaded state dict contains a parameter group that "
                "doesn't match the size of optimizer's group")
        slots = []
        for pid in saved_params:
            missing = object()
            try:
                st = saved_state.get(pid, missing)
                if st is missing:
                    st = saved_state.get(str(pid), {})
            except (TypeError, ValueError) as error:
                raise TypeError("loaded optimizer state key is invalid") from error
            if not isinstance(st, Mapping):
                raise TypeError("loaded optimizer parameter state must be a mapping")
            st = dict(st)
            step = 1 if st else 0
            if "step" in st:
                value = st["step"]
                if isinstance(value, jt.Var):
                    value = value.item()
                try:
                    numeric = float(value)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError("loaded optimizer step must be numeric") from error
                if not np.isfinite(numeric) or numeric < 0 or numeric != int(numeric):
                    raise ValueError(
                        "loaded optimizer step must be a non-negative integer")
                step = int(numeric)
            max_step = max(max_step, step)
            slots.append((st, step))
        load_plan.append((dict(saved_pg), slots))
    # Apply only after the complete input has been validated. This keeps
    # malformed loads atomic instead of leaving half-reset moments.
    for pg in self.param_groups:
        steps = _torch_param_steps(pg)
        for i in range(len(steps)):
            steps[i] = 0
        for key in ("m", "values", "v", "d", "pre_grad"):
            buffers = pg.get(key)
            if not isinstance(buffers, list):
                continue
            for i, buffer in enumerate(buffers):
                if isinstance(buffer, jt.Var):
                    buffers[i] = jt.zeros_like(buffer).stop_grad()
    for gi, (saved_pg, slots) in enumerate(load_plan):
        pg = self.param_groups[gi]
        steps = _torch_param_steps(pg)
        for k, v in saved_pg.items():
            if k == "params":
                continue
            pg[k] = v
        for i, (st, step) in enumerate(slots):
            if kind in ("adam", "adamw"):
                if "m" in pg and i < len(pg["m"]) and "exp_avg" in st:
                    pg["m"][i] = st["exp_avg"]
                if "values" in pg and i < len(pg["values"]) \
                        and "exp_avg_sq" in st:
                    pg["values"][i] = st["exp_avg_sq"]
            elif kind == "sgd" and "momentum_buffer" in st:
                pg["values"][i] = st["momentum_buffer"]
            elif kind == "rmsprop" and "square_avg" in st:
                pg["values"][i] = st["square_avg"]
            elif kind == "adan":
                for source, target in (
                        ("m", "exp_avg"), ("v", "exp_avg_sq"),
                        ("d", "exp_avg_diff"),
                        ("pre_grad", "pre_grad")):
                    if target in st and source in pg and i < len(pg[source]):
                        pg[source][i] = st[target]
            steps[i] = step
    self.n_step = max_step
    return None


def _zero_grad_compat(self, set_to_none=True):
    _context = get_install_context(jt)
    _orig_zero = _context.state["optimizer_native_api"]['_orig_zero']
    for _pg in getattr(self, "param_groups", []):
        _params = list(_pg.get("params", []))
        _new_grads: List[Any] = []
        if set_to_none:
            _pg.pop("grads", None)
        else:
            _old_grads = list(_pg.get("grads") or [])
            _new_grads = []
            for _i, _p in enumerate(_params):
                _old = _old_grads[_i] if _i < len(_old_grads) else None
                _published = (
                    getattr(_p, "_torch_grad", None)
                    if isinstance(_p, jt.Var) else None
                )
                if isinstance(_p, jt.Var) and (
                        isinstance(_old, jt.Var)
                        or isinstance(_published, jt.Var)):
                    _existing = (
                        _old if isinstance(_old, jt.Var)
                        else _published
                    )
                    _existing.update(jt.zeros_like(_existing))
                    _existing.stop_grad().stop_fuse()
                    _new_grads.append(_existing)
                else:
                    _new_grads.append(None)
            if any(isinstance(_g, jt.Var) for _g in _new_grads):
                _pg["grads"] = _new_grads
            else:
                _pg.pop("grads", None)
        for _i, _p in enumerate(_params):
            if not isinstance(_p, jt.Var):
                continue
            _value = None
            if not set_to_none and _i < len(_new_grads):
                _value = _new_grads[_i]
            try:
                _p.grad = _value
            except (AttributeError, TypeError) as exc:
                swallowed("torch/optimizers.py _zero_grad_compat: _p.grad = _value", exc)
                object.__setattr__(_p, "_torch_grad", _value)
    try:
        object.__setattr__(self, "_grad_map", {})
    except (AttributeError, TypeError) as exc:
        swallowed("torch/optimizers.py _zero_grad_compat: object.__setattr__(self, '_grad_map', {})", exc)
    # The compatibility pass above has already materialized zero tensors for
    # existing gradients and preserved None for untouched parameters. Jittor's
    # native implementation assumes every entry in ``pg['grads']`` is a Var,
    # so calling it for the mixed list used by set_to_none=False raises.
    result = _orig_zero(self) if set_to_none else None
    _fsdp2_zero = _fsdp_hooks.provider()
    if _fsdp2_zero is not None and _fsdp2_zero.optimizer_has_fsdp_params(self):
        _fsdp2_zero.refresh_visible_full_grads(self)
    object.__setattr__(self, "_torch_backward_advanced_n_step", False)
    return result


def _backward_with_step_marker(self, *args, **kwargs):
    _context = get_install_context(jt)
    _orig_backward = _context.state["optimizer_native_api"]['_orig_backward']
    result = _orig_backward(self, *args, **kwargs)
    object.__setattr__(self, "_torch_backward_advanced_n_step", True)
    return result


def _optimizer_has_ready_grads(opt):
    for _pg in getattr(opt, "param_groups", []):
        _grads = _pg.get("grads")
        if not _grads:
            continue
        for _p, _g in zip(_pg.get("params", []), _grads):
            if isinstance(_p, jt.Var) and isinstance(_g, jt.Var) and list(_p.shape) == list(_g.shape):
                return True
    return False


def _advance_ready_param_steps(opt):
    for pg in getattr(opt, "param_groups", []):
        steps = _torch_param_steps(pg)
        grads = pg.get("grads") or []
        for i, (param, grad) in enumerate(zip(pg.get("params", []), grads)):
            if isinstance(param, jt.Var) and isinstance(grad, jt.Var) \
                    and list(param.shape) == list(grad.shape):
                steps[i] = int(steps[i]) + 1


def _advance_trainable_param_steps(opt):
    for pg in getattr(opt, "param_groups", []):
        steps = _torch_param_steps(pg)
        for i, param in enumerate(pg.get("params", [])):
            if isinstance(param, jt.Var) and param.requires_grad:
                steps[i] = int(steps[i]) + 1


def _load_fsdp2_for_optimizer(opt):
    # The seam, not an import: this file is below fsdp2 in the dependency
    # order (jittor/compat/fsdp_hooks.py). The guard is what makes the
    # `None` answer safe -- `_jittor_fsdp2_state` is set only by fsdp2, and
    # fsdp2 registers itself when imported, so an optimizer holding a
    # sharded parameter always finds a provider here.
    if not _optimizer_maybe_has_fsdp_params(opt):
        return None
    return _fsdp_hooks.provider()


class LBFGS(jt.optim.Optimizer):
    def __init__(self, params, lr=1.0, *args, **kwargs):
        super().__init__(params, lr)
    def step(self, closure=None):
        raise NotImplementedError("torch.optim.LBFGS is not implemented by the jittor torch shim")


def _lsd(self, state):
    _context = get_install_context(jt)
    _orig_lsd = _context.state["optimizer_native_api"]['_orig_lsd']
    trainable = []
    try:
        for pg in self.param_groups:
            for p in pg.get("params", []):
                if p.requires_grad:
                    trainable.append(p)
    except EXPECTED as exc:
        swallowed("torch/optimizers.py _lsd: for pg in self.param_groups:", exc)
    r = _orig_lsd(self, state)
    for p in trainable:
        try: p.start_grad()
        except EXPECTED as exc: swallowed("torch/optimizers.py _lsd: p.start_grad()", exc)
    return r


def _state_getter(self):
    return get_install_context(jt).target_namespace.optim.Optimizer._OptState(self)


def _lbfgs_type(base):
    if base is jt.optim.Optimizer:
        return LBFGS
    return type("LBFGS", (LBFGS, base), {"__module__": "torch.optim"})


def _step_with_closure(self, loss, retain_graph, closure, kwargs, native_kind):
    _orig_step = get_install_context(jt).state["optimizer_native_api"]["steps"][native_kind]
    called_closure = False
    native_fsdp_loss = None
    if closure is None and callable(loss):
        closure = loss
        loss = None
    if closure is not None:
        loss = closure()
        called_closure = True
    _fsdp2_step = _load_fsdp2_for_optimizer(self)
    if _fsdp2_step is not None and _fsdp2_step.optimizer_has_fsdp_params(self):
        if loss is not None and not called_closure:
            native_fsdp_loss = loss
            loss.backward(retain_graph=retain_graph)
            loss = None
        if not _fsdp2_step.optimizer_step(
                self, None, retain_graph=retain_graph, native_kind=native_kind):
            raise NotImplementedError(
                f"FSDP2 optimizer step is not implemented for {type(self).__name__}")
        if not _fsdp2_step.optimizer_has_non_fsdp_params(self):
            if native_fsdp_loss is not None:
                try:
                    self.post_step()
                except EXPECTED as exc:
                    swallowed("torch/optimizers.py _step_torch_closure: self.post_step()", exc)
            else:
                jt.flags.node_order = 0
                object.__setattr__(
                    self, "_torch_backward_advanced_n_step", False)
            return loss if called_closure else None
        _fsdp2_step.clear_fsdp_optimizer_grads(self)
    torch_style = native_fsdp_loss is None and (loss is None or called_closure)
    if torch_style and not _optimizer_has_ready_grads(self):
        jt.flags.node_order = 0
        object.__setattr__(
            self, "_torch_backward_advanced_n_step", False)
        return loss if called_closure else None
    if not torch_style:
        if loss is None and not getattr(
                self, "_torch_backward_advanced_n_step", False):
            self.n_step = int(getattr(self, "n_step", 0)) + 1
        if loss is None:
            _advance_ready_param_steps(self)
        else:
            _advance_trainable_param_steps(self)
        out = _orig_step(self, loss, retain_graph=retain_graph)
        object.__setattr__(self, "_torch_backward_advanced_n_step", False)
        return out
    # Native SGD/RMSprop/Adan always post_step()->zero_grad(). Torch
    # keeps parameter.grad until the caller explicitly clears it.
    previous_post = self.__dict__.get("post_step", None)
    had_post = "post_step" in self.__dict__
    previous_step = int(getattr(self, "n_step", 0))
    if not getattr(self, "_torch_backward_advanced_n_step", False):
        self.n_step = previous_step + 1
    _advance_ready_param_steps(self)
    self.post_step = _torch_post_step
    try:
        out = _orig_step(self, None, retain_graph=retain_graph)
    finally:
        object.__setattr__(self, "_torch_backward_advanced_n_step", False)
        if had_post:
            self.post_step = previous_post
        else:
            self.__dict__.pop("post_step", None)
    return loss if called_closure and out is None else out


def _update_in_target_dtype(target, value):
    if _dtype_to_str(value.dtype) != _dtype_to_str(target.dtype):
        value = value.cast(_dtype_to_str(target.dtype))
    target.update(value)


def _adam_step(self, loss, retain_graph, closure, kwargs, decoupled_weight_decay):
    native_fsdp_loss = None
    if closure is None and callable(loss):
        closure = loss
        loss = None
    if closure is not None:
        loss = closure()
    _fsdp2_step = _load_fsdp2_for_optimizer(self)
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
    self.pre_step(None if closure is not None and _optimizer_has_ready_grads(self) else loss, retain_graph)
    if not _optimizer_has_ready_grads(self):
        if native_fsdp_loss is not None:
            self.post_step()
        else:
            jt.flags.node_order = 0
            object.__setattr__(
                self, "_torch_backward_advanced_n_step", False)
        return native_fsdp_loss if native_fsdp_loss is not None else loss
    if not getattr(self, "_torch_backward_advanced_n_step", False):
        self.n_step = int(getattr(self, "n_step", 0)) + 1
    jt.flags.node_order = 1
    for pg in self.param_groups:
        lr = pg.get("lr", self.lr)
        eps = pg.get("eps", self.eps)
        weight_decay = pg.get("weight_decay", self.weight_decay)
        b0, b1 = pg.get("betas", self.betas)
        param_steps = _torch_param_steps(pg)
        # torch permits optimizers containing frozen or otherwise
        # unused parameters. loss.backward() then leaves the group
        # without gradients and step() must be a no-op, not KeyError.
        grads = pg.get("grads") or [None] * len(pg["params"])
        fused = (decoupled_weight_decay and jt.flags.use_acl and
                 pg.get("fused", getattr(self, "fused", None)) is True)
        if fused:
            active = []
            for i, (p, g, v, m) in enumerate(zip(
                    pg["params"], grads, pg["values"], pg["m"])):
                if not p.requires_grad or not isinstance(g, jt.Var) \
                        or list(g.shape) != list(p.shape):
                    continue
                param_steps[i] = int(param_steps[i]) + 1
                active.append((p, m, v, g, param_steps[i] - 1))
            from jittor.optim.algorithms.adam import _acl_fused_adamw_updates
            updates = _acl_fused_adamw_updates(
                active, lr, b0, b1, weight_decay, eps)
            for (p, m, v, _, _), (new_p, new_m, new_v) in zip(
                    active, updates):
                _update_in_target_dtype(p, new_p)
                _update_in_target_dtype(m, new_m)
                _update_in_target_dtype(v, new_v)
                if p.is_stop_grad():
                    p.start_grad()
            continue
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


def _torch_post_step():
    jt.flags.node_order = 0


def adam_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    return _adam_step(self, loss, retain_graph, closure, kwargs, False)


def adamw_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    return _adam_step(self, loss, retain_graph, closure, kwargs, True)


def _initialize_default(self, params, lr, args, kwargs, name):
    initializer = get_install_context(jt).state["optimizer_native_api"]["initializers"][name]
    return initializer(self, params, lr, *args, **kwargs)

def sgd_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    return _step_with_closure(self, loss, retain_graph, closure, kwargs, 'sgd')


def rmsprop_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    return _step_with_closure(self, loss, retain_graph, closure, kwargs, 'rmsprop')


def adan_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    return _step_with_closure(self, loss, retain_graph, closure, kwargs, 'adan')


_STEP_APIS = {"sgd": sgd_step, "rmsprop": rmsprop_step, "adan": adan_step}

def adam_init(self, params, lr=1e-3, *args, **kwargs):
    return _initialize_default(self, params, lr, args, kwargs, 'Adam')


def adamw_init(self, params, lr=1e-3, *args, **kwargs):
    return _initialize_default(self, params, lr, args, kwargs, 'AdamW')


def rmsprop_init(self, params, lr=1e-3, *args, **kwargs):
    return _initialize_default(self, params, lr, args, kwargs, 'RMSprop')


def adan_init(self, params, lr=1e-3, *args, **kwargs):
    return _initialize_default(self, params, lr, args, kwargs, 'Adan')


_INITIALIZER_APIS = {
    'Adam': adam_init,
    'AdamW': adamw_init,
    'RMSprop': rmsprop_init,
    'Adan': adan_init,
}

"""Stable optimizer state and update adapters, sharing native mathematics."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name, var_dtype_name
from collections.abc import Mapping
import weakref as _weakref
import jittor as jt
import numpy as np
from .context import get_install_context
from .types import _dtype_to_str
from ..diagnostics import EXPECTED, swallowed
from typing import Any, Dict, List
from .. import fsdp_hooks as _fsdp_hooks
from .optimizer_state import (
    _OptState, _ParamState, _adagrad_step_tensor,
    _torch_optimizer_kind, _torch_param_steps,
)
from .tensor_state import get_tensor_state
from .installers.tensor.autograd_api import _optimizer_maybe_has_fsdp_params
from jittor.optim.algorithms.adam import adam_update
from jittor.optim.base import _state_buffer


def _reset_adam_state_to_lazy(opt):
    """Match torch by leaving Adam moments unallocated until the first step.

    Native Jittor creates both full-sized moment tensors in its constructor.
    Besides differing from torch, that is wrong for partitioning runtimes such
    as DeepSpeed ZeRO: they replace a model parameter group with rank-local
    flat partitions before the first step, so eager states retain the original
    model shapes and can consume several extra device gigabytes.
    """
    for pg in getattr(opt, "param_groups", []):
        size = len(pg.get("params", ()))
        pg["m"] = [None] * size
        pg["values"] = [None] * size


def _ensure_adam_group_state(pg):
    """Create Adam moments for the group's current parameter objects."""
    from .frontend import tensor_frontend
    tensor_type = get_install_context(jt).target_namespace.Var
    params = list(pg.get("params", ()))
    for key in ("m", "values"):
        buffers = pg.get(key)
        if not isinstance(buffers, list) or len(buffers) != len(params):
            buffers = pg[key] = [None] * len(params)
        for index, param in enumerate(params):
            buffer = buffers[index]
            if not isinstance(buffer, jt.Var) or list(buffer.shape) != list(param.shape):
                # Native zeros_like constructs by shape, so it needs the
                # optimizer owner's frontend scope to preserve Tensor state.
                with tensor_frontend(tensor_type, like=param):
                    buffers[index] = _state_buffer(param)


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
                     "pre_grad", "_torch_steps", "_adagrad_steps"):
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
            if pid is None or (kind != "adagrad" and int(steps[i]) <= 0):
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
            elif kind == "adagrad":
                entry["sum"] = pg["values"][i]
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
                    entry["step"] = (steps[i] if kind == "adagrad" else
                                     g.tensor(float(steps[i]), dtype=g.float32))
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
        if kind == "adagrad":
            from jittor.optim.algorithms.adagrad import validate_adagrad_options
            validate_adagrad_options(dict(current_pg, **saved_pg))
        slots = []
        for param_index, pid in enumerate(saved_params):
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
            if kind == "adagrad":
                value = st.get("sum")
                param = current_pg["params"][param_index]
                if not isinstance(value, jt.Var) or list(value.shape) != list(param.shape):
                    raise ValueError("loaded Adagrad sum must match its parameter shape")
                if _jittor_dtype_name(value.dtype) != "float32":
                    raise NotImplementedError("loaded Adagrad sum must be float32")
                if "step" not in st:
                    raise ValueError("loaded Adagrad state requires step")
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
        if kind in ("adam", "adamw"):
            # Loading an optimizer is itself state materialization. Allocate
            # every current slot so absent entries are reset to zero exactly as
            # torch's partial state dictionaries require.
            _ensure_adam_group_state(pg)
        steps = _torch_param_steps(pg)
        for i in range(len(steps)):
            if kind != "adagrad":
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
            elif kind == "adagrad":
                pg["values"][i] = st["sum"].clone().stop_grad()
            elif kind == "adan":
                for source, target in (
                        ("m", "exp_avg"), ("v", "exp_avg_sq"),
                        ("d", "exp_avg_diff"),
                        ("pre_grad", "pre_grad")):
                    if target in st and source in pg and i < len(pg[source]):
                        pg[source][i] = st[target]
            if kind == "adagrad":
                steps[i].update(_adagrad_step_tensor(step))
            else:
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
    result = _orig_zero(self)
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


def _sync_published_parameter_grads(opt):
    """Make dynamic optimizer groups consume each current parameter's grad.

    PyTorch optimizers read param.grad at step time. Jittor's native
    optimizers keep a parallel param_group["grads"] list instead. Frameworks
    such as DeepSpeed ZeRO-3 temporarily replace a group's params with a flat
    partition after assigning its grad, so the backward-time registry cannot
    have populated that parallel list. Rebuild it from the public grad slots
    immediately before a torch-style step.
    """
    for pg in getattr(opt, "param_groups", []):
        grads = []
        has_grad = False
        for param in pg.get("params", []):
            grad = (
                getattr(param, "_torch_grad", None)
                if isinstance(param, jt.Var) else None
            )
            if isinstance(grad, jt.Var) and list(grad.shape) == list(param.shape):
                grads.append(grad)
                has_grad = True
            else:
                grads.append(None)
        if has_grad:
            pg["grads"] = grads
        else:
            pg.pop("grads", None)


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
    from jittor._runtime import step_capture   # only SGD hands its books to a capture
    if native_kind != "sgd": step_capture.refuse(f"a captured step cannot replay {native_kind}")
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
    if torch_style and not getattr(
            self, "_torch_backward_advanced_n_step", False):
        # Native optimizer.backward() already populated param_group["grads"].
        # Rebuilding from public param.grad here would erase those gradients;
        # only torch's loss.backward() spelling needs the public-slot bridge.
        _sync_published_parameter_grads(self)
    if torch_style and not _optimizer_has_ready_grads(self):
        jt.flags.node_order = 0
        object.__setattr__(
            self, "_torch_backward_advanced_n_step", False)
        return loss if called_closure else None
    if not torch_style:
        if loss is None and not getattr(
                self, "_torch_backward_advanced_n_step", False):
            self.n_step = int(getattr(self, "n_step", 0)) + 1
        if native_kind != "adagrad":
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
    if native_kind != "adagrad":
        _advance_ready_param_steps(self)
    from .optimizers import replay_books_of_step
    replay_books_of_step(self)
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
    if var_dtype_name(value) != var_dtype_name(target):  # native, not `.dtype`
        value = value.cast(var_dtype_name(target))
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
    torch_style = loss is None or closure is not None
    if torch_style and not getattr(
            self, "_torch_backward_advanced_n_step", False):
        # Native optimizer.backward() already populated param_group["grads"].
        # Rebuilding from public param.grad here would erase those gradients;
        # only torch's loss.backward() spelling needs the public-slot bridge.
        _sync_published_parameter_grads(self)
    self.pre_step(
        None if closure is not None and _optimizer_has_ready_grads(self)
        else loss,
        retain_graph,
    )
    if not _optimizer_has_ready_grads(self):
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
        param_steps = _torch_param_steps(pg)
        # torch permits optimizers containing frozen or otherwise
        # unused parameters. loss.backward() then leaves the group
        # without gradients and step() must be a no-op, not KeyError.
        grads = pg.get("grads") or [None] * len(pg["params"])
        if not any(
                isinstance(p, jt.Var) and isinstance(g, jt.Var)
                and p.requires_grad and list(g.shape) == list(p.shape)
                for p, g in zip(pg["params"], grads)):
            continue
        # Do this only once a group has a real update. All slots in that group
        # are initialized together so unused parameters expose zero moments
        # after the first group step, while optimizer.state remains absent for
        # them until their own per-parameter step becomes nonzero.
        _ensure_adam_group_state(pg)
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


def adagrad_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    """Reuse native Adagrad, preserving Torch gradients and closure returns."""
    if kwargs:
        raise TypeError("unexpected Adagrad.step options: %s" % sorted(kwargs))
    if _optimizer_maybe_has_fsdp_params(self):
        raise NotImplementedError("Adagrad FSDP updates are not supported")
    from jittor.optim.algorithms.adagrad import _validate_tensor, validate_adagrad_options
    for group in self.param_groups:
        validate_adagrad_options(group)
        for grad in group.get("grads", ()):
            if grad is not None:
                _validate_tensor(grad, "gradients")
    return _step_with_closure(self, loss, retain_graph, closure, kwargs, 'adagrad')


_STEP_APIS = {"sgd": sgd_step, "rmsprop": rmsprop_step, "adan": adan_step,
              "adagrad": adagrad_step}

def adam_init(self, params, lr=1e-3, *args, **kwargs):
    result = _initialize_default(self, params, lr, args, kwargs, 'Adam')
    _reset_adam_state_to_lazy(self)
    return result


def adamw_init(self, params, lr=1e-3, *args, **kwargs):
    result = _initialize_default(self, params, lr, args, kwargs, 'AdamW')
    _reset_adam_state_to_lazy(self)
    return result


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

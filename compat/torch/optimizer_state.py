"""Torch-compatible optimizer state views and serialization."""
from collections.abc import Mapping
import weakref as _weakref
from typing import Dict

import jittor as jt
import numpy as np
from jittor.optim.algorithms.sgd import (
    MOMENTUM_INITIALIZED as _MOMENTUM_INITIALIZED,
    momentum_initialized_flags as _momentum_initialized_flags,
)

from .context import get_install_context


def _optimizer_api():
    from . import optimizer_api
    return optimizer_api


def _torch_param_steps(pg):
    return _optimizer_api()._torch_param_steps(pg)


def _torch_optimizer_kind(opt):
    return _optimizer_api()._torch_optimizer_kind(opt)


class _ParamState(dict):
    """A stable public dictionary reflecting the canonical native state lists."""
    def __init__(self, owner, param, values):
        dict.__init__(self, values)
        self._owner = owner
        self._param = param
    def _refresh(self):
        if self._owner is not None:
            if self._owner._opt is None:
                self._owner = None
            else:
                values = self._owner._native_values(self._param)
                if values is not None:
                    dict.update(self, values)
        return self
    def _detach(self):
        self._refresh()
        self._owner = None
    def __getitem__(self, key):
        self._refresh()
        return dict.__getitem__(self, key)
    def get(self, key, default=None):
        self._refresh()
        return dict.get(self, key, default)
    def __contains__(self, key):
        self._refresh()
        return dict.__contains__(self, key)
    def __iter__(self):
        self._refresh()
        return dict.__iter__(self)
    def __len__(self):
        self._refresh()
        return dict.__len__(self)
    def keys(self):
        self._refresh()
        return dict.keys(self)
    def values(self):
        self._refresh()
        return dict.values(self)
    def items(self):
        self._refresh()
        return dict.items(self)
    def copy(self):
        self._refresh()
        return dict.copy(self)
    def __repr__(self):
        self._refresh()
        return dict.__repr__(self)
    def __eq__(self, other):
        self._refresh()
        if isinstance(other, _ParamState):
            other._refresh()
        return dict.__eq__(self, other)
    def __ne__(self, other):
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result
    def __setitem__(self, key, value):
        self._refresh()
        if self._owner is not None:
            self._owner._set_field(self._param, key, value)
        dict.__setitem__(self, key, value)
    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value
    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]
    def clear(self):
        if self._owner is not None:
            pg, index = self._owner._find(self._param)
            if pg is not None:
                self._owner._reset_slot(pg, index)
                self._owner._sync_n_step()
        dict.clear(self)


class _OptState:
    def __init__(self, opt):
        # The optimizer caches this view. Avoid making that cache a strong cycle.
        self._opt_ref = _weakref.ref(opt)
        self._entries = {}
    @property
    def _opt(self):
        return self._opt_ref()
    def _find(self, param):
        opt = self._opt
        if opt is not None:
            for pg in opt.param_groups:
                for i, p in enumerate(pg.get("params", [])):
                    if p is param:
                        return pg, i
        return None, None
    def _params(self):
        opt = self._opt
        if opt is not None:
            for pg in opt.param_groups:
                for p in pg.get("params", []):
                    if self.get(p, None) is not None:
                        yield p
    def _reset_slot(self, pg, i):
        _torch_param_steps(pg)[i] = 0
        if _torch_optimizer_kind(self._opt) == "sgd":
            _momentum_initialized_flags(pg)[i] = False
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
        if kind == "sgd" and key == "momentum_buffer" and value is None:
            # torch's "no buffer yet": the next momentum update seeds it.
            if isinstance(buffers, list) and i < len(buffers) \
                    and isinstance(buffers[i], jt.Var):
                buffers[i] = jt.zeros_like(buffers[i]).stop_grad()
            _momentum_initialized_flags(pg)[i] = False
            return
        if isinstance(buffers, list) and i < len(buffers):
            buffers[i] = value
    def _native_values(self, param):
        pg, i = self._find(param)
        if pg is None:
            return None
        steps = _torch_param_steps(pg)
        if int(steps[i]) <= 0:
            return None
        kind = _torch_optimizer_kind(self._opt)
        if kind in ("adam", "adamw") and "m" in pg and "values" in pg:
            return {
                "exp_avg": pg["m"][i],
                "exp_avg_sq": pg["values"][i],
                "step": float(steps[i])}
        if kind == "sgd" and "values" in pg and pg.get(
                "momentum", getattr(self._opt, "momentum", 0)):
            return {
                "momentum_buffer": pg["values"][i]}
        if kind == "rmsprop" and "values" in pg:
            return {
                "square_avg": pg["values"][i],
                "step": float(steps[i])}
        if kind == "adan":
            out = {"step": float(steps[i])}
            for source, target in (
                    ("m", "exp_avg"), ("v", "exp_avg_sq"),
                    ("d", "exp_avg_diff"),
                    ("pre_grad", "pre_grad")):
                if source in pg and i < len(pg[source]):
                    out[target] = pg[source][i]
            return out
        return None
    def _record(self, param, values):
        entry = _ParamState(self, param, values)
        self._entries[id(param)] = entry
        return entry
    def _entry(self, param, create=False):
        pg, _ = self._find(param)
        if pg is None:
            return None
        entry = self._entries.get(id(param))
        if entry is None:
            values = self._native_values(param)
            if values is not None or create:
                entry = self._record(param, values or {})
        return entry
    def get(self, param, default=None):
        entry = self._entry(param)
        return default if entry is None else entry
    def __getitem__(self, param):
        entry = self._entry(param, create=True)
        if entry is None:
            raise KeyError(param)
        return entry
    def __setitem__(self, param, values):
        pg, i = self._find(param)
        if pg is None:
            raise KeyError(param)
        if not isinstance(values, Mapping):
            raise TypeError("optimizer state must be a mapping")
        values = dict(values)
        previous = self._entries.pop(id(param), None)
        if previous is not None:
            previous._detach()
        self._reset_slot(pg, i)
        entry = self._record(param, {})
        entry.update(values)
        if values and "step" not in values:
            self._set_field(param, "step", 1)
        self._sync_n_step()
    def __delitem__(self, param):
        pg, i = self._find(param)
        entry = self._entry(param)
        if pg is None or entry is None:
            raise KeyError(param)
        entry._detach()
        self._entries.pop(id(param), None)
        self._reset_slot(pg, i)
        self._sync_n_step()
    def clear(self):
        for param in list(self._params()):
            del self[param]
    def _detach_all(self):
        for entry in self._entries.values():
            entry._detach()
        self._entries.clear()
    def __contains__(self, param):
        return self._entry(param) is not None
    def __iter__(self):
        return self._params()
    def __len__(self):
        return sum(1 for _ in self._params())
    def keys(self):
        return list(self._params())
    def values(self):
        return [self[p] for p in self._params()]
    def items(self):
        return [(p, self[p]) for p in self._params()]
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
                     "pre_grad", "_torch_steps", "_torch_adam_params", _MOMENTUM_INITIALIZED):
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
            group.setdefault("decoupled_weight_decay", kind == "adamw")
        elif kind == "sgd":
            for key, default in (
                    ("momentum", 0), ("dampening", 0),
                    ("weight_decay", 0), ("nesterov", False)):
                group.setdefault(key, pg.get(
                    key, getattr(self, key, default)))
            group.setdefault("maximize", False)
            group.setdefault("foreach", getattr(self, "foreach", None))
            group.setdefault("differentiable", False)
            group.setdefault("fused", getattr(self, "fused", None))
        elif kind == "rmsprop":
            group.setdefault("alpha", pg.get(
                "alpha", getattr(self, "alpha", 0.99)))
            group.setdefault("eps", pg.get(
                "eps", getattr(self, "eps", 1e-8)))
            group.setdefault("weight_decay", pg.get(
                "weight_decay", getattr(self, "weight_decay", 0)))
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)
            group.setdefault("capturable", False)
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
        group["params"] = params
        param_groups.append(group)
    state = {}
    for parameter, entry in self.state.items():
        pid = param_ids[id(parameter)]
        values = dict(entry)
        if "step" in values and not isinstance(values["step"], jt.Var):
            values["step"] = g.tensor(float(values["step"]), dtype=g.float32)
        state[pid] = values
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
                    st = saved_state.get(str(pid), missing)
                present = st is not missing
                if not present:
                    st = {}
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
            slots.append((st, step, present))
        load_plan.append((dict(saved_pg), slots))
    # Apply only after the complete input has been validated. This keeps
    # malformed loads atomic instead of leaving half-reset moments.
    public_state = self.state
    public_state._detach_all()
    for pg in self.param_groups:
        steps = _torch_param_steps(pg)
        for i in range(len(steps)):
            steps[i] = 0
        if kind == "sgd":
            flags = _momentum_initialized_flags(pg)
            for i in range(len(flags)):
                flags[i] = False
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
            if k in ("params", _MOMENTUM_INITIALIZED):
                continue
            pg[k] = v
        for i, (st, step, present) in enumerate(slots):
            if kind in ("adam", "adamw"):
                if "m" in pg and i < len(pg["m"]) and "exp_avg" in st:
                    pg["m"][i] = st["exp_avg"]
                if "values" in pg and i < len(pg["values"]) \
                        and "exp_avg_sq" in st:
                    pg["values"][i] = st["exp_avg_sq"]
            elif kind == "sgd" and st.get("momentum_buffer") is not None:
                pg["values"][i] = st["momentum_buffer"]
                _momentum_initialized_flags(pg)[i] = True
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
            if present:
                public_state._record(pg["params"][i], st)
    self.n_step = max_step
    return None

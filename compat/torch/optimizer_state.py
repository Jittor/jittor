"""Live optimizer state mappings and per-parameter step metadata.

Update arithmetic stays in native optimizer algorithms. Public optimizer API
bindings stay in optimizer_api; this module owns their state-view helpers.
"""
from collections.abc import Mapping
import weakref as _weakref
import jittor as jt
from .context import get_install_context
from .optimizer_adam import _AdamStep
from .. import optimizer_kinds as _optimizer_kinds
from jittor.optim.base import _state_buffer
from jittor.optim.algorithms.sgd import (
    momentum_initialized_flags as _momentum_initialized_flags,
)


def _param_steps(pg):
    """Read step counts through optimizer_api, their registered owner."""
    from .optimizer_api import _torch_param_steps
    return _torch_param_steps(pg)


#: Torch state-dict key -> native optimizer buffer key, per optimizer kind.
#: This table is the single owner of the name translation: `state_dict`,
#: `load_state_dict` and the live `state` view all read it, so adding a state
#: key means editing one place instead of three.
_TORCH_TO_NATIVE_STATE = {
    "adam": {"exp_avg": "m", "exp_avg_sq": "values"},
    "adamw": {"exp_avg": "m", "exp_avg_sq": "values"},
    "sgd": {"momentum_buffer": "values"},
    "rmsprop": {"square_avg": "values"},
    "adagrad": {"sum": "values"},
    "adan": {"exp_avg": "m", "exp_avg_sq": "v",
             "exp_avg_diff": "d", "pre_grad": "pre_grad"},
}
_NATIVE_TO_TORCH_STATE = {
    kind: {native: torch for torch, native in names.items()}
    for kind, names in _TORCH_TO_NATIVE_STATE.items()
}


def _torch_state_key(kind, native):
    """The Torch spelling of a native buffer key, or None."""
    return _NATIVE_TO_TORCH_STATE.get(kind, {}).get(native)


def _native_state_key(kind, torch):
    """The native buffer key for a Torch state key, or None."""
    return _TORCH_TO_NATIVE_STATE.get(kind, {}).get(torch)


def _native_optimizer_buffer(value, existing=None):
    """Give a loaded buffer the same base type as the one it replaces.

    ``load_state_dict`` materialises state through the Torch facade, whose
    Tensor is a ``jt.Var`` subclass with its own promotion rules. Native
    algorithms (Adam, AdamW, Adagrad) keep native Vars in ``param_groups``;
    loading a frontend Tensor there makes the next update take the frontend
    operator path and drift in the last bits. Algorithms whose buffers are
    frontend Tensors (SGD's momentum) must keep that type, because the public
    ``state`` view exposes the object and callers read ``.device`` from it.
    """
    if (existing is not None and type(existing) is jt.Var
            and isinstance(value, jt.Var) and type(value) is not jt.Var):
        return jt.Var(value)
    return value


def _adagrad_step_tensor(value):
    """Create Torch's host step scalar through its placement-aware factory."""
    torch_owner = get_install_context(jt).target_namespace
    return torch_owner.tensor(float(value), dtype=torch_owner.float32, device="cpu").stop_grad()


def _step_count(value):
    return int(value.item()) if isinstance(value, jt.Var) else int(value)


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
    """A stable public dictionary reflecting the canonical native state lists."""
    def __init__(self, owner, param, values):
        dict.__init__(self, values)
        self._owner = owner
        self._param = param
    @property
    def _owner(self):
        return self._owner_ref() if self._owner_ref is not None else None
    @_owner.setter
    def _owner(self, value):
        # The view caches its entries. An entry must not retain that view,
        # or an observed counter keeps discarded parameters in a cycle.
        self._owner_ref = _weakref.ref(value) if value is not None else None
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
        steps = _param_steps(pg)
        if "_adagrad_steps" in pg:
            steps[i].update(_adagrad_step_tensor(0))
        else:
            list.__setitem__(steps, i, 0)
        if _torch_optimizer_kind(self._opt) == "sgd":
            _momentum_initialized_flags(pg)[i] = False
        for key in ("m", "values", "v", "d", "pre_grad"):
            buffers = pg.get(key)
            if not isinstance(buffers, list) or i >= len(buffers):
                continue
            buffer = buffers[i]
            buffers[i] = (_state_buffer(buffer)
                          if isinstance(buffer, jt.Var) else None)
    def _sync_n_step(self):
        self._opt.n_step = max(
            (_step_count(step) for pg in self._opt.param_groups
             for step in _param_steps(pg)), default=0)
    def _set_field(self, param, key, value):
        pg, i = self._find(param)
        if pg is None:
            raise KeyError(param)
        kind = _torch_optimizer_kind(self._opt)
        if key == "step":
            if isinstance(value, jt.Var) and kind not in ("adam", "adamw"):
                value = value.item()
            steps = _param_steps(pg)
            if kind == "adagrad":
                steps[i].update(_adagrad_step_tensor(value))
            else:
                steps[i] = value if isinstance(value, jt.Var) else int(value)
            self._sync_n_step()
            return
        target = _native_state_key(kind, key)
        buffers = pg.get(target) if target is not None else None
        if kind == "sgd" and key == "momentum_buffer" and value is None:
            # torch's "no buffer yet": the next momentum update seeds it.
            if isinstance(buffers, list) and i < len(buffers) \
                    and isinstance(buffers[i], jt.Var):
                buffers[i] = _state_buffer(buffers[i])
            _momentum_initialized_flags(pg)[i] = False
            return
        if isinstance(buffers, list) and i < len(buffers):
            # A live `state[p][key] = value` keeps the caller's object: the
            # Torch contract exposes that exact tensor. Only checkpoint
            # loading has to restore the native base type.
            buffers[i] = value
            if kind == "sgd" and key == "momentum_buffer":
                _momentum_initialized_flags(pg)[i] = True
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
        if (_torch_optimizer_kind(self._opt) == "sgd"
                and set(values) == {"momentum_buffer"}
                and values["momentum_buffer"] is None):
            # Mainline's explicit buffer reset removes initialized state;
            # the next real momentum update seeds from its gradient.
            entry._detach()
            self._entries.pop(id(param), None)
            self._sync_n_step()
            return
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
    @property
    def _opt(self):
        return self._opt_ref()
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
    def clear(self):
        for param in list(self._params()):
            del self[param]
    def _detach_all(self):
        for entry in self._entries.values():
            entry._detach()
        self._entries.clear()
    def _native_values(self, param, default=None):
        pg, i = self._find(param)
        if pg is None:
            return default
        steps = _param_steps(pg)
        kind = _torch_optimizer_kind(self._opt)
        if kind == 'adagrad':
            return {'sum': pg['values'][i], 'step': steps[i]}
        if not isinstance(steps[i], _AdamStep) and int(steps[i]) <= 0:
            return default
        names = _NATIVE_TO_TORCH_STATE.get(kind, {})

        def present():
            return {torch: pg[native][i] for native, torch in names.items() if native in pg and i < len(pg[native])}
        if kind in ('adam', 'adamw') and 'm' in pg and ('values' in pg):
            return {**present(), 'step': steps[i].tensor() if isinstance(steps[i], _AdamStep) else float(steps[i])}
        if kind == 'sgd':
            if 'values' in pg and _momentum_initialized_flags(pg)[i]:
                return present()
            return default
        if kind == 'rmsprop' and 'values' in pg:
            return {**present(), 'step': float(steps[i])}
        if kind == 'adan':
            return {**present(), 'step': float(steps[i])}
        return default

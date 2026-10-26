"""Live optimizer state mappings and per-parameter step metadata.

Update arithmetic stays in native optimizer algorithms. Public optimizer API
bindings stay in optimizer_api; this module owns their state-view helpers.
"""
from collections.abc import Mapping
import jittor as jt
from .context import get_install_context
from .. import optimizer_kinds as _optimizer_kinds
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
        steps = _param_steps(pg)
        if "_adagrad_steps" in pg:
            steps[i].update(_adagrad_step_tensor(0))
        else:
            steps[i] = 0
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
            (_step_count(step) for pg in self._opt.param_groups
             for step in _param_steps(pg)), default=0)
    def _set_field(self, param, key, value):
        pg, i = self._find(param)
        if pg is None:
            raise KeyError(param)
        kind = _torch_optimizer_kind(self._opt)
        if key == "step":
            if isinstance(value, jt.Var):
                value = value.item()
            steps = _param_steps(pg)
            if kind == "adagrad":
                steps[i].update(_adagrad_step_tensor(value))
            else:
                steps[i] = int(value)
            self._sync_n_step()
            return
        target = _native_state_key(kind, key)
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
            if kind == "sgd" and key == "momentum_buffer":
                _momentum_initialized_flags(pg)[i] = True
    def get(self, param, default=None):
        pg, i = self._find(param)
        if pg is None:
            return default
        steps = _param_steps(pg)
        kind = _torch_optimizer_kind(self._opt)
        if kind == "adagrad":
            return _ParamState(self, param, {
                "sum": pg["values"][i], "step": steps[i]})
        if int(steps[i]) <= 0:
            return default
        names = _NATIVE_TO_TORCH_STATE.get(kind, {})

        def present():
            return {torch: pg[native][i] for native, torch in names.items()
                    if native in pg and i < len(pg[native])}

        if kind in ("adam", "adamw") and "m" in pg and "values" in pg:
            return _ParamState(self, param, {
                **present(), "step": float(steps[i])})
        if kind == "sgd":
            # Present exactly while a buffer exists, like torch: independent
            # of the *current* momentum, so pausing momentum keeps it.
            if "values" in pg and _momentum_initialized_flags(pg)[i]:
                return _ParamState(self, param, present())
            return default
        if kind == "rmsprop" and "values" in pg:
            return _ParamState(self, param, {
                **present(), "step": float(steps[i])})
        if kind == "adan":
            return _ParamState(self, param, {
                **present(), "step": float(steps[i])})
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

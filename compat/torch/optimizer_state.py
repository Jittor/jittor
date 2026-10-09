"""Optimizer state views and lazily allocated Adam moments."""
from collections.abc import Mapping
import jittor as jt
from jittor.optim.base import _state_buffer
from .context import get_install_context
from .. import optimizer_kinds as _optimizer_kinds

def _reset_adam_state_to_lazy(opt):
    """Leave moments unallocated until a group's first actual update."""
    for pg in opt.param_groups:
        size = len(pg.get("params", ()))
        pg["m"] = [None] * size
        pg["values"] = [None] * size


def _ensure_adam_group_state(pg):
    """Create moments for the current parameter objects after partitioning."""
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
                with tensor_frontend(tensor_type, like=param):
                    buffers[index] = _state_buffer(param)


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
        pg, _ = self._find(param)
        if pg is None:
            raise KeyError(param)
        r = self.get(param, None)
        # torch.optim.Optimizer.state is a defaultdict(dict): ZeRO reads a
        # registered partition's empty state before its first optimizer step.
        return _ParamState(self, param, {}) if r is None else r
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

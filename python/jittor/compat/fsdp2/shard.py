"""FSDP2 parameter metadata, sharding, and forward lifecycle."""

import types

import jittor as jt
from jittor import nn

from . import common, dtensor
from ..diagnostics import EXPECTED, swallowed


_EXPORTS = (
    "_flat_local_overlap",
    "_flat_entry_slices",
    "_refresh_flat_entry_shards",
    "_mark_fsdp_param_var",
    "_fsdp_param_entry",
    "is_fsdp_managed_param",
    "_fsdp_var_to_local",
    "_fsdp_var_full_tensor",
    "_fsdp_var_redistribute",
    "_named_parameters_with_owner",
    "_iter_modules",
    "_iter_fsdp_modules",
    "_apply_fsdp_attr",
    "_init_true_fsdp_state",
    "_unshard_module_params",
    "_reshard_module_params",
    "_release_full_params",
    "_execute_with_true_fsdp",
    "_install_true_fsdp_execute",
)


def _flat_local_overlap(state, entry):
    rank_start = int(state.true_fsdp_rank) * int(state.true_fsdp_flat_shard_numel)
    rank_end = rank_start + int(state.true_fsdp_flat_shard_numel)
    param_start = int(entry.flat_offset)
    param_end = param_start + int(entry.numel)
    overlap_start = max(rank_start, param_start)
    overlap_end = min(rank_end, param_end)
    if overlap_end <= overlap_start:
        return 0, 0
    return overlap_start - rank_start, overlap_end - overlap_start


def _flat_entry_slices(state, flat_var):
    out = []
    for entry in state.true_fsdp_params:
        local_start, local_len = _flat_local_overlap(state, entry)
        out.append(common._slice_flat(flat_var, local_start, local_len))
    return out


def _refresh_flat_entry_shards(state):
    for entry, shard in zip(state.true_fsdp_params,
                            _flat_entry_slices(
                                state, state.true_fsdp_flat_shard)):
        entry.shard = shard
        entry.shard_numel = int(shard.shape[0]) if len(shard.shape) else int(entry.numel)
        _mark_fsdp_param_var(shard, state, entry, "shard")
        if getattr(entry, "requires_grad", True):
            if shard.is_stop_grad():
                shard.start_grad()
        elif not shard.is_stop_grad():
            shard.stop_grad()


def _mark_fsdp_param_var(var, state, entry, role):
    try:
        object.__setattr__(var, "_jittor_fsdp2_state", state)
        object.__setattr__(var, "_jittor_fsdp2_entry", entry)
        object.__setattr__(var, "_jittor_fsdp2_module", getattr(state, "true_fsdp_module", None))
        object.__setattr__(var, "_jittor_fsdp2_role", role)
        object.__setattr__(var, "_dtensor_device_mesh",
                           getattr(state, "mesh", None) or dtensor.DeviceMesh("cuda", (common._world_size(),)))
        object.__setattr__(var, "_dtensor_placements", (dtensor.Shard(0),))
        object.__setattr__(var, "device_mesh", getattr(var, "_dtensor_device_mesh"))
        object.__setattr__(var, "placements", getattr(var, "_dtensor_placements"))
        object.__setattr__(var, "_spec", types.SimpleNamespace(
            mesh=getattr(var, "_dtensor_device_mesh"),
            placements=getattr(var, "_dtensor_placements")))
        object.__setattr__(var, "_local_tensor", entry.shard if entry is not None else var)
        object.__setattr__(
            var, "to_local", types.MethodType(_fsdp_var_to_local, var))
        object.__setattr__(
            var, "full_tensor", types.MethodType(_fsdp_var_full_tensor, var))
        object.__setattr__(
            var, "redistribute", types.MethodType(_fsdp_var_redistribute, var))
    except EXPECTED as exc:
        swallowed("fsdp2/shard.py _mark_fsdp_param_var: object.__setattr__(var, '_jittor_fsdp2_state', state)", exc)
    return var


def _fsdp_param_entry(param):
    state = getattr(param, "_jittor_fsdp2_state", None)
    entry = getattr(param, "_jittor_fsdp2_entry", None)
    if state is None or entry is None:
        return None, None
    if not getattr(state, "true_fsdp_initialized", False):
        return None, None
    return state, entry


def is_fsdp_managed_param(param):
    state, entry = _fsdp_param_entry(param)
    return state is not None and entry is not None


def _fsdp_var_to_local(self, *args, **kwargs):
    state, entry = _fsdp_param_entry(self)
    if state is None or entry is None:
        return self
    return entry.shard


def _fsdp_var_full_tensor(self, *args, **kwargs):
    state, entry = _fsdp_param_entry(self)
    if state is None or entry is None:
        if getattr(self, "_jittor_fsdp2_role", None) == "flat_shard":
            return common._all_gather_shards(self)
        return self
    if getattr(state, "true_fsdp_unsharded", False) and getattr(entry, "full_param", None) is not None:
        return entry.full_param
    if getattr(state, "true_fsdp_flat", False):
        full_flat = common._all_gather_shards(state.true_fsdp_flat_shard)
        return common._slice_flat(full_flat, entry.flat_offset, entry.numel).reshape(entry.shape)
    gathered = common._all_gather_shards(entry.shard)
    full_flat = gathered if entry.padded_numel == entry.numel else common._slice_flat(gathered, 0, entry.numel)
    return full_flat.reshape(entry.shape)


def _fsdp_var_redistribute(self, device_mesh=None, placements=None, **kwargs):
    try:
        if device_mesh is not None:
            object.__setattr__(self, "_dtensor_device_mesh", device_mesh)
            object.__setattr__(self, "device_mesh", device_mesh)
        if placements is not None:
            object.__setattr__(self, "_dtensor_placements", tuple(placements))
            object.__setattr__(self, "placements", tuple(placements))
        object.__setattr__(self, "_spec", types.SimpleNamespace(
            mesh=getattr(self, "_dtensor_device_mesh", None),
            placements=getattr(self, "_dtensor_placements", ())))
    except EXPECTED as exc:
        swallowed("fsdp2/shard.py _fsdp_var_redistribute: if device_mesh is not None:", exc)
    return self


def _named_parameters_with_owner(module, recurse=True):
    out = []
    seen = set()

    def child_items(mod):
        try:
            items = mod.named_children()
            if items is not None:
                return list(items)
        except EXPECTED as exc:
            swallowed("fsdp2/shard.py child_items: items = mod.named_children()", exc)
        try:
            modules = getattr(mod, "_modules", None)
            if callable(modules):
                modules = modules()
            if isinstance(modules, dict):
                return list(modules.items())
        except EXPECTED as exc:
            swallowed("fsdp2/shard.py child_items: modules = getattr(mod, '_modules', None)", exc)
        return []

    def visit(mod, prefix=""):
        dc = getattr(mod, "__dict__", {})
        try:
            if isinstance(mod, nn.ParameterList):
                dc = mod.params
        except EXPECTED as exc:
            swallowed("fsdp2/shard.py visit: if isinstance(mod, nn.ParameterList):", exc)
        bufnames = getattr(mod, "__dict__", {}).get("_buffer_names", ())
        for name, value in list(dc.items()):
            if isinstance(name, str) and name.startswith("_"):
                continue
            if isinstance(value, jt.Var):
                if id(value) in seen:
                    continue
                if getattr(value, "is_buffer", False) or not getattr(value, "persistent", True) or name in bufnames:
                    continue
                seen.add(id(value))
                pname = f"{prefix}.{name}" if prefix else str(name)
                out.append((pname, mod, name, value))
        if recurse:
            for name, value in child_items(mod):
                if isinstance(value, nn.Module):
                    child_prefix = f"{prefix}.{name}" if prefix else str(name)
                    visit(value, child_prefix)

    visit(module)
    return out


def _iter_modules(module, recurse=True):
    if recurse and hasattr(module, "modules"):
        try:
            return list(module.modules())
        except EXPECTED as exc:
            swallowed("fsdp2/shard.py _iter_modules: return list(module.modules())", exc)
    return [module]


def _iter_fsdp_modules(module, recurse=True):
    return [m for m in _iter_modules(module, recurse)
            if getattr(m, "_is_fsdp_module", False)]


def _apply_fsdp_attr(module, name, value, recurse=True):
    targets = _iter_fsdp_modules(module, recurse) or [module]
    for m in targets:
        st = getattr(m, "_fsdp_state", None)
        if st is None:
            st = types.SimpleNamespace()
            object.__setattr__(m, "_fsdp_state", st)
        setattr(st, name, value)
    return module


def _init_true_fsdp_state(module, state):
    if getattr(state, "true_fsdp_initialized", False):
        return state
    state.true_fsdp_module = module
    if not common._in_true_distributed():
        state.true_fsdp_initialized = False
        return state
    ws = common._world_size()
    rank = common._rank()
    entries = []
    params = [
        item for item in _named_parameters_with_owner(module, recurse=True)
        if not is_fsdp_managed_param(item[3])
    ]
    total_numel = sum(common._param_numel(param) for _, _, _, param in params)
    if common._fsdp2_flat_enabled(ws, total_numel) and params and len({str(param.dtype) for _, _, _, param in params}) == 1:
        flat_shard_numel = common._ceil_div(total_numel, ws)
        flat_padded_numel = flat_shard_numel * ws
        flat_full = common._pad_flat(jt.concat([common._flatten_var(param) for _, _, _, param in params], dim=0),
                                     flat_padded_numel)
        flat_shard = common._slice_flat(flat_full, rank * flat_shard_numel, flat_shard_numel)
        flat_shard.sync()
        offset = 0
        for name, owner, attr, param in params:
            numel = common._param_numel(param)
            entries.append(types.SimpleNamespace(
                name=name,
                owner=owner,
                attr=attr,
                shape=tuple(int(x) for x in param.shape),
                dtype=param.dtype,
                numel=numel,
                padded_numel=numel,
                shard_numel=0,
                shard=None,
                full_param=None,
                flat_offset=offset,
                requires_grad=not param.is_stop_grad(),
            ))
            offset += numel
        state.true_fsdp_initialized = True
        state.true_fsdp_rank = rank
        state.true_fsdp_world_size = ws
        state.true_fsdp_params = entries
        state.true_fsdp_flat = True
        state.true_fsdp_flat_total_numel = total_numel
        state.true_fsdp_flat_padded_numel = flat_padded_numel
        state.true_fsdp_flat_shard_numel = flat_shard_numel
        state.true_fsdp_flat_shard = _mark_fsdp_param_var(
            flat_shard, state, None, "flat_shard")
        if any(entry.requires_grad for entry in entries):
            if state.true_fsdp_flat_shard.is_stop_grad():
                state.true_fsdp_flat_shard.start_grad()
        elif not state.true_fsdp_flat_shard.is_stop_grad():
            state.true_fsdp_flat_shard.stop_grad()
        _refresh_flat_entry_shards(state)
        for entry in entries:
            object.__setattr__(entry.owner, entry.attr, entry.shard)
        state.true_fsdp_unsharded = False
        return state

    state.true_fsdp_flat = False
    for name, owner, attr, param in params:
        numel = common._param_numel(param)
        shard_numel = common._ceil_div(numel, ws)
        padded_numel = shard_numel * ws
        flat_full = common._pad_flat(common._flatten_var(param), padded_numel)
        local = common._slice_flat(flat_full, rank * shard_numel, shard_numel)
        local.sync()
        entries.append(types.SimpleNamespace(
            name=name,
            owner=owner,
            attr=attr,
            shape=tuple(int(x) for x in param.shape),
            dtype=param.dtype,
            numel=numel,
            padded_numel=padded_numel,
            shard_numel=shard_numel,
            shard=local,
            full_param=None,
            requires_grad=not param.is_stop_grad(),
        ))
        _mark_fsdp_param_var(local, state, entries[-1], "shard")
        if entries[-1].requires_grad:
            if local.is_stop_grad():
                local.start_grad()
        elif not local.is_stop_grad():
            local.stop_grad()
        object.__setattr__(owner, attr, local)
    state.true_fsdp_initialized = True
    state.true_fsdp_rank = rank
    state.true_fsdp_world_size = ws
    state.true_fsdp_params = entries
    state.true_fsdp_unsharded = False
    return state


def _unshard_module_params(module):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return module
    if getattr(state, "true_fsdp_unsharded", False):
        return module
    if getattr(state, "true_fsdp_flat", False):
        full_flat = common._all_gather_shards(state.true_fsdp_flat_shard)
        state.true_fsdp_flat_full_param = full_flat
        for entry in state.true_fsdp_params:
            full = common._slice_flat(full_flat, entry.flat_offset, entry.numel).reshape(entry.shape)
            entry.full_param = full
            _mark_fsdp_param_var(full, state, entry, "full")
            if getattr(entry, "requires_grad", True):
                if full.is_stop_grad():
                    full.start_grad()
            elif not full.is_stop_grad():
                full.stop_grad()
            object.__setattr__(entry.owner, entry.attr, full)
    else:
        for entry in state.true_fsdp_params:
            gathered = common._all_gather_shards(entry.shard)
            full_flat = gathered if entry.padded_numel == entry.numel else common._slice_flat(gathered, 0, entry.numel)
            full = full_flat.reshape(entry.shape)
            entry.full_param = full
            _mark_fsdp_param_var(full, state, entry, "full")
            if getattr(entry, "requires_grad", True):
                if full.is_stop_grad():
                    full.start_grad()
            elif not full.is_stop_grad():
                full.stop_grad()
            object.__setattr__(entry.owner, entry.attr, full)
    state.true_fsdp_unsharded = True
    return module


def _reshard_module_params(module):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return module
    if not getattr(state, "true_fsdp_unsharded", False):
        return module
    for entry in state.true_fsdp_params:
        object.__setattr__(entry.owner, entry.attr, entry.shard)
        # Keep the full Var from the just-finished forward alive for
        # sync_sharded_grads(loss): Jittor's autograd needs the exact Var object
        # that participated in the forward graph.
    state.true_fsdp_unsharded = False
    return module


#: Attribute on the FSDP state counting how deep we are inside this module's
#: own forward. See :func:`_execute_with_true_fsdp`.
_EXECUTE_DEPTH_ATTR = "true_fsdp_execute_depth"


def _release_full_params(state):
    """Drop the gathered full parameters once their gradients are sharded.

    This is what makes FSDP save memory. Three things each held a full-size
    copy after every step, and none of them was ever cleared:

    * ``entry.full_param`` -- the gathered parameter. ``_reshard_module_params``
      deliberately keeps it, because the backward needs the exact Var that took
      part in the forward graph. That is true *until the gradients have been
      reduce-scattered into the shards*; after that nobody needs it, and the
      next forward gathers a fresh one.
    * ``entry.full_public_grad`` -- a full-size gradient that
      ``refresh_visible_full_grads`` re-materialises on every ``zero_grad()``
      so that ``.grad`` on a full parameter reads correctly. With the full
      parameter gone it has nothing to attach to, and stops being made.
    * ``state.true_fsdp_flat_full_param`` -- the flat gathered buffer, written
      in ``_unshard_module_params`` and cleared nowhere in the tree.

    Measured on 2 ranks, 64 MB of parameters (``nn.Linear(2048, 2048)`` x4):
    sharding correctly halved the resident parameters (base 129 -> 97 MB) and
    then drove the steady state to 513 MB, against 209 MB for the same model
    not sharded. FSDP was not merely failing to save memory, it cost 2.5x.

    Not called while the parameters are legitimately still gathered:
    ``reshard_after_forward=False`` asks for exactly that, and the module
    attributes point at the full Vars.
    """
    if getattr(state, "true_fsdp_unsharded", False):
        return state
    for entry in getattr(state, "true_fsdp_params", ()):
        entry.full_param = None
        entry.full_public_grad = None
    state.true_fsdp_flat_full_param = None
    return state


def _execute_with_true_fsdp(module, orig_execute, *args, **kwargs):
    """Run one forward with this module's parameters unsharded.

    Two hooks call this for the same forward, and they are both needed because
    they cover different dispatch paths:

    * ``Module.__call__`` (installed once, in ``compat/torch/installers/nn.py``)
      is the only one that sees a torch-style ``forward`` override, a
      per-instance ``self.forward``, or the fused RMSNorm shortcut -- none of
      which reach ``execute``;
    * the per-instance ``execute`` wrapper (installed by ``fully_shard`` in
      :func:`_install_true_fsdp_execute`) is the only one that sees a direct
      ``module.execute(...)`` call that bypasses ``__call__``.

    On the ordinary jittor-style path both fire, nested. Unsharding and
    resharding are individually idempotent, so the answer was never wrong, but
    the inner hook resharded on the way out and the outer one then re-entered
    the whole guard sequence for a second time on every single forward.

    The depth counter makes the pair behave as one hook: whichever fires first
    owns the unshard/reshard for that forward, and any nested call is a plain
    pass-through. That also fixes the case the idempotence was quietly
    covering: with ``reshard_after_forward`` true, the inner hook used to put
    the shards back before the outer hook's ``finally`` had run, so the window
    in which the module's parameters were unsharded ended in the middle of the
    outer hook rather than at its end.
    """
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return orig_execute(*args, **kwargs)
    depth = getattr(state, _EXECUTE_DEPTH_ATTR, 0)
    if depth:
        return orig_execute(*args, **kwargs)
    setattr(state, _EXECUTE_DEPTH_ATTR, depth + 1)
    try:
        _unshard_module_params(module)
        try:
            return orig_execute(*args, **kwargs)
        finally:
            if getattr(state, "reshard_after_forward", True):
                _reshard_module_params(module)
    finally:
        setattr(state, _EXECUTE_DEPTH_ATTR, depth)



def _install_true_fsdp_execute(module):
    state = getattr(module, "_fsdp_state", None)
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return module
    if getattr(module, "_fsdp_orig_execute", None) is not None:
        return module
    orig_execute = getattr(module, "execute", None)
    if not callable(orig_execute):
        return module
    object.__setattr__(module, "_fsdp_orig_execute", orig_execute)

    def _wrapped_execute(self, *args, **kwargs):
        return _execute_with_true_fsdp(
            self, self._fsdp_orig_execute, *args, **kwargs)

    object.__setattr__(module, "execute", types.MethodType(_wrapped_execute, module))
    return module

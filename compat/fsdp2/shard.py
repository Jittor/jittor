"""FSDP2 parameter metadata, sharding, and forward lifecycle."""

import contextlib
import types
import sys

from jittor.distributed.fsdp import storage as _storage
import weakref

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor import nn

from . import common, dtensor
from ..torch.parameter_containers import ParameterDictAdapter, ParameterListAdapter
from ..torch.tensor_state import get_tensor_state
from ..diagnostics import EXPECTED, swallowed


_EXPORTS = (
    "_flat_local_overlap",
    "_flat_entry_slices",
    "_refresh_flat_entry_shards",
    "_mark_fsdp_param_var",
    "_fsdp_param_entry",
    "_fsdp_norm_grad",
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


def _materialize_initial_shard(shard):
    return _storage._materialize_initial_shard(shard)


def _flat_local_overlap(state, entry):
    return _storage._flat_local_overlap(state, entry)


def _flat_entry_slices(state, flat_var):
    return _storage._flat_entry_slices(state, flat_var, api=sys.modules[__name__])


@common._state_frontend
def _refresh_flat_entry_shards(state):
    result = _storage._refresh_flat_entry_shards(state, api=sys.modules[__name__])
    for entry in state.true_fsdp_params:
        get_tensor_state(jt).leaf_params[id(entry.shard)] = entry.shard
    _bind_parameter_aliases(state, "shard")
    return result


def _bind_entry_aliases(entry, value):
    _storage._set_module_parameter(entry, value)


def _bind_parameter_aliases(state, role):
    for entry in getattr(state, "true_fsdp_params", ()):
        value = entry.full_param if role == "full" else entry.shard
        _bind_entry_aliases(entry, value)


class _ShardTensorMethod:
    """Resolve the current shard through weak metadata, never retain its Var."""

    def __init__(self, function, state, entry, role):
        self.function = function
        self.state = weakref.ref(state)
        self.entry = weakref.ref(entry) if entry is not None else None
        self.role = role

    def _resolve_tensor(self, state, entry):
        if entry is None:
            return state.true_fsdp_flat_shard
        if self.role == "grad_shard":
            for current, gradient in zip(
                state.true_fsdp_params, getattr(state, "true_fsdp_last_grads", ())
            ):
                if current is entry:
                    return gradient
            gradient = getattr(entry.shard, "_torch_grad", None)
            if isinstance(gradient, jt.Var):
                return gradient
            raise ReferenceError("the FSDP gradient has been released")
        if self.role == "full" and getattr(entry, "full_param", None) is not None:
            return entry.full_param
        return entry.shard

    def __call__(self, *args, **kwargs):
        state = self.state()
        entry = self.entry() if self.entry is not None else None
        if state is None or self.entry is not None and entry is None:
            raise ReferenceError("the FSDP parameter state has been released")
        tensor = self._resolve_tensor(state, entry)
        return self.function(tensor, *args, **kwargs)


def _fsdp_var_size(var, dim=None):
    """A sharded parameter's Torch-facing size is its global parameter size."""
    entry = getattr(var, "_jittor_fsdp2_entry", None)
    shape = tuple(getattr(entry, "shape", tuple(var.shape)))
    if dim is None:
        return shape
    dim = int(dim)
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        raise IndexError("dimension out of range")
    return shape[dim]


def _parameter_requires_grad(param):
    return _storage._parameter_requires_grad(param)


def _mark_fsdp_param_var(var, state, entry, role):
    try:
        object.__setattr__(var, "_jittor_fsdp2_state", state)
        object.__setattr__(var, "_jittor_fsdp2_entry", entry)
        object.__setattr__(var, "_jittor_fsdp2_module", getattr(state, "true_fsdp_module", None))
        object.__setattr__(var, "_jittor_fsdp2_role", role)
        object.__setattr__(
            var, "size", _ShardTensorMethod(_fsdp_var_size, state, entry, role)
        )
        object.__setattr__(
            var,
            "_dtensor_device_mesh",
            getattr(state, "mesh", None)
            or dtensor.init_device_mesh("cuda", (common._world_size(),)),
        )
        object.__setattr__(var, "_dtensor_placements", (dtensor.Shard(0),))
        object.__setattr__(var, "device_mesh", getattr(var, "_dtensor_device_mesh"))
        object.__setattr__(var, "placements", getattr(var, "_dtensor_placements"))
        object.__setattr__(
            var,
            "_spec",
            types.SimpleNamespace(
                mesh=getattr(var, "_dtensor_device_mesh"),
                placements=getattr(var, "_dtensor_placements"),
            ),
        )
        # Resolve through weak metadata. Bound native methods retain each
        # temporary Var and leak every superseded all-gathered parameter.
        object.__setattr__(
            var, "to_local", _ShardTensorMethod(_fsdp_var_to_local, state, entry, role)
        )
        object.__setattr__(
            var, "full_tensor", _ShardTensorMethod(_fsdp_var_full_tensor, state, entry, role)
        )
        object.__setattr__(
            var, "redistribute", _ShardTensorMethod(_fsdp_var_redistribute, state, entry, role)
        )
        getattr(var, "__dict__", {}).pop("_local_tensor", None)
    except EXPECTED as exc:
        swallowed(
            "fsdp2/shard.py _mark_fsdp_param_var: object.__setattr__(var, '_jittor_fsdp2_state', state)",
            exc,
        )
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


def _fsdp_norm_grad(param, public_gradient=None):
    """Return this rank's current shard of an FSDP gradient for norm clipping.

    Trainer clipping runs before the optimizer adapter localizes the public
    full-parameter gradient into shard-shaped optimizer gradients. Prefer that
    current public gradient, then project it with the same shard slicing helper
    used by the optimizer adapter. Internal last-gradient slots can be full on
    one rank and empty on another at this point.
    """
    state, entry = _fsdp_param_entry(param)
    if state is None or entry is None:
        return None

    if not isinstance(public_gradient, jt.Var):
        public_gradient = getattr(param, "_torch_grad", None)
    if not isinstance(public_gradient, jt.Var):
        full = getattr(entry, "full_param", None)
        public_gradient = getattr(full, "_torch_grad", None) if full is not None else None
    if not isinstance(public_gradient, jt.Var):
        public_gradient = getattr(param, "grad", None)

    local_numel = int(entry.shard.numel())
    local_shape = list(entry.shard.shape)
    full_numel = int(entry.numel)

    def as_local_shard(gradient):
        if not isinstance(gradient, jt.Var):
            return None
        if int(gradient.numel()) == local_numel and list(gradient.shape) == local_shape:
            local = gradient
        elif int(gradient.numel()) == full_numel:
            from . import grad_sync
            local = grad_sync._local_grad_from_visible_full(state, entry, gradient)
            if int(local.numel()) != local_numel:
                return None
            local = local.stop_grad()
        else:
            return None
        object.__setattr__(
            local, "_fsdp_norm_group", getattr(state, "shard_group", None)
        )
        return local

    # The public gradient is the newest value. If it is a full logical
    # parameter, produce exactly the local slice Torch exposes through
    # DTensor.grad.to_local().
    local = as_local_shard(public_gradient)
    if local is not None:
        return local

    current_grads = getattr(state, "true_fsdp_last_grads", None)
    if isinstance(current_grads, (list, tuple)):
        for current, gradient in zip(state.true_fsdp_params, current_grads):
            if current is entry:
                local = as_local_shard(gradient)
                if local is not None:
                    return local
                break
    for gradient in (
        getattr(entry, "last_grad", None),
        getattr(entry.shard, "_torch_grad", None),
    ):
        local = as_local_shard(gradient)
        if local is not None:
            return local
    return None


def _fsdp_var_to_local(self, *args, **kwargs):
    state, entry = _fsdp_param_entry(self)
    if state is None or entry is None:
        return self
    if getattr(self, "_jittor_fsdp2_role", None) == "grad_shard":
        return self
    return entry.shard


def _fsdp_var_full_tensor(self, *args, **kwargs):
    raw_state = getattr(self, "_jittor_fsdp2_state", None)
    raw_entry = getattr(self, "_jittor_fsdp2_entry", None)
    if (
        raw_state is not None
        and raw_entry is None
        and getattr(raw_state, "true_fsdp_initialized", False)
        and getattr(self, "_jittor_fsdp2_role", None) == "flat_shard"
    ):
        return common._all_gather_shards(self, getattr(raw_state, "shard_group", None))
    state, entry = _fsdp_param_entry(self)
    if state is None or entry is None:
        return self
    if getattr(self, "_jittor_fsdp2_role", None) == "grad_shard":
        return common._full_gradient_from_shard(self, state, entry)
    if (
        getattr(state, "true_fsdp_unsharded", False)
        and getattr(entry, "full_param", None) is not None
    ):
        return entry.full_param
    if getattr(state, "true_fsdp_flat", False):
        full_flat = common._all_gather_shards(
            state.true_fsdp_flat_shard, getattr(state, "shard_group", None)
        )
        return common._slice_flat(full_flat, entry.flat_offset, entry.numel).reshape(entry.shape)
    gathered = common._all_gather_shards(entry.shard, getattr(state, "shard_group", None))
    full_flat = (
        gathered
        if entry.padded_numel == entry.numel
        else common._slice_flat(gathered, 0, entry.numel)
    )
    return full_flat.reshape(entry.shape)


def _fsdp_var_redistribute(self, device_mesh=None, placements=None, **kwargs):
    try:
        if device_mesh is not None:
            object.__setattr__(self, "_dtensor_device_mesh", device_mesh)
            object.__setattr__(self, "device_mesh", device_mesh)
        if placements is not None:
            object.__setattr__(self, "_dtensor_placements", tuple(placements))
            object.__setattr__(self, "placements", tuple(placements))
        object.__setattr__(
            self,
            "_spec",
            types.SimpleNamespace(
                mesh=getattr(self, "_dtensor_device_mesh", None),
                placements=getattr(self, "_dtensor_placements", ()),
            ),
        )
    except EXPECTED as exc:
        swallowed("fsdp2/shard.py _fsdp_var_redistribute: if device_mesh is not None:", exc)
    return self


def _named_parameters_with_owner(module, recurse=True, remove_duplicate=True):
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
        for name, value, role in mod._var_roles():
            if role != "parameter" or (remove_duplicate and id(value) in seen):
                continue
            seen.add(id(value))
            pname = f"{prefix}.{name}" if prefix else str(name)
            out.append((pname, mod, name, value))
        if recurse:
            for name, value in child_items(mod):
                if isinstance(value, nn.Module):
                    child_prefix = f"{prefix}.{name}" if prefix else str(name)
                    visit(value, child_prefix)

    try:
        visit(module)
    finally:
        # The recursive visitor captures itself and out. Break that cycle now
        # so superseded Parameters do not wait for cyclic GC to be released.
        visit = None
    return out


def _parameter_aliases(module, recurse=True):
    aliases = {}
    visited = set()

    def visit(mod):
        if id(mod) in visited:
            return
        visited.add(id(mod))
        for name, value, role in mod._var_roles():
            if role == "parameter":
                aliases.setdefault(id(value), []).append((mod, name))
        if recurse:
            for _, child in child_items(mod):
                if isinstance(child, nn.Module):
                    visit(child)

    def child_items(mod):
        try:
            items = mod.named_children()
            if items is not None:
                return list(items)
        except EXPECTED as exc:
            swallowed("fsdp2/shard.py _parameter_aliases: items = mod.named_children()", exc)
        modules = getattr(mod, "_modules", None)
        if callable(modules):
            modules = modules()
        return list(modules.items()) if isinstance(modules, dict) else []

    visit(module)
    return aliases


def _iter_modules(module, recurse=True):
    if recurse and hasattr(module, "modules"):
        try:
            return list(module.modules())
        except EXPECTED as exc:
            swallowed("fsdp2/shard.py _iter_modules: return list(module.modules())", exc)
    return [module]


def _iter_fsdp_modules(module, recurse=True):
    return [m for m in _iter_modules(module, recurse) if getattr(m, "_is_fsdp_module", False)]


def _apply_fsdp_attr(module, name, value, recurse=True):
    targets = _iter_fsdp_modules(module, recurse) or [module]
    for m in targets:
        st = getattr(m, "_fsdp_state", None)
        if st is None:
            st = common.StateRecord()
            object.__setattr__(m, "_fsdp_state", st)
        setattr(st, name, value)
    return module


def _init_true_fsdp_state(module, state):
    tensor_types = {
        getattr(type(param), "_frontend_result_type", type(param))
        for param in module.parameters()
        if getattr(type(param), "_frontend_backend", None) is not None
    }
    if len(tensor_types) > 1:
        raise TypeError("FSDP parameters must use one tensor frontend")
    state.frontend_type = next(iter(tensor_types), None)
    with common._frontend_scope(state):
        return _init_true_fsdp_state_impl(module, state)


def _init_true_fsdp_state_impl(module, state):
    if getattr(state, "true_fsdp_initialized", False):
        return state
    state.true_fsdp_module = module
    group = getattr(state, "shard_group", None)
    if not common._can_initialize_fsdp(group):
        state.true_fsdp_initialized = False
        return state
    ws = common._world_size() if group is None else group.size()
    rank = common._rank() if group is None else group.rank()
    ignored_ids = {id(param) for param in getattr(state, "ignored_params", ())}
    params = [
        item
        for item in _named_parameters_with_owner(module, recurse=True)
        if not is_fsdp_managed_param(item[3]) and id(item[3]) not in ignored_ids
    ]
    aliases = _parameter_aliases(module, recurse=True)
    for _, owner, _, _ in params:
        if isinstance(owner, (nn.ParameterList, ParameterListAdapter, ParameterDictAdapter)):
            raise NotImplementedError(
                "FSDP2 cannot shard ParameterList/ParameterDict: its parameter "
                "storage needs container-aware replacement at every shard transition"
            )
    total_numel = sum(common._param_numel(param) for _, _, _, param in params)
    initialized = _storage._init_true_fsdp_state_impl(
        module,
        state,
        params,
        world_size=ws,
        rank=rank,
        flat=common._fsdp2_flat_enabled(ws, total_numel),
        api=sys.modules[__name__],
    )
    # Optimizers may already own the full parameters. Give those exact objects
    # the same entry identity as the new shard, without retaining them in state.
    for (_, _, _, original), entry in zip(params, state.true_fsdp_params):
        _mark_fsdp_param_var(original, state, entry, "retired")
        # Public backward is independent of whether an optimizer exists. Index
        # the current holder, with its explicit entry metadata, in the existing
        # weak leaf registry; never retain a full parameter or a retired alias.
        get_tensor_state(jt).leaf_params[id(entry.shard)] = entry.shard
    for entry, (_, owner, attr, param) in zip(state.true_fsdp_params, params):
        entry.aliases = aliases.get(id(param), ((owner, attr),))
    _bind_parameter_aliases(state, "shard")
    return initialized


def _unshard_module_params(module):
    with common._frontend_scope(getattr(module, "_fsdp_state", None)):
        return _unshard_module_params_impl(module)


def _compute_shard(state, master_shard):
    return _storage._compute_shard(state, master_shard)


def _cast_floating_tree(value, target):
    return _storage._cast_floating_tree(value, target)



def _fsdp_forward_amp_scope(policy):
    """Apply the FSDP compute dtype to floating ops inside its forward.

    Parameters that are outside this FSDP state (for example, root-owned
    weights around an auto-wrapped transformer layer) can remain FP32 while
    their inputs have already been cast to the policy dtype. Mirror the
    forward compute preference for those mixed operands, without changing the
    optimizer's FP32 master shards or leaking the flag past this module call.
    """
    if not getattr(policy, "cast_forward_inputs", True):
        return contextlib.nullcontext()
    param_dtype = getattr(policy, "param_dtype", None)
    target = _jittor_dtype_name(param_dtype) if param_dtype is not None else None
    if target == "bfloat16":
        requested = jt.amp_flags.prefer_bfloat16
    elif target == "float16":
        requested = jt.amp_flags.prefer16
    else:
        return contextlib.nullcontext()
    precision_mask = (
        jt.amp_flags.prefer32
        | jt.amp_flags.prefer16
        | jt.amp_flags.prefer_bfloat16
    )
    current = int(jt.flags.amp_reg)
    amp_reg = (current & ~precision_mask) | requested
    from ..torch.amp import _fsdp_mixed_precision_scope
    return _fsdp_mixed_precision_scope(amp_reg)


def _unshard_module_params_impl(module):
    result = _storage._unshard_module_params_impl(module, api=sys.modules[__name__])
    state = getattr(module, "_fsdp_state", None)
    if state is not None and getattr(state, "true_fsdp_unsharded", False):
        _bind_parameter_aliases(state, "full")
    return result


def _reshard_module_params(module):
    result = _storage._reshard_module_params(module)
    state = getattr(module, "_fsdp_state", None)
    if state is not None and not getattr(state, "true_fsdp_unsharded", False):
        _bind_parameter_aliases(state, "shard")
    return result


#: Attribute on the FSDP state counting how deep we are inside this module's
#: own forward. See :func:`_execute_with_true_fsdp`.
_EXECUTE_DEPTH_ATTR = "true_fsdp_execute_depth"


def _release_full_params(state):
    return _storage._release_full_params(state)


def _execute_with_true_fsdp(module, orig_execute, *args, **kwargs):
    return _storage._execute_with_true_fsdp(
        module, orig_execute, *args, api=sys.modules[__name__], **kwargs
    )


def _wrapped_execute(self, *args, **kwargs):
    return _execute_with_true_fsdp(self, self._fsdp_orig_execute, *args, **kwargs)


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

    object.__setattr__(module, "execute", types.MethodType(_wrapped_execute, module))
    return module

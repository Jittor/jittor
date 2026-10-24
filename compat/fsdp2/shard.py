"""FSDP2 parameter metadata, sharding, and forward lifecycle."""

import types
import sys

from jittor.distributed.fsdp import storage as _storage
import weakref

import jittor as jt
from jittor import nn
from jittor._core.module import module_protocol

from . import common, dtensor
from . import _resident_backward as _resident
from ..torch.parameter_containers import ParameterDictAdapter, ParameterListAdapter
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


def _materialize_initial_shard(shard):
    return _storage._materialize_initial_shard(shard)


def _flat_local_overlap(state, entry):
    return _storage._flat_local_overlap(state, entry)


def _flat_entry_slices(state, flat_var):
    # CANN 9's SliceV2 is off by one for some sufficiently large tail slices.
    # Flat FSDP shards are already a contiguous partition, so on ACL express
    # every non-empty parameter view with one native SplitWithSize operation.
    # Besides avoiding that backend defect, one split describes the ownership
    # invariant more directly than hundreds of independent slices. These are
    # storage publications; FSDP constructs/synchronizes gradients explicitly,
    # so the inference-only ACL split kernel has the required semantics.
    if getattr(getattr(jt, "compiler", None), "has_acl", False):
        backend, _device_id = jt.core.dispatch_context([flat_var])
        if backend in ("acl", "acl_legacy"):
            overlaps = [_flat_local_overlap(state, entry)
                        for entry in state.true_fsdp_params]
            cursor = 0
            sections = []
            for local_start, local_len in overlaps:
                if not local_len:
                    continue
                if int(local_start) != cursor:
                    raise RuntimeError(
                        "non-contiguous ACL flat FSDP entry layout at %s, expected %s" %
                        (local_start, cursor)
                    )
                sections.append(int(local_len))
                cursor += int(local_len)
            total = int(flat_var.shape[0])
            if cursor > total:
                raise RuntimeError("ACL flat FSDP entry layout exceeds local shard")
            if cursor < total:
                sections.append(total - cursor)  # rank-tail padding, not a parameter
            with jt.flag_scope(no_grad=1):
                pieces = (tuple(jt.split(flat_var, sections, dim=0))
                          if len(sections) > 1 else (flat_var,))
            out, piece_index = [], 0
            for _local_start, local_len in overlaps:
                if local_len:
                    out.append(pieces[piece_index])
                    piece_index += 1
                else:
                    # Do not share one empty Var: each entry receives role and
                    # owner metadata below.
                    out.append(common._zeros_like_shape(flat_var, (0,)))
            return out
    return _storage._flat_entry_slices(state, flat_var, api=sys.modules[__name__])


@common._state_frontend
def _refresh_flat_entry_shards(state):
    return _storage._refresh_flat_entry_shards(state, api=sys.modules[__name__])


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
            # Public .grad is the accumulated (and possibly clipped) optimizer
            # slot. The last reduce-scatter is only one backward contribution;
            # resolving it first discards earlier microbatches at to_local()/
            # full_tensor() without changing the actual optimizer gradient.
            gradient = getattr(entry.shard, "_torch_grad", None)
            if jt.core._is_tensor_carrier(gradient):
                return gradient
            for current, gradient in zip(
                state.true_fsdp_params, getattr(state, "true_fsdp_last_grads", ())
            ):
                if current is entry:
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


def _parameter_requires_grad(param):
    return _storage._parameter_requires_grad(param)


def _mark_fsdp_param_var(var, state, entry, role):
    if (role == "grad_shard" and getattr(state, "frontend_type", None) is not None
            and getattr(state, "true_fsdp_world_size", None) == 1):
        from .owned_dtensor import owned_dtensor_type
        carrier = owned_dtensor_type()
        if type(var) is not carrier:
            var = carrier.from_local(var, getattr(state, "mesh", None), (dtensor.Shard(0),))
    try:
        object.__setattr__(var, "_jittor_fsdp2_state", state)
        object.__setattr__(var, "_jittor_fsdp2_entry", entry)
        object.__setattr__(var, "_jittor_fsdp2_module", getattr(state, "true_fsdp_module", None))
        object.__setattr__(var, "_jittor_fsdp2_role", role)
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


def _world_one_plain_local(state, entry):
    """Make a real native storage view, not a Parameter/DTensor projection."""
    with common._frontend_scope(state):
        local = jt.view(entry.shard, tuple(entry.shard.shape))
        if not _parameter_requires_grad(entry.shard):
            local.stop_grad()
    if local is entry.shard or dtensor.is_dtensor(local):
        raise RuntimeError("FSDP world-one local view must be a distinct plain tensor")
    return local


def _fsdp_var_to_local(self, *args, **kwargs):
    state, entry = _fsdp_param_entry(self)
    if state is None or entry is None:
        return self
    if getattr(self, "_jittor_fsdp2_role", None) == "grad_shard":
        if (getattr(state, "frontend_type", None) is not None
                and getattr(state, "true_fsdp_world_size", None) == 1
                and not getattr(state, "true_fsdp_flat", False)
                and not _parameter_requires_grad(self)):
            # A first-order shard gradient owns a real plain local alias, just
            # as the canonical Parameter owns its local storage. Cache on the
            # gradient holder, not the entry: a replacement grad has new storage.
            local = getattr(self, "_stored_grad_local", None)
            if local is None:
                with common._frontend_scope(state):
                    from .owned_dtensor import plain_alias
                    local = plain_alias(self)
                if local is self or dtensor.is_dtensor(local):
                    raise RuntimeError("FSDP gradient local must be a distinct plain tensor")
                object.__setattr__(self, "_stored_grad_local", local)
            return local
        return self
    if (getattr(state, "frontend_type", None) is not None
            and getattr(state, "true_fsdp_world_size", None) == 1
            and not getattr(state, "true_fsdp_flat", False)):
        # The global Parameter remains canonical. The stopped local is cached
        # separately. Same-layout frontend optimizer writes preserve its
        # shared allocation rather than rebinding the Parameter holder.
        if bool(jt.flags.no_grad):
            local = getattr(entry, "_stored_local", None)
            if local is None:
                # Native DTensor returns its persistent plain local here,
                # not an autograd view constructed under the caller's no_grad.
                # Keep shared storage/version without a no-grad view history.
                with common._frontend_scope(state):
                    from .owned_dtensor import plain_alias
                    local = plain_alias(entry.shard)
                if local is entry.shard or dtensor.is_dtensor(local):
                    raise RuntimeError("FSDP stored local must be a distinct plain tensor")
                entry._stored_local = local
            return local
        return _world_one_plain_local(state, entry)
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
                if module_protocol(mod).is_module(value):
                    child_prefix = f"{prefix}.{name}" if prefix else str(name)
                    visit(value, child_prefix)

    visit(module)
    return out


def _entry_owners(entry):
    return getattr(entry, "aliases", ((entry.owner, entry.attr),))


def _bind_entry(entry, value):
    """Publish one logical parameter at every location which aliases it."""
    for owner, attr in _entry_owners(entry):
        object.__setattr__(owner, attr, value)


def _collect_owned_parameters(module, ignored_ids=()):
    """Deduplicate storage, not its owning module attributes.

    A shared parameter crossing FSDP units needs a coordinated lifecycle which
    is not implemented. Remember replaced originals weakly so wrapping a
    sibling/parent cannot silently shard a stale second reference separately.
    """
    params, aliases = [], {}
    module_ids = {id(child) for child in _iter_modules(module)}
    for item in _named_parameters_with_owner(module, remove_duplicate=False):
        _name, owner, attr, param = item
        if id(param) in ignored_ids:
            continue
        managed_state, entry = _fsdp_param_entry(param)
        original_owner = getattr(param, "_fsdp_original_owner", None)
        if managed_state is not None:
            if id(managed_state.true_fsdp_module) not in module_ids:
                raise NotImplementedError("shared parameters across FSDP units are unsupported")
            if not any(owner is parent and attr == name
                       for parent, name in _entry_owners(entry)):
                raise NotImplementedError("shared parameters across FSDP units are unsupported")
            continue
        if original_owner is not None and original_owner() is not None:
            raise NotImplementedError("shared parameters across FSDP units are unsupported")
        identity = id(param)
        if identity not in aliases:
            params.append(item)
            aliases[identity] = []
        if not any(owner is parent and attr == name for parent, name in aliases[identity]):
            aliases[identity].append((owner, attr))
    return params, aliases


def _remember_original_owner(param, state):
    object.__setattr__(param, "_fsdp_original_owner", weakref.ref(state))


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
    if not common._in_true_distributed() and common._world_size() > 1:
        state.true_fsdp_initialized = False
        return state
    group = getattr(state, "shard_group", None)
    ws = common._world_size() if group is None else group.size()
    rank = common._rank() if group is None else group.rank()
    ignored_ids = {id(param) for param in getattr(state, "ignored_params", ())}
    params, aliases = _collect_owned_parameters(module, ignored_ids)
    for _, owner, _, _ in params:
        if isinstance(owner, (nn.ParameterList, ParameterListAdapter, ParameterDictAdapter)):
            raise NotImplementedError(
                "FSDP2 cannot shard ParameterList/ParameterDict: its parameter "
                "storage needs container-aware replacement at every shard transition")
    total_numel = sum(common._param_numel(param) for _, _, _, param in params)
    if ws == 1:
        # Torch FSDP publishes a new canonical Parameter even at world size 1.
        # Preserve the native no-frontend identity branch below.
        parameter_type = None
        frontend_type = getattr(state, "frontend_type", None)
        if frontend_type is not None:
            from ..torch.tensor_state import compatibility_owner
            torch_owner = compatibility_owner(jt)
            if torch_owner.Var is not frontend_type:
                raise TypeError("FSDP parameter frontend differs from its Torch owner")
            parameter_type = torch_owner.nn.Parameter
        entries = []
        for name, owner, attr, param in params:
            numel = common._param_numel(param)
            requires_grad = _parameter_requires_grad(param)
            canonical = param
            if parameter_type is not None:
                from .owned_dtensor import owned_dtensor_type
                local = _materialize_initial_shard(param)
                distributed = owned_dtensor_type().from_local(
                    local, getattr(state, "mesh", None), (dtensor.Shard(0),))
                canonical = parameter_type(distributed,
                                           requires_grad=requires_grad)
            entry = common.StateRecord(name=name, owner=owner, attr=attr,
                aliases=tuple(aliases[id(param)]), shape=tuple(int(x) for x in param.shape),
                dtype=param.dtype, numel=numel, padded_numel=numel, shard_numel=numel,
                shard=canonical, full_param=None, requires_grad=requires_grad)
            entries.append(entry)
            _mark_fsdp_param_var(canonical, state, entry, "shard")
            _remember_original_owner(param, state)
            _bind_entry(entry, canonical)
        state.true_fsdp_initialized = True
        state.true_fsdp_rank = rank
        state.true_fsdp_world_size = ws
        state.true_fsdp_params = entries
        state.true_fsdp_flat = False
        state.true_fsdp_unsharded = False
        return state
    result = _storage._init_true_fsdp_state_impl(
        module, state, params, world_size=ws, rank=rank,
        flat=common._fsdp2_flat_enabled(ws, total_numel), api=sys.modules[__name__])
    if len(state.true_fsdp_params) != len(params):
        raise RuntimeError("shared FSDP init changed parameter-entry cardinality")
    for entry, (name, owner, attr, param) in zip(state.true_fsdp_params, params):
        if entry.name != name or entry.owner is not owner or entry.attr != attr:
            raise RuntimeError("shared FSDP init changed parameter-entry ordering")
        entry.aliases = tuple(aliases[id(param)])
        _remember_original_owner(param, state)
        _bind_entry(entry, entry.shard)
    return result


def _unshard_module_params(module):
    with common._frontend_scope(getattr(module, "_fsdp_state", None)):
        return _unshard_module_params_impl(module)


def _compute_shard(state, master_shard):
    return _storage._compute_shard(state, master_shard)


def _cast_floating_tree(value, target):
    return _storage._cast_floating_tree(value, target)


def _unshard_module_params_impl(module):
    if _resident.selected(getattr(module, "_fsdp_state", None)):
        return _resident.unshard(module)
    result = _storage._unshard_module_params_impl(module, api=sys.modules[__name__])
    state = getattr(module, "_fsdp_state", None)
    if state is not None and getattr(state, "true_fsdp_unsharded", False):
        for entry in state.true_fsdp_params:
            if entry.full_param is not None:
                # A no-grad-created view can be natively active while its
                # public requires_grad flag is disabled. Publish the entry's
                # real trainability, not just its native stop-grad state.
                if (getattr(entry, "requires_grad", True)
                        and not _parameter_requires_grad(entry.full_param)):
                    entry.full_param.start_grad()
                _bind_entry(entry, entry.full_param)
    return result


def _reshard_module_params(module):
    if _resident.selected(getattr(module, "_fsdp_state", None)):
        return _resident.reshard(module)
    result = _storage._reshard_module_params(module)
    state = getattr(module, "_fsdp_state", None)
    if state is not None and getattr(state, "true_fsdp_initialized", False):
        for entry in state.true_fsdp_params:
            _bind_entry(entry, entry.shard)
    return result


#: Attribute on the FSDP state counting how deep we are inside this module's
#: own forward. See :func:`_execute_with_true_fsdp`.
_EXECUTE_DEPTH_ATTR = "true_fsdp_execute_depth"


def _begin_no_grad_forward(state):
    """Own only a new graph-free forward; never discard an older live graph."""
    if state is None or not getattr(state, "true_fsdp_initialized", False):
        return False
    if (not bool(jt.flags.no_grad) or _resident.selected(state)
            or not getattr(state, "reshard_after_forward", True)):
        return False
    entries = getattr(state, "true_fsdp_params", ())
    if (getattr(state, "true_fsdp_unsharded", False)
            or getattr(state, "true_fsdp_flat_full_param", None) is not None
            or any(entry.full_param is not None
                   or getattr(entry.shard, "_torch_grad", None) is not None
                   for entry in entries)):
        raise NotImplementedError(
            "FSDP no-grad forward requires no prior live full parameters or pending gradients")
    return True


def _no_grad_output_requires_grad(out):
    # core carrier routing includes the Torch Tensor frontend, not only jt.Var.
    return any(bool(value.requires_grad) for value in common._tensor_values(out))


def _finish_no_grad_forward(state, out, owned):
    if (not owned or _no_grad_output_requires_grad(out)
            or any(getattr(entry.shard, "_torch_grad", None) is not None
                   for entry in state.true_fsdp_params)):
        return False
    roots = common._tensor_values(out)
    if roots:
        jt.submit_pending(*roots, device_sync=True)
    return True


def _release_full_params(state):
    if _resident.selected(state):
        _resident.owners(state)
        return state  # Explicitly resident; no storage-release claim.
    return _storage._release_full_params(state)


def _fsdp_forward_pre_hook(module, args, kwargs):
    state = module._fsdp_state
    frames = state.true_fsdp_forward_hook_frames
    if frames and not frames[-1]["dispatch_depth"]:
        raise RuntimeError("FSDP forward hook lifecycle was not completed")
    owner = not frames and not getattr(state, "true_fsdp_execute_depth", 0)
    policy = None
    resident_frame = None
    no_grad_owner = False
    if owner:
        no_grad_owner = _begin_no_grad_forward(state)
        if _resident.selected(state):
            resident_frame = _resident.forward_enter(module)
        try:
            args, kwargs, policy = _storage._prepare_forward_inputs(state, args, kwargs)
            _unshard_module_params(module)
            if resident_frame is not None:
                args, kwargs = _resident.wrap_inputs(resident_frame, args, kwargs)
        except BaseException as error:
            if resident_frame is not None:
                try:
                    _resident.forward_exit(resident_frame, None, False)
                except BaseException as cleanup_error:
                    raise error from cleanup_error
            raise
    frames.append({"owner": owner, "dispatch_depth": 0, "policy": policy,
                   "resident": resident_frame, "no_grad_owner": no_grad_owner})
    return args, kwargs


def _close_fsdp_forward_hook(module, frame, frozen_forward_synced=False,
                             no_grad_forward_synced=False):
    state = module._fsdp_state
    frames = state.true_fsdp_forward_hook_frames
    if not frames or frames[-1] is not frame:
        raise RuntimeError("FSDP forward hook ownership changed")
    try:
        resident_frame = frame.get("resident")
        if resident_frame is not None:
            frame["resident"] = None
            _resident.forward_exit(resident_frame, None, False)
        if frame["owner"] and getattr(state, "reshard_after_forward", True):
            _reshard_module_params(module)
            if no_grad_forward_synced:
                _release_full_params(state)
            if frozen_forward_synced:
                jt.gc()
    finally:
        frames.pop()


def _fsdp_forward_post_hook(module, args, kwargs, out):
    state = module._fsdp_state
    frames = state.true_fsdp_forward_hook_frames
    if not frames:
        raise RuntimeError("FSDP forward post hook has no owned invocation")
    frame = frames[-1]
    frozen_forward_synced = False
    no_grad_forward_synced = False
    try:
        if frame["owner"]:
            out, frozen_forward_synced = _storage._finish_forward_output(
                module, state, args, kwargs, out, frame["policy"], api=sys.modules[__name__]
            )
            no_grad_forward_synced = _finish_no_grad_forward(
                state, out, frame["no_grad_owner"])
        resident_frame = frame.get("resident")
        if resident_frame is not None:
            frame["resident"] = None
            _resident.forward_exit(resident_frame, out, True)
        return out
    finally:
        _close_fsdp_forward_hook(module, frame, frozen_forward_synced,
                                 no_grad_forward_synced)


def _execute_with_true_fsdp(module, orig_execute, *args, **kwargs):
    state = getattr(module, "_fsdp_state", None)
    frames = getattr(state, "true_fsdp_forward_hook_frames", ())
    if frames:
        frame = frames[-1]
        depth = frame["dispatch_depth"]
        frame["dispatch_depth"] = depth + 1
        try:
            # The actual public pre/post hooks own this __call__. A nested
            # execute wrapper must not reshard before user prepend post hooks.
            return orig_execute(*args, **kwargs)
        except BaseException as error:
            if not depth:
                try:
                    _close_fsdp_forward_hook(module, frame)
                except BaseException as cleanup_error:
                    raise error from cleanup_error
            raise
        finally:
            frame["dispatch_depth"] = depth
    if _resident.selected(state):
        raise NotImplementedError("resident FSDP direct execute lacks public forward ownership")
    # Direct execute keeps the shared executor as sole unshard/reshard owner.
    owned = (not getattr(state, "true_fsdp_execute_depth", 0)
             and _begin_no_grad_forward(state))
    if owned:
        release_ready = False

        def completed_execute(*inner_args, **inner_kwargs):
            nonlocal release_ready
            out = orig_execute(*inner_args, **inner_kwargs)
            release_ready = _finish_no_grad_forward(state, out, True)
            return out

        out = _storage._execute_with_true_fsdp(
            module, completed_execute, *args, api=sys.modules[__name__], **kwargs)
        if release_ready and not _no_grad_output_requires_grad(out):
            _release_full_params(state)
        return out
    return _storage._execute_with_true_fsdp(
        module, orig_execute, *args, api=sys.modules[__name__], **kwargs
    )


def _wrapped_execute(self, *args, **kwargs):
    frames = getattr(getattr(self, "_fsdp_state", None), "true_fsdp_forward_hook_frames", ())
    if frames and not frames[-1]["dispatch_depth"]:
        raise RuntimeError("direct execute cannot resume an incomplete FSDP hook invocation")
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
    if getattr(state, "frontend_type", None) is not None:
        # Only the Torch frontend changes public hook ordering. The native
        # execute path, including direct execute calls, retains the old owner.
        state.true_fsdp_forward_hook_frames = []
        pre = None
        try:
            pre = module.register_forward_pre_hook(
                _fsdp_forward_pre_hook, prepend=True, with_kwargs=True
            )
            post = module.register_forward_hook(
                _fsdp_forward_post_hook, prepend=False, with_kwargs=True
            )
        except BaseException as error:
            try:
                try:
                    if pre is not None:
                        pre.remove()
                finally:
                    object.__setattr__(module, "execute", orig_execute)
                    object.__delattr__(module, "_fsdp_orig_execute")
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        state.true_fsdp_forward_hook_handles = (pre, post)
    return module

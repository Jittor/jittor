"""Persistent flat parameter/state layout for the ACL fused AdamW.

The ACL fused update issues one ``aclnnApplyAdamWV2`` launch per tensor:
200 launches for ViT-B/16 and 686 for the SD15 UNet. torch_npu's
``NpuFusedAdamW`` instead points every parameter, state and gradient at one
flat buffer (``npu_change_data_ptr``) and updates the flat tensors with a few
element-wise ops. This module gives the same layout to the ACL path: at
optimizer construction each parameter and state is re-pointed at a slice of
one flat buffer, and each step concatenates the gradients and submits one
fused update (KI-EXEC-016).

The layout is worthwhile while the per-launch overhead dominates: ViT-B/16
and BERT-base step ~3% faster, while the 860M-parameter SD15 UNet, whose
optimizer is bandwidth-bound, measured slower and is left on the per-tensor
path (``_MAX_FLAT_ELEMENTS``).

Everything the fused op cannot represent -- a capture, a gradient scaler's
overflow decision, a guarded learning rate, a device move, a replaced state
list -- permanently falls back to the per-tensor update for that group, and
the step re-checks that the parameters still share the flat buffer so a move
cannot silently update a block nobody reads.
"""

import weakref

import jittor as jt

from ..._core.dtypes import var_dtype_name
from ..._core.var import device_scope_like
from ..._runtime.dispatch import dispatch_context

_DTYPE_SIZE = {"float32": 4, "float16": 2, "bfloat16": 2}
_OWNER = "_acl_flat_owner"
# One flat gradient is built per step on top of the flat layout. Past this
# size the optimizer is bandwidth-bound and the extra gradient copy costs
# more than the launches it removes: SD15 UNet (860M parameters) measured
# 3.5 ms slower, while ViT (86M) and BERT (110M) improved by 3%.
_MAX_FLAT_ELEMENTS = 1 << 28


def _last_line(exc):
    lines = str(exc).splitlines()
    return (lines[-1] if lines else type(exc).__name__)[:160]


class _FlatState:
    """One param group's flat buffers and the slices they were built from."""

    __slots__ = (
        "flat_params", "flat_moments", "flat_variances",
        "params", "moments", "variances", "param_list", "moment_list",
        "variance_list", "hold", "broken",
    )


def _decline(message):
    # A declined layout is a fallback, not a failure: the per-tensor fused
    # path still computes the same update. Keep it observable.
    jt.LOG.w("[acl-flat-adamw] falling back to per-tensor update:", message)


def _same_dtype(values, dtype):
    return all(var_dtype_name(value) == dtype for value in values)


def setup_group(optimizer, group):
    """Flatten ``group``'s parameters and states in place; return the state.

    Called after the optimizer created its per-parameter state buffers and
    before the first step, i.e. before any forward graph reads the
    parameters.
    """
    params = group.get("params") or []
    moments = group.get("m") or []
    variances = group.get("values") or []
    if not params or len(moments) != len(params) or len(variances) != len(params):
        return None
    if len(params) < 2:
        # A single tensor already costs one launch; re-pointing it would only
        # change the pointer callers and tests hold.
        return None
    dtype = var_dtype_name(params[0])
    if dtype not in _DTYPE_SIZE:
        return None
    try:
        if dispatch_context(params[0]).backend != "acl":
            return None
    except Exception:
        return None
    if not (_same_dtype(params, dtype) and _same_dtype(moments, dtype)
            and _same_dtype(variances, dtype)):
        return None
    if any(not value.is_contiguous() for value in params):
        return None
    # A parameter that is a view of someone else's storage must keep writing
    # through that view; re-pointing only this holder would split them.
    if getattr(params[0], "_is_view", None) is not None:
        if any(value._is_view() for value in params):
            return None
    # A second optimizer on the same parameters must not re-point them: its
    # view would free the block the first one is still updating.
    owner = getattr(params[0], "__dict__", {}).get(_OWNER)
    if owner is not None and owner() is not None and owner() is not optimizer:
        return None

    offsets = []
    total = 0
    for value in params:
        offsets.append(total)
        total += int(value.numel())
    if total <= 0 or total > _MAX_FLAT_ELEMENTS:
        return None
    dsize = _DTYPE_SIZE[dtype]

    try:
        with device_scope_like(params[0]):
            # The parameters are already materialised, so one concat builds
            # the flat in a single op. The states start at zero; allocating
            # them dense matters, because the CANN update writes them in
            # place and a broadcast-zero view has no full-size block.
            flat_params = jt.concat(
                [value.reshape((-1,)) for value in params], dim=0)
            flat_moments = jt.zeros((total,), dtype=dtype)
            flat_variances = jt.zeros((total,), dtype=dtype)
            # `zeros` is a host constant that the construction placement does
            # not move; the flat copy must live where the parameters do, or
            # the first `pre_step` realigns the states off the flat block.
            device = max(int(params[0].device_id), 0)
            if flat_moments.placement_backend != params[0].placement_backend:
                flat_moments = flat_moments.to_device(device)
                flat_variances = flat_variances.to_device(device)
        # The concat must land before the parameters stop naming their own
        # block, otherwise it would read the zeroed slices it is replacing.
        jt.sync_all(True)
        for i, value in enumerate(params):
            offset_bytes = offsets[i] * dsize
            value._repoint_share(flat_params, offset_bytes)
            moments[i]._repoint_share(flat_moments, offset_bytes)
            variances[i]._repoint_share(flat_variances, offset_bytes)
    except Exception as exc:
        _decline("%s: %s" % (type(exc).__name__, _last_line(exc)))
        return None

    state = _FlatState()
    state.flat_params = flat_params
    state.flat_moments = flat_moments
    state.flat_variances = flat_variances
    state.params = list(params)
    state.moments = list(moments)
    state.variances = list(variances)
    # The step writes slices of these lists; a replaced list (a state-dict
    # load that rebinds the buffers) would leave it updating unreferenced
    # storage, so the identity is checked every step.
    state.param_list = group.get("params")
    state.moment_list = group.get("m")
    state.variance_list = group.get("values")
    state.hold = []
    state.broken = False
    states = getattr(optimizer, "_acl_flat_states", None)
    if states is None:
        states = {}
        object.__setattr__(optimizer, "_acl_flat_states", states)
        # A captured step cannot replay a flat layout; undo it before the
        # recording starts, where the copies are not part of the graph.
        from jittor._runtime import step_capture
        owner = weakref.ref(optimizer)

        def _deflate():
            target = owner()
            if target is not None:
                drop_layout(target)

        step_capture.register_pre_capture(_deflate)
    states[id(group)] = state
    marker = weakref.ref(optimizer)
    for value in list(params) + list(moments) + list(variances):
        try:
            value.__dict__[_OWNER] = marker
        except Exception:
            pass
    return state


def drop_layout(optimizer):
    """Give every parameter and state its own storage again.

    A captured step treats its tensors as independent leaves, which a slice
    of a shared flat block is not; the capture runs this before recording.
    """
    states = getattr(optimizer, "_acl_flat_states", None)
    if not states:
        return
    for state in states.values():
        state.broken = True
        for values in (state.params, state.moments, state.variances):
            for value in values:
                # A one-input concat is a dense device-side copy; `update`
                # rebinds the holder to it and leaves the flat ring.
                copy = jt.concat([value.reshape((-1,))], dim=0)
                value.update(copy.reshape(tuple(value.shape)))
        state.hold = []
    states.clear()


def _shares(value, flat):
    check = getattr(value, "_shares_storage_with", None)
    return check is not None and check(flat)


def flat_step(optimizer, group, active, lr, beta1, beta2, weight_decay, eps,
              param_steps):
    """Run one flat fused AdamW step for ``group``; False to fall back.

    ``active`` is the shim's list of ``(param, moment, variance, grad,
    step_value)`` entries, already filtered to parameters that have a
    gradient. The flat path requires all of the group's parameters to be
    active and to share one step value. ``param_steps`` is the list to
    advance after a successful step, or None when the caller already did.

    Any fallback is permanent for the group: the per-tensor update rebinds
    each parameter to its own new tensor, so the ones it touched no longer
    live in the flat block and a later flat step would update a slice nobody
    reads.
    """
    states = getattr(optimizer, "_acl_flat_states", None)
    state = states.get(id(group)) if states else None
    if state is None or state.broken:
        return False

    def fall_back(message=None):
        state.broken = True
        if message is not None:
            _decline(message)
        return False

    if len(active) != len(state.params):
        return fall_back("only %d of %d parameters have gradients"
                         % (len(active), len(state.params)))
    if (group.get("params") is not state.param_list
            or group.get("m") is not state.moment_list
            or group.get("values") is not state.variance_list):
        return fall_back("the group's parameter or state list was replaced")
    if not isinstance(lr, (int, float)):
        return fall_back("learning rate is not a plain number")
    # A captured step replays without this Python path, and a gradient
    # scaler's overflow decision is made by the caller; both keep the
    # per-tensor update.
    from jittor._runtime import step_capture
    if step_capture.active():
        return fall_back("a step is being captured")
    if getattr(optimizer, "_amp_found_inf", None):
        return fall_back("the gradient scaler found inf/nan")
    step_values = {int(entry[4]) for entry in active}
    if len(step_values) != 1:
        return fall_back("parameters carry %d different step counts"
                         % len(step_values))
    # A device move after construction re-allocates a parameter or state;
    # updating the old block would then update nobody.
    for name, values, flat in (
            ("param", state.params, state.flat_params),
            ("moment", state.moments, state.flat_moments),
            ("variance", state.variances, state.flat_variances)):
        for index in (0, len(values) - 1):
            if not _shares(values[index], flat):
                return fall_back("%s[%d] is no longer on the flat buffer"
                                 % (name, index))

    try:
        from jittor.backends.acl.kernels.ops.adamw_op import fused_adamw_acl
    except Exception as exc:
        return fall_back("import: %s" % (_last_line(exc),))

    try:
        # One concat per group instead of one small copy per parameter; the
        # CANN update reads a single dense gradient tensor.
        gradient = jt.concat(
            [entry[3].reshape((-1,)) for entry in active], dim=0)
        step = jt.array(float(next(iter(step_values))),
                        dtype="float32").stop_grad()
        updates = fused_adamw_acl(
            [state.flat_params], [state.flat_moments], [state.flat_variances],
            [gradient], step, lr, beta1, beta2, weight_decay, eps)
    except Exception as exc:
        return fall_back("%s: %s" % (type(exc).__name__, _last_line(exc)))

    flat_updates = [value for values in updates for value in values]
    for i, entry in enumerate(active):
        if param_steps is not None:
            param_steps[i] = int(param_steps[i]) + 1
    # `FusedAdamwOp::infer_shape` makes every output share its input's
    # storage, so submitting the graph allocates the output as an alias of
    # the flat block and the CANN update writes it in place. Until the
    # executor has resolved that share the alias is not yet in the ring.
    # One root is enough: the three outputs are one op. Submitting here is
    # also what keeps the op from being pruned.
    jt.submit_pending(flat_updates[0], device_sync=False)
    state.hold = flat_updates
    for entry in active:
        if entry[0].is_stop_grad():
            entry[0].start_grad()
    for out, flat in ((updates[0][0], state.flat_params),
                      (updates[1][0], state.flat_moments),
                      (updates[2][0], state.flat_variances)):
        if out is flat or _shares(out, flat):
            continue
        # The allocator refused the alias (e.g. a block that cannot hold the
        # sub-range): the op wrote its own buffer, so publish it back. The
        # copy is ordered behind the update on the same device stream.
        flat._copy_into(out, False)
    return True

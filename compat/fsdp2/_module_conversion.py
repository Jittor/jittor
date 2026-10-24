"""Move owned FSDP storage before rebinding its rank-local parameter views."""
from functools import partial

import jittor as jt
from jittor._core.carrier_protocol import copy_native_tensor, native_api

from . import common, shard


def _conversion_input(source, convert):
    """Stage a placed Module.to through an independent, storage-sharing holder.

    The owned Module.to callback changes residency by device_copy + _update,
    not by moving a native alias group or writing its input's values. Its input
    therefore needs a separate holder, not a whole same-device allocation.
    Arbitrary _apply callbacks and legacy FollowRuntime migration retain their
    owned copy: either can mutate the input's shared storage in place.
    """
    from ..torch.installers.nn.module_methods import _module_to_conversion

    if (source.placement_backend < 0 or not isinstance(convert, partial)
            or convert.func is not _module_to_conversion):
        return copy_native_tensor(source).stop_grad()
    # Bypass public DTensor.detach decoration. This transport holder must not
    # carry FSDP state/snapshot methods that describe the *old* storage.
    if isinstance(source, jt.core.Var):
        value = jt.core.Var.detach(source)
    else:
        value = native_api(type(source), source)["detach"](source)
    return value.stop_grad()


def _convert_module_storage(module, convert):
    """Handle initialized shards for Module.to without assigning into a view.

    Only quiescent states are supported. Converting a live forward/backward
    graph would invalidate its saved full parameters; pending gradients require
    additional optimizer-state coordination and are explicitly rejected.
    Parameter objects are retained, including references held by optimizers.
    """
    states = []
    for child in shard._iter_fsdp_modules(module, recurse=True):
        state = getattr(child, "_fsdp_state", None)
        if state is None or not getattr(state, "true_fsdp_initialized", False):
            continue
        if getattr(state, "true_fsdp_unsharded", False) or any(
                getattr(entry, "full_param", None) is not None
                or getattr(entry.shard, "_torch_grad", None) is not None
                for entry in state.true_fsdp_params):
            raise NotImplementedError("FSDP Module.to requires resharded parameters with no pending gradients")
        states.append(state)
    plans = []
    for state in states:
        with common._frontend_scope(state):
            sources = ([state.true_fsdp_flat_shard] if state.true_fsdp_flat else
                       [entry.shard for entry in state.true_fsdp_params])
            converted = []
            trainability = [not entry.shard.is_stop_grad() for entry in state.true_fsdp_params]
            for source in sources:
                # The plans remain private until every conversion has synced.
                # Never give an in-place residency callback the public holder.
                value = convert(_conversion_input(source, convert))
                value.sync()
                converted.append(value)
            plans.append((state, sources, converted, trainability))
    handled = set()
    for state, sources, converted, trainability in plans:
        with common._frontend_scope(state):
            for source, value in zip(sources, converted):
                trainable = not source.is_stop_grad()
                # Residency changes storage, not the logical parameter value.
                source._update(value, logical_write=False)
                source.start_grad() if trainable else source.stop_grad()
            if state.true_fsdp_flat:
                slices = shard._flat_entry_slices(state, state.true_fsdp_flat_shard)
                for entry, value in zip(state.true_fsdp_params, slices):
                    # Raw storage rebinding (not slice assignment) retains the
                    # public Parameter identity and all shared module aliases.
                    entry.shard._update(value, logical_write=False)
                    entry.shard._set_view_of(value, Ellipsis, False)
            for entry, trainable in zip(state.true_fsdp_params, trainability):
                entry.requires_grad = trainable
                entry.shard.start_grad() if trainable else entry.shard.stop_grad()
                entry.dtype = entry.shard.dtype
                shard._bind_entry(entry, entry.shard)
                handled.add(id(entry.shard))
            if state.true_fsdp_flat:
                (state.true_fsdp_flat_shard.start_grad() if any(trainability)
                 else state.true_fsdp_flat_shard.stop_grad())
    # Grad-policy rebinding can publish a lazy storage alias. Complete those
    # aliases here so Module.to returns the requested resident storage.
    for state, sources, converted, trainability in plans:
        with common._frontend_scope(state):
            for source in sources:
                source.sync()
            for entry in state.true_fsdp_params:
                entry.shard.sync()
    return handled

"""Native adapters for the shared FSDP storage owner."""

import sys

from . import common as common, storage


_materialize_initial_shard = storage._materialize_initial_shard
_flat_local_overlap = storage._flat_local_overlap
_compute_shard = storage._compute_shard
_cast_floating_tree = storage._cast_floating_tree
_reshard_module_params = storage._reshard_module_params
_release_full_params = storage._release_full_params


def _mark_fsdp_param_var(value, state, entry, role):
    # Common metadata is frontend-independent. Torch adds its DTensor surface
    # in its own adapter; native arithmetic never imports that frontend.
    for name, item in (
        ("_jittor_fsdp2_state", state),
        ("_jittor_fsdp2_entry", entry),
        ("_jittor_fsdp2_module", state.true_fsdp_module),
        ("_jittor_fsdp2_role", role),
        ("_jittor_native_fsdp_state", state),
    ):
        object.__setattr__(value, name, item)
    return value


def _flat_entry_slices(state, value):
    return storage._flat_entry_slices(state, value, api=sys.modules[__name__])


def _refresh_flat_entry_shards(state):
    return storage._refresh_flat_entry_shards(state, api=sys.modules[__name__])


def _unshard_module_params(module):
    return storage._unshard_module_params_impl(module, api=sys.modules[__name__])


def _execute_with_true_fsdp(module, execute, *args, **kwargs):
    return storage._execute_with_true_fsdp(
        module, execute, *args, api=sys.modules[__name__], **kwargs
    )


def _init_state(module, state, parameters):
    return storage._init_true_fsdp_state_impl(
        module,
        state,
        parameters,
        world_size=state.native_world,
        rank=state.native_rank,
        flat=False,
        api=sys.modules[__name__],
    )

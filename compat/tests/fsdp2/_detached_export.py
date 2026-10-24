"""Detached-export regression in an already-owned real NCCL CUDA2 rank/cache scope.

No launcher, cache, fallback policy, model/PPO replacement, or source-pin bypass
is provided here. The caller remains responsible for those admission checks.
``engine`` is the caller-owned tiny fixture with the ORIGINAL bound
FSDPEngine.get_per_tensor_param method; it is not a full-engine-init claim.
Returned records contain only type/shape/device/count metadata, never values.
"""
import os
import types
import gc
import weakref


def _require(value, message):
    if not value:
        raise AssertionError(message)


def _equal(torch, actual, expected):
    _require(actual.device == expected.device, "CUDA comparison device differs")
    _require(tuple(actual.shape) == tuple(expected.shape), "full tensor shape differs")
    _require(actual.dtype == expected.dtype, "full tensor dtype differs")
    _require(bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)),
             "entire CUDA tensor comparison differs")


def _full(value):
    method = getattr(value, "full_tensor", None)
    return method() if callable(method) else value


def _types(value, public_dtensor):
    return dict(actual_type=type(value).__module__ + "." + type(value).__qualname__,
                actual_bases=[base.__module__ + "." + base.__qualname__
                              for base in type(value).__bases__],
                exact_public_DTensortype=type(value) is public_dtensor,
                legacy_instance_admission=isinstance(value, public_dtensor),
                shape=[int(dim) for dim in value.shape],
                device_type=value.device.type, device_index=value.device.index)


def _original_export(torch, engine, expected):
    # The caller must verify this bound method's complete code/source identity
    # against the pinned upstream checkout before invoking this draft.
    method = engine.get_per_tensor_param
    _require(isinstance(method, types.MethodType), "original bound export method required")
    stream, peft = method(layered_summon=False, base_sync_done=True)
    _require(peft is None, "no PEFT branch in this detached export regression")
    seen, failures = [], []
    # Exhaust the original iterator even if one item's shape/copy is wrong.
    for name, value in stream:
        seen.append(name)
        if name not in expected or seen.count(name) != 1:
            failures.append("unexpected or repeated original export key")
            continue
        target = torch.empty_like(expected[name])
        if tuple(value.shape) != tuple(target.shape):
            failures.append("original export retained a local shard")
            continue
        with torch.no_grad():
            target.copy_(value)
        if not bool(torch.allclose(target, expected[name], atol=1e-5, rtol=1e-5)):
            failures.append("original export full CUDA copy differs")
    _require(set(seen) == set(expected) and len(seen) == len(expected),
             "complete original export inventory differs")
    _require(not failures, "one or more exhausted original export copies failed")
    return dict(iterator_exhausted=True, complete_items=len(seen),
                all_full_shape_copies_passed=True,
                original_full_engine_initialization_claimed=False)


def _fixed_snapshot_replacement_probe(torch, model, expected):
    """Candidate-specific metadata contract, NOT an upstream training rewrite.

    Temporarily replace state/entry references in this isolated fixture only;
    full_tensor on the already-created detached alias must not read them.
    The original metadata is restored before any further model operation.
    """
    params = dict(model.named_parameters())
    source = params["large"]
    state = source._jittor_fsdp2_state
    entry = source._jittor_fsdp2_entry
    alias = source.detach()
    _equal(torch, alias.full_tensor(), expected["large"])
    old_shard, old_shape = entry.shard, entry.shape
    had_offset = hasattr(entry, "flat_offset")
    old_offset = getattr(entry, "flat_offset", None)
    old_flat = getattr(state, "true_fsdp_flat_shard", None)
    try:
        entry.shard = torch.full_like(old_shard, 99.)
        entry.shape = (1,)
        if had_offset:
            entry.flat_offset = 0
        if old_flat is not None:
            state.true_fsdp_flat_shard = torch.full_like(old_flat, 101.)
        # Include detach-of-detach: it must retain the old storage snapshot too.
        _equal(torch, alias.full_tensor(), expected["large"])
        _equal(torch, alias.detach().full_tensor(), expected["large"])
    finally:
        entry.shard, entry.shape = old_shard, old_shape
        if had_offset:
            entry.flat_offset = old_offset
        if old_flat is not None:
            state.true_fsdp_flat_shard = old_flat
    return dict(old_storage_preserved_after_entry_and_flat_replacement=True,
                metadata_restored=True, scope="candidate_metadata_contract_only")


def _holder_collectability_probe(torch, source):
    """Retain the callable, not its holder; gc disabled detects a self-cycle."""
    alias = source.detach()
    local = alias.to_local()
    if hasattr(local, "sync"):
        local.sync()
    torch.cuda.synchronize()
    callback = alias.full_tensor
    reference = weakref.ref(alias)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        del alias
        _require(reference() is None,
                 "detached holder retained by its installed callable/snapshot")
        try:
            callback()
        except ReferenceError:
            pass
        else:
            raise AssertionError("dead detached owner did not fail explicitly")
    finally:
        if was_enabled:
            gc.enable()
    return dict(holder_released_with_gc_disabled_and_callable_retained=True,
                dead_owner_raises_ReferenceError=True)


def run(torch, *, jt=None, engine=None):
    """Run serially on BOTH actual ranks with the same per-rank call ordering."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor

    _require(torch.cuda.is_available(), "real CUDA is required")
    _require(torch.distributed.is_initialized(), "caller-owned process group required")
    _require(torch.distributed.get_world_size() == 2, "exactly two CUDA ranks required")
    _require(torch.distributed.get_backend() == "nccl", "actual NCCL backend required")
    candidate = hasattr(torch, "_torch_compat_install_context")
    if candidate:
        _require(jt is not None and int(jt.core.backend_fallback_count()) == 0,
                 "candidate actual core and zero initial fallback required")

    class ExportModel(torch.nn.Module):
        def __init__(self, include_empty):
            super().__init__()
            self.large = torch.nn.Parameter(
                torch.arange(15, dtype=torch.float32, device="cuda:0").reshape(3, 5) / 8.)
            self.small = torch.nn.Parameter(torch.tensor([2.], device="cuda:0"))
            self.tiny = torch.nn.Parameter(torch.tensor([3.], device="cuda:0"))
            if include_empty:
                self.empty = torch.nn.Parameter(torch.empty((0, 3), device="cuda:0"))
            self.tied = self.large
            self.register_buffer("resident_buffer", torch.tensor([1., 2.], device="cuda:0"))

    mesh = init_device_mesh("cuda", (2,))
    rows = []
    prior_mode = os.environ.get("JITTOR_FSDP2_FLAT")
    try:
        # total_numel=17 gives padding. Flat mode puts small/tiny wholly on
        # rank 1, hence rank 0 has an empty LOCAL intersection even when the
        # corresponding global parameter is not empty. The global empty tensor
        # is an additional independent edge; native rejection is not skipped.
        for flat in (False, True):
            os.environ["JITTOR_FSDP2_FLAT"] = "1" if flat else "0"
            for include_empty in (False, True):
                model = ExportModel(include_empty)
                expected = {name: value.detach().clone()
                            for name, value in model.state_dict(keep_vars=True).items()}
                fully_shard(model, mesh=mesh, reshard_after_forward=True,
                            mp_policy=MixedPrecisionPolicy(
                                param_dtype=torch.float32, reduce_dtype=torch.float32))
                named = dict(model.named_parameters())
                default, live = model.state_dict(), model.state_dict(keep_vars=True)
                _require(set(default) == set(live) == set(expected), "state inventory differs")
                row = dict(requested_flat_mode=flat, global_empty_parameter=include_empty,
                           complete_state_items=len(expected), physical_type_observations=[],
                           global_shape_or_genuine_DTensortype_acceptance_claimed=False)
                if candidate:
                    _require(bool(model._fsdp_state.true_fsdp_flat) == flat,
                             "candidate did not enter the requested flat mode")
                for name in expected:
                    value = default[name]
                    _require(not value.requires_grad, "default state_dict did not detach")
                    _equal(torch, _full(value), expected[name])
                    _equal(torch, _full(live[name]), expected[name])
                    if name != "resident_buffer":
                        _equal(torch, _full(value.detach()), expected[name])
                        row["physical_type_observations"].append(
                            dict(name=name, **_types(value, DTensor)))
                        if candidate:
                            attrs = vars(value)
                            _require(not any(key in attrs for key in
                                ("_jittor_fsdp2_state", "_jittor_fsdp2_entry", "_jittor_fsdp2_role")),
                                "detached export leaked live optimizer metadata")
                            _require(not getattr(value, "_is_param", False),
                                     "detached export retained Parameter role")
                    else:
                        _require(not isinstance(value, DTensor), "ordinary buffer became distributed")
                _equal(torch, _full(default["large"]), _full(default["tied"]))
                if engine is not None:
                    engine.module = model
                    row["original_export"] = _original_export(torch, engine, expected)
                else:
                    row["original_export"] = {"executed": False}
                if candidate:
                    row["replacement"] = _fixed_snapshot_replacement_probe(torch, model, expected)
                    row["collectability"] = _holder_collectability_probe(torch, named["large"])
                # A detached alias observes writes to the SAME allocation.
                source = named["large"]
                alias = source.detach()
                local = source.to_local()
                with torch.no_grad():
                    local.add_(.25)
                _equal(torch, alias.full_tensor(), expected["large"] + .25)
                _require(not alias.requires_grad and bool(source.requires_grad),
                         "detach changed original trainability")
                row["same_storage_inplace_write_visible"] = True
                torch.cuda.synchronize()
                if candidate:
                    _require(int(jt.core.backend_fallback_count()) == 0,
                             "candidate fallback occurred during CUDA2 regression")
                rows.append(row)
    finally:
        if prior_mode is None:
            os.environ.pop("JITTOR_FSDP2_FLAT", None)
        else:
            os.environ["JITTOR_FSDP2_FLAT"] = prior_mode
    return dict(scope="real_CUDA2_detached_FSDP_export_component_regression_NOT_PPO",
                cases=rows, cases_passed=len(rows), original_export_required=engine is not None,
                type_MRO_global_shape_mismatches_not_waived=True,
                raw_tensor_or_config_saved=False,
                L1_L2_L3_L4_L5_acceptance_claimed=False)

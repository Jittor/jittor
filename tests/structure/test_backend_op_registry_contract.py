from jittor._runtime.registry import (
    BackendRegistry,
    BackendSpec,
    DuplicateRegistration,
    MissingCapability,
    MissingKernel,
    OpRegistry,
    RegistrySnapshot,
    UnknownBackend,
)


def test_default_registry_exposes_cpu_and_cuda_capabilities():
    registry = BackendRegistry.default()
    assert registry.names() == ("cpu", "cuda")
    cpu = registry.get("cpu")
    assert cpu.device_count() == 1
    assert cpu.supports("allocator")
    assert cpu.allocator is not None
    first = cpu.allocator(4)
    second = cpu.allocator(4)
    assert first == bytearray(4)
    assert isinstance(first, bytearray)
    first[0] = 7
    assert second[0] == 0
    assert registry.get("cuda").supports("synchronize")


def test_backend_snapshot_freezes_nested_capabilities_and_preserves_order():
    capabilities = {"stream": True}
    registry = BackendRegistry((BackendSpec("cpu", capabilities=capabilities),
                                BackendSpec("cuda")))
    snapshot = registry.snapshot()
    assert tuple(spec.name for spec in snapshot) == ("cpu", "cuda")
    assert snapshot[0] is registry.get("cpu")
    capabilities["allocator"] = True
    assert "allocator" not in registry.get("cpu").capabilities
    try:
        snapshot[0].capabilities["allocator"] = True
    except TypeError:
        pass
    else:
        raise AssertionError("backend capability snapshot remained mutable")
    assert registry.get("cpu").supports("stream")
    assert not registry.get("cpu").supports("allocator")


def test_backend_state_snapshot_is_isolated_across_provider_lifecycle():
    registry = BackendRegistry((BackendSpec("cpu"), BackendSpec("cuda")))
    before = registry.snapshot_state()
    assert isinstance(before, RegistrySnapshot)
    assert tuple(spec.name for spec in before.backends) == ("cpu", "cuda")
    assert before.kernels == ()

    replacement = registry.set_capability("cuda", "stream")
    assert replacement.supports("stream")
    after = registry.snapshot_state()
    assert after.backends[1].supports("stream")
    assert not before.backends[1].supports("stream")

    registry.unregister("cuda")
    assert tuple(spec.name for spec in before.backends) == ("cpu", "cuda")
    assert tuple(spec.name for spec in after.backends) == ("cpu", "cuda")


def test_registry_snapshot_normalizes_mutable_constructor_inputs():
    backends = [BackendSpec("cpu")]
    kernels = [["add", "cpu"]]
    snapshot = RegistrySnapshot(backends, kernels)
    backends.clear()
    kernels[0][0] = "mutated"
    assert tuple(spec.name for spec in snapshot.backends) == ("cpu",)
    assert snapshot.kernels == (("add", "cpu"),)


def test_registry_snapshot_exposes_read_only_lifecycle_queries():
    snapshot = RegistrySnapshot(
        [BackendSpec("cpu"), BackendSpec("cuda")],
        [["add", "cpu"], ["add", "cuda"], ["copy", "cuda"]],
    )
    assert snapshot.backend("cuda").name == "cuda"
    assert snapshot.supported_ops("cuda") == ("add", "copy")
    assert snapshot.has_kernel("add", "cpu")
    assert not snapshot.has_kernel("missing", "cpu")
    assert snapshot.provider_for("copy", "cuda").name == "cuda"
    try:
        snapshot.provider_for("missing", "cuda")
    except MissingKernel:
        pass
    else:
        raise AssertionError("snapshot provider query accepted a missing kernel")
    try:
        snapshot.backend("metal")
    except UnknownBackend:
        pass
    else:
        raise AssertionError("snapshot lookup accepted an unknown backend")


def test_registry_snapshot_rejects_duplicate_or_dangling_ownership():
    cpu = BackendSpec("cpu")
    for backends, kernels in (
            ([cpu, cpu], []),
            ([cpu], [["add", "cpu"], ["add", "cpu"]]),
            ([cpu], [["add", "cuda"]])):
        try:
            RegistrySnapshot(backends, kernels)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("invalid snapshot ownership was accepted")


def test_registry_snapshot_rejects_ambiguous_kernel_iterables():
    """Ownership snapshots must not reinterpret strings as kernel pairs."""
    try:
        RegistrySnapshot([BackendSpec("cpu")], ["ab"])
    except TypeError as exc:
        assert "(op, backend) pairs" in str(exc)
    else:
        raise AssertionError("string kernel ownership was accepted")


def test_old_snapshot_survives_provider_unregister_without_aliasing():
    backends = BackendRegistry((BackendSpec("cpu"), BackendSpec("cuda")))
    ops = OpRegistry(backends)
    ops.register("copy", "cuda", lambda value: value)
    snapshot = ops.snapshot_state()

    removed = ops.unregister_backend("cuda")
    assert removed.name == "cuda"
    assert snapshot.backend("cuda").name == "cuda"
    assert snapshot.has_kernel("copy", "cuda")
    assert backends.names() == ("cpu",)


def test_cpu_provider_rejects_invalid_allocation_sizes():
    allocator = BackendRegistry.default().get("cpu").allocator
    try:
        allocator(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative CPU allocation was accepted")
    try:
        allocator(1.5)
    except TypeError:
        pass
    else:
        raise AssertionError("non-integer CPU allocation was accepted")


def test_operator_registry_dispatches_and_reports_supported_ops():
    backends = BackendRegistry((BackendSpec("cpu", device_count=lambda: 1),))
    ops = OpRegistry(backends)
    ops.register("add", "cpu", lambda left, right: left + right)
    assert ops.dispatch("add", "cpu", 2, 3) == 5
    assert ops.supported_ops("cpu") == ("add",)
    assert backends.supported_ops(ops, "cpu") == ("add",)
    assert ops.snapshot() == (("add", "cpu"),)


def test_dispatch_value_selects_backend_from_runtime_location():
    class CpuValue:
        def location(self):
            return "cpu"

    class CudaValue:
        def location(self):
            return "cuda:0"

    backends = BackendRegistry((BackendSpec("cpu"), BackendSpec("cuda")))
    ops = OpRegistry(backends)
    ops.register("where", "cpu", lambda value, suffix: ("cpu", suffix))
    ops.register("where", "cuda", lambda value, suffix: ("cuda", suffix))
    assert ops.dispatch_value("where", CpuValue(), 1) == ("cpu", 1)
    assert ops.dispatch_value("where", CudaValue(), 2) == ("cuda", 2)


def test_dispatch_value_accepts_backend_level_cuda_location_and_rejects_unknown():
    class BackendCudaValue:
        def location(self):
            return "cuda"

    class UnknownValue:
        def location(self):
            return "metal:0"

    backends = BackendRegistry((BackendSpec("cpu"), BackendSpec("cuda")))
    ops = OpRegistry(backends)
    ops.register("identity", "cuda", lambda value, suffix: ("cuda", suffix))
    assert ops.dispatch_value("identity", BackendCudaValue(), 7) == ("cuda", 7)
    try:
        ops.dispatch_value("identity", UnknownValue(), 7)
    except UnknownBackend as exc:
        assert "metal:0" in str(exc)
    else:
        raise AssertionError("unknown runtime location was silently dispatched")


def test_capability_gated_dispatch_fails_closed_and_preserves_kernel_contract():
    backends = BackendRegistry((
        BackendSpec("cpu", capabilities={"stream": True}),
        BackendSpec("cuda"),
    ))
    ops = OpRegistry(backends)
    ops.register("copy", "cpu", lambda value, suffix: (value, suffix))
    assert ops.dispatch_capability("copy", "cpu", "stream", "x", 1) == ("x", 1)
    try:
        ops.dispatch_capability("copy", "cpu", "allocator", "x", 1)
    except MissingCapability as exc:
        assert "allocator" in str(exc)
    else:
        raise AssertionError("dispatch ignored a missing backend capability")
    try:
        ops.dispatch_capability("copy", "cuda", "stream", "x", 1)
    except MissingKernel:
        raise AssertionError("capability check should precede kernel lookup only for supported providers")
    except MissingCapability:
        pass
    else:
        raise AssertionError("unsupported capability was silently accepted")


def test_capability_gated_value_dispatch_uses_runtime_backend():
    class CpuValue:
        def location(self):
            return "none"

    backends = BackendRegistry((
        BackendSpec("cpu", capabilities={"synchronize": True}),
    ))
    ops = OpRegistry(backends)
    ops.register("sync", "cpu", lambda value: value)
    value = CpuValue()
    assert ops.dispatch_value_capability("sync", value, "synchronize") is value
    try:
        ops.dispatch_value_capability("sync", value, "stream")
    except MissingCapability:
        pass
    else:
        raise AssertionError("value dispatch ignored missing capability")


def test_provider_capability_registration_is_atomic_and_preserves_kernel_hooks():
    backends = BackendRegistry((BackendSpec("cpu"),))
    ops = OpRegistry(backends)
    kernel = lambda value: ("ok", value)
    ops.register("sync", "cpu", kernel)
    value = object()
    try:
        ops.dispatch_capability("sync", "cpu", "synchronize", value)
    except Exception as exc:
        assert isinstance(exc, MissingCapability)
    else:
        raise AssertionError("unpublished capability was accepted")

    updated = backends.set_capability("cpu", "synchronize")
    assert updated.supports("synchronize")
    assert updated.allocator is None
    assert ops.dispatch_capability("sync", "cpu", "synchronize", value) == ("ok", value)

    revoked = backends.set_capability("cpu", "synchronize", False)
    assert not revoked.supports("synchronize")
    assert ops.has_kernel("sync", "cpu")
    try:
        ops.dispatch_capability("sync", "cpu", "synchronize", value)
    except MissingCapability:
        pass
    else:
        raise AssertionError("revoked capability remained dispatchable")


def test_provider_capability_registration_rejects_bad_updates():
    backends = BackendRegistry((BackendSpec("cpu"),))
    for args, error in ((('cpu', ''), ValueError),
                        (('cpu', 'stream', 1), TypeError),
                        (('missing', 'stream'), UnknownBackend)):
        try:
            backends.set_capability(*args)
        except error:
            pass
        else:
            raise AssertionError("invalid capability update was accepted")


def test_capability_snapshot_updates_only_through_atomic_registry_transition():
    registry = BackendRegistry((BackendSpec("cpu", capabilities={"stream": True}),))
    before = registry.snapshot()[0]
    after = registry.remove_capability("cpu", "stream")
    assert before.supports("stream")
    assert not after.supports("stream")
    assert registry.snapshot()[0] is after


def test_provider_capability_removal_is_atomic_and_preserves_other_contracts():
    backends = BackendRegistry((BackendSpec(
        "cpu", capabilities={"stream": True, "allocator": True}),))
    ops = OpRegistry(backends)
    kernel = lambda value: value
    ops.register("copy", "cpu", kernel)

    updated = backends.remove_capability("cpu", "stream")
    assert "stream" not in updated.capabilities
    assert updated.supports("allocator")
    assert ops.has_kernel("copy", "cpu")
    try:
        ops.dispatch_capability("copy", "cpu", "stream", object())
    except MissingCapability:
        pass
    else:
        raise AssertionError("removed capability remained dispatchable")

    # Idempotent removal is useful during provider teardown and must not
    # replace the provider object when there is no declaration to withdraw.
    assert backends.remove_capability("cpu", "stream") is updated
    try:
        backends.remove_capability("missing", "stream")
    except UnknownBackend:
        pass
    else:
        raise AssertionError("capability removal accepted an unknown backend")


def test_native_cpu_location_none_resolves_to_cpu():
    class NativeCpuValue:
        def location(self):
            return "none"

    backends = BackendRegistry((BackendSpec("cpu"),))
    ops = OpRegistry(backends)
    ops.register("outer", "cpu", lambda value, suffix: ("cpu", suffix))
    assert ops.dispatch_value("outer", NativeCpuValue(), 3) == ("cpu", 3)


def test_registry_rejects_duplicate_and_missing_entries():
    backends = BackendRegistry((BackendSpec("cpu"),))
    try:
        backends.register(BackendSpec("cpu"))
    except DuplicateRegistration:
        pass
    else:
        raise AssertionError("duplicate backend registration was accepted")
    ops = OpRegistry(backends)
    try:
        ops.register("add", "cuda", lambda: None)
    except UnknownBackend:
        pass
    else:
        raise AssertionError("unknown backend was accepted")
    try:
        ops.dispatch("add", "cpu")
    except MissingKernel:
        pass
    else:
        raise AssertionError("missing kernel was silently accepted")


def test_operator_registry_kernel_lifecycle_is_explicit():
    backends = BackendRegistry((BackendSpec("cpu"),))
    ops = OpRegistry(backends)
    kernel = lambda value: value + 1
    ops.register("add1", "cpu", kernel)
    assert ops.has_kernel("add1", "cpu")
    assert ops.unregister("add1", "cpu") is kernel
    assert not ops.has_kernel("add1", "cpu")
    try:
        ops.unregister("add1", "cpu")
    except MissingKernel:
        pass
    else:
        raise AssertionError("unregister silently accepted an unknown kernel")


def test_backend_teardown_removes_provider_and_all_registered_kernels():
    backends = BackendRegistry((BackendSpec("cpu"), BackendSpec("cuda")))
    ops = OpRegistry(backends)
    ops.register("add", "cuda", lambda value: value + 1)
    ops.register("mul", "cuda", lambda value: value * 2)
    ops.register("identity", "cpu", lambda value: value)

    removed = ops.unregister_backend("cuda")
    assert removed.name == "cuda"
    assert backends.names() == ("cpu",)
    assert ops.supported_ops("cpu") == ("identity",)
    try:
        ops.supported_ops("cuda")
    except UnknownBackend:
        pass
    else:
        raise AssertionError("torn-down backend remained addressable")
    try:
        ops.unregister_backend("cuda")
    except UnknownBackend:
        pass
    else:
        raise AssertionError("backend teardown was not fail-closed")


def test_provider_replacement_clears_old_kernel_ownership():
    backends = BackendRegistry((BackendSpec("cpu"),))
    ops = OpRegistry(backends)
    ops.register("add", "cpu", lambda value: ("old", value))
    replaced = ops.register_backend(
        BackendSpec("cpu", capabilities={"replacement": True}), replace=True)
    assert replaced.capabilities["replacement"]
    assert ops.supported_ops("cpu") == ()
    try:
        ops.dispatch("add", "cpu", 1)
    except MissingKernel:
        pass
    else:
        raise AssertionError("provider replacement retained an old kernel")


def test_registry_snapshot_is_coherent_across_provider_replacement():
    backends = BackendRegistry((BackendSpec("cpu"), BackendSpec("cuda")))
    ops = OpRegistry(backends)
    ops.register("add", "cuda", lambda value: value + 1)
    before = ops.snapshot_state()
    assert isinstance(before, RegistrySnapshot)
    assert tuple(spec.name for spec in before.backends) == ("cpu", "cuda")
    assert before.kernels == (("add", "cuda"),)

    ops.register_backend(BackendSpec("cuda", capabilities={"stream": True}),
                         replace=True)
    after = ops.snapshot_state()
    assert tuple(spec.name for spec in after.backends) == ("cpu", "cuda")
    assert after.backends[1].supports("stream")
    assert after.kernels == ()
    # Snapshots are values, so a later lifecycle transition cannot rewrite the
    # earlier diagnostic view.
    assert before.backends[1].capabilities == {}
    assert before.kernels == (("add", "cuda"),)


def test_snapshot_provider_query_survives_provider_teardown():
    """A captured provider/kernel pair remains queryable after teardown."""
    backends = BackendRegistry((BackendSpec("cpu"), BackendSpec("cuda")))
    ops = OpRegistry(backends)
    ops.register("copy", "cuda", lambda value: value)
    snapshot = ops.snapshot_state()

    ops.unregister_backend("cuda")
    provider = snapshot.provider_for("copy", "cuda")
    assert provider.name == "cuda"
    assert provider is snapshot.backend("cuda")
    try:
        ops.dispatch("copy", "cuda", 1)
    except UnknownBackend:
        pass
    else:
        raise AssertionError("live dispatch remained available after teardown")


def test_provider_registration_is_fail_closed_without_replacement():
    backends = BackendRegistry((BackendSpec("cpu"),))
    ops = OpRegistry(backends)
    try:
        ops.register_backend(BackendSpec("cpu"))
    except DuplicateRegistration:
        pass
    else:
        raise AssertionError("provider replacement happened without explicit opt-in")


def test_native_cpu_registry_dispatch_matches_outer_and_clamp_values():
    # Keep the registry contract test lightweight at collection time; the
    # local import still exercises the actual CPU runtime dispatch path.
    import numpy as np
    import jittor as jt

    x = jt.array([1, 2, 3])
    y = jt.array([4, 5])
    np.testing.assert_array_equal(
        jt.outer(x, y).numpy(), np.outer([1, 2, 3], [4, 5])
    )
    np.testing.assert_allclose(
        jt.clamp(jt.array([-2.0, 0.5, 3.0]), 0.0, 1.0).numpy(),
        [0.0, 0.5, 1.0],
    )


def test_native_cpu_registry_dispatches_flatten_and_reports_it():
    import numpy as np
    import jittor as jt
    from jittor._runtime.core_api import _runtime_op_registry

    assert "flatten" in _runtime_op_registry.supported_ops("cpu")
    value = jt.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(
        jt.flatten(value).numpy(), np.array([1, 2, 3, 4]))
    np.testing.assert_array_equal(
        jt.flatten(value, 0, 1).numpy(), np.array([1, 2, 3, 4]))

from jittor._runtime.registry import (
    BackendRegistry,
    BackendSpec,
    DuplicateRegistration,
    MissingKernel,
    OpRegistry,
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

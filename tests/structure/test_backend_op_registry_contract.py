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
    assert registry.get("cpu").device_count() == 1
    assert registry.get("cpu").supports("allocator")
    assert registry.get("cuda").supports("synchronize")


def test_operator_registry_dispatches_and_reports_supported_ops():
    backends = BackendRegistry((BackendSpec("cpu", device_count=lambda: 1),))
    ops = OpRegistry(backends)
    ops.register("add", "cpu", lambda left, right: left + right)
    assert ops.dispatch("add", "cpu", 2, 3) == 5
    assert ops.supported_ops("cpu") == ("add",)
    assert backends.supported_ops(ops, "cpu") == ("add",)


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

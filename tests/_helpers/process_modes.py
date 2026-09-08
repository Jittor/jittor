"""Shared path ownership for native and Torch-mode pytest processes."""

# This CPU recorder contract keeps native semantics despite its structural home.
NATIVE_MODE_PATHS = ("tests/structure/backends/acl/test_acl_dtype_preservation.py",)


def is_torch_mode_path(path):
    return path.startswith(TORCH_MODE_PATHS) and path not in NATIVE_MODE_PATHS


TORCH_MODE_PATHS = (
    "compat/tests/torch",
    "compat/tests/structure",
    "adapters/tests/vllm",
    # The OpInfo runner exercises Torch-facing signatures for the shared
    # numerical surface. The rest of tests/ops asserts native Jittor behavior.
    "tests/ops/test_ops.py",
    # Device parity consumes the same OpInfo registry and therefore the same
    # Torch-facing signatures while still executing both CPU and accelerator.
    "tests/backends/parity/test_device_parity.py",
    # The harness cases import test_device_parity and run its CPU side, so they
    # belong in the session that owns that module's semantics.
    "tests/backends/parity/test_parity_harness.py",
    # Its oracle cache is checked against the same battery: the reproducibility
    # cases import test_device_parity and run its CPU side.
    "tests/backends/parity/test_reference_cache.py",
    "tests/backends/acl/test_acl_torch_compat.py",
    # These suites intentionally lock Torch defaults and dtype semantics.
    "tests/core/test_regression.py",
    "tests/type/test_type_system.py",
    "tests/structure",
    "compat/tests/triton/test_triton_torch_compat.py",
)

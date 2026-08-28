"""Shared path ownership for native and Torch-mode pytest processes."""


TORCH_MODE_PATHS = (
    "tests/compat/torch",
    # The OpInfo runner exercises Torch-facing signatures for the shared
    # numerical surface. The rest of tests/ops asserts native Jittor behavior.
    "tests/ops/test_ops.py",
    # Device parity consumes the same OpInfo registry and therefore the same
    # Torch-facing signatures while still executing both CPU and accelerator.
    "tests/backends/parity/test_device_parity.py",
    "tests/backends/npu/test_acl_torch_compat.py",
    # These suites intentionally lock Torch defaults and dtype semantics.
    "tests/core/test_regression.py",
    "tests/core/test_type_system.py",
    "tests/structure",
    "tests/backends/triton/test_triton_torch_compat.py",
)

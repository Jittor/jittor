"""Shared path ownership for native and Torch-mode pytest processes."""


TORCH_MODE_PATHS = (
    "tests/compat/torch",
    # vLLM compatibility is reached through the shim: its attention entry
    # points import torch, which installs Torch mode into whatever process
    # touches them.
    "tests/compat/vllm",
    # The OpInfo runner exercises Torch-facing signatures for the shared
    # numerical surface. The rest of tests/ops asserts native Jittor behavior.
    "tests/ops/test_ops.py",
    # Device parity consumes the same OpInfo registry and therefore the same
    # Torch-facing signatures while still executing both CPU and accelerator.
    "tests/backends/parity/test_device_parity.py",
    # The harness cases import test_device_parity and run its CPU side, so they
    # belong in the session that owns that module's semantics.
    "tests/backends/parity/test_parity_harness.py",
    "tests/backends/npu/test_acl_torch_compat.py",
    # These suites intentionally lock Torch defaults and dtype semantics.
    "tests/core/test_regression.py",
    "tests/core/test_type_system.py",
    "tests/structure",
    "tests/backends/triton/test_triton_torch_compat.py",
)

"""Shared path ownership for native and Torch-mode pytest processes."""

import os


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
    "tests/backends/npu/test_acl_torch_compat.py",
    # These suites intentionally lock Torch defaults and dtype semantics.
    "tests/core/test_regression.py",
    "tests/core/test_type_system.py",
    "tests/structure",
    "tests/backends/triton/test_triton_torch_compat.py",
)


#: Wall-clock budget for a helper subprocess that cold-starts jittor.
#:
#: These probes run ``sys.executable -c ...`` in a fresh interpreter, so each one
#: imports jittor from scratch and may compile the core against a cold cache.
#: 180s was comfortable on an idle machine. It stopped being comfortable once a
#: dozen agents shared the box: the probe is correct and still gets killed, which
#: reads as a red gate and costs someone a bisect. Wall-clock is not the property
#: under test here -- the assertions are about module surfaces -- so the budget
#: exists only to turn a genuine hang into a failure rather than a hung session.
#:
#: Kept under the gates' ``--timeout=900`` so an overrun still surfaces as this
#: assertion (naming the subprocess) instead of pytest killing the whole test.
SUBPROCESS_TIMEOUT = int(os.environ.get("JITTOR_TEST_SUBPROCESS_TIMEOUT", "600"))

# Error Categories

The 2.19 split has two explicit entry points:

- `USER_CHECK` and `USER_CHECKop` validate user-provided arguments and raise a
  catchable `RuntimeError`.
- `ASSERT` and `ASSERTop` protect internal invariants and backend/runtime state.
  They must remain non-user errors unless a public argument is proven to reach
  the check directly.

CUDA backend checks that report failed plans, CUDA/NCCL/CUDNN status, or test
harness failures are classified as internal invariants. They are tracked by
the structure gate in `tests/structure/test_error_categories.py` so future
error-boundary migrations do not accidentally weaken backend diagnostics.
This includes CUB test-kernel CUDA status assertions, which are not user input
validation and therefore remain `ASSERT` checks.

Hardware-only negative tests may be registered as static contracts when the
host has no matching CUDA, CUDNN, CUB, or NCCL device. Static evidence does not
claim that the hardware path ran.

The current backend manifest has no additional safe user-input assertions;
future entries should first demonstrate a direct public-argument path before
changing an `ASSERT` to `USER_CHECK`.

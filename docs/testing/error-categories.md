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

## The binding layer: what `is_type` already covers

`py_converter.h` looks like a large block of unmigrated user boundaries and is
mostly not one. The generated wrapper resolves an overload by asking
`is_type<T>` about each argument, and only then calls `from_py_object<T>`. Bad
input is therefore rejected *before* conversion, by the parser, with a
catchable `RuntimeError` that names the op and lists the types it was given.
The `CHECK(is_type<...>)` calls inside `from_py_object` fire only when those
two halves of the same generated binding disagree, which is the framework
contradicting itself. They stay internal invariants.

The exception is a value coming back *out* of a binding, where no parser has
looked at it: `GradCallback` receives whatever the user's `Function.grad`
returned. A wrong gradient count or a non-Var is the caller's mistake and is a
`USER_CHECK`. Reaching a `from_py_object` check from Python, if some future
argument does get past `is_type` with an element the converter rejects, is what
would move one of the others across.

## Destructors and signal handlers

Neither may report by throwing, and `LOGf` throws. Two things decide whether
that holds, and a grep for `LOGf` is only the first of them.

**Where to look.** The scan has to cover every root that holds C++, which is
now `python/jittor/src`, `python/jittor/extern` **and `backends/`** -- the
CUDA, ACL and ROCm kernels moved into the last one, and a scan naming only the
first two goes quiet about them while still reporting a healthy total. It also
has to cover `.cu` and `.cuh`, not just `.cc` and `.h`. This is why
`test_destructor_and_handler_contract.py` asserts that *each* root contributed
a destructor rather than asserting a total.

**A literal grep is not sufficient, only necessary.** A destructor that calls
anything not marked `noexcept` can still throw: `~VarHolder` reached an
`ASSERT` through `release_both_liveness()`, the static scan saw nothing, and
the process aborted anyway. Checking a destructor therefore means reading the
call graph one hop out and, where the destructor calls into the runtime at all,
having a teardown test that runs (`tests/backends/cuda/test_var_holder_teardown.py`).

The signal handler side is `segfault_sigaction` plus everything it reaches:
`sig_write*`, `print_trace_from_signal`, and the `sigquit_callback` list. These
may only `write(2)` and `_exit`. `handle_signal` is the Windows-only handler;
it throws nothing but does use `std::cerr` and `abort()`, which is not
async-signal-safe and is tracked separately from the two-tier split.

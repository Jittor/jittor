# Error categories

Jittor splits errors into **two tiers**, each with a clear entry point:

| Entry | Purpose | Result |
| --- | --- | --- |
| `USER_CHECK` / `USER_CHECKop` | validate arguments passed in by the user | raises a **catchable** `RuntimeError` |
| `ASSERT` / `ASSERTop` | protect internal invariants and backend/runtime state | internal error |

**Unless it can be shown that a public argument reaches the check directly, `ASSERT` must stay
a non-user error.**

Checks in the CUDA backend that report planning failures, CUDA/NCCL/CUDNN status or test
harness failures are all classified as internal invariants. They are tracked by the structural
gate in `tests/structure/core/test_error_categories.py`, so that future error-boundary
migrations **do not accidentally weaken backend diagnostics**. The CUDA status assertions of
the CUB test kernels belong to this category as well -- they are not user-input validation,
so they stay `ASSERT`.

When the host has no corresponding CUDA, CUDNN, CUB or NCCL device, hardware-only negative
tests may be registered as static contracts. **Static evidence does not claim the hardware
path actually ran.**

The current backend inventory contains no other item that can safely become a user-input
assertion; to turn an `ASSERT` into a `USER_CHECK` in the future, **a direct public-argument
path must first be demonstrated**.

## Binding layer: what `is_type` already covers

`py_converter.h` looks like a large unmigrated user boundary, but **most of it is not**. The
generated wrapper resolves overloads by asking `is_type<T>` about every argument, and **only
afterwards** calls `from_py_object<T>`. Illegal input is therefore rejected by the resolver
**before** conversion, raising a catchable `RuntimeError` that names the operator and lists the
types received.

The `CHECK(is_type<...>)` calls inside `from_py_object` fire only when the two halves of the
same generated binding contradict each other -- that is **the framework contradicting itself**,
so they remain internal invariants.

The exception is values **returned from** a binding, because no resolver has looked at them:
`GradCallback` receives whatever the user's `Function.grad` returned. A wrong number of
gradients, or a return that is not a Var, is **the caller's error** and belongs to
`USER_CHECK`. If some argument ever really passes `is_type` while carrying an element the
converter rejects, that is when another check crosses this boundary.

## Destructors and signal handlers

Neither can **report by throwing**, and `LOGf` throws. Whether this constraint holds depends
on two things, and grepping for `LOGf` is only the first of them.

**Scan scope.** The scan must cover every root that holds C++ -- currently the top-level `src/`
**and `backends/`** (`SOURCE_ROOTS`). The CUDA, ACL and ROCm kernels live in the latter, and a
scan that names only the core root would **stay silent about them while still reporting a
healthy total**. The scan must also cover `.cu` and `.cuh`, not only `.cc` and `.h`. This is
why `test_destructor_and_handler_contract.py` asserts that **every root contributes a
destructor** instead of asserting a total.

**A literal grep is necessary, not sufficient.** A destructor that calls anything not marked
`noexcept` can still throw: `~VarHolder` once reached an `ASSERT` through
`release_both_liveness()`, the static scan saw nothing, and the process aborted anyway.
Checking a destructor therefore means **reading one hop out along the call graph**, and where a
destructor really does call into the runtime there must be a destruction test that actually
runs (`tests/backends/cuda/test_var_holder_teardown.py`).

On the signal-handler side there is `segfault_sigaction` and everything it can reach:
`sig_write*`, `print_trace_from_signal` and the `sigquit_callback` list. **They are allowed
only `write(2)` and `_exit`.** `handle_signal` is a Windows-only handler; it does not throw,
but it uses `std::cerr` and `abort()`, neither of which is async-signal-safe, so it is tracked
separately from these two tiers.

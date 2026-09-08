# Installer and backend boundaries

- Status: Implemented; task 11.04 accepted with the API owner migration
- Reviewed: 2026-09-08
- Owner: Runtime, build and compatibility maintainers
- Recheck when: a lifecycle callback, provider hook or allocator family changes

## Frontend installation

`compat/torch/contracts.py` defines `Installer(context: InstallContext) -> object`
and `InstallStep`. The required and optional composition tables use this same
contract. InstallContext validates a callback's signature before invoking it;
validation does not execute the callback or create a frontend. A callback can
be a module function, a bound method or an explicit callable object, but must
accept one positional context with no other required argument.

The context separates the publication target, native backend, module registry,
mutable installation state, reports and transaction. A family publishes stable
implementations and joins the active transaction for persistent writes; it
must not hide installation failure in a placeholder or mutate native types in
independent mode. The coordinator treats return values as opaque. Required
errors become named `InstallStepError` reports and propagate to rollback;
optional errors are reported once per step and remain queryable. This reporting
applies to the shared `EXPECTED` exception set. Unexpected errors, including
`AssertionError` and custom exception types outside that set, propagate to the
outer rollback without being converted to step reports. Keyboard interrupts
and process-exit exceptions are not converted into optional success.
Completed markers prevent repeating successful steps; a rolled-back install
must rebuild them. See [transaction ownership](runtime-hook-transactions.md)
and [API ownership](torch-api-ownership.md).

The old audit's 19 installer modules and seven functions named `check` are not
a current inventory. Current installation has 24 required and six optional
steps. Deployment `check(target)` functions inspect installed files; distribution
constraint `check(value)` evaluates values; neither is an installer or a hardware
provider. Their unrelated arguments are not forced into this protocol.

## Build providers and runtime backends

`jittor_utils.backend_discovery.Backend` describes a selected SDK provider:

| Entry | Input | Result and ownership |
| --- | --- | --- |
| `configure` | immutable `BuildContext` | a new `BuildConfig`; no mutation of compiler globals or the input config |
| `install_extern` | context with library-publication services | boolean indicating whether this provider handled external library setup |
| optional `post_process` | context after core construction | binds runtime operators; the coordinator ignores the return value |

`BackendPostProcess` models the optional hook separately. Discovery imports only
the selected entry point and validates all required hooks and every provided
optional hook before `configure` can compile anything. Missing hooks or wrong
signatures name the provider and method in `TypeError`. CPU/CUDA builtin selection
continues to bypass optional SDK entry points. Missing SDKs and runtime failures
remain the selected provider's explicit failures, not implicit CPU fallback.
The ACL/ROCm/Corex provider signatures use the common `BuildContext`/`BuildConfig`
types. Python 3.7 obtains `Protocol` from the conditional `typing_extensions`
dependency; Python 3.8 and later use the standard library.

The Python build protocol does not replace `src/runtime/backend.h`'s versioned
`BackendOps` table. That table owns device/memory/stream/event callbacks and
execution capabilities. The native registry validates its ABI and required
callbacks, and copies the table; code and allocator pools must outlive its
Runtime. Stream/event handles belong to their declared device. The allocation
is the authority for residency; frontend placement is a separate graph
constraint, currently being integrated under 7.12.

## Execution and allocation

`src/core/executor.h`, `exec_plan.h` and `exec_runner.h` define batch ownership,
dependency order, compilation before execution, device waits and serialized
entry. Allocation owns the tuple `(allocator, pointer, requested size, token)`;
the token is allocator-local and must be returned unchanged on release. Wrapper
allocators must preserve their underlying allocator's identity and handle,
even when the pointer is shared or a block is cached. `share_with` adds an owner
only when it succeeds, and `can_share` is the preflight for preserving a Var
sharing group during migration. See the declarations in `src/mem/allocator.h`.

These contracts establish lifecycle boundaries. They do not certify absent
hardware or mark unfinished type-factory implementation ownership as complete.

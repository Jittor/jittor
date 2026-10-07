# Torch install transactions

- Status: Implemented
- Reviewed: 2026-10-05 (baseline `e3c369acb`)
- Owner: Torch compatibility maintainers
- Recheck when: an installer starts writing a new kind of process state, a
  delayed-hook owner is added, or the rollback helpers in `compat/transaction.py`
  change

Installing the Torch frontend writes process-global state: namespace bindings,
Jittor flags, environment variables, import machinery and class attributes. This
page states which of those writes are reversible, who owns them, and what a test
may claim about rollback. The installer composition itself (steps, markers,
`InstallContext`) is described in
[Torch lowering](../compatibility/torch-lowering.md) section 2 and in the
compatibility API part of [source architecture](source-architecture.md).

## The contract: a reversible mutation ledger

`compat/transaction.py` implements a **reversible mutation ledger**, not a
hard-failure contract that aborts before the first irreversible write. Every
write routed through it is recorded with its previous value and undone in
reverse order when installation fails. A rollback that finds a slot replaced by
another actor does not overwrite it: it keeps restoring the independent
entries, then raises `TransactionConflict` naming every slot it could not
restore, and leaves the transaction in the queryable `failed` state. A retry
builds a fresh transaction and re-runs installation steps instead of trusting
completion markers that were reverted.

Two ledger types share one reentrant process lock (module-patch and
source-resolution locks are always taken after it):

- `InstallTransaction` records one installation until commit.
- `RuntimeHook` records one delayed activation and keeps its undo ledger for as
  long as the hook is active.

## What is recorded

| State | Written through | Rolled back |
| --- | --- | --- |
| the `torch*` namespace (bindings, deletions, delete masks), install markers and the published registry | publication, a namespace snapshot undo, `record_object_diffs`/`record_mapping_diffs` | yes |
| `jt.flags` | `set_flag(flags, name, value)` | yes, per named flag |
| `os.environ` | `set_env(key, value)` | yes |
| object and class attributes | `set_attr`, `module_patcher.patch_method`, `RuntimeHook.mutate_attr` | yes; class descriptors are read from the local `__dict__`, so staticmethods and inherited attributes survive |
| `sys.modules` | `publish_module`, `RuntimeHook.replace_module` | yes; a foreign module already in the slot is a conflict, never overwritten |
| `sys.path` | `mutate_path` | only the owned insertion is removed |
| `sys.meta_path` finders | `record_undo` in `module_patcher` and the permissive-package finder | yes |
| the `module_patcher` registry | `record_undo(restore_registry)` | yes |
| `builtins.__import__` | no installer writes it | -- one that starts to must go through the ledger |
| side effects of third-party module-patch callbacks | arbitrary code | **no** |

`jt.flags` cannot be snapshotted: its native object exposes dynamic flag
attributes, so `flags.__dict__` is not an authoritative enumeration. The ledger
therefore records only the flags that installers write through `set_flag` --
an explicit allowlist in code: the install-time flags of
`compat/torch/installers/core.py` (`_set_install_flag`) and `use_cuda` in
`installers/factories.py` and `installers/distributed.py`.

The distributed installer writes `JT_NCCL_WORLD_SIZE`, `JT_NCCL_RANK`,
`JT_NCCL_LOCAL_RANK`, `JT_NCCL_ROOTINFO_FILE`, `use_nccl` and `use_mpi` through
`set_env`, and then `use_cuda` through `set_flag`. Because rollback runs in
reverse order, the flag is restored before the environment it depends on.

## One owner for the write helpers

`compat/transaction.py` owns the lookup and the three writes:
`active_transaction()`, `set_flag()`, `set_env()` and `set_attr()`. Installers
call those; none of them re-derives the active transaction from
`jt._torch_compat_install_context`.

- `active_transaction()` returns a ledger only while its state is `open`.
  `InstallTransaction.record()` refuses a committed or rolled-back ledger, and
  the same helpers run at runtime long after installation (`torch.zeros(device=
  "cuda")` reaches `set_flag` through the factory owner), so a closed ledger
  must turn the call into a direct write rather than `RuntimeError: transaction
  is rolled_back`.
- `set_env()` applies `str()` on both the recorded and the direct path. A raw
  integer stored on one path and its text on the other made the rank variables
  fail their own owner check during rollback.

One `use_cuda` write is deliberately outside the ledger: `Module.to(device=...)`
in `installers/nn/module_methods.py` expresses a user request made after
installation, and rolling it back with an install would undo something the
caller asked for. The `torch.backends` TF32 switches no longer write a Jittor
flag at all: they set frontend-owned precision tiers in
`installers/cuda/api.py` (see [float32 precision](../notes/float32-precision-policy.md)).

## Delayed hooks

`runtime_hook(owner)` and the `owned_runtime_hook(owner)` decorator establish the
current recording scope without importing Jittor; `current_transaction()` is
bootstrap-free as well. Attribute writes through `set_attr` or `patch_method`
and module publication through `RuntimeHook.replace_module` belong to that
scope. Cleanup checks the current object's identity, so foreign replacements
are preserved and reported as `TransactionConflict`.

- `release_runtime_hooks(owner)` releases an owner's effects in reverse order; a
  conflict does not stop independent earlier entries from being restored.
- Delayed module-patch callbacks have owners of the form
  `("module_patch", module_name, callback)` and are released by
  `release_module_patch_hooks`. Callback authors must use the mutation helpers
  for class or registry changes: arbitrary third-party side effects cannot be
  inferred or reversed from a module-dictionary snapshot.
- `compat.torch.install(torch, parent_transaction=activation)` transfers a
  successful child ledger with `activation.adopt(child)` before the child
  commits, so a later outer failure still restores classes, modules, completion
  markers and cached tensor state. Fidelity registration records individual API
  records through the same scope.
- `torch.library` registry and metadata writes join the current
  `InstallTransaction` or `RuntimeHook`; `Library` advertises
  `_transactional_registry = True`, and adapters must not add a second snapshot
  undo for the same slots.
- The vLLM adapter's activation owns its extension stand-ins, API-version
  selection, registered operators and installed marker; flash-attention
  publication has a nested, independently releasable owner. Existing foreign
  module slots or operator names are not overwritten.
- The C++ extension facade publishes before executing a module so recursive
  imports see the same object; a failed loader restores the prior owned slot,
  and a foreign replacement is a hard failure. Build products are not deleted
  by module rollback.
- Source-root resolution uses identity-tagged path insertions and a temporary,
  current-thread loader tracker. Rollback touches only recorded publications;
  concurrent path changes and unrelated imports stay intact. Custom source
  loaders publish through `publish_source_module`; an untracked publication
  under the source root is preserved and reported as a hard conflict. The public
  resolver entry is `load()`; helpers that modify import state require its
  active scope.

## What tests may claim

Child-process tests are isolated by `_helpers.child_process.child_env()` and
must verify the child's `PYTHONPATH` and mode variables explicitly. A clean
child environment is not evidence that a failed parent install restored its
state.

Tests may assert rollback of the recorded owners in the table above and child
isolation. They must not claim full install rollback for state outside that
table -- in particular for third-party side effects of module-patch callbacks.

- `compat/tests/torch/test_transaction.py`, `test_install_context.py` and
  `test_runtime_hook_ownership.py` exercise the real ledger, hook,
  extension-publication and source-loader code against controlled state owners;
  they need neither Jittor compilation nor accelerator hardware.
- `compat/tests/structure/test_torch_install_state_boundary.py` pins this page
  and the explicit `set_env`/`set_flag` writes of the distributed installer.
- Integrated activation/context tests remain the control for the complete
  frontend.

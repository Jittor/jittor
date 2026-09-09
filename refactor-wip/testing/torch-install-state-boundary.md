# Torch Install State Boundary

`InstallContext` currently provides a transaction for the `torch*` namespace
only. It does not claim to snapshot `jt.flags`, `os.environ`,
`builtins.__import__`, `sys.meta_path`, module class dictionaries, or the
`module_patcher` registry.

Child-process tests are isolated by `_helpers.child_process.child_env()` and
must verify the child's `PYTHONPATH` and mode variables explicitly. A clean
child environment is not evidence that a failed parent install restored its
state.

The next 7.05 implementation must choose one explicit contract before coding:

- a reversible mutation ledger covering every listed state owner and restoring
  in reverse order; or
- a hard-failure contract that aborts before any irreversible mutation and
  reports the first owner that cannot be rolled back.

Until then, tests may assert namespace rollback and child isolation, but must
not claim full install rollback.

## Snapshot inventory

`jt.flags` is not a complete snapshot source: its native object exposes dynamic
flag attributes, so `flags.__dict__` is not an authoritative enumeration. A
future ledger must receive an explicit allowlist of mutable flags and record
their values through attribute access.

The environment mutation inventory currently includes `JT_NCCL_WORLD_SIZE`,
`JT_NCCL_RANK`, `JT_NCCL_LOCAL_RANK`, `JT_NCCL_ROOTINFO_FILE`, `use_nccl`, and
`use_mpi` in `installers/distributed.py`. The ledger must also account for the
`jt.flags.use_cuda` writes in `installers/core.py`, `tensor.py`, `nn.py`, and
`distributed.py`, restoring flags before environment-dependent teardown.

## One owner for the write helpers

`jittor/compat/transaction.py` owns the lookup and the three writes:
`active_transaction()`, `set_flag()`, `set_env()`, and `set_attr()`. Installers
call those; none of them may re-derive the active transaction from
`jt._torch_compat_install_context` themselves.

Six installers used to inline that lookup, and the copies had drifted. Five of
them tested only whether a transaction *existed*, not whether it was still open.
`InstallTransaction.record()` refuses a transaction that has been committed or
rolled back, so a ledger left behind by an earlier failure turned the next write
into `RuntimeError: transaction is rolled_back`. That is reachable outside
install: `_set_use_cuda` is called from `torch.zeros(device="cuda")`, and
`_mutate_import` from the optional integration steps, long after any ledger has
closed. `active_transaction()` therefore requires `state == "open"` and returns
`None` otherwise, so the write falls back to a direct one.

`set_env()` applies `str()` on both paths. The direct path used to store the raw
object for some callers while `mutate_env` recorded the normalised text, which
made the integer-valued rank variables fail their own owner check at rollback.

Two `use_cuda` writes are deliberately *outside* the install ledger, because
they express a user request at runtime rather than an installation step: the
`Module.to(device=...)` path in `installers/nn.py` and the reduced-precision
switches in `installers/cuda.py` (`torch.backends.cuda.matmul.allow_tf32`).
Rolling those back with an install would undo something the caller asked for
after the install finished.

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

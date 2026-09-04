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

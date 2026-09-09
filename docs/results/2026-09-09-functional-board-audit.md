# Functional board audit (2026-09-09)

## Counting rule

The rendered board contains historical evidence and aggregate rows, so a raw
Markdown line count is not a task count. The latest board header reports **264
rows marked `已合并`**; code/performance and hardware rows are tracked
separately. A historical parser may report 263 because it treats the long 2.19
evidence row as a separate aggregate entry. That is a ledger-format discrepancy,
not a new code gap.

The code evidence below records functional sub-batches; it does not silently
change a task-row status when performance, hardware, or aggregate acceptance is
still open.

This note records the current evidence behind the two rows that were most often
reported as “almost closed”. It does not close either row by itself.

## 2.19 user and internal errors

The Python native and ACL owner scans now have no production user-facing
`assert` left in the scanned API/operator paths. The migrated paths raise
`ValueError`, `TypeError`, `IndexError`, `FileNotFoundError`, or
`NotImplementedError` as appropriate. The remaining assertions are build or
resource invariants, dataset internal state, or documentation examples.

The row remains open until the aggregate C++/CUDA boundary inventory is rerun
and its cross-`pyjt` negative cases are collected in one final report.

## 10.21 import layering and typing

The import-layer lint contract is closed. The typing scope now includes the
clean modules in `python/jittor/build`, `compat/torch`, and the nox entrypoint;
each added module was checked with mypy and `py_compile`. The clean compat
modules include `functional.py`, `fidelity.py`, `context.py`, `library.py`,
`lr_scheduler.py`, `nn_modules.py`, `nn_frontend.py`, `optim_frontend.py`,
`autograd.py`, `distribution_api.py`, and `core_install_api.py`.

The row remains partially merged because `optimizer_api.py` still depends on
the external writable `Var.grad` contract. The `types.py` protocol batch is now
clean and formally in scope; remaining work requires a shared core stub or
Protocol design for optimizer gradient ownership.

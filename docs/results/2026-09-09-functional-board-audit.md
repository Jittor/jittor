# Functional board audit (2026-09-09)

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

The row remains partially merged: the remaining `compat/torch/types.py`
errors depend on the external `jittor_core.Var` type contract (dynamic device
methods and residency markers), while `optimizer_api.py` still depends on the
external writable `Var.grad` contract. These require a shared core stub or
Protocol design before expanding the scope safely.

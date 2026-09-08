# 原生与 Torch 类型系统测试隔离

- Status: fixed; type-system contracts execute in their owning process mode
- Date: 2026-08-21
- Baseline: `2e34fee4`
- Owner: test infrastructure and Torch compatibility maintainers
- Review when: Torch installation, dtype promotion/casts, `TORCH_MODE_PATHS`, or
  complete-suite orchestration changes

## Finding

After native/Torch startup isolation became explicit, two old test modules still
assumed that plain `import jittor` installed Torch semantics. A native run of
`tests/compiler/test_kernel_traps.py` failed two Torch expectations:
`float32 + int64 -> float32` and `.long() -> int64`. The same native process ran
`tests/core/test_type_system.py` with 18 failures, 12 passes, and two skips;
missing `result_type`, `promote_types`, `can_cast`, `.type()`, and Torch cast
aliases accounted for the failures.

These were test-mode errors, not missing native APIs. Native Jittor 2.0 retains
width-based mixed dtype inference and `Var.long = Var.int32`. The compatibility
installer deliberately overrides that surface. Before editing, running the type
system module with `JITTOR_TORCH_SHIM=1` produced 30 passes and two historical
no-op skips, proving that its executable assertions belonged to the Torch
process.

## Change

`tests/core/test_type_system.py` is now listed in `TORCH_MODE_PATHS`, alongside
the existing Torch-default regression module. Broad native collection excludes
it; explicit selection and the complete-suite runner execute it in the Torch
process. A structure test locks that ownership.

The kernel-trap module now asserts the native contracts and points Torch
promotion coverage to `tests/compat/torch/`. Two empty skipped methods that only
repeated historical divergence prose were removed. No executable check was
lost: native behavior is asserted in the trap module, while Torch promotion,
casts, values, and typed-tensor names remain positive tests in the type-system
and compatibility modules.

The maintained test-system document now states that `python -m pytest tests`
means native-only coverage and that `tools/run_test_suite.py` is required for a
combined native plus Torch result.

## Verification

Tests used Python 3.12.13 and an isolated CPU cache below
`$JITTOR_LAB_ROOT/_state/`. This is test ownership and orchestration work; runtime
operator code and backend dispatch did not change.

| Gate | Result |
| --- | --- |
| Pre-change type-system module in native mode | `18 failed`, 12 passed, 2 skipped |
| Pre-change explicit Torch-mode control | `30 passed`, 2 skipped |
| Post-change automatically selected type-system module | `30 passed` |
| Native kernel-trap module | `9 passed`, 29 subtests passed |
| Torch promotion and dtype modules | `45 passed`, 2 skipped |
| Focused pytest-structure contract | `15 passed` |
| Repository layout | passed |
| `tests/structure` | `212 passed`, 2 skipped, 1503 subtests passed |

## Maintained commands

With the environment configured according to `agent/manuals/environment.md`:

```bash
python -m pytest -q tests/compiler/test_kernel_traps.py
python -m pytest -q tests/core/test_type_system.py
python -m pytest -q tests/compat/torch/test_torch_compat_promotion.py \
  tests/compat/torch/test_torch_compat_dtype.py
python tools/run_test_suite.py
```

## Limits

The complete two-session repository suite was not rerun end to end in this
focused change; the native and Torch modules that previously failed were run in
their correct processes, and the complete-suite routing contract is covered
statically. No CUDA, NPU, or ROCm behavior changed or is claimed by this report.

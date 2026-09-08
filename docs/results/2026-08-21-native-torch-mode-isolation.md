# 原生与 Torch 模式启动隔离修复验证

- Status: Verified on CPU composition probes and real CUDA startup
- Date: 2026-08-21
- Baseline: `20af7de3`
- Owner: compatibility composition and Triton bridge maintainers
- Review when: root compatibility composition, deployed Torch entry, or Triton
  detection/bridge activation changes

## Problem and fix

In an environment whose top-level `torch` package was the deployed Jittor shim,
plain CUDA-enabled `import jittor` could silently activate the process-wide Torch
installer despite `JITTOR_TORCH_SHIM=0`. Root composition always imports the
canonical Triton domain. Its automatic real-Triton probe imports the upstream
package, which may import `torch`; the deployed placeholder then re-entered the
installer before native composition finished. Native APIs consequently exposed
Torch return types and defaults without explicit mode selection.

Root composition already decides the mode before importing Triton. It now marks
the short Triton-import window when that decision is native, and the Torch
installer rejects re-entrant activation during that window. The real-Triton
probe can then continue without Torch or fall back to the Jittor Triton shim.
Explicit Torch startup never sets the native marker and retains the original
installation path.

## Verification

A deterministic CPU subprocess regression puts two packages first on
`PYTHONPATH`: a fake real Triton that imports `torch`, and the actual deployed
Torch entry resource. Plain `import jittor` must retain the native median owner
and no Torch install context; explicit `import torch` must still publish Jittor
as the same top-level module with a complete installer graph. This reproduces the
dependency relationship without relying on a particular upstream Triton version.

The original environment reproduction was also rerun with Python 3.11.15, the
deployed Torch shim, CUDA 12.2 and an RTX 4090. Native mode retained
`jittor.misc.tensor_ops` as the median owner, did not register `torch`, and
executed/read a CUDA multiplication. Explicit Torch mode installed the
compatibility owner, created accelerator-resident inputs and outputs, and read
the same CUDA result.

| Gate | Result |
| --- | --- |
| Indirect-vs-explicit deployed entry subprocess regression | `1 passed` |
| Real native CUDA startup and computation | passed |
| Real explicit Torch CUDA startup and computation | passed |
| Installer transaction/idempotence regressions | `18 passed` |
| Shim import-order and identity regressions | `9 passed` |
| `tests/structure` | `211 passed, 2 skipped` |
| Repository layout, syntax, and `git diff --check` | passed |

The guard is about API-mode selection, not which Triton implementation ultimately
loads. An upstream Triton that tolerates unavailable Torch may remain the active
module; another version may fail its probe and yield the Jittor shim. Both are
acceptable as long as plain Jittor remains native and explicit Torch startup
remains functional.

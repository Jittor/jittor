# 浮点 NaN 比较 CPU/CUDA 修复验证

- Status: CPU/CUDA verified; NPU/ROCm real-device verification pending
- Date: 2026-08-21
- Baseline: `8929db65`
- Owner: compiler and binary-operator maintainers
- Review when: compiler optimization flags, fused choices, floating comparison
  code generation, or an unverified accelerator backend changes

## Finding and correction

The original ledger described a same-`Var` equality fold. The minimized CPU
reproduction showed a wider problem: under both fused and `no_fuse=1` execution,
`==` and `!=` mishandled NaN for distinct inputs as well, while same-input `<=`
and `>=` were also folded incorrectly. Generated source used ordinary C++
comparisons, but CPU kernels ended with `-Ofast`, whose finite-math assumptions
permit these transformations. The same pre-fix matrix already produced correct
results on real CUDA.

Floating and complex `BinaryOp` comparisons now add an `-O3` graph choice. On
CPU, its position after `-Ofast` restores IEEE floating semantics while retaining
level-three optimization. `OpCompiler` now obtains flags from the fused graph's
aggregated loop options, so choices from intermediate operators reach the actual
compile command and remain represented in the JIT key. CPU float16 and bfloat16
wrappers also define the previously missing `<=`, `>=`, and `!=` overloads.

## Verification

CPU validation used Python 3.11.15 and an isolated cache below
`$JITTOR_LAB_ROOT/_state/`. CUDA validation used Python 3.12.13, CUDA 12.2, an
RTX 4090, and a separate cache. First JIT work was serialized. CUDA tests selected
the accelerator explicitly and evaluated complete result arrays on the device.

| Gate | Result |
| --- | --- |
| Fused/unfused, same/distinct float32, six comparisons, CPU + real CUDA | `2 passed`, `28 subtests passed` |
| Same-input float16/bfloat16/float32/float64/complex64 matrix, CPU | `1 passed` |
| Same dtype matrix, CPU + real CUDA | `1 passed`, `38 subtests passed` |
| Fusion, compile-options, NaN predicates, and CPU fp16 BinaryOp expansion | `19 passed` |
| Repository layout | passed |
| `tests/structure` after the required JIT refresh rerun | `213 passed` |

The generated CPU command contains `-Ofast -O3` for affected comparison graphs.
The CUDA command accepts the graph choice and the public real-device tests pass;
there is no CPU fallback evidence in the accelerator result.

## Focused performance check

A synchronized CPU microbenchmark compared an 8,388,608-element finite float32
`<` kernel with the candidate `-O3` choice against an explicit legacy `-Ofast`
choice. After warming both JIT entries, three alternating profiler samples had
median times of approximately `2.058 ms` for the candidate and `2.069 ms` for the
legacy path, about `0.5%` faster for the candidate. This supports only the local
comparison-kernel no-regression claim; no downstream model performance claim is
made from this microbenchmark.

## Limitations

No real NPU or ROCm device was available. The known-issues entry remains active
until the same dtype and fused/unfused matrices pass on those backends. A broader
native-mode run of `test_kernel_traps.py` also retained two unrelated pre-existing
Torch-dtype expectation failures (`float32 + int64` and `.long()`); the comparison
and compiler-option selections in that run passed.

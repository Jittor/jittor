# Ascend `arg_reduce` backward 原生执行修复

- Status: verified on a real Ascend 910B3
- Date: 2026-08-30
- Baseline: `139aac05` plus this change
- Owner: ACL backend and reduction maintainers
- Review when: `arg_reduce`, ACL Scatter, custom-function autograd, or the NPU
  gate changes

## Scope

This follow-up closes the maintained float16/float32 `arg_reduce` backward gap.
Validation used Python 3.9.25, NumPy 1.26.4, CANN 9.0.0, and driver version
25.5.1 on Linux aarch64. The selected 910B3 was restricted before Jittor import.
The baseline and fixed runs used separate caches below
`$JITTOR_LAB_ROOT/_state/npu-arg-reduce-grad/`; first compilation was serial.

## Reproduction

The previous value-output gradient was numerically correct only because ACL
fell back to CPU. A two-row max reduction with nonuniform upstream gradients
produced the expected gradient, but its captured execution log contained one
`fallback cpu` and one `compile cpu`. The unsupported fused graph was:

```text
index -> reindex_reduce.add
```

The failure reason was `op index not supported`. This establishes that the old
result was not evidence of NPU training support.

## Fix

The ACL function registry now routes float16/float32 `arg_reduce` through a
custom autograd function. Forward still uses the existing CANN
`aclnnMaxDim`/`aclnnMinDim` runner and preserves Jittor's `(indices, values)`
result. Backward reshapes the selected indices and upstream value gradient to a
singleton reduction dimension, then scatters the gradient into a zero tensor
with the existing CANN Scatter path. Ties therefore retain the forward-selected
first index. Non-ACL execution and other dtypes continue to call the original
2.0 implementation.

The regression covers max and min, dimensions 0, 1, and -1, keepdims on and
off, first-index tie behavior, nonuniform upstream gradients, and both float16
and float32. It requires ACL execution and fails on any CPU compilation or
fallback diagnostic.

## Verification

| Gate | Result |
| --- | --- |
| Baseline focused reproduction | exact values, 1 CPU fallback, 1 CPU compile |
| Fixed focused real-NPU regression | `1 passed`, zero fallback and CPU compile |
| Complete ACL core module | `29 passed` |
| Original CPU arg-reduce forward/backward matrix | `2 passed` |
| NPU OpInfo inventory | `218 passed, 9 skipped` |
| ACL extension inventory | `110 passed, 2 skipped` |
| ACL Torch compatibility | `3 passed` |
| ACL indexing | `2 passed` |
| NPU floor-divide | `2 passed` |
| NPU NaN/Inf predicates | `1 passed` |
| NPU fused NaN comparisons | `1 passed` |
| Maintained NPU inventory total | `366 passed, 11 skipped` |

The NPU inventory was run with `JITTOR_TEST_DEVICES=npu` and the same
file-by-file process boundaries as the maintained Nox session. An exploratory
combined-process run was discarded after cross-file state caused a float64
dropout fallback; the official isolated dropout test and complete ACL extension
file passed.

## Limits

This result establishes float16/float32 value-output backward for the supported
single-axis max/min `arg_reduce` contract. It does not add general ACL float64
support or remove the remaining reduction, `atan2`, FFT, or optional
FlashAttention skips. The broader downstream NPU training matrix remains a
separate goal.

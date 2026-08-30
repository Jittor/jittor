# Diffusers UNet2D Ascend 数值、梯度与性能

- Status: correctness accepted on one real Ascend 910B3; performance not accepted
- Last reviewed: 2026-08-30
- Baseline: `9351cd8d` plus the changes described in this report
- Owner: Torch compatibility and ACL backend maintainers
- Review when: Diffusers, Torch promotion, GroupNorm, SiLU, nearest upsample,
  scaled-dot-product attention, ACL execution, or graph construction changes

## 结论

Diffusers 0.36.0 的维护用 `UNet2DModel` 现在使用相同权重分别在 Jittor ACL
和独立 `torch_npu` 进程中完成前向、loss 和反向。前向输出及 145 个输入/参数
梯度均通过仓库维护阈值，Jittor 捕获窗口内没有 CPU fallback 或 CPU compilation。

这只接受正确性，不接受性能。当前固定协议的 10-repeat 最小值为：

| Runtime | Step | Relative to `torch_npu` |
| --- | ---: | ---: |
| `torch_npu` | 31.053 ms | 1.000x |
| Jittor ACL | 54.750 ms | 1.763x |

Jittor 从本轮优化前约 72.09 ms 降至 54.75 ms，仍不满足“不慢于 PyTorch”的
项目目标。现有 profile 显示主要剩余时间在每步 Jittor forward/backward 图构建，
而不是已提交到 NPU 的算子执行；下一阶段应优化动态图构建和反向图复用。

## 环境与协议

- Device: one Ascend 910B3
- CANN: 9.0.0
- Jittor interpreter: Python 3.9.25, Jittor 1.3.11.0
- Oracle interpreter: Python 3.10.20, PyTorch 2.10.0, torch_npu 2.10.0
- Downstream: Diffusers 0.36.0
- Precision: float32
- Timing: one warmup followed by 10 synchronized repeats, reporting the minimum
- JIT: serial first compilation with isolated `HOME`, `JITTOR_HOME`, `TMPDIR`,
  and `cache_name`
- Runtime state and result tensors: unversioned under
  `$JITTOR_LAB_ROOT/_state/npu-ecosystem/20260830/`

The two interpreters use the same downstream package site and exact Diffusers
origin. The site contains the pure-Python dependency used by this case; the
cross-ABI override must not be generalized to sites containing incompatible
CPython-minor-specific extensions.

The result runner reports 146 tensors on each side: one output plus 145
gradients. Current normalized divergences are:

| Measurement | Result | Maintained threshold |
| --- | ---: | ---: |
| Forward output | 0.0005214 | 0.005 |
| Worst gradient | 0.0021763 | 0.02 |

The worst gradient is `grad::mid_block.attentions.0.to_q.bias`. Jittor reports
`has_acl/use_acl/use_cuda=true`, `fallback_count=0`, and
`cpu_compile_count=0`. The losses are 12.708706 for Jittor and 12.712467 for
`torch_npu`; the full tensor comparison, rather than loss proximity alone, is
the correctness criterion.

## Findings and fixes

The initial model ran on ACL but spent most of the step in decomposed graph
construction and exposed one dtype promotion mismatch. This batch adds the
following verified routes:

1. Python-float true division keeps the target float dtype on ACL. CPU keeps
   the existing PyTorch-compatible wide calculation and exact float32 rounding.
2. Nearest-neighbor 2-D interpolation and its backward route to native
   `aclnnUpsampleNearest2d` operations for verified 4-D floating inputs.
3. Module and functional GroupNorm route through the existing backend hook to
   native `aclnnGroupNorm` forward/backward for verified float32 shapes.
4. Float32 SiLU routes through the native ACL operation.
5. Training SDPA uses the fused ACL path for the verified float32, equal-head,
   unmasked, noncausal, zero-dropout subset. Unsupported combinations retain the
   existing decomposition instead of being claimed as native support.

The new C++ runners stop after workspace-query failures and own temporary ACL
arrays through RAII. Tests execute the NPU candidate before entering a CPU
reference scope because nested `flag_scope` use does not restore the enclosing
device flags in this runtime.

## Verification

The maintained NPU targets were run as separate pytest processes, matching
`noxfile.py`'s process-mode contract:

```text
tests/backends/npu/test_acl.py: 33 passed
tests/backends/npu/test_acl_torch_compat.py: 10 passed
tests/backends/npu/test_aclop.py: 110 passed, 2 skipped
tests/backends/npu/test_acl_indexing.py: 2 passed
tests/ops/test_ops.py: 220 passed, 7 skipped
NPU floor-divide: 2 passed
NPU NaN kernel trap: 1 passed
NPU fused NaN comparison: 1 passed
Total: 379 passed, 9 skipped
```

Additional current-worktree verification:

```text
CPU Torch promotion: 24 passed
Diffusers Jittor ACL vs torch_npu parity: 1 passed in 163.47s
Structure excluding two external-worktree scans: 217 passed, 2 skipped,
2 deselected
```

Running native and Torch-mode targets in one pytest process is invalid: Torch
compatibility changes reduction return types process-wide. The maintained NPU
gate deliberately invokes each target in its own process.

The complete structure run reports `217 passed, 2 skipped, 2 failed`. Both
failures scan pre-existing `.claude/worktrees` containing retired documentation
trees. `check_repo_layout.sh` reports the same worktrees, the user's untracked
root `TODO.md`, and ignored stale `python/jittor.egg-info` metadata. None belongs
to this change; they were not removed or staged.

## Boundaries

- The correctness claim covers this maintained float32 `UNet2DModel` training
  step, not every Diffusers pipeline, scheduler, dtype, attention mask, or model.
- Nearest upsample, GroupNorm, SiLU, and training SDPA keep explicit verified
  guards; unsupported inputs fall back to the pre-existing Jittor graph, not CPU.
- The current 1.763x ratio fails the broader performance target. No NPU
  Diffusers performance acceptance is claimed.
- ms-swift, verl, vLLM, and TRELLIS NPU coverage remains separate work.

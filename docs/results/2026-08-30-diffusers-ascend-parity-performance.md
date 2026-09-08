# Diffusers UNet2D Ascend 数值、梯度与性能

- Status: correctness and maintained-case performance accepted on one real Ascend 910B3
- Last reviewed: 2026-08-31
- Baseline: `0ad9b903` plus the changes described in this report
- Owner: Torch compatibility and ACL backend maintainers
- Review when: Diffusers, Torch promotion, GroupNorm, SiLU, nearest upsample,
  scaled-dot-product attention, ACL execution, or graph construction changes

## 结论

Diffusers 0.36.0 的维护用 `UNet2DModel` 现在使用相同权重分别在 Jittor ACL
和独立 `torch_npu` 进程中完成前向、loss 和反向。前向输出及 145 个输入/参数
梯度均通过仓库维护阈值，Jittor 捕获窗口内没有 CPU fallback 或 CPU compilation。

同一固定协议现在也满足维护用例的性能目标。10-repeat 同步最小值为：

| Runtime | Step | Relative to `torch_npu` |
| --- | ---: | ---: |
| `torch_npu` | 31.053 ms | 1.000x |
| Jittor ACL | 29.939 ms | 0.964x |

Jittor 从首轮 ACL 专项优化前约 72.09 ms 降至原生算子接入后的 54.75 ms，
再通过移除热路径中的 Python autograd 和输出分配开销降至 29.939 ms。在本协议下，
它比 31.053 ms 的 `torch_npu` 参考快 3.59%。该结论只覆盖此维护模型、精度、
形状和训练步骤。

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
| Forward output | 0.0005214 | 0.002 |
| Worst gradient | 0.0021763 | 0.01 |

The worst gradient is `grad::mid_block.attentions.0.to_q.bias`. Jittor reports
`has_acl/use_acl/use_cuda=true`, `fallback_count=0`, and
`cpu_compile_count=0`. The losses are 12.708706 for Jittor and 12.712467 for
`torch_npu`; the full tensor comparison, rather than loss proximity alone, is
the correctness criterion.

## Findings and fixes

The initial model ran on ACL but spent most of the step in decomposed graph
construction and exposed one dtype promotion mismatch. The complete verified
route now includes:

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
6. `CodeOp` supports one joint backward invocation returning gradients for
   multiple inputs. ACL convolution with bias, GroupNorm, concat, nearest
   upsample, matmul, transpose, SiLU, getitem, and FlashAttention use native
   CodeOp gradients instead of Python `Function.grad` graph construction where
   their verified subsets permit it.
7. ACL CodeOps construct outputs directly from known shape and dtype metadata,
   removing temporary `empty` graph nodes from the model hot path. Native Jittor
   Functions also bypass Torch-style context bookkeeping while Torch-style
   custom autograd Functions retain it.
8. Basic unit-stride slice gradients stay lazy, zero-initialize their destination
   inside the ACL runner, and avoid a Python-side synchronization.
9. The O(N^2) DFT constants used by the compatibility FFT implementation are
   retained in a bounded, backend-aware cache. This fixes an async rFFT lifetime
   failure exposed by the full NPU operator order and avoids rebuilding identical
   matrices.

The new C++ runners stop after workspace-query failures and own temporary ACL
arrays through RAII. Tests execute the NPU candidate before entering a CPU
reference scope because nested `flag_scope` use does not restore the enclosing
device flags in this runtime.

## Verification

The maintained NPU targets were run as separate pytest processes, matching
`noxfile.py`'s process-mode contract:

```text
tests/backends/npu/test_acl.py: 34 passed
tests/backends/npu/test_acl_torch_compat.py: 10 passed
tests/backends/npu/test_aclop.py: 112 passed, 2 skipped
tests/backends/npu/test_acl_indexing.py: 4 passed
tests/ops/test_ops.py: 220 passed, 7 skipped
NPU floor-divide: 2 passed
NPU NaN kernel trap: 1 passed
NPU fused NaN comparison: 1 passed
Total: 384 passed, 9 skipped
```

Additional current-worktree verification:

```text
CPU CodeOp and Torch autograd: 42 passed, 3 skipped
CPU Torch-compatible FFT/complex/einsum: 43 passed
Diffusers final runner: 146/146 tensors, 145 gradients, zero fallback,
29.939 ms minimum over 10 synchronized repeats
Structure: 217 passed, 2 skipped, 2 failed
```

Running native and Torch-mode targets in one pytest process is invalid: Torch
compatibility changes reduction return types process-wide. The maintained NPU
gate deliberately invokes each target in its own process.

The two structure failures scan pre-existing `.claude/worktrees` containing
retired documentation trees. `check_repo_layout.sh` reports the same external
worktrees, generated `python/jittor.egg-info` metadata, and the user's untracked
root `TODO.md`. None belongs to this change; they were not removed or staged.

## Boundaries

- The correctness claim covers this maintained float32 `UNet2DModel` training
  step, not every Diffusers pipeline, scheduler, dtype, attention mask, or model.
- Nearest upsample, GroupNorm, SiLU, and training SDPA keep explicit verified
  guards; unsupported inputs fall back to the pre-existing Jittor graph, not CPU.
- The `0.964x` result accepts only the maintained float32 UNet2D training step;
  it is not a blanket Diffusers or NPU performance claim.
- ms-swift, verl, vLLM, and TRELLIS NPU coverage remains separate work.

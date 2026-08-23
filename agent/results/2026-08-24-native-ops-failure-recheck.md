# 原生 ops 历史失败复验与收口

- Status: CPU/real-CUDA test gates accepted
- Last reviewed: 2026-08-24
- Commits: `3a1d6a12`, `33b7db56`, `384adab9`
- Owner: operator and CUDA runtime maintainers
- Review when: runtime storage reuse, scalar ArrayOp fusion, AMP test setup, or unary
  dtype policy changes

## 结论

历史 `tests/ops` 失败清单混合了三类问题：已修复的跨模块 device/Torch-shim
污染、引用 runtime storage 的测试快照，以及强制 CUDA 下 fused scalar `ArrayOp`
缺少 capability 标记。当前独立原生进程已清理该清单中的 argsort、merge、matmul、
transpose、unary 和 where 模块。

- CUTT transpose 输出正确；旧测试先保存 `a.data` view，再同步 `b.data`，runtime
  storage 复用使 reference 随后被覆盖。测试现在立即复制输入和输出。
- `_force_fuse` 标量 `ArrayOp` 由生成 kernel 在 CPU/CUDA 内联发射，但默认 host
  allocator 下只声明 CPU capability；`use_cuda=2` 因而正确拒绝了这条隐式 fallback。
  两个 ArrayOp 构造入口现在都声明 fused scalar CUDA capability。
- fp16 `erfinv`/`safe_clip` 实际 CPU/CUDA 误差分别不超过 `5.30e-4`、`9.77e-5`；
  测试改为对真正的 SciPy `erfinv` 输出，并只对 AMP 分支使用 `1e-3` 容差。
  CPU fp16 fixture 也恢复了明确的 device 和 AMP 状态管理。
- where 的旧 `test_doc` 失败在原生独立进程未复现。

## 验证

- `tests/ops/test_argsort_op.py`：`5 passed`。
- `tests/ops/test_merge_single_array_op.py`：`9 passed`。
- `tests/ops/test_transpose_op.py`：`17 passed`，真实 CUDA CUTT。
- `tests/ops/test_unary_op.py`：`20 passed`，覆盖 CPU/CUDA 与常规/fp16 AMP。
- `tests/ops/test_where_op.py`：`18 passed`。
- `tests/backends/cuda/test_cuda.py`：`6 passed, 1 skipped`。
- 强制 CUDA 共享回归：binary `36 passed`，ternary `6 passed`。
- 布局检查通过；`tests/structure`：`218 passed`。

matmul 的 float64 cuBLAS 根因和 `13 passed` 完整模块结果单独记录在
[`2026-08-24-cuda-float64-matmul.md`](2026-08-24-cuda-float64-matmul.md)。

所有 CUDA 验证均使用 `use_parallel_op_compiler=0`、隔离 cache，并在真实 RTX 4090
上执行；强制 CUDA 用例使用 `use_cuda=2`，不允许 CPU fallback。

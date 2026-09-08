# arg_reduce 测试快照 alias 修复

- Status: CPU/CUDA test gate accepted
- Last reviewed: 2026-08-24
- Commit: `212954bd`
- Owner: operator test maintainers
- Review when: runtime storage ownership or arg-reduce output capture changes

## 结论

`tests/ops/test_arg_reduce_op.py` 原先直接保存 `Var.data` 视图作为 NumPy reference；
后续 CUB output 同步可能复用/覆盖 runtime storage，造成随机失败。测试现在立即用
`np.array(..., copy=True)` 捕获输入、index 和 value snapshots。CUDA CUB kernel 本身
保持不变。

## 验证

- 真实 RTX 4090 新进程运行完整模块：`4 passed in 3.99s`。
- 覆盖 CPU/CUDA forward、min/max、keepdims True/False 和 backward。
- 独立 copy 后的 NumPy reference 与 CUDA CUB 输出逐样本一致。

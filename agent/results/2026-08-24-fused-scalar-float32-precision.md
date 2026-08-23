# 融合标量 float32 精度语义修复

- Status: CPU/CUDA accepted
- Last reviewed: 2026-08-24
- Commit: `21166f28`
- Owner: compiler numerical-semantics maintainers
- Review when: fused loop options or float32 arithmetic contract changes

## 结论

当 binary add/subtract 的两侧都是 float32 且至少一侧是 scalar，JIT loop 现在使用
`-O3` 而不是 `-Ofast`，禁止编译器跨逐步 add/subtract 链重排或收缩。普通向量运算、
其他 dtype 和比较运算路径不变。

这修复了 `(x + 1e8) - 1e8` 中严格 float32 应丢失的小数项：CPU/CUDA 现在均与
NumPy/Torch 的逐步 float32 结果一致。

## 验证

- CPU 定向 precision regression：由 strict `expectedFailure` XPASS 为正常通过。
- 真实 CUDA 定向 precision regression：同样 XPASS 为正常通过。
- 完整 `tests/core/test_edge_cases.py`：`24 passed`。
- `tests/structure` 与仓库布局门禁在相邻提交中通过，均为 `218 passed` / OK。

`KI-SEMANTICS-002` 已从 known-issues ledger 移除；NPU/ROCm 未在本机验证。

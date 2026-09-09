# 空轴 mean NaN 语义修复

- Status: CPU/CUDA accepted
- Last reviewed: 2026-08-23
- Commit: `2c45e558`
- Owner: reduction semantics maintainers
- Review when: reduction initialization or floating-point empty-tensor semantics change

## 结论

空 reduction 的 `sum` 仍返回加法 identity `0`；`mean` 现在遵循 NumPy/Torch 的
`0 / 0 -> NaN` 语义。JIT reduction key 对空 mean 记录字面量分支，在初始化输出时
写入 NaN；CUDA 使用 `::nanf("")`，避免 host-only `numeric_limits::quiet_NaN()`
被 NVCC 拒绝。非空 mean 和其他 reduction 路径不变。

## 验证

- CPU/CUDA `tests/core/test_edge_cases.py -k mean_over_empty_axis`：各 `1 passed`。
- 覆盖 `float32`、`float64`，形状 `(2, 0)` 按 dim、`(0, 3)` 按 dim，以及全空
  `(0,)` reduction；全空输出仍保持 Jittor 的 `(1,)` shape。
- 完整 `tests/core/test_edge_cases.py`：`23 passed, 1 xfailed`；唯一 xfail 是既有
  fused scalar float32 精度契约。
- CPU/CUDA 均在真实 device/path 执行，未接受 fallback。

`KI-SEMANTICS-001` 已从 known-issues ledger 移除；NPU/ROCm 未在本机验证。

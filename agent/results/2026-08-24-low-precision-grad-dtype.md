# 低精度 elementwise backward dtype 修复

- Status: CUDA accepted
- Last reviewed: 2026-08-24
- Commit: `747e2eee`
- Owner: dtype/autograd maintainers
- Review when: low-precision promotion or elementwise backward construction changes

## 结论

fp16/bfloat16 elementwise backward 现在保持目标输入 dtype。`TernaryOp::grad` 会把
upstream gradient 和 zero branch 对齐到目标输入 dtype；`BinaryOp::grad` 的非复数
multiply 也会在构造结果后恢复目标 dtype。这覆盖了 scalar loss gradient 常见的
float32 上游路径，同时保持 float32/float64 和 complex 分支原有行为。

## 验证

- `tests/backends/cuda/test_low_precision.py`：`2 passed`。
- 覆盖 fp16/bf16 的 ReLU、ReLU + scalar multiply、direct scalar multiply，并检查
  forward 和 gradient dtype。
- 修复前的 `expectedFailure` 已 XPASS，随后转为正常断言并通过。
- CUDA 计算在真实 RTX 4090 上执行；NPU/ROCm 未在本机验证。

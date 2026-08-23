# CPU DepthwiseConv backward 修复

- Status: CPU/CUDA accepted
- Last reviewed: 2026-08-23
- Commit: `7666e91b`
- Owner: neural-network and autograd maintainers
- Review when: depthwise dispatch or grouped-convolution backward changes

## 结论

`jittor.nn.DepthwiseConv` 在 CPU 上原本只执行 grouped `conv2d` forward，却仍由
CUDA 专用 `Function.grad` 接管 backward，导致 `AttributeError: save_vars`。现在 CPU
调用直接绕过自定义 Function tape，返回 grouped `conv2d`，由其维护的 CPU backward
计算输入和权重梯度；CUDA 继续使用原有 custom kernel。

## 验证

- CPU `tests/nn/test_depthwise_conv.py -k 'cpu_fallback or cpu_backward'`：`2 passed,
  1 deselected`。
- CPU 输入和权重梯度均与独立 NumPy 解析梯度逐元素比较，`atol=rtol=1e-6`。
- 真实 RTX 4090 CUDA 小形状 forward/backward smoke：`CUDA_DEPTHWISE_OK`，输出、输入
  梯度和权重梯度 shape 正确且全部有限。
- 仓库结构门禁在相邻阶段通过：`tests/structure` `218 passed`，布局检查通过。

NPU/ROCm 未在本机验证；本修复不改变这些后端路径。

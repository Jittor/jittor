# 验证结论

这一节保存**可复现的维护者验证结论**：一个具体的现象、它的根因、修复，以及修复
之后在真实设备上量到的数字。写的是"我们查过什么、结果怎么样"，不是项目历史，也
不是教程——想让某个模型或后端跑起来、想对照某个已知问题的结论时看这里。

每条结论都注明状态、日期、基线提交和复查条件。过期的结论会被删除或归档，不在多
个文件里重复维护。

```{toctree}
:maxdepth: 1

2026-09-12-minimax-h3-torch-compat
2026-09-12-cuda-metaop-launch-index-scalar
2026-09-13-host-path-per-node-cost
2026-09-14-autocast-conv-mixed-dtype
2026-09-14-exit-heap-corruption
2026-09-14-jittor-vs-pytorch
2026-09-14-vllm-omni-h3-enablement
```

## 按主题索引

| 结论 | 状态 | 日期 |
| --- | --- | --- |
| [MiniMax-H3 在 Torch 兼容层下跑通](2026-09-12-minimax-h3-torch-compat.md) | 端到端跑通，达到 parity 速度 | 2026-09-12，2026-09-14 复验 |
| [autocast 请求了混合 dtype 的卷积](2026-09-14-autocast-conv-mixed-dtype.md) | 已修复，真实 CUDA 设备验证 | 2026-09-14 |
| [退出期 "corrupted double-linked list"](2026-09-14-exit-heap-corruption.md) | 已修复，多次真实运行验证 | 2026-09-14 |
| [jittor + torch-compat + vLLM-Omni 接入 H3 暴露的 shim 缺陷](2026-09-14-vllm-omni-h3-enablement.md) | 完整请求跑通且速度正常（710.7 s → 39.6 s） | 2026-09-14，2026-09-15 解决速度 |
| [CUDA 元算子：发射配置、strided 下标与标量常量融合](2026-09-12-cuda-metaop-launch-index-scalar.md) | 部分完成，H20 上实测 | 2026-09-12 |
| [主机受限步长：把每算子的 Python 开销从图构建里拿掉](2026-09-13-host-path-per-node-cost.md) | Python 路径已完成 | 2026-09-13 |
| [Jittor vs 真 PyTorch 2.9.1：现在差在哪](2026-09-14-jittor-vs-pytorch.md) | 进行中，赢 6 平 5 输 1 | 2026-09-14 |

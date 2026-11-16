# 注意力 API

[本栏目目录](index.md) · [English](nn-attention.en.md)

`jittor.nn.attention` 提供注意力算子，包括缩放点积注意力及其融合实现。

选择注意力接口时，先核对输入形状、mask、dtype 与设备要求。融合实现的可用性取决于后端，名称存在不等于任意设备都能执行。

配置 `JITTOR_FLASH_ATTN_JITTOR_SRC` 后，可直接编译 FlashAttention 官方 CUDA 源码。
该后端使用 FP16/BF16；FP32 不会默认降精度。普通执行支持前向和已实现的自定义反向。
CUDA graph 目前支持无梯度、无 dropout、对齐 head 维度的 dense 调用，包括 GQA 与
矩形 causal attention；输入和输出由图节点管理。其它扩展调用会明确拒绝整步捕获并
正常执行 Python 路径，避免重放与新输入无关的旧结果。实际可用范围仍取决于编译的
head/dtype、官方源版本和目标设备；导入成功不代表这些功能都已验证。

:::{autopublicmodule} jittor.nn.attention
:::

## 相关页面

- [神经网络 API](nn.md)
- [核心 API](core.md)
- [快速开始](../getting-started/quickstart.md)

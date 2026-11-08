# 注意力 API

[本栏目目录](index.md) · [English](nn-attention.en.md)

`jittor.nn.attention` 提供注意力算子，包括缩放点积注意力及其融合实现。

选择注意力接口时，先核对输入形状、mask、dtype 与设备要求。融合实现的可用性取决于后端，名称存在不等于任意设备都能执行。

:::{autopublicmodule} jittor.nn.attention
:::

## 相关页面

- [神经网络 API](nn.md)
- [核心 API](core.md)
- [快速开始](../getting-started/quickstart.md)

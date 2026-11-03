# 优化器 API

[本栏目目录](index.md) · [English](optim.en.md)

优化器在 `jittor.optim`，学习率调度在 `jittor.lr_scheduler`。

原生训练通常把 loss 交给优化器的 `step(loss)`；Torch 兼容训练使用另一套调用约定，见[Torch 使用指南](../compatibility/torch.md)。

:::{autopublicmodule} jittor.optim
:::

## 学习率调度

:::{autopublicmodule} jittor.lr_scheduler
:::

## 相关页面

- [神经网络 API](nn.md)
- [核心 API](core.md)
- [快速开始](../getting-started/quickstart.md)

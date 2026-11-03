# API 参考

[文档首页](../index.md) · [English](index.en.md)

按任务查找原生 Jittor 的模块与函数。页面中的参数和公开对象来自构建文档时安装的包，具体可用性还取决于你的版本和设备。已有 Torch 代码请读[Torch 兼容指南](../compatibility/torch.md)。

- 张量与求导：[核心](core.md)。
- 模型与训练：[网络层](nn.md)、[注意力](nn-attention.md)、[模型库](models.md)、[初始化](init.md)、[优化器与调度器](optim.md)。
- 数据准备：[数据集](dataset.md)、[数据变换](transform.md)。
- 专项计算：[线性代数](linalg.md)、[概率分布](distributions.md)、[三维损失](loss3d.md)、[贡献工具](contrib.md)。

还没写过训练脚本时，先看[快速开始](../getting-started/quickstart.md)。

:::{autosummary}
jittor
jittor.nn
jittor.nn.attention
jittor.models
jittor.optim
jittor.lr_scheduler
jittor.init
jittor.contrib
jittor.dataset
jittor.transform
jittor.linalg
jittor.distributions
jittor.loss3d
:::

<!-- sphinx-only
:::{toctree}
:hidden:
:maxdepth: 2

core
nn
nn-attention
models
optim
init
contrib
dataset
transform
linalg
distributions
loss3d
:::
-->
# 运行已有 Torch 代码

[本栏目目录](index.md) · [English](torch.en.md)

这页帮助你先跑通一个 CPU 训练循环，再决定如何迁移已有脚本。请先按[安装说明](../getting-started/installation.md)安装配套的 Jittor 和 `jittor-compat`。下游库的可用功能取决于具体版本、模型和设备，需查对应集成仓库的支持范围。

## 一个完整的 CPU 示例

在新的 Python 进程中运行。激活必须发生在导入真实 PyTorch 之前：

```python
import jittor as jt
jt.flags.use_cuda = 0

from jittor.compat.shim import activate
activate()
import torch

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
x = torch.tensor([[0.0], [1.0], [2.0]])
y = 2 * x + 1

for step in range(100):
    optimizer.zero_grad()
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(step, loss.item())
```

`activate()` 为整个进程选择兼容前端，不是一个局部上下文。需要独立 PyTorch 参考结果时，请在另一个环境和进程中运行。

## 迁移时先检查这几处

| 项目 | 使用方式 |
| --- | --- |
| 导入 | 普通 `import jittor as jt` 保持原生行为。`import jittor as torch` 只是 Python 别名，不会激活兼容层。 |
| 训练循环 | Torch 模式用 `zero_grad()`、`backward()`、`step()`；原生 Jittor 常用 `step(loss)`。 |
| 包安装 | `jittor` 与 `jittor-compat` 是两个发行物。仅添加 `PYTHONPATH=python` 不会安装兼容包。 |
| 第三方库 | 使用独立集成仓库要求的版本；不要把某个小模型通过理解为整库可用。 |
| 设备与精度 | 分别验证目标芯片、dtype、前向、反向与训练；CPU 通过不能替代加速器验证。 |

旧的 `activate(independent_namespace=False)` 与 `JITTOR_TORCH_INDEPENDENT=0` 模式已移除。保留正常的 `activate()` 调用，删除旧开关；Torch 前端与原生 Jittor 共享运行时，但不是同一个 Python 模块。

## 让现有应用直接导入 torch

兼容 wheel 提供 `torch` 入口；需要部署随包 stub 时，运行：

```bash
jittor-torch-shim
jittor-torch-shim --check
```

命令会改动当前环境的包文件，应使用为该应用准备的独立环境。安装来源、入口优先级和排查方法见[安装与部署](torch-shim.md)。

## 设备与混合精度

Jittor 的设备执行方式与原生 PyTorch 不完全相同。当前兼容层的 `cuda` 接口也可映射到 ACL 加速后端；因此 `torch.cuda.is_available()` 的结果不能单独说明芯片厂商。NPU 的具体配置与支持状态应以对应后端和集成文档为准。

autocast 基于 Jittor 的精度控制，不是 PyTorch 的逐算子白名单；某些算子和嵌套设备上下文的行为存在差异。迁移时先保留 float32 基线，再逐项检查混合精度输出、梯度和训练曲线。不要根据显存占用大小判断算子是否在目标设备执行。

## 使用下游库

[JittorEcosystem](https://github.com/JittorEcosystem) 的每个集成仓库维护自己的安装组合、官方用例入口、测试与功能/芯片支持表。优先查明目标任务所需的功能是否已验证，再复用该版本的官方教程。

发现问题时保留最小脚本、包版本、芯片与完整错误日志。导入、编译、数值不一致和性能问题需要不同的复现材料；验收标准见[兼容原则](../development/compatibility/principles.md)。

继续学习：[Torch API 教程](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/torch_compat.md)、[脚本迁移教程](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/torch_compat_migration.md)。

# 使用 Torch 兼容接口

[文档首页](../index.md) · [English](index.en.md)

已有 PyTorch 代码时，可以通过 `jittor-compat` 使用 Torch 风格的张量、模型和优化器接口，计算由 Jittor 执行。先验证你要用的 API，再按下游库的固定版本和芯片支持记录选择集成。

- 第一次尝试：先[安装兼容包](../getting-started/installation.md)，再跑[CPU 训练示例](torch.md)。
- 需要让应用直接 `import torch`：[安装与部署](torch-shim.md)。
- 使用 vLLM、Transformers 等库：查看 [JittorEcosystem](https://github.com/JittorEcosystem) 对应仓库的版本要求、教程与支持范围。

这些接口不是所有 PyTorch 版本、模型与设备的无条件替代。成功导入一个库，也不代表它的训练、分布式或加速器路径已通过验证。

```{toctree}
:hidden:

torch
torch-shim
```

## 修改兼容实现

维护者先读[兼容原则](../development/compatibility/principles.md)，再读[分层架构](../development/compatibility/torch-architecture.md)和[调用下沉](../development/compatibility/torch-lowering.md)。这些文档说明代码归属和验收规则，运行普通应用不需要先读它们。

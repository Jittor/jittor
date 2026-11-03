# Jittor 文档

[English](index.en.md)

Jittor 是一个基于即时编译的深度学习框架。你可以使用原生 Jittor API，
或通过 `jittor-compat` 运行符合其支持范围的 PyTorch 代码。

## 开始使用

第一次使用请按顺序阅读：

1. [安装与环境](getting-started/installation.md)：选择安装方式，确认解释器与编译器。
2. [第一个训练示例](getting-started/quickstart.md)：在 CPU 上完成前向、反向和参数更新。
3. [继续学习](getting-started/index.md)：选择原生 Jittor 或 Torch 兼容教程。

已有 PyTorch 项目，请从 [Torch 兼容](compatibility/index.md) 开始；
接入 vLLM、Transformers 等库时，还需核对 [JittorEcosystem](https://github.com/JittorEcosystem)
中对应项目的版本和设备支持。

## 按任务查找

| 你要做什么 | 阅读入口 |
| --- | --- |
| 配置硬件、分布式训练或排查运行问题 | [使用指南](guides/index.md) |
| 查看函数、类和参数 | [API 参考](api/index.md) |
| 理解设备、精度和惰性执行 | [核心概念](notes/index.md) |
| 分析耗时、显存或对比性能 | [性能优化](performance/index.md) |
| 修改框架、添加功能或运行测试 | [开发指南](development/index.md) |
| 了解升级变化、社区项目和研究方向 | [项目资料](project/index.md) |

## 使用前确认

[平台支持](guides/hardware/platform-support.md) 区分持续测试与历史验证；
[已知限制](guides/known-limitations.md) 列出使用时需要注意的边界。
历史测量只适用于原记录的提交与环境，不代表当前版本已经重新验收。

```{toctree}
:hidden:
:maxdepth: 2
:caption: 使用 Jittor

getting-started/index
guides/index
compatibility/index
performance/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: 参考与开发

api/index
notes/index
development/index
project/index
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Language

index.en
```

[源码](https://github.com/Jittor/jittor/tree/2.0-refactor) ·
[问题反馈](https://github.com/Jittor/jittor/issues) ·
[社区论坛](https://discuss.jittor.org/)

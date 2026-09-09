# Jittor 文档

Jittor（计图）是一个基于即时编译和元算子的深度学习框架。前端使用 Python 动态图
接口，后端把算子编译成针对实际负载调优的 C++ 与 CUDA 代码。

本站包含安装与上手、教程、API 参考、使用指南、Torch 兼容说明、框架机制说明，
以及面向贡献者的开发文档。

```{toctree}
:maxdepth: 2
:caption: 开始使用

guides/index
tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: 参考

api/index
compatibility/index
notes/index
performance/index
```

```{toctree}
:maxdepth: 2
:caption: 项目

development/index
releases/index
community/index
contributing
```

## 从哪里开始

| 你想做的事 | 去哪里 |
| --- | --- |
| 装好并跑通第一个模型 | [安装与快速开始](https://github.com/Jittor/jittor#install)、[教程](tutorials/index.md) |
| 查某个函数怎么用 | [API 参考](api/index.md) |
| 把 PyTorch 代码跑在 Jittor 上 | [Torch 兼容](compatibility/index.md) |
| 在昇腾、天数或多卡上运行 | [使用指南](guides/index.md) |
| 弄清显存、精度或异步执行的行为 | [机制说明](notes/index.md) |
| 调查一个报错或性能问题 | [调试指南](guides/debugging.md)、[性能](performance/index.md) |
| 给 Jittor 提交代码 | [贡献指南](contributing.md)、[开发文档](development/index.md) |

## 项目链接

- [官网](https://cg.cs.tsinghua.edu.cn/jittor/)
- [源码仓库](https://github.com/Jittor/jittor)
- [问题追踪](https://github.com/Jittor/jittor/issues)
- [论坛](https://discuss.jittor.org/)

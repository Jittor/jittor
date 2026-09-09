# 开发文档

面向 Jittor 贡献者：代码怎么组织、测试怎么跑、当前有哪些已知缺陷。

先读仓库根目录的 [`CONTRIBUTING.md`](https://github.com/Jittor/jittor/blob/master/CONTRIBUTING.md)
和 [贡献指南](../contributing.md)，再按需要查下面的文档。

```{toctree}
:maxdepth: 1

repository-layout
source-architecture
test-system
known-issues/parallel-compiler-segfault
```

| 文档 | 内容 |
| --- | --- |
| [仓库布局](repository-layout.md) | 顶层目录的职责与打包边界 |
| [源码架构](source-architecture.md) | Python 侧的模块归属与依赖方向 |
| [测试体系](test-system.md) | 测试分层、门禁与验证口径 |
| [已知问题](known-issues/parallel-compiler-segfault.md) | 并行编译器可能破坏进程状态 |

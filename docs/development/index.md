# 开发文档

面向 Jittor 贡献者：代码怎么组织、测试怎么跑、当前有哪些已知缺陷。

已知缺陷的**总账**是仓库里的
[`agent/manuals/known-issues.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)：
每个条目带严重度、证据、workaround 和退出条件。`known-issues/` 下只放那些需要
长篇调查记录的条目，目前只有一个。

先读仓库根目录的 [`CONTRIBUTING.md`](https://github.com/Jittor/jittor/blob/master/CONTRIBUTING.md)
和 [贡献指南](../contributing.md)，再按需要查下面的文档。

```{toctree}
:maxdepth: 1

repository-layout
source-architecture
storage-layout
jit-operator-source
error-categories
async-error-diagnostics
test-system
known-issues/parallel-compiler-segfault
```

| 文档 | 内容 |
| --- | --- |
| [仓库布局](repository-layout.md) | 顶层目录的职责与打包边界 |
| [源码架构](source-architecture.md) | Python 侧的模块归属与依赖方向 |
| [存储布局契约](storage-layout.md) | stride 与存储归原生所有，前端不维护第二份 |
| [JIT 算子源码契约](jit-operator-source.md) | 算子源文件被编译两遍，`jit_run` 要同时满足 C++ 与 KernelIR |
| [错误分级](error-categories.md) | 用户错误与内部不变量各走哪个入口 |
| [异步错误诊断契约](async-error-diagnostics.md) | 有界发射记录环，与图元数据的区别 |
| [测试体系](test-system.md) | 测试分层、门禁与验证口径 |
| [并行编译器段错误](known-issues/parallel-compiler-segfault.md) | `KI-COMPILER-001` 的调查记录：症状、复现协议与验收门槛 |

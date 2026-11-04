# 开发与维护

[文档首页](../index.md) · [English](index.en.md)

按你要做的改动阅读。先按[贡献指南](contributing.md)准备环境，再用下面前三篇
确定代码放哪里、该跑什么测试。专题页是按需查阅的实现契约，不要求从头读完。

## 开始一次改动

- [仓库、打包与资源边界](repository-layout.md)
- [查找源码归属](source-architecture.md)
- [选择并运行测试](testing/test-system.md)

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

repository-layout
source-architecture
testing/test-system
```
-->

## 修改核心或算子

- [运行时状态与图提交](runtime/native-runtime.md)
- [存储、stride 与别名](runtime/storage-layout.md)
- [算子注册、分派与回退](runtime/operator-dispatch.md)
- [编写 JIT 算子源码](runtime/jit-operator-source.md)
- [后端构建与资源打包](backends/backend-build.md)
- [ACL 发射器、属性与描述符](backends/acl-backend-contracts.md)
- [图回放与训练步捕获](runtime/static-graph-replay.md)

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

runtime/native-runtime
runtime/storage-layout
runtime/operator-dispatch
runtime/jit-operator-source
backends/backend-build
backends/acl-backend-contracts
runtime/static-graph-replay
```
-->

## 修改 Torch 兼容或下游集成

- [前端对象归属与 dtype](compatibility/frontend-boundaries.md)
- [安装事务与回滚](compatibility/install-transactions.md)
- [下游支持范围与验收证据](testing/downstream-scorecards.md)

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

compatibility/frontend-boundaries
compatibility/install-transactions
testing/downstream-scorecards
```
-->

## 验证并排查问题

- [数值、梯度与覆盖](testing/test-evidence.md)
- [进程隔离、skip 与完成判据](testing/test-isolation.md)
- [JIT 缓存与耗时](testing/jit-cache.md)
- [用户错误与内部错误](runtime/error-categories.md)
- [异步错误定位](runtime/async-error-diagnostics.md)
- [记录 NPU 复现和性能证据](testing/npu-validation-templates.md)
- [并行编译故障调查](known-issues/parallel-compiler-segfault.md)

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

testing/test-evidence
testing/test-isolation
testing/jit-cache
runtime/error-categories
runtime/async-error-diagnostics
testing/npu-validation-templates
known-issues/parallel-compiler-segfault
```
-->

## 记录结果与缺陷

任务、缺陷、待硬件验收与实际执行结果在 GitHub issue/PR 维护。
协作规则见 [AGENTS.md](https://github.com/Jittor/jittor/blob/2.0-refactor/AGENTS.md)，
按任务使用[技能目录](https://github.com/JittorEcosystem/jittor-dev-skills/blob/main/README.md)。
旧问题和硬件验收记录通过该规则页的固定提交链接查阅，未复验不等于已解决。

## 兼容架构与贡献流程

修改兼容前端或文档时阅读这些专题；运行模型请先看用户兼容指南。

<!-- sphinx-only
```{toctree}
:hidden:
:maxdepth: 1

contributing
compatibility/torch-architecture
compatibility/torch-lowering
compatibility/principles
```
-->
- [贡献流程](contributing.md)
- [共享计算图与前端分层](compatibility/torch-architecture.md)
- [调用下沉到核心](compatibility/torch-lowering.md)
- [兼容验收原则](compatibility/principles.md)

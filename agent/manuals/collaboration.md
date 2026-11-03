# Jittor 主仓协作手册

本页补充 [`AGENTS.md`](../../AGENTS.md) 的协作方式；同步、工作区边界、提交与测试要求以该文件为准。
文档入口是 [agent-index.md](agent-index.md)，当前源码与工具入口见 [project-context.md](project-context.md)。

## 先确定改动归属

| 内容 | 维护位置 |
| --- | --- |
| 原生张量、自动微分、编译器、运行时与后端 | 本仓源码、`tests/` 与公开开发文档 |
| 通用 Torch API、shim、序列化与分布式兼容语义 | 本仓 `compat/`、对应测试与开发文档 |
| 某个下游库的版本适配、教程、官方测试、测速与功能/芯片支持范围 | [JittorEcosystem](https://github.com/JittorEcosystem) 中对应仓库 |
| 跨仓发现的核心或 compat 缺陷 | 在主仓修复并加入最小回归；下游仓库保留集成验收 |

下游适配按 [downstream-library-adaptation](../skills/downstream-library-adaptation/SKILL.md) 分流。
主仓不维护第二份下游安装矩阵、支持表或项目运行手册。迁移材料前核对承接仓库；没有承接的未解决问题不能因清理而消失。

## 记录只保留一个归属

- [project-context.md](project-context.md)：当前结构与入口，不复制问题列表或执行历史。
- [environment.md](environment.md)：主仓环境、缓存隔离与后端前置条件。
- [known-issues.md](known-issues.md)：主仓尚未解决的缺陷与限制，保留最小复现、证据和关闭条件。
- [deferred-hardware.md](deferred-hardware.md)：主仓待硬件验收项；缺设备不等于通过。
- GitHub issue/PR：任务状态、交接、执行命令与结果。旧材料由 Git 历史保留，不在 `agent/` 新建报告树。
- `docs/`：用户指南、长期机制与实现契约。不要在 skill 中复制整篇公开文档。

维护条目时区分“旧证据”“当前实现”和“本次实际执行”。文档整理不刷新硬件验收基线，也不关闭未复验缺陷。

## Skill 与工具

主仓 skill 只描述核心、后端、通用 compat 或仓库维护中可复用的方法；项目专属流程由下游仓库维护。
新增前先查 [索引](agent-index.md)，能合并到现有方法就不另建入口。命令引用当前树中的真实路径；
机器地址、凭据、设备分配和个人目录只放在本地运行配置。

通用仓库命令放在 `tools/`；skill 配套的小型验证脚本可以跟随 skill。
独立实验、下游 checkout 与原始产物按 [环境规则](environment.md) 放在仓库外。

## 验证与并行协作

先最小复现，再修改；验证结果须包含实际执行数量、失败和 skip 原因。
多个 agent 可按互不重叠的文件或验证任务分工；同步远端期间停止同一工作树的写入与测试。

首次 JIT 或扩展编译串行完成。并行验证必须使用独立的 `JITTOR_HOME` 或 cache name，
benchmark 与 unittest 不共用编译缓存。长时间无输出时先检查编译进程与 `jittor.lock`，不要把等待锁当成算子挂死。

## 工作区边界

源码、测试、公开文档与可复用维护方法留在主仓。实验、worktree、缓存、日志和构建产物的位置由
[`AGENTS.md`](../../AGENTS.md#repository-boundary) 与 [环境模板](environment.md#workspace-roots) 统一规定；本页不另设规则。

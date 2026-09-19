# Project Rules

每次开始新任务或恢复中断任务时，先获取并整合目标远端分支的最新代码，记录同步后的
提交 SHA，再开始修改和验证。先检查未提交改动及正在运行的测试；保留本地工作，
以远端为准完成同步后，再重新应用和调整本地修改；重叠处优先保留远端实现，
不得用旧本地代码覆盖远端更新。禁止通过丢弃改动或自动 stash 清场来同步。整合期间冻结同一工作树的
测试与协作者写入；同步前的结果保留原基线，按实际变更复验受影响范围。

## Start Here

开始任务前依次阅读：

1. [`agent/manuals/collaboration.md`](agent/manuals/collaboration.md)：协作、验证和 JIT 并发规则。
2. [`agent/skills/jittor-dev-context/SKILL.md`](agent/skills/jittor-dev-context/SKILL.md)：上下文入口。
3. [`agent/manuals/project-context.md`](agent/manuals/project-context.md)：当前状态和主题索引。
4. 与任务相关的架构文档、已知问题和既有结果报告。

整改任务的唯一状态源是 [`refactor-wip/architecture/refactor-board.md`](refactor-wip/architecture/refactor-board.md)。
按同目录的 [交接](refactor-wip/architecture/refactor-handoff.md)与
[分工](refactor-wip/architecture/refactor-dispatch.md)协作，不在 agent/ 下另建看板或结果树。

## Working Method

- verify-then-fix：先用最小复现确认问题，再修改。
- 新增或修复的计算必须在所声明的真实 device 上执行；导入成功和 CPU fallback
  不能证明 CUDA、ROCm 或 NPU 支持。
- 首次 JIT 或扩展编译串行执行。并行任务使用不同的 `JITTOR_HOME` 或 `cache_name`，
  unittest 与 benchmark 不得共享同一编译缓存。
- 公共行为变化需要定向回归测试；涉及共享语义时扩大到 CPU 与相关加速后端。
- 不使用宽泛异常捕获隐藏安装、注册、编译或后端错误。
- 接入或验证下游 Torch 生态库（新库对拍、生态门禁 case、`adapters/` 新增或修改）
  按 [`agent/skills/downstream-library-adaptation/SKILL.md`](agent/skills/downstream-library-adaptation/SKILL.md) 执行：先按其三行判据分流到
  jittor 核心、`jittor.compat.torch` 或 adapter，再按其 device 阶梯和准入清单验收。
  不在该流程外新增 adapter。

## Repository Boundary

- 主仓库只保存源码、测试、公开文档、`agent/` 协作文档和可复用工具。
- 独立实验、下游 checkout 和大体积产物放在 `$JITTOR_LAB_ROOT/<topic>/`。
- `HOME`、`JITTOR_HOME`、`TMPDIR`、模型缓存、编译缓存和原始日志放在
  `$JITTOR_LAB_ROOT/_state/<topic>/<run>/`。
- Git worktree 放在 `$JITTOR_LAB_ROOT/worktrees/`。
- 未设置 `JITTOR_LAB_ROOT` 时，使用仓库同级的 `jittor-lab/`；不要在规则或文档中
  写入个人主目录、机器地址或设备编号。
- 不在主仓库顶层创建 `jittor_fsdp2`、`*_work`、`*_probe` 等实验目录，也不把
  下游兼容文件放进 `python/jittor/` 根目录。

## 单测分层

一次全量跑完十几分钟起，拿它当「刚改的东西坏没坏」的答案太贵。三层回答三个问题：

| 层 | 问题 | 选择方式 | 实测 |
| --- | --- | --- | --- |
| `core` | 改完一处，有没有立刻坏 | `tests/_helpers/tiers.CORE_FILES` 白名单 | 44 s，双进程模式 |
| `smoke` | 一个 PR 该等多久 | 整棵树减去 `tiers.SLOW_FILES` | 约 390 s |
| `full` | 这棵树到底什么状态 | 整棵树 | 十几分钟起 |

```bash
python tools/run_test_suite.py --tier core    # 改完就跑，默认串行
python tools/run_test_suite.py --tier smoke   # 提 PR 前
python tools/run_test_suite.py                # 全量，按需，不要每次都跑
```

- **不要动不动跑全量。** 改完先 `--tier core`；只有在定位面广、或要给出「这棵树现在
  什么状态」这种结论时才跑 `full`。
- `core` 是白名单，`smoke` 是黑名单，因为漏掉一个文件的后果不同：`smoke` 漏了是
  覆盖的洞，`core` 漏了只是晚几分钟发现。往 `core` 加文件是在花所有人每一次编辑的
  时间，条目必须写清实测秒数和它守的基本面，由 `tests/structure/test_gate_tiers.py`
  检查。
- compat 有两种测试，`core` 两种各有一条：行为（`compat/tests/torch`，Torch 模式下
  真的跑张量）和结构契约（`compat/tests/structure`，断言名字归属与命名空间发布）。
- 预算按**算术**校验（各文件实测秒数求和 vs 预算），不断言墙上时钟——后者在忙机器上
  失败的样子和真回归一模一样。

提交前运行：

```bash
bash tools/check_repo_layout.sh
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

`tests/structure` 属于 Torch 模式路径（见 `tests/_helpers/process_modes.py`），
不带 `JITTOR_TORCH_SHIM=1` 会直接被 pytest 策略拒绝收集。

顶层或文档结构有意调整时，同步更新结构门禁。只暂存当前任务涉及的文件，不使用
`git add -A`；完成后提交，提交信息用简明中文。

## Documentation Ownership

- 根目录只保留一份双语 [`README.md`](README.md)。不要新增生成版或按语言复制的 README。
- 长期机制说明放在 `docs/notes/`，仓库布局、源码架构、测试体系与已知问题放在
  `docs/development/`，研究提案放在 `docs/research/`。整改期的过程文档在
  `refactor-wip/`，收口后整个目录删除。
- [`agent/manuals/project-context.md`](agent/manuals/project-context.md) 只做当前状态索引；
  环境规则与问题总账分别维护在
  [`agent/manuals/environment.md`](agent/manuals/environment.md) 和
  [`agent/manuals/known-issues.md`](agent/manuals/known-issues.md)。
- `docs/results/` 保存可复现的维护者验证结论，不复制成长篇项目历史。原始日志、缓存、
  二进制和 benchmark 数据不进入文档树。
- 稳定结论注明状态、对应提交、验证范围、维护者和复查条件；过期信息应删除或归档，
  不在多个文件中重复维护。

# Project Rules

每次开始新任务或恢复中断任务时，先获取并整合目标远端分支的最新代码，记录同步后的
提交 SHA，再开始修改和验证。先检查未提交改动及正在运行的测试；保留本地工作，
以远端为准完成同步后，再重新应用和调整本地修改；重叠处优先保留远端实现，
不得用旧本地代码覆盖远端更新。禁止通过丢弃改动或自动 stash 清场来同步。整合期间冻结同一工作树的
测试与协作者写入；同步前的结果保留原基线，按实际变更复验受影响范围。

## 按任务查阅

直接按当前任务查阅[整体架构与原理图](docs/development/architecture-overview.md)、
[开发文档](docs/development/index.md)或[技能目录](https://github.com/JittorEcosystem/jittor-dev-skills/blob/main/README.md)，
不再要求依次阅读协作手册和项目上下文。同步、缓存、验证、多仓协作与提交规则只在本文件维护。

任务、缺陷、待硬件验收事项与验证结果记在 GitHub issue/PR。
既有的[问题记录](https://github.com/Jittor/jittor/blob/3679de22c02c83b982432a0e1bdb3da148411c50/agent/manuals/known-issues.md)和
[硬件验收记录](https://github.com/Jittor/jittor/blob/3679de22c02c83b982432a0e1bdb3da148411c50/agent/manuals/deferred-hardware.md)保留在固定提交中，
未因文档清理而关闭或重新验收。涉及其中条目时，按最新源码复现并在 issue/PR 记录后续处置。

## 多仓协同与 PR 流程（`Jittor/jittor` × `JittorEcosystem`）

| 仓库层 | 归属内容 | 分支与合入规则 | 冲突裁决与同步 |
| --- | --- | --- | --- |
| 主仓 `Jittor/jittor` | 核心计算、延迟图、JIT、多后端及通用 `compat/` 语义；主仓不保留 `adapters/` | 在个人 fork 基于最新 `2.0-refactor` 建分支，向 `Jittor/jittor:2.0-refactor` 提 PR；**禁止直接 push 或自行合并上游 `2.0-refactor`** | 维护者合并后产生新的框架 SHA |
| 下游仓 `JittorEcosystem/jittor-*` | 库私有 adapter、`upstream/` 版本锁、官方测试/示例与双语支持/验证页 | 在目标下游仓建 `task/<slug>` 分支，向 `main` 提 PR；必需检查与证据齐全后合并 | 通过 `compatibility.json` 固定上游与框架 SHA；换 pin 必须重跑受影响证据并运行 `support_matrix.py --update` |
| 技能仓 `JittorEcosystem/jittor-dev-skills` | 跨仓可复用的工程技能与检查工具；本地克隆至 `$JITTOR_LAB_ROOT/jittor-dev-skills` | 在技能仓建 `task/<slug>` 分支向 `main` 提 PR；修改后运行 `python3 tools/check_skills.py` | 目标项目 `AGENTS.md` 优先于共享技能；工具与参数以各自源码仓为准 |

跨仓问题分流与闭环按
[downstream-library-adaptation](https://github.com/JittorEcosystem/jittor-dev-skills/blob/main/skills/downstream-library-adaptation/SKILL.md)
与 [ecosystem-repo-change-workflow](https://github.com/JittorEcosystem/jittor-dev-skills/blob/main/skills/ecosystem-repo-change-workflow/SKILL.md) 执行：

1. **核心能力缺失或计算错误** → 在主仓核心与后端修复，补不依赖下游大模型的最小回归测试；
2. **公共 Torch 签名、对象身份或共享语义不兼容** → 在主仓 `compat/` 修复，补公共 API 行为/结构测试；
3. **下游库私有 API、插件注册、入口或版本差异** → 在对应 `JittorEcosystem/jittor-*` 仓库修复并取证。

涉及主仓修复的下游任务按闭环推进：**主仓 fork PR 过检并合入 → 下游更新 `compatibility.json` pin → 重跑受影响证据并重新生成双语支持页**；主仓 PR 尚未合入而需临时在下游复验时，须在 PR 描述与支持页细节中显式注明所依赖的 PR 与未合入状态，不得把未绑定当前 pin 的旧证据当作已验收。

## Working Method

- verify-then-fix：先用最小复现确认问题，再修改。
- 新增或修复的计算必须在所声明的真实 device 上执行；导入成功和 CPU fallback
  不能证明 CUDA、ROCm 或 NPU 支持。
- 首次 JIT 或扩展编译串行执行。并行任务使用不同的 `JITTOR_HOME` 或 `cache_name`，
  unittest 与 benchmark 不得共享同一编译缓存。
- 公共行为变化需要定向回归测试；涉及共享语义时扩大到 CPU 与相关加速后端。
- 性能结论必须先有同规格正确性证据，再在同一硬件与独立缓存上 1:1 复测（区分冷编译与稳态），不引入通用路径负优化。
- 不使用宽泛异常捕获隐藏安装、注册、编译或后端错误。
- 不在 `downstream-library-adaptation` 流程外新增 adapter；下游实现与专属测试在 `JittorEcosystem` 独立仓库维护，入口见根 README。

## Repository Boundary & Safety

- 主仓库只保存源码、测试、公开文档、开发技能入口和可复用工具。
- 独立实验、下游 checkout 和大体积产物放在 `$JITTOR_LAB_ROOT/<topic>/`。
- `HOME`、`JITTOR_HOME`、`TMPDIR`、模型缓存、编译缓存和原始日志放在
  `$JITTOR_LAB_ROOT/_state/<topic>/<run>/`；普通运行默认保留用户 `HOME`，仅在显式测试全新主目录时为子进程覆写。
- Git worktree 放在 `$JITTOR_LAB_ROOT/worktrees/`；一任务一 worktree，不与并发任务共享工作树或缓存。
- 未设置 `JITTOR_LAB_ROOT` 时，使用仓库同级的 `jittor-lab/`；不要在规则、脚本或文档中
  写入个人主目录、机器地址、凭据或设备编号。
- 不在主仓库顶层创建 `jittor_fsdp2`、`*_work`、`*_probe` 等实验目录，也不把
  下游兼容文件放进 `python/jittor/` 根目录。
- 安全清理原则：严禁对未做非空与路径前缀断言的动态变量执行 `rm -rf`；删除任何缓存前先确认本任务相关 worker 进程已退出，且不得清理他人正在使用的缓存目录。
- 安全与凭据卫生：GitHub 优先使用 SSH remote；token 不得写入代码、提交、remote URL、日志或文档。

## Agent 文档边界

- 主仓只保留本 `AGENTS.md` 作为开发代理规则与导航入口，不保留 `agent/`。
- 可复用技能在 [JittorEcosystem/jittor-dev-skills](https://github.com/JittorEcosystem/jittor-dev-skills) 维护；
  需要本地读取时克隆到 `$JITTOR_LAB_ROOT/jittor-dev-skills`，不要复制回主仓。
- 技能中的命令从目标 Jittor checkout 根目录执行；主仓测试所需工具继续在 `tools/`、`benchmarks/` 随源码维护。
- 下游专属流程在 `JittorEcosystem` 对应仓库维护；主仓只保留统一的问题分流规则。
- 仍被主仓门禁调用的生态验证脚本位于 `tools/ecosystem/`，性能探针位于 `benchmarks/`；不要将它们藏在逐库技能目录中，也不要将工具存在视为支持验收。
- 协作规则在根 `AGENTS.md` 维护；问题状态与执行记录使用 GitHub issue/PR。

## 单测分层

一次全量跑完十几分钟起，拿它当「刚改的东西坏没坏」的答案太贵。三层回答三个问题：

| 层 | 问题 | 选择方式 | 实测 |
| --- | --- | --- | --- |
| `core` | 改完一处，有没有立刻坏 | `tests/_helpers/tiers.CORE_FILES` 白名单 | 44 s，双进程模式 |
| `smoke` | 一个 PR 该等多久 | 整棵树减去 `tiers.SLOW_FILES` | 约 390 s |
| `full` | 这棵树到底什么状态 | 整棵树 | 十几分钟起 |

```bash
python tools/run_test_suite.py --tier core # 改完就跑，默认串行
python tools/run_test_suite.py --tier smoke # 提 PR 前
python tools/run_test_suite.py # 全量，按需，不要每次都跑
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

## 提交与提 PR 前检查

提交前运行：

```bash
bash tools/check_repo_layout.sh
python tools/build/generate_manifest.py --check
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

按改动范围补充：

- **修改文档或链接**：额外运行 `python tools/docs/check_links.py`，并确认中英文页面成对且均纳入 `toctree`。
- **修改 `src/` 或 `backends/` C++ 源码**：在 CUDA 与 CPU-only 两种宏配置下确认新增/改动编译单元无新增告警；新增仅源码文件时先运行 `python tools/build/generate_manifest.py` 更新 `MANIFEST.in`。
- **修改 `compat/`**：补充运行 `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q python/jittor/compat/tests/structure`。

`tests/structure` 属于 Torch 模式路径（见 `tests/_helpers/process_modes.py`），
不带 `JITTOR_TORCH_SHIM=1` 会直接被 pytest 策略拒绝收集。

顶层或文档结构有意调整时，同步更新结构门禁。只暂存当前任务涉及的文件，不使用
`git add -A`；完成后提交（提交信息用简明英文），推送至个人 fork 并向 `Jittor/jittor` 的 `2.0-refactor` 创建 PR。

## Documentation Ownership

- 根目录保留中文默认 [`README.md`](README.md) 与英文
  [`README.en.md`](README.en.md)；两份内容保持同步，不新增其它语言或生成版本。
- `docs/` 下每篇文档成对维护：`X.md` 是中文默认页，`X.en.md` 是同目录英文页；
  索引页同样成对（`index.md` / `index.en.md`）。英文页的相对链接指向 `X.en.md`，
  中文页指向 `X.md`。修改一侧时必须同步另一侧。
- 长期机制说明放在 `docs/notes/`，仓库布局、源码架构、整体架构图、测试体系与已知问题放在
  `docs/development/`，研究提案放在 `docs/project/research/`。仓库不保留任务看板、计划或交接
  这类过程文档。
- 执行记录、阶段性验证与测速结论写入 GitHub issue/PR，不在 `docs/` 新建日期命名文档、
  结果目录或演示产物。历史报告通过固定提交引用；原始日志、缓存、二进制和 benchmark
  数据留在仓库外。可复用机制与操作说明按主题维护。
- 稳定结论注明状态、对应提交、验证范围、维护者和复查条件；过期信息应删除或归档，
  不在多个文件中重复维护。

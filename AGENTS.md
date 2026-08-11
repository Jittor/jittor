# Project Rules

## Start Here

开始任务前依次阅读：

1. [`agent/manuals/collaboration.md`](agent/manuals/collaboration.md)：协作、验证和 JIT 并发规则。
2. [`agent/skills/jittor-dev-context/SKILL.md`](agent/skills/jittor-dev-context/SKILL.md)：上下文入口。
3. [`agent/manuals/project-context.md`](agent/manuals/project-context.md)：当前状态和主题索引。
4. 与任务相关的架构文档、已知问题和既有结果报告。

## Working Method

- verify-then-fix：先用最小复现确认问题，再修改。
- 新增或修复的计算必须在所声明的真实 device 上执行；导入成功和 CPU fallback
  不能证明 CUDA、ROCm 或 NPU 支持。
- 首次 JIT 或扩展编译串行执行。并行任务使用不同的 `JITTOR_HOME` 或 `cache_name`，
  unittest 与 benchmark 不得共享同一编译缓存。
- 公共行为变化需要定向回归测试；涉及共享语义时扩大到 CPU 与相关加速后端。
- 不使用宽泛异常捕获隐藏安装、注册、编译或后端错误。

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

提交前运行：

```bash
bash agent/scripts/check_repo_layout.sh
python -m pytest -q tests/structure
```

顶层或文档结构有意调整时，同步更新结构门禁。只暂存当前任务涉及的文件，不使用
`git add -A`；完成后提交，提交信息用简明中文。

## Documentation Ownership

- 根目录只保留一份双语 [`README.md`](README.md)。不要新增生成版或按语言复制的 README。
- 长期架构决策放在 `docs/architecture/`，测试契约放在 `docs/testing/`，开发指南和
  已知问题放在 `docs/development/`，研究提案放在 `docs/research/`。
- [`agent/manuals/project-context.md`](agent/manuals/project-context.md) 只做当前状态索引；
  环境规则与问题总账分别维护在
  [`agent/manuals/environment.md`](agent/manuals/environment.md) 和
  [`agent/manuals/known-issues.md`](agent/manuals/known-issues.md)。
- `agent/results/` 保存可复现的维护者验证结论，不复制成长篇项目历史。原始日志、缓存、
  二进制和 benchmark 数据不进入文档树。
- 稳定结论注明状态、对应提交、验证范围、维护者和复查条件；过期信息应删除或归档，
  不在多个文件中重复维护。

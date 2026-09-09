# 2026-08-10 仓库工作区整理

## 结论

主仓库中的独立实验、运行缓存和 Git worktree 已迁至
`/home/zy/projects/jittor-lab`。整理前工作区约 29G；整理后主仓库源码、测试和
文档约 9.7M，连同 `.git` 约 69M。迁移过程没有删除实验数据，也没有丢失三个
worktree 的未提交修改。

## 新目录边界

| 内容 | 位置 |
| --- | --- |
| Jittor 源码、测试、公开文档 | `/home/zy/projects/jittor` |
| 独立实验和下游 checkout | `${JITTOR_LAB_ROOT}/<topic>` |
| 缓存、临时文件、原始日志 | `${JITTOR_LAB_ROOT}/_state/<topic>/<run>` |
| Git worktree | `${JITTOR_LAB_ROOT}/worktrees` |
| 本次迁出的旧运行状态 | `${JITTOR_LAB_ROOT}/_state/legacy` |

`JITTOR_LAB_ROOT` 默认取 `/home/zy/projects/jittor-lab`。完整路径与大小映射见
lab 根目录的 `MOVED_FROM_JITTOR_2026-08-10.tsv`。

## 迁移内容

- 九个实验目录已移出仓库顶层：`diffusers_video_jittor`、`jittor_fsdp2`、
  `jittor_lightning_backend`、`jittor_torchmetrics_work`、
  `jittor_transformers_perf`、`ms_swift_jittor_work`、
  `transformers_jittor_cuda`、`transformers_jittor_probe`、`verl_jittor`。
- `transformers_jittor_probe` 只有运行产物，保存在 `jittor-lab/archive`。
- 仓库根缓存、嵌套 `__pycache__`、egg-info 和原始日志保存在
  `jittor-lab/_state/legacy`。
- 三个 `.claude/worktrees` 通过 `git worktree move` 迁至 lab；分支、HEAD 和
  dirty 状态保持不变。
- Transformers 性能脚本的规范副本放在
  `agent/skills/jittor-transformers-perf/scripts`，旧 lab 入口使用符号链接兼容。

## 防复发措施

- `agent/scripts/check_repo_layout.sh` 对仓库顶层做允许名单检查，并接入 CI。
- `.gitignore` 不再隐藏实验目录专名；误放 `jittor_fsdp2` 等目录时，
  `git status` 会直接暴露。
- `.dockerignore` 改为默认拒绝，仅允许 Python 包构建所需文件进入 context；
  Dockerfile 也改为明确复制包文件。
- torch shim 的默认运行目录改为
  `${XDG_CACHE_HOME:-~/.cache}/jittor/torch-shim/<project>-<hash>`，显式
  `JITTOR_TORCH_RUNTIME_ROOT` 仍有最高优先级。
- 两个会在源码树创建临时目录的测试，以及 TorchQuantum、torch-diff、
  Transformers perf 的运行说明，均改用 lab 或用户缓存目录。

## 验证

- `agent/scripts/check_repo_layout.sh`：通过。
- `git diff --check`：通过。
- 相关 Python 文件与全部 Transformers perf 脚本 `py_compile`：通过。
- 相关 Shell 脚本 `bash -n`：通过。
- `jittor.test.test_cuda_wheel` 与 `jittor.test.test_torch_bootstrap`：13 项通过。
- 迁移映射逐项核对：所有来源均已从主仓库消失，所有目标均存在。
- `git worktree list` 与三个 worktree 的 dirty 状态核对：保持完整。

测试缓存统一写入
`/home/zy/projects/jittor-lab/_state/verify/repository-cleanup`，没有重新污染主仓库。

# 仓库结构现代化阶段 0：目标架构 RFC

## 状态

✅ 已完成目标架构决策落盘；阶段 1 打包现代化待开始。

## 背景证据

- 第十、十一批已由提交 `47e7ec75` 提交，不存在待提交的 staged 文件。
- 当前工作树中 `python/jittor/src/var_holder.cc` 与
  `python/jittor/test/test_setitem.py` 是用户未提交修改，本阶段未触碰或暂存。
- 旧架构规范把 `公开文件 + _xxx/` 和 `jittor.test` 公开路径定义为长期目标，和新的
  仓库现代化计划冲突。

## 决策

新增 `docs/architecture/repository-layout.md`，锁定两项核心决定：

1. `nn.py + _nn/` 等配对最终合并成常规领域包，`__init__.py` 作为组合入口，真实
   实现路径成为反射来源；重复 runtime proxy 和 `preserve_facade_origins` 被列为迁移
   脚手架。
2. Torch 兼容按框架能力、通用机制、Jittor 根因、下游项目粘合四层分离；TRELLIS、
   Gaussian Splatting 和 Transformers 版本粘合最终进入可选发行包。

RFC 同时记录不能机械移动的 JIT 资源路径、目标仓库树、打包和测试边界、工具链、
阶段顺序及逐阶段验收门槛。独立审计补充锁定了三项迁移前提：五个领域配对包都要
消除；`jittor.nn.functional` 的物理包与 Torch shim 必须保持单一模块对象；当前测试
依赖图是 76 个文件、123 条跨测试 import，迁移不能继续沿用旧的 69 文件估计。

## 被取代的旧决策

`agent/manuals/design/source-architecture.md` 已明确更新：

- 前导下划线实现包不再是长期模式；已有 `_nn/_misc/_pool/_torch_compat/
  _torch_fsdp2` 仅作为前十一批历史脚手架。
- `python/jittor/test` 不再长期随 wheel 发布；先由 `jittor.selftest` 承接安装后自检，
  再迁移到仓库根 `tests/`。
- 不再计划创建 `_third_party_compat/`；下游粘合应移出主仓库。
- `extern/llvm` 仍被 Clang/Linux 编译路径消费，在替代消费者和回归证据就绪前不得按
  “死目录”直接删除。

## 本阶段验证

- RFC 与旧规范交叉引用使用仓库内相对路径。
- `git diff --check` 与 `agent/scripts/check_repo_layout.sh` 作为提交前门禁执行。
- 本阶段只修改文档，不改变 Python/C++、打包内容或运行时行为；设备数值回归沿用
  `47e7ec75` 已记录的 CPU、CUDA 和真实双卡 NCCL 结果，不把它扩写为 NPU 证据。

## 下一阶段

阶段 1 将建立 `pyproject.toml`、完整包发现、显式资源清单和 wheel 内容基线。最新
1,053 条目基线 wheel 已含 `UnpackRaw.cuh`，阶段 1 要把它从偶然的深层通配结果变成
显式受测资源；部署版 `flash_attn_interface.py` 的复制缺口则需要实际修复。

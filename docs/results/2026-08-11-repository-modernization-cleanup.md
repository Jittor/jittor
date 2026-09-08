# 仓库结构现代化阶段 6：杂物清理与发行边界

日期：2026-08-11

## 状态

Stage 6 已完成运行时发行边界清理。安装包不再携带开发脚本、示例、教程、离线
打包工具、`version` 文件或未使用的 `vcompiler`；仓库根的 `tools/` 与 `examples/`
成为对应资产的唯一位置。用户已有的三处未暂存修改没有进入验收索引或制品。

`extern/llvm/` 没有按原计划删除。Clang 14 冷启动验证依次暴露 libomp 缺失、
`data.gz` 路径的成员函数指针转换失败，以及禁用 `data.gz` 后
`jittor::tflag_count` 未定义。删除 `compiler.compile_extern()` 与
`extern/llvm/jt_alignment_from_assumptions.cc` 会破坏现有编译链，因此本阶段明确
延期，而不是把未通过的删除伪装成完成。

## 仓库边界

- `python/jittor/script/` 的仍有用途入口迁到 `tools/`，旧安装、文档、发布入口标成
  `legacy`，均支持无副作用导入或 dry-run；6 个 shell 入口通过 `bash -n` 且保持
  可执行位。
- `python/jittor/demo/` 迁到 `examples/gan/`；`python/jittor/notebook/` 改为
  `examples/notebooks/` 下 11 组 Jupytext MyST Markdown 与 notebook 配对。所有
  notebook 已清除输出、执行计数、附件和本机路径，并修正旧 API。
- 删除 `python/jittor/vcompiler/`、`python/jittor/version`、旧 notebook 转换器、
  `python/jittor_utils/pack_offline.py` 及运行时中的 polish/release 脚本。
  `jittor.vcompiler` 的 breaking removal 和替代入口已记录在 `docs/releases/2.0.md`。
- `.dockerignore`、Dockerfile、MANIFEST、README、贡献指南、release/container workflow
  已同步新路径。sdist 对完整 `examples/`、`tools/` 和 `requirements/examples.txt`
  做双向成员检查，缓存、字节码和额外文件均会使门禁失败。
- `check_repo_layout.sh` 对运行时、工具、示例、workflow、根文档和现行 docs 做集中
  旧路径扫描。仓库根 `jittor_fsdp2`、根单文件
  `python/jittor/torch_fsdp2_compat.py` 以及本阶段退休路径都会 fail closed。

## 制品

最终制品从独立 Git 索引导出；索引以 HEAD 为基线加入 Stage 6 全部修改，再把
`agent/manuals/README.md`、`python/jittor/src/var_holder.cc` 和
`tests/core/test_setitem.py` 精确还原为 HEAD blob。

| 制品 | SHA-256 |
| --- | --- |
| direct wheel | `d9f7fe785ebad7d7562e3b17f968f746a46299bfe2081f6f944a54ccdf1c76ad` |
| sdist | `e0cc2c667a63021849fe4bb2ceacebb50396f3acd4a975cbfc6bf8fb44754ff1` |
| sdist-derived wheel | `d9f7fe785ebad7d7562e3b17f968f746a46299bfe2081f6f944a54ccdf1c76ad` |

两个 wheel 均有 728 个成员，路径集合及逐成员 SHA-256 完全一致。相对 Stage 5
的 762 成员 accepted baseline，精确差异为 0 additions、3 项批准内容变化
（selftest 后端识别、METADATA、RECORD）、34 项批准删除；unexpected 和未消费
allowance 均为 0。
sdist 有 918 个成员，完整包含受治理的工具、示例和示例依赖文件，且不包含缓存。

## 验证

| 验证 | 结果 |
| --- | --- |
| 全树 pytest collect | 3104 项，0 collection error |
| 结构与制品 checker | 132 passed，2 个环境 skip；checker 单测 15/15 |
| notebook | 5/5；Jupytext 精确同步、可移植性和离线 CPU 执行通过 |
| CPU 代表组 | custom op 2、compiler utils 3、autograd engine 9、silent regression 11，合计 25/25 |
| ASV | `asv check` 通过；Jittor CPU GELU setup/time/RSS/teardown 通过，RSS 266842112 bytes |
| 静态门禁 | Ruff lint/format、Mypy、YAML、`git diff --check` 均通过 |
| Python 3.7 语法 | Stage 6 索引 546 个 Python 文件经 3.7 grammar AST 检查通过；本机无真实 3.7 解释器 |
| 最终 wheel CPU | 隔离安装来源确认；forward `(1,4,9)`、gradient `(2,4,6)` |
| wheel CUDA | RTX 4090、JTCUDA CUDA 12.2/cuDNN 8；device count 1，loss 14，gradient `[2,4,6]` |

旧 selftest 曾把 CUDA 误报为 `npu`，根因是 `flags.use_acl/use_rocm` 只是
`use_cuda` 的绑定别名，并非独立后端状态。最终实现改用
`compiler.has_acl/has_rocm` 判别真实能力；最终 wheel 在
`has_cuda=1`、`has_acl=0`、`use_cuda=1` 下明确报告 `cuda`。

## 已知边界

- `importlib.metadata.version("jittor")` 与 `jittor.__jittor_version__` 均为
  `1.3.11.0`；现有 Torch 兼容 installer 会把 `jittor.__version__` 改为 Torch API
  版本 `2.11.0`。这是 Stage 7 统一 installer/composition root 必须解决的版本所有权
  冲突，本阶段没有静默改写既有 Torch 兼容语义。
- 本机 Docker daemon 无访问权限，未执行真实 image build；container workflow 已把
  Dockerfile、README 和 license 变更纳入触发路径。机器没有 NPU/CANN，因此没有声称
  NPU 回归通过。

最终索引、构建日志、成员对比、隔离安装和 CPU selftest 位于
`${JITTOR_LAB_ROOT}/_state/verify/repository-modernization/stage6-index-final/`；
明确标识 CUDA 的最终日志为同级 `stage6-index-reviewed/cuda-final-fixed-pass.log`。
LLVM 诊断位于同级 `stage6-clang-before/` 与 `stage6-clang-runtime/`，ASV 日志位于
`stage6-benchmark-final/`。

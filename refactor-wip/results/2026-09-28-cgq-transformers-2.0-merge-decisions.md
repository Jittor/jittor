# `cgq_transformers` 合并 `2.0-refactor` 决策记录

状态：合并冲突已解决，定向验证已通过，已创建本地合并提交。此文件记录在独立 worktree
`/home/cgq/jittor-lab/worktrees/cgq-transformers-full-2.0` 中进行的合并，不代表已经
提交或推送到远端。

## 基线

- 目标分支：`origin/cgq_transformers`，当前 SHA `a2850846a`。
- 来源分支：`origin/2.0-refactor`，当前 SHA `cc1b7bda3`。
- 合并基点：`0ce0e43736`。
- 分支关系：相对合并基点，目标分支有 78 个独有提交，来源分支有 608 个独有提交。
- 执行方式：在 `integrate/cgq-transformers-full-2.0` 上执行
  `git merge --no-commit --no-ff origin/2.0-refactor`。

原生单机多卡的 `python/jittor/distributed/`、`python/jittor/nn/parallel/`、启动器、
进程组、DDP 和对应测试没有产生冲突，当前合并树保留了目标分支实现。运行条件仍按
项目约定冻结为 `NCCL_P2P_DISABLE=1`；本轮只处理代码基线兼容，不在未确认前推送。

## 可以直接合并的部分

1. 根目录 `AGENTS.md` 合并两边的同步、测试分层和文档归属规则，保留仓库当前对
   `refactor-wip/`、布局检查和未提交文件的约束。
2. `python/jittor/__init__.py` 合并两边的公共导出：保留原生多卡相关导出，同时加入
   2.0 基线的图重放和性能接口。
3. `src/type/common_op_type.cc`、`src/type/fp16_op_type.cc`、`src/utils/log.h` 采用
   2.0 的最新实现，并检查原生多卡所需的编译接口没有被删掉。它们的冲突是核心算子
   数值/编译更新与并发编译安全修复，不涉及 DDP API。
4. 结构和回归测试文件优先采用 2.0 较新的断言；若目标分支只增加了独立的原生多卡
   测试，则两个集合并保留。
5. `MANIFEST.in` 在冲突消解后重新运行仓库的 manifest 生成器，不手工维护生成结果。

## 冲突取舍

### 1. cuRAND 状态模型（合并后复核已调整）

冲突文件：

- `backends/cuda/kernels/curand/curand_random_op.cc`
- `backends/cuda/libraries/curand/include/curand_wrapper.h`
- `backends/cuda/libraries/curand/src/curand_wrapper.cc`

初始合并曾整体采用 2.0 的 `curand_generator_seed/offset/restore_state` 实现，目标分支的
`JITTOR_CURAND_XORWOW_U32_V1` 格式因此没有进入合并树。合并后对 Accelerate 的随机状态
和显式生成器路径复核发现，这个取舍只有 compat 层的 seed+offset 计数，没有统一的 Jittor
native state owner，也没有 native generator-aware CPU/CUDA `randint`，无法满足“Jittor 自己
的状态可以保存恢复”的要求。因此在当前分支中已将随机数部分调整为 Jittor 原生 Philox
方案：`curand` wrapper 维护版本化的 seed/counter 字符串，发布
`jt.get_cuda_rng_state`、`jt.set_cuda_rng_state` 等 API；`jt.Generator` 和
`generator_randint` 在 CPU/CUDA 下直接使用同一套 Jittor-owned stream，Torch compat 只做
状态字节编码、参数检查和 sampler 调用转发。该方案不承诺与 Torch 的随机值逐位一致。

### 2. Var 的隐式设备与显式放置

冲突文件：`python/jittor/_core/var.py`。

目标分支的 `_factory_scope_like`/placement 逻辑服务于多设备张量创建和 DDP；2.0 引入
`device_scope_like`、dispatch table 和更完整的设备归属判定。这里不能简单按文件一侧
覆盖，否则可能把未放置 Var 的 ambient CUDA 语义或 DDP 的显式设备语义删掉。需要确认
最终实现采用 2.0 的显式 placement、当前 device 和 dispatch backend 组合规则，并在
`_factory_scope_like` 外层保留目标分支的 frontend type token。没有留下待定冲突。

### 3. 优化器状态、图捕获与原生 DDP

冲突文件：

- `python/jittor/optim/base.py`
- `python/jittor/optim/algorithms/adam.py`
- `python/jittor/optim/algorithms/adan.py`
- `python/jittor/optim/algorithms/rmsprop.py`
- `python/jittor/optim/algorithms/sgd.py`

目标分支包含原生 DDP 需要的梯度/状态同步和 FP32 算术路径；2.0 增加优化器状态设备
对齐、zero-grad 缓存、fused CUDA optimizer 及 step capture 拒绝规则。建议最终做语义
混合实现保留目标分支的 `_optimizer_arithmetic`、DDP reducer/step-ready 检查，同时
使用 2.0 的 state buffer 对齐、fused kernel 和 step capture 规则；验证重点是
多卡梯度同步与 fused optimizer 的共同路径。

### 4. Torch compat/FSDP2

冲突文件集中在 `compat/fsdp2/`、`compat/torch/` 和对应 Torch 测试。按任务范围说明，
这些不是原生 DDP 核心；本次统一采用 `origin/2.0-refactor` 的最新版本，避免将目标分支
此前的兼容层实验与新独立 Torch 前端混合。后续只需运行来源分支对应的 compat 结构和
行为门禁确认基线没有回归，不需要维护者逐文件选择。

## 已处理的冲突

截至当前阶段，以下部分已在合并树中完成并暂存：

- 文档规则、公共导出、验证结果索引和 `MANIFEST.in`；随机数新增的三个 native 源文件
  已登记到 manifest。当前工作区的 `python tools/build/generate_manifest.py --check`
  仍会看到一条由合并前已存在的未跟踪文档
  `docs/development/2.0-refactor-onboarding.md` 引起的生成差异；本次没有把该无关文档
  纳入提交。
- 算子类型和日志头文件，采用 2.0 的 `expand_op(..., bool is_cuda)` 接口、数值身份和
  并发编译安全实现。
- `python/jittor/_core/var.py`：组合了 2.0 的显式/隐式 device scope、`keepdim` 和
  reshape/transpose 优化，以及目标分支的前端类型传播。
- `python/jittor/ops/` 三个冲突文件：保留整数数组索引和视图语义，采用 2.0 的非连续
  storage 防护与按输入设备分配输出。
- `python/jittor/optim/` 五个冲突文件：保留目标分支 DDP 梯度同步、step-ready 检查和
  FP32 frontend arithmetic，同时保留 2.0 的状态设备对齐、fused CUDA/ACL 路径和图捕获
  参数检查。
- ACL/runtime 结构门禁和 AMP 测试：采用 2.0 的最新测试，并保留原生多卡测试文件。
- 发现并补齐 2.0 最新基线中 `compat/torch/serialization/portable.py`、
  `compat/torch/nn_modules.py` 使用而 `compat/torch/types.py` 缺失的
  `_set_meta_placeholder` 辅助函数；实现沿用该符号在历史基线中的标记语义，不改变
  原生多卡代码。

上述冲突取舍均已落到合并树，当前没有未解决的 index entry；后续差异属于运行验证中
发现的真实回归，而不是尚未决定的文本冲突。

## 验证记录

以下命令均在本 worktree 中执行，Jittor 编译缓存位于
`/home/cgq/jittor-lab/_state/cgq-transformers-full-2.0/structure/`，并通过
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` 使用与 `/usr/bin/g++` 匹配的
系统运行库：

- `bash tools/check_repo_layout.sh`：通过。
- `python tools/build/generate_manifest.py --check`：通过。
- `python -m compileall -q python compat tests`：通过。
- 原生模式下 `tests/distributed/test_launch.py`、`test_native_process_group.py`、
  `test_native_ddp.py` 和 `tests/data/test_dataset_native_distributed.py`：42 passed。
- `JITTOR_TORCH_SHIM=1` 下的优化器、runtime、ACL 结构门禁：127 passed。
- `CUDA_VISIBLE_DEVICES=5,7,8,9 NCCL_P2P_DISABLE=1` 下通过
  `python -m jittor.distributed.launch --backend nccl --standalone --nproc-per-node=4`
  运行 `tests/distributed/native_nccl_smoke.py`：4 个 rank 返回 0，`NATIVE_NCCL_SMOKE_OK
  checks=20 nccl_version_code=22105 bf16_tested=True`。

本轮没有宣称默认 P2P 路径；`NCCL_P2P_DISABLE=1` 是当前机器的冻结条件。正式 P2P
主机仍需另行复验。
- 完整 `tests/structure`：`1394 passed, 4 skipped`（收集 1398、实际执行 1396）。
  为适配 2.0 新增的两个 Accelerate 测试 fixture，结构门禁登记了重复的
  `_isolated_state` 测试脚手架名称；其余重复实现检查保持原规则。

合并后随机数专项复核（当前工作树）使用独立 `JITTOR_HOME` 和 `cache_name=rng_native_merge`：

- `tests/runtime/test_philox_rng.py` 与 `tests/runtime/test_generator_randint.py` 的 CPU
  用例 7/7 通过；包含 CPU 状态恢复、惰性反向求值、宽范围 int64 和全局 RNG 隔离。
- `tests/runtime/test_generator_randint.py` 在真实单卡 CUDA 上 7/7 通过；
  `tests/backends/cuda/test_cuda_rng_state.py` 5/5 通过，双卡用例因只有一张可见卡跳过。
- `compat/tests/torch/test_torch_sampler_rng.py`、`test_generator_streams.py` 的专项用例
  11/11 通过；`compat/tests/torch/test_torch_cuda_rng_state.py` 8/8 通过；显式 CPU/CUDA
  generator `randint` 的 11 个兼容用例通过。
- CPU `torch.random.fork_rng` 的原 expected-failure 已转为普通测试并通过，证明 CPU state
  现在保存了随机位置而不只是 seed。
- Accelerate CUDA fresh-process resume 仍有一个独立的设备放置失败：输入已经在 CUDA，
  但 `accelerator.prepare` 后的线性层权重仍在 CPU；该失败发生在矩阵乘法 dispatch，
  不是 RNG state 保存/恢复路径，本轮没有把它伪装成随机数回归。

## 当前状态

本地已创建消息为 `合并 2.0-refactor 完整基线` 的双父合并提交
`085e5d51b`，尚未推送；本次随机数复核和 native 修复作为该合并提交之后的当前工作树
变更维护。主工作区和远端 `cgq_transformers` 仍保持 `a2850846a`。后续若要更新远端，
应以随机数修复提交后的 HEAD 为起点执行独立的远端兼容复核和推送流程。

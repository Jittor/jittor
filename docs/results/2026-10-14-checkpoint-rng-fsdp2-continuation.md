# Native RNG 恢复与 FSDP2 FULL checkpoint 续跑

Status: CPU 与 CUDA 单进程 Accelerate RNG 与数值续跑已验证；双 rank CUDA FSDP2 FULL
模型/优化器续跑已对照独立 PyTorch 验证。分片分布式 checkpoint I/O 与 ACL 仍未验证。
本报告取代此前[优化器 checkpoint 报告](2026-10-14-optimizer-checkpoint-continuation.md)
中的 RNG 与多 rank 限制。

Baseline: `7602f9040c1d63a61209bc50a575cb440c787d57`，整合第一个 checkpoint 实现
`1da40ecc0` 与拉取的 `origin/2.0-refactor` 提交 `7ed9b3b52`。
实现：包含本报告的后续提交。维护者：Jittor 兼容性维护者。
复查条件：native RNG 快照格式、Torch RNG owner、checkpoint FQN 或 rank-zero 变换、
FSDP shard 替换或下游版本发生变化。

## 已验证的失败与修改

一个真实的 `Accelerator.save_state`/`load_state` 探针最初恢复了 Python 与 NumPy
流，却把 Torch CPU 流倒回其种子。前端现在把现有 native CPU generator 快照编码为
CPU `uint8` tensor，并通过 native owner 恢复该完整快照。这保留了已推进的流位置，
且不会重设 CUDA 种子。Native CUDA 已提供 cuRAND seed/offset 快照，无需修改。
非法字节载荷与仅含种子的旧版整数 tensor 会明确失败；非法恢复会保持当前流位置不变。
快照字节是不透明的，不能在 native Jittor 与 PyTorch generator 之间移植。

已有的 advanced-prefix `torch.random.fork_rng` 回归现在无需 expected-failure
标记即可通过。CPU RNG 测试即使在 CUDA 进程中也会显式选择 host 后端。实际的
Accelerate 回归在恢复后检查 Python、NumPy、Torch CPU 以及所选设备的下一批随机值，
然后检查续跑后的模型、moment、step、梯度、scheduler 与 loss。这是固定输入的
训练用例，带有独立的流续跑断言；它不覆盖打乱的 dataloader 或任意自定义对象恢复。

实际的双 rank FULL 保存/加载用例暴露了三个额外边界：

1. 即使对 FULL 文件，Accelerate 也会导入 `checkpoint.default_planner`。两个 planner
   名字现在都可导入、注册为 `UNIMPLEMENTED`，且其构造函数无条件抛出
   `NotImplementedError`。FULL 路径使用普通 `torch.save`/`torch.load`；这不实现分片规划。
2. 一个非零 rank 在 rank-zero 序列化前丢弃了惰性 FULL gather，同时推进到 optimizer
   元数据集合通信。两个 rank 在不同的 NCCL 操作处挂起。现在 host 卸载的 tensor 会在
   任何 rank 丢弃其 FULL 结果之前物化。保留的超时 trace 识别出 rank zero 位于序列化的
   `sync_all` 内，rank one 位于元数据 all-gather 内。
3. FULL 模型加载会刷新 flat shard 视图。优化器加载现在在解析 FQN 之前使用现有的
   FSDP 参数刷新 owner，在 shard 替换后保留活优化器所有权。没有更新数学变化。

## 运行与数值证据

两个运行时都使用 Accelerate 1.10.1、离线确定性权重与输入。独立 oracle 为二进制
PyTorch 2.11.0+cu128，shim 身份不存在且 Jittor 未加载。缺失的环境包在外置隔离
overlay 中提供；共享环境未被修改。CPU 与 CUDA 使用独立缓存，初始 native/NCCL
编译在双 rank 执行前完成。Checkpoint 数值验收用例禁止后端回退，未观察到计数变化。

| 验证 | 结果 |
| --- | --- |
| 相同的单进程 checkpoint 用例，CPU | PyTorch 7/7；shim 7/7 |
| 相同的单进程 checkpoint 用例，CUDA | PyTorch 7/7；shim 7/7 |
| 合并的 CPU 序列化/FSDP 所有权/RNG/checkpoint 回归 | 79 passed，2 个加速器 skip |
| CUDA RNG/checkpoint/planner 回归 | 24 passed，无 skip |
| 来自 `core` 白名单的 Torch 兼容文件 | 224 passed，1 个加速器 skip |
| 双 rank FSDP2 FULL Accelerate 保存/加载 | 两个运行时、两个 rank 均通过 |
| 单进程最大绝对误差，CPU / CUDA | 1.49e-8 / 3.73e-9 |
| FULL checkpoint 最大绝对误差，任一 rank | 1.87e-9 |

跨运行时比较包含每个保存/恢复/续跑的模型 tensor、优化器 moment 与 step、学习率与
续跑 loss。单进程比较另包含所有参数梯度与输入梯度。容差为 `atol=2e-6, rtol=2e-5`。
PyTorch 额外的 AdamW group 字段 `decoupled_weight_decay=True` 已显式验证并从共享
元数据相等性中排除；其余所有元数据键均比较。Generator 算法与不透明 RNG 字节未在
框架之间比较：每个运行时必须在恢复自己的 checkpoint 后精确复现自己的下一批值。

双 rank 用例写入了实际的 `pytorch_model_fsdp.bin`、`optimizer.bin` 以及两个每 rank
的 `random_states_*.pkl` 文件。报告确认两个 NCCL rank、真实 CUDA 参数设备、非零
storage 指针以及 shim/oracle 身份。FSDP2 用例检查固定输入数值续跑；详细的 RNG 断言
在单进程 CPU 与 CUDA 用例中运行。

必需的仓库布局检查通过。完整 `tests/structure` 运行完成，1426 passed、4 skipped、
4 failed：隔离环境中 ASV 不可用、既有 ACL Dropout 梯度清单不一致、未登记的报告
toctree 条目，以及生成的打包 manifest。后两项需要集成 owner 更新共享 index/manifest；
这些文件明确不在本工作树的编辑所有权内。原始输出为 `logs/followup-structure.log`。
这不是全绿结构声明。

## 复现与证据边界

2026-10-15 的整合复验从同步后的主树基线 `ad47b58143f5c54e83114f923e23c0c69322130a`
开始。完整的 native 加 Torch CPU `core` runner 通过，322 passed、42 skipped 和一个
既有预期失败。合并的真实 CUDA AMP/RNG/checkpoint 选择全部 127 个用例通过，无 skip。
在更新共享报告索引、打包 manifest 与 CodeOp 梯度清单后，完整结构选择通过，
1432 passed、6 skipped、无失败。两个结构 skip 涉及此仅源码解释器中的 distribution
解析；其余为声明或加速器能力 skip。仓库布局与空白检查通过。本次复验取代上面的工作树
结构失败，且不增加 ACL 或分片 checkpoint 验收。未版本化整合日志位于
`$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/`，名为
`amp-rng-integration-core.log`、`amp-rng-integration-cuda.log` 与
`amp-rng-integration-structure.log`。

使用独立的、版本匹配的环境，shim 只部署在其自己的解释器中。先在 CPU、再在 CUDA 上
运行这些选择，为 checkpoint 测试设置 `CHECKPOINT_DEVICE` 并显式选择匹配的 native
后端。设置 `JITTOR_REQUIRE_ACCELERATE=1` 以拒绝缺失依赖：

```bash
python compat/tests/torch/test_optimizer_checkpoint_state.py -v
JITTOR_TORCH_SHIM=1 python -m pytest -q --rootdir=. \
  compat/tests/torch/test_torch_fork_rng.py \
  compat/tests/torch/test_core_misc_owner.py \
  compat/tests/torch/test_checkpoint_planner_boundary.py \
  compat/tests/torch/test_optimizer_checkpoint_state.py
```

双 rank shim 执行时，启用已安装的 NCCL 后端并使用分配两张卡的 native launcher。
首次编译必须串行完成。设置 `JITTOR_ACCELERATE_DISTRIBUTED_REQUIRED=1`、
`ACCELERATE_USE_FSDP=true`、`FSDP_VERSION=2`，然后执行：

```bash
python python/jittor/distributed/launch.py -n 2 --backend nccl --standalone \
  --logdir "$JITTOR_LAB_ROOT/_state/checkpoint-optimizer/fsdp-shim-logs" \
  -m pytest -q -s --rootdir=. --timeout=90 --timeout-method=thread \
  compat/tests/torch/test_accelerate_distributed_integration.py::test_fsdp2_full_model_optimizer_checkpoint_resume
```

用独立 PyTorch 的双进程 launcher 运行同一用例。oracle 运行使用了测试的字节一致外置
副本，不含仓库 conftest 安装钩子，并在每份报告中断言 `torch_compat_active=False` 与
`jittor_loaded=False`。这避免了通过测试收集把 oracle 变成 shim。

原始日志、JSON 报告、修复前失败的 RNG 探针、双 rank 超时 trace 以及严格数值比较脚本
保留在仓库外的 `$JITTOR_LAB_ROOT/_state/checkpoint-optimizer/` 下。相关日志为
`logs/rng-cpu-targeted.log`、`logs/rng-cuda-targeted.log`、`logs/followup-targeted.log`、
`logs/core-torch.log`、`logs/fsdp-reference.log`、`logs/fsdp-shim-refresh.log`、
`logs/numeric-comparison.log` 以及对应的每 rank 日志目录。`core` 结果是 Torch 兼容
白名单选择，而非声称完整 native 加 Torch runner 已运行。

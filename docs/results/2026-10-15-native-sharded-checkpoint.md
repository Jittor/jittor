# 原生分片 checkpoint 与训练续训验证

- Status: 本报告限定的 native checkpoint I/O、CPU / 单机双 CUDA rank 续训通过；训练 FSDP / TP / ZeRO 与真实多机仍未验收。
- Date: 2026-10-15。
- Baseline: 初始 `a1a3f44afdcec9d0ce3670ff77e78c09f16b80a2`；停止测试、保留所有工作并 fetch / fast-forward 后，最终基线 `8e388d108eb83c1c438936709f29855f47ba2a69`。
- Owner: 原生 serialization、distributed、optimizer 维护者。
- Review when: 原生 Module / Optimizer / RNG 协议、Store、tensor placement、BF16 storage、NCCL 或 rank/world 布局变化。

新增 `jittor.distributed.checkpoint` 公共入口与 native serialization 叶子 codec，
不 import Torch/compat、不复制 FSDP 或 optimizer 数学。各真实 rank 分别写出自己的
payload，Store 只交换元数据和状态，完整验证后原子发布 manifest。
`save_training_state` / `load_training_state` 消费现有 Module、Optimizer 与原生 RNG
owner，保存 FQN、trainability、shape/dtype、optimizer class 与有序绑定、超参数、
moment / step、Python / NumPy / native CPU / CUDA RNG 以及额外状态。

模型读取通过现有 native Module.update owner，它会继承 source 的 gradient flag；
因此不能把 generic I/O 返回的 detached tensor 直接用于训练恢复。helper 单独保存
并在加载 source 上恢复 flags，保存过程不改变原模型 flags，不修改 Module owner。
恢复前检查全部 payload 与 FQN/shape/dtype/flags/optimizer 绑定和 state slot，
非法 RNG 也回滚当前随机流。原模型与新模型、冻结 bias 与可训练 weight 均进入实际续训。

原生 BF16 `numpy()` 导出 FP32，不能据导出 dtype 推断参数或 momentum storage。
codec 通过 UInt16 reinterpret view 保存并恢复 BF16 原始位，包含 signed zero、
非有限值、NaN payload 与 subnormal。训练 fixture 同时断言 parameter / native SGD
momentum 实际为 BF16；使用可精确表示的 dyadic 输入，独立 NumPy FP32 SGD 每一步
检查低 16 位为零，再比较完整 BF16 原始位、momentum、实际梯度、loss 与 counter。

本次另修复 native optimizer 的 MPI-free world=1 NCCL legacy 断点：distributed
状态为真而历史 Var.mpi_all_reduce 别名未安装。MPI-free 同步薄转发现有 collective
owner；真实 MPI 保留原 method 调用，DDP 已管理参数仍排除，singleton owner 返回
原 tensor 时不做 self-assign。没有改变平均、参数广播或更新公式。

## 最终验证

Python 3.11.16、NumPy 1.26.4、pytest 7.4.4，独立环境与源码路径；没有修改共享
Python 环境或安装新下游库。两个 NVIDIA RTX 4090，nvcc 12.2.140 / GCC 11.5，
实际加载 NCCL `21803`，MPI 关闭。CPU、各 CUDA rank、launcher 与 core 两模式缓存
独立，首次 core / NCCL / BF16 编译依次预热。最终两 rank 累计 fallback 均为 0，
全过程断言没有 import Torch，数学参考为独立 NumPy 实现。

| 验证 | 实际结果 |
| --- | --- |
| native CPU checkpoint / 精确续训 / 冻结参数 / BF16 / 拒绝路径 | 21 passed，0 skipped |
| 真实 host TCPStore 两 client 元数据交换 | 包含在上述 21 项，公共 wait(timeout) + get(key) 合同 |
| NCCL world=1 unmanaged / DDP optimizer | 2 passed，5 步独立 NumPy Adam |
| 最终单机两 rank native optimizer + checkpoint | 每 rank 3 passed、0 skipped，launcher rc 0 |
| existing / fresh model 恢复后两个更新 | 全部模型、m/v、梯度、loss、step 精确相等 |
| Python / NumPy / native CPU / CUDA 下一组随机值 | 每 rank 全部精确相等 |
| native BF16 参数 / BF16 momentum SGD 3 步保存 + 2 步恢复 | CPU 与真实 CUDA 通过，原始位与 NumPy dyadic 轨迹全等 |
| CPU serialization / optimizer / Store 邻近回归 | 35 passed / 2 Torch-mode skips / 1 xfailed |
| CPU core 两模式 | 322 passed / 42 skipped / 1 xfailed，两个模式 exit 0 |
| 完整 structure | 1451 passed / 6 skipped / 2 central-document-registration failures |
| 布局 / whitespace | 通过 |

完整 structure 的两个失败仅为本次新 note / report 尚未进入对应 toctree，以及生成
MANIFEST 尚待集中登记；没有声称整个结构门禁通过。测试 helper 依赖与 optimizer
communication ownership 的实际违规已修复，定向 8 passed，并完成上述完整复验。

CUDA FP32 Adam 的独立 FP64 NumPy 参考使用原生 epsilon-before-bias-scaling
约定；它与 `torch.optim.Adam` 的 epsilon 位置不同，未擅自对齐数学。
五步轨迹最大绝对差：参数 `1.3598102172052151e-8`，moments
`5.952429563138573e-9`，实际均值梯度 `2.5982546839747878e-8`，两 rank
loss 分别 `4.183892484202478e-9` / `4.018982271980498e-9`。
阈值仍为 `2e-6 + 2e-5 * abs(reference)`；恢复自身轨迹用逐值相等，无放宽。

CUDA 计算的 prediction、loss、梯度和更新后的参数/moments 在 NumPy 读回前
同步并断言 device residency，CUDA driver pointer attribute 9 验证真实 allocation
属于该 rank 的 logical device 0。`numpy()` 的显式 host snapshot 会把 residency
变成 CPU 而保留 logical placement；load / I/O 后通过现有 device pointer accessor
物化 CUDA placement 再验指针，不能把 host snapshot 误认为计算 fallback。

两 rank 还实际验证不同 local 输入的 flat interval 重组与局部恢复、replicated
元数据不一致和仅一个 rank writer 失败；后两者在所有参与 rank 抛错且无完整 marker。
CPU 检查 missing/corrupt payload、FQN/group 顺序、flags、shape/dtype、moment、
RNG payload/backend 拒绝且目标参数/flags/LR/随机流保持不变；失败 overwrite
保留旧完整 manifest，overlap/gap 拒绝。

## 边界与复现

`TensorShard` 描述 checkpoint 的元素区间，不负责训练 sharding。普通输入是
replicated：每个 rank 完整 host snapshot 后校验全量 digest，再写自己的区间；
这一条没有 bounded host-memory 承诺。显式 TensorShard 才消费既有 local 输入，
局部恢复只分配所请求的输出范围。源码/测试不测显存峰值或性能，不作 L4 结论。

训练 helper 仅支持 completed optimizer step / zero_grad 边界，明确拒绝 pending
accumulation；严格校验同 rank set / world、执行 backend 与可见 CUDA RNG device
数量。world-size RNG 迁移、custom load hook 的完全事务性、多节点、native FSDP / TP /
ZeRO、ROCm/NPU/Windows、MPI 实际回归和掉电/worker-death容错不在本次通过范围。
NPU storage 显式拒绝，桥接的 HCCL 分支未授予硬件验收。

公共 API 与 shared-filesystem / Store / generation 清理规则见
[机制说明](../notes/native-sharded-checkpoint.md)。入口：

```bash
JITTOR_TORCH_SHIM=0 JT_BACKEND=cpu JT_USE_CUDA=0 JT_BUILD_NVCC_PATH= \
  PYTHONPATH=python <jittor-python> -m pytest -q tests/serialization/test_sharded_checkpoint.py
# world=1 使用 JT_NCCL_WORLD_SIZE=1 的真实 CUDA 环境，运行 optimizer_sync 文件。
# 双 rank 使用现有 launcher，每个 child 在导入前按 RANK 指定独立 cache/home。
<jittor-python> python/jittor/distributed/launch.py --backend nccl -n 2 \
  --device-ids=<two-gpus> -- <jittor-python> <isolated-rank-wrapper>
```

设置 `JITTOR_CHECKPOINT_TEST_DIR` 为新的 shared path；rank wrapper 在当前 worktree
调用 `pytest -q -s tests/backends/comm/nccl/test_native_optimizer_sync.py
 tests/backends/comm/nccl/test_native_sharded_checkpoint.py`。shim 关闭，CUDA kernel math
strict，真实 NCCL include/lib，fallback error scope。NCCL timeout 由 Store 和本 API
共同约束，Store.get 的签名不接受 timeout，应使用 wait 的公共合同。

原始记录、rank wrapper、manifest/payload、失败部分 generation 和独立缓存位于
`$JITTOR_LAB_ROOT/_state/native-sharded-checkpoint/`，未版本化。最终关键日志为
`logs/cpu-neighbors-8e-owner.log`（包含全部 21 项新增测试）、`logs/cuda-world1-optimizer-8e-owner.log`、
`logs/two-rank-8e-owner-launch.log`、`logs/two-rank-8e-owner/rank{0,1}.log`、
`logs/core-cpu-8e-owner.log` 和 `logs/structure-8e-owner.log`。初始 a1 基线成功日志
`logs/two-rank-tcp/rank{0,1}.log` 单独保留；最终基线重新执行未复用旧通过标记。

## 主分支整合复验

在 `b930461562e73c67332581fe2384eaf8cfd7d838` 上整合子任务提交
`82fb93150fc51e0608db7f95e08f8d48bb7f78ec`，补齐两份文档的 toctree 并重新生成
MANIFEST。八份 Python 源码/测试与已完成双卡验证的子任务版本逐字节一致；主分支
相对其原测量基线的其他变化仅为 Transformers 报告和 skill 文档。

整合后 CPU core 为 322 passed / 42 skipped / 1 existing xfailed，新 checkpoint
CPU 测试为 21 passed / 0 skipped，完整 structure 为 1453 passed / 6 skipped / 0 failed。
布局与 whitespace 检查通过。前述中央登记失败至此已修复；保留初次结果以区分两轮验证。
原始整合日志和源码 SHA256 对照位于
`$JITTOR_LAB_ROOT/_state/native-checkpoint-integration/`，未版本化。

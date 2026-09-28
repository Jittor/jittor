# 原生单机多卡收口验证（2026-09-28）

## 范围与基线

本轮以远端 `2.0-refactor` 的 `d28bd5d98` 为代码基线。远端没有原生 DDP 文件，这是本任务的
实现范围，不视为缺失；Torch compatibility 层的其他变更也不纳入本轮。原生入口由
`jtrun + NCCL + jittor.distributed + DistributedDataParallel` 组成，训练更新仍使用
`optimizer.backward(loss)` 与 `optimizer.step()`。多机、FSDP、TP、ZeRO 和 `compat.torch`
核心逻辑不在范围内。

本机验证条件固定为 `NCCL_P2P_DISABLE=1`，设备为物理卡 `5,7,8,9`。`jtrun` 对每个 NCCL
子进程强制注入该变量，避免不同 rank 选择不同传输路径；因此本报告不宣称 P2P 已验证。

## 已完成适配

- `jtrun` 支持单机每卡一个进程、设备映射、TorchRun 风格环境变量、独立 rendezvous、日志、
  失败 rank 传播和清理；显式 NCCL 启动不需要父进程导入 Jittor。
- `jittor.distributed` 提供初始化/销毁、rank/world-size/local-rank、NCCL/HCCL 后端选择、
  `all_reduce`、`broadcast`、`all_gather`、`reduce_scatter`、`barrier`、`new_group` 和
  同步 `Work` 兼容句柄。
- 原生 DDP 在构造时校验模块签名并广播参数/buffer；同步 backward 对稳定参数顺序做梯度
  mean-reduce，支持 `no_sync()`、缺失/重复参数检测和未同步梯度的 step 拒绝。NCCL 梯度
  归约使用已有 bucket scope，将同一轮的多个 all-reduce 合并为一个 NCCL group。
- Dataset 改用原生 process group 的 rank/world-size，支持 `set_epoch` 和按
  `seed/epoch/rank/worker` 派生 worker seed；非整除 global batch 的尾部按固定 local batch
  大小补齐尾样本，保证各 rank 的 collective shape 一致。Sync BatchNorm 改走原生
  `distributed.all_reduce`。
- 训练基准同时保存模型和 optimizer state；恢复后用同一个 continuation step 与原训练器
  逐 loss、逐参数对拍。

## 证据

CPU 定向回归：`test_native_ddp.py`、`test_native_process_group.py` 和 Dataset 原生分片
测试共 **27 passed**；启动器测试（含 P2P 环境冻结）**15 passed**。GPU 4-rank 冒烟使用
`CUDA_VISIBLE_DEVICES=5,7,8,9`，共 **20 项检查通过**，覆盖 NCCL sum/broadcast/all-gather、
BF16 归约、Dataset 分片、SyncBN、参数/buffer 广播、梯度平均、`no_sync`、最终副本一致性，
日志中的 NCCL 版本为 `2.21.5`。

在同一合成 MLP（input 256、hidden 512、16 类、50 步 Adam、warmup 10）上，4 卡 global
batch 128 的稳定吞吐实测约 **6.5k--8.7k samples/s**（卡上已有负载，结果有波动）；最新一
轮为 `6492.6 samples/s`，loss 从 `2.7680` 降到 `1.4126`，replica 一致、模型恢复和
`optimizer_resume_matches=true`。固定 global batch 32（每卡 batch 8）时 bucket 版本为
`1800.4 samples/s`，未分组归约对照为 `1263.2 samples/s`，说明 bucket 确实降低通信开销。
单卡 batch 32 同模型约 `5275.4 samples/s`；该数字只用于同机参考，不能与 global batch 128
直接解释为线性加速。

## 尚未宣称完成的事项

`Work(async_op=True)` 当前仍是同步完成句柄，NCCL/HCCL communicator 没有可调用 teardown，
默认 P2P 路径未在正式 P2P 主机验证，吞吐仍受本机共享内存路径和外部 GPU 负载影响。还需要
在 P2P 可用机器上复验传输路径，并在更多常见模型/优化器和冷缓存环境下扩大回归；这些不改变
本轮单机、NCCL、原生 DDP 的功能边界，也不引入多机或 FSDP 目标。

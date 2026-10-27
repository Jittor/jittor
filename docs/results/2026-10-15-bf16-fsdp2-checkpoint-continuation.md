# 双卡 bf16 FSDP2 FULL 保存与续训验证

- Status: 本报告限定的 CUDA FULL 保存、恢复和固定输入续训通过；跨框架 bf16 数值保留近似边界。
- Date: 2026-10-15。
- Baseline: `ad47b58143f5c54e83114f923e23c0c69322130a`，顺序合入 RNG `1e2ff1aa6b3a391c0c84747413c01bc2731e5626` 与 AMP `de23770c70f993b4d4fe818a763f9edca7486381`；隔离验证基底 `5274aeb6ecf1791029889263bc3ab6cc88324a4f`。
- Owner: Torch 兼容层、FSDP2 与 checkpoint 维护者。
- Review when: FSDP2 mixed precision、FULL checkpoint、RNG、Linear 舍入策略、Accelerate 或独立 Torch 版本变化。

本次只增加回归和证据，没有更改计算实现。单机两个真实 CUDA rank，固定 4×4 输入、
3×4 Linear 权重与 bias，两组 AdamW。FP32 保留一更新后保存；新增 bf16 行先更新
三步，保存 FULL 模型与 optimizer，再运行两步作为预期轨迹，恢复后重新运行两步。
完整模型、FQN optimizer 状态、实际梯度、loss 和超参数均进入每个 rank 的记录。

Linear 子类在 Accelerate 输出转换之前观察 input、weight、bias、output；七次前向
的四种张量全部为 bf16。CUDA driver 的 pointer attribute 9 验证每个观察张量确实
分配在该 rank 的 CUDA device。每步检查 FP32 主分片、FP32 实际梯度、FP32
`exp_avg` / `exp_avg_sq`；loss 明确为 FP32。FULL shape 从实际完整状态推导，
避免把 native flat 本地 shape 当成 global shape。

恢复断言为逐值精确相等，不使用跨框架数值容差。保存前先推进 Python、NumPy、
Torch CPU、Torch CUDA 随机流；恢复后的下一组值也逐值相等。训练输入本身固定，
这不是 dropout 或随机采样训练的额外覆盖。Jittor 两 rank 均处于 fallback `error`
scope，最终累计 fallback 为 0。两侧 matmul / cuDNN TF32 均关闭，fixture 恢复原配置。

## 实际结果

Python 3.11，pytest 7.4.4、Accelerate 1.10.1 两侧相同。Jittor 2.0.0 的 Torch
声明 API 为 2.6.0；独立参考为真实二进制 PyTorch 2.11.0+cu128，报告确认未加载
Jittor、未激活 shim。硬件为两个 NVIDIA RTX 4090；Jittor GCC 11.5、nvcc 12.2.140。
CPU、两个 CUDA rank 使用独立 JITTOR_HOME，首次核心、扩展与 bf16/FP32 NCCL
算子暖机依次串行完成。没有修改共享 Python 环境或安装新的下游库。

| 验证 | 结果 |
| --- | --- |
| CPU 独立 Torch / Jittor bf16 前后向预检 | 输出一致，真实 CPU，FP32 梯度；Jittor fallback 0 |
| 最终 fresh 两 rank，FP32 × 两 runtime | 每 rank 1 passed，两个 launcher rc 0，0 skipped |
| 最终 fresh 两 rank，bf16 × 两 runtime | 每 rank 1 passed，两个 launcher rc 0，0 skipped |
| FULL 恢复、两步续训、四类 RNG | 两 runtime 各 rank 精确相等 |
| CPU Torch core 白名单 + FSDP mixed precision / optimizer checkpoint / AMP / planner 邻近选择 | 253 passed、1 accelerator skip，257.56 s，包含冷算子编译 |
| 完整 `tests/structure` | 1426 passed、4 skipped、4 failed，213.80 s |

结构失败为此隔离基底已有的 ASV 依赖缺包、ACL Dropout 梯度 manifest，以及等待
集中维护的结果索引 / MANIFEST；没有声称完整结构门禁通过。新增报告也需要集中
更新索引与 manifest。布局检查和 diff whitespace 检查通过。
报告落盘后的文档、打包与私人路径选择为 25 passed、2 failed，仍为共享索引与
MANIFEST 两项；私人路径及其余文档契约通过。

两侧最终八份 JSON 的测试源码 SHA256 均为
`5edc207e7892c8367cc91e17f54d45b6b68209c1bab7a891ec4aa47c2e3581af`。
FULL weight / bias shape 均为 `[3,4]` / `[3]`，本地分片布局允许两个 owner 不同。

## 外部数值对拍与边界

FP32 全轨迹最大绝对差 `1.862645149230957e-9`，通过既有
`2e-6 + 2e-5 * abs(reference)` 阈值。bf16 最大绝对差为
`1.388988457620144e-4`；上述 FP32 阈值有 194 / 188 个标量比较不通过，原始结果保留。

对每个浮点 tensor，以最大绝对差除以参考 tensor 的最大绝对值，bf16 全轨迹上限为：
loss 0.736%、模型 0.604%、梯度 2.381%、moments 1.882%。相对于 bf16 epsilon
`2^-7`，分别为 0.942、0.773、3.048、2.409 倍；近零梯度与 EMA 的相消会放大
这一归一化误差。这些是本固定五步规格的实测值，不把它们变成全库容差或 EXACT 承诺。

最小 CPU/CUDA 复现使用完全相同的 bf16 x / weight / 非零 bias，分别比较融合
`F.linear` 与 `x @ weight.T + bias`。Jittor CUDA 的 output、loss、input grad、
weight grad、bias grad 与独立 Torch 的后者逐位相同；Torch 融合 Linear 有不同
舍入位置，单步 output 最大差 `4.8828125e-4`、bias grad 最大差
`2.44140625e-4`。因此差距不由错误 dtype、NCCL 或缺失梯度链造成，既有 Linear
`APPROXIMATE` fidelity 保留，没有为逐位匹配扩写 kernel 或降回 FP32 训练。

CPU 另发现 lazy fusion 边界：不物化 bf16 输出时，声明为 FP32 的 loss 与 Torch
unfused 最多差 `1.8621794879436493e-6`；先物化输出后差不超过
`1.1641532182693481e-10`，全部梯度仍一致。保留为 CPU 精度限制，未在本次修改
compiler；本报告不声称 CPU FSDP2 或所有无物化 bf16 图已通过。

同进程连续跑两个 precision 行，独立 Torch 曾在第二次 `ProcessGroupGloo`
初始化超时 180 s，尚未进入 bf16 计算。原先合并执行也曾通过；最终验收为每个
precision 使用新的两 rank launcher，所有最终 launcher 均正常退出。该 oracle
重建竞态日志保留，不能用进程组初始化失败推断 checkpoint 计算失败。

## 复现入口

测试 node 为
`compat/tests/torch/test_accelerate_distributed_integration.py::test_fsdp2_full_model_optimizer_checkpoint_resume[bf16]`；
FP32 对应 `[no]`。环境需配置两个可见 GPU、NCCL include/lib、真实参考解释器与
当前源码 frontend。设置 `ACCELERATE_USE_FSDP=true`、`FSDP_VERSION=2`、
`JITTOR_ACCELERATE_DISTRIBUTED_REQUIRED=1` 和独立的 JSON 输出目录。

Jittor 用现有 `python/jittor/distributed/launch.py -n 2 --backend nccl --standalone`；
每个 child 在导入前按 RANK 指定独立 JITTOR_HOME / TMPDIR，再运行上述 pytest node。
参考用 `torch.distributed.run --standalone --nproc-per-node=2`，同一 node 的字节相同
外部副本、`--noconftest` 与外部 rootdir，避免 monorepo conftest 激活 shim；源文件
provenance 目录以只读链接补齐。每个 precision 使用新的 launcher。

原始日志、byte-identical 参考副本、rank wrapper、独立 cache 与八份完整 JSON 位于
`$JITTOR_LAB_ROOT/_state/bf16-fsdp2-checkpoint/`，未版本化。关键日志为
`logs/reference-exact-{no,bf16}.log`、`logs/shim-exact-{no,bf16}.log`、
`shim-exact-{no,bf16}-rank-logs/rank{0,1}.log`、`logs/final-comparison.log`、
`logs/final-quantification.log` 与 `logs/structure.log`。

未验证多节点、TP、ZeRO、SHARDED/DCP planner、跨 runtime checkpoint 互换、ROCm/NPU，
也没有据此声称整个 Accelerate 或 R17 已完成。

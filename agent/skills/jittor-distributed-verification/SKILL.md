---
name: jittor-distributed-verification
description: 修改 MPI、NCCL、rendezvous 或 Native FSDP 时，验证真实 rank、通信数值、生命周期与故障退出。
---

# 验证分布式执行与通信

## 先确定协议

- 区分 MPI、无 MPI 的 NCCL launcher 和 HCCL。记录所有 rank 的来源、设备、WORLD/subgroup
  及实际 backend；单 rank 快捷路径不能证明通信库执行。
- 首次编译串行准备。通信初始化必须让所有 rank 共同参与，不能持编译锁等待 peer；
  `JT_NCCL_WORLD_SIZE` / `JT_HCCL_WORLD_SIZE` 配置与 MPI 构建模式分开验证。
- 当前 launcher 的后端、多机范围与参数以源码和 `--help` 为准，机器上的 MPI 路径不是接口合同。

## 数值、顺序和生命周期

以 rank/iteration 构造独立参考，覆盖 all-reduce、gather、broadcast、reduce 的实际实现，
检查 dtype、shape、finite 与所有 rank 结果。通信前有设备生产者、通信后有消费者，避免
每轮主机读回把竞态同步掉。真实设备与无 fallback 由运行证据证明。

NCCL group 的 join/event 需在 `ncclGroupEnd` 之后，延迟 join 必须持有输入/输出 storage。
WORLD 与 subgroup 分别覆盖，不能将一种成功外推另一种。修改进程状态时只写唯一所有者，
检查所有读取视图跟随变化，而不是测试先手动更新每一份副本。

## 故障与证据

缺失 peer 要显式触发 `compile_extern.setup_nccl()`；lazy import 本身未必进入 rendezvous。
使用维护的故障测试按本任务 PID 和哨兵注入，不按进程名批量终止。记录超时、所有 worker
退出码、收集数量和清理结果；单 rank 退出成功不等于整个任务完成。

不要把所有非 NCCL kernel 都叫 default stream。重叠分析应按 device/context/stream
分组并取时间区间并集，简单相加会重复计算，甚至得到超过 100% 的“重叠率”。

## 按任务使用工具

- [collective_oracle.py](scripts/collective_oracle.py)：分别运行 Jittor/真实 Torch 的集合通信参考。
- [Native FSDP 协议](references/native-fsdp.md)：做训练对拍、fresh-launch 恢复或双机验收时阅读；
  包含 oracle、校准、runner 与原始证据复核工具，不放宽既有数值预算。
- [HCCL 设备验收](../hccl-on-device-verification/SKILL.md)：Ascend 多卡与同步策略。

宿主脚本测试只证明协议和证据处理，不能替代 GPU、多机或网络拓扑验收。

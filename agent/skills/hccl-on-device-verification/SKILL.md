---
name: hccl-on-device-verification
description: 在真实多卡 Ascend 环境验证 HCCL 初始化、集合通信数值和 full/stream-order 同步策略。用于修改 HCCL 发射或同步逻辑；主机桩测试、导入和单卡运行不能替代该验收。
---

# 验证 HCCL 同步策略

当前实现的 `hccl_collective_begin/end()` 在 `full` 模式下保留调用前后的设备与流同步；
`stream-order` 省去这些同步，依赖 `aclstream` 上的顺序。`JT_HCCL_COLLECTIVE_SYNC` 在
进程内首次读取后缓存，A/B 必须启动不同进程。默认仍是 `full`；文档整理、主机测试或
单次模型成功都不能授权修改默认值。

先读[Ascend 环境指南](../../../docs/guides/hardware/ascend-910b.md)和
[隔离验证环境](../isolated-validation/SKILL.md)。检查当前实现：

```bash
rg -n 'hccl_collective_(begin|end|full_sync)' backends/comm/hccl
rg -n 'JT_HCCL_|HcclCommInitRootInfo' backends/comm/hccl/src/hccl_wrapper.cc
```

## 1. 准备真实多卡任务

- 使用至少两张已分配的 Ascend 卡，记录型号、驱动、固件、CANN 与提交。
- 确认 CANN 头文件/库和实际设备可见；不能只以环境变量或 `has_acl` 宣称设备执行。
- 优先使用 `jittor.distributed.launch --backend hccl` 的 env/file rendezvous。
  当前 launcher 明确拒绝多机 HCCL；本技能不把单机验收外推为多机支持。
- `JT_HCCL_WORLD_SIZE/RANK/LOCAL_RANK/ROOTINFO_FILE` 由 launcher 为每个 rank 设置，
  不复用旧任务的 rootinfo 文件，不把物理卡号当进程内逻辑 rank。
- 首次核心/JIT/扩展编译串行准备，A/B 与其它任务使用隔离缓存和日志目录；通信初始化
  本身要求所有 rank 按协议共同参与。launcher 退出后确认没有残留 worker。

## 2. 先运行已有的设备准入用例

现有 `test_hccl_backend_status.py` 同时包含 CPU/mock 测试和显式 NPU 分支。
只运行下面的设备用例；`JITTOR_HCCL_STATUS_TEST_MODE=npu` 会要求真实已初始化的 WORLD、
rank/size 一致、实际 all-reduce 与空张量 broadcast，缺少设备或初始化会失败而不是跳过。

完成安装、CANN 加载和缓存准备后，在源码根目录运行：

```bash
export JITTOR_TORCH_SHIM=0
export JITTOR_HCCL_STATUS_TEST_MODE=npu
for mode in full stream-order; do
  JT_HCCL_COLLECTIVE_SYNC="$mode" \
    python -m jittor.distributed.launch --backend hccl --nproc-per-node "$NPROC" \
    --log-dir "$RUN_DIR/$mode" --timeout 120 \
    -- python -m pytest -q \
    tests/backends/comm/hccl/test_hccl_backend_status.py::test_hccl_status_on_declared_runtime
done
```

`NPROC` 不得大于实际分配的设备数，`RUN_DIR` 必须在 lab 状态目录内。外部指定
`ASCEND_RT_VISIBLE_DEVICES`，不要照抄机器设备编号。源码与安装产物的导入证据分别记录。
该用例使用可精确表示的小整数构造 all-reduce 参考，但不代表任意浮点归约都能逐位相同。

## 3. 补齐四个算子的数值和竞态覆盖

已有主机测试 `test_hccl_check_macros.py` / `test_hccl_collective_sync_switch.py` 不执行真实
集合通信。修改同步策略时，还需要为 `all_reduce`、`all_gather`、`broadcast`、`reduce`
分别提供设备用例；目前没有可直接照跑的 `test_hccl_ops_on_device.py`。

- 用 rank 和迭代号构造独立参考，不能拿 full 的输出作为唯一正确答案。
- gather 按 rank 顺序拼接；broadcast 检查所有 rank 的 root 数据；reduce 只按当前公开
  契约检查 root 输出。当前 `hccl_reduce_op.cc` 没有承诺非 root 输出清零，不能移植 NCCL
  测试的额外断言。HCCL 没有本目录的 reduce-scatter 实现，也不能按名称假设存在。
- 整数或可证明精确可表示的简单用例可要求精确相等；一般浮点归约预先定义 dtype、规模
  对应的绝对/相对误差预算，检查 finite，并解释容差来源。不要强制所有误差都等于零。
- 输入由设备上的前序计算生成，通信结果立即被后续计算消费；避免每轮读回造成隐式同步。
  跨多种大小重复至少 200 轮，并独立重跑，最后同步后对所有轮次检查。
- 全程使用 `forbid_backend_fallbacks()` 并核对实际设备；全部 rank 的退出、完成数量和
  日志齐全后才形成结论。超时、崩溃、跳过和少跑不能记为通过。

初始化的负向测试要隔离运行，覆盖缺少 rank/rootinfo、无效 rank、写入失败和 peer 超时。
要求整个任务非零退出，不允许按单卡继续；不要在数值任务中注入错误污染其它 rank。

## 4. 证明同步策略与收益

数值相同并不能证明同步调用已经改变。用 CANN profiling 的实际调用或 timeline 确认
full/stream-order 的同步差异，再在相同输入、预热与同步口径下比较端到端耗时和波动。
省去同步不保证通信与计算一定重叠；同一流上的操作仍顺序执行，不能把重叠当必要判据。

broadcast root 分支目前另有同步 `aclrtMemcpy` 和 `aclrtSynchronizeDevice()`，不由
begin/end 的开关控制。若改这段需单独证明数据生命周期、时序与数值，不凭源码阅读宣布
同步冗余。结果记录到对应 GitHub issue/PR，注明未覆盖的硬件、dtype、反向和多机范围。

# ACL 私有 Generator：状态隔离与算子接入提案

[研究提案](index.md) · [English](acl-private-generator.en.md)

状态：**提案，未实现**。维护者审查入口为
[#18162](https://github.com/Jittor/jittor/issues/18162)。本文不宣布 ACL 私有
Generator、seeded sampling 或 offset/discard 已受支持；进入实现前需要原生运行时和
compat 维护者确认状态表示、算法及消耗单位。

## 问题与范围

默认设备 RNG 与私有 Generator 是两个合同。默认流的 seed、重放和全局状态恢复成功，
不证明 `tensor.exponential_(generator=g)` 使用了 `g` 的独立状态。私有对象能创建、
`manual_seed` 能返回，也不证明之后的随机算子消费了该对象。

已有合法的无请求 seed 推理路径；不能把此缺口描述成所有 Engine 或随机数均不可用。
真正需要独立状态的调用方包括按请求设置 seed 的采样、交错请求，以及某些 warmup 和
discard 路径。各条路径要单独验收，不得用 greedy 输出代替私有 RNG 验收。

### 已核对的源码边界

以下源码结论固定于 `f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb`，不代表对该提交
做过设备复验。历史真机记录、命令、输出和对应 Core 身份写在关联 PR 中。

1. [compat Generator](https://github.com/Jittor/jittor/blob/f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb/compat/torch/generator.py#L19)
   只发布 CPU/CUDA 私有对象语义，未发布 `get_offset` / `set_offset`。
2. [工厂的 owner 检查](https://github.com/Jittor/jittor/blob/f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb/compat/torch/installers/factories.py#L761)
   将 backend 2 归为 `unsupported`；显式随机调用会在分配和发射之前被拒绝。
3. [RandomOp](https://github.com/Jittor/jittor/blob/f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb/src/ops/composite/random_op.cc#L39)
   虽保存传入的 Generator state，
   [ACL 发射器](https://github.com/Jittor/jittor/blob/f01650afa2c83d4a4fb49f1c6bee78c7a603e4cb/backends/acl/src/acl_op_exec.cc#L714)
   却从 `reserve_acl_random(device, numel)` 获取默认流的 seed/offset，没有消费 `op.gen`。

因此仅删除 compat 拒绝检查，或把 ACL 字符串改成 CUDA，都不能实现状态隔离。
也不能由默认 ACL 发射器按元素推进的行为，推断所有 CANN 分布的 counter 单位。

## 最小实现边界

| 层 | 需要审查的能力 |
| --- | --- |
| 原生运行时 | 明确 backend/device 的私有 owner；独立、可保活的状态；验证后原子恢复；每次 draw 的不可变 reservation |
| ACL 随机发射 | 显式 owner 使用该 owner 的 seed/counter；只有 `generator=None` 使用默认流；不支持的分布在消费前拒绝 |
| compat | 通过正式设备 resolver 接入 backend 2；校验 owner 与输出 ordinal；公开 seed/state 及经定义的 offset 接口 |
| 下游 Adapter | 能力/版本准入、请求绑定及真实消费链验收；不读取私有 C++ ABI，不伪造 Generator，不交换全局流冒充私有状态 |

现有 CPU/CUDA 算法、状态格式和放置检查须保持。原生状态标签和发射接入需要匹配的
Core；不能只更新 Python 包后继续使用来源或 ABI 不匹配的二进制。

### 状态与惰性执行

- 两个 owner 的交错 draw、默认流 draw 和其它请求不得改变彼此的序列。
- seed 重置只影响指定 owner。状态帧必须识别版本、算法、backend、设备和当前位置；
  错误帧、跨设备恢复和溢出在变更前拒绝。
- draw 绑定不可变 seed/起点/跨度及所需生命周期；删除 Python wrapper、之后 reseed、
  完成顺序变化或同一 op 重计算不能改变已发布 draw 或重复消费状态。
- 验证失败不消费状态；已发布后发射失败的状态处理必须有明确合同，不能静默重用区间。

### offset 不是任意 Python 计数器

必须先定义算法、offset 单位和各分布/dtype/shape 的消费规则。vLLM 某些路径的
`set_offset(get_offset() - 4)` 是调用方约定，不证明 ACL 算子消费相同的单位或跨度。
不能用普通指数与 SIMT 算子的推进数互相替换，也不能只加两个方法就宣布 discard 正确。

如果所选算法不符合该假设，应单独审查使用公开完整状态快照的请求级恢复：绑定请求、
owner 和采样步，拒绝过期或有后续未处理 draw 的快照。CUDA 原有分支不变。
图捕获、异步/投机采样和跨进程共享不隐含在首版支持范围中。

## 分层验收

N0–N5 与 A0–A1 是待实现的验收项目，不是已通过结果。

- **N0**：真实 ACL owner、backend/ordinal、伪句柄及跨设备拒绝。
- **N1**：同 seed 重放；多 owner 与默认流交错隔离。
- **N2**：seed/reset/initial_seed 以及高位 seed 往返。
- **N3**：选定算法的参考向量、边界长度、拆分 draw、空 draw、溢出和准确消费跨度。
- **N4**：state/offset 恢复、非法恢复原子性、惰性 op 保活、reseed 后已发布 draw 不变。
- **N5**：真正的 `exponential_(generator=g)` 保持原地对象和存储，并在 ACL 消费指定 owner，零回退。
- **A0**：真正需要私有 Generator 的原始 warmup/请求调用链与自然退出。
- **A1**：原始 discard 的状态恢复、多请求隔离及过期快照拒绝。

CPU 检查、导入成功、默认 RNG 或同一随机算子的一次执行，不代替上述验收。
不要求 CPU/CUDA/ACL 浮点输出逐 bit 相同。

## 手动最小复现

[复现工具](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/probes/acl/private_generator_repro.py)
只比较默认流与显式 Generator 的小规模调用，并记录 offset 接口是否存在。
须先由操作者完成设备准入和独立缓存配置；工具不代替资源许可，不构建 Core。

```bash
# 使用与所选框架匹配的已构建 ACL Core，并先完成设备/资源准入。
JT_BACKEND=acl JITTOR_NO_BUILD=1 python tools/probes/acl/private_generator_repro.py \
  --run --ordinal "$ACL_USER_ORDINAL"
```

调用使用逻辑 `cuda:<user ordinal>` API，必须由 Jittor-owned Torch 和真实 backend 2
证明 ACL 归属；该拼写不代表 CUDA 硬件。报告包含实际接口来源、原始错误和回退计数。
非零退出保留失败，不把捕获异常或默认流通过当作私有 Generator PASS。

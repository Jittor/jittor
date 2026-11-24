# ACL 私有 Generator：状态隔离与算子接入

[研究提案](index.md) · [English](acl-private-generator.en.md)

状态：**首版实现，范围限于下述合同**。审查入口为
[#18162](https://github.com/Jittor/jittor/issues/18162)。实际硬件结果、匹配 Core 身份和
剩余验收缺口记录在实现 PR 中。本文不宣布下游 seeded/discard 工作流通过。

## 支持的随机流

`jt.Generator("acl:N")` 是 ACL 用户设备 ordinal `N` 上的原生私有 owner。
激活 Jittor Torch 前端后，`torch.Generator("npu:N")` 和原生 `acl:N` 拼写都选择该 owner。
逻辑 `cuda:N` 通过现有前端 resolver 选择后端；只有 ACL 运行时才映射为 ACL owner。
owner 与输出的 backend/ordinal 必须一致。必须重建匹配的 Core；只更新 Python
不能让旧二进制获得此原生接口。

首版原生分布是稠密 **float32 uniform**，通过 AscendC Philox4x32-10 内核在 NPU 上生成。
不使用主机随机数，也不借用进程默认 CANN RNG。既有 compat 逆 CDF 实现让
`exponential_(generator=g)` 通过指定 owner 消费这条 uniform 流。
未支持的原生分布/dtype 在 reservation 前拒绝；这不代表所有 Torch 随机工厂或所有
exponential 输出 dtype 已通过。`generator=None` 保持既有 CANN 实现与状态格式。
CPU/CUDA 算法和状态帧保持原样。

## 精确的 seed、位置与消耗合同

- seed 是 uint64 位模式，通过原生 signed-int64 API 传递。
- 一个位置代表一个 **原始 32 位随机字**，范围 `[0, 2**63 - 1]`，不代表 draw 次数、
  CANN/SIMT offset 或 CUDA launch offset。
- 对位置 `p`，counter 为 `(low32(p//4), high32(p//4), 0, 0)`，key 为
  `(low32(seed), high32(seed))`，选取 lane `p % 4`。
- float32 uniform 为 `(word >> 8) * 2**-24`，精确落在 `[0, 1)`。
  每个输出元素预留一个字；空 draw 消耗零；拆分 shape 后拼接的序列不变。
  不承诺与 PyTorch、torch-npu 或 CANN 分布逐 bit 相同。
- `manual_seed` 将位置重置为零。`get_offset`/`set_offset` 只在 ACL 公开这个 raw32
  位置。非法、负值和溢出位置在改变状态前拒绝。本次 CPU/CUDA offset 接口仍明确拒绝。

算法依据原始 [Random123 论文](https://thesalmons.org/john/random123/papers/random123sc11.pdf)。
测试包含公开的零 key/零 counter
[Philox 已知答案向量](https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors)，
以及非零 seed 和高位位置的独立整数参考。

## 状态与惰性执行

原生 owner 使用 mutex 保护 seed 和位置。状态帧包含版本 `JITTOR_ACL_RNG_1`、算法
`philox4x32-10-raw32-v1`、backend、用户 ordinal、seed 与位置。恢复前验证全部字段和
多余字节，验证完成后才修改状态；其它算法、后端或设备的帧拒绝恢复。
compat 通过 CPU ByteTensor 传递状态帧，不自行解释字段。

RandomOp 在构造时原子预留并捕获不可变 seed/base/span。之后 reseed/restore、反序执行、
删除 Python wrapper 或重计算，均不改变已发布 draw，也不重复消费跨度。
不同 owner 与默认流互不推进。验证失败不消费状态；draw 发布后的编译/发射失败不会
静默回退位置或重用区间，而是让操作失败。私有 draw 不允许回退 CPU 或默认设备 RNG。

首版内核使用一个逻辑 block，通过具有显式 scalar/DMA 同步的 UB-to-GM copy 写入，
支持尾部不足一个 tile 的情况，不猜测物理 core 数量。需要已安装的 CANN AscendC
编译器与实际 SoC 的 platform 配置。这是正确性实现，不声明吞吐性能、异步或图捕获能力。

## 验收边界

`tests/backends/acl/test_acl_private_generator.py` 覆盖算法参考、lane/tile 边界、
高位位置、reservation、owner 生命周期和原子拒绝。
`compat/tests/torch/test_acl_private_generator_compat.py` 覆盖真实 owner 路由、隔离、
state/offset 重放、原地 exponential 消费及非法输入。两者均要求实际准入的 ACL 设备与
零回退；CPU 检查、导入成功及默认 RNG 检查不能代替真机验收。

原提案 N0–N5 对应这些原生/compat 语义。**A0**（原始 warmup/请求使用与自然退出）及
**A1**（原始 discard 恢复、请求绑定及过期快照拒绝）仍需下游独立验收。
特别是 `set_offset(get_offset() - 4)` 只回退四个 raw32 字，不能撤销包含任意词表元素数的 draw。
接口存在不代表 vLLM 该假设成立。图捕获、投机采样及跨进程状态共享不在首版合同中；
下游适配必须使用经过审查的公开状态接口，并保留已有 CUDA 分支。

## 手动复现

[复现工具](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/probes/acl/private_generator_repro.py)
记录默认流/私有流的小规模调用、真实接口来源、原始错误与回退计数。
须先完成资源准入并准备独立、匹配的 Core；工具不授予设备许可，也不构建 Core。

```bash
JT_BACKEND=acl JITTOR_NO_BUILD=1 python tools/probes/acl/private_generator_repro.py \
  --run --ordinal "$ACL_USER_ORDINAL"
```

逻辑 `cuda:<user ordinal>` 输入不代表 CUDA 硬件；实际 backend 2、Tensor 放置与 SDK
映射证明物理 ACL 设备归属。捕获异常、默认流通过或单独退出零均不能代表私有 Generator 通过。

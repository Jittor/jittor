# ACL 单算子性能对齐 torch_npu

> 研究提案，未实现。描述问题、关键认知、设计原则与分阶段验收，不描述当前行为。
> 进入实现前需单独做架构评审，并在真机(910B)上验证。

## 问题陈述

在 Ascend 910B 上，jittor 的 ACL 后端在**训练类 / 可融合 / 可图重放**的工作负载上已能与
torch_npu 打平甚至反超(训练四项几何均值 ≤ 1.0)。但在**单算子 device-bound**的工作负载上仍
落后(见 `docs/guides/known-limitations.md` 的 KI-BACKEND-017)。本提案要把"落后"拆成可定位、
可逐项消除的具体原因，并建立"**凡是能从 torch_npu 迁移过来的更优调用方式，就迁移进来**"的
定点流程。

## 关键认知:底层是同一批 CANN 算子

910B 上真正执行的 kernel 是华为 CANN 的 aclnn/aclop 算子(`libascendcl`)。jittor-ACL 与
torch_npu **都调用这同一批算子**。因此同一个 `aclnnXxx` 调用的 device 时间对两个框架是一样的
——**不存在"torch_npu 有一个更快的秘密 kernel 而我们没用"**。性能差异不来自 kernel 本身，
而来自**怎么调**。可归为五个轴:

1. **算子选择 / 融合**:torch_npu 常调一个专用融合算子;jittor 可能调通用算子，或拆成一串
   aclnn 调用的 composite。
2. **数据格式**:CANN 算子内部偏好 NZ/fractal 格式。保持 ND 会让算子内部自己转置/pad，多出
   `TransData`/`Dilation` 这类额外 device op。
3. **精度模式**:CANN 的 matmul 可走 HF32(类 TF32)。同一算子在 HF32 下远快于真 FP32——差的
   是一个精度 flag，不是 kernel。
4. **host 发射流水**:torch_npu 用 `TASK_QUEUE_ENABLE` host 异步排队发射，device 不挨饿;串行
   eager 发射会 host-bound，device 空等。
5. **算了不需要的结果**:例如给"永不需要梯度"的输入算 grad_input。

## 已验证的证据(这套拆法是对的)

- **算子选择(轴 1)**:`qwen3_prefill` 曾慢 3.2x。根因是 SDPA 因 bool-mask 的 dtype 守卫拒绝了
  融合的 `scaled_dot_product_attention_acl`，掉到不融合的 composite(> 60% device 时间)。接上
  融合算子后 0.163 → 0.056 s/step，**反超 torch_npu ~1.6x**。—— 这就是"torch_npu 用哪个算子
  我们就用哪个"的直接胜利。
- **算了不需要的结果 + 格式(轴 5/2)**:`vit_b16_train` 的 `Dilation` + `Conv2DBackpropInput`
  占 ~90% device 时间，其中 patch-embed conv 的 grad_input(图像不需要梯度)是纯浪费;接入
  `Var::is_result_needed()` 后按需裁剪。
- **发射流水(轴 4)**:ACL 异步执行 worker(建图与设备执行重叠)把几何均值从 0.917 降到 0.880。

## 设计原则

- **定点迁移，不整套照搬**:torch_npu 是大团队多年手工维护的"一 torch op → 指定 aclnn op(s)
  + 格式管理"适配层;jittor 走的是自己的 IR + 融合编译范式,正是融合让 jittor 在训练类反超。
  整套替换会丢掉训练胜场。因此只对**落后的 device-bound 算子**，逐个把 torch_npu 更优的
  **选择 / 格式 / 精度**迁移过来。
- **先测后改**:若我们已在调同一个 aclnn 算子，"换算子"是空操作——真实差异在格式/精度/发射。
  任何一处改动都必须先由 op-timeline 对拍确认它落在哪一轴。
- **不回归训练胜场与正确性**:每次改动都要 3-way(CPU/CUDA/NPU)正确性复核，并确认训练几何
  均值不退。
- **CANN kernel 真实慢的情况如实上报**:若对齐选择/格式/精度后，某算子的 device 时间仍慢于
  torch_npu 调用的同名算子,那才是 CANN kernel 层面的差距,向上游报告,不在本仓库强行绕。

## 分阶段方法

### 阶段一:建立 op-timeline 对拍

对每个 device-bound 落后层，用 aclprof/msprof(peer 已用 ctypes `aclprofInit/Start/Stop/
Finalize` + 离线 `msprof --export` 驱动)同时采 jittor 与 torch_npu 的 op timeline，逐算子导出:

- 算子名(是否同一个 `aclnnXxx`;jittor 是否 composite 成多个)
- 输入/输出格式(NZ vs ND)
- dtype 与精度模式(是否 HF32)
- workspace、tiling / 关键 attr
- 单算子 device 时间、该层 host 时间占比

### 阶段二:按轴归因

对每个落后层，判断差距落在轴 1–5 的哪一个(或多个)。

### 阶段三:定点修复 + 真机验证

- 轴 1 → 把该 torch op 接到 torch_npu 用的那个融合 aclnn 算子(如 SDPA 的做法)。
- 轴 2 → 在该路径上对齐设备偏好格式(NZ),减少 `TransData`。
- 轴 3 → 对齐 HF32 等精度默认(需同时给出数值影响评估与开关)。
- 轴 4 → 复用已有的异步 worker / 图重放,减少 host 发射占比。
- 轴 5 → 用 `is_result_needed` 等信号裁剪不需要的输出。
- 每项:3-way 正确性 + before/after device 时间 + 训练几何均值不退。

## 当前候选目标

以最近一次记录的 14 工作负载快照为起点(commit `9e37dc005` 一线,**在 SDPA / conv / 异步 worker
修复之前**，数字已过时，需**重新基线**后再认领):曾落后的有 `sd15_vae_decode`、`resnet50_infer`
(b64)、`bert_base_train`、`sd15_sample`、`ddpm_unet_train`、`vit_b16_train` 等;`qwen3_decode` 的
2.23x 当时判为 bf16 argmax 的良性差异。重新基线后,按"落后幅度 × 出现频率"排序逐个对拍。

## 验收标准

- 每个认领的 device-bound 算子:对齐选择/格式/精度后,device 时间进入 torch_npu 的合理邻域,
  **或**把残差根因钉到 CANN kernel 层(并上报上游)。
- 训练几何均值相对对齐前**不退**;所有改动 3-way 正确性通过。
- 结论写回 `docs/guides/known-limitations.md` 的 KI-BACKEND-017,过期条目删除。

## 分工与状态

- **状态**:研究提案,未实现。
- **分工**:核心/派发/格式/精度接线与 CUDA 基线由主仓库侧实现并自测;NPU 上的 aclprof 采集、
  对拍与端到端验证由 NPU 侧(910B 真机)负责。参见 `docs/development/npu-validation-templates.md`
  与 `docs/development/acl-backend-contracts.md`。
- **复查条件**:每认领一个算子前做架构评审;阶段三每项改动需真机验证记录。

# verl PPO NPU / CUDA 历史验证记录

- Status: 历史限定范围的证据摘要；不是本次发布提交的完整 L1-L5 验收。
- Date: 2026-10-24 文档收口；以下测量日期为 2026-09-29、2026-10-21。
- Baseline: NPU 原记录只给出 `40c73324edf7` cache-segment 标识，不补写为完整 Git SHA；
  CUDA V55 为 `b2fa61449531341caabf418e639fd41e5e4ff0c7` 加当时保留的工作树修复。
  本次发布集成基线 `ca56aa802c2acf9a22bca32d1bee3276381ed203` 不继承旧测量的授级。
- Owner: Jittor compatibility maintainers
- Review when: 模型/权重、训练协议、verl、Transformers、TensorDict、源码/安装来源、
  CUDA/ACL/HCCL、DDP/FSDP2、保存恢复、device/RNG facade，或 adapter 依赖的
  `nested_tensor_type`、`nested_tensor_rebuild`、`namespace_owner`、`owned_runtime_hook`、
  `transaction`、`module_patcher` 公开入口改变。

本文件仅保留两项 PPO 历史结论及其边界。完整原记录、失败尝试和原始产物仍在原开发
工作树/仓外状态目录保存，未删除或改为通过；它们不随此摘要进入发行物。本文没有
为本次发布重新生成全级验收结果；**L4 速度本轮未严格验证**。

## 2026-09-29：PPO 在 910B3 上的 L1–L5 受限闭环

本节只签署明确的 PPO 验收合同，不把它扩大为整个 verl、Ray rollout server 或任意配置。
环境为 Ascend 910B3、CANN 9.0.0、官方 Torch/torch_npu 2.7.1 与当前 Jittor 源码
`40c73324edf7` cache segment；实模固定为本地锁定的 `Qwen/Qwen3.5-0.8B` text actor，
参数量 `752,393,024`。官方与候选始终使用独立 Python 解释器和相同初值，候选禁止 backend
fallback。物理卡号只用于分配；报告内的逻辑设备均为 `npu:0`/`npu:1`。

| 等级 | 结论 | 本轮真实 NPU 证据 |
| --- | --- | --- |
| L1 import/forward | 通过 | 当前 Jittor frontend 导入 verl/Transformers，Qwen3.5-0.8B 参数、输入和输出实际驻留 NPU；独立官方 torch_npu 的 seq 8/32/65 functional 回归覆盖输出、state 与 query/key/value/g/beta 五路梯度 |
| L2 单步正确性 | 通过 | tiny PPO actor + learned critic 对拍 243 个张量：actor 参数梯度最大差 `8.94e-8`、critic `2.98e-7`、输入梯度最大差 `3.725e-8`、更新后权重最大差 `3.725e-9`；候选 fallback `0` |
| L3 多步训练闭环 | 通过 | 同一受控 PPO/GAE actor + learned critic 连续 3 步，全部前向、全参数梯度与更新轨迹通过；真实 verl `TrainingWorker` 双 rank actor 150 tensors/rank、critic 170 tensors/rank，各连续 3 步通过 |
| L4 实模性能与数值 | 通过（单卡、双卡） | Qwen3.5-0.8B、FP32/eager、batch 1/rank、seq 32、`lr=5e-10`、`eps=1e-5`、2 warmup + 3 timed steps，每种拓扑三轮 Torch/Jittor 交错运行；单卡比值 `1.93243×`、双卡比值 `1.94002×`，均 `≤2.0×`，固定正确性门禁全部通过 |
| L5 设备/kernel/通信 | 通过 | 单卡真实 NPU 驻留和 fallback `0`；双卡 FSDP2、真实 HCCL、不同 rank 输入、梯度同步、全局裁剪、连续 SGD 与负控 `no_sync` 均通过；日志观察到 HCCL 初始化和 `HcclAllGatherOp` |

上述 L3 的受控工作流和 TrainingWorker 使用固定 response/reward，L4 是真实 0.8B actor
update；仍未包含 Ray、在线生成、独立 reward model 服务、actor/critic 联合调度或 vLLM
weight transfer。因此可以写为“PPO 在上述 NPU 验收合同内 L1–L5 完成”，不能写为
“verl 全库或任意 PPO 配置 L1–L5 完成”。BF16 和若干更大学习率的失败尝试继续保留，
没有并入通过合同或删除。

仓外历史单卡 / 双卡 L4 汇总 JSON 的 SHA256（未版本化）：
`9284d79e5aa8d276d22e853efa46097236833f6d7dbda3d6622d80d190028404` /
`bfa634610263a4f561cfde76d5ee4756dc20883ceee6d4a0b5d2d330f778df19`。

## 2026-10-21 V55：单卡 FP32 完整 PPO 的限定数值范围

状态：**原始完整 PPO 的所列数值范围通过；严格元数据仍未通过，不能记为严格 L2
或累计 L1–L5 全通过**。Owner：Torch compatibility maintainers。模型为 Qwen3.5-0.8B，
原始 verl v0.9.1（`1876b06d0a3e4e71e06230be10af14492ca8a75b`），单张 A800 80GB PCIe，
FP32/strict、batch=2、prompt 上限 64、response 上限 8、一个完整 PPO 训练步。
本节实测基线为 `b2fa61449531341caabf418e639fd41e5e4ff0c7`，不移作后续源码的精度结论。
模型、训练配置、采样/输入重放、compat、源码身份或 CUDA 依赖变化后复查。

数值验收使用独立真实 PyTorch CUDA（2.8.0+cu128）为基线，不以 NumPy 充当模型 oracle。
原始输入 producer、PyTorch、Jittor 三轮原始 fit 与实际子进程等待均为退出 0：

- 初始 Actor/Critic 参数与缓冲区合计 **6,823,892,100 字节**全部逐字节比较，差异 0；
  两角色、两框架四个数值 scope 全部排空。初始 Actor 严格元数据一致，Critic 不一致。
- 全部 **948 组参数梯度 / 1,705,972,865 元素**完整比较，未抽样；
  按 `abs(candidate-reference) <= 1e-4 + 1e-4*abs(reference)`，失败组/元素均为 **0**。
  梯度最大绝对误差 **1.850128173828125e-4**，不是声称所有绝对误差小于 1e-4。
  以全网独立 oracle 梯度最大值归一化，最大误差比为 **3.5489837320439717e-6**；
  Actor 单独归一化为 **1.9537204253152646e-5**，不隐去角色差异。
- 实际 loss/输入边界 **2111 元素**，同一容差失败 0，最大绝对误差 **4.57763671875e-5**。
  保留更严格 `atol=rtol=1e-5` 的诊断失败：梯度 5176 元素、loss/输入 11 元素。
- Jittor Actor/Critic 真实数值角色退出回执均为 fallback=0；三轮分别包含外层启动器的
  **9/9/8 个已记录 PID/start tick 全部排空**。单卡结果不外推多卡、NPU 或全部 verl。

仍开放：**874 组梯度的 `gradient_local.storage_offset`**、3 个输入微批元数据 scope、
9 个配置字段记录及初始 Critic 元数据。旧严格失败原样保留，没有归一化设备、类型或
存储偏移字段。本轮不是全 AdamW 状态轨迹、完整更新后参数对拍或任意中间张量 VJP 验收；
不声称全工作者隐私或所有辅助路径零回退。完整状态观察及对拍开销使本轮不具备正常
训练性能验收资格，未提供速度比或 L4 达标结论。

复现入口（仓外、未版本化）：`verl_d0_full_PPO_runtime51_controller_MAIN_v55.py`，
配套 39 项哈希闭包；安全数值汇总为 `numeric-complete.json`。原始 stdout/stderr 为
DEVNULL，原始张量、UID、输入与随机状态仅 RAM，不落盘。

- runtime / full 清单 SHA256：
  `1ca8fb8ec77ded1a35ab3d66c94bddd438827ab76699ae1911fe047916896653` /
  `e9890f173a162e8cae79d0ef64b84e7e8cb701122dc3ffab0e037b2eaaeb22c8`；
  1114 runtime / 813 core / 8 generators，完整树 3247 文件，运行前后逐项一致。
- 实际 core / 新 stamp：
  `8e95e0808856cf70fb31f60f4b8c4d6b9213554afe02c958595680efaeb17714` /
  `3bbaed3058fe4d11f99eab1f3fc14f30b696e915a69d38508f72728e0dd5042f`。
- V55 controller / 工具闭包清单 / 数值汇总 SHA256：
  `3235cb224aa0022883c4bd4649079edc9edb91eaa245d6550a1d7f1b2da3105a` /
  `9d0916c4c2ef30f50b464bba585c4d3571625d885150b29436a4fde444454410` /
  `6cb2180c3e0e508cea3e351be840d914e0a9da9eba0d3fba2b003b2ed950e557`。

当前开放合同见 [KI-COMPAT-016](../../agent/manuals/known-issues.md#ki-compat-016-complete-cuda-ray-ppo-correctness-remains-unverified)。

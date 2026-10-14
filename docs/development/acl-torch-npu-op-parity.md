# ACL 对齐 torch_npu 单算子性能 — 迁移 TODO

跟踪"把能从 torch_npu 迁移过来的更优调用方式都迁进来"这件事的工作清单。对应
`agent/manuals/known-issues.md` 的 KI-BACKEND-017。

**背景(一句话)**:jittor-ACL 与 torch_npu 底层跑的是同一批 CANN aclnn 算子,单算子慢**不是
kernel 更慢**,而是**怎么调**不同——五轴:① 算子选择/融合 ② 格式(NZ vs ND)③ 精度(HF32 vs
FP32)④ host 发射流水 ⑤ 算了不需要的结果。打法是**定点迁移、先测后改**,不整套照搬(会丢掉
jittor 融合在训练类的胜场)。详细论证见下方"原则"。

## 阶段一:对拍基建

- [ ] 在**当前分支**(SDPA / conv grad / 异步 worker 修复**之后**)重新基线 14 工作负载,得到最新
      jittor vs torch_npu 比值(旧快照 commit `9e37dc005` 一线已过时)。
- [ ] 建立 aclprof / msprof op-timeline 采集脚本(peer 已用 ctypes `aclprofInit/Start/Stop/
      Finalize` + 离线 `msprof --export` 驱动),对同一层同时采 jittor 与 torch_npu。
- [ ] 每个算子导出字段:算子名(是否同一 `aclnnXxx` / jittor 是否 composite)、输入输出格式
      (NZ vs ND)、dtype 与是否 HF32、workspace/tiling、单算子 device 时间、该层 host 占比。

## 阶段二:逐个 device-bound 落后层归因

对每个落后层,勾选 = 已对拍 + 已判定差距落在哪一轴。候选(重新基线后据实增删):

- [ ] `vit_b16_train`(复测 conv grad_input 裁剪后的残差)
- [ ] `sd15_vae_decode`
- [ ] `sd15_sample`
- [ ] `resnet50_infer`(b64)
- [ ] `ddpm_unet_train`
- [ ] `bert_base_train`
- [ ] `qwen3_decode`(旧判为 bf16 argmax 良性差异,复核确认)

## 阶段三:按轴定点迁移(每个落后算子)

- [ ] **轴1 算子选择/融合**:把落后的 torch op 接到 torch_npu 用的那个融合 aclnn 算子
      (参照 SDPA bool-mask → `scaled_dot_product_attention_acl` 的做法)。
- [ ] **轴2 格式**:在落后路径上对齐设备偏好格式(NZ),减少 `TransData`/`Dilation`。
- [ ] **轴3 精度**:对齐 HF32 等精度默认,带开关 + 数值影响评估。
- [ ] **轴4 发射**:复用已有异步 worker / 图重放,降低 host 发射占比。
- [ ] **轴5 裁剪**:用 `Var::is_result_needed()` 等信号裁掉不需要的输出(如不需梯度的 grad_input)。
- [ ] 每项改动:3-way(CPU/CUDA/NPU)正确性 + before/after device 时间 + 训练几何均值不退。

## 收尾

- [ ] 对齐选择/格式/精度后仍慢的,把残差根因钉到 CANN kernel 层,**如实上报上游**,不在本仓库强行绕。
- [ ] 结论写回 KI-BACKEND-017,删除过期条目。

## 原则(执行时别踩)

- **定点迁移,不整套照搬**:torch_npu 是手工维护的"一 torch op → 指定 aclnn op + 格式管理"适配层;
  jittor 走自己的 IR + 融合范式,正是融合让训练类反超。整套替换会丢掉训练胜场。只对落后的
  device-bound 算子逐个迁移 torch_npu 更优的选择/格式/精度。
- **先测后改**:若已在调同一个 aclnn 算子,"换算子"是空操作,真实差异在格式/精度/发射——必须先
  由 op-timeline 对拍确认落在哪一轴,再动。
- **已验证证据**:SDPA bool-mask 接融合算子(3.2x 慢 → 1.6x 快,轴1);vit conv grad_input 裁剪
  (轴5);异步 worker(几何均值 0.917 → 0.880,轴4)。

## 分工

- 核心 / 派发 / 格式 / 精度接线 + CUDA 基线:主仓库侧实现并自测。
- aclprof 采集、对拍、910B 真机端到端验证:NPU 侧。参见 `docs/development/npu-validation-templates.md`
  与 `docs/development/acl-backend-contracts.md`。

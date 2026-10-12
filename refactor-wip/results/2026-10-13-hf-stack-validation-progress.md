# HF stack 组合验证进展

- 状态：部分通过，按计划继续扩展
- 日期：2026-10-13（Asia/Shanghai）
- 维护者：cgq / Jittor Torch 兼容层维护者
- 实验启动源码基线：`fe4e738338de5e275471da125816968af7d945fc`
- 本轮修复和结果文档提交：`2a5b0123a`
- 目标远端：`origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，已确认是当前 `HEAD` 的祖先
- 工作区：dirty；本报告不把其他优化器、看板和开发文档改动归入本轮实验提交
- 环境脚本：`env_jittor.sh` SHA-256 `5c31abdf9f8921e66e331dc5b0c0ad1bfe44ce1ce80abc78c0cd9d20dac0daca`；`env_pytorch.sh` SHA-256 `a664bfed00a10840a1134f7e6f4778d9153022fdc4c959bd87f7ca4afbd34d33`
- 本轮 dirty diff SHA-256：`16eb8bf1c0c4f3b8f601e459a647f3e43bda2a3854146a093e5fd659fd226e76`（记录时点）
- 环境：Python 3.11.3；Transformers 4.56.2；PEFT 0.17.1；Tokenizers 0.22.1；Safetensors 0.6.2；Accelerate 1.10.1；Jittor 1.3.11.0；独立 PyTorch 2.6.0+cu124；A800 80GB CUDA

实验原始日志保存在未版本化目录 `/home/cgq/jittor-lab/_state/hf-stack-validation/unit-20261013/logs/`。每个对拍都由独立 PyTorch oracle 先生成权重和输入，Jittor runner 再加载相同权重；Jittor 侧启用 `backend_fallback="error"`，报告实际 device 与 fallback 计数。性能数字只作为冷启动成本记录，当前不作为速度门禁。

## 已完成

| 计划项 | 实际范围 | 结果和证据 |
| --- | --- | --- |
| `ENV-01/02` | runner 记录 oracle/shim 导入 origin、库版本、实际 CPU/CUDA device、backend 和 fallback | PASS（随下列 runner case 执行；无意外 skip，Jittor fallback 为 0） |
| `TOK-02` | 固定 WordLevel tokenizer、Seq2Seq collator、`BatchEncoding.to(device)`、embedding 交接；PyTorch/Jittor CPU 与 CUDA | PASS；`tokenizer_handoff_{pytorch,jittor}_{cpu,cuda}.log`；这是固定 fixture，不是 32k/128k 词表性能验收 |
| `SER-01/SER-05` | 有限 dtype/metadata owner case、PyTorch↔Jittor 文件互读 fixture、CUDA owner placement、畸形 bytes 拒绝 | PASS（声明范围内）；`safetensors_owner_cuda_final.log`、`safetensors_validation_after.log`、`safetensors-cross-runtime/`；尚未覆盖完整模型大分片和恶意 offsets 全矩阵 |
| `PEF-01` | tiny Llama/Qwen LoRA 单步 forward/backward；trainable/frozen 集合、实际梯度键、CUDA placement 和数值对拍 | PASS（CPU `2 passed`，CUDA `2 passed`）；`peft_parity_cpu_after_owner_fix.log`、`peft_parity_cuda_after_owner_fix.log`；不代表多步 optimizer、adapter merge 或 resume |
| `ACC-05` | 混合 FP32/FP64 uniform/normal、dropout、Python/NumPy RNG 的 fresh-process 连续训练/恢复；CPU 与 CUDA | PASS（CPU `1 passed`，CUDA `2 passed`）；`accelerate_rng_mixed_cpu.log`、`accelerate_rng_mixed_cuda.log`；仅证明当前 fixture 的恢复协议，不外推所有 cuRAND 历史 |
| `ACC-01` / placement 子项 | CPU 创建的 model/optimizer 经 `Accelerator.prepare` 放到 CUDA；CPU checkpoint 恢复到 CUDA model，保持参数 identity 和 optimizer 引用 | PASS；`accelerate_prepare_contract.log`、`module_load_state_dict_cuda_placement.log`、`module_load_state_dict_cpu.log` |
| `INF-01` | Transformers tiny GPT-2、Llama、BERT、ViT、T5、Whisper，独立 PyTorch 对拍全部参数梯度；FP32 CPU/CUDA | PASS：CPU `6/6`（52.75 s），CUDA `6/6`（353.63 s）；`transformers_parity_cpu_fixed.log`、`transformers_parity_cuda_fixed.log`。单独 Whisper CPU/CUDA 也各 `1 passed`（`9.23/11.04 s`） |
| `INF-01` harness/parser | marker 前缀 JSON 在同一行尾随 Jittor runtime log 时仍能解析；无 marker 仍失败 | PASS：`compat/tests/torch/test_ecosystem_result_scan.py` 5 cases；修复 `_result_from_stdout` 使用 `JSONDecoder.raw_decode` |
| Transformers freeze contract | Whisper `encoder.embed_positions` 在 `@torch.no_grad()` 的 `parameter.data.normal_()` 后保持冻结；CPU/CUDA 定向回归 | PASS：`module_data_freeze_regression.log` `2 passed`；修复位于 `compat/torch/installers/tensor/method_api.py`，新增 owner `requires_grad` 回写 |

## 当前缺陷与边界

本轮没有把 tiny forward/backward 结果升级为完整库或组合结论。以下场景仍为 `NOT_RUN`：

- `TRN-01～06` 的十步/多步 optimizer、梯度累积、AMP、scheduler 和有效 token loss 轨迹；
- `PEF-02～06` 的多步 LoRA/DoRA/IA3、adapter 切换/merge、safe merge、导出后原生 PyTorch reload；
- `E2E-01～04` 的 Trainer/Accelerate + tokenizer + safetensors + fresh-process resume 真实数据链；
- `DST-01～04`、`E2E-05` 的当前源码双 rank/四 rank DDP、no-sync、尾批、恢复和独立 PyTorch DDP 对拍；
- `FSD-01～08`、`E2E-06` 的 FSDP2 FULL/SHARDED、reshard、LoRA、AMP、真实峰值和通信收益；
- `SER-02～06` 的 Transformers 分片模型 round-trip、完整 tied-storage/stride 错误矩阵、64 MiB～1 GiB 多 shard；
- `TOK-03～06` 的 MLM/SFT packing、workers 2/4、spawn、跨 epoch/rank sampler、词表规模与 soak；
- `INF-02～07` 的 cache/generation/beam/sampling、AMP/SDPA/fused attention、长上下文和多模态真实 processor；
- `PRF-01～05` 的稳定负载吞吐、TTFT/TPOT、step breakdown、显存峰值、strong/weak scaling 与 30 分钟 soak；
- `PORT-01～03` 的 clean wheel、候选库版本矩阵、ROCm/NPU 及多机部署。

旧的 17-model CUDA、PEFT 真实 0.5B、Accelerate 两 rank/FSDP2 FULL 和 safetensors 基础报告只作为 `prior_evidence`，没有覆盖当前 dirty tree 上的上述组合。尤其不能用已有 uniform-only CUDA RNG 记录替代完整混合历史 resume 门禁。

## 后续顺序

先实现并执行 `TRN-01` + `PEF-01` 多步/恢复短轨迹，再进入 `E2E-02` 单卡六库链；随后用 tiny 双卡执行 `DST-01` 和 `FSD-02`，确认数值、collective 顺序、各 rank 恢复和实际峰值后再扩大模型。性能实验必须使用独立 benchmark cache 和稳定设备负载，与 unittest 分离；大模型训练、长 soak 和 70B 仍不自动启动。

# Llama 3.1 70B 真实 BF16 六卡短 SFT 对拍

- 状态：PASS（代表性真实 CUDA 短训练；不是全参数 70B SFT 或 BF16 exact parity）
- 验收日期：2026-09-07（Asia/Shanghai）
- 分支：`feature/cgq_transformers`
- Jittor 提交：`9dced1879c1190645964b1d5f41646a1748a2f94`
- PyTorch oracle 提交：`093a4e4937618ae83abf66ed2a315a666b4db020`
- 共同基线：已推送 `508f816b81bdacba5ebdb72ce82b9bab3181b2ec`
- 维护者：Jittor Transformers compatibility

## 验收结论

在同一份真实 Llama 3.1 70B 权重、tokenizer、SFT 样本顺序、padding、seed、optimizer、
precision 和有效 batch 下，原生 PyTorch 与 Jittor Torch adapter 都在六张真实 A800 CUDA
卡上完成完整 80 层 forward、last-block/final-RMSNorm/lm-head 的 BF16 backward 和 SGD
update。五步 loss 曲线均稳定下降且几乎重合：曲线起止绝对降幅比为 `1.0062765178`，在
`[0.8, 1.25]` 门槛内。逐 step loss 数值相等不是本轮门槛。

step 0 的 11 个可训练张量全部有完整梯度和参数更新；梯度和更新从 FSDP2 shards 重建
后均 shape matched、finite、无缺失。BF16 舍入下 whole-tensor L2 只记录为诊断，不作为
硬 gate。Jittor harness 最终 `comparison.passed=true`、`tensor_evidence_passed=true`。
没有 CPU fallback：Jittor 进程验证 `torch is jittor`、`jt.has_cuda=1`，六个 rank 均以
NCCL `world_size=6` 初始化并在目标卡执行计算。

## 固定配置

| 项目 | 值 |
| --- | --- |
| Transformers / oracle | `4.56.2` / PyTorch `2.6.0+cu124` |
| Jittor runtime | `1.3.11.0` Torch shim |
| Hardware | 6 x NVIDIA A800 80GB PCIe CUDA |
| Physical devices | `0,3,4,5,6,8` |
| Checkpoint | `meta-llama/Llama-3.1-70B@349b2ddb53ce8f2849a6c168a81980ab25258dac` |
| Checkpoint index SHA256 | `2abe0910e23770a30ccf9b1b91804c64831c47f9c98defaa5293aa999433fc2b` |
| Checkpoint bytes | `141107412992` |
| Loaded scope | 723 keys, 80 layers, streamed unit construction/load/immediate FSDP shard |
| Train scope | last decoder block + final RMSNorm + lm_head |
| Trainable / frozen | 11 tensors / `1,906,335,744` params; 712 tensors frozen |
| Dtype / attention | BF16 parameters and forward, eager attention |
| Data | 24 authored instruction/response rows |
| Data SHA256 | `e0d9519a6c1b2c3e66d52cfe4ee28071841855af92f579ad2f96d9571c45e716` |
| Tokenized SHA256 | `3bd376868190c2d3eb674b381621d5ff6e593c4841b8a3c3fe6964ee690edff6` |
| Padding / schedule | right padding; repeated rank-local `sft-000..005` batch |
| Batch / sequence | per-rank 1, effective 6, sequence length 64, 267 supervised tokens/step |
| Seed / optimizer | seed `20260906`; SGD `lr=0.001`, momentum `0`, weight decay `0` |
| Extra branches | no LoRA, autocast, GradScaler, CPU offload or host compute |

## Loss and tensor evidence

| Step | PyTorch loss | Jittor loss |
| ---: | ---: | ---: |
| 0 | 2.9905923207600913 | 2.9917977253595986 |
| 1 | 2.9090225299199424 | 2.909897526105245 |
| 2 | 2.8394821087519326 | 2.8396405776341758 |
| 3 | 2.778156359990438 | 2.776845335960388 |
| 4 | 2.7179112235705056 | 2.717405140399933 |

PyTorch change is `-0.27268109718958566` and Jittor change is `-0.2743925849596658`; the
ratio is `1.0062765178361086`, and both fitted slopes are negative (`-0.0676228364` and
`-0.0681837360`). The plot is the primary compact curve artifact. Full tensor evidence is
written only for step 0; steps 1-4 retain scalar loss/timing/memory records.

All 11 gradient and 11 update records are present, shape matched and finite. Diagnostic maximum
whole-tensor L2 values are `0.0247801` for gradients and `0.2073825` for updates; these reflect
BF16 rounding and are explicitly not acceptance gates. The failed r4 run used the old strict
numeric gate and is retained as fail-first evidence; r5 reran after the full-gradient semantics
fix in `9dced187` and passed the revised gate.

## Performance

Stable steps are steps 1-4, excluding step 0 full-tensor evidence and Jittor cold JIT compilation.

| Metric | PyTorch | Jittor | Jittor / PyTorch |
| --- | ---: | ---: | ---: |
| Mean step time | `38.9064034475 s` | `42.2975912094 s` | `1.08716x` |
| Mean throughput | `6.8626535166 token/s` | `6.3127138445 token/s` | `0.91986x` |
| Framework peak | `27.4968 GiB` | `34.5537 GiB` | `1.25664x` |

The independent memory sampler recorded Jittor peaks of GPU `0:36835 MiB` and GPUs
`3/4/5/6/8:36037 MiB`. The corresponding PyTorch peaks were GPU `0:35131 MiB`,
`3/4/5/6:33461 MiB`, and `8:34997 MiB`. The Jittor step-0 wall time includes complete
gradient evidence collection and is not used in the steady-state table; cold JIT compilation
is also excluded.

## Reproduction

All large files, caches and raw logs are unversioned under `${JITTOR_LAB_ROOT}`. Set
`JITTOR_LAB_ROOT` to the lab root and `LLAMA_CHECKPOINT` to the fixed local snapshot before
running. The launcher itself uses isolated `JITTOR_HOME`, `HOME`, `TMPDIR`, CUDA and Torch-shim
paths under `$JITTOR_LAB_ROOT/_state/transformers_70b_sft/representative/<run>/`.

```bash
export JITTOR_LAB_ROOT=/path/to/jittor-lab
export LLAMA_CHECKPOINT=/path/to/Llama-3.1-70B/snapshots/349b2ddb53ce8f2849a6c168a81980ab25258dac
export JITTOR_REPO_ROOT=/path/to/jittor

# The lab launcher reads the checkpoint from $LLAMA_CHECKPOINT.
$JITTOR_LAB_ROOT/transformers_compat/envs/pytorch/bin/python \
  $JITTOR_LAB_ROOT/transformers_70b_sft/launch_jittor_six_rank.py \
  --state $JITTOR_LAB_ROOT/_state/transformers_70b_sft/representative/jittor-r5 \
  --output $JITTOR_LAB_ROOT/transformers_70b_sft/runs/representative-80layer-5step-jittor-r5 \
  --devices 0,3,4,5,6,8

$JITTOR_LAB_ROOT/transformers_compat/envs/pytorch/bin/python \
  $JITTOR_LAB_ROOT/transformers_70b_sft/plot_loss_curves.py \
  --pytorch $JITTOR_LAB_ROOT/transformers_70b_sft/runs/representative-80layer-5step-pytorch-r2 \
  --jittor $JITTOR_LAB_ROOT/transformers_70b_sft/runs/representative-80layer-5step-jittor-r5 \
  --output $JITTOR_LAB_ROOT/transformers_70b_sft/runs/representative-80layer-5step-jittor-r5/loss-curves.png
```

The actual retained local result is `$JITTOR_LAB_ROOT/transformers_70b_sft/runs/representative-80layer-5step-jittor-r5/result.json`;
the six NCCL prewarm files, rank logs and memory CSV are under the matching
`$JITTOR_LAB_ROOT/_state/transformers_70b_sft/representative/jittor-r5/` directory.
The curve PNG SHA256 is `b7a9ae61e89decff5f0849f34913aee77bf41252c7a1e9a986b3c0247cde1151`.
The post-change harness, launcher, plotting script and dataset SHA256 values are respectively
`2a7cd1b8a745a5954ad90ccb472ecaf9a21b91dc75f73b3cfb9e4f7478d63d52`,
`88f96e6eb841ba9a12888e91cc7aa80e1bec0f059e6f08d912eaab53e345a083`,
`92514183fe23798e1dd1c36d5c3fc9f31f6203aa0bf6745a1bd17dbd54fab70f`, and
`e0d9519a6c1b2c3e66d52cfe4ee28071841855af92f579ad2f96d9571c45e716`.

## Limitations

This is a short, fixed-batch, BF16 training protocol with only the final block, final norm and
lm-head trainable. It establishes real-device execution, complete step-0 tensor presence and a
representative loss trajectory, not full-parameter optimizer parity. Whole-tensor BF16 numeric
differences, longer training, all-parameter updates, other attention implementations, mixed
precision variants, CPU/ROCm/NPU, Trainer/Accelerate/PEFT and broader L5 performance remain open.

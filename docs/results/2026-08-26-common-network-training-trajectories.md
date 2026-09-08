# 常见网络三步训练轨迹对拍

- Status: CPU/CUDA correctness accepted
- Last reviewed: 2026-08-26
- Baseline: `4dbda832`
- Owner: model, optimizer, normalization, and test maintainers
- Review when: common parity builders, SGD, BatchNorm state, CUDA accumulation,
  or the maintained network set changes

## 结论

ResNet18、ViT、GPT-2 和 diffusion UNet 现在不仅完成单次 forward/backward 对拍，
还在 CPU 与真实 CUDA 上完成三步 SGD 训练轨迹。每个 case 从同一 PyTorch state
初始化，使用同一输入和固定 MSE target；门禁逐步比较 loss，最终比较完整 trainable
parameter 集合与数值、shared buffers，并确认参数确实更新。

新增轨迹暴露并修复两个原先单步 eval 对拍无法发现的问题：

1. Jittor `BatchNorm.num_batches_tracked` 在 train forward 中从不递增；现在仅在
   `track_running_stats=True` 时随 batch 更新，三步后与 PyTorch 同为 3；
2. PyTorch diffusion fixture 的 timestep frequency 固定创建在 CPU，导致真实 CUDA
   input device mismatch；现在跟随 `t.device`。

最终八项 CPU/CUDA 三步轨迹全部通过。后续新增的双语 ResNet tutorial 也在离线 CPU
notebook gate 中真实执行三步训练、BatchNorm counter 和 state restore。常见网络 todo
中的完整训练轨迹与本轮 notebook/示例扩充已完成；真实规模性能门禁仍保持开放。
后续 real-scale 复验中 UNet 达到 `0.79x`，ConvNet 从 `1.18x` 收窄到 `1.08x`，
ViT 仍约 `1.33x`。

## 轨迹合同

- Networks: compact ResNet18、2-block ViT、2-block GPT-2、compact diffusion UNet
- Optimizer: native Jittor SGD vs independent PyTorch SGD, no momentum/weight decay
- Steps: 3
- Learning rate: ResNet18 `1e-4`；其余 `1e-3`
- Target: fixed random float32 MSE target generated from a shared seed
- Device: CPU and real RTX 4090 CUDA
- Parameters: trainable name sets must match exactly; every final shared parameter
  is compared and at least one must change
- Buffers: every shared buffer is compared, including BatchNorm running state and
  `num_batches_tracked`

ResNet 使用较小学习率是为了让门禁衡量 update parity，而不是让三层累计 CUDA
浮点差异被 residual BatchNorm stack 在高学习率下放大成收敛实验。所有网络仍使用
相同误差门禁：CPU loss/parameter/buffer 为 `0.4%/0.3%/0.4%`，CUDA 为
`1.0%/0.8%/1.0%`，并分别设置 `2e-5/5e-5` 的 near-zero absolute parameter 和
buffer floor。

## Verify then fix

首次 CPU 运行中 ViT、GPT-2、UNet 通过；ResNet 只有近零 `bn1.bias` 触发比例放大，
最大绝对差为 `1.11e-6`。双门禁加入合理 absolute floor 后，四项 CPU 通过。

首次 CUDA 运行中：

- ViT/GPT-2 通过；
- UNet 在 PyTorch reference 的 CPU frequency tensor 处 device mismatch，模型未执行；
- ResNet 在 `lr=1e-3` 的第三步 loss 差 `2.56%`，部分近零 BatchNorm bias 更新差被
  放大。改用 parity-oriented `1e-4` 后在原严格 CUDA 误差带内通过。

加入 shared-buffer 比较后，CPU/CUDA ResNet 同时发现 `num_batches_tracked` 为 0 对
PyTorch 3。修复 counter 后，八项轨迹通过。

## 验证

- BatchNorm running variance + batch counter focused CPU/CUDA: `1 passed`;
- final three-step trajectory matrix: `8 passed in 43.34s`;
- original one-step forward/backward plus trajectory matrix: `16 passed`;
- Ruff lint ratchet: passed;
- repository layout/document governance: passed;
- complete normalization regression: `12 passed`;
- complete NN capability suite: `33 passed`;
- complete structure gate: `218 passed in 74.59s`.

## Notebook follow-up

`examples/notebooks/resnet_training.md` 使用 MyST/Jupytext 作为唯一源文件，包含：

- 两卷积 residual block、channel projection 和 compact ResNet；
- fixed synthetic batch 上的三步 native SGD；
- stem 参数真实更新与 `num_batches_tracked == 3` 断言；
- `state_dict` 到 fresh model 的 evaluation-logit restore。

不需要网络、CUDA 或外部数据。notebook source/fence/tag checks `3 passed`，Jupytext
无输出物化 `1 passed`，包含全部九个 smoke topics 的离线 CPU 执行
`1 passed in 468.92s`。

## Real-scale performance follow-up

同版本 package、双方 TF32/cuDNN benchmark、GPU 预热后的 compute-only step：

| Case | PyTorch | Jittor | Ratio | Result |
| --- | ---: | ---: | ---: | --- |
| ConvNet | `0.0162s` | `0.0175s` | `1.08x` | open |
| ViT | `0.0258s` | `0.0344s` | `1.33x` | open |
| Diffusers UNet2D | `0.0679s` | `0.0535s` | `0.79x` | accepted |

ConvNet profile 中旧 BatchNorm/elementwise 部分约 `7.1ms/step`，PyTorch native
BatchNorm 约 `1.5ms`。新增 4-D float32 affine CUDA BatchNorm 后：

- train forward/backward 每 channel 使用 CUB block，三类 gradients 一次生成；
- eval backward 固定 running stats，同一 kernel 生成 input/weight/bias gradients；
- cuDNN Conv 的 channel bias 使用专用 forward 与 per-channel bias reduction。

ConvNet wall 从 `0.0195s` 依次降到 `0.0180s`、`0.0175s`。剩余约 8% 主要在
cuDNN convolution 本体；profile 中 BatchNorm 已接近 PyTorch。

ViT profile 的普通 GEMM 每 step 约 `27.97ms`，PyTorch 约 `16.11ms`；attention
本身都约 `3-4ms`，LayerNorm A/B 只有约 `0.3ms`。shape-specific cuBLASLt 对
`1576x768 @ 3072x768^T` 从现有约 `132us` 退化到约 `381us`，未进入主仓库。
因此 ViT 剩余缺口需要更强的 CUDA GEMM backend/algorithm，而不是继续改 attention
或 LayerNorm 路由。

Artifacts under `$JITTOR_LAB_ROOT/_state/sdpa-training/`：

| Artifact | SHA-256 |
| --- | --- |
| `large_convnet_profile_20260826.json` | `619b7c3ff475494279e998a07c0badaff085c724b274b52d3d0274758353614b` |
| `large_convnet_profile_20260826_batchnorm.json` | `6125da76c33d0f3a840f4ccf5d385e335542eb744fcd9f005255f0777c924ac1` |
| `large_convnet_torch_profile_20260826.json` | `4eaf6570badda4c7d07e0a2479dbf9d1f3ee801574204c04bcb7124d0fc60f15` |
| `large_vit_profile_20260826.json` | `6acd9f4467041d54ccc1997339e0b9832970b32a93c78fd68a4f10e2289e145d` |
| `large_vit_torch_profile_20260826.json` | `16d6a7e276d58d2f2114bcc27d59c03fb4d6df6435aa9441ee5e8911cf21b4f2` |
| `cublas_lt_vit_forward.json` | `a0d05538685e9321e21a9115bc5397af63baa615b39ead73ebbcf73c5c93f8df` |

## Boundaries

- 当前门禁和教程证明紧凑网络三步更新语义，不代表 ImageNet、长序列 GPT-2 或大规模
  diffusion 的收敛/吞吐已经与 PyTorch 相同。
- ConvNet 与 ViT 的 real-scale 性能仍未接受；UNet 的受控 CUDA autotune 结果已接受。
- 使用 float32 plain SGD；mixed precision、AdamW、scheduler 和 checkpoint resume
  由各自维护门禁负责，不由本报告外推。
- NPU/ROCm 未执行；它们保持现有独立设备门禁。

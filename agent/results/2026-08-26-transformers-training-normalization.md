# Transformer 训练归一化与 float32 fused-attention 复验

- Status: CUDA correctness accepted; default performance improved, overall gate open
- Last reviewed: 2026-08-26
- Baseline: `295c3227`
- Owner: neural-network, Torch compatibility, and CUDA performance maintainers
- Review when: LayerNorm/RMSNorm CUDA training, module dispatch, float32 SDPA,
  cuBLAS matmul, or Transformers versions change

## 结论

真实 PyTorch profiler 证明参考 GPT-2/Llama 的 float32 SDPA 使用 CUTLASS
memory-efficient attention，而不是 math fallback。Jittor 原路径每 step 的三次 batched
GEMM、mask、softmax及其 backward 约 `9.64ms`；PyTorch fused 前后向约 `4.57ms`。

本轮落地两个不依赖外部 package 的框架能力：

1. 单末维、float32、affine CUDA LayerNorm 使用 fused forward、input backward 与
   affine-gradient kernels；
2. 标准 `variance_epsilon` RMSNorm 模块使用 fused float32 CUDA training capability，
   instance-level `forward`、gated/offset 变体和不支持的 device/dtype 保持原路径。

默认 math-attention 下，GPT-2 从 `0.0490s` 降到 `0.0455s`，Llama 从 `0.0478s`
降到 `0.0465s`。默认结果仍分别比 PyTorch 慢 `1.13x/1.22x`，所以生态性能总目标
保持开放。

配置官方 FlashAttention source 并显式设置 float32-to-fp16 cast 后，GPT-2 两轮预热
配对均快于 PyTorch；Llama 的 Jittor step 稳定在 `0.0391-0.0394s`，但最快干净
PyTorch reference 为 `0.0379s`，保守口径仍慢约 `3-4%`。该 cast 仍是 opt-in，
没有改变 float32 默认精度策略。

## 环境

- Code baseline: `295c3227`
- Python: 3.11.15
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9
- Jittor CUDA: 12.2.140; serial JIT compiler
- Real PyTorch: 2.12.1+cu130
- Shared Transformers: 4.56.2
- TF32: matmul/cuDNN enabled on both runtimes
- cuDNN benchmark: enabled on both runtimes
- Benchmark and unittest use separate caches under `$JITTOR_LAB_ROOT`

## Verify before fix

同轮初始基线：

| Case | Jittor | PyTorch | Ratio | Gradients |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 | `0.0490s` | `0.0419s` | `1.17x` | 100 |
| Llama | `0.0478s` | `0.0394s` | `1.21x` | 75 |

Jittor profile 中普通 cuBLAS matmul 占 GPT-2/Llama kernel 时间约 `50%/56%`。PyTorch
CUDA profile 进一步显示 float32 attention 使用
`_efficient_attention_forward/backward`；四个 step 的 GPT-2 fused attention 总计
`18.29ms`，Llama 总计 `18.93ms`。

真实 `[B=4,H=16,L=512,D=64]` causal float32 前反向 micro：

| Runtime/path | Median per call |
| --- | ---: |
| Jittor math | `1.460ms` |
| Jittor explicit fp16 cast + native FlashAttention | `0.574ms` |
| PyTorch default efficient attention | `0.827ms` |

数组级比较覆盖 output 和 Q/K/V gradients。cast 相对 Jittor math 的 relative L2 为
`2.09e-4` 至 `3.17e-4`；相对真实 PyTorch 为 `4.12e-4` 至 `5.49e-4`，所有结果
finite 且梯度非零。

## 实现边界

LayerNorm training capability 只接受：

- real CUDA、非 ACL、非 `no_grad`；
- float32 input/weight/bias；
- 单个 normalized dimension，末维和 affine 参数严格匹配；
- finite positive epsilon。

forward 保存逐行 mean/rstd；backward 使用稳定闭式公式分别计算 input 与 affine
gradients。CPU、float16/bfloat16、多维 normalized shape、无 affine、NPU 和 ROCm
继续原实现。

RMSNorm module dispatch 只接受类名以 `RMSNorm` 结尾、单位置输入、无 kwargs、实例
含 `variance_epsilon` 和匹配的一维 weight。instance `forward` override 优先，
`RMSNormGated`、Gemma 风格 unit-offset 和其他不满足契约的模块不会被替换。training
kernel 只覆盖 float32 CUDA；已有 inference capability 继续处理其原合同。

## 性能结果

### 默认 math attention

GPU 预热后、无外部 FlashAttention source/cast：

| Case | Jittor | PyTorch | Ratio |
| --- | ---: | ---: | ---: |
| GPT-2 | `0.0455s` | `0.0402s` | `1.13x` |
| Llama | `0.0465s` | `0.0381s` | `1.22x` |

这只接受 normalization 的绝对收益，不接受整体“不慢”结论。

### Opt-in fused attention

官方 source、native-required、显式 float32-to-fp16 cast，GPU 预热后两轮：

| Run | GPT-2 J/P | Ratio | Llama J/P | Ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | `0.0392/0.0435s` | `0.90x` | `0.0394/0.0379s` | `1.04x` |
| 2 | `0.0405/0.0429s` | `0.94x` | `0.0391/0.0511s` | `0.77x` |

PyTorch Llama 在共享主机上仍有明显波动，因此只接受 GPT-2 的 opt-in 不慢结论；
Llama 以最快 reference 判断仍保留约 `3-4%` 差距。

## Rejected experiments

- float32 TF32 `cublasGemmEx` 强制 `CUBLAS_GEMM_DEFAULT_TENSOR_OP`：GPT-2 profile
  从 `168ms` 退化到 `173ms`，已撤回。
- shape-specific cuBLASLt、零 workspace：`2048x1024 @ 2816x1024^T` 与现有
  GEMM 均为约 `174us`，逐元素一致但无收益。
- 同一 cuBLASLt shape、32 MiB workspace heuristic：退化到约 `359us`，未进入主仓库。

## 验证

- LayerNorm/RMSNorm CUDA focused + dispatch: `3 passed`;
- native normalization: `9 passed`;
- Torch normalization compatibility: `28 passed`;
- Torch NN compatibility: `31 passed`;
- complete NN capability suite: `33 passed`;
- final GPT-2/Llama ecosystem correctness: `2 passed`, 100/75 gradients;
- Ruff lint ratchet: passed;
- repository layout/document governance: passed;
- complete structure gate: `218 passed in 89.30s`.

## Artifacts

| Artifact under `$JITTOR_LAB_ROOT` | SHA-256 |
| --- | --- |
| `_state/sdpa-training/large_gpt2_profile_20260826.json` | `f7ce37e9647882e739ccd4d43254a68515b26d0b55a94f0908ba2de07762dae1` |
| `_state/sdpa-training/large_llama_profile_20260826.json` | `f0f5c83066b9a939e980447b6ce5a6dea277176e77d8ccd5277edc18cf76309d` |
| `_state/sdpa-training/large_gpt2_torch_profile_20260826.json` | `e52a0b509cec90714bb86397312b231042335ba4fe3277aa53dc17a1a96525aa` |
| `_state/sdpa-training/large_llama_torch_profile_20260826.json` | `63f1c40ba9199f10ec6c535319b7a4ed6a02e5d71a7635aaffaba3d6f5ce8302` |
| `jittor_transformers_perf/results/fp32_math_h16_l512_20260826.npz` | `dce6af03b255752b17695473ecea00c5cbb72d0550e45a61d95aa3c208269d1c` |
| `jittor_transformers_perf/results/fp32_flash_cast_h16_l512_20260826.npz` | `7d6c1533a9b2940443b3fd1712d69f69de7de9cb294f9c929bd5db005c530f68` |
| `jittor_transformers_perf/results/fp32_torch_h16_l512_20260826.npz` | `334d8ec78bba51b76c7f58f2b9304dfc8a78a641b29009eb1e0d1cace4fd956b` |

## Remaining work

- 默认 float32 path 仍缺少不降精度、训练可用的 fused CUDA attention backend。
- Llama opt-in 路径相对最快 PyTorch reference 仍慢约 `3-4%`；剩余主要是普通
  GEMM、SiLU/gating 和 graph-build overhead。
- NPU/ROCm 未执行新 CUDA capability；它们保持原路径，不据此宣称跨后端性能。

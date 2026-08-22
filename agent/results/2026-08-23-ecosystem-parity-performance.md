# 下游生态同版本对拍与 CUDA 训练热点优化

- Status: CPU/CUDA correctness accepted; performance partially accepted
- Last reviewed: 2026-08-23
- Baseline: `49ff7c7a`
- Owner: Torch compatibility, CUDA backend, and performance maintainers
- Review when: downstream versions, ecosystem harness timing, GroupNorm/CUDA,
  scalar power lowering, cuDNN policy, or training SDPA changes

## 结论

下游生态对拍现在在真实 PyTorch 与 Jittor 各自占有 `torch` namespace 后，强制加载
同一套下游 Python 包，并把依赖版本、来源、实际 device、TF32 和 cuDNN benchmark
状态纳入硬断言。当前基线下，十二个 Transformers、Diffusers、PEFT、ms-swift、
MMCV、MMEngine 用例在 CPU 与真实 CUDA 上均通过前向、输入梯度和全部参数梯度对拍。

性能剖析落地了两个框架级优化：

1. 4-D float32 affine GroupNorm 使用专用 CUDA forward/backward kernels；
2. float32 Var 的标量三次幂降为 `x*x*x`，消除 Transformers NewGELU 中昂贵的
   generic `powf`。

真实规模 Diffusers UNet 的 Jittor step 从约 `0.151s` 降到默认约 `0.080s`；双方启用
cuDNN autotune 后，Jittor 三次独立运行稳定在 `0.0530/0.0567/0.0551s`。GPT-2 从
约 `0.114s` 降到 `0.051s`。剩余 Transformer 约 1.3x 差距主要对应 Jittor 的训练
SDPA math fallback；本机 flash-attn Jittor backend 明确只支持 inference，未伪报训练
加速。

## 环境和版本

- Git baseline: `49ff7c7a`
- Python: 3.11.15
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9
- CUDA toolkit: 12.2.140; driver 595.84
- Real PyTorch: 2.12.1+cu130, binary `_C` validated
- Shared Transformers: 4.56.2
- Diffusers: 0.38.0
- PEFT: 0.17.1
- ms-swift: 4.5.2
- MMCV-lite: 2.1.0; MMEngine: 0.10.7
- JIT policy: isolated `$JITTOR_LAB_ROOT/_state/` caches,
  `use_parallel_op_compiler=0`

## Harness findings

### Same-version dependency ownership

修复前，Jittor 解释器使用 Transformers 4.56.2，而真实 PyTorch 解释器默认加载
5.12.1。runner 现在先加载并验证各自的 `torch`（以及真实侧 `torchvision`），再将
调用方的共享 package site 置顶。每个结果回报依赖版本和绝对 origin；两侧不一致会在
数值比较前失败。

### Timing and snapshot correctness

旧计时每 step 重新创建输入/H2D，并逐个把数百个梯度拷回 CPU。新计时预分配四个不同
device input slots 和一个 loss-weight tensor；Jittor 对 `[loss] + gradients` 显式
`jt.sync`，PyTorch backward 后统一 device sync，计时窗口内均不做梯度 D2H。

CPU 还暴露了一个只影响测试结果的 alias：Jittor `.numpy()` 可共享 runtime storage，
correctness 数组在 timing warmup 的 `zero_grad` 后被覆盖，造成十二项假失败。捕获现在
立即 `np.array(..., copy=True)`，并有源 buffer 覆盖回归测试。

CUDA matmul 与 cuDNN TF32 默认同时开启；`JITTOR_ECOSYSTEM_CUDNN_BENCHMARK=1` 会在
两个 runtime 同时开启 autotune，避免只调优一侧。

## Correctness matrix

| Gate | Result | Time |
| --- | --- | ---: |
| CPU ecosystem parity | 12 passed | 7:09 |
| CUDA ecosystem parity | 12 passed | 6:56 |
| GroupNorm CUDA forward + all gradients | 1 passed | 44.28s |
| complete normalization regression | 7 passed | 2:54 |
| GroupNorm CUDA OpInfo + device parity | 2 passed | 29.95s |
| complete core regression | 13 passed | 2:31 |

十二项包括 GPT-2、Llama、BERT、ViT、T5、Whisper、Diffusers UNet2D/DiT、PEFT
LoRA、ms-swift LoRA、MMCV ConvModule 和 MMEngine BaseModule。CUDA 计算在真实 device
执行，未接受 CPU fallback。

## Performance findings

### GroupNorm and Diffusers

修复前 UNet profile 的 generic GroupNorm forward/backward reductions 约占 kernel
时间 22%。专用实现每个 `(sample, group)` 使用一个 CUB block，两遍计算 mean/variance；
backward 用稳定闭式公式计算 input gradient，并用每 channel block 计算 affine gradients。
fast path 只覆盖 4-D float32 affine CUDA；CPU、ACL、其他 rank/dtype 和无 affine 路径
保持通用实现。

双方 cuDNN autotune 下三次独立 UNet 配对：

| Run | PyTorch | Jittor | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 0.1003s | 0.0530s | 0.53x |
| 2 | 0.0595s | 0.0567s | 0.95x |
| 3 | 0.0444s | 0.0551s | 1.24x |

Jittor 绝对时间稳定；PyTorch 在共享主机上波动较大。中位数时间比约 `0.93x`，因此只对
此受控 autotune 口径接受“不慢”结论，不外推到默认 CPU 或其他形状。

### GPT-2 NewGELU

Transformers `NewGELUActivation` 使用 `torch.pow(input, 3.0)`。优化前，每层融合
forward/backward kernel 含 `pow + tanh + cosh`，约占 GPT-2 kernel 时间 58%。只对
float32 Var 的标量指数 3 降为三次乘法后，完整 100-gradient step 从约 `0.114s` 降到
`0.051s`；同次 PyTorch 为 `0.040s`。其他 dtype 和指数仍走原 `pow` 语义。

### Final real-scale CUDA snapshot

相同 package、TF32 和双方 cuDNN autotune，修正后的 compute-only training 口径：

| Case | PyTorch | Jittor | Ratio |
| --- | ---: | ---: | ---: |
| ConvNet | 0.0161s | 0.0191s | 1.19x |
| Diffusers UNet2D | 0.0919s | 0.0554s | 0.60x |
| BERT | 0.0283s | 0.0369s | 1.30x |
| GPT-2 | 0.0402s | 0.0512s | 1.27x |
| Llama | 0.0384s | 0.0499s | 1.30x |
| ViT | 0.0261s | 0.0336s | 1.29x |

单次绝对值受共享主机影响；结论以热点变化、Jittor 重复稳定性和多次 UNet 配对共同支撑。

## Supporting gates

- Ruff lint ratchet: passed.
- Changed-file Ruff lint: passed; new format-ratchet files passed Ruff format.
- Repository layout and documentation governance: passed.
- `tests/structure`: `218 passed in 103.46s`.
- Full CPU normalization/core regression: `19 passed, 1 skipped`.
- Torch GroupNorm and optional Diffusers focused group: `4 passed, 5 skipped`.

The full `nox -s format` session remains red because seven unrelated pre-existing
ratchet files would be reformatted; no such files were changed in this work.

## Boundaries

- Transformer real-scale training remains about 1.27-1.30x slower; a proven training-capable
  fused SDPA backend is still required.
- CPU UNet/MMCV remain slower than PyTorch and are not accepted for performance.
- TRELLIS, verl, and vLLM were not revalidated on this baseline.
- The GroupNorm fast path does not claim float16/bfloat16, 3-D/5-D, non-affine, NPU, or ROCm.
- Raw profiles, generated kernels, and benchmark artifacts remain unversioned under
  `$JITTOR_LAB_ROOT`.

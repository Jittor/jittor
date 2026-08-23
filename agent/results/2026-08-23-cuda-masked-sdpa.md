# CUDA masked SDPA 防护图优化

- Status: CUDA correctness accepted; performance improved, remaining gap open
- Last reviewed: 2026-08-23
- Baseline: `aae5f3e6`
- Owner: neural-network and Torch compatibility maintainers
- Review when: SDPA masks, CUDA softmax, Transformers attention dispatch,
  cuBLAS matmul, or training performance changes

## 结论

实际测量否定了“先接外部 FlashAttention backward”这一未经验证的方案。当前
Jittor math SDPA 在 `[B=4,H=12,L=128,D=64]` 的 fp16/float32 前反向微基准中并不慢；
真实 8-layer GPT-2 的 profile 才显示问题来自 L=512 mask 路径为 fully-masked row
无条件构造的 reduction 与前后 ternary 防护图。

CUDA softmax 已经支持“输入整行为 `-inf` 时输出全零”，并且其闭式 backward 对全零
输出自然产生零梯度。本阶段让显式 mask 直接使用该能力；纯 causal mask 每行必有有效
对角元素，只跳过 row-valid 图并继续普通 softmax。CPU 和非 CUDA 后端保持原来的显式
row-valid 防护。

真实 CUDA GPT-2 step 从上一基线约 `0.0512s` 降到 `0.04791s`，约改善 `6.4%`；
同轮真实 PyTorch 为 `0.04018s`，比值从约 `1.27x` 收敛到 `1.192x`。Llama 从约
`0.0499s` 降到 `0.04698s`，同轮 PyTorch `0.03801s`，比值 `1.236x`。性能目标仍未
完成，剩余热点主要是线性层 cuBLAS matmul、attention batched matmul 和图构建。

## 验证环境

- Jittor code baseline: `aae5f3e6`
- Python: 3.11.15
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9
- CUDA: 12.2.140; JIT compiler serial
- Real PyTorch: 2.12.1
- Shared Transformers: 4.56.2
- TF32: matmul and cuDNN enabled on both runtimes
- cuDNN benchmark: enabled on both runtimes

所有缓存、profile、NPZ 和 runner 输出位于 `$JITTOR_LAB_ROOT`；benchmark 与
unittest 使用不同 `JITTOR_HOME`。

## Verify before fix

### SDPA micro baseline

形状 `[4,12,128,64]`，十个独立输入 slot，前向与 Q/K/V 三梯度全部保留并同步：

| Runtime | dtype/backend | Forward + backward |
| --- | --- | ---: |
| PyTorch | fp16 math | `2.584 ms` |
| PyTorch | fp16 flash | `1.418 ms` |
| Jittor | fp16 math | `0.379 ms` |
| PyTorch | float32 default | `1.469 ms` |
| PyTorch | float32 math | `1.923 ms` |
| Jittor | float32 math | `0.311 ms` |

数组级比较而非 checksum-only：

- float32 forward/Q/K/V gradients relative L2: `7.96e-7` to `1.07e-6`;
- fp16 forward/Q/K/V gradients relative L2: `5.63e-4` to `6.58e-4`;
- every result and gradient is finite and nonzero.

这证明常见短序列 math SDPA 本身不是当前性能缺陷，也说明只支持 fp16/bf16 的外部
FlashAttention backward 不能直接解决本轮 float32 真实模型目标。

### Large GPT-2 profile

profile 使用正式 `_ecosystem_speed` 的 GPT-2 配置：8 layers、hidden 1024、16 heads、
batch 4、sequence 512、vocab 32000。每个 kernel 2 次 warmup、5 次 rerun。

修改前后的 profile row 数从 `54` 降到 `50`。按文件身份比较，旧图删除：

- 一条 `20.83ms` 的 row-valid/mask backward 融合图；
- 三条 `4.81/4.81/4.87ms` 的 reduction/ternary 防护图；
- 少量 bool/int helper 图。

新图只有两条约 `4.81ms` 的简化 causal mask 图。五次 rerun 的同口径净减少约
`25.9ms`，即约 `5.2ms/step`。profiler 汇总从 54 rows / 108 GB memory access 变为
50 rows / 99.9 GB。整份 profiler 总时间还受同机 cuBLAS 波动影响，不单独作为 wall
time 结论。

Artifacts:

| Artifact | SHA-256 |
| --- | --- |
| before profile | `e22606b2f0a21345f091c9967bd37ade9274a59b30d3402e784f779771cff525` |
| after profile | `5878d94e89594f9b5f6c2d980c72761e25b42d15278b89dcf09b0bb632799a02` |
| SDPA micro JSONL | `d373119aeb681e403bff81608af4cbc64e81b96e450a2d328f3dca06e92722c5` |

## Real-scale paired results

每个 runtime 使用四个 resident input slots；计时包含 forward、loss、zero-grad、
backward 和 device synchronization，不包含逐梯度 D2H。

| Case | Jittor | PyTorch | Ratio |
| --- | ---: | ---: | ---: |
| GPT-2 | `0.04791s` | `0.04018s` | `1.192x` |
| Llama | `0.04698s` | `0.03801s` | `1.236x` |
| BERT | `0.03773s` | `0.02917s` | `1.294x` |

GPT-2 和 Llama 相对上一报告的 Jittor 绝对时间分别改善约 `6.4%` 和 `5.8%`。BERT
在同机重复运行中接近原基线，未据此宣称额外收益。

## Correctness gates

- selected ordinary/causal/fp16 masked/fully-masked SDPA: `4 passed`;
- new CUDA routing regression: `1 passed`; explicit mask uses
  `zero_all_neg_inf=True`, pure causal uses ordinary softmax;
- complete Torch attention suite: `33 passed, 3 skipped`; skips require an
  unconfigured external native FlashAttention source;
- CPU/CUDA native SDPA device parity: `2 passed`;
- independent Transformers CUDA parity: GPT-2 28 gradients, Llama 21 gradients,
  BERT 37 gradients; `3 passed`;
- `test_nn_structure`: `30 passed`;
- complete structure gate: `218 passed`.

The first Transformers parity attempt used an environment whose torchvision
binary did not match its PyTorch and failed before any Jittor case ran. The
accepted run used the maintained matching PyTorch/torchvision reference
environment; this environment failure is not counted as a framework regression.

## Remaining work

- The 8-layer GPT-2 profile now spends about half of measured kernel time in
  three linear-layer cuBLAS matmul families. Bias/activation fusion and matmul
  algorithm selection are higher-priority than adding an fp16-only Flash bridge.
- Attention still has three batched matmul directions plus mask and softmax
  kernels. A float32/TF32 fused backend may help at L=512, but must independently
  match forward and all three gradients before becoming default.
- GPT-2/Llama/BERT remain `1.19-1.29x` slower in this real-scale gate; the todo
  performance requirement remains open.
- NPU and ROCm were not exercised. They retain the pre-existing row-valid path;
  no result here claims accelerator parity beyond CUDA.

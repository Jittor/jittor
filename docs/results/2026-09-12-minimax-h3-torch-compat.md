# Torch 兼容层下的 MiniMax-H3

- 状态：在发布的 33B checkpoint 上端到端跑通，`generate_seconds` 与真 PyTorch 持平（同一
  时段 73.78 s 对 74.90 s）；「Open items」中的音轨对齐与首次运行的 JIT 代价仍开放
- 日期：2026-09-12，2026-09-14 复验
- 基线提交：`89622bb3`（首次运行，见「Changes」）；复验在 `a6b2c9c6`（基于 `5a084737`）
- 验证范围：单卡 CUDA 12.9、sm_90；`transformers==5.5.3`、diffusers `0.41.0.dev0`
  （`a71e62e`）、flash-attention `v2.7.4.post1`。缩减 checkpoint 在 CPU 上做数值对拍；发布的
  `MiniMaxAI/MiniMax-H3`（`FL2VA`）在 CUDA 上 512x512、124 帧、6 步端到端；对照为同机真
  PyTorch 2.9.1。同机其他租户占卡，只引用同一时段的比值
- 维护者：Torch 兼容层维护者
- 复查条件：「Open items」中受阻的条目落地，或 JIT 算子编译器的逐形状成本变化

## 问题

MiniMax-H3（33B 全模态视频+音频扩散 transformer、Qwen3-VL-32B 条件编码器、视频与音频
VAE）能否通过 `jittor.compat.torch`、走原生融合 FlashAttention 路径运行，并产出一段
视频？

上游集成是一个 diffusers `ModularPipeline`（`MiniMaxH3Blocks`），因此这属于下游库
适配，遵循
[`agent/skills/downstream-library-adaptation`](https://github.com/Jittor/jittor/blob/master/agent/skills/downstream-library-adaptation/SKILL.md)：
模型不做修改地直接使用，每个断点都被归到 jittor 核心、`jittor.compat.torch` 或
shim。

## 环境

- 使用本 checkout 的 Jittor，装入专用 venv：`python3.12`，以
  `pip install --no-deps --no-build-isolation .` 安装，再加 `./compat`，然后
  `jittor-torch-shim`。
- `transformers==5.5.3`（adapter 支持的版本），`diffusers` 为 `0.41.0.dev0`（main，
  `a71e62e`；MiniMax-H3 的类尚未进入发布版）。
- CUDA 12.9、sm_90；上游 `flash-attention` `v2.7.4.post1` 连同其 CUTLASS 子模块，经
  `JITTOR_FLASH_ATTN_JITTOR_SRC` 构建，并设置
  `JITTOR_FLASH_ATTN_HEAD_DIMS=128 JITTOR_FLASH_ATTN_DTYPES=bf16`。
- checkpoint：`MiniMaxAI/MiniMax-H3`，`FL2VA` 分片，diffusers 布局（transformer
  62 GB，文本编码器 63 GB，VAE 9.8 GB，音频 VAE 0.6 GB）。
- 对照：同一台机器上的真 PyTorch 2.9.1，相同的 diffusers checkout。

运行状态位于 `$JITTOR_LAB_ROOT/_state/h3/`；harness（`infer_h3.py`、
`probe_audio_vae.py`、`env-jittor.sh`）在 `$JITTOR_LAB_ROOT/minimax-h3/`，刻意放在
本仓库之外。

## 结果

### 缩减 checkpoint 上的数值对拍（L2，CPU）

`hf-internal-testing/tiny-minimax-h3-modular-pipe`，`t2va`，124 帧，2 步，并注入
视频与音频噪声，使两次运行使用相同的 latent：

| 输出 | 最大绝对误差 | 平均绝对误差 |
| --- | --- | --- |
| 视频帧（uint8，0-255） | 1 | 0.046 |
| 音轨（float） | 0.43 | 0.14 |

视频在舍入精度内一致。音轨不相关（`corr ≈ 0.05`，RMS 比值 0.53），因此音频 VAE
路径仍存在结构性差异；没有进一步定位，因为同一缩减 checkpoint 只有在注入噪声后才能
逐 bit 复现两次运行，而注入噪声正是视频路径关心的部分。

### CUDA 上的端到端

同一缩减 checkpoint 通过 shim 在 CUDA 上运行 `t2va`，写出带音轨的 124 帧 mp4
（64×64 与 512×288 画布），由原生融合 FlashAttention 内核服务
`scaled_dot_product_attention`。

### 发布的 checkpoint

加载可行：135 GB 的组件装入主机内存（≈300 s），组件管理器按需逐个搬到加速器上。随后
transformer 开始运行，但它的第一次前向把时间花在**编译 JIT 内核**上，而不是计算：
第一个小时内缓存了约 730 个不同内核，之后每分钟还会出现约 30 个，逐个出现。单次
去噪步骤在 75 分钟内未完成，因此本次会话没有从发布的 checkpoint 产出视频。

两个环境事实加剧了这一点，值得在下一次尝试前修复：

- shim 将 `JT_USE_PARALLEL_OP_COMPILER=0` 固定下来，即使从调用者环境导出，该值也不
  会改变；
- `auto_cpu_offload` 加上 shim 默认在加速器上创建的 tensor，意味着组件先被加载到
  主机内存再被搬走，因此加载路径与执行路径对默认设备的选择不一致。

### 在发布的 checkpoint 上复验（2026-09-14）

发布的 33B checkpoint 现在能产出视频。`run-jittor-parity.sh`（512x512，124 帧，
6 步，`--vae-dtype float16`，flash VAE attention）在 `a6b2c9c6` 上运行了两次；两次
都以 **exit 0、没有 `best_algo_idx`、没有 `corrupted`** 结束，并写出一个 512x512
H.264 mp4，含 124 帧和 AAC 音轨。

此前失败背后的两个阻塞点已修复，且两处修复都在 jittor 而非模型中：autocast 混合
dtype 请求（`63fc1485`、`4c3ab0e4`），以及退出时的堆破坏及其 relay 子图
use-after-free（`f43b90d6`、`c6b56a61`）。见
[退出破坏报告](2026-09-14-exit-heap-corruption.md)与
[autocast 报告](2026-09-14-autocast-conv-mixed-dtype.md)。

速度，相同请求，`generate_seconds`（可比较的指标——对照侧汇总按惰性方式计量组件
加载，而本 harness 采用急切加载）：

| 运行 | dit | text_encoder | vae.video | vae.audio | generate |
| --- | --- | --- | --- | --- | --- |
| 本 harness，热缓存 | 28.32 | 3.72 | 14.78 | 4.54 | **73.78 s** |
| 本 harness，热缓存，上一次会话（`al-jittor`） | 25.85 | 3.79 | 14.14 | 5.26 | **73.17 s** |
| PyTorch 对照，同一时段（`al-oracle`） | 21.47 | 10.53 | 6.42 | 0.19 | **74.90 s** |
| PyTorch 对照（`fin-oracle-s6`） | 22.92 | 10.67 | 6.25 | 0.20 | 79.67 s |

总计相差几个百分点；分阶段看，这里的文本编码器约快 2.8x，而视频/音频 VAE 慢
2.3x/24x，二者相互抵消。修复后的**第一次**运行耗时 300.9 s，因为卷积、matmul 与
relay 的改动使这些算子的 JIT 内核失效；稳态是 73-74 s 那一行。

两点说明：机器并非空闲（这些运行期间，所有 GPU 都报告来自其他租户的 94-100%
利用率），因此绝对秒数在不同时段之间会变化，只有上表同一时段的成对数据可以当作
比值来读；另外，急切组件加载（此处约 270-340 s）不计入 `generate_seconds`。

## 改动

在把流水线跑起来的过程中修复。每一项都是真实缺陷而非绕过；改变了共享行为的项都
列出了能捕获回归的证据。

### jittor 核心

| 改动 | 原因 |
| --- | --- |
| `ops/numerical.py`：新增 `sinc`，发布为 `jt.sinc`/`Var.sinc` | jittor 没有归一化 sinc；`torch.special.sinc` 本就期望 `jt.sinc`，而 H3 音频 VAE 会调用它 |
| `src/type/fp16_compute.h`：为 `float16`/`bfloat16` 增加默认构造函数 | 代码生成器为分块归约声明 `decltype(acc) stack[N]`；没有默认构造函数时，该 pass 作用到的每个 fp16/bf16 归约都无法编译 |
| `backends/cuda/kernels/cub/cub_cumsum_op.cc`：把 block size 纳入代码生成 key | JIT 源码变换把 C++ 模板参数压平为单个 `#define`，于是 `BlockScanKernel<512>` 与 `<1024>` 合并成同一个，8 字节类型拿到了 1024 线程的 block，其启动被拒绝（92 寄存器 × 1024 > 64k）。float64/int64 `cumsum` 以 `cudaErrorLaunchOutOfResources` 失败 |

### `jittor.compat.torch`

| 改动 | 原因 |
| --- | --- |
| `installers/numerical/signal.py`：`kaiser_window` | H3 音频 VAE 用它构建重采样滤波器 |
| `native_api.py`：native API 列表中加入 `sinc` | 即使核心已提供，`torch.sinc` 仍不可达 |
| `installers/tensor/method_api.py`：`contiguous` 实体化带步长的存储 | 它无条件返回 `self`，于是广播/扩展后的 Var 的存储小于其形状，紧接着的 `reshape` 以 "call contiguous() first" 失败，而本意是修正布局的这次调用什么也没做 |
| `installers/tensor/shape_api.py`：`reshape` 回退为拷贝 | torch 的 `reshape` 是「尽可能返回视图，否则拷贝」；原生路径只实现了视图那一半 |
| `frontend.py`：`Tensor(...)` 接受 `dtype`/`device`/`requires_grad`/`pin_memory` | `accelerate` 以 `param_cls(value, requires_grad=...)` 构造 tensor；拒绝所有关键字导致设备搬移失败 |

### shim

| 改动 | 原因 |
| --- | --- |
| `resources/torch_dist_info/top_level.txt` + `deploy.py` | 没有它，`importlib.metadata.packages_distributions()` 只把 `torch` 映射到 `jittor-torch`，Transformers 把 torch 版本读成 1.3.x 并禁用其模型代码 |
| `backends/flash_attention/__init__.py`：head dim 160 | 上游 flash-attention 提供 160 宽的内核，`flash_api.cpp` 会引用 160 的 split-kv dispatch；省略它留下未定义符号，整个原生后端导入失败 |
| `cpp_extension/src/jtorch_aten.cu`：接受 Var 子类 | shim 下的 `torch.Tensor` *就是* Var 子类，精确类型判断以 "q must be a Jittor Var" 拒绝了每个经 torch API 构造的 tensor |

## 未结事项

1. **大模型的 JIT 编译成本。** 33B transformer 的一次前向需要数千个不同内核。
   `jt.flags.use_parallel_op_compiler` 到达时为 `0`——从调用者环境导出
   `use_parallel_op_compiler` 或 `JT_USE_PARALLEL_OP_COMPILER` 都不会改变它——关闭时
   内核逐个构建，实测每分钟 15-30 个。该 flag *确实*可在运行时设置，运行 harness 在
   调用流水线前将它设为 16；框架侧的问题是为什么环境变量的写法被忽略，以及兼容层
   是否应为此类大模型默认打开它。2026-09-14 实测：jittor 源码改动后的首次
   512x512x124 运行 `generate_seconds` 为 **300.9 s**（该改动使受影响算子的内核
   失效），而缓存预热后的下一次运行为 **73.8 s**。冷启动成本是真实的，且按 jit-key
   集合计；它不是稳态吞吐。
2. **Autocast 混合 dtype。已在 `63fc1485` 与 `4c3ab0e4` 中修复。** H3 的视频解码在
   `torch.autocast(float16)` 下运行；shim 的 amp 寄存器改变了算子的*结果* dtype 却
   不转换其操作数，且在各算子之间应用不一致。一个 float32 卷积被要求产出 float16
   输出，cuDNN 无法提供这种算法，触发了 `cudnn_conv3d` 的 `best_algo_idx != -1`
   不变式；修复后，视频 VAE 的 attention 又以同样方式失败，因为 CUDA cuBLAS matmul
   忽略了该寄存器，而分解后的 `linear` 让 float32 bias 把结果抬回 float32。现在
   cuDNN 卷积算子、两个 cuBLAS matmul 算子以及 `linear` 都会把浮点操作数转换到共享
   推断选定的 dtype，与 torch 的 autocast 一致。视频解码现在能在 autocast 下使用
   float32 VAE 完成，不再需要 `--vae-dtype float16`。见
   [修复报告](2026-09-14-autocast-conv-mixed-dtype.md)。
3. **音轨对齐。** 在缩减 checkpoint 上，视频路径与对照在舍入精度内一致；音频 VAE
   路径仍不相关。
4. **Attention dtype 门控。** float32 的 attention 需要
   `JITTOR_FLASH_ATTN_CAST_FLOAT32=bf16` 才能到达 flash 内核；没有它，shim 会静默
   实体化一个稠密的 `[heads, seq, seq]` score 矩阵，在 H3 的序列长度下这是 80 GB。

## 复现

```bash
export JITTOR_LAB_ROOT=<lab-root>
source $JITTOR_LAB_ROOT/minimax-h3/env-jittor.sh
export use_cuda=1
export JITTOR_FLASH_ATTN_JITTOR_SRC=$JITTOR_LAB_ROOT/flash-attention
export JITTOR_FLASH_ATTN_JITTOR_REQUIRED=1
export JITTOR_FLASH_ATTN_HEAD_DIMS=128 JITTOR_FLASH_ATTN_DTYPES=bf16
export JITTOR_FLASH_ATTN_CAST_FLOAT32=bf16
$VENV/bin/python $JITTOR_LAB_ROOT/minimax-h3/infer_h3.py \
  --model $JITTOR_LAB_ROOT/_state/h3/models/tiny-h3 \
  --outdir $JITTOR_LAB_ROOT/_state/h3/runs/tiny-cuda --tag tiny-cuda \
  --height 64 --width 64 --num-frames 124 --steps 2 \
  --device cuda --vae-dtype float16
```

对照运行是在 `env-oracle.sh` 下执行同一命令，并设置
`PYTHONPATH=$JITTOR_LAB_ROOT/diffusers-main/src`。

原始日志、checkpoint、FlashAttention 构建与 JIT 缓存均不入库，位于
`$JITTOR_LAB_ROOT/_state/h3/`。

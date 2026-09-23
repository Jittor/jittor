# 下游库性能台账

- 状态：维护中
- 上次复查：2026-09-09
- 基线提交：`5932d7c9b`
- Owner：性能与兼容性维护者
- 复查触发：任何一次库级性能测量之后，或某一行的证据过期时

[基准测试方法](benchmarking.md)讲的是**怎么测**；本文件记的是**现在测到哪儿了**。

分开两份是因为它们的失效方式不同：方法学基本不变，而数字每次测量都会变。此前这些
数字散在 25 份带日期的报告里，格式、基线、日期、硬件各不相同——要回答「某个库现在
什么水平」得把 25 份读一遍。

## 口径

**比值 = Jittor 耗时 / 对照耗时。小于 1 表示 Jittor 更快。** 全文统一，包括引用的报告。

对照物按路径不同：

| 路径 | 对照 |
|---|---|
| Torch 兼容层 | 独立安装的二进制 PyTorch（**不是** Jittor 自己的 shim） |
| 昇腾 | 同机同卡的 `torch_npu` |
| vLLM 昇腾 | 原生 `vllm-ascend` |
| 原生计图 API | 见下方「没有对照的那一类」 |

**一个没有日期、提交和硬件的数字不是数字。** 本表每一行都必须带这三样，否则删掉
而不是留着——一个来历不明的比值比没有比值更糟，因为它会被引用。

## 两种行，不要混

**门禁行**由维护的 nightly 重新测量，过期会被自己发现。
**一次性行**来自某次调查，此后无人重测，**会静默变旧**。

目前只有第一类的一小部分进了门禁。

### 门禁行：`nox -s ecosystem` 的速度半

七个用例，走 Torch 兼容层对独立 PyTorch，阈值 `ECOSYSTEM_SPEED_RATIO = 1.07`。
比值**总是被报告**，只在 nightly 断言——墙钟上界是唯一能被机器负载单方面搞红的
断言，而速度门禁上的一次假红，代价大于一次迟到的真红：它会教人忽略那个同时还在
查数值的门禁。

| 用例 | 出处 |
|---|---|
| `large_convnet` / `large_transformers_bert` / `large_transformers_gpt2` | `compat/tests/torch/test_ecosystem_speed.py` |
| `large_transformers_llama` / `large_transformers_qwen3` / `large_transformers_vit` | 同上 |
| `large_diffusers_unet2d` | 同上 |

### 一次性行：报告里的当前站位

以下数字**不在任何门禁里**，来自各自的调查报告。日期即测量日期；此后无人重测。

| 库 / 用例 | 设备 | 比值 | 状态 | 证据 |
|---|---|---|---|---|
| ViT | CUDA | `1.33x` | **开口**，但归因已失效，见下 | `2026-08-26-common-network-training-trajectories.md` |
| ConvNet | CUDA | `1.08x` | 已改善 | 同上 |
| UNet | CUDA | `0.79x` | 已接受 | 同上 |
| GPT-2（数学注意力） | CUDA | `1.13x` | 配置相关，见下 | `2026-08-23-ecosystem-parity-performance.md` |
| GPT-2（显式 FlashAttention + fp16 转换） | CUDA | `0.90–0.94x` | 已接受 | 同上 |
| Llama（数学注意力） | CUDA | `1.22x` | 配置相关 | 同上 |
| Llama（显式 FlashAttention） | CUDA | 约 `1.03–1.04x` | 保守差距 | 同上 |
| TRELLIS.2 端到端 | CUDA | `1.093x` | **开口**（曾为 `1.20x`） | `2026-08-23-verl-vllm-trellis-current-baseline.md` |
| vLLM Qwen3-0.6B | CUDA | 约 `0.795x`（快约 20.5%） | 已接受，4-token 协议 | 同上 |
| Diffusers | 昇腾 910B3 | `0.964x` | 已接受 | `2026-08-30-diffusers-ascend-parity-performance.md` |
| MMCV / MMEngine（tiny） | 昇腾 910B3 | `0.927x` / `0.796x` | 已接受 | `2026-08-30-mmcv-mmengine-ascend-parity.md` |
| ms-swift LoRA（tiny，融合 fp32 因果 SDPA） | 昇腾 910B3 | `0.969x` | 已接受 | `2026-08-31-ms-swift-ascend-parity-performance.md` |
| Qwen3-0.6B 解码 | 昇腾 910B3 | 15.90 对 16.19 token/s | 已接受 | `transformers/2026-08-28-qwen3-ascend-performance.md` |
| Qwen3-0.6B BF16 SDPA 生成 | 昇腾 910B3 | 14.92 对 15.31 token/s | 已接受，零 CPU fallback | `transformers/2026-08-30-qwen3-ascend-training.md` |
| Qwen3-0.6B FP32 前向/损失/反向 | 昇腾 910B3 | `1.07–1.12x` | 已接受 | 同上 |
| Qwen3 同设备训练协议 | 昇腾 910B3 | `1.063x`（曾为 `1.195x`） | **开口**，精确路径性能门禁未建 | 同上 |
| vLLM Qwen3-0.6B 暖态请求中位数 | 昇腾 910B3 | `0.36330s` 对原生 `vllm-ascend` `0.38998s` | 已接受，单请求短上下文 TP=1 | `2026-08-31-vllm-ascend-jittor-bootstrap.md` |

报告路径均相对 `refactor-wip/results/`。**整改收口后该目录整体删除**，届时仍需保留的
行应把证据迁往 `docs/performance/` 或重测。

### `fuse_op_limit` 已经在好的区间里（2026-09-22 实测，负面结果）

视频 VAE 的 profile 是平的——最高的算子只占 `11.8%`，一次 decode 有 `499GB`
访存而输出只有 `41MB`。这种形状的直觉是「融合得更狠一点，少几遍」，所以扫了
`fuse_op_limit`（H3 video VAE `decode_base`，4 帧）：

| `fuse_op_limit` | decode | 相对默认 |
| --- | --- | --- |
| `4` | `1.748s` | 慢 37% |
| `8` | `1.236s` / `1.240s` | — |
| `12` | `1.266s` | — |
| `16`（默认） | `1.275s` | 基准 |
| `32` | `1.369s` | 慢 7% |
| `64` | `1.529s` | 慢 20% |
| `0`（无上限） | `2.694s` | **慢 111%** |

**方向和直觉相反：融合越宽越慢**，无上限直接翻倍。flag 自己的说明早写了原因——
很宽的融合要把每个活跃中间量留在寄存器里。默认值 `16` 已经在曲线的好区间里，
**「提高融合上限」这条路是死的**，记在这里免得后人重走。

**`8` 看起来比 `16` 快 3%，但这台机器测不出 3%。** 串行 A/B/A/B 的四次里，
`fuse=8` 两次是 `1.236`/`1.240`（差 0.3%），`fuse=16` 两次是 `1.275`/`2.402`
——后者几乎翻倍。机器负载在测量期间漂移，**运行间方差可达 90%**，所以低于约
10% 的差异在这里不可测量。上表里 `4`、`32`、`64`、`0` 那几档远在噪声之上，可信；
`8` 与 `16` 之间的差不可信，没有据此改默认值。

### 两条 2026-09-22 的加速测量

**分组 `conv_transpose` 现在会问加速 kernel。** `conv_transpose` 的
`groups == 1` 分支一直调 `select_kernel`，分组分支从来没调过，所以每个 depthwise
转置卷积都走八维 `reindex * reindex -> reindex_reduce` 降级——而 cuDNN 适配器本身
就接收 `groups` 并传给 `cudnn_conv_backward_x`。能力一直在，没人问。

| 用例 | 设备 | 改前 | 改后 | 比值 |
| --- | --- | --- | --- | --- |
| H3 音频 VAE `decode` | CUDA | `2.962s` | `1.404s` | `0.47x` |

那一个融合算子原先占整个 decode 的 `79.8%`，带宽 `344MB/s`。两条路径在
depthwise、`groups < C`、padding、output_padding 和 dilation 下**逐位相同**
（`rel = 0`），用例断言相等而非容差。

**`transpose_storage_view` 的默认值维持关闭，因为两个方向的测量互相矛盾。**
它让permutation 成为输入分配的视图而不是拷贝。哪边快取决于谁消费这个转置：

| 负载 | 关 | 开 | 变化 |
| --- | --- | --- | --- |
| H3 视频 VAE `decode_base`（4 帧） | `1.581s` | `1.272s` | **快 24%** |
| `1576x768 @ 768x3072`（转置操作数喂 cuBLAS） | `594us` | `702us` | **慢 18%** |
| `4096³` | `9964us` | `10260us` | 慢 3% |
| `8192³` | `76485us` | `77812us` | 慢 1.7% |

转置喂逐元素/卷积链时视图省下一次拷贝；喂 cuBLAS 时 GEMM 要的是稠密操作数。
视频 VAE 这一档的数值差（`0.00151`）**小于**同一档自己的运行间抖动
（`0.00166`），所以不是精度换速度。**没有单一正确的默认值**，逐模型测。

### ViT 行的归因已经过期（2026-09-22 实测）

那一行写的是「主导的 CUDA GEMM 落后于参考」，依据是 `27.97ms` 对 `16.11ms` 的
每步普通 GEMM，结论是「需要更强的 CUDA GEMM backend/algorithm」。**这条归因今天
不成立。** 同一张卡、同一进程节奏、来回背靠背测的四个形状：

| 形状 | Jittor | 独立 PyTorch | 比值 |
| --- | --- | --- | --- |
| `1576x768 @ 768x3072`（ViT 那个） | `593.0us` | `583.4us` | `1.02x` |
| `1576x768 @ 768x768` | `230.2us` | `229.6us` | `1.00x` |
| `4096³` | `9963us` | `12798us` | `0.78x` |
| `8192³` | `76497us` | `78463us` | `0.97x` |

两边都是 `float32_matmul_precision = highest`（两个框架的默认值都是它，所以差距
也不是精度档）。**普通 GEMM 已经持平**——前两行的差在百分之二以内，而本文件另一节
记下这台机器的运行间方差可达 90%，所以「持平」正是这两行能支持的全部措辞，不能
读成任何一方更快。`4096³` 那行的 22% 高于噪声底线，但**只测了一次**，够不上
「Jittor 更快」这个断言；它只说明这里没有一个能盖过噪声的反向差距。

**没有一并重测的是端到端的 `1.33x` 本身**：它要 `nox -s ecosystem` 的口径，也就是
环境里装有独立二进制 PyTorch，而做这次测量的环境没有，用例直接 skip。所以这里只
推翻归因，不宣布比值。手搓一个 ViT 来代替那个口径是行不通的——本文件开头就写了
为什么：那比的是两套实现。

这一行也是「一次性行会静默变旧」的实例：数字放了一个月，它给出的方向（换 GEMM
backend）今天会让人白做。

**这台机器上的绝对吞吐不可引用**：一个 `gpu_occupy` 作业在全部八张卡上空转到
100% 利用率，上表每个数字都被它压低。同机来回测出的比值不受影响，峰值百分比受。

一条被记录过的反例值得留着：昇腾上直接用 CANN RoPE 可达 `0.988x`，**被拒绝**，
因为它的 logits 与梯度轨迹与参考不同。**更快但不一致的路径不是性能成果。**

## 没有对照的那一类

原生计图的下游库——[JSeg](https://github.com/Jittor/JSeg)、
[JDet](https://github.com/Jittor/JDet)、
[JittorGeometric](https://github.com/AlgRUC/JittorGeometric)——**目前一个性能用例都没有**，
本仓的 `benchmarks/` 是框架级的（算子、优化器步、归约、主机↔设备传输、tiny_llama），
不是库级的。

这类库还有一个方法上的问题必须先解决，否则测出来的数字没有意义：**它们没有直接的
PyTorch 对照物**。JDet 不是 mmdetection，JSeg 不是 mmsegmentation，模型定义、数据增强
和训练调度都不同。所以可选的口径只有三种，各有代价：

1. **对等价的 PyTorch 实现**（如 JDet 对 mmdetection 的同名模型）——数字有意义，
   但需要逐个确认两边确实在算同一件事，否则比的是两套实现而不是两个框架；
2. **对自己的历史**——回归检测有效，但回答不了"比 PyTorch 慢多少"；
3. **只测框架级算子**——已经有了，但下游库的真实瓶颈往往在数据管线和调度上。

**在选定口径之前不要产出比值。** 一个口径不明的"JSeg 比 PyTorch 慢 1.4 倍"会被引用，
然后无法被反驳。

## 维护规则

- 新增一行必须带日期、提交、硬件和证据链接；
- 门禁行由 nightly 更新；一次性行**在其证据报告更新时**同步，否则视为过期；
- 一行的状态只有「已接受」「开口」两种。"接受"意味着有人判断这个差距可以带着走，
  不是"测完了"；
- **拒绝掉的更快路径要记下来**（如上面的 CANN RoPE），否则后人会重新发现它、
  重新采纳它，再重新发现它不对。

### 原生 SDPA 在 CUDA 上没有注册 kernel，长序列上差 2.72x（2026-09-22 实测）

`python/jittor/nn/functional/attention.py` 会问
`try_dispatch("nn.scaled_dot_product_attention", ...)`。CUDA 上没有任何东西注册
在这个名字下——`backends/acl/kernels/install.py:53` 注册了它，**只给昇腾**——所以
每一次原生调用都落到数学降级，物化一个 N×N 的分数矩阵。flash 路径存在，但住在
`compat/torch/installers/nn/attention.py`，只有走 torch shim 才够得着。

同一批 q/k/v，交错 A/B/A/B 取中位数，fp16，`hits=6 misses=0`（flash 每次都命中）：

| 形状 | 原生（数学） | shim（flash） | 比值 |
| --- | --- | --- | --- |
| H3 video VAE 注意力 `2x32x1797x1797x64` | `0.0081s` | `0.0030s` | **`2.72x`** |
| GPT-2 medium `4x16x1024x1024x64` | `0.0030s` | `0.0026s` | `1.14x` |
| Llama prefill `1x32x2048x2048x128` | `0.0038s` | `0.0031s` | `1.23x` |

**差距随序列长度增长**，和 O(N²) 显存对 O(N) 的预期一致。五次重复之间的离散度
低于 1%，远在这台机器的噪声之上（占卡程序在跑，绝对值不可比，比值可比）。

`no_grad` 和带梯度两档结果相同：只要反向 kernel 已经编出来，
`attention.py:212` 的 training 能力检查就过得去，所以「带梯度会挡住 flash」这个
担心是多余的。

**这个口子已经补上。** `backends/cuda/kernels/nn/flash_attention_cuda.py`
把桥注册到了这个名字下，`nn/functional/attention.py` 在第一次调用时延迟导入它
（和 `softmax.py` 对 `softmax_cuda` 的做法同形）。原生入口本身的前后对比，两个
独立进程各跑一档，避免进程内注销 kernel 扰动派发状态：

| 形状 | 改前（数学） | 改后（flash） | 比值 |
| --- | --- | --- | --- |
| H3 video VAE `2x32x1797x1797x64` | `0.0081s` | `0.0029s` | **`2.79x`** |
| GPT-2 medium | `0.0030s` | `0.0026s` | `1.15x` |
| Llama prefill | `0.0037s` | `0.0030s` | `1.23x` |

**同一批输入、同一进程内**比对两条路径：四个形状上
`max|flash-math|` 都是 `0.000488`（即 `2**-11`，相对 `4.9e-4`），就是 flash 把
softmax 累加换了顺序在 float16 下的舍入，不随形状变化。跨进程的 checksum 只能
说明大和会集中，不是证据，所以没有用它下结论。

**没装 flash 的人一个比特都不会变。** `load_backend_for` 找不到源码 checkout
就没有 backend，kernel 返回 None，`try_dispatch` 原样传回，调用方走的还是那条
数学路径。会被拒的情况都列在 kernel 的注释里：mask、dropout、非四维、q/k/v 头数
不等、flash 没有模板的 head_dim，以及 `is_causal` 且 q/k 长度不等——最后一条是
因为 flash 的因果掩码在长度不等时对齐右下，而这里的降级用 `triu(diagonal=1)`
对齐左上，形状一样答案不一样。

一个必须记下来的坑：kernel 最初直接返回扩展输出的 `reshape().permute()` 惰性
视图，之后同进程里的一次数学注意力就在 `matmul` 里带着一个乱掉的 NanoVector
形状崩了。输出和输入一样要 `clone()` 落地。

复现：`$JITTOR_LAB_ROOT/sdpa-dispatch/bench_sdpa_paths.py`。跑之前需要 pybind11
头文件和 `JITTOR_FLASH_ATTN_JITTOR_SRC`，见 `examples/flash-attention/README.md`。

### 补上对 PyTorch 的那一栏：接上 flash 之后仍慢 6–14%，因为 kernel 选错了

上一节的 `2.79x` 是 **jittor 改前对 jittor 改后**，不是对 PyTorch。拿它当「加速」
的结论汇报是误导——PyTorch 的 SDPA 本来就走融合注意力，所以那次改动是**追平**，
不是超过。这一节把缺的那一栏补上。

真 PyTorch `2.13.0+cu129`，同一张 H20，同样的形状/dtype/`is_causal`，同样的
10 次 warmup + 20 次重复取中位数（5 次重复对 torch 不够：raw 会从 `0.0024`
一路掉到 `0.0001`，那是没热起来，不是结果）：

| 形状 | jittor + flash-attn 2 | torch 默认 | 比值 |
| --- | --- | --- | --- |
| H3 video VAE `2x32x1797x1797x64` | `0.00292s` | `0.00262s` | 慢 `11%` |
| GPT-2 medium | `0.00258s` | `0.00244s` | 慢 `6%` |
| Llama prefill | `0.00303s` | `0.00266s` | 慢 `14%` |

**差距的来源不是 jittor 的调用路径，是算法选择。** 把 torch 的后端逐个钉死来问：

| 形状 | torch cudnn（= 默认） | torch flash | torch mem_efficient | jittor flash |
| --- | --- | --- | --- | --- |
| H3 video VAE | `0.00262s` | `0.00280s` | `0.00296s` | `0.00292s` |
| GPT-2 medium | `0.00244s` | `0.00250s` | `0.00256s` | `0.00258s` |
| Llama prefill | `0.00266s` | `0.00289s` | `0.00303s` | `0.00302s` |

两条结论：

1. **jittor 的 flash 路径和 torch 的 flash 路径只差 `3–5%`**。layout 转换、
   派发、跨扩展边界这些加起来就这么多，管道基本追平了。
2. **在 H20 上 cuDNN 的融合注意力比 flash-attn 2 快 `6–9%`**，torch 默认选它。
   jittor 跑的是慢的那个算法。

所以「让 jittor 的注意力不慢于 torch」这件事的下一步是**给 CUDA 后端写一个
cuDNN 融合注意力 kernel**，注册到同一个 `nn.scaled_dot_product_attention` 名字
下、优先级高于 flash 桥。收益已经量出来了：对 flash-attn 2 再快 `6–9%`，并且
cuDNN 随 CUDA 栈发货，不需要 flash 源码 checkout 和 pybind11。

**一条被推翻的猜想，记下来免得后人重走。** 我以为差距来自这个 kernel 对 q/k/v
各做的 `permute().reshape().clone()` 和输出那次 `clone()`——四次整张量拷贝，
H3 那档约 117MB 额外访存，量级上足够解释。去掉输入那三次 clone 后实测
`0.00292 / 0.00258 / 0.00302`，和保留时的 `0.00292 / 0.00258 / 0.00303`
**逐档相同**，正确性也不变。jittor 的图把冗余拷贝消掉了，那几次 clone 不花钱，
这条路是死的。clone 予以保留（它挡的是扩展边界上的悬挂视图，见上节）。

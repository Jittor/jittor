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
| ViT | CUDA | `1.33x` | **开口**，主导的 CUDA GEMM 落后于参考 | `2026-08-26-common-network-training-trajectories.md` |
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

# Qwen3-0.6B Ascend 推理性能诊断与优化

- Status: Accepted for float32 eager and SDPA greedy inference on one Ascend 910B3
- Last reviewed: 2026-08-29
- Source baseline: `f8e39607` plus the changes in this report's commit
- Owner: Torch compatibility and ACL backend maintainers
- Review when: Transformers masking, vmap compatibility, ACL RMSNorm/RoPE/SDPA,
  Torch constructor wrapping, CANN, checkpoint, dtype boundary, or timing
  protocol changes

## 结论

Qwen3-0.6B 原先的稳态慢速不是矩阵乘计算能力不足，主要来自两个算子提交放大点：
Transformers causal mask 的四层嵌套 `torch.vmap` 被兼容层展开成 Python 标量循环，
以及每层 RMSNorm 被拆成多个 ACL 算子。加上 ACL getitem 中两个不必要的显式同步，
一个 22-token prefill 共提交 10,096 个 ACL 调用。

本次将 mask 构建限定在 `TransformGetItemToIndex` 上下文内做广播向量化，移除
getitem 的显式同步，并为无梯度推理接入 `aclnnRmsNorm`。ACL 调用降至 1,493，
减少 85.2%。Jittor prefill 中位数从 0.8191 s 降至 0.1226 s，单 token generation
从 0.8139 s 降至 0.1409 s，分别加速 6.68 倍和 5.77 倍。

多 token decode 还存在一个独立阻塞：ACL 的混合 `slice + tensor index` 不受
`GetItemACL` 支持，异常 fallback 又调用同一个 ACL wrapper，形成无限递归。显式
lowering 到已有 Index/IndexPut 路径后，8-token greedy generation 可稳定完成并输出
`2 + 2 = 4.`。

最终 greedy selection 原先仍把 `arg_reduce` 编译到 CPU。本次用 CANN
`aclnnMaxDim`/`aclnnMinDim` 返回 values 和 indices；0.6B 的 8-token generation
与 8B 的 1-token generation 均在真实 NPU 上达到 `cpu_compile_count=0`、
`fallback_count=0`。单 token generation 中位数从 0.1409 s 降至 0.1358 s，改善
3.7%；prefill 未触发该算子，0.1226 s 到 0.1235 s 的 0.7% 变化视为测量波动。

剩余 profile 中，Qwen3 每层的 RoPE 仍由 Slice、Neg、Concat、Mul 和 Add 组合，
28 层共涉及 392 次算子提交。新的无梯度快路径使用
`aclnnRotaryPositionEmbedding`，每层 q/k 各提交一次，净减少 336 次提交；训练和
未验证形状仍走可微组合实现。同一张 NPU 上的严格成对 A/B 中，prefill 从
0.1248 s 降至 0.0934 s，单 token generation 从 0.1389 s 降至 0.1084 s，分别
改善 25.1% 和 21.9%；8-token generation 从 1.0591 s 降至 0.8195 s，改善 22.6%。

RoPE 融合后，attention 仍由 QK matmul、scale、mask、softmax 和 value matmul
组合提交。本次把 Torch SDPA 的已验证无梯度 FP32 子集接到 CANN
`aclnnFlashAttentionScoreV2`，保留 BNSD、GQA、方形 causal prefill、矩形 decode
和 float additive mask 语义。Transformers 的 SDPA mask 判断还会调用
`padding_mask.all()`；最初通过 nonzero、int32 min 和 bool cast 的精确 lowering
避免底层 `reduce.logical_and` 回退。0.6B prefill 进一步降至 0.0658 s，
单 token generation 降至 0.0792 s，8-token generation 降至 0.5791 s，较 ACL
RoPE 分别改善 29.6%、26.9% 和 29.3%。

当前源码复验又发现两个较小但稳定的提交开销。ACL fused executor 原先在每个成功
图的入口无条件同步整条 stream；成功路径改为保持同 stream 异步，只在准备 CPU
fallback 前 drain。CANN 9 已提供 `aclnnAll` 和 `aclnnAny`，因此公开 truth
reduction 也由多算子组合改为单个原生归约，numeric 输入只保留必要的 nonzero
比较。10-sample 8-token generation 最终为 0.5459 s，较当前源码修改前的
0.5690 s 改善 4.1%，输出和完整 logits 不变且仍为零 fallback。

同版本 Python profile 随后确认 Torch 兼容构造器还拦截了 ACL adapter 内部的
`jt.empty`：单次 8-token generation 有 7,564 次 native `empty` 调用进入完整的
device、shape 和 dtype 兼容流程。仅对无关键字参数且 shape 已是 native int、
tuple、list、NanoVector 或一组 Python int 的 `empty` 调用走原始构造器，Torch
Size、NumPy/Var 维度和所有显式 Torch 关键字仍保留完整适配。调用数从 835,904 降至
592,450；10-sample prefill 从 0.06325 s 降至 0.05796 s，8-token generation 从
0.54590 s 降至 0.50309 s，分别改善 8.4% 和 7.8%。

## 环境与口径

- Device: one allocated Ascend 910B3
- Driver / CANN: 25.5.1 / 9.0.0
- Jittor: 1.3.11.0 from the current `2.0` source tree
- PyTorch / torch_npu: 2.10.0 / 2.10.0
- Transformers: 4.56.2 on both sides
- Model: local Qwen3-0.6B, 596,049,920 parameters
- Input: batch 1, 22-token chat prompt, eager or SDPA attention, KV cache, float32
- Timing: current SDPA single-token rerun uses 2 warmups and 7 synchronized samples;
  historical 8-token rows use 2 warmups and 5 synchronized samples; the latest
  paired rerun uses 3 warmups and 10 synchronized samples; median and p90 reported
- Isolation: separate processes, Python environments, JIT caches, homes, and
  temporary directories; raw logs remain under `$JITTOR_LAB_ROOT/_state/`

Checkpoint loading and first JIT are not mixed into steady-state timing. The
first SDPA prefill and generation in the measured Jittor process took 1.69 s and
0.53 s; these include first-call work and are not the per-request steady-state
latency below. The earlier empty ACL cache took 120.09 s and 40.02 s for the
same phases, which is compilation rather than inference latency.

## 性能结果

| Backend | Prefill median / p90 | Generate median / p90 | Relative to paired PyTorch |
| --- | ---: | ---: | ---: |
| Jittor before | 0.8191 s / 0.8228 s | 0.8139 s / 0.8162 s | 11.60x / 10.78x |
| Jittor optimized before ACL arg-reduce | 0.1226 s / 0.1243 s | 0.1409 s / 0.1414 s | 1.74x / 1.87x |
| Jittor with ACL arg-reduce | 0.1235 s / 0.1266 s | 0.1358 s / 0.1367 s | 1.77x / 1.84x |
| Paired Jittor before ACL RoPE | 0.1248 s / 0.1282 s | 0.1389 s / 0.1420 s | 1.78x / 1.88x |
| Jittor with ACL RoPE | 0.0934 s / 0.0963 s | 0.1084 s / 0.1098 s | 1.34x / 1.47x |
| Native PyTorch NPU, eager rerun | 0.0699 s / 0.0713 s | 0.0737 s / 0.0738 s | 1.00x |
| Jittor with ACL SDPA | 0.0658 s / 0.0661 s | 0.0792 s / 0.0794 s | 1.02x / 1.12x |
| Native PyTorch NPU, SDPA rerun | 0.0648 s / 0.0650 s | 0.0709 s / 0.0710 s | 1.00x |

The historical rows retain their original paired PyTorch ratios. The current
Jittor and PyTorch rows were rerun back-to-back on the same selected 910B3.

启用 ACL RoPE 后，prefill 可分为约 0.0648 s Python/graph/operator submission 和
0.0285 s final synchronization；融合前的成对结果是 0.0858 s 和 0.0390 s。
原生 PyTorch 总 prefill 为 0.0699 s，说明剩余差距仍主要与 Jittor 的细粒度调用及
主机提交路径有关，不能归因成单个 GEMM 太慢。

ACL SDPA 后，Jittor 的 0.0658 s prefill 已接近原生 PyTorch SDPA 的 0.0648 s；
单 token generation 仍慢 11.8%，8-token generation 慢 23.0%。Jittor SDPA
prefill 中位数由约 0.0472 s graph/operator submission 和 0.0186 s final
synchronization 构成。剩余 decode 差距因此主要落在框架 generate 调度、KV cache
更新和主机提交，而不是未融合 attention 或 GEMM 本身。

| Backend | 8-token median / p90 | Throughput | Output |
| --- | ---: | ---: | --- |
| Jittor with ACL arg-reduce | 1.0845 s / 1.0874 s | 7.38 token/s | `2 + 2 = 4.` |
| Paired Jittor before ACL RoPE | 1.0591 s / 1.0753 s | 7.55 token/s | `2 + 2 = 4.` |
| Jittor with ACL RoPE | 0.8195 s / 0.8325 s | 9.76 token/s | `2 + 2 = 4.` |
| Native PyTorch NPU, eager rerun | 0.5843 s / 0.5899 s | 13.69 token/s | `2 + 2 = 4.` |
| Jittor with ACL SDPA | 0.5791 s / 0.5842 s | 13.81 token/s | `2 + 2 = 4.` |
| Native PyTorch NPU, SDPA rerun | 0.4708 s / 0.4728 s | 16.99 token/s | `2 + 2 = 4.` |
| Current Jittor before async/truth-reduce follow-up | 0.5690 s / not retained | 14.06 token/s | `2 + 2 = 4.` |
| Current Jittor with async ACL and native truth reduction | 0.5459 s / 0.5482 s | 14.65 token/s | `2 + 2 = 4.` |
| Current Jittor with native `empty` fast path | 0.5031 s / 0.5037 s | 15.90 token/s | `2 + 2 = 4.` |
| Current native PyTorch NPU | 0.4940 s / 0.4966 s | 16.19 token/s | `2 + 2 = 4.` |
| Rejected Jittor `pipeline_ops=1600` | 0.5718 s / 0.5737 s | 13.99 token/s | `2 + 2 = 4.` |

8-token 默认 Transformers 路径已不再卡死；启用 ACL SDPA 后的总延迟为同口径
原生 PyTorch 的 1.23 倍。这里报告整次 `generate` 延迟，不用总时间简单除以 8
伪装逐 token 首包延迟。

最新 10-sample 同时段复验中，Jittor prefill 为 0.05796 s，原生 PyTorch 为
0.05867 s；8-token generation 分别为 0.50309 s 和 0.49400 s，Jittor 当前仍慢
约 1.8%，所以“不慢于 PyTorch”尚未达成。曾在旧组合路径改善约 2% 的
`pipeline_ops=1600` 与原生 truth reduction 叠加后反而变慢 3.9%，因此保持默认关闭。

## 根因证据

原始 ACL 调用计数中，第一次 attention softmax 之前已有 7,493 个调用。对应的
Transformers masking code 在 `TransformGetItemToIndex` 中用四层嵌套 vmap 构造
22x22 causal mask；原兼容实现逐元素循环，产生 484 次标量函数求值。上下文限定的
广播向量化将整次 prefill 的 ACL 调用从 10,096 降至 2,736。

Qwen3-0.6B 的 28 层还触发 113 次 RMSNorm。原生 `aclnnRmsNorm` 替换无梯度分解后，
总调用进一步从 2,736 降至 1,493。小尺寸 GEMM 单独测试约 0.17 至 0.18 ms，且删除
入口 stream synchronize 的 A/B 仅改善约 5%，都不足以解释原来的数量级差距。

已有 `aclnnSilu` 也做了独立 A/B。它把每层 Sigmoid 和 Mul 合成一个调用，但 prefill
从 0.1206 s 变慢到 0.1237 s，generation 为 0.1354 s，因此该实验未纳入生产改动。

原始 RoPE 在 28 层中产生 112 次 Slice、56 次 Neg、56 次 Concat、112 次 Mul 和
56 次 Add。直接使用旧的 `aclnnApplyRotaryPosEmb` 不成立：该接口是原地更新接口，
既有 runner 还把 BNSD 的 Qwen 张量硬编码成 BSND，并返回未真实写入的输出。本次改用
CANN 9.0 的非原地 `aclnnRotaryPositionEmbedding`，mode 0 对应 Qwen 的 half rotation，
输出由独立张量承载。

SDPA 直连探针确认 CANN 的 bool causal mask 与 NumPy 参考最大误差低于 `3e-8`。
任意 additive bias 必须以 `[batch, query_heads, query_length, source_length]`
形状作为 `realShift` 传入，并使用 `pseType=0`；广播 mask 在调用前物化到该精确
形状。无 `realShift` 时必须改用 `pseType=1`。最终 Qwen3-0.6B 单 token 运行命中
`acl_flash_attention_score_v2` 700 次，8-token 运行命中 2,268 次，miss 均为空。

Transformers `_ignore_causal_mask_sdpa` 会检查 `padding_mask.all()`。原路径生成
`reduce.logical_and` 并触发一次 CPU fallback；第一版修复使用 nonzero、int32 min
和 bool cast 的公开组合。最新 profiler 显示该链在每个 decode step 重复提交，随后
改为 CANN 9 的 `aclnnAll`/`aclnnAny` runner。bool 输入直接归约，numeric 输入先做
一次 nonzero 比较；真实 NPU 的 full reduction、按维归约、负维度和 Qwen 整模均为
零 CPU compile/fallback。底层 Jittor core bool `all_`/`any_` OpInfo 仍是独立能力
边界，不由公开 ACL wrapper 推导为完整支持。

新增的 benchmark profile 在计时结束后单独执行一次 generation，共报告 69 个聚合
row。8-token 中算子时间合计约 223 ms，而整次墙钟约 550 ms；其中 matmul 1,576
次、transpose 896 次、FlashAttention 224 次、RMSNorm 904 次、RoPE 448 次。
因此剩余差距主要是每 token 约 4,100 个 Jittor IR op 的 Python/lazy graph 构建和
细粒度提交，不能仅靠继续减少 stream synchronize 消除。

cProfile 进一步记录到 7,604 次 Torch factory wrapper，其中 7,564 次来自 ACL
adapter 的 `jt.empty`。原完整 wrapper 累计 0.203 s，native-shape 快路径后降至
0.080 s，Python 调用数减少 29.1%。profile 中的 45 次 `Var.sync` 一度看似占用
0.151 s，但延迟所有 lazy CUDA residency 后生成了非法 token；只延迟无 copy 的
同设备转换也从约 0.55 s 变慢到 0.585 s。这些同步包含真实执行和跨 decode step
正确性边界，因此没有把删除同步的实验纳入实现。

第二步 decode 使用 `input_ids[:, cache_position]`。原 ACL wrapper 在
`GetItemACL` 拒绝该混合索引后执行 `x[slices]`，这会重新进入自身，造成 Python
单核 100%、NPU AICore 0% 的无限递归。新的受限 lowering 只接受单个 1-D integer
tensor index 且其他轴均为完整 slice，通过广播坐标复用 Index/IndexPut；独立真实
NPU 前向和反向均与 NumPy 完全一致。

## 正确性

优化后的 Jittor 和原生 PyTorch 分别保存完整的末 token logits。两者 shape 都是
`[1, 151936]`，数值全部 finite：

| Metric | Result |
| --- | ---: |
| Maximum absolute error | `3.8035214e-05` |
| Mean absolute error | `5.1964222e-06` |
| RMSE | `6.6203959e-06` |
| Cosine similarity | `0.9999999999977` |
| Argmax | both token 17 |
| Top-10 / Top-20 overlap | 10/10 and 20/20 |

两侧单步 greedy 输出均为 token 17、文本 `2`；8-token token ids 都是
`[17, 488, 220, 17, 284, 220, 19, 13]`，文本均为 `2 + 2 = 4.`。这证明性能变化
没有依赖错误 mask 或 CPU fallback 产生相同表面文本。

当前 10-sample 复验保留相同误差：最大绝对误差 `3.8035214e-05`、平均绝对误差
`5.1964222e-06`，argmax 与 Top-5/10/50 token 集合全部一致。新的 Jittor logits
还与优化前 Jittor 结果逐元素完全一致；native `empty` 快路径前后的 `.npy` 文件
SHA256 均为 `84ade73520acdc79db62415b89c296e633bbf75c6bbba5925e801cdd990de43c`。

最终源码稳定性复验中，首轮曾在第二次 generation 的 device sync 收到一次 CANN
`507018`；进程退出后以相同设备、缓存和参数连续执行两轮完整 3-warmup、10-sample
复验，均正常完成，中位延迟分别为 0.50007 s 和 0.50480 s。两轮 token、文本和
logits SHA256 均与上段一致且 `fallback_count=0`；异常未复现，继续作为环境稳定性
观察项，不据此改动同步语义。

ACL RoPE 候选与融合前 Jittor 的完整末 token logits 逐元素完全一致，最大绝对误差和
平均绝对误差均为 0。启用 ACL SDPA 后，相对原生 PyTorch SDPA 的完整 logits 指标
如上，argmax、top-10 和 top-20 token ids 全部一致。Qwen3-8B 严格 1-token SDPA
复验继续输出 token 19、文本 `4`，8,190,735,360 个参数驻留 NPU，命中 fused
attention 216 次，且 `cpu_compile_count=0`、`fallback_count=0`。

## 实现边界

- vmap 快路径只在 `TransformGetItemToIndex` 活跃时生效；普通 `torch.vmap` 保持原有
  循环语义。异常退出也恢复嵌套深度。
- getitem 不再提前同步，后继 slice/index 依赖通过真实 NPU 回归覆盖。
- ACL 混合索引快路径仅覆盖一个 1-D integer tensor index 和其余完整 slice；复杂
  advanced indexing 不在本改动中猜测轴重排语义。
- RMSNorm 只接受 ACL、CUDA flag、no-grad、合法末维 weight 和受支持 dtype；训练及
  其他形状继续走原有可微分实现。
- RoPE 融合只接受 ACL、CUDA flag、no-grad、4-D 广播兼容张量、相同受支持 dtype，
  以及已验证的 64 对齐 head dim；其他情况走可微组合实现。FP32、FP16、BF16 算子
  数值和非融合 FP32 梯度已在真实 NPU 验证。
- ACL SDPA 快路径只接受 ACL、CUDA flag、no-grad、dropout 0、4-D BNSD、FP32、
  8 对齐且不超过 256 的 head dim，以及已验证的 GQA、mask 和 causal 组合。FP16、
  BF16、训练、bool public mask 和矩形 causal 请求 fail closed 到原数学路径；不把
  CANN abort 当作半精度支持证据。
- float additive mask 只接受 FP32 rank-2/rank-4 可广播形状，并在调用前扩成 CANN
  要求的完整 query-head 形状。该路径保留语义，但长序列 mask 会承担物化成本。
- 公开 ACL `all`/`any` 在 CANN 9 上使用原生 truth-reduction runner；numeric 输入
  先比较 nonzero。底层 Jittor core bool `all_`/`any_` reduction 仍按 maintained
  OpInfo 精确 skip。
- `empty` 快路径只接受无关键字参数和 native 可直接消费的 int、tuple、list、
  NanoVector shape 或一组 Python int，并继续标记 extension mutable；Torch Size、
  NumPy/Var 维度、device、dtype 和 requires_grad 仍走完整兼容路径。
- Transformers 4.56.2 的 Qwen3 版本胶水通过外置 `jittor.module_patches` 适配接入
  `jt.nn.rotary_emb`，不把项目/版本特定 monkeypatch 放进 Jittor 核心。
- float16/float32 `arg_reduce` forward 通过 ACL MaxDim/MinDim 执行；generic backward
  仍因 index operation 不受支持而 fallback，不在本次 NPU 训练支持声明内。
- float32 Qwen3-0.6B 已验证。bfloat16 整模仍受 `KI-BACKEND-005` 限制，不在本报告
  的支持声明内。

## 验证

- Focused CPU Torch SDPA dispatch regressions: `3 passed`
- Real-NPU indexing, mixed slice/tensor-index forward/backward, and asynchronous
  dependency regression: `2 passed`
- Real-NPU float32/bfloat16 RMSNorm inference and training fallback: `1 passed`
- Real-NPU float32/float16/bfloat16 RoPE inference and composite gradient:
  `2 passed`
- Full current real-NPU ACL backend file: `26 passed`
- Real-NPU FP32 fused SDPA causal/decode/GQA/additive-mask reference and
  FP16/BF16 fail-closed boundary: `1 passed`
- Full Qwen3-0.6B SDPA logits parity: passed
- Fail-closed Qwen3-0.6B 8-token and Qwen3-8B 1-token generation:
  `cpu_compile_count=0`, `fallback_count=0`, fused SDPA hit
- Jittor and native PyTorch current single-token 7-sample and 8-token 5-sample
  synchronized benchmarks: passed
- Current real-NPU native `all`/`any`, SDPA, and asynchronous indexing focused
  regression: `13 passed, 103 deselected`
- Current Jittor/native `torch_npu` 8-token, 3-warmup, 10-sample synchronized
  comparison after the native `empty` fast path: `0.50309 s` / `0.49400 s`;
  logits parity passed
- Focused Torch constructor CPU regression: `23 passed, 2 skipped`
- Real-NPU native-shape `empty` residency regression: `3 passed`
- Full maintained NPU gate: `362 passed, 11 skipped`

可复现入口是
[`benchmark_qwen3_ascend.py`](../../skills/jittor-transformers-perf/scripts/benchmark_qwen3_ascend.py)。

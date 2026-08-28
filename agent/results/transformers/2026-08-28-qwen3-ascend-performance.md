# Qwen3-0.6B Ascend 推理性能诊断与优化

- Status: Accepted for float32 eager greedy inference on one Ascend 910B3
- Last reviewed: 2026-08-29
- Source baseline: `e3e82d98` plus the changes in this report's commit
- Owner: Torch compatibility and ACL backend maintainers
- Review when: Transformers masking, vmap compatibility, ACL RMSNorm, CANN,
  checkpoint, or timing protocol changes

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

## 环境与口径

- Device: one allocated Ascend 910B3
- Driver / CANN: 25.5.1 / 9.0.0
- Jittor: 1.3.11.0 from the current `2.0` source tree
- PyTorch / torch_npu: 2.10.0 / 2.10.0
- Transformers: 4.56.2 on both sides
- Model: local Qwen3-0.6B, 596,049,920 parameters
- Input: batch 1, 22-token chat prompt, eager attention, KV cache, float32
- Timing: current single-token rerun uses 2 warmups and 10 synchronized samples;
  8-token uses 2 warmups and 5 synchronized samples; median and p90 reported
- Isolation: separate processes, Python environments, JIT caches, homes, and
  temporary directories; raw logs remain under `$JITTOR_LAB_ROOT/_state/`

Checkpoint loading and first JIT are not mixed into steady-state timing. In the
current isolated benchmark cache, the first prefill took 121.72 s and first
generation took 40.63 s; these include cold operation compilation and are not
the per-request steady-state latency below.

## 性能结果

| Backend | Prefill median / p90 | Generate median / p90 | Relative to paired PyTorch |
| --- | ---: | ---: | ---: |
| Jittor before | 0.8191 s / 0.8228 s | 0.8139 s / 0.8162 s | 11.60x / 10.78x |
| Jittor optimized before ACL arg-reduce | 0.1226 s / 0.1243 s | 0.1409 s / 0.1414 s | 1.74x / 1.87x |
| Jittor with ACL arg-reduce | 0.1235 s / 0.1266 s | 0.1358 s / 0.1367 s | 1.77x / 1.84x |
| Native PyTorch NPU, current rerun | 0.0699 s / 0.0713 s | 0.0737 s / 0.0738 s | 1.00x |

The historical rows retain their original paired PyTorch ratios. The current
Jittor and PyTorch rows were rerun back-to-back on the same selected 910B3.

当前 prefill 可分为约 0.0820 s Python/graph/operator submission 和 0.0414 s
final synchronization。原生 PyTorch 总 prefill 为 0.0699 s，说明剩余差距仍主要与
Jittor 的 1,493 个细粒度调用及主机提交路径有关，不能归因成单个 GEMM 太慢。

| Backend | 8-token median / p90 | Throughput | Output |
| --- | ---: | ---: | --- |
| Jittor with ACL arg-reduce | 1.0845 s / 1.0874 s | 7.38 token/s | `2 + 2 = 4.` |
| Native PyTorch NPU, current rerun | 0.5843 s / 0.5899 s | 13.69 token/s | `2 + 2 = 4.` |

8-token 默认 Transformers 路径已不再卡死；Jittor 当前总延迟仍为原生 PyTorch 的
1.86 倍。这里报告整次 `generate` 延迟，不用总时间简单除以 8 伪装逐 token 首包延迟。

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
| Maximum absolute error | `3.8146973e-05` |
| Mean absolute error | `5.5334233e-06` |
| RMSE | `7.0331342e-06` |
| Cosine similarity | `0.9999999999974` |
| Argmax | both token 17 |
| Top-10 / Top-20 overlap | 10/10 and 20/20 |

两侧单步 greedy 输出均为 token 17、文本 `2`；8-token token ids 都是
`[17, 488, 220, 17, 284, 220, 19, 13]`，文本均为 `2 + 2 = 4.`。这证明性能变化
没有依赖错误 mask 或 CPU fallback 产生相同表面文本。

## 实现边界

- vmap 快路径只在 `TransformGetItemToIndex` 活跃时生效；普通 `torch.vmap` 保持原有
  循环语义。异常退出也恢复嵌套深度。
- getitem 不再提前同步，后继 slice/index 依赖通过真实 NPU 回归覆盖。
- ACL 混合索引快路径仅覆盖一个 1-D integer tensor index 和其余完整 slice；复杂
  advanced indexing 不在本改动中猜测轴重排语义。
- RMSNorm 只接受 ACL、CUDA flag、no-grad、合法末维 weight 和受支持 dtype；训练及
  其他形状继续走原有可微分实现。
- float16/float32 `arg_reduce` forward 通过 ACL MaxDim/MinDim 执行；generic backward
  仍因 index operation 不受支持而 fallback，不在本次 NPU 训练支持声明内。
- float32 Qwen3-0.6B 已验证。bfloat16 整模仍受 `KI-BACKEND-005` 限制，不在本报告
  的支持声明内。

## 验证

- CPU Torch compatibility regressions: `5 passed`
- Real-NPU indexing, mixed slice/tensor-index forward/backward, and asynchronous
  dependency regression: `2 passed`
- Real-NPU float32/bfloat16 RMSNorm inference and training fallback: `1 passed`
- Full Qwen3-0.6B logits parity: passed
- Fail-closed Qwen3-0.6B 8-token and Qwen3-8B 1-token generation:
  `cpu_compile_count=0`, `fallback_count=0`
- Jittor and native PyTorch current single-token 10-sample and 8-token 5-sample
  synchronized benchmarks: passed
- Full maintained NPU gate: `357 passed, 11 skipped`

可复现入口是
[`benchmark_qwen3_ascend.py`](../../skills/jittor-transformers-perf/scripts/benchmark_qwen3_ascend.py)。

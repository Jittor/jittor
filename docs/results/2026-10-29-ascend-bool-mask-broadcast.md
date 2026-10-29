# Ascend 上 boolean 注意力掩码的按 score 展开

- 状态：已验证（Ascend 910B3 + CANN，含修复前后对照）。
- 日期：2026-10-29
- 基线提交：`008cae211`；修复为 `21e7dfca2`。
- 验证范围：Ascend 910B3（CANN 9.0.0），单卡；Qwen3-0.6B bf16、2048 上下文
  prefill；对照 PyTorch torch_npu 2.10.0 的同一 workload。
- 维护者：Jittor 核心维护者
- 复查条件：`backends/acl/kernels/ops/flashattention_op.py` 的 bool 掩码分支、
  aclnn FlashAttentionScore 的 attenMask 形状约定、或 CANN 大版本变化。

## 现象

`benchmarks/torch_compat/run.py --device npu --size full` 的 `qwen3_prefill`
一步 54.99 ms，设备侧 55.07 ms（设备受限），而 PyTorch 的同一 workload 是
77.41 ms、**几乎全是 host**。设备 profile（msprof）里每步有 28 次
（= 层数）张量级算子：

| 算子 | 每步次数 | 每步耗时 | 单次 |
| --- | ---: | ---: | ---: |
| Cast | 28 | 11.36 ms | 406 µs |
| BroadcastTo | 28 | 5.79 ms | 207 µs |
| LogicalNot | 28 | 2.31 ms | 83 µs |

真正做计算的 `FlashAttentionScore` 是 9.82 ms、`MatMulV2` 8.20 ms，说明多出来的
19.5 ms 不是注意力本身。

## 根因

`scaled_dot_product_attention_acl` 对 bool 掩码的处理是
`reshape → broadcast([b, n, s, s]) → logical_not`：

- Torch 的 bool 掩码是 `[1, 1, s, s]`（真 = 保留），CANN 的 attenMask 是“丢弃”
  语义，所以必须取反；
- 但展开到 `[b, n, s, s]` 在 2048 上下文、16 头时是 64 MiB，每层一次；
- ACL 后端上这次展开还会带出两个与 score 同尺寸的 Cast（复现：一次调用固定
  2 个 Cast + 1 个 BroadcastTo + 1 个 LogicalNot，尺寸随 score 线性放大）。

对照实验（同一 provider、bf16、因果 bool 掩码，每次调用均值）：

| 输入 | 附加算子 |
| --- | --- |
| `[1,1,S,S]`，provider 内部广播（修前） | Cast ×2 + BroadcastTo + LogicalNot |
| `[1,16,S,S]`，调用方已广播 | 只剩 LogicalNot |
| `[1,1,S,S]` 直接交给 `FlashAttentionACL` | **无** |

即 aclnn 自己会按 batch/head 广播掩码，展开是多余的。

## 修复

bool 分支只在小掩码上取反；只有 query/key 轴仍需要扩展时才广播。float 的
“pse” 分支保持原样（aclnn 没有同样的承诺）。

## 结果

掩码语义在 910B3 上与 float64 参考对拍（bf16 量化级误差）：

| 用例 | max\|diff\| |
| --- | ---: |
| `[1,1,S,S]`，batch 1 | 0.0074 |
| `[1,1,S,S]`，batch 2 | 0.0075 |
| `[B,1,S,S]`，batch 2 | 0.0074 |
| `[1,1,1,S]`（key 轴扩展，走旧广播） | 0.0040 |

`qwen3_prefill`（同一张卡、同一缓存）：

| 指标 | 修前 | 修后 |
| --- | ---: | ---: |
| 墙钟一步 | 54.99 ms | **35.69 ms** |
| 设备一步 | 55.07 ms | **34.64 ms** |
| 对 PyTorch（77.41 ms） | 0.710 | **0.461** |
| Cast / 步 | 11.36 ms | 0.013 ms |
| BroadcastTo（掩码）/ 步 | 5.79 ms | 0（其余是 10 µs 级小算子，合计 1.43 ms） |
| LogicalNot / 步 | 2.31 ms | 0.310 ms |

设备侧已经低于 PyTorch 的 38.94 ms；PyTorch 的整步耗时仍被 host 主导。

## 复现

```bash
# 910B3 上，Jittor 侧
python benchmarks/torch_compat/run.py --device npu --gpu 6 --size full \
  --runtimes jittor --workloads qwen3_prefill
# 对照
python benchmarks/torch_compat/run.py --device npu --gpu 6 --size full \
  --runtimes torch --workloads qwen3_prefill
```

设备侧算子表由 msprof 采集，或在本机用 `jt.profile(device=True)`（需 CUPTI）。

## 回归覆盖

- ACL 全量：1 failed / 440 passed / 11 skipped；唯一失败是
  `test_acl_indexing.py::test_acl_rfft_keeps_lazy_dft_constants_alive`
  （`unary/imag` 未注册，见 KI-COMPLEX-001），与本改动无关。
- `tests/structure/backends/acl/`：295 passed。

## 未结

- `qwen3_decode` 两侧都 ~100% host（每步约 15 万个算子），比值 1.065 在漂移带内；
  要收窄只能减少每 token 的算子数或每算子开销，不是设备问题。
- `is_causal=True` 的快路径（45.9 µs 对 54.5 µs）需要“掩码恰为因果掩码”的判定，
  有语义风险，暂不做。
- `qwen3_train`（422.6 ms，host 主导，比值 1.190）不再需要逐项定位：优化器写回改成
  每组只解析一次读者后，配对测得 412 → **265 ms**（对 torch 357 约 0.74），稳态显存
  26.9 → 28.92 GB；把参数拼成扁平缓冲的方案测得 264–270 ms 但多 6.1 GB 显存，因此
  不合并。

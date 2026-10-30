# CUDA float32 注意力的行和与模块 zero_grad：两处训练路径修复

- 状态：已验证（CUDA 实机，含修复前后对照）。待 910B 上的等价改动另记。
- 日期：2026-10-29
- 基线提交：`008cae211`；本报告的两处修改为 `b0dcb366d`（注意力行和）与
  `d2bd0de56`（`zero_grad` 合并同步）。
- 验证范围：RTX 4090 24 GB，CUDA 12.2.140 + `jittor[cuda12]` 组件轮子
  （`pipcu122`），对照 PyTorch 2.11.0+cu128；两侧同一张卡、同一份 Transformers
  包、同一 TF32 策略。
- 维护者：Jittor 核心维护者
- 复查条件：`backends/cuda/kernels/nn/fused_attention_f32_cuda.py` 的行和逻辑、
  `compat/torch/installers/nn/module_methods.py` 的 `zero_grad`、或 float32
  SDPA 的分派条件变化；换卡或换 PyTorch 版本

## 现象

`compat/tests/torch/test_ecosystem_speed.py` 的 CUDA 大模型用例里，BERT-large
（12 层、hidden 768、8×256）前反向一步是 `torch 27.4 ms / jittor 179.2 ms`
（6.55x），而 convnet/llama 等在 1.0x 附近。分段计时把它定位到反向：

| 段 | Jittor | PyTorch |
| --- | ---: | ---: |
| 前向构图 + 执行 | 10.7 + 5.3 ms | 12.3 ms |
| 反向构图 + 执行 | 15.5 + **148.1 ms** | 25.2 ms |
| `zero_grad(set_to_none=False)` | **11.9 ms** | 1.5 ms |

设备侧 profile（CUPTI，`JITTOR_CUPTI_LIB` 指向 12.8 的 `libcupti.so.12`）显示
12 次调用、每次 11.3 ms 的 `mea::row_delta`：float32 记忆高效注意力的反向在**每一
层**重走一遍 `lq×lk×d` 的 score 矩阵，只为重建一个前向已经算过的量。

## 根因

反向里每个 query 行的 `sum_k p_k·dP_k` 与 `dO·O` 是同一个数，`O` 是小
`[b,h,lq,d]` 张量。重建只在“该行概率几乎压在一个 key 上”时才有意义：那时
`O` 的 float32 舍入会把 Q/K 梯度整体吃掉（`4c722fee0` 为此引入的精确路径）。
普通分布下重建的代价是每层 `2·lq·lk·d` 次读写，2048 上下文时是 GB 级流量。

`Module.zero_grad(set_to_none=False)` 同理：每个参数的 `zero_` 都先做一次
`sync_all` 再原地写，199 个参数就是 199 次全量同步，把设备在写之间反复抽干。

## 修复

- 行和：先算 `dO·O` 作为默认值；`_row_delta` 只对
  `1/sum_k exp(score_k-max) > 0.5`（概率质量摊在不到两个 key 上）的行做精确重建，
  其余行直接取默认值。阈值是模块常量 `_RECOMPUTE_INV_ABOVE`。饱和行与普通行混在
  同一次 launch 里，逐行判定。
- `zero_grad`：先收集所有 `.grad`，用一次 `sync_all` 解析读者，再对整批做原地写；
  单张量路径仍负责延迟捕获、未落盘张量、可微张量等特例。

## 结果

`compat/tests/torch/test_ecosystem_speed.py -k EcosystemSpeedCUDA`（同一张卡，
`JITTOR_ECOSYSTEM_LARGE=1`，比值 = Jittor / PyTorch）：

| 用例 | PyTorch | Jittor | 比值 |
| --- | ---: | ---: | ---: |
| large_convnet | 17.3 ms | 17.7 ms | 1.02x |
| large_transformers_bert | 27.5 ms | 34.6 ms | **1.26x**（修前 6.55x） |
| large_transformers_gpt2 | 43.3 ms | 37.7 ms | 0.87x |
| large_transformers_llama | 39.3 ms | 38.8 ms | 0.99x |
| large_transformers_qwen3 | 48.1 ms | 38.5 ms | 0.80x |
| large_transformers_vit | 23.8 ms | 30.4 ms | **1.28x**（修前 1.67x） |
| large_diffusers_unet2d | 26.0 ms | 44.3 ms | **1.71x** |

分步效果（BERT-large，4090）：一步 186 ms → 注意力修复后 50 ms → 合并
`zero_grad` 后 41 ms；反向执行 148 ms → 12 ms；`zero_grad` 11.4 ms → 3.4 ms。
ViT 的注意力同样走 float32 融合核，1.67x → 1.28x 来自同两处修改。表中这两行测于
优化器改动之前；本页末尾的优化器与 `order_after_readers` 修复把 `vit_b16_train`
进一步降到 0.73x、`qwen3_train` 到 0.66x。
`large_diffusers_unet2d` 用的是 `diffusers-installed-source/20261015` 的包站点；
它的差距不是某一处内核：设备侧 20.9 ms 与 PyTorch 接近，1787 个算子里有 63%
的墙钟花在主机侧构图与启动上（`cudnn_conv*` 51/50/51 次、GroupNorm 的 `code` 90 次
都不是异常项）。

## 附带修复：bf16 掩码 prefill 走因果路径

`qwen3_prefill` 的 SDPA 收到的是 Transformers 构造的 `[1,1,2048,2048]` bool 因果
掩码；带掩码时融合核只能把它当 score 尺寸的偏置处理，一步 88.4 ms（PyTorch 36.4，
2.43x）。`jittor.nn.functional.scaled_dot_product_attention` 现在先逐元素比对掩码
是否就是 `tril(ones)`，是则按 `is_causal=True` 调用（两者语义完全相同），两次
测量为 **88.4 → 30.2 / 24.7 ms，比值 2.43x → 0.83x / 0.68x**。非因果掩码
（padding 等）不受影响，仍走原路径。独立对拍里同一输入的两条路径在 2048 上输出
逐位相同（max abs 0.0）。

同一次排查还发现：若一个环境里 cuDNN 与 `nvidia-cudnn-frontend` 主版本不匹配
（例如 cuDNN 8.9 配 frontend 1.30），融合核的支撑性探测对所有形状都返回否，注意力
静默退回 matmul 组合。现在 `_library()` 首次加载时会用一个必然受支持的形状自检，
失败则告警一次。

## 附带修复：优化器写回每组只解析一次读者

每个原地写回都会先 `sync_all` 解析目标存储的读者；AdamW 每个参数写三次，Qwen3 的
一步就是几百次全量同步。优化器更新里一个参数的读者都在它新值的上游，所以改成
每组参数开始前解析一次、每个写回只把自己的源算出来。4090 上 `qwen3_train`
212.2 → **177.7 ms**（对 PyTorch 267.8 为 0.66x）；910B3 上由协作者配对测得
412 → **265 ms**（对 torch 357 为 0.74），稳态显存 26.9 → 28.92 GB。同一条路径上
"把参数拼成扁平缓冲"的方案测得 264–270 ms 但要多 6.1 GB 显存，因此不合并。

同一轮还把原地写回的控制边构建 `order_after_readers` 里的线性查重换成了哈希集合
（`src/core/op.cc`；向量仍保留插入顺序，控制边不变）。这一步对逐参数路径同样有效：
4090 上 `vit_b16_train` 与上面两项合计 266.8 → **141.5 ms**（对 PyTorch 194.4
为 0.73x，此前是 1.67x）；910B3 上这棵树的一步是 174.2–174.7 ms。

## 真实模型矩阵（`benchmarks/torch_compat`，14 个任务）

同一张 4090、两侧配对、热缓存、无其它租户负载，提交 `86fa5dc73`：

| 任务 | PyTorch | Jittor | 比值 |
| --- | ---: | ---: | ---: |
| qwen3_decode_static | 2714.9 ms | 542.5 ms | **0.20x** |
| resnet50_infer_b1 | 3.2 ms | 0.6 ms | **0.18x** |
| bert_base_infer | 3.1 ms | 1.2 ms | 0.38x |
| sd15_unet_train | 314.6 ms | 197.5 ms | 0.63x |
| sd15_vae_decode | 33.2 ms | 21.1 ms | 0.64x |
| qwen3_decode | 2122.4 ms | 1389.6 ms | 0.65x |
| qwen3_prefill | 36.8 ms | 24.4 ms | 0.66x |
| qwen3_train | 267.8 ms | 177.7 ms | 0.66x |
| resnet50_infer | 14.1 ms | 9.4 ms | 0.67x |
| bert_base_train | 66.1 ms | 57.3 ms | 0.87x |
| vit_b16_train | 193.2 ms | 172.0 ms | 0.89x |
| sd15_sample | 441.4 ms | 400.5 ms | 0.91x |
| resnet50_train | 91.7 ms | 85.5 ms | 0.93x |
| ddpm_unet_train | 75.4 ms | 74.4 ms | 0.99x |

几何平均 **0.59x**（同一提交、同一张卡的第二次全量跑为 0.62x；Jittor 侧的绝对值两次
一致，PyTorch 侧在 159–193 ms 之间漂移约 20%，所以单行比值以同一次配对为准）。
14 个任务里只有 `ddpm_unet_train` 在第二次跑中读到 1.03x，其余均不慢于 PyTorch；
`vit_b16_train`/`bert_base_train`/`ddpm_unet_train` 在空闲卡上重新配对为 0.88x /
0.86x / 1.03x。`qwen3_decode` 与 `qwen3_decode_static` 的 agree 列分别是
9.5e-02 / 1.2e-01（harness 在 5e-2 以上打标记）：这两行比较的是生成的 token id，
随机权重下贪心解码的 near-tie 会翻转个别 token，序列长度两侧都是 136，不是长度
漂移。

## 复现

```bash
# Jittor 侧用 jittor[cuda12] 组件轮子的解释器；对照用真实 PyTorch 的解释器
JITTOR_HOME=/path/to/jittor-home \
JITTOR_TORCH_SHIM=1 JITTOR_ECOSYSTEM_LARGE=1 \
REAL_TORCH_PYTHON=/path/to/torch/bin/python \
JITTOR_ECOSYSTEM_PACKAGE_SITE=/path/to/transformers-site \
python -m pytest -q compat/tests/torch/test_ecosystem_speed.py -s \
  -k 'EcosystemSpeedCUDA and large_transformers_bert'
```

CUDA 设备侧 profile 需要 CUPTI：`JITTOR_CUPTI_LIB=<...>/nvidia/cuda_cupti/lib/libcupti.so.12`，
随后 `with jt.profile(device=True): ...`。

## 回归覆盖

- `tests/backends/cuda/test_fused_attention_f32.py`：25 条，含新增的“饱和行与普通行
  混在同一次 launch”用例；饱和行、常量 V 两条原有实验保持通过。
- `compat/tests/torch/test_torch_compat_attention.py`：48 passed / 38 skipped
  （跳过项为未配置 flash-attn 源码）。
- `compat/tests/torch/test_torch_module_method_owner_fidelity.py`、
  `test_optimizer_storage_updates.py`、`test_tensor_inplace_storage.py`、
  `test_torch_compat_optim.py`：177 passed。
- `tests/structure`：1571 passed / 4 skipped；唯一失败是 `test_asv_entrypoints`
  所在解释器没有装 `asv`，与本改动无关。
- 生态 parity（`test_ecosystem_parity.py -k EcosystemParityCUDA`）：10 passed /
  6 skipped / 1 failed。唯一失败是 `transformers_t5` 的一个解码器 MLP 梯度
  3.337e-02（带宽 2e-2），在 `008cae211` 上数值完全相同，属既有问题，记在
  KI-COMPAT-021。

## 未结

- diffusers UNet 的 1.71x 由主机侧构图主导，不是内核；要收窄得减少每步的算子数或
  每个算子的 Python 开销。
- BERT/ViT 剩余的 1.26–1.28x 同样由主机侧构图主导（每 op 约 5–7 µs，PyTorch 约
  1–2 µs）。

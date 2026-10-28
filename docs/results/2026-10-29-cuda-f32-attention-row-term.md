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

分步效果（BERT-large，4090）：一步 186 ms → 注意力修复后 50 ms → 合并
`zero_grad` 后 41 ms；反向执行 148 ms → 12 ms；`zero_grad` 11.4 ms → 3.4 ms。
ViT 的注意力同样走 float32 融合核，1.67x → 1.28x 来自同两处修改。`large_diffusers_unet2d`
因对照站点缺 diffusers 未跑。

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

## 未结

- diffusers UNet 的 CUDA 大模型用例尚未在对照站点补上 diffusers 后测量。
- BERT/ViT 剩余的 1.26–1.28x 已由主机侧构图主导（每 op 约 5–7 µs，PyTorch 约
  1–2 µs），不是某一处内核的问题。

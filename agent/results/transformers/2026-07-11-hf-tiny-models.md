# Transformers 4.56.2 Tiny 模型性能对比（2026-07-11）

## 结论

在同一份 jt311 `transformers==4.56.2` 源码、RTX 4090 GPU2、fp32/TF32-on/SDPA 口径下：

| 模型 | phase | Jittor | 真 PyTorch | Jittor / PyTorch |
| --- | --- | ---: | ---: | ---: |
| Tiny Llama | forward | 1.9222 ms | 2.0645 ms | **0.93x** |
| Tiny BERT | forward | 1.4989 ms | 1.0696 ms | **1.40x** |
| Tiny ViT | forward | 1.7640 ms | 0.9561 ms | **1.85x** |
| Tiny Llama | forward + backward | 3.2109 ms | 9.7066 ms | **0.33x** |

Llama forward 已持平略快，说明当前差距不是统一的框架常数开销；BERT 和 ViT 仍分别慢约 40% 和 85%。Tiny Llama 的反向优势来自 Jittor 统一图/融合在小模型、多小算子场景下对 eager launch 开销的摊薄，但本项不含 optimizer、grad clip、scaler 或参数更新，不能外推为大模型完整训练快 3 倍。

## 同版本隔离

- Jittor：`/home/zy/miniconda3/envs/jt311/bin/python`，脚本显式 `import jittor as torch`，源码为当前仓库 `python/jittor`。
- PyTorch：`/home/zy/rt_venv/bin/python`，真 `torch 2.12.1+cu130`。
- 两端 Transformers 均断言来自 `/home/zy/miniconda3/envs/jt311/lib/python3.11/site-packages/transformers/__init__.py`，版本均为 `4.56.2`。`rt_venv` 自带的 Transformers `5.12.1` 没有参与测试。
- PyTorch 端先加载并锁定 `rt_venv` 的真 torch/torchvision，再把 jt311 site-packages 置顶加载 Transformers，避免 jt311 的 Jittor torchvision 污染 oracle。
- 无网络下载，无预训练权重，无新增依赖。

## 模型和计时口径

三者均为 eval、dropout=0、`_attn_implementation="sdpa"`：

| 模型 | 配置 | 参数量 | 输出形状 |
| --- | --- | ---: | --- |
| LlamaModel | 2 层，H=256，FFN=768，8 heads/4 KV heads，B=2，S=128 | 2,098,432 | `[2,128,256]` |
| BertModel | 2 层，H=256，FFN=1024，8 heads，B=2，S=128 | 2,170,368 | `[2,128,256]` |
| ViTModel | 2 层，H=256，FFN=1024，8 heads，64x64/patch8，B=2 | 1,646,336 | `[2,65,256]` |

- forward：20 个不同的预分配输入槽；计时前每槽完整 warm 一次，计时内保留 20 个输出，末尾设备同步。Jittor lazy 图因此不会只执行最后一次输出。
- Llama forward+backward：8 个不同输入槽，每槽 warm 一次；随机上游梯度，`autograd.grad` 返回并保留 20 组参数梯度，末尾同步。两端 20/20 梯度均 finite 且非零。
- JIT/首次模型编译、输出转 NumPy和正确性检查均在计时区间外。
- PyTorch 模型保持 eager 执行（未使用 `torch.compile`），SDPA 由 default dispatcher 自行选择优化 kernel；Jittor 当前条件走自身 SDPA 路径。该口径代表相同 Transformers 代码的默认实际表现，不是强制相同底层 attention kernel 的 microbenchmark。
- 两端独立随机初始化；本轮只验证版本、参数量、输出形状和 finite，不是数值精度对拍。

## 显存

脚本记录的 `max_memory_allocated`：

| 模型 / phase | Jittor | PyTorch | Jittor / PyTorch |
| --- | ---: | ---: | ---: |
| Llama forward | 20.00 MiB | 25.56 MiB | 0.78x |
| BERT forward | 20.00 MiB | 24.24 MiB | 0.83x |
| ViT forward | 14.00 MiB | 20.69 MiB | 0.68x |
| Llama forward+backward | 100.00 MiB | 108.37 MiB | 0.92x |

Jittor 的 allocator 统计在计时前后相同，而 PyTorch 会在执行中更新 peak；因此这些数字只能作为本 harness 的工作集比较，不能替代进程级 NVML 峰值。

## 瓶颈判断

- Llama 使用 RMSNorm、RoPE、SwiGLU/GQA；当前 Jittor 的图融合和 matmul 路径已能抵消 SDPA fallback、小算子调度等成本，forward 接近 PyTorch。
- BERT 的 LayerNorm、GELU、残差和 attention 小算子占比更高，1.40x 差距与既有 LayerNorm/GELU/softmax microbenchmark 的热点一致。
- ViT 的 token 数只有 65，除 LayerNorm/GELU/attention 外还包含 patch Conv2d；计算规模更小后，kernel launch、建图/dispatch 和未融合边界占比最高，形成 1.85x 最大差距。
- Tiny Llama backward 的 0.33x 是统一图融合的正面信号，但需要在含 optimizer 的真实 train step、更大 hidden/sequence 和多轮复测中确认；不能仅凭本结果改变整体 G4 结论。

## 使用方法

先验证版本和输出形状：

```bash
CUDA_VISIBLE_DEVICES=2 cache_name=hf_tiny_gpu2 \
  agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/jittor-transformers-perf/scripts/benchmark_hf_tiny_models.py \
  --backend jittor --model llama --smoke-only

CUDA_VISIBLE_DEVICES=2 cache_name=hf_tiny_torch_gpu2 \
  agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/rt_venv/bin/python \
  agent/skills/jittor-transformers-perf/scripts/benchmark_hf_tiny_models.py \
  --backend torch --model llama --smoke-only
```

正式 forward 或 forward+backward：

```bash
CUDA_VISIBLE_DEVICES=2 cache_name=hf_tiny_gpu2 \
  agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/jittor-transformers-perf/scripts/benchmark_hf_tiny_models.py \
  --backend jittor --model bert --phase forward --repeats 20 \
  --jsonl ${JITTOR_LAB_ROOT}/jittor_transformers_perf/results/hf_tiny.jsonl

CUDA_VISIBLE_DEVICES=2 cache_name=hf_tiny_torch_gpu2 \
  agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/rt_venv/bin/python \
  agent/skills/jittor-transformers-perf/scripts/benchmark_hf_tiny_models.py \
  --backend torch --model llama --phase fwd_bwd --repeats 8 \
  --jsonl ${JITTOR_LAB_ROOT}/jittor_transformers_perf/results/hf_tiny.jsonl
```

## 产物与缺失

- 脚本：`benchmark_hf_tiny_models.py`。
- smoke：`results/hf_tiny_smoke_{jittor,torch}_gpu2_20260711.jsonl`。
- forward：`results/hf_tiny_forward_{jittor,torch}_gpu2_20260711.jsonl`。
- Llama 训练：`results/hf_tiny_train_{jittor,torch}_gpu2_20260711.jsonl`。
- 未完成：BERT/ViT backward、完整 optimizer step、fp16/bf16、NPU、同权重逐层精度对拍、重复进程统计。

环境排障记录：初版 PyTorch 隔离仅 pin 真 torch，随后 jt311 Transformers 经 `image_utils` 导入了 jt311 的 Jittor torchvision；脚本现同时 pin `rt_venv` 的真 torchvision。Jittor 直接使用 `autograd.grad` 且不构造 optimizer 时需要显式激活参数 leaf，脚本在 `fwd_bwd` 模式对两端统一执行 `requires_grad_(True)`。

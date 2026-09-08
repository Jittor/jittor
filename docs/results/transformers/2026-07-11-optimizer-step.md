# Jittor optimizer-step 热路径量化（GPU3）

日期：2026-07-11
状态：✅ 真实 SGD/AdamW 集成、完整 step 与分段计时完成；未修改核心源码，未执行 git。

## 环境与方法

- Python：`/home/zy/miniconda3/envs/jt311/bin/python`。
- GPU：`CUDA_VISIBLE_DEVICES=3`，RTX 4090。
- 当前共享树：`torch_compat.py` mtime `2026-07-11 03:45:58`。
- 探针：`agent/skills/jittor-transformers-perf/scripts/benchmark_optimizer_step.py`。
- 原始结果：
  - `results/optimizer_step_gpu3_20260711.jsonl`（首次 5 repeats）
  - `results/optimizer_step_gpu3_rerun_20260711.jsonl`（热缓存 7 repeats，最终数据）

使用真实 `torch.optim.SGD(lr=1e-3, momentum=0.9, weight_decay=0.01)` 与 `torch.optim.AdamW(lr=1e-3, weight_decay=0.01)`。梯度已就绪，排除 backward 和每轮梯度重置，只计梯度管理与 optimizer update。

- 总梯度元素固定为 `262144`：128 张量时每张量 2048 元素，512 张量时每张量 512 元素。
- dtype 为 float32，避免 fp16 AdamW 数值路径干扰 launch/图调度开销判断。
- `none`：直接 `optimizer.step()`。
- `clip`：当前 `_clip_grad_norm_device(..., max_norm=1, p=2)` 后 step。输入全局 norm 为 512，确保确实发生裁剪。
- `scaler`：梯度预乘 2，当前 `GradScaler(init_scale=2).step()` + `update()`，包含 unscale、finite reduction、一次 host overflow 判定和真实 optimizer step。
- 完整计时只在末尾 `sync_all(True)`；分段计时在 management 后和 optimizer 后分别同步。最终采用 warmup=3、repeats=7 的中位数，规避少量周期性 outlier。

## 完整 step 结果

单位 ms，括号为相对同 optimizer、同张量数的 `none` 增量。

| optimizer | 张量数 | 无 clip | 当前 clip + step | GradScaler + step |
|---|---:|---:|---:|---:|
| SGD | 128 | 3.245 | 5.448（+67.9%） | 5.814（+79.2%） |
| SGD | 512 | 13.782 | 21.465（+55.7%） | 23.164（+68.1%） |
| AdamW | 128 | 7.937 | 10.248（+29.1%） | 10.163（+28.0%） |
| AdamW | 512 | 30.696 | 39.190（+27.7%） | 40.222（+31.0%） |

所有路径参数 sample checksum 有限；GradScaler 路径与无管理路径 checksum 一致，说明 scale=2 的梯度在 step 前正确还原。clip 路径使用主动裁剪，不能与无 clip 比较最终参数值。

## 分段占比

下表为 management 与随后 optimizer update 各自强制同步的中位数。占比按两段中位数之和计算。

| optimizer | 张量数 | 管理方式 | management (ms) | update (ms) | management 占比 |
|---|---:|---|---:|---:|---:|
| SGD | 128 | clip | 2.380 | 3.482 | 40.6% |
| SGD | 128 | scaler | 2.339 | 3.520 | 39.9% |
| SGD | 512 | clip | 8.979 | 14.254 | 38.6% |
| SGD | 512 | scaler | 9.380 | 14.099 | 39.9% |
| AdamW | 128 | clip | 2.455 | 7.835 | 23.9% |
| AdamW | 128 | scaler | 2.392 | 7.815 | 23.4% |
| AdamW | 512 | clip | 9.254 | 31.495 | 22.7% |
| AdamW | 512 | scaler | 9.541 | 31.575 | 23.2% |

结论：继续优化 clip/scaler 对 SGD 完整热路径有约 40% 的分段空间，对 AdamW 有约 23%。但 optimizer update 本身仍占 60%-77%，不能只优化梯度管理。

## 图融合与同步

当前 clip 不做常规 `.item()`，完整路径可以把 device 上的 coefficient 与 optimizer update 留在同一 lazy 图。比较“末尾统一同步”与“两段分别同步”：

- SGD clip：128/512 张量完整路径比两段之和低约 `7.1%/7.6%`。
- AdamW clip：128/512 张量低约 `0.4%/3.8%`。
- GradScaler 因 overflow 决策需要 `.item()`，management 内部已有 host 边界；完整与分段中位数只差约 `0.4%-2.2%`。

因此 device-only clip coefficient 的价值在完整 step 中确实存在，不能用孤立 helper benchmark 完全评价。但 GradScaler 当前无法跨 finite/overflow host 判定与 update 融合。

## 底层瓶颈判断

总元素不变时，张量数从 128 增到 512：

- 无管理 SGD 从 `3.245` 境长到 `13.782 ms`，为 `4.25x`。
- 无管理 AdamW 从 `7.937` 增长到 `30.696 ms`，为 `3.87x`。
- clip/scaler management 从约 `2.4 ms` 增长到约 `9.0-9.5 ms`，也接近 4 倍。

这说明当前主要随 **tensor count** 而不是总元素数扩展。源码中 SGD/AdamW 和梯度管理都按 parameter Var 构建独立 `update()` 子图；小 tensor 多时，Python 图构建、executor 节点和 kernel launch 主导。

AdamW 无管理 step 在 512 张量时仍需 `30.7 ms`，显著大于 management 的 `9.3-9.5 ms`。所以最高优先级不是继续微调 Python concat 表达式，而是：

1. 原生 foreach/multi-tensor SGD、AdamW，把参数、momentum/exp_avg/exp_avg_sq 更新合并到少量 CUDA kernel。
2. 把 unscale、nonfinite reduction、clip coefficient 应用融合进 multi-tensor optimizer。GradScaler 可保留一次必要的 overflow host 决策，但不要为每个参数创建独立写回图。
3. 对无 overflow 的常见路径，评估 device flag/条件化 optimizer launch，或至少在单次 D2H 后调用 fused optimizer，而不是回到 N 个 Var update。
4. benchmark 与 profiler 应同时按“总元素”和“tensor 数”分桶；只测大 flat tensor 会漏掉 Transformers 参数碎片化的主要成本。

## 限制

- 本次只测 CUDA GPU3、float32、均匀 tensor 尺寸和单 param group；未覆盖 fp16/bf16、稀疏梯度、多个学习率组或 NPU。
- GradScaler 的 dtype 选择用于量化其图/同步结构，不代表 fp16 数值吞吐；真实混合精度还需要单独验证 optimizer state dtype。
- 计时从“梯度已就绪”开始，不包含 backward、DDP/FSDP reduction、zero_grad 或数据加载。
- 本报告采用分段 wall-clock 计时而非开启 Jittor profiler，以避免 profiler instrumentation 改变数百个小 update op 的绝对耗时；两次独立运行趋势一致。

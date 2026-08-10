# `clip_grad_norm_` CUDA 性能消融与实现建议

状态：✅ GPU5 独立缓存实测完成；✅ 退化根因定位；✅ 当前 flat/device
实现复验；本审计未修改核心源码、未执行 git 操作。

## 1. 结论

1. 用户观察到的变慢是真实的，但退化源不是“device-only”本身，而是上一版
   **per-gradient device reduction**：每个参数分别做 reduce + sqrt，再归约这些标量，
   大模型数百个参数 tensor 时产生 `O(count)` 个小 reduction kernel。固定总元素
   `262144` 时，旧 per-grad device 在 `count=128/512` 分别比原始
   concat+`.item()` 慢 `25.4%/43.5%`。
2. 当前正式 helper 已改成 **flat reduction + device coefficient**。同一口径下，真实发生
   裁剪（`max_norm=1`）时，当前实现相对原始 `.item()` 版本：
   - `262144` 元素、`count=1/16/128/512`：分别快
     `3.06x/2.05x/1.29x/1.26x`；
   - `4194304` 元素：分别快 `6.27x/2.88x/1.52x/1.31x`。
   因此当前 flat/device 方案已消除本轮回归，并在常见真实裁剪路径上有净收益。
3. flat/device 有两个明确代价：
   - 逻辑上构造一个总参数量大小的 flat 临时量，fp32 临时空间为 `4*N` bytes；本轮两档
     分别是 `1 MiB` 和 `16 MiB`，超大模型可能上升到 GiB 级。
   - 为保持零 D2H，同一个 device `coef` 会无条件写回每个 gradient。即使高有限阈值使
     `coef==1`，仍执行 scale。`count=512` 的 no-clip 情况下，当前实现比旧 host-branch
     版本慢 `14.5%`（262K）和 `9.2%`（4M）。
4. 当前剩余第一瓶颈已不是 norm，而是**逐 tensor scale 写回**。`count=512` 时，单独 scale
   约 `3.60-3.64 ms`，且 profiler 显示每个 tensor 各有 multiply + setitem 两个动态
   kernel 调用。下一步应把 `coef` 融入 optimizer update，而不是重新引入 `.item()`。

## 2. 环境与口径

- GPU：NVIDIA RTX 4090，仅 `CUDA_VISIBLE_DEVICES=5`。
- Python：`conda jt311`。
- CUDA：12.2，sm89。
- 独立缓存：`cache_name=kernel_audit_clip_gpu5`。
- HOME、Jittor JIT、CUDA cache 均位于
  `${JITTOR_LAB_ROOT}/jittor_transformers_perf/runtime/kernel_audit_clip_gpu5/`。
- 工具：`agent/skills/jittor-transformers-perf/scripts/probe_clip_grad_norm.py`。
- 原始数据：
  - `results/clip_audit_gpu5_matrix.jsonl`
  - `results/clip_audit_gpu5_profiles.jsonl`

所有 timed gradients 在计时前都执行一次 CUDA `update(+0)` 并同步，确保已经 device-resident。
否则 `jt.array(numpy)` 的首次使用会把 H2D page migration 算进被测实现，不能代表 optimizer
热路径。

每个 timed slot 的计时拆为：

- `call_build_ms`：进入实现到函数返回。device-only 模式主要是 Python/图构造；旧实现的
  `.item()` 在函数内部，因此此项还包含 norm 执行、D2H 和 host 同步。
- `final_sync_ms`：函数返回后把 total 与全部 gradients 同步完成。
- `total_ms`：二者之和。

矩阵为 warmup 1、独立输入 3 repeats。绝对值会受主机状态影响，结论主要看同进程、同
device-resident 口径下的模式差值。

## 3. 被测方案

| 名称 | norm | 系数判断 | scale |
|---|---|---|---|
| `old` | concat 全梯度后单 reduction | `total.item()` + Python branch | 仅实际 clip 时逐 tensor update |
| `per_grad_device` | 每 tensor reduce+sqrt，再对标量 reduce | device minimum | 每次逐 tensor update |
| `sum_device` | 每 tensor 只求 `sum(g^2)`，最后一次 sqrt | device minimum | 每次逐 tensor update |
| `current` | concat 全梯度后单 reduction | device minimum | 每次逐 tensor update |
| `norm_*` | 仅对应 norm | 无 | 无 |
| `scale_device/host` | 无 | 固定 0.5 | 仅逐 tensor scale |

`sum_device` 验证了去掉 per-gradient `sqrt -> square` 虽有收益，但不能消除每 tensor
reduction 的 launch 数，仍明显慢于 flat。

## 4. 真实裁剪结果

### 4.1 固定总元素 262144

单位：ms/step，输入已驻留 GPU。

| tensor 数 | old | 旧 per-grad device | sum-device | 当前 flat/device |
|---:|---:|---:|---:|---:|
| 1 | 0.442 | 0.137 | 0.126 | 0.144 |
| 16 | 0.821 | 0.634 | 0.540 | 0.402 |
| 128 | 2.991 | 3.752 | 3.240 | 2.313 |
| 512 | 10.369 | 14.880 | 13.006 | 8.228 |

`count=512` 的分解：

| 方案 | build / 内部同步 | final sync | total |
|---|---:|---:|---:|
| old | 8.237 | 2.132 | 10.369 |
| 旧 per-grad device | 6.344 | 8.536 | 14.880 |
| 当前 flat/device | 4.413 | 3.815 | 8.228 |

旧实现看起来 final sync 很短，是因为 `.item()` 已经在 `call_build_ms` 内执行并同步了 norm；
它不是异步优势。

### 4.2 固定总元素 4194304

| tensor 数 | old | 旧 per-grad device | sum-device | 当前 flat/device |
|---:|---:|---:|---:|---:|
| 1 | 0.814 | 0.144 | 0.131 | 0.130 |
| 16 | 1.124 | 0.587 | 0.536 | 0.390 |
| 128 | 3.206 | 3.709 | 3.240 | 2.105 |
| 512 | 10.713 | 15.991 | 14.723 | 8.202 |

从 262K 增到 4M 后，`count=512` 的耗时变化很小，说明此区域主要受 Python 建图和大量小
kernel launch 支配，而不是总字节数。相反，`count=1` 时才更接近带宽型归约。

## 5. No-clip tradeoff

这里的 no-clip 指 `max_norm=1e9`：norm 仍计算，但随机梯度的 `coef==1`。

### 5.1 262144 元素

| tensor 数 | old host branch | 当前 flat/device | 当前相对 old |
|---:|---:|---:|---:|
| 1 | 0.409 | 0.124 | 快 69.7% |
| 16 | 0.648 | 0.408 | 快 37.1% |
| 128 | 2.045 | 2.116 | 慢 3.4% |
| 512 | 7.104 | 8.131 | 慢 14.5% |

### 5.2 4194304 元素

| tensor 数 | old host branch | 当前 flat/device | 当前相对 old |
|---:|---:|---:|---:|
| 1 | 0.794 | 0.125 | 快 84.3% |
| 16 | 1.009 | 0.395 | 快 60.8% |
| 128 | 2.463 | 2.119 | 快 14.0% |
| 512 | 7.505 | 8.192 | 慢 9.2% |

原因不是 norm 回退，而是旧实现从 host 得知 `coef>=1` 后完全跳过 gradient 写回；当前
device-only 路径不能用 Python 分支，所以仍乘 1。若用户明确把 `max_norm=inf`，当前源码会
跳过 scale；高有限阈值则不会。

不建议为了这一小段 no-clip 区间恢复每步 `.item()`：Trainer 默认 `max_grad_norm=1.0`，真实
训练通常需要裁剪或至少不能预知是否裁剪，D2H 会切断 GPU pipeline。正确解法是让 optimizer
本来就要执行的参数更新同时消费 `coef`。

## 6. Norm / scale 单独消融

`count=512`：

| 总元素 | per-grad norm | per-grad sum norm | flat norm | device scale | host-scalar scale |
|---:|---:|---:|---:|---:|---:|
| 262144 | 11.024 | 9.076 | 3.977 | 3.637 | 3.310 |
| 4194304 | 11.034 | 8.939 | 3.986 | 3.596 | 3.313 |

结论：

- 去掉每 tensor sqrt 只能把 norm 从约 `11.0 ms` 降到 `9.0 ms`，仍然有 512 个 reduction。
- flat norm 降到约 `4.0 ms`，证明上一版的主要问题是 reduction 数，而不是 sqrt 算术本身。
- scale 本身约 `3.6 ms`，已经与 flat norm 同量级，是当前 helper 的下一主要优化点。
- device scalar 比 host scalar 只慢约 `0.28-0.33 ms`；主要成本是逐 tensor 图节点和 launch，
  不是读一个 device scalar。

## 7. Profiler / kernel 根因

对 `count=512,total=262144,max_norm=1` 使用完全相同的 profiler 配置：

- 当前 flat/device：profiler total `17.6 ms`。
- 旧 per-grad device：profiler total `35.9 ms`，约 `2.04x`。

动态 `Count`（约四轮有效 rerun）揭示真实 launch 结构：

### 当前 flat/device

- 全量 multiply+reduce：`Count=4`，约每轮一次；该生成源码内部有 reduce init + reduce
  两个 CUDA launch。
- gradient multiply：`Count≈2050`，约每轮 512 次。
- setitem/update：`Count≈2050`，约每轮 512 次。
- 其余 divide/minimum/isnan/ternary/sqrt 是少量标量 kernel。

其中 multiply 与 setitem 两行合计占 profiler 总时约 90.6%，即逐 tensor scale 写回已经主导。

### 旧 per-grad device

- per-gradient multiply+reduce：`Count≈2050`。
- per-gradient sqrt：`Count≈2050`。
- gradient multiply：`Count≈2050`。
- setitem/update：`Count≈2050`。

即每轮约有 512 组 `reduce + sqrt + scale + update`，动态小 kernel 数接近当前 flat 的两倍，
与 profiler `2.04x` 和 wall-time 退化一致。

注意 `source_launch_sites=15` 只是 15 个**唯一生成源码位置**，不能代表动态 launch 数；分析这类
问题必须看 profiler `Count`。

## 8. 内存与下一步

当前 flat helper 的 reduction kernel 接收一个连续 flat 指针，说明 concat 结果是 reduction 的
逻辑输入。其 fp32 大小为：

```text
temporary_bytes = total_gradient_elements * 4
```

本轮是 1 MiB / 16 MiB；10 亿 fp32 gradients 则是约 3.73 GiB。当前方案恢复了旧实现已有的
内存特征，并未比原始 concat+`.item()` 更差，但它仍不满足最终的超大模型显存目标。

建议按以下顺序继续：

1. **短期保留当前 flat/device helper**：它已经修掉 per-grad regression，真实 clip 全矩阵均比
   old 快，且没有 D2H。
2. **把 `coef` 融入 optimizer update**：optimizer 本来就逐参数读 gradient 并更新参数，直接用
   `effective_grad = grad * coef`，可删除本轮 `count=512` 下约 1024 次 multiply+setitem 动态
   kernel，以及一整遍 gradient 读写。
3. **实现 multi-tensor norm 元算子**：输入为 gradient 指针/长度表，分块产生 partial sums，
   再归约成一个 device scalar。目标是固定数量 launch、无全模型 flat 临时量；CUDA 与 NPU
   分别实现 backend kernel，CPU 保留组合 fallback，继续保持 Jittor 元算子路线。
4. 在 multi-tensor op 落地前，如超大模型无法承受 flat 临时量，可按显存预算选择
   `sum_device` fallback；它省内存但在 512 tensors 上仍比 flat/current 慢约 `58-79%`，不能
   作为默认性能路径。
5. 标量 `divide/minimum/isnan/ternary` 目前仍是多个约 3-4 us 的小 kernel，可后续融合为一个
   coefficient kernel，但优先级远低于 optimizer-scale 融合。

## 9. 正确性

固定 gradients `[3,4]` 与 `[0,-3]`、L2 norm `sqrt(34)`、`max_norm=1`：

- old、当前 flat/device、旧 per-grad device、per-grad-sum device 全部通过；
- 返回 norm 最大绝对误差约 `2.04e-7`；
- 裁剪后 gradients 最大绝对误差 `5.96e-8`。

不同 reduction tree 会产生 float32 末位差异，矩阵中约为 `1e-5` 相对量级；没有发现 dtype、
非有限值传播或 clip 语义的新回归。正式 helper 的完整 norm-type/nonfinite 回归由主任务统一执行。

新增依赖：无。

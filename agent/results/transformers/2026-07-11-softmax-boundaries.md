# Softmax 大词表调度边界审计

状态：✅ 已完成（2026-07-11）

## 结论

当前 `code_softmax.py` 在 RTX 4090 / SM89 上，指定的 7 个边界长度均能完成编译，
普通 softmax、log-softmax 的 forward/backward 均通过 float64 NumPy 参考对照；没有
NaN/Inf 或特殊值不匹配，也没有 CUDA resource launch failure。

但是，当前调度存在两个性能边界问题：

1. `length=10000` 被第一段调度到 register/125 threads，资源为 forward 112、
   backward 196 registers/thread。强制改为 register/500 threads 后，forward 从
   `7.764 us` 降至 `5.379 us`，forward+backward 从 `20.937 us` 降至
   `14.379 us`，均约快 31%。建议让 10000 进入第二段，即将第一段的
   `length <= 10000` 改为 `length < 10000`。
2. `65535` 虽然可以运行，但 register backward 已经打满 255 registers/thread，
   每线程 stack frame 为 1112 B；其 forward+backward 为 `120.641 us`，比流式
   内核的 `99.622 us` 慢 17.4%。相邻的 `65536` 因为能走 ILP8 向量加载，register
   反而显著更快。因此不能简单把全局 cutoff 从 65536 降到某个更小常数。

建议保留 register forward，并在 `grad()` 内对大且不能向量化的宽度单独选流式
backward。当前探针支持的候选条件是：非向量化（ILP1）且 `length >= 49153` 时走
streaming backward；ILP8 对齐宽度继续走 register backward。该策略同时覆盖
GPT-2 的 50257 和最坏边界 65535，又不会拖慢 50000、65536 这类向量化宽度。

若实现层面暂时只能使用一个 forward/backward 共用的全局 cutoff，建议暂时保留
65536；直接降到 16384 或 49152 会使若干常见对齐词表明显回退。

## 环境与方法

- Python：conda `jt311`
- GPU：GPU6，NVIDIA GeForce RTX 4090 24 GB，driver 595.71.05
- CUDA：12.2，编译目标 SM89
- 独立 cache：`${JITTOR_LAB_ROOT}/jittor_transformers_perf/runtime/kernel_audit_softmax_gpu6/`
- dtype：float32
- 正确性：每个 length 使用独立 Python 进程，2 rows；以实际输入值构造 float64
  NumPy 参考，覆盖 softmax/log-softmax forward 和随机上游梯度 backward
- 性能：总元素数约 1M，profiler warmup 5 次、稳定采样 20 次
- `pair` 时间包含 softmax forward、backward 和两种方案共有的轻量乘法 kernel，
  因此用于训练路径的同口径比较
- 资源：`cuobjdump --dump-resource-usage`；另用 `--dump-sass` 检查 local-memory
  `LDL/STL` 指令

## 正确性结果

表中为相对 L2 误差；四类输出的 `special_mismatch` 均为 0。

| lastdim | 当前调度 | softmax FWD | softmax BWD | log FWD | log BWD | 结果 |
|---:|---|---:|---:|---:|---:|---|
| 10000 | register/125 | 9.82e-8 | 1.03e-7 | 6.48e-8 | 2.89e-8 | ✅ |
| 10001 | register/512 | 8.70e-8 | 9.64e-8 | 3.60e-8 | 2.64e-8 | ✅ |
| 16384 | register/512 | 1.32e-7 | 1.35e-7 | 3.74e-8 | 2.62e-8 | ✅ |
| 16385 | register/256 | 9.65e-8 | 1.03e-7 | 3.66e-8 | 2.57e-8 | ✅ |
| 65535 | register/256 | 9.33e-8 | 9.81e-8 | 4.74e-8 | 2.59e-8 | ✅ |
| 65536 | register/256 | 9.86e-8 | 1.04e-7 | 3.28e-8 | 2.55e-8 | ✅ |
| 65537 | stream/512 | 1.12e-7 | 1.17e-7 | 3.89e-8 | 2.59e-8 | ✅ |

完整数据：`results/softmax_boundaries_gpu6_correctness.jsonl`。

## 稳定短性能与资源

单位为微秒；每行 current 与 forced stream 使用相同 shape。资源列为当前 register
内核的 `REG/STACK`，顺序为 forward/backward。65537 当前本身即为 stream。

| lastdim (rows) | current FWD | stream FWD | current pair | stream pair | current FWD/BWD 资源 |
|---:|---:|---:|---:|---:|---|
| 10000 (100) | 7.764 | 7.406 | 20.937 | 19.871 | 112/0, 196/0 |
| 10001 (100) | 6.481 | 7.518 | 16.690 | 19.346 | 55/0, 80/0 |
| 16384 (64) | 8.111 | 10.037 | 21.267 | 26.107 | 58/0, 87/0 |
| 16385 (64) | 8.899 | 10.859 | 20.453 | 27.352 | 152/0, 255/0 |
| 65535 (16) | 33.822 | 58.019 | 120.641 | 99.622 | 255/136, 255/1112 |
| 65536 (16) | 28.969 | 57.973 | 63.313 | 99.745 | 255/448, 255/2208 |
| 65537 (16) | 58.294 | 58.306 | 101.098 | 100.983 | 22/0, 22/0 |

完整 profiler/resource 摘要：`results/softmax_boundaries_gpu6_profiles.jsonl`。

### 65536 的 ptxas/SASS 证据

- forward：255 registers/thread，448 B stack/thread，301 条 `LDL/STL`
- backward：255 registers/thread，2208 B stack/thread，986 条 `LDL/STL`
- streaming forward/backward：22 registers/thread，0 B stack/thread

`LOCAL:0` 并不表示没有 local-memory 流量；非零 stack frame 和大量 `LDL/STL`
已经证明寄存器数组被放入线程本地地址空间。256 threads x 255 registers = 65280
registers/block，也使该 kernel 的 occupancy 对 GPU 架构非常敏感。

65536 尽管 spill 很重，ILP8 向量加载仍使它在本机快于流式方案。65535 不能走
ILP，反向 spill 的代价超过了流式内核重新读取数据的代价，因而出现相反结论。

## 阈值补充探针

为避免根据两个端点误判，额外保持约 1M 元素测试了以下长度：

| lastdim | 特征 | current pair | stream pair | 结论 |
|---:|---|---:|---:|---|
| 32000 | t250, ILP8 | 33.310 | 57.350 | register 快 41.9% |
| 32769 | t256, ILP1 | 35.065 | 59.516 | register 快 41.1% |
| 49153 | t256, ILP1 | 83.136 | 80.999 | stream 快 2.6% |
| 50000 | t250, ILP8 | 42.236 | 87.332 | register 快 51.6% |
| 50257 | t256, ILP1 | 84.681 | 81.656 | stream 快 3.6% |
| 65535 | t256, ILP1 | 120.641 | 99.622 | stream 快 17.4% |
| 65536 | t256, ILP8 | 63.313 | 99.745 | register 快 36.5% |

补充摘要：`results/softmax_threshold_probes_gpu6.jsonl`。

这组数据说明长度本身不是充分条件。应先看 `_softmax_v1_cls` 最终得到的 ILP，
再决定 backward kernel；把 49152 作为非向量化 backward 的候选 crossover，比单纯
修改 65536 的全局 cutoff 更符合实测。

## 建议改动顺序

1. 低风险：将第一段边界改成 `< 10000`，让 10000 使用 register/500 threads。
2. 中风险：保留 register forward；在大词表 `grad()` 中按 ILP/length 选择 register
   或 streaming backward，候选初值为 `ILP == 1 and length >= 49153`。
3. 回归时至少覆盖 10000、10001、32000、49153、50000、50257、65535、65536、
   65537，并在另一种 GPU 架构上复查 255-register 内核。

本次只修改独立审计脚本与结果文件，没有修改核心源码。

## 复现

正确性示例：

```bash
CUDA_VISIBLE_DEVICES=6 cache_name=kernel_audit_softmax_gpu6 \
  agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/jittor-transformers-perf/scripts/audit_kernel_changes_gpu.py \
  --task softmax-case --length 65536 --rows 2 --dtype float32
```

稳定 profile 示例：

```bash
CUDA_VISIBLE_DEVICES=6 cache_name=kernel_audit_softmax_gpu6 \
  agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/jittor-transformers-perf/scripts/audit_kernel_changes_gpu.py \
  --task profile --op softmax --length 65536 --rows 16 \
  --warmup 5 --rerun 21 --slots 8
```

10000/t500 调度探针：

```bash
CUDA_VISIBLE_DEVICES=6 cache_name=kernel_audit_softmax_gpu6 \
  agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/jittor-transformers-perf/scripts/audit_kernel_changes_gpu.py \
  --task softmax-forced-schedule --length 10000 --threads 500 --rows 100 \
  --warmup 5 --rerun 21 --slots 8
```

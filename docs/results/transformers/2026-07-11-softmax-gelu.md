# Jittor CUDA Softmax / GELU 性能深挖

状态：✅ 源码定位、GPU0 实证、优化原型与风险分析已完成。本文只记录分析；本 agent 未修改核心实现，也未执行 git 操作。

## 1. 范围与环境

- GPU：NVIDIA GeForce RTX 4090，sm89，仅使用 GPU0。
- Python：`/home/zy/miniconda3/envs/jt311/bin/python`。
- 真 PyTorch：`/home/zy/rt_venv/bin/python`，`torch 2.12.1+cu130`。
- Jittor 源码：当前工作树，`import jittor as torch` 路径。
- 所有 JIT/cache/log 均在 `${JITTOR_LAB_ROOT}/jittor_transformers_perf/`，未写 `/tmp`。
- Jittor profiler 的 `AvgTime` 是重复执行同一输入/输出 buffer 的 L2-hot kernel 时间；wall-time 使用不同输入槽并保留全部输出，且拆成 Python/建图 `build_ms` 与执行等待 `sync_ms`。两种口径不能混用。

复现入口：

```bash
CUDA_VISIBLE_DEVICES=0 agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/jittor-transformers-perf/scripts/probe_softmax_gelu.py --section gelu

CUDA_VISIBLE_DEVICES=0 agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/jittor-transformers-perf/scripts/probe_softmax_gelu.py --section softmax

CUDA_VISIBLE_DEVICES=0 agent/skills/jittor-transformers-perf/scripts/run_perf_env.sh \
  /home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/jittor-transformers-perf/scripts/probe_softmax_gelu.py --section softmax_vocab
```

主要日志：

- `logs/probe_gelu_gpu0.log`
- `logs/probe_softmax_gpu0.log`
- `logs/probe_softmax_vocab_gpu0.log`
- `logs/probe_softmax_gelu_training_gpu0.log`
- `logs/probe_torch_softmax_gelu_gpu0.log`
- `logs/probe_torch_softmax_vocab_gpu0.log`
- `logs/probe_softmax_gelu_fp16_gpu0.log`

## 2. 调用链与实现定位

### GELU

1. `torch.nn.functional.gelu` 在 `python/jittor/torch_compat.py` 中映射到 `jittor.nn.gelu`。
2. Python 实现在 `python/jittor/nn.py::gelu`。
3. exact 路径原式为 `0.5*x*(1+erf(x/sqrt(2)))`；tanh 路径是组合逐元素表达式。
4. Jittor fuser 会把 exact/tanh 整条表达式融合为一个 CUDA kernel。因此 GELU 的主问题不是 launch 数，而是生成 kernel 内的 dtype 和主机建图开销。

旧 float32 exact 生成源码包含：

```cpp
auto op1_yd = (float64)(x[i]);
auto op7_zd = float64(float64(op1_yd) / float64(sqrt2));
auto op8_yd = (float32)(op7_zd);
auto op9_yd = ::erff(op8_yd);
```

即每个 float32 元素先转 double、做一次 double divide，再转回 float 调 `erff`。RTX 4090 的 FP64/双精度除法吞吐远低于 FP32，这正是 exact GELU 的决定性瓶颈。tanh 路径生成代码全为 float32，没有这个问题。

### Softmax

1. `F.softmax` / `Tensor.softmax` 经 `torch_compat.py` 进入 `python/jittor/nn.py::softmax`。
2. CUDA、最后一维、长度 `<=10000` 时进入 `python/jittor/other/code_softmax.py::softmax_v1`。
3. 快路径前向是一枚 `jt.code` CUDA kernel；反向也是一枚独立 `jt.code` kernel。
4. 长度 `>10000`、非末维或非 CUDA 时回退到通用图：`max -> subtract/exp/sum -> divide`。

快路径长度 1024 的实际配置是：

- `tnum=128`。代码中先计算 500/512，下一行又无条件覆盖为 125/128，第一行实际是死选择。
- 每行一个 block，128 threads。
- 每线程 ILP=8，向量读取 8 个 float，值保留在寄存器。
- 一次 BlockReduce max、一次 BlockReduce sum、两次 block 同步。
- 每元素全局内存只读一次、写一次，不物化中间 exp。

通用大词表路径 profile 为 3 个 profiler op、生成源码内 5 次 launch：

1. max 输出初始化 kernel。
2. max reduce kernel。
3. exp/sum 输出初始化 kernel。
4. exp/sum reduce kernel。
5. 再次计算 subtract+exp，并除以 row sum 的输出 kernel。

它没有物化完整 exp tensor，但会完整读取 logits 三次、计算 exp 两次，并支付 5 次 launch；快路径只读一次、算 exp 一次、写一次。

## 3. float32 前向结果

形状均为 8,388,608 个元素（`8192x1024`；等价于原 benchmark 的 `32x256x1024` 元素数）。

### GELU

| 路径 | L2-hot kernel | wall total | build | sync | 相对旧实现 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧 exact 组合式 | 169.33 us | 0.20037 ms | - | - | 1.00x |
| 强制 FP32 `x * 1/sqrt(2)` | 19.03 us | 0.08569 ms | - | - | 2.34x |
| 专用 `jt.code` 256-thread | 19.19 us | 0.07410 ms | - | - | 2.70x |
| 原生 tanh approximate | 18.60 us | 0.08992 ms | - | - | 2.23x |
| 真 PyTorch exact | - | 0.07124 ms CUDA event | - | - | - |

结论：

- 仅消除 float64 divide，即把 Jittor/PyTorch 差距从约 2.7x 缩到约 1.2x。
- 专用单 kernel 版本为 0.0741 ms，和 PyTorch 0.0712 ms 基本持平。
- 强制 FP32 原型相对旧 exact 输出最大绝对差 `2.3841858e-7`。
- exact backward 本身生成纯 FP32 fused kernel；float64 污染集中在独立 exact forward。

### Softmax，长度 1024

| 路径 | L2-hot kernel | wall total | build | sync |
| --- | ---: | ---: | ---: | ---: |
| 原生 `softmax_v1` | 38.51 us | 0.10390 ms | 0.02853 ms | 0.07538 ms |
| 模块级 hoisted Function 原型，128 threads | 38.51 us | 0.08214 ms | 0.00782 ms | 0.07431 ms |
| 直接 `jt.code`，128 threads | 38.28 us | 0.07724 ms | 0.00326 ms | 0.07398 ms |
| 直接 `jt.code`，256 threads | 18.78 us | 0.07770 ms | 0.00322 ms | 0.07448 ms |
| 直接 `jt.code`，512 threads | 23.94 us | 0.07742 ms | 0.00323 ms | 0.07420 ms |
| 真 PyTorch | 68.75 us CUDA event | - | - | - |

原始 corrected benchmark 中 Jittor/PyTorch 为 `0.1480/0.0762 ms = 1.94x`；隔离复跑会因主机状态出现绝对值变化，但 build 拆分稳定地指出热点：`softmax_v1()` 每次调用都重新定义 `CodeSoftmax(jt.Function)` 并重建相同 CUDA source，单次约多 20--25 us Python/建图成本。

正确性：

- 256-thread vs 原生最大绝对差 `5.59e-9`。
- 512-thread vs 原生最大绝对差 `7.45e-9`。

256 threads 把 L2-hot kernel 减半，但 64 MB 流式输入已经接近显存带宽上限，所以 wall-time 没有同步减半。它可能有利于 attention score 刚由 matmul 写入、仍驻留 L2 的情况，但不是可以无条件采用的结论。

## 4. 训练路径

### GELU

Jittor 在同时保留 `y` 和 `dx` 时，可把 GELU 前向和反向融合为一枚 CUDA kernel：

- old exact materialized pair：148.87 us，1 launch。
- FP32 prototype materialized pair：148.64 us，1 launch。
- wall：旧 exact 0.19989 ms，FP32 原型 0.19099 ms。
- PyTorch 独立 forward/backward event 合计约 `71.24 + 109.06 = 180.30 us`。

因此 GELU forward 单算收益很大；完整训练图因 Jittor 元算子融合、backward 和内存流量占主导，收益约 4--5%，整体已经接近 PyTorch。

### Softmax

原生完整梯度探针保持定制反向：

- 前向 code kernel：约 38.4 us。
- 反向 code kernel：约 106 us。
- 上游逐元素 kernel：约 18 us。
- materialized pair 共 3 launches、162.72 us；wall 0.27385 ms。
- PyTorch forward+backward event 合计约 `68.75 + 189.97 = 258.72 us`。

曾用组合公式 `y * (g - sum(y*g))` 模拟 hoisted class 的反向，只用于诊断：它需要额外通用 reduction，变成源码 4 launches、wall 0.41086 ms。该方案不可采用。真正的 class 缓存必须原样保留现有单-kernel custom backward。

## 5. 大词表性能悬崖

以下形状都约 1M 元素，Jittor 与 PyTorch 都是 L2-hot kernel 口径；wall 是 Jittor 不同输入槽口径。

| shape | Jittor 原生 kernel / launches | Jittor wall | 实验快路径 | 快路径 wall | PyTorch kernel |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128x8192 | 7.79 us / 1 | 0.02264 ms | t512 7.54 us | 0.01170 ms | 5.73 us |
| 104x10001 | 244.00 us / 5 | 0.26617 ms | t512 6.49 us | 0.01137 ms | 6.26 us |
| 21x50257 | 278.61 us / 5 | 0.29529 ms | t256 20.55 us | 0.02598 ms | 14.09 us |

关键结论：

- `10000 -> 10001` 会从单 kernel 突然退到 5 launches；同规模 kernel 慢约 37.6x。
- 10001 维直接扩展现有算法、用 512 threads，已与 PyTorch 基本持平。
- 50257 维用 256 threads 可比通用路径快 13.6x，距 PyTorch约 1.46x。
- 50257 维的 512-thread 实验因寄存器/launch 资源超限而失败。block size 必须按长度、dtype、每线程 item 数和寄存器预算自适应。
- 50257/t256 输出 vs 原生最大绝对差 `2.56e-9`，最大 row-sum error `2.62e-6`。
- 128K 词表继续使用单 block、把所有元素留在寄存器已不可扩展，需要多 block 或重读输入的算法。

## 6. fp16 结果

8M 元素：

| 路径 | kernel | wall | build | sync |
| --- | ---: | ---: | ---: | ---: |
| fp16 旧 exact GELU | 21.43 us | 0.06904 ms | 0.01915 ms | 0.04988 ms |
| 显式升 FP32 的 typed-constant 原型 | 20.24 us | 0.06208 ms | 0.01428 ms | 0.04779 ms |
| fp16 专用 GELU code | 14.38 us | 0.04223 ms | 0.00172 ms | 0.04051 ms |
| fp16 原生 softmax t128 | 10.57 us | 0.04873 ms | 0.01032 ms | 0.03841 ms |
| fp16 softmax t256 | 13.66 us | 0.04124 ms | 0.00343 ms | 0.03781 ms |

风险点：

- fp16 旧 exact 路径本来就没有 float64 token；typed-constant 原型 vs 旧输出最大绝对差 `0.001953125`，约一个 fp16 ULP。必须以真 PyTorch fp16 forward/backward 为 oracle确认，不能只引用 float32 的 `2.38e-7`。
- fp16 t256 的 hot kernel 比原生 t128 更慢，说明 float32 长度 1024 的线程调优不能跨 dtype 直接采用。
- fp16 t256 softmax vs 原生最大绝对差 `3.81e-6`。

## 7. 已采用与未采用方案

### 已采用的低风险方向（由主 agent 集成）

1. GELU typed constant：避免 float32 exact 路径的 float64 divide；低精度显式用 fp32 compute 后 cast 回输入 dtype。
2. Softmax Function-class LRU：按 `(length, log)` 缓存生成的 `CodeSoftmax` class，仍为每次调用创建 Function 实例，并原样保留现有定制 backward。

预期收益：

- float32 GELU forward 约 2.3x；完整局部训练约 4--5%。
- softmax 长度 1024 的 build 约从 28.5 us 降至接近 7--8 us，总 wall 接近 PyTorch。

### 明确未采用：固定调整 softmax threads

本轮没有把 128 固定改成 256/512，原因：

- float32/1024：t256 的 L2-hot kernel 快约 2x，但流式 wall 几乎不变。
- fp16/1024：t256 反而比 t128 慢约 29%。
- float32/50257：t512 直接资源超限失败，t256 才可运行。
- 改变 reduction tree 会带来末位差异，虽当前 float32 最大约 `1e-8`，仍需覆盖 NaN/Inf、极值、fp16/bf16 和 backward。
- 需要长度、dtype、row 数、register items 联合启发式，不能用单一常数。

## 8. 后续可落地优化

优先级 P0：验证当前两项低风险修复。

- CPU/CUDA/NPU 三后端 forward/backward。
- `float16/bfloat16/float32/float64` 对真 PyTorch。
- 确认 LRU key 包含影响源码的全部变量，目前为 `(length, log)`；验证多线程调用和最多 128 类的内存占用。

优先级 P1：扩大 softmax 单-block快路径并自适应 block。

- 约 10K 可优先 512 threads；约 50K 可尝试 256 threads。
- 以 `ceil(length/threads)` 和编译后的寄存器数设硬上限，资源不满足时回退。
- 前向与 custom backward 必须使用同一调度选择。
- 在 64/128/256/512/1024/2048/4096/8192/10000/10001/32768/50257 上离线 autotune，固化少量分段规则，不在运行时动态试错。

优先级 P1：重写任意长度大词表 softmax。

- 单 block 版本不必把全部元素保存在寄存器：可分三遍读取 logits（max、sum(exp)、normalize），用更多带宽换取可控寄存器和任意长度支持，仍优于当前 5-launch通用图。
- row 很少、词表 128K 以上时采用多 block/row：partial max -> row max -> partial sum -> normalize，或 CUB segmented reduction；目标是消除当前通用 reduce 的两次初始化 kernel，并避免 exp 重算/低占用。
- 训练需同步实现 log-softmax 和 backward，避免仅优化 sampling forward。

优先级 P2：CUDA 专用 GELU op / GEMM epilogue。

- 单独 CUDA GELU kernel 已达到 PyTorch；若落地，必须有 custom backward。
- `jt.code` 在 ACL/NPU 不支持，必须 CUDA guard，CPU/NPU保留组合原语或增加 ACL 原生绑定。
- MLP 更高收益方向仍是 cublasLt bias+GELU epilogue，省去一次中间 tensor 写回。

## 9. 建议测试矩阵

- dtype：fp16、bf16、fp32、fp64。
- 长度：1、31、32、33、64、125、127、128、129、256、512、1024、4096、8192、9999、10000、10001、32768、50257、128256。
- row 数：1、21、104、128、8192，覆盖低占用和大吞吐。
- 数据：普通随机、大正负值、全相等、单极大值、`+/-inf`、NaN；log-softmax 单独覆盖。
- 正确性：对真 PyTorch forward；随机上游梯度的 backward（不能用 `sum(softmax)` 这种恒定目标）；row sum；输出 dtype。
- 性能：同时记录 Python build、CUDA kernel、总 wall、launch 数和峰值显存；warm/cold 两种 cache。
- 回归：attention softmax、LM sampling 50K/128K vocab、cross-entropy/log-softmax、transformer MLP exact/tanh GELU。

新依赖：无。

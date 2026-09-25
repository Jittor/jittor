# 性能与显存画像：`jt.profile`

- 状态：维护中
- 对应提交：`wip/profiler-rework` 分支（基线 `cb93ba04`）
- Owner：Jittor 核心维护者
- 复查触发：执行器批次/算子发射路径、分配器（SFRL/Temp）记账、图重放、CUPTI 版本变化

一个训练步慢、显存高的时候，要回答的问题总是那几个：主机时间花在哪（Python 构图、
执行器规划、逐算子发射，还是在等设备）？设备时间花在哪几个算子、哪几个 kernel？这一
步是主机瓶颈还是设备瓶颈？显存峰值是多少、峰值那一刻谁活着、是谁建的？

`jt.profile` 用一个入口回答这些问题：

```python
import jittor as jt

with jt.profile() as prof:          # 默认：设备时间（有 CUPTI 时）+ 显存
    train_step()
print(prof.summary())               # 主机拆分、设备拆分、瓶颈判断、算子/kernel/调用点表、显存
prof.export_chrome_trace("step.json")   # chrome://tracing 或 Perfetto 打开
```

Torch 兼容模式下，`torch.profiler.profile` 与 `torch.autograd.profiler.profile` 就是
它：`key_averages().table()`、`export_chrome_trace`、`schedule`/`on_trace_ready`、
`record_function` 都可用。

## 它测什么，怎么测

**不改变被测的东西。** 区域内不插同步、不重跑算子、不改编译选项。旧的
`jt.profile_scope` 做相反的事——每个算子后同步、可选重跑——所以它回答的是"这个 kernel
单独跑多快"，不是"这一步时间去哪了"（见下文"旧工具"）。区域两端默认各做一次
`jt.sync_all(True)`，让区域外排队的工作不漏进来、区域内建的工作在报告前做完；
`sync=False` 关掉。

### 主机拆分

执行器在 `src/runtime/profiler/step_trace.{h,cc}` 里给每个批次、每次设备等待、每个
发射的算子打时间戳（CLOCK_MONOTONIC，与 Python 的 `time.perf_counter_ns` 同一个钟）。
区域的墙钟时间被**划分**为（各项之和等于墙钟）：

| 项 | 含义 |
| --- | --- |
| Python + 构图 | 墙钟减去执行器与图发射：Python 代码、前端、pyjt 入口、建 Op/Var |
| 执行器规划 | `build_exec_plan`：遍历、融合、调度 |
| JIT 编译 / kernel 查找 | `parallel_compile_all_ops`；缓存命中时是查表开销 |
| 算子发射 | 执行循环里除设备等待外的时间：分配、准备、发射 |
| 等待设备 | `DeviceWaitScope`：批次末尾同步、CPU 算子前的同步、阻塞拷贝 |
| 设备图发射 | `graph_launch`（`jt.graph_replay` 录好的图） |

每个算子另记"分配结束"时刻，所以还能看出发射时间里有多少是分配。算子内部发生的设备
等待单列为该算子的 `wait us`——一个本不该同步的算子在同步，会在这一列直接露出来。

`python=True` 另开一个采样线程，按与 `Op::launch_origin` 相同的规则（跳过 `jittor.*`、
`torch.*` 模块的帧）把主机时间记到 Python 行上。采样要拿 GIL，期间把解释器切换间隔调到
采样间隔，是统计量：几个样本以内的行是噪声，多跑几步再看。

### 设备拆分

有 CUPTI 时，每个 kernel、memcpy、memset 带着设备时间戳进来。执行器发射每个算子时把
算子序号作为 CUPTI external correlation id 压栈，所以每个 kernel（包括 cuBLAS/cuDNN
内部的、设备图里的）都挂到发射它的算子上；报告里的 "attributed to operators" 是挂上
的比例，挂不上的记为 "(device work outside any operator)"，例如图重放在 Python 里做的
输入/输出拷贝。设备忙碌时间取区间并集。

瓶颈判断：设备忙碌 ≥ 80% 墙钟为 device-bound；< 50% 为 host-bound，并给出主机时间在
Python/执行器/等待之间的比例；其间为 mixed。没有 CUPTI 时退回用"等待设备"占比估计，
并标注 *(estimated)*。

CUPTI 不是 Jittor 的依赖。查找顺序：`$JITTOR_CUPTI_LIB`、`nvidia-cuda-cupti-cu12`
wheel（`site-packages/nvidia/cuda_cupti`）、编译用 CUDA toolkit 的 `extras/CUPTI`、
加载器路径。找不到时报告写明"device timing unavailable"和原因，不编数。
`device=True` 则直接报错。建议装与 CUDA runtime 同版本的 wheel：

```bash
pip install "nvidia-cuda-cupti-cu12==12.2.*"   # 与 jittor 用的 CUDA 12.2 对应
```

### 显存

`memory=True`（默认）时，SFRL 池每次分配/释放、TempAllocator 每次工作区分配/释放、
池向驱动申请/归还内存，都记一条带时间戳的事件；池分配还带着 Var 的 dtype/shape、
生产它的算子（`name_ex`）和建这个算子的 Python 行。区域开始时从被持有的 Var 出发
遍历图，把已经活着的分配记为 "allocated before profile"（参数、优化器状态）。

回放这份日志得到：

- **精确峰值**及其时刻。它必须等于池自己的 `device_memory_peak`（区域开始时已重置），
  不等时报告会同时写出两者；
- **含工作区的峰值**：cuDNN/cub 工作区走 TempAllocator，不在 `device_memory_peak`
  里，这里加回来；
- **峰值时刻活着的分配**，按生产算子、Python 调用点、张量（dtype+shape）、区域前/内
  分组；
- 池的保留量峰值（新计数器 `jt.core.device_memory_reserved_peak`）、区域末尾的缓存
  块数、最大空闲块和碎片率（`1 - 最大空闲块 / 缓存总量`）；
- NVML 看到的本进程占用（起止与约 2 ms 采样峰值），以及"池外"部分（CUDA 上下文、
  模块、库句柄）。

## 结果对象

`prof.summary()` 之外，结构化数据都在：

| 属性/方法 | 内容 |
| --- | --- |
| `prof.host` | 上表各项（ns）、批次数、算子数 |
| `prof.device` | `busy`、`kernel`、`copy`、`kernels`、`attributed`、`utilization`；无 CUPTI 为 `None` |
| `prof.bound` | `(判断, 依据)` |
| `prof.op_stats(group_by_shapes=False)` / `prof.table(sort_by=...)` | 每算子：调用次数、主机时间、等待、设备时间、kernel 数、输入 shape、调用点 |
| `prof.kernel_stats()` / `prof.kernel_table()` | 每 kernel：次数、设备时间、由哪些算子发射 |
| `prof.site_stats()` / `prof.site_table()` | 每 Python 行：建的算子数、发射时间、设备时间、采样主机时间 |
| `prof.memory` | `MemoryReport`：`peak_allocated`、`peak_with_workspace`、`pool_peak_reserved`、`live_by("op"/"site"/"tensor"/"origin")`、`fragmentation`、`timeline` |
| `prof.replay` | 区域内设备图发射次数；注册的重放对象的计数增量与拒绝原因 |
| `prof.op_records` / `kernel_records` / `copy_records` | 原始记录 |

`prof.step()` 标记步边界。`jt.profiling.record_function(name)` 在 trace 里标一段主机
区间。

## 图重放

`jt.graph_replay` 录成设备图之后，一步只剩一次 `graph_launch`，执行器不运行——旧的
逐算子画像因此是空的。现在 `graph_launch` 本身是一条记录（`cuda_graph_launch`），图里
的 kernel 通过 correlation 挂在它下面；`GraphReplay` 实例自动注册，报告给出它在区域内的
`stats` 增量和拒绝原因。其他捕获/重放机制调用
`jittor.profiling.register_replay_source(obj)` 即可进入报告（对象需有 `stats` 字典，
可选 `refused`/`_graph_refused`）。

## 开销

同一进程、同一缓存，10–20 步取三次最好（RTX 4090）：

| 负载 | 不画像 | 仅主机 | 默认（+CUPTI+显存+shape） | 默认+Python 采样 |
| --- | --- | --- | --- | --- |
| MLP 训练，device-bound（2.3 ms/步） | 2.287 | 2.294（+0.3%） | 2.344（+2.5%） | — |
| Torch 模式 GPT-2 小模型，host-bound（10.1 ms/步） | 10.10 | 10.31（+2%） | 11.86（+17%） | 12.16（+20%） |

host-bound 负载上的开销主要来自 CUPTI 的 runtime API 记录（correlation 需要它）。因为
主机拆分里的 "Python + 构图" 是"墙钟减执行器"，画像本身的开销会记在这一项；精确比较
主机时间时用 `device=False, memory=False`。单步画像的第一步会多出 CUPTI 首次启用的
一次性成本，多步画像摊掉。

## 边界

- 调用点规则跳过所有 `jittor.*`/`torch.*` 帧：原生 `jittor.models` 里的模型，调用点
  会落在用户调用模型的那一行；Torch 模式下的 Transformers/Diffusers 模型落在库源码行。
- 惰性执行下算子在同步处才发射；`record_function` 只框住主机区间和区间内的即时发射
  （例如设备图），惰性建的算子靠调用点归属。
- `profile` 不嵌套。主机拆分按一条时间线计算；多个 Python 线程同时驱动执行器的情形没有验证。
- 记录超过每类 2^20 条时截断，报告会写明。
- 只验证了 CUDA（RTX 4090，CUPTI 12.2）；ROCm/NPU 上主机拆分与显存部分走同一条执行器
  路径，设备时间没有实现。

## 旧工具

| 工具 | 现状 |
| --- | --- |
| `jt.profile_scope` / `jt.profiler` | 保留，定位为**算子级 kernel 微基准**：每个算子后同步、可重跑，报告时间是"发射+kernel+同步"，不代表一步。区域内只有设备图发射时，警告会指向 `jt.profile` |
| `profile_mark` | 保留，只对 `profile_scope` 生效；本轮未复核 |
| `profile_memory_enable` + `get_max_memory_info` / `get_max_memory_treemap` / `get_peak_*` | 保留；重新开启时从零开始（原来沿用上一次会话的峰值与 Var 列表）。调用点需要 `trace_py_var=3`，代价高；推荐 `jt.profile(memory=True)` |
| `jt.core.device_memory_used/reserved/peak/reset_device_memory_peak/allocated_total` | 正确，保留；新增 `device_memory_reserved_peak`，`reset_device_memory_peak` 同时重置两个峰值 |
| `display_memory_info`、`liveness_info`、`jt.introspection.counters` | 调试用，保留 |
| `graph_build_profile_*` | 需 `JT_GRAPH_BUILD_PROFILE=1` 重编核心，给 C++ 构图各阶段的自身时间；开发者工具，保留 |
| `trace_py_var>=2` + `dump_trace_data` | 调试追踪；它在每个算子后同步，不能用来量时间 |

# 性能与显存画像：`jt.profile`

[本栏目目录](index.md) · [English](profiling.en.md)

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
它，而且产物与 PyTorch 2.11 的一样（`compat/torch/profiler.py`）：

- `events()`/`key_averages()` 是 PyTorch 的 `FunctionEvent`/`FunctionEventAvg`：每个算子、每个
  `record_function` 区间是 CPU 事件，每个 kernel/拷贝是 CUDA 事件，kernel 的时间同时记为发射它的
  算子的 self 设备时间；区间的 total = 自身 + 子事件。惰性执行下，区间里**建**的算子往往在区间
  结束后才执行（在另一个区间里的 `.item()`/sync 处，或在所有区间之外），它们仍算建它的区间的子事件
  （执行器按建算子时打开的区间记，`op_records[i]["range"]`）；执行它们的那个区间也不把这段时间算作
  自身时间。所以 `forward` 的 total 是前向建出的全部算子，而做 sync 的区间只剩自己的时间。
- 新 kernel 第一次需要时才生成和编译：执行器规划一批算子时，或算子第一次运行时。这段时间是一个
  `jit_compile` 事件（后者挂在该算子下），不算进算子的发射时间；原生 `jt.profile` 的主机拆分同样记在
  "JIT compile / kernel lookup"。没有新 kernel 时它只是很短的查缓存。
- 一个融合算子由几行代码建出来，就有几个成员：它的栈先列每个成员最内层的那一帧（按成员顺序），
  再接最后一个成员的调用方，Source Location 因此能看到所有融合进来的行。
- 区间边界：进入时 `jt.sync_all(True)`，所以在 profile 之前建好、还没执行的图在进入时就执行完，
  不出现在结果里；退出时同样 sync，区间里建的图在退出前一定执行完。原生 `jt.profile(sync=False)`
  两边都不 sync，区间里建而没 sync 的图在区间外执行，不被记录。
- `table()` 的列、宽度、截断、单位、排序键和页脚与 PyTorch 的 `_build_table` 相同；
  `profile_memory` 给出每个算子分配减释放的 CPU/CUDA Mem（self 与 total），`record_shapes` +
  `group_by_input_shape` 给出 `[[64, 256], [512, 256]]` 形式的 Input Shapes，`with_stack` +
  `group_by_stack_n` 给出 Source Location（建算子时的 Python 栈，`file(line): func`，最内层在前，
  去掉 jittor 自己的帧）。kernel 名按 C++ 反修饰显示。
- `export_chrome_trace` 写 Kineto 格式（顶层键、`cpu_op`/`user_annotation`/`kernel`/`gpu_memcpy`、
  ac2g 流、`[memory]` 瞬时事件、`GPU n`/`stream n` 元数据、kernel 的 grid/block/寄存器/共享内存），
  TensorBoard profiler 插件、HTA、Perfetto 按 PyTorch trace 读；`export_stacks` 写火焰图格式；
  `export_memory_timeline` 写 `[times, sizes]`（json/json.gz）或 matplotlib 图（html），版式与
  PyTorch 相同，但 jittor 不把分配分成参数/梯度/激活，全部记在 "Unknown"。
- 没有的：Python 函数调用事件（`python_function`）、CUDA runtime API 事件、`with_flops`、
  `with_modules`。原生 `jt.profile(stacks=True)` 在 `result.stacks` 里给出同样的栈。

## 读报告，再决定下一步

先准备能正常运行的训练步骤；测稳态时，在画像区域之外预热。示例中的 `train_step()` 代表这个已有函数。导出的 trace 保存在源码树之外。

| 报告现象 | 下一步 |
| --- | --- |
| 主机工作占主导 | 查 Python 构图和执行器规划；可用 `device=False, memory=False` 做主机侧分析 |
| 设备工作占主导 | 查算子表与 kernel 表 |
| 显存受限 | 查 `prof.memory`，再按[显存优化](memory-optimization.md)处理 |
| 无设备计时 | 按下文检查 CUPTI；主机时间不能代替设备实测 |

画像用于归因；验证改动收益时，关闭画像重新测量。CUDA 设备计时需要 CUPTI；这里尚未实现 ROCm/NPU 设备计时。下文按需查阅同步方式、归因规则和字段含义。

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

CUPTI 是可选依赖（extra `profile`），不装也能用 `jt.profile`，只是没有设备时间。
查找顺序：`$JITTOR_CUPTI_LIB`、`nvidia-cuda-cupti-cu12` wheel
（`site-packages/nvidia/cuda_cupti`）、编译用 CUDA toolkit 的 `extras/CUPTI`、加载器路径。
找不到时报告写明"device timing unavailable"和原因，不编数；`device=True` 则直接报错。

```bash
pip install "jittor[profile]"   # nvidia-cuda-cupti-cu12==12.2.142，对应 jittor[cuda12] 的 CUDA 12.2
```

用别的 CUDA runtime 时，装同版本的 `nvidia-cuda-cupti-cu12` 或把 `JITTOR_CUPTI_LIB` 指向
对应的 `libcupti.so`。

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
| `prof.stacks` | `stacks=True` 时，栈 id → 帧列表（`op_records[i]["stack"]` 索引它）；否则 `None` |

`prof.step()` 标记步边界。`jt.profiling.record_function(name)` 在 trace 里标一段主机
区间。

## 整段运行的显存：快照与分配器统计

`jt.profile` 回答一个区域；要看一整段训练里显存怎么涨、碎在哪，用 PyTorch 的那套工具，
接口与产物和 PyTorch 2.11 相同：

```python
jt.profiling.record_memory_history(max_entries=100000)   # torch.cuda.memory._record_memory_history
train()
jt.profiling.dump_memory_snapshot("snap.pickle")          # torch.cuda.memory._dump_snapshot
jt.profiling.record_memory_history(None)
print(jt.profiling.memory_summary())                      # torch.cuda.memory_summary 的表
```

Torch 兼容模式下 `torch.cuda.memory._record_memory_history/_snapshot/_dump_snapshot`、
`torch.cuda.memory_snapshot`、`memory_stats`、`memory_summary`、`reset_accumulated_memory_stats`
就是这些函数。

- **快照**：pickle 的键、动作（`segment_alloc`/`alloc`/`free_requested`/`free_completed`/
  `segment_free`/`oom`/`snapshot`）和嵌套与 PyTorch 的相同，可以直接拖进
  <https://pytorch.org/memory_viz>，也能用 `torch.cuda._memory_viz`（`segsum`、`trace_plot`、
  `segment_plot`）读。记录来自 SFRL 池本身（`src/runtime/profiler/memory_history.{h,cc}`）：
  段是池向驱动要的那一块，块是段里切出来的那一块，地址是真实设备地址。
  `enabled="state"` 只记当前活着的块，`"all"` 再记全过程；`context` 决定记不记栈；
  `max_entries` 每卡保留最新的若干条；停止时清掉记录，与 PyTorch 一样。
  `jt.profiling.record_function` 的区间写进 `external_annotations`。
- **栈**：惰性执行下，块分配发生在执行时，那一刻的 Python 栈是触发执行的那行。所以张量块记的是
  **建出它的算子时**的完整 Python 栈（最内层在前）；库的工作区记到正在跑的算子；释放和执行器外的
  分配记当下的栈（持有 GIL 时）。只有 Python 帧，没有 C++ 帧（`stacks="all"` 与 `"python"` 相同）。
- **统计**：`memory_stats()` 有 `torch.cuda.memory_stats` 的全部键——分配、段、活跃与
  inactive split 的块数和字节、保留与请求字节，按 all/small/large 池各给 current/peak/
  allocated/freed，以及 `num_alloc_retries`、`num_ooms`、`num_device_alloc/free`。计数在池的
  每次分配、释放、段增减时更新（`sfrl_device_memory_stats`）；`memory_summary()` 的版式与
  PyTorch 逐字符相同，只是标题写 Jittor（兼容模式下也是）。

**显存不够时**：设备拒绝分配（池已归还缓存并重试之后）抛 `jittor.JittorOutOfMemoryError`
（`RuntimeError` 的子类；Torch 兼容模式下 `torch.OutOfMemoryError` 和
`torch.cuda.OutOfMemoryError` 就是这个类，所以按 torch 名字捕获照样接得住）。消息开头是 PyTorch 的那几句——
`CUDA out of memory. Tried to allocate …. GPU n has a total capacity of … of which … is free.
Including non-Jittor memory, this process has … memory in use. Of the allocated memory … is
allocated by Jittor, and … is reserved by Jittor but unallocated.`（兼容模式下同样写 Jittor），
数字取自池自己的计数和 NVML；随后是 jittor 的建议（请求比整卡还大时提示查形状）和出错算子的
创建位置。PyTorch 末尾那句 `PYTORCH_CUDA_ALLOC_CONF` 的建议在这里不适用，没有照抄。
`memory_stats()` 的 `num_ooms` 与快照里的 `oom` 条目同时记一次。

与 PyTorch 数字不同的地方来自分配器本身，不是记录方式：小段是 1 MiB（PyTorch 2 MiB），块按 512 B
对齐，块属于它所在段的池；释放不排在流后面，所以 `free_requested` 与 `free_completed` 同时发生；
没有 `max_split_size`，因此没有 oversize 分配和段；cuBLAS 工作区不从池里分配，不出现在快照里。

## 图重放

`jt.graph_replay` 录成设备图之后，一步只剩一次 `graph_launch`，执行器不运行——旧的
逐算子画像因此是空的。现在 `graph_launch` 本身是一条记录（`cuda_graph_launch`），图里
的 kernel 通过 correlation 挂在它下面；`GraphReplay` 实例自动注册，报告给出它在区域内的
`stats` 增量和拒绝原因。其他捕获/重放机制调用
`jittor.profiling.register_replay_source(obj)` 即可进入报告（对象需有 `stats` 字典，
可选 `refused`/`_graph_refused`）。

## 处理画像开销

CUPTI 记录、分配事件与 Python 采样都会增加工作。主机受限的模型尤其敏感：画像开销会计入按差值计算的“Python + 构图”项。仅分析主机时用 `device=False, memory=False`；多步画像可摊薄 CUPTI 首次初始化成本。不要把画像中的延迟直接作为未插桩的基准结果。


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

## 维护参考

- 状态：维护中
- 对应提交：`wip/profiler-rework` 分支（基线 `cb93ba04`）
- Owner：Jittor 核心维护者
- 复查触发：执行器批次/算子发射路径、分配器（SFRL/Temp）记账、图重放、CUPTI 版本变化

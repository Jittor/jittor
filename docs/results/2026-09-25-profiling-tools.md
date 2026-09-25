# 性能/显存分析工具：审计与重写

- 状态：已实现并在真实 CUDA 设备上验证；未合入 `2.0-refactor`
- 日期：2026-09-25
- 基线提交：`cb93ba04`（分支 `wip/profiler-rework`）
- 硬件：RTX 4090 24 GB（驱动 580.178.04），nvcc 12.2.140，CUPTI 12.2.142
  （`nvidia-cuda-cupti-cu12` wheel，经 `JITTOR_CUPTI_LIB` 指向；jittor 环境默认不带 CUPTI）
- Owner：Jittor 核心维护者
- 复查触发：执行器发射路径、SFRL/Temp 分配器记账、图重放、CUPTI 版本变化
- 机制说明：[性能与显存画像](../notes/profiling.md)

## 结论

最近的性能工作要回答的问题——一步的主机时间在 Python、规划、发射、等设备之间怎么分；
设备时间在哪些算子和 kernel；是主机还是设备瓶颈；显存峰值多少、峰值时谁活着、是哪行
代码建的；重放的一步是否真的在重放——原有工具一个也答不全，只能靠 `perf`、`nsys` 和临时
脚本。原有工具里**计数器是对的**（池峰值 `device_memory_peak`、`max_memory_allocated`
已修好），**画像器不是**：逐算子同步、看不到重放、Torch 侧 `profiler` 是空壳。

新增 `jt.profile` 一个入口；Torch 的 `torch.profiler`/`torch.autograd.profiler` 改为
它的外壳；旧工具保留，修掉会误导的部分并在文档里改正定位。

## 审计

每一项都先做了最小复现（脚本不入库，放在 `$JITTOR_LAB_ROOT/_state/profiler-rework/repro/`）。

| 工具 | 声称 | 实测 | 结论 |
| --- | --- | --- | --- |
| `jt.profile_scope` / `jt.profiler`（`Profiler::record_and_run`） | 区域内各算子耗时 | 每个算子执行后同步设备（可重跑）。MLP 训练步（1024→4096→4096→1024，batch 256，AdamW）稳态 2.289 ms，开着它 3.53–3.76 ms；表中总时间 2.71 ms 是"发射+kernel+同步"的和，既不是设备时间也不是主机时间；表头没有主机/设备列，行名是原始 JIT key，没有 Python 行（除非 `trace_py_var`） | 保留，重新定位为算子微基准；整步画像由 `jt.profile` 替代 |
| 同上，对图重放的一步 | — | `jt.graph_replay` 录成 CUDA graph 后，一步 0 行，并发出指向 `auto_flush_ops`（KI-EXEC-002）的错误解释 | 修：检测到区域内有设备图发射时改为说明重放绕过执行器、指向 `jt.profile` |
| `profile_memory_enable` + `get_max_memory_info` / `get_max_memory_treemap` / `get_peak_device_used_memory` | 峰值与峰值时的 Var、调用点 | 峰值数对（4×(2048→8192→2048) MLP，batch 4096：3456.7 MiB，与 `device_memory_peak` 一致）。但**重新开启沿用上次会话的峰值和 Var 列表**：先开一次、再带 `trace_py_var=3` 开一次，树图给出的是第一次的无栈 Var；调用点必须 `trace_py_var=3`（该步 68.0 → 84.6 ms），开启前创建的张量无调用点；不含 cuDNN/cub 工作区；文本用 `[!@#div1!@#]` 分隔 | 修：开启时清零（`setter_profile_memory_enable`）；保留，推荐 `jt.profile(memory=True)` |
| `jt.core.device_memory_used/reserved/peak`、`reset_device_memory_peak`、`device_memory_allocated_total` | torch 的 `memory_allocated/reserved/max_memory_allocated` 对应物 | 正确：峰值在每次池分配时记录（上例 3456.7 MiB；NVML 1 ms 采样 4112 MiB，差额是上下文和缓存） | 保留；新增 `device_memory_reserved_peak`，`reset_device_memory_peak` 同时重置两个峰值 |
| `torch.cuda.max_memory_allocated`（"22 GB 的步只报 0.1–1.3 GB"） | — | 已修（`device_memory_peak`）：复现里 reset 后一步 140.1 MiB，与 `memory_allocated`/`memory_reserved` 读数一致 | 保留 |
| `torch.cuda.max_memory_reserved` / `max_memory_cached` | 保留量峰值 | 返回的是**分配**峰值：140.1 MiB，小于同时刻 `memory_reserved()` 163.0 MiB | 修：读池的保留量峰值 |
| `torch.cuda.memory_stats` / `memory_summary` | — | `memory_stats` 只有 3 个键；`memory_summary` 不存在 | 补：保留量峰值、累计分配；新增 `memory_summary` |
| `torch.profiler.profile` / `record_function` / `schedule` / `tensorboard_trace_handler` | PyTorch 画像 API | 空壳：`profile` 什么都不录，没有 `key_averages`，`export_chrome_trace` 不写文件且不报错 | 替换为 `jt.profile` 外壳 |
| `torch.autograd.profiler.profile` / `EventList` | 同上 | 空壳：`key_averages()`/`function_events` 返回空表，`table()` 返回空串 | 替换 |
| `display_memory_info`、`get_mem_info`、`liveness_info`、`jt.introspection.counters` | 调试转储/计数 | 按文档工作（`total_cuda_used` 合计所有卡，已有文档说明） | 保留，未改 |
| `graph_build_profile_*` | C++ 构图阶段自身时间 | 需要 `JT_GRAPH_BUILD_PROFILE=1` 重编核心；未开时正确地报空 | 保留（开发者工具），本轮没有用探针构建复核 |
| `trace_py_var>=2` + `dump_trace_data` / `print_trace` | 追踪 | 执行器在每个算子后同步（`exec_runner.cc`），不能用来量时间 | 保留，文档注明 |
| `profile_mark` | 给 `profile_scope` 分段 | 未复核 | 保留 |
| `bench/torch_compat/worker.py` 的 NVML 采样器 | 跨框架进程显存峰值 | 20 ms 间隔采样，可能漏掉短峰；作为两侧同一把尺子仍合理 | 保留 |

## 改了什么

- **执行器步追踪**（`src/runtime/profiler/step_trace.{h,cc}`）：批次（规划/编译/结束）、
  设备等待（`DeviceWaitScope`）、每个发射的算子（开始/分配完/结束、可读名、输入 shape、
  输出字节、`Op::launch_origin` 调用点）、设备图发射；发射期间把算子序号压为 CUPTI
  external correlation id。未开启时每个钩子一个分支。挂钩点：`executor.cc`、
  `exec_runner.cc`、`executor_entry.cc`、`graph_capture.cc`、`var.cc`。
- **分配器事件与保留量峰值**：`sfrl_allocator.cc`（分配/释放/向驱动申请与归还，新增
  按设备的保留量与其峰值）、`temp_allocator.cc`（工作区）、`mem_info.*`
  （`device_memory_reserved_peak`）；`launch_diagnostics` 增加调用点 id → 文件行的查询。
- **`jittor.profiling`**（`jt.profile`）：CUPTI（ctypes，按需加载）、主机拆分、设备拆分、
  瓶颈判断、算子/kernel/调用点表、显存日志回放（`MemoryReport`）、Python 采样器、NVML、
  Chrome trace 导出。`GraphReplay` 自动注册为重放来源。
- **Torch 外壳**：`compat/torch/profiler.py`（新）；`installers/utilities.py`、
  `autograd.py`、`installers/autograd.py` 改为发布它；`installers/cuda/{api,bindings}.py`
  修 `max_memory_reserved`、扩 `memory_stats`、加 `memory_summary`。
- **旧工具**：`profile_memory_enable` 开启即清零；`profile_scope` 的空报告警告区分设备图
  发射；文档与 notebook 改正（原文"峰值没有重置接口"已不成立）。

## 证据

### 原生 ResNet18 训练（batch 64，224²，SGD momentum），3 步

稳态 30.04 ms/步。`jt.profile(python=True)` 三步 91.24 ms：

- 主机：Python+构图 11.05 ms（12.1%）、规划 1.37、编译/查找 3.07、发射 14.38
  （1401 个算子，10.3 µs/个，其中分配 2.6 µs）、等待设备 61.37 ms（67.3%）；
- 设备：忙碌 88.48 ms（97.0%），1308 个 kernel 全部归属到算子 → **device-bound**；
- 设备时间前几位：`cudnn_conv_backward_w` 22.79 ms（25.7%）、BatchNorm 的 `code` 算子
  21.86 ms（24.6%，kernel 为 `batch_norm_backward` 10.35 ms、`batch_norm_forward`
  6.47 ms）、`cudnn_conv_backward_x` 14.84 ms、`cudnn_conv` 14.37 ms；
- 显存：Var 峰值 1645.0 MiB（与池计数器一致），**含 cuDNN 工作区 1929.0 MiB**（原先的
  任何计数器都看不到这 284 MiB），池保留峰值 2080.0 MiB，NVML 2816 MiB（池外 423 MiB）；
  峰值时活着的：`cudnn_conv` 输出 606.4 MiB、`ternary` 算子输出 514.5 MiB、
  区域前已有（参数/动量）170.5 MiB；按张量最大的是 9 个 `float32[64,64,56,56]` 共
  441.0 MiB；区域末尾缓存 1909.4 MiB 分在 122 个空闲块，碎片率 90%。

### Torch 模式 Transformers GPT-2（4 层，宽 256，batch 8×128，AdamW），5 步

稳态 10.00 ms/步。`jt.profile(python=True, sample_interval=0.0005)` 五步 63.48 ms：

- **host-bound**：设备只忙 22.8%；主机 61.1% 在 Python+构图，38% 在执行器规划/发射，
  等设备 0.7%；
- 设备时间：`cublas_matmul` 6.04 ms（41.7%）、`fused_adamw` 2.40 ms（16.6%）、
  `cublas_batched_matmul` 1.49 ms；
- 调用点：`transformers/pytorch_utils.py:122`（Conv1D 的 addmm）建了 800 个算子、发射
  5.80 ms、设备 5.32 ms；`transformers/integrations/sdpa_attention.py:83` 500 个算子；
- 显存：峰值 220.3 MiB；最大的是 `loss_utils.py:36` 处的 logits `float32[8,128,8192]`
  32 MiB。

### 重放的一步

`jt.graph_replay` 录成 CUDA graph 的 MLP 推理一步：报告显示 1 次设备图发射，6 个 kernel
（cuBLAS、融合算子）挂在 `cuda_graph_launch` 下，`GraphReplay` 计数增量
`{'replayed': 1, 'graph': 1}`；图外的输入/输出拷贝单列为"不属于任何算子"。旧工具在同一
场景报 0 行。

### 设备时间的独立校验

16 个连续 2048³ 矩阵乘（设备打满）时，CUPTI 给出的每次 `cublas_matmul` 设备时间落在
墙钟/次的 0.5–1.1 倍之间（回归测试 `test_device_time_agrees_with_a_device_bound_wall_clock`）。
显存峰值与池计数器逐字节相等（`test_the_traced_peak_is_the_pools_peak_and_names_what_was_live`）。

### 开销

同一进程、同一缓存，取三次最好：

| 负载 | 不画像 | 仅主机 | 默认 | 默认 + Python 采样 |
| --- | --- | --- | --- | --- |
| MLP 训练（device-bound，20 步） | 2.287 ms | 2.294 | 2.344（+2.5%） | — |
| Torch 模式 GPT-2（host-bound，10 步） | 10.10 ms | 10.31（+2%） | 11.86（+17%） | 12.16（+20%） |

host-bound 负载上主要是 CUPTI runtime API 记录的成本（correlation 需要它；driver API
记录已关闭，对归属没有影响——ResNet18 与 cuDNN 情形归属率仍为 100%）。

## 测试

修前失败、修后通过，均在 RTX 4090 上执行：

- `tests/runtime/test_step_profile.py`（12 项）：主机拆分之和等于墙钟、调用点、设备时间
  归属与校验、Chrome trace、显存峰值与活跃集、保留量峰值与重置、旧内存画像开启清零、
  重放可见、`profile_scope` 的重放警告、CPU 区域。修前：模块导入失败；把导入去掉后，
  旧画像清零、保留量峰值、重放警告三项失败。设备时间三项在找不到 CUPTI 时以
  "CUPTI ... cannot be loaded" 跳过。
- `compat/tests/torch/test_torch_profiler_facade.py`（4 项）：`key_averages`/`table`/
  Chrome trace、`schedule` 只保存一次、`torch.autograd.profiler`、`max_memory_reserved`
  与 `memory_summary`。修前 4 项全部失败。

门禁结果见提交说明。

## 没有做 / 没有验证

- jittor 环境默认没有 CUPTI；本次验证用独立下载的 `nvidia-cuda-cupti-cu12==12.2.142`。
  没有把它加进依赖，也没有在别的 CUDA 版本上验证 CUPTI 记录布局（只读 CUPTI 11 起稳定的
  字段）。
- ROCm、Ascend：主机拆分与显存走同一执行器路径，但没有在真机上跑；设备时间未实现。
- 原生 `jittor.models` 模型的调用点落在用户调用模型的行（`launch_origin` 跳过所有
  `jittor.*` 帧）；要按模块归属需要另一套规则。
- 主进程之外的 `StepCapture`（主检出里未提交）没有接入；它调用
  `jittor.profiling.register_replay_source(self)` 即可进入报告。
- 多个 Python 线程同时驱动执行器时的主机拆分没有验证。

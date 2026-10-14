# 异步错误诊断契约

已有的 `TraceData` 栈是**图的元数据**：它以节点为键，并且可以被 `trace_py_var` 关闭。
**它不是最近发射历史，也不能当作发射历史来读。**

原生 Runtime 拥有一个**每线程、有界的发射记录环**，每条记录包含：

- 算子 id / 名字，以及融合成员的 id；
- 发射时刻拷贝下来的 Python 文件与行号；
- 设备 / 流标识和一个单调递增的序号。

初始化之后，**发射路径上的记录是免分配的**。每个线程 64 条记录，同时注册的线程槽最多
64 个；已退役的槽保留其记录，直到后续写入者覆盖它们。带互斥保护的快照容忍并发写入者。
一个 Runtime 级的序号为记录排序；报告展示最近 16 条匹配项，并显式给出**覆盖数、
省略数和线程不可用数**。线程租约持有的是原生实现存储而非裸的 owner 指针，因此即使一个
独立的历史对象已被销毁，线程退出也是安全的。

Python 文件与行号在构造时捕获，**不持有 Python 帧、code 对象、Var 或 Op**。源位置是
内化的原生值；发射会在图节点可能被释放之前把位置拷进环里。源表最多容纳 8192 个位置，
且在 Op 可能仍持有 id 期间**绝不复用 id**。溢出会产生"来源不可用"和一个显式的
`unrecorded_origin_captures` 计数，而不是让动态源无界增长。超过 4096 字节的源键同样被
拒绝并计数，因此动态 `exec` 的文件名无法绕过该表的字节上限。

融合记录最多包含 8 个成员 id，被截断时给出显式总数。Python 路径与算子名的存储都有上限。
**没有可用 Python 帧时不会编造来源**：那条记录打印 `not-found`。

CUDA 错误会自动附上最近的发射候选。流等待与回读使用已知的流；没有已知流的设备等待或
event 等待会报告设备范围的候选，并把流标注为未知。

> **候选不是"哪次发射导致了故障"的证明**：异步错误可能由更晚的 API 调用报告，也可能
> 沿依赖传播。原始的 CUDA 错误会被保留。

`core.async_launch_history(backend, device, stream)` 是只读的原生快照，`stream=-1`
表示该设备上的所有流。受支持的 Python 观察入口是

```python
jt.introspection.diagnostics.launch_history(backend="cuda", device=0, stream=None)
```

`None` 会选中原生的"全部流"哨兵值而**不查询设备**。它校验后端拼写和非负的设备/流值，
过程中不初始化也不查询硬件后端（`-1` 是唯一的流哨兵值）。

## 没有被读到的设备故障

`Var.sync()` 与 `jt.sync_all()` 默认**不等设备**（`device_sync=False`），只把工作交出去。
最后一次读回结果之后才出错的 kernel，此前会让进程以 0 退出，stderr 上只剩一行指向 cuDNN
的 teardown 错误。现在正常退出时（无未捕获异常、退出码为 0），`jittor_exit` 调用
`core._wait_for_devices()`：先等异步执行队列，再同步加速器设备，**只在本进程确实向加速器
发射过工作时才同步**，纯 CPU 进程不会因此创建设备上下文。若这时抛出错误，打印完整诊断并以
退出码 1 结束；sticky 错误之后的 teardown 句柄销毁失败不再逐条打印。程序已经捕获过的
sticky 错误不会在退出时再报一次。需要在某处确认工作已完成时，用 `jt.sync_all(True)`。

带越界下标记录的设备故障以 `JittorIndexError`（`IndexError` 子类）抛出，第一行就是
`index ... is out of bounds ...`，CUDA 错误码随后。

**不要用 `trace_py_var>=2` 来测这个**：旧的图追踪器会在每个 Op 之后刻意同步。新的历史
与该追踪器无关，**不引入额外同步**。

本实现**无法在纯 CPU 主机上完整验收**：静态契约可以校验记录布局、容量上界、覆盖顺序和
报告格式，但最终门禁需要一次 CUDA 探测——注入一个异步错误并检查报告出来的 Python 行号。

建议的 CUDA 验收命令：

```bash
CUDA_VISIBLE_DEVICES=0 JITTOR_TORCH_SHIM=1 JT_SYNC=0 trace_py_var=0 \
  python -m pytest -q -s tests/backends/cuda/test_async_error_location.py
```

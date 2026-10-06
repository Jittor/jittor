# 运行时内省

- 状态：已实现
- 上次复查：2026-10-05（基线 `e3c369acb`）
- Owner：原生 Runtime 与测试基础设施维护者
- 复查触发：新增能力查询、策略字段或计数器，或 `jt.runtime.scope` 的写入语义变化时

`jt.introspection` 是给测试与诊断用的、**受支持的只读观察面**。它的各层读的都是已有的
原生/运行时服务，**不拥有第二份配置、后端注册表或计数表**；写入仍然只经过
`jt.runtime.scope(...)`。实现模块（`python/jittor/_runtime/introspection.py`）只依赖标准库
与已有的状态和能力定义，导入它不需要原生 bootstrap。

## 能力：哪些后端、几张卡、哪些库

```python
inventory = jt.introspection.capabilities.devices("cuda")
if inventory.capability.failed:
    raise RuntimeError(inventory.capability.reason)
if inventory.capability.enabled:
    print(inventory.count, inventory.devices)
```

- `backends()` 返回原生核心**认识**的后端名；`registered_backends()` 返回本运行时**注册了**
  的后端名。
- `backend(name)` 返回不可变的 `Capability`；`devices(name)` 返回不可变的
  `DeviceInventory`，含这条记录和一组 `Device(backend, index)`。编号是该后端当前可见设备
  命名空间里的**逻辑**编号，不是物理 GPU 标识；CPU 表示为 `Device("cpu", 0)`。
- 已注册后端的查询走 `core.backend_device_count(name)`：**不切换当前设备，也不拿
  `use_cuda` 当可用性判据**。所以一个默认在 CPU 上跑的运行时照样可以报告两张可见的
  CUDA 卡。
- 查询抛异常或原生返回负数时结果是 FAILED，附原因与错误证据，此时 `inventory.count` 是
  `None`——**绝不伪造一个 0**。已注册但可见设备为 0 的后端是 DISABLED；未注册的加速器
  沿用 `jt.capability` 的物理存在/构建证据，且不可能报告为可用。未知名字抛
  `ValueError`，不会默认当成 CPU。
- `libraries()` 与 `library(name)` 转发给已有的能力服务，但**总是 `load=False`**：
  UNPROBED 保持 UNPROBED，不会因观察而编译或加载可选扩展；这个命名空间没有
  `load=True`。需要初始化某个可选库的测试，把那次显式操作与观察分开写。
  `jt.capability` 原有的显式加载接口仍可用。

能力记录与设备清单**拒绝隐式布尔转换**。调用方必须区分 `.enabled`、`.failed`、
`.disabled`、`.absent` 与 `.unprobed`：一个坏掉的构建不能变成一次 skip。`present`
描述的是能力背后的证据，不承诺某个数值运算一定成功；设备清单是一次只读的驱动查询，
不是数值硬件测试。

## 生效策略

```python
view = jt.introspection.policy
before = view.snapshot()
with jt.runtime.scope(no_grad=not before.runtime["no_grad"]):
    assert view.runtime.no_grad != before.runtime["no_grad"]
```

- `policy.startup` 读 `jt.config`，`policy.runtime` 读 `jt.runtime.context`，都支持属性
  与映射两种访问。规范的运行时 flag 在 runtime 视图里；`cache_path`、编译器路径、架构
  选择这类启动设置在 startup 视图里。计数器不是策略。未知键抛 `KeyError`，未知属性抛
  `AttributeError`。
- 视图在原生 scope 变化之间保持"活"的。返回的字典与列表递归冻结为 mapping proxy 与
  tuple，嵌套的 `compile_options` 也改不动原生策略。`policy.snapshot()` 返回不可变的
  `PolicySnapshot(startup, runtime)`，值已与源分离，之后的策略变化不会改写旧快照。
  通过任何观察命名空间赋值或删除都抛 `AttributeError`。
- `runtime.use_cuda` 是默认的执行/放置策略，**不证明设备可用**，也不描述混合 CPU/CUDA
  图里每个张量的位置。可用性看能力层，单个张量的位置看张量放置 API（见
  [设备与放置](device-placement.md)）。

## 计数器

- `counters` 提供实时只读的 `exec_calls`、`allocator`、`held_vars`、`live_vars`、
  `live_ops`；`snapshot()` 返回分离的不可变 `CounterSnapshot`。计数的唯一归属者仍是原生
  liveness 查询，查询本身不额外持有张量引用。
- `allocator` 是 `AllocatorCounters(enabled, alloc_calls, allocated_bytes, free_calls,
  freed_bytes)`：原生统计分配器的**累计**分配/释放流量，不是 RSS，也不是当前占用。
  `enabled=False` 明示没有插桩。原生插桩模式变化会重置这些计数，**跨模式变化相减不是
  有效的流量测量**；这个 API 不提供 reset。
- `live_vars` 是进程全局的，带着进程里已缓存 Var 的"地板"。断言泄漏时先 `jt.clean()`
  再取基线，比较回到基线，而不是比较绝对的 0。
- 快照**不提交挂起的图、不同步设备、不跑 GC**。它们是先后几次观察，不是跨线程/跨设备的
  原子检查点；需要"工作已完成"边界的调用方在观察之外显式同步。`exec_calls` 数的是原生
  执行器计数，不是 kernel 发射次数，也不是耗时。

## 发射诊断

`jt.introspection.diagnostics.launch_history(backend="cuda", device=0, stream=None)`
返回描述最近发射候选的分离文本。`stream=None` 选该设备记录过的所有流，整数选那个原生
流句柄（不是序号）。查询不提交工作、不同步、不初始化后端；负的设备号与负的显式流句柄
被拒绝。原生历史拥有有界的、拷贝来的源码位置与记录，在图释放之后依然存在，不持有张量
或 Python 帧；文本会写明省略与截断。**候选不证明是哪一个运算引发了异步故障**；CUDA
错误处理会自动附上这段信息，调用方不必先查询。契约见
[异步错误诊断](../development/async-error-diagnostics.md)。

测试代码从旧的读法迁到这些 API 的对照表见[测试体系](../development/test-system.md)
的"运行时内省"一节。

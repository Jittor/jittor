# Torch 兼容原则

- 状态：已接受
- 上次复查：2026-10-05（基线 `e3c369acb`）
- Owner：兼容与核心维护者
- 复查触发：提出新的兼容层、回退策略或下游补丁机制

Torch 兼容是让有用的工作负载以 Jittor 语义和 Jittor 设备运行的手段。它不是把每个
Torch 命名空间都复制进一个模块的许可，也不是在计算已经逃到另一个运行时的情况下报告
成功的许可。前端如何构建见[分层架构](torch-architecture.md)与
[Torch 调用下沉](torch-lowering.md)；本页是评判工作的验收规则。

## 目标

兼容工作必须满足五道门：

- **G1，语义正确性：** 对受支持的输入，输出、dtype、形状、错误、梯度与状态转换都
  与文档化契约一致。
- **G2，设备执行：** 声称支持的 CUDA、ROCm 或 NPU 算子在该后端上执行；主机转换不算
  加速器实现。
- **G3，组合性：** API 通过真实应用使用的公开导入路径工作，包括注册与序列化边界。
- **G4，可维护性：** 一项能力只有一个 owner，失败可观测，版本特定行为被隔离。
- **G5，证据：** 回归测试以及（在性能要紧时）可复现的基准支撑结论。

## 归属决策

使用能正确拥有该行为的最低可复用层：

| 行为 | Owner |
| --- | --- |
| 张量语义、梯度、dtype/device 规则 | Jittor 核心或原生领域 |
| 通用神经网络算子 | `jittor.nn` |
| Torch 拼写或参数适配 | `jittor.compat.torch` |
| 可复用的导入/构建/补丁机制 | `jittor.compat` |
| Torch shim 运行时/部署与顶层 `torch` 表面 | `jittor.compat.shim` |
| Triton API 桥 | `jittor.compat.triton` |
| 单个项目的模型/版本粘合 | 可选适配器包（`adapters/`） |

如果多个下游项目需要同一个算子，就把算子提升到原生领域。如果它们只需要同一套补丁
机制，就提升该机制，同时把每个补丁留在核心之外。新下游库的分流判据在
`agent/skills/downstream-library-adaptation/SKILL.md`。

## 激活边界

普通 `import jittor as jt` 保留原生 Jittor 契约，不把 Jittor 发布为顶层 `torch`
模块。Torch 兼容由显式的 `jittor.compat.shim.activate()` 调用、`JITTOR_TORCH_SHIM=1`、
部署后的 `torch` 入口或历史路径 `jittor.torch_compat` 导入激活。仅仅把本地别名写成
`import jittor as torch` 不会激活 shim。每个激活入口都发布独立的 Torch 命名空间，
拥有自己的 Tensor、Parameter、Module 与 NN 类型。原生 Var/Op 图是共享的；原生 Var
方法保持不变。使用返回的 `result["torch"]`，或在激活后 `import torch`。

"原生即 Torch"的安装模式已移除。`activate(independent_namespace=False)`、
`JITTOR_TORCH_INDEPENDENT=0` 与直接调用 `compat.torch.install(jittor)` 都会显式失败；
保留下来的关键字只产生迁移错误。独立前端每次调用都会应用并恢复它的 autograd 策略；
原生 Var 行为不变。

Torch 测试与应用使用 `import torch`，并单独导入 `jittor` 以使用原生 flag、编译器
诊断或同步工具。`torch.tensor(...)` 取代已移除的 `torch.array(...)`；dtype 转换用
`tensor.to(dtype=...)`，而原生 Jittor 保持 `jt.float32(...)`。更新既有环境时要重新
部署旧的生成式 Torch 入口。

因此原生测试与兼容测试在独立进程中运行。原生测试不得依赖其他被收集测试安装的 Torch
包装；Torch 兼容测试必须证明其显式入口完成了所需的安装器图。

## 正确性规则

- 与独立 oracle 或书面契约比较，而不是与被测实现的另一个别名比较。
- 在相关时覆盖位置参数与关键字参数形式、dtype 提升、广播、空输入与非连续输入、设备
  选择、梯度与有状态行为。
- 不支持的契约抛出聚焦的错误。返回看似合理但错误的值永远不是可接受的回退。
- 兼容包装不得全局改变无关的原生 Jittor 语义，除非原生行为本身被修正并测试。
- 导入与安装器步骤是幂等的，并独立报告每个失败组件。

### 存储共享

Torch 的别名共享存储；Jittor 的 Var 不共享。这个差异在要紧的方向上是静默的——一次
写入落到临时张量而不是目标张量时什么都不会报告，很久以后才以错误数值的形式浮现。

- 基础索引切片是视图。`param[:rows] = w`、`param[:rows].copy_(w)` 与
  `param[:rows].data.copy_(w)` 都必须真正写到 `param` 上；分片权重加载器正是这样写的，
  丢掉写入会让参数停留在初始值而没有任何报告。
- `Tensor.numpy()` 在 Torch 里共享存储，在这里是拷贝。不要靠交出 Jittor 自己的缓冲区
  （`Var.data`）来模拟共享：图一旦重新物化该 Var，视图就脱离，于是第一次写入生效、
  之后每次写入都消失——比拷贝更糟，因为它看起来像是工作的。忠实的方向是写穿数组，
  用延迟图能看到的 setitem 把每次主机写入镜像回去。

## 设备与回退规则

主机侧 NumPy 可以作为测试 oracle 或明确记录为 CPU-only 的实现。它不得藏在声称支持
CUDA、ROCm 或 NPU 的 API 之后。只有以下条件全部成立时，后端回退才可接受：

1. 公开契约允许该回退；
2. 设备转移是显式且可观测的；
3. 梯度与 dtype 行为保持正确；
4. 性能影响有文档记录；
5. 测试能区分回退与原生执行。

后端可用性通过一次小的真实操作探测。环境故障与框架缺陷分别记录；两者都不会变成笼统
的 skip。硬件验收使用原生回退计数器，而不是日志匹配（见
[源码架构](../development/source-architecture.md)中的回退策略）。

## 性能规则

优化之前必须先保证正确性。性能结论使用预热、同步、固定形状/dtype、隔离缓存与精确
修订号。比较等价的执行模式，当变更影响 latency 或内存时两者都报告。见
[性能基准](../performance/benchmarking.md)。

优化必须保留可靠的回退与证明实际执行路径的方法。它不得为了提升头条数字而削弱错误
报告或测试覆盖。

## 兼容别名

别名保留导入；它们不保留重复的源码树。在可观测身份要紧时，遗留名字必须解析到规范
模块或可调用对象。每个新别名都要记录：

- 规范路径；
- 它服务的兼容承诺；
- 身份与导入顺序的测试；
- 可以退役的条件。

特别是，`jittor.torch_shim` 是规范包 `jittor.compat.shim` 的同对象遗留别名。部署后的
顶层 `torch` 表面委托给该规范运行时，而不是拥有第二份实现。

## 最低证据

兼容变更应包含最小适用集合：

- 聚焦单元测试与错误契约测试；
- CPU 参考对拍；
- 在支持时做梯度与二阶梯度检查；
- 在每个声称支持的后端上做 CPU 到加速器的对拍；
- 导入顺序、身份与重复安装测试；
- 面向集成的工作做真实下游 smoke 测试；
- 声称的快路径给出基准证据。

持续存在的限制记录到
[已知问题总账](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)，
并附 owner 以及一个能指示限制何时消失的测试或复现。可复现的维护者结论记录到
[结果](../results/index.md)。

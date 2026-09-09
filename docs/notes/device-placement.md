# 设备与放置

这篇讲清楚三件事：一个张量到底在哪张卡上、怎么让它换一张卡、以及什么时候
Jittor 会拒绝你。

## 张量的设备是它自己的属性

每个 `Var` 带一个 `device_id`，表示它所在（或将被计算于）的加速器编号，创建时
确定：

- 算子的输出**继承输入的设备**；
- 源算子（`array`、`random`、`zeros` 等）落在**当前设备**上；
- 被搬到主机内存的 Var 仍记得自己的设备，回到设备时用的还是它。

**当前设备**是 `jt.flags.device_id` / `jt.current_device()`，用 `jt.set_device(i)`
设置，或用 `jt.flag_scope(device_id=i)` 限定作用域。设置它会调用 `cudaSetDevice`
并让各库句柄切到该设备。**进程不会重启**，其它卡照常可见可用。

算子在**它的输出所在的设备**上执行：执行器先把该设备设为当前，再分配输出、发射
kernel。一次运行结束时会等待它触及过的每一张卡，并恢复调用者原本的当前设备。

## 换设备只有一条路：`device_copy`

`Var.to_device(i)` 就是 `device_copy` 算子，是数据改变设备的唯一途径。它：

- 在**目标设备的流**上执行，等待源设备上生产者的 event；源设备的流也要等这次拷贝
  完成才能复用那块显存；
- **可导**，梯度是一次反向拷贝；
- 在硬件允许的设备对上，每对只启用一次 peer access。

面向用户的写法：

```python
b = a.cuda(3)            # 复制到 3 号卡；已在 3 号卡则返回自身
b = a.to("cuda:3")       # 同上
c = b.cpu()              # 取回主机内存，返回新 Var，源不变
```

`.cpu()` 的目标缓冲分配在**主机侧**，不会在设备上再占一份等大显存。

## 混设备是构图期的错误

一个算子的输入落在不同卡上会**在建图时**报错，和 torch 一致：

```
Expected all inputs to be on the same CUDA device
```

这条检查在 `Op::init` 里，所以 `jt.grad` 新建的反向算子走同一条规则——**前向被拒绝
的组合，不会出现一个悄悄混设备的反向**。

唯一的例外对应 torch 的 CPU 标量：**既未完成计算、又带 `_is_scalar` 标记**的 Var
（`x * 2` 里的 `2`、梯度起始的 `1`）会跟随它遇到的操作数，连同它背后那一小段待定
子图一起走。两个条件缺一不可——只看元素个数会放过用户真实存放在某张卡上的单元素
张量；只看"未完成"会把用户特意建在 `cuda:0` 上、只是还没同步的 `jt.array` 悄悄
挪走，而 torch 在这里是报错的。

已知边界：`jt.zeros(n)` / `jt.ones(n)` 实现为 `unary(0).broadcast(n)`，`_is_scalar`
会穿过 broadcast，因此一个未同步的 `jt.zeros(1000)` **会**跟随另一张卡上的操作数，
而 torch 会报错。这一条是**接受**而非修补：它是编译期常量，任何一张卡上按位一致地
产生，没有用户算出来的数据被搬动。凡是真正携带数据的路径（多于一个元素的
`jt.array`，或任何已经算出来的值）仍然被拒绝。

## 每张卡各有一份的状态

| 资源 | 粒度 |
| --- | --- |
| 显存池 | 每设备一个；全局那个实例是 0 号卡的 |
| cuDNN / cuBLAS / cuSPARSE 句柄 | 每设备一个；全局名字始终指当前设备的句柄 |
| cuRAND 生成器 | 每设备一个，统一种子 |
| cuFFT plan 缓存 | 每设备一个缓存 |
| 同步 | 等待本次运行发射过的每一张卡 |
| NCCL | 走同一套设备切换，句柄与显存池保持一致 |

## Torch 兼容写法

以下都按 torch 的语义工作：`torch.cuda.device_count()`、`current_device()`、
`set_device()`、`torch.cuda.device(i)` 与 `device_of(tensor)` 上下文、
`with torch.device("cuda:1")` 作为新张量的默认设备、带真实编号的 `Tensor.device`、
`Tensor.get_device()`、`.to("cuda:N")` 与 `.cuda(N)`（编号不同则复制，就是本卡则
返回自身）、工厂函数的 `device="cuda:N"`（**直接在 N 上创建**，不是先建后搬）、
以及原地生效且保持参数对象标识的 `Module.to("cuda:N")`。不带编号的 `.to("cuda")`
表示当前设备，与 torch 相同。

## 不在范围内

- **流与事件**：只用每设备的默认流，`torch.cuda.Stream` 仍是空对象。
- **异构架构**：kernel 按 `query_cuda_cc` 探测到的计算能力编译，同一进程里混用不同
  架构的卡未做处理。
- **显存换出**（`save_mem`）仍假定 0 号卡。
- **非 CUDA 后端**：设备放置与后端选择是两个维度，见
  [多后端设计](../../refactor-wip/architecture/multi-backend-design.md)。

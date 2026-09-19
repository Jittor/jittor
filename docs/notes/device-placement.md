# 设备与放置

这篇讲清楚三件事：一个张量到底在哪张卡上、怎么让它换一张卡、以及什么时候
Jittor 会拒绝你。

## 张量的设备是它自己的属性

每个 `Var` 带一个 `device_id`，表示它所在（或将被计算于）的加速器编号，创建时
确定：

- 算子的输出**继承输入的设备**；
- 源算子（`array`、`random`、`zeros` 等）落在**当前设备**上；
- 被搬到主机内存的 Var 仍记得自己的设备，回到设备时用的还是它。

**当前设备**是 `jt.current_device()`，用 `jt.set_device(i)` 设置，或用
`jt.flag_scope(device_id=i)` 限定作用域。设置它会调用 `cudaSetDevice` 并让各库句柄
切到该设备。**进程不会重启**，其它卡照常可见可用。

`jt.flags.device_id` 是它背后的原始状态，两者不是同一个读数：进程启动时
`jt.flags.device_id == -1`，表示"还没有人选过"，而 `jt.current_device()` 在有加速器
可见时把这个哨兵值解析为 `0`。一旦调用过 `jt.set_device(i)`，两者都是 `i`。要判断
"当前在哪张卡上"一律读 `jt.current_device()`；`jt.flags.device_id` 只在需要区分
"未设置"与"显式设为 0"时才有意义（`tests/runtime/test_runtime_device_state.py`
直接写它）。

**没有加速器的构建**（`jt.get_device_count() == 0`，CPU-only 会话）里
`jt.current_device()` 返回 **-1**，不是 0：`src/runtime/device.h` 的契约是"当前
设备，没有可见加速器时为 -1"，回答 0 会声称存在一张 `get_device_count()` 说不
存在的卡。任何跨两种构建的代码都要按这个写，`tools/run_test_suite.py --backend
cpu|cuda` 两侧都要过。

`Var.device` 给出 `"cpu"` / `"cuda:N"` 这一对，`Var.device_id` 只给编号。一个还没
被执行的 Var 没有分配可问，于是 `device` 报告它*将会*落在哪里；唯一需要特别处理的
是待定的 `.cpu()`——它的目的地已经定了，但 `device_id` 故意保留来源设备（这样它才
能回去），所以 `x.cuda(3).cpu().device` 在同步前也读作 `"cpu"`，不是 `"cuda:3"`。

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
b = a.cuda("cuda:3")     # 同上，也接受带 .type/.index 的设备对象
b = a.to("cuda:3")       # 同上
b = a.to(device="cuda:3")
b = a.to("cuda:3", "float16")   # 换设备顺带换 dtype，顺序无关
b = a.to(other)          # 取 other 的 dtype 和设备，other 未同步也算数
c = b.cpu()              # 取回主机内存，返回新 Var，源不变
```

`.cpu()` 的目标缓冲分配在**主机侧**，不会在设备上再占一份等大显存。

不接受的写法，每一条都会**报错**而不是悄悄不动：

```python
a.to(1)            # TypeError：裸整数不是设备，torch 在这里同样报错
a.to("cuda1")      # TypeError：拼错的设备名不会被当成 dtype
a.cuda(-1)         # RuntimeError：设备编号不能为负
a.cuda(99)         # RuntimeError：超出可见设备数
```

（torch 兼容层比这里宽一点：`Tensor.to(1)` / `Module.to(1)` 在那边读作
`cuda:1`，因为在多卡推理栈里裸编号很常见，见
`docs/results/2026-09-14-vllm-omni-h3-enablement.md` 第 31 节。宽是可以的，
悄悄丢掉不行。）

**不带编号的 `cuda` 在两层里含义不同**，这是有意的：

| 写法 | 原生 `jt.Var` | torch 兼容层（`torch.Tensor`） |
| --- | --- | --- |
| `x.cuda()` / `x.to("cuda")` | 它**自己那张卡**（`device_id`），即"确保它在设备上"，host-park 的 Var 由此回到它原本的卡 | **当前设备**，与 torch 相同 |
| `x.cuda(1)` / `x.to("cuda:1")` | 1 号卡 | 1 号卡 |

原生这一条不是疏漏：`.cuda()` 在这里读作"把它放回它所属的加速器"，而 Var 一直
记得那是哪张卡。同一个进程里两种 Var 各按各自的规则走——`type(x) is jt.Var` 的走
上面这列，`torch.Tensor` 的走右边这列。

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

这条豁免**不越过一次显式的 `device_copy`**。`device_copy` 的输出带
`_placement_published`：调用者已经点名了它的设备，它就不再是"可以跟着别人走"的
待定标量。在此之前 `jt.ones(3).cuda(2)` 的输出继承了 `_is_scalar`，于是

```python
a = jt.ones(3).cuda(1); b = jt.ones(3).cuda(2)   # 两个都没同步
(a + b).sync()      # 整段跑在 cuda:1 上，而且 b.device 读作 "cuda:1"
```

——一次显式的 `.cuda(2)` 被无声地改写了，而同一个表达式只要先同步就会报错；
`.cpu()` 也一样丢。`_is_scalar` 本身保留在拷贝输出上，因为 dtype 提升要用它
（`x.cuda(1) * 2` 必须和 `x * 2` 提升得一样）。

## 每张卡各有一份的状态

| 资源 | 粒度 |
| --- | --- |
| 显存池 | 每设备一个；全局那个实例是 0 号卡的。用量按设备查：`jt.core.device_memory_used(i)`（在用字节）与 `jt.core.device_memory_reserved(i)`（池持有字节，含缓存空闲块），`i == -1` 是主机侧。`MemInfo.total_cuda_used` 是**所有卡之和**，回答不了"第 i 张卡上有多少" |
| 优化器状态 | 跟随它那个参数所在的设备：构造时按 `zeros_like(param)` 分配，`step()` 前再对齐一次，所以 `Module.to("cuda:1")` 之后原有的 optimizer 继续可用（torch 靠"第一次 step 才建状态"达到同样效果） |
| cuDNN / cuBLAS / cuSPARSE 句柄 | 每设备一个；全局名字始终指当前设备的句柄 |
| cuRAND 生成器 | 每设备一个，统一种子 |
| cuFFT plan 缓存 | 每设备一个缓存 |
| 同步 | 等待本次运行发射过的每一张卡 |
| NCCL | 走同一套设备切换，句柄与显存池保持一致 |

## Torch 兼容写法

以下都按 torch 的语义工作：`torch.cuda.device_count()`、`current_device()`、
`set_device()`（非 CUDA 设备按 torch 报 `ValueError`）、`torch.cuda.device(i)` 与
`device_of(tensor)` 上下文、`with torch.device("cuda:1")` 作为新张量的默认设备、
带真实编号的 `Tensor.device`、`Tensor.get_device()`、`.to("cuda:N")` 与 `.cuda(N)`
（编号不同则复制，就是本卡则返回自身）、工厂函数的 `device="cuda:N"`（**直接在 N
上创建**，不是先建后搬）、`x.new_*` 与 `*_like` 继承参考张量的设备、以及原地生效
且保持参数对象标识的 `Module.to("cuda:N")`。不带编号的 `.to("cuda")` 表示当前设备，
与 torch 相同。

`torch.device` 自身也按 torch 解析：`torch.device(1)` 是 `cuda:1`（**不是 cpu**），
负编号和不存在的设备类型都报错，`torch.device("cuda")` 与 `torch.device("cuda:0")`
是两个不同的对象。`torch.cuda` 的每设备查询——`get_device_name(i)`、
`get_device_capability(i)`、`get_device_properties(i)`、`mem_get_info(i)`、
`memory_allocated(i)` 一族、`current_stream(i)`/`default_stream(i)`、
`can_device_access_peer(a, b)`——都按传入的编号回答，不带参数时是**当前设备**而不是
0 号卡。`with torch.cuda.stream(s)` 同时也是一个设备上下文：块内当前设备是
`s.device`。

两处有意与 torch 不同，都写在这里而不是留给人去撞：

- `with torch.device("cuda:3"):` 会把 jittor 的当前设备也切到 3（torch 只改新张量
  的默认设备，`torch.cuda.current_device()` 不变）。这样块内那些不经过前端的原生
  分配也落在 3 上，否则它们会散到 0 号卡去。
- `Tensor.pin_memory()` 对已经在加速器上的张量返回一份**主机**拷贝而不是报错
  （torch 报错）。在这里一个没写 `device=` 的张量本来就在加速器上，报错会打断在
  torch 上跑得好好的暂存代码。缓冲区不是 page-locked，所以 `non_blocking` 的
  H2D 仍是同步的；`is_pinned()` 与之一致地返回 True。

## 不在范围内

- **流与事件**：只有每设备的默认流。`torch.cuda.Stream` 有设备身份并且
  `with torch.cuda.stream(s)` 会切设备，但所有逻辑流都序列化在同一条物理流上，
  `wait_stream`/`wait_event` 因此是诚实的空操作而不是缺失的实现。
- **异构架构**：kernel 按 `query_cuda_cc` 探测到的计算能力编译，同一进程里混用不同
  架构的卡未做处理。
- **显存换出**（`save_mem`）仍假定 0 号卡。
- **非 CUDA 后端**：设备放置与后端选择是两个维度，见
  [多后端设计](../../refactor-wip/architecture/multi-backend-design.md)。

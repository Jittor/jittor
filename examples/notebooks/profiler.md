---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  python: python3
  name: python3
---

# 性能与显存：先测量，再优化

模型跑通之后，"能不能更快、能不能放下更大的 batch" 就成了下一个问题。这一篇讲怎么在
Jittor 里**测量**，以及三个最容易踩的坑：

1. Jittor 是**异步**的，不 `sync` 直接掐表，量到的只是"派发"的时间；
2. 找一个算子慢不慢，要用 `jt.profiler`，而不是猜；
3. 第一次调用包含**即时编译**，它比稳定态慢一到两个数量级。

本篇全部在 CPU 上跑得动。涉及显卡的显存峰值测量单独标出。

## 1. 异步：必须先 `sync` 再掐表

Jittor 的算子是异步派发的：`c = a @ b` 这一行**只把任务交给后端**，不等它算完就返回。
所以直接掐表量到的是派发开销，不是计算时间。

强制等待结果的方式有三种，**它们不是一回事**：

| 写法 | 等什么 |
| --- | --- |
| `c.sync()` | 等 `c` 这一个张量的计算图算完 |
| `jt.sync_all()` | 等**所有**已派发的任务算完 |
| `jt.sync()` | **不带参数时什么都不等**，立刻返回 |

最后一条是个真实的坑：`jt.sync()` 要传张量才有意义（`jt.sync([c])` 等价于 `c.sync()`），
**空调用不会等**。写了它、看到它没报错，很容易以为自己在测量，其实什么都没测。

`.numpy()` 和 `.item()` 也会强制等待——但它们要把数据搬回主机，本身有开销，掐表时不要用它们
代替 `sync`。

先看这个差别有多大：

```{code-cell} ipython3
import time

import jittor as jt

jt.flags.use_cuda = 0        # 本篇用 CPU；有显卡时它会自动打开

a = jt.randn((512, 512))
b = jt.randn((512, 512))

# 不 sync：只量到派发
start = time.perf_counter()
for _ in range(50):
    c = a @ b
no_sync = (time.perf_counter() - start) / 50 * 1000

# sync：量到真正的计算
c.sync()                                  # 先把前面排队的任务清掉
start = time.perf_counter()
for _ in range(50):
    c = a @ b
    c.sync()
with_sync = (time.perf_counter() - start) / 50 * 1000

print("不 sync:     %.3f ms/次" % no_sync)
print("Var.sync():  %.3f ms/次" % with_sync)
```

两个数字都会随机器变化，这里要记住的是**量级差别**：不 `sync` 时你测的是"派发一次 512×512
矩阵乘"要多久，而不是"算完它"要多久。用这种方式对比两个算子，很容易得出相反的结论。

再把上面那张表的第三行亲眼看一次——空 `jt.sync()` 之后工作量还在：

```{code-cell} ipython3
work = jt.randn((1024, 1024))
work.sync()                              # 保证下面的对比从"没有待办"开始

start = time.perf_counter()
result = work @ work
jt.sync()                                # 空调用：不等
empty_sync = (time.perf_counter() - start) * 1000

start = time.perf_counter()
jt.sync_all()                            # 这一次才真的等
real_sync = (time.perf_counter() - start) * 1000

print("jt.sync()     立刻返回: %.2f ms" % empty_sync)
print("jt.sync_all() 真正等待: %.2f ms" % real_sync)
```

如果第一行和第二行差不多大，说明前面那次真的算完了；正常的机器上第二行会是第一行的几十倍，
差出来的那部分就是"你以为在等、其实没等"的时间。

> 一个例外：如果循环里下一步依赖上一步的结果，异步流水线会被自动"顶住"，两者就接近了。
> 但只要不是这种情况，就必须显式 `sync`。

## 2. 找瓶颈：`jt.profiler`

`jt.profiler.start()` / `stop()` 之间执行的算子会被记录：每个算子被调用了多少次、总耗时、
平均耗时，以及访存和计算吞吐。

```{code-cell} ipython3
from jittor import nn

model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
inputs = jt.randn((128, 256))

jt.profiler.start()
outputs = model(inputs)
loss = (outputs ** 2).mean()
jt.sync_all()                  # 先让任务真的算完，再停止记录（不是 jt.sync()）
jt.profiler.stop()
```

`stop()` 会直接打印一张表。表里的列值得认识一下：

| 列 | 含义 |
| --- | --- |
| `Count` | 这个算子被调用了多少次 |
| `TotalTime` / `AvgTime` | 总耗时 / 单次平均耗时 |
| `Input` / `Output` / `InOut` | 读 / 写 / 读写总带宽 |
| `Compute` | 计算吞吐 |

`jt.profiler.report()` 还能把同一张表**以数据结构的形式**取回来（返回的是"行"组成的
列表，第一行是表头），方便你自己排序或统计。注意表里打印的数字是排版过的（`6.4ms`），
而 `report()` 给的是**原始浮点数**，时间列的单位是**纳秒**：

```{code-cell} ipython3
rows = jt.profiler.report()
header = rows[0]
print("列:", header)
print("共 %d 个算子被记录" % (len(rows) - 1))

time_column = header.index("TotalTime")
count_column = header.index("Count")
slowest = sorted(rows[1:], key=lambda row: float(row[time_column]), reverse=True)[:5]
for row in slowest:
    name = row[0].split("«")[0].strip()
    if not name:                         # 融合算子的名字以 « 开头，取整串前一段
        name = row[0][:24]
    print("  %-24s count=%-3s total=%8.2f ms"
          % (name, row[count_column], float(row[time_column]) / 1e6))
```

读这张表的顺序是：**先按 `TotalTime` 排序**，看大头在哪；再看那个算子的 `Count` 是不是
被调用了太多次；最后看 `Input`/`Output` —— 如果带宽很低而耗时很高，往往不是算得慢，而是
访存模式不友好。

> 真实的模型里，瓶颈经常不在你以为的那个矩阵乘上，而在 reshape、transpose、concat 这些
> "搬运数据"的算子上。这正是要用 profiler 而不是猜的原因。

## 3. 第一次调用为什么特别慢

Jittor 在**第一次**执行某个算子时会即时编译它。编译一次的代价不小——在本篇这种小算子上
也是几百毫秒级别，远大于稳定态的单次执行。

编译结果**落在磁盘上**，位置是 `JITTOR_HOME` 下的缓存目录，按算子签名和 Jittor 版本索引。
所以"第一次"真正的含义是"这个缓存里第一次"，而不是"这个进程第一次"：

```{code-cell} ipython3
print("缓存目录:", jt.flags.cache_path)

x = jt.randn((448, 448))
timings = []
for _ in range(5):
    start = time.perf_counter()
    _ = x @ x
    jt.sync_all()
    timings.append((time.perf_counter() - start) * 1000)

steady = max(min(timings[1:]), 1e-9)      # 防止稳定态太快导致除零
print("每次耗时(ms):", ["%.1f" % value for value in timings])
print("第 1 次 / 稳定态 的倍数: %.1fx" % (timings[0] / steady))
```

这里的倍数**可能是几十也可能接近 1**，两种结果都是对的，原因就是缓存：

* 倍数大 → 这次真的付了编译钱；
* 倍数小 → 这个算子的编译产物早就在 `JITTOR_HOME` 里了（可能是你上一次运行留下的，
  也可能是本篇前面的单元格已经触发过）。

这带来三条实际规则：

* **基准测试要丢掉前几次**。只跑一次就下结论，量到的可能是编译器的时间，也可能不是——
  而你看不出是哪一种。
* **换缓存就换了一套数字**。`JITTOR_HOME` 或 `cache_name` 一变，第一次运行要重新编译，
  两次测量不能直接比较。跨机器的对比同理。
* 第一次慢不代表模型慢。想对比两个实现，就让它们在同一个进程、同一个缓存里都跑热之后再比。

> 判断"是不是在编译"可以打开日志：`jt.flags.log_v = 1` 会打印编译过程。它输出很多，
> 只在排查时开。

## 4. 显存与内存

Jittor 自己管着一层内存池：张量申请到的内存，释放后**留在池子里**备用，不一定还给系统。
所以有两个不同的数字，别混用：

* **占用（used）**：现在真正被活着的张量拿着的字节数；
* **预留（reserved）**：池子从系统要来的总字节数，等于占用加上缓存着备用的部分。

`device=-1` 表示主机（CPU）侧：

```{code-cell} ipython3
mib = 1024 ** 2
free_before = jt.core.device_memory_reserved(-1)

big = jt.randn((1024, 1024))          # 4 MiB 的 float32
big.sync()

used = jt.core.device_memory_used(-1)
reserved = jt.core.device_memory_reserved(-1)
print("活着: %.1f MiB   池子: %.1f MiB" % (used / mib, reserved / mib))
print("这一次申请让池子长了: %.1f MiB" % ((reserved - free_before) / mib))
```

`jt.get_mem_info()` 给的是另一组量——**容量**和池子的合计，它是**属性**不是方法，直接读：

```{code-cell} ipython3
info = jt.get_mem_info()
print("CPU  池内合计 %.1f MiB / 物理内存 %.0f MiB" % (info.total_cpu_used / mib,
                                                 info.total_cpu_ram / mib))
print("GPU  池内合计 %.1f MiB / 显存总量 %.0f MiB" % (info.total_cuda_used / mib,
                                                 info.total_cuda_ram / mib))
```

> `total_cpu_used` 是 Jittor 内存池的合计，**不是整个进程的常驻内存**。它答的是
> "我的张量占了多少"，答不了"这个进程吃了多少"。要看后者得用操作系统的工具。
>
> 还有一个坑：`total_cuda_used` 把**所有**显卡的池子加在了一起，所以它回答不了
> "第 3 号卡用了多少"。要按卡问，用 `jt.core.device_memory_used(3)`。

### 峰值：必须开画像，而且不要自己采样

想知道"峰值到哪里"，靠 Python 循环定期读 `device_memory_used` 是量不准的：采样器只有在
解释器放掉 GIL 时才跑得动，而一个训练步里那基本只发生在等待设备的时候。快速的一步跑完，
采样器读到的只是它碰巧撞上的某个中间值。

Jittor 内部记录了真正的峰值。它默认关闭，查询前必须先打开，否则会直接报错
`memory profiling must be enabled before querying its results`：

```{code-cell} ipython3
:tags: [cuda, skip-execution]

# 以下几行需要显卡
jt.flags.profile_memory_enable = 2     # 2 = 连 CPU 上的张量也一起统计

device = 0                              # 显卡编号
work = jt.randn((4096, 4096))
_ = (work @ work).sync()

peak = jt.core.get_peak_device_used_memory(device)
print("第 %d 号卡峰值: %.1f MiB" % (device, peak / mib))
```

这个峰值是**进程生命周期内的高水位，没有重置接口**。它答的是"打开画像以来最高到过多少"。
想单独量某个阶段，就在那个阶段开始时才打开 `profile_memory_enable`。

`jt.get_max_memory_info()` / `jt.display_max_memory_info()` 也受同一个开关约束，它们返回的
是"哪些张量活得最久"的文字报告。在纯 CPU 环境下 `get_peak_device_used_memory` 返回 0。

显存不够时，按代价从低到高依次尝试：减小 batch、用混合精度（见
[混合精度](mixed_precision.md)）、检查是否有张量被意外地一直持有（比如在循环里把
`loss` 累积进了一个 list）、最后才是换更小的模型。

## 5. 一次体检的完整流程

把上面的工具串起来，分析一个训练步骤可以固定成这四步：

1. **热身**：先跑几步，把编译开销排除掉。
2. **掐表**：整段跑完后 `jt.sync_all()`，测稳定态的每步耗时。
3. **分解**：用 `jt.profiler` 看这一步里时间花在哪些算子上。
4. **对照**：改一处，只改一处，再测一次。

```{code-cell} ipython3
optimizer = jt.optim.SGD(model.parameters(), lr=1e-3)
data = jt.randn((128, 256))


def one_step():
    loss = (model(data) ** 2).mean()
    optimizer.step(loss)


for _ in range(3):              # 1. 热身
    one_step()
jt.sync_all()

start = time.perf_counter()     # 2. 掐表
for _ in range(20):
    one_step()
jt.sync_all()
print("稳定态每步: %.3f ms" % ((time.perf_counter() - start) / 20 * 1000))

jt.profiler.start()             # 3. 分解
one_step()
jt.sync_all()
jt.profiler.stop()
```

## 6. 排查清单

**测量本身出错**，比对结果没有结论更常见。动手优化前先确认这几条：

- 掐表时等对了吗？用 `jt.sync_all()` 或 `张量.sync()`。**空的 `jt.sync()` 不等任何东西**——
  它不报错，只是直接返回，写错了很难发现。
- 丢掉前几次了吗？第一次包含即时编译。
- 改一处了吗？一次只改一个变量，否则无法归因。
- 量的是稳定态吗？样本太少时，单次测量在忙机器上波动很大，多测几次取最小值。
- 缓存是热的吗？换了 `JITTOR_HOME` 或 `cache_name` 会重新编译，两次数值不可直接比较。

数据加载也可能是瓶颈，而且它不在算子画像里——见 [数据加载与预处理](data_pipeline.md)。

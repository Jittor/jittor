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
  language: python
  name: python3
---

# JIT、算子融合与异步执行

同一个计算，为什么写成一串小算子反而更慢？这一篇讲 Jittor 的**执行模型**：

1. 它是**先建图、后执行**的——`y = x * 2` 这一行并不立刻算；
2. 执行时，一串小算子会被**融合**成一个核，这是它快的主要原因；
3. 一旦你把中间结果搬回主机，融合就被**打断**了，而且代价比想象中大。

这是 Jittor 区别于其它框架的地方，和 [元算子](meta_op.md) 同级。**怎么测量**（时间怎么掐、
显存怎么看）在 [性能分析器](profiler.md) 里，这一篇只讲机制。

本篇固定用 CPU，这样在任何机器上结论都一样；融合与异步的机制在 GPU 上是相同的。

## 1. 先建图，后执行

先看一个反直觉的事实：`c = a @ b` 这一行**几乎不花时间**，时间花在"要结果"的那一刻。

```{code-cell} ipython3
import time

import jittor as jt

jt.flags.use_cuda = 0        # 本篇用 CPU，这样在任何机器上结论都一样

a = jt.randn((2048, 2048))
b = jt.randn((2048, 2048))


def build_and_fetch():
    """建一次图，再把结果取回来，两段分别计时。"""
    start = time.perf_counter()
    out = a @ b                            # 只建图：登记一个节点，不算
    build = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    _ = out.numpy()                        # 取值：算出来并搬回主机
    return build, (time.perf_counter() - start) * 1000


build_ms, first_ms = build_and_fetch()     # 第一次：含即时编译
_, again_ms = build_and_fetch()            # 换一张新图再取：编译已就绪

print("建图              : %8.3f ms" % build_ms)
print("第一次取值        : %8.1f ms   <- 大部分是即时编译" % first_ms)
print("换张新图再取值    : %8.1f ms   <- 这才是计算本身" % again_ms)
```

建图是**微秒级**的：`out` 只是一个 `jt.Var`，它代表"这次矩阵乘"，而不是"乘完的结果"。
只有取值时才真的算——而第一次取值里还混着即时编译的成本（第 5 节专门讲它）。

还有一个细节值得知道：**同一个 `Var` 的结果会被留着**，再取一次不会重算。

```{code-cell} ipython3
c = a @ b
_ = c.numpy()                              # 先算一次

start = time.perf_counter()
_ = c.numpy()                              # 同一个 Var 再取一次
print("对同一个 Var 反复取值: %.2f ms" % ((time.perf_counter() - start) * 1000))

launches = 0
jt.profiler.start()
_ = c.numpy()
jt.sync_all()
jt.profiler.stop()
for row in jt.profiler.report()[1:]:
    try:
        launches += int(row[2])
    except (ValueError, IndexError):
        pass
print("这几毫秒里启动了几个算子:", launches)
```

两次取的是同一份已经算好的结果，这几毫秒只是把 16 MB 数据搬回主机。**想让它重算，就得重新
建图**（比如再写一次 `a @ b`，或者改动输入）——下一节会用到这一点。

**触发计算**的写法有：

| 写法 | 触发计算 |
| --- | --- |
| `v.numpy()` / `v.item()` | 是 |
| `v.sync()` / `jt.sync_all()` | 是（只等，不取数据） |
| `print(v)` / `repr(v)` | 是（要显示内容就得先算出来） |
| `float(v)` / `int(v)` | 是（只对单元素张量） |

这也解释了第一次跑 Jittor 代码为什么慢：**它得先把图建出来，才知道要编译什么**。

## 2. 一串算子会被融合成一个核

既然计算发生在取值的那一刻，那么"该编译哪些代码"就是此刻决定的——决策者能看到**整条链**。
这就是融合的来源：链上 12 个"乘 + 加"不必编译成 12 个核，可以合成一个。

数一数就知道。`jt.profiler` 记录的是**实际执行的核**，每个核被启动几次记在 `Count` 列：

```{code-cell} ipython3
x = jt.ones([512, 512])


def chain(t, n):
    """把 n 个"乘 + 加"串成一条链，返回链尾的 Var。"""
    y = t
    for _ in range(n):
        y = y * 1.1 + 0.01
    return y


def profile(build):
    """执行 build() 并返回被记录的算子行（第一行是表头，去掉）。"""
    jt.profiler.start()
    build()
    jt.sync_all()                  # 先算完，再停止记录
    jt.profiler.stop()
    return jt.profiler.report()[1:]


def kernel_stats(rows):
    """返回（核启动次数，链上一共几个 multiply）。

    每个名字里的 "OP:multiply" 出现几次，说明**这一个核**里装了几次乘法。
    """
    launches = 0
    multiplies = 0
    for row in rows:
        name = row[0]
        if "OP:" not in name:      # 只看真正干活的算子核
            continue
        times = int(row[2])        # Count：这个核被启动了几次
        launches += times
        multiplies += times * name.count("OP:multiply")
    return launches, multiplies


launches, multiplies = kernel_stats(profile(lambda: chain(x, 12).numpy()))
print("12 个乘加：启动了 %d 个核，核里一共 %d 次乘法" % (launches, multiplies))
```

`1` 个核里装下 `12` 次乘法——融合了。可以直接在算子的名字里读出来：把它打印出来会发现
`OP:multiply` 和 `OP:add` 各出现 12 次，还带着 `JIT:1` 的标记。

```{code-cell} ipython3
rows = profile(lambda: chain(x, 12).numpy())
longest = max(rows, key=lambda row: len(row[0]))[0]
print("最长的算子名有 %d 个字符" % len(longest))
print("  里面的 OP:multiply  :", longest.count("OP:multiply"))
print("  里面的 OP:add       :", longest.count("OP:add"))
print("  里面的 broadcast_to :", longest.count("broadcast_to"))
print("  带即时编译标记       :", "JIT:1" in longest)
```

这个长长的名字就是 Jittor 的**算子签名**：它把 dtype、形状维度、算子类型和整段融合后的
依赖关系编码了进去，用它当编译缓存的键。名字不同就是另一段代码，要重新编译。

> 融合的长度有上限（`jt.flags.fuse_op_limit`，默认 16）。特别长的链会被切成几个核，
> 所以你在分析器里看到的核数不一定正好是 1——**要看的不是行数，而是"乘法总共几次、
> 分在几个核里"**。

## 3. 打断融合的代价

融合的前提是"整条链一起编译"。任何一步把数据搬回主机（`.numpy()`、`.item()`、`print`），
这条链就被切断了：调度器下一次只能看见一两个算子，只能各编译各的。

```{code-cell} ipython3
def broken_chain(t, n):
    """每一步都把结果搬回主机再重新包成 Var——等于把图切成 n 段。"""
    y = t
    for _ in range(n):
        y = jt.array((y * 1.1 + 0.01).numpy())
    return y


for label, build in (("融合", lambda: chain(x, 12).numpy()),
                     ("打断", lambda: broken_chain(x, 12).numpy())):
    launches, multiplies = kernel_stats(profile(build))
    print("%s：核启动 %2d 次，乘法总共 %d 次" % (label, launches, multiplies))
```

两种写法的**计算量完全相同**（都是 12 次乘法），区别只在核的个数。而核的个数直接决定时间：

```{code-cell} ipython3
def timed(fn, repeat=15):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best * 1000


fused_ms = timed(lambda: chain(x, 12).numpy())
split_ms = timed(lambda: broken_chain(x, 12).numpy())

print("融合：一遍算完   %.3f ms" % fused_ms)
print("打断：每步搬回来 %.3f ms" % split_ms)
print("倍数：%.1fx" % (split_ms / max(fused_ms, 1e-9)))
```

具体的倍数随机器变化（这里从几倍到几十倍都出现过），但方向不变：**每次算子启动都有固定
开销，核越多越亏**。

最容易被忽视的打断方式是在循环里打印损失值：

```{code-cell} ipython3
def train_step_quiet():
    y = x
    for _ in range(12):
        y = y * 1.1 + 0.01
    return y


def train_step_logging():
    y = x
    for _ in range(12):
        y = y * 1.1 + 0.01
        float(y.mean().item())        # 只想看看这一步步的值
    return y


quiet_ms = timed(train_step_quiet)
logging_ms = timed(train_step_logging)
print("不打日志：%.3f ms" % quiet_ms)
print("打日志  ：%.3f ms" % logging_ms)
print("倍数    ：%.1fx" % (logging_ms / max(quiet_ms, 1e-9)))
```

顺带一个 `item()` 的坑：它**只对单元素张量有效**，直接对一张图调用会报错。

```{code-cell} ipython3
try:
    (x * 2).item()
except RuntimeError as exc:
    print("对 512x512 调 .item():", str(exc)[:80])
print("先归约成单元素就没问题:", float((x * 2).mean().item()))
```

所以训练循环里的正确做法是**按间隔记录**，而不是每一步都取：

```{code-block} python
for step in range(total_steps):
    loss = model(batch)                   # 全程留在设备上
    optimizer.step(loss)
    if step % 50 == 0:                    # 每隔 50 步才搬回一次
        print("step %d loss %.4f" % (step, float(loss.item())))
```

## 4. 异步：为什么"不等"是对的

Jittor 的算子是**异步派发**的：`y = x * 2` 只是把任务交给后端就返回了。所以融合才可能发生——
调度器手里总是攒着"还没执行的一段图"，而不是一个孤零零的算子。

这也意味着**你必须主动等待，才能量到时间**：

| 写法 | 等什么 |
| --- | --- |
| `v.sync()` | 等 `v` 这一张图算完 |
| `jt.sync_all()` | 等**所有**已派发的任务算完 |
| `jt.sync()` | **不带参数时什么都不等**，立刻返回 |

最后一行是真实的坑：写了它、没报错，很容易以为自己在测量，其实什么都没测。怎么量、量出来
怎么读，见 [性能分析器](profiler.md) 第一节；这里要记住的是**机制**：

> 融合要求调度器看到一整段图。你越频繁地 `sync` / `.numpy()` / `.item()`，它能看到的部分
> 就越短，能融合的就越少。

## 5. 编译一次，之后都走缓存

第一次执行一段新表达式要**即时编译**（生成 C++/CUDA 源码、调用编译器、加载动态库），
之后同一段代码直接用。下面这段表达式前面没出现过，所以第一次一定会编译：

```{code-cell} ipython3
def fresh(t, n=5):
    y = t
    for _ in range(n):
        y = y * 1.3 + 0.07
    return y


start = time.perf_counter()
_ = fresh(jt.ones([512, 512])).numpy()
first_ms = (time.perf_counter() - start) * 1000

start = time.perf_counter()
_ = fresh(jt.ones([512, 512])).numpy()
second_ms = (time.perf_counter() - start) * 1000

print("第一次（要编译）: %8.1f ms" % first_ms)
print("第二次（走缓存）: %8.2f ms" % second_ms)
```

缓存的键是前面那个**算子签名**：它由 dtype、算子组合和链的长度决定，**不含张量的具体形状**。
所以换成同样结构的新输入、甚至换一个形状，都不会再编译一次：

```{code-cell} ipython3
for shape in ((512, 512), (768, 768), (512, 512)):
    start = time.perf_counter()
    _ = fresh(jt.ones(shape)).numpy()      # 每次都是新的输入 Var
    print("%s: %.2f ms" % (shape, (time.perf_counter() - start) * 1000))
```

编译结果同时**写到磁盘**，路径在 `jt.flags.cache_path` 下，所以下一个进程也不用重编译：

```{code-cell} ipython3
import os

print("jt.flags.cache_path =", jt.flags.cache_path)
print("这个目录存在        :", os.path.isdir(jt.flags.cache_path))
```

> 具体快多少倍**不固定**：取决于这段代码以前有没有编译过、JITTOR_HOME 是不是复用、以及
> 机器上有没有其他编译任务在抢 CPU。看到"快了几十倍"和"几乎没差别"都是正常的——
> 关键是**第一次包含编译，别把第一次的数字当成性能**。
>
> 想让测量反映真实速度，就先**预热**：正式计时前把每个形状/分支都跑一遍，然后丢掉这些
> 数字重测。

## 6. 写高效循环的几条规则

- **别在训练循环里 `.item()` / `.numpy()` / `print(tensor)`**。按间隔记录，或者用
  `jt.profiler` 在循环外统计。
- **让算子留在设备上过完一整段**。中间结果搬回主机一次，融合就断一次。
- **别在 Python 里拼几千个算子的长链**。建图本身要花时间（在 512×512 上串 200 个算子，
  实测建图约 2.6 ms、每加一个算子约 13 µs），而且全在 Python 侧。能写成一次覆盖整批的
  张量运算，就不要在 Python 里循环。
- **别为了"看对不对"每步取值**。要调试就先跑几步、把值存下来，再统一打印。
- **计时前先预热**，并记住第一次包含编译。
- **比较两种写法时，先确认计算量一样**。融合会改变核的个数，不会改变计算量——
  像本文那样把"乘法总共几次"数出来，才知道对比是否公平。
- 想量时间或显存，见 [性能分析器](profiler.md)；想理解求导那一侧的行为，见
  [自动求导](autograd_details.md)。

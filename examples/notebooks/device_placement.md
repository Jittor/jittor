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

# 设备与驻留：数据到底在哪

这一章回答一个看起来简单、实际最容易出错的问题：**我的张量现在在 CPU 还是在显卡上？**

Jittor 的设备模型和 PyTorch 不同，不了解这个差异会写出「看起来对、跑起来慢」甚至
「读了一下就搬家了」的代码。本章把这些行为一次讲清，并给出可以自己验证的方法。

## 一个全局开关，不是每个张量一个设备

PyTorch 里每个张量自带设备，`a.cuda()` 只影响 `a`。Jittor 不是：**`use_cuda` 是一个
全局标志，它移动整张计算图**。

```{code-cell} ipython3
import jittor as jt

print("当前 use_cuda =", jt.flags.use_cuda)
with jt.flag_scope(use_cuda=0):
    a = jt.ones((4, 4))
    a.sync()
    print("标志关闭时新建的张量：", a.location())
```

这个差异有个直接后果：**在一台有显卡的机器上，`use_cuda` 默认就是开的**。所以
「我没有调用 `.cuda()`，所以它在 CPU 上」这个推断是错的——新建的张量会跟随全局标志。

想确认一段代码真的跑在 CPU 上，要显式关掉，而不是「不去打开」：

```{code-cell} ipython3
with jt.flag_scope(use_cuda=0):
    cpu_var = jt.ones((100, 100))
    cpu_var.sync()
    print("显式关闭后：", cpu_var.location())
```

## 三个不同的问题，三个不同的接口

关于「在哪」有三个接口，它们回答的**不是同一个问题**：

| 接口 | 回答 | 典型值 |
| --- | --- | --- |
| `location()` | 存储**当前实际**在哪 | `"none"` / `"cpu"` / `"device"` / `"disk"` |
| `device` | torch 风格的拼写 | `"cpu"` / `"cuda:N"` |
| `device_id` | 设备**索引**，不含种类 | `0`、`3` |

最常见的误用是拿 `device_id` 判断张量在不在显卡上。它只是索引：一个已经被复制回
主机的张量，`device_id` 仍然保留它来自哪张卡。

```{code-cell} ipython3
with jt.flag_scope(use_cuda=0):
    v = jt.ones((8, 8))
    v.sync()
    print("location =", v.location(), " device =", v.device, " device_id =", v.device_id)
```

## 惰性：还没算出来的张量没有「在哪」

Jittor 的计算是异步的。一个刚构造、还没被实体化的张量**没有存储**，所以它的
`location()` 是 `"none"`——这不是错误，是一个真实的第三种状态。

```{code-cell} ipython3
with jt.flag_scope(use_cuda=0):
    lazy = jt.ones((64, 64))
    print("sync 之前：", lazy.location())
    lazy.sync()
    print("sync 之后：", lazy.location())
```

`device` 在这种时候会回答「它将会落在哪里」——按当前的全局标志推断。这是一个承诺，
而实体化之后必须与之相符。写测试时如果想要确定的答案，先 `sync()`。

## 在设备之间搬运

```{code-cell} ipython3
:tags: [cuda, skip-execution]

# 需要真实的 CUDA 设备。
a = jt.ones((1000, 1000)).cuda()      # 当前设备
b = jt.ones((1000, 1000)).cuda(2)     # 指定 2 号卡
c = jt.ones((1000, 1000)).to("cuda:2")  # 等价拼写
a.sync(); b.sync()
print(a.device, b.device)
```

几点容易踩到的：

* **`.cuda("0")` 不合法。** 设备参数接受整数（`.cuda(0)`）或 torch 风格的字符串
  （`.to("cuda:0")`）；裸数字字符串解析不了。
* **`.to()` 不接受裸整数**，`.to(2)` 会报 `TypeError`；`.cuda(2)` 可以。
* **`.cpu()` 返回一个新张量，源张量不动。** 想替换就得接住返回值：`x = x.cpu()`。

```{code-cell} ipython3
:tags: [cuda, skip-execution]

g = jt.ones((512, 512)).cuda()
g.sync()
h = g.cpu()
h.sync()
print("副本：", h.location(), " 源：", g.location())   # cpu / device
print("device_id 跨 cpu() 保留源设备：", h.device_id == g.device_id)
```

`device_id` 跨 `.cpu()` 保留是**有意的**：它让 `x.cpu().cuda()` 能回到原来那张卡。
判断数据在哪要看 `location()` 或 `device`，它们跟随数据。

## 读一个设备张量会把它搬到主机

这是本章最重要的一条，也是最容易在性能上吃亏的地方。

`numpy()`、`repr()`（也就是在 REPL 里直接打印它）以及**读取单个元素**，都会把设备
张量的存储真的迁回主机。`location()` 随之变成 `"cpu"`，它说的是实话。

```{code-cell} ipython3
:tags: [cuda, skip-execution]

import time

x = jt.ones(10_000_000).cuda()   # 40 MiB
x.sync()
print("分配后：", x.location())

x.numpy()
print("numpy() 后：", x.location())   # cpu —— 存储真的搬走了

t = time.time(); (x * 2).sum().item(); jt.sync_all(True)
print("读过之后的第一次设备运算耗时 %.4fs（它要先把数据搬回去）" % (time.time() - t))

t = time.time(); (x * 2).sum().item(); jt.sync_all(True)
print("同样运算，这次已在卡上： %.4fs" % (time.time() - t))
```

在一台 RTX 4090 上实测，这两个数字相差约 **215 倍**，而且迁移期间显存里同时存在
两份数据。所以：

* 调试时随手加的一个 `print(x)` 会**永久改变**这个张量住在哪，代价落在后面某一行；
* 读**一个元素**（`x[0].item()`）会把**整个**张量搬走。

**归约结果不受影响。** `loss.item()` 里的 `loss` 是归约产生的新张量，读它不会动源
数据——训练循环里最常见的那个写法是安全的：

```{code-cell} ipython3
:tags: [cuda, skip-execution]

y = jt.ones((1024, 1024)).cuda()
y.sync()
total = y.sum().item()          # 新标量，不是 y 的视图
print("读归约结果之后，源仍在：", y.location())
```

想要一份主机副本又不想动原张量，用 `jt.array(x.numpy())`；只要数值就用
`x.sum().item()` 这类归约。

> 这条行为已作为缺陷记录在 `agent/manuals/known-issues.md` 的 `KI-MEM-002`：读是
> 查询，而这个查询改变了放置，API 里没有任何地方这么说。修复之后本节需要更新。

## 显存：释放了为什么 `nvidia-smi` 不降

Jittor 用缓存分配器。释放一个张量后显存回到**它自己的池子**，不还给驱动，所以
`nvidia-smi` 的读数不会下降。这是设计，不是泄漏。

```{code-cell} ipython3
:tags: [cuda, skip-execution]

big = jt.ones((512, 1024, 1024)).cuda()   # 2 GiB
big.sync()
del big
# 此时 nvidia-smi 仍然显示占用；再分配同样大小会复用这块缓存，不会继续上涨。
jt.clean()      # 真正把缓存交还驱动
```

另外，CUDA context 本身固定占用几百 MiB，只要进程活着就不释放——任何框架都一样。
判断有没有泄漏，要看**重复分配是否持续上涨**，而不是看单次释放后的绝对值。

## 自己验证的方法

不要靠猜。这几条随时可以自己测：

```{code-cell} ipython3
with jt.flag_scope(use_cuda=0):
    probe = jt.ones((16, 16))
    probe.sync()
    checks = {
        "location": probe.location(),
        "device": probe.device,
        "device_id": probe.device_id,
        "全局标志": jt.flags.use_cuda,
    }
    for key, value in checks.items():
        print("%-10s %s" % (key, value))
```

仓库里对应的契约测试是 `tests/core/test_var_residency_contract.py`，它把本章讲的每
一条都固定成断言，包括那些**目前还不成立、以严格 xfail 标记**的——修复之后会 XPASS
并强制更新记录。想知道当前哪些行为是有意为之、哪些是待修缺陷，读那个文件最快。

## 小结

* `use_cuda` 是全局的；在有显卡的机器上默认开启，「没调用 `.cuda()`」不代表在 CPU 上。
* `location()` 说当前实际在哪，`device` 是 torch 风格拼写，`device_id` 只是索引。
* 未实体化的张量没有驻留（`"none"`），需要确定答案就先 `sync()`。
* `.cpu()` 复制而不移动源；`device_id` 跨 `.cpu()` 保留是为了能回到原卡。
* **读一个设备张量会把它搬到主机**，下一次设备运算要付往返代价；归约结果除外。
* `nvidia-smi` 不下降通常是缓存分配器而非泄漏，`jt.clean()` 才交还驱动。

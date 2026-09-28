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

# 自动求导：梯度是怎么来的

[模型定义与训练](example.md) 里 `optimizer.step(loss)` 一行就把参数更新掉了。这一篇拆开
看它内部的那一步：**梯度**。

Jittor 的求导入口是 `jt.grad(loss, targets)`。它和 PyTorch 的 `backward()` 有一个重要区别：
**梯度不会留在张量上**，而是作为**返回值**给你。理解这一点，剩下的坑都好躲。

本篇全程 CPU，几秒钟跑完。

## 1. 一次求导

```{code-cell} ipython3
import numpy as np
import jittor as jt

jt.flags.use_cuda = 0

x = jt.array([1.0, 2.0, 3.0])
loss = (x * x).sum()

gradient = jt.grad(loss, x)
print("loss      =", float(loss.item()))
print("dloss/dx  =", gradient.numpy(), "  （应为 2x）")
```

`jt.grad(某个标量, 你要导的那个 Var)` 返回一个和 `x` 同形状的 Var。

**输出不是标量也可以**——`jt.grad` 内部等价于先把它加起来再求导：

```{code-cell} ipython3
v = jt.array([1.0, 2.0])
print("grad(v * 2, v)  =", jt.grad(v * 2, v).numpy())
print("grad(v * v, v)  =", jt.grad(v * v, v).numpy(), "  （应为 2v）")
```

第二行说明 `grad(v*v, v)` 拿到的是 `d(sum(v²))/dv`。如果你要的是一条链上某个非标量节点的
雅可比，这里不会给你——它给的是"把所有输出加起来的标量"的梯度。

## 2. 什么时候会拿到 0

这是这一篇最需要记住的一节。**当目标不在计算图上时，`jt.grad` 不会报错，它会返回全 0，
只在日志里写一行警告。**

```{code-cell} ipython3
a = jt.array([2.0])
b = (a * 3).stop_grad()          # 切断"b 来自 a"这层联系
print("stop_grad 之后:", jt.grad(b.sum(), a).numpy())

c = jt.array([2.0])
d = (c * 3).detach()             # detach 的效果一样
print("detach 之后   :", jt.grad(d.sum(), c).numpy())
```

两行都会打印 `[0.]`，而不是抛异常。上面那些 `[w] ... doesn't have gradient. It will be
set to zero` 的日志就是 jittor 在提示你——**但只有你去看日志时才知道**。

`jt.no_grad()` 是同一个道理：块里**根本不建计算图**，所以出来之后也导不了。

```{code-cell} ipython3
e = jt.array([2.0])
with jt.no_grad():
    out = e * 3
print("no_grad 里的结果:", out.numpy())
print("对它的梯度    :", jt.grad(out.sum(), e).numpy(), " <- 全 0")
```

所以：

| 写法 | 作用 |
| --- | --- |
| `v.stop_grad()` | 标记这个 Var 不参与求导（数据、常数用） |
| `v.detach()` | 同上，返回一个断开联系的新 Var |
| `with jt.no_grad():` | 块内**不建图**，适合推理和算指标 |

**看到 `[0.]` 先怀疑"是不是根本没连上图"，不要以为梯度真的是 0。**

## 3. 一张图能反传几次

`jt.grad` 的第三个参数是 `retain_graph`，**默认是 `True`**——也就是说默认允许你在同一张图上
再导一次：

```{code-cell} ipython3
w = jt.array([1.0, 2.0])
loss = (w * w).sum()
print("第一次:", jt.grad(loss, w).numpy())
print("第二次:", jt.grad(loss, w).numpy(), " <- 默认 retain_graph=True，还能导")
```

显式关掉之后再用同一张图，就会报错：

```{code-cell} ipython3
w2 = jt.array([1.0, 2.0])
loss2 = (w2 * w2).sum()
jt.grad(loss2, w2, retain_graph=False)
try:
    jt.grad(loss2, w2)
    print("第二次居然成功了")
except RuntimeError as exc:
    print("第二次报错:", str(exc)[:100], "...")
```

保留图是有代价的：整张图的中间结果都得留着，显存/内存会涨。**导完不再用就传
`retain_graph=False`**，或者干脆重新算一遍 `loss`——训练循环里就是这么做的。

## 4. 训练循环里的梯度在哪

`jt.grad` 是"函数式"的，梯度是返回值。那优化器训练时把梯度放哪了？

先看一个**不存在**的接口。`p.grad` 在 Jittor 里没有——调用它会给一段很清楚的提示：

```{code-cell} ipython3
from jittor import nn

jt.set_global_seed(0)
model = nn.Linear(2, 1)
optimizer = jt.optim.SGD(model.parameters(), lr=0.1)

data = jt.array([[1.0, 2.0], [3.0, 4.0]])
target = jt.array([[1.0], [0.0]])

try:
    model.weight.grad
except RuntimeError as exc:
    print(str(exc)[:420], "...")
```

Jittor 的梯度存在**优化器**里。而且必须在 `backward` 之后才能取：

```{code-cell} ipython3
loss = nn.mse_loss(model(data), target)

try:
    model.weight.opt_grad(optimizer)
except KeyError as exc:
    print("backward 之前取梯度:", repr(exc), " <- 还没算呢")

optimizer.backward(loss)                   # 求梯度，存进优化器
gradient = model.weight.opt_grad(optimizer)
print("backward 之后取到梯度:", [round(float(v), 5) for v in gradient.numpy().ravel()])
```

顺序是：**`backward` → `opt_grad` → `step`**。

## 5. 梯度会累加，`zero_grad` 负责清零

`backward` 是**累加**的。从零开始连续调两次，梯度就翻倍：

```{code-cell} ipython3
def one_backward():
    optimizer.backward(nn.mse_loss(model(data), target))
    return model.weight.opt_grad(optimizer).numpy().copy()


optimizer.zero_grad()                      # 从零开始，否则上一个单元格的梯度还在
once = one_backward()

optimizer.zero_grad()
one_backward()
twice = one_backward()

print("一次 backward :", once.ravel())
print("两次 backward :", twice.ravel())
print("正好是两倍吗  :", np.allclose(twice, 2 * once, atol=1e-6))

optimizer.zero_grad()
after = one_backward()
print("zero_grad 之后:", after.ravel(), " 回到单次:", np.allclose(after, once))
```

这段代码里 `zero_grad()` 出现在三个地方，正说明一件事：**缓冲区不清，梯度就一直累加**。

这正是 [模型定义与训练](example.md) 里 `optimizer.step(loss)` 不需要手动清零的原因：它在
`post_step` 里替你清了。而**自己拆开写 `backward` + `step` 时，必须自己调 `zero_grad()`**。

累加这个性质有个正经用途：**梯度累积**。显存装不下大 batch 时，可以分几个小批分别
`backward`，累加完再更新一次：

```{code-cell} ipython3
accumulator = jt.optim.SGD(model.parameters(), lr=0.1)
for _ in range(4):
    accumulator.backward(nn.mse_loss(model(data), target) / 4)   # 别忘除以份数

full = jt.optim.SGD(model.parameters(), lr=0.1)
full.backward(nn.mse_loss(model(data), target))

accumulated_grad = accumulator.param_groups[0]["grads"][0].numpy()
full_grad = full.param_groups[0]["grads"][0].numpy()
print("累积 4 小批:", accumulated_grad.ravel())
print("一次整批   :", full_grad.ravel())
print("一致       :", np.allclose(accumulated_grad, full_grad, atol=1e-6))
```

注意那个 `/ 4`：损失默认是对 batch 取平均的，累加 N 份就要除以 N，否则梯度会放大 N 倍。

## 6. 高阶导数

`jt.grad` 的结果本身也在图上，所以可以再导一次：

```{code-cell} ipython3
z = jt.array([3.0])
first = jt.grad((z ** 3).sum(), z)          # 3z² = 27
second = jt.grad(first.sum(), z)            # 6z  = 18
print("一阶导 3z²（z=3）:", first.numpy())
print("二阶导 6z （z=3）:", second.numpy())
```

## 7. 在梯度上加钩子

`Var.register_hook(fn)` 的回调**只接一个参数：梯度**。返回一个值就替换掉梯度，返回 `None`
就只是看看：

```{code-cell} ipython3
p = jt.array([0.0, 0.0])
q = p * jt.array([1.0, 2.0])

q.register_hook(lambda g: g * 2)            # 把梯度放大一倍
print("挂钩子后 dq/dp =", jt.grad(q.sum(), p).numpy(), "  （没挂钩子会是 [1. 2.]）")

seen = []
r = jt.array([1.0, 2.0])
s = r * r
s.register_hook(lambda g: seen.append(np.asarray(g).tolist()))   # 返回 None，不改梯度
print("只观察  ds/dr =", jt.grad(s.sum(), r).numpy(), " 钩子看到:", seen)
```

调试梯度异常（爆炸、全 0）的时候，用钩子打印比在每层中间插 `print` 干净得多。

## 8. 检查清单

- 拿到全 0 的梯度了吗？先确认目标真的在图上——中间有没有 `stop_grad()` / `detach()`，
  或者整段是不是在 `no_grad()` 里。
- 导完还要再用这张图吗？不用就传 `retain_graph=False`，别白白留着中间结果。
- 取参数梯度用 `p.opt_grad(optimizer)`，`p.grad` 不存在；而且要在 `backward` 之后取。
- 自己拆开写循环时调用 `zero_grad()` 了吗？`backward` 是累加的。
- 做梯度累积时除以份数了吗？
- 想量训练速度和显存，见 [性能与显存](profiler.md)；想存下训练状态，见
  [检查点与断点续训](checkpointing.md)；想弄清"算子在什么时候才真的执行"，
  见 [JIT、算子融合与异步执行](jit_and_async.md)。

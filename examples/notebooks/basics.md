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

# 算子与 Var

用 Jittor 训练模型只需要先理解两个概念：

* **Var**：Jittor 的基本数据类型，相当于其它框架里的张量。
* **算子**：作用在 Var 上的运算，写法与 NumPy 基本一致。

本篇只讲这两个。数据在 CPU 还是显卡上、什么时候会被搬动，是下一篇
[设备与驻留](device_placement.md) 的内容。

## Var

Var 是 Jittor 的基本数据类型。**Jittor 的计算是异步的**——写下一个表达式并不会
立刻算出结果，它只是进入计算图。想要拿到数值时用 `Var.numpy()`，它会同步等待
计算完成再返回。

```{code-cell} ipython3
import jittor as jt
a = jt.array([1,2,3], dtype="float32")
print (a)
print (a.numpy())
# Output: float32[3,]
# Output: [ 1. 2. 3.]
```

打印 Var 本身看到的是它的形状和类型；`numpy()` 才给出数值。异步是 Jittor 做算子
融合与调优的前提，代价是「什么时候真正算」不由你写代码的顺序决定——需要确定的
时机时显式调用 `sync()` 或 `numpy()`。

## 算子

算子的写法与 NumPy 类似。下面用 `jt.array` 创建两个 Var 再相加；打印出来可以看到
它们形状与类型相同。

```{code-cell} ipython3
import jittor as jt
a = jt.array([1,2,3], dtype="float32")
b = jt.array([4,5,6], dtype="float32")
c = a+b
print(a,b,c)
```

每个 `jt.xxx(Var, ...)` 形式的算子都有一个等价的方法写法 `Var.xxx(...)`，两者是
同一个实现，选顺手的即可：

```{code-cell} ipython3
c.max() # 等价于 jt.max(c)
c.add(a) # 等价于 jt.add(c, a)
c.min(keepdims=True) # 等价于 jt.min(c, keepdims=True)
```

具体有哪些算子可以在 `jt.ops` 上查看。在 `jt.ops.xxx` 里找到的算子同样可以用
`jt.xxx` 这个别名调用：

```{code-cell} ipython3
for name in ("add", "max", "min"):
    print(name, getattr(jt.ops, name))
```

## 接下来

* [设备与驻留](device_placement.md)：数据在哪、什么时候会被搬动。
* [模型定义与训练](example.md)：把这些拼成一个能训练的模型。

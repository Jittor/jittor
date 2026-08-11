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

# Basics: Op, Var

# 基本概念：Op, Var

To train your model with jittor, there are only two main concept you need to know:

要使用jittor训练模型，您需要了解两个主要概念：

* Var: basic data type of jittor
* Var：Jittor的基本数据类型
* Operations: Jittor'op is simular with numpy
* Operations：Jittor的算子与numpy类似

## Var
First, let's get started with Var. Var is the basic data type of jittor. Computation process in Jittor is asynchronous for optimization. If you want to access the data, `Var.numpy()` can be used for synchronous data accessing.

首先，让我们开始使用Var。Var是jittor的基本数据类型，为了运算更加高效Jittor中的计算过程是异步的。 如果要访问数据，可以使用`Var.numpy()`进行同步数据访问。

```{code-cell} ipython3
import jittor as jt
a = jt.array([1,2,3], dtype="float32")
print (a)
print (a.numpy())
# Output: float32[3,]
# Output: [ 1. 2. 3.]
```

## Op
Jittor'op is simular with numpy. Let's try some operations. We create Var `a` and `b` via `jt.array`, and add them. Printing those variables shows they have the same shape and dtype.

 Jittor的算子与numpy类似。 让我们尝试一些操作， 我们通过`jt.array`创建Var `a`和`b`，并将它们相加。 输出这些变量相关信息，可以看出它们具有相同的形状和类型。

```{code-cell} ipython3
import jittor as jt
a = jt.array([1,2,3], dtype="float32")
b = jt.array([4,5,6], dtype="float32")
c = a+b
print(a,b,c)
```

Beside that, All the operators we used `jt.xxx(Var, ...)` have alias `Var.xxx(...)`. For example:

除此之外，我们使用的所有算子`jt.xxx(Var,...)`都具有别名`Var.xxx(...)`。 例如：

```{code-cell} ipython3
c.max() # alias of jt.max(a)
c.add(a) # alias of jt.add(c, a)
c.min(keepdims=True) # alias of jt.min(c, keepdims=True)
```

You can inspect individual operations on `jt.ops`. Operations found as `jt.ops.xxx` can also be used through the `jt.xxx` alias.

您可以在`jt.ops`上查看具体的算子。 您在`jt.ops.xxx`中找到的操作也可以通过别名`jt.xxx`使用。

```{code-cell} ipython3
for name in ("add", "max", "min"):
    print(name, getattr(jt.ops, name))
```

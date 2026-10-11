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

# 模型定义与训练

上一篇讲的是 Var 和算子。这一篇把它们拼成一个能训练的模型。

一条完整链路：**定义模型 → 找到参数 → 算损失 → 更新参数 → 检查结果对不对**。
全程 CPU，后面的训练示例几秒钟跑完；第一次执行会额外花十几秒编译算子，那是 Jittor
的正常行为（见 [性能与显存](profiler.md)）。

## 1. 定义模型

继承 `jt.nn.Module`：把层写在 `__init__` 里，把前向计算写在 **`execute`** 里。

```{code-cell} ipython3
import numpy as np
import jittor as jt

jt.flags.use_cuda = 0
from jittor import nn


class Model(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(1, 16)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(16, 1)

    def execute(self, x):
        return self.fc2(self.act(self.fc1(x)))


model = Model()
print(model(jt.ones((3, 1))).shape)
```

两条规则：

* **前向方法叫 `execute`，不叫 `forward`。** 写 `forward` 的话，调用 `model(x)` 会报
  `NotImplementedError: Please implement 'execute' method`——报错很直接，不会静默跑错。
* **只有赋给 `self` 的层和 `jt.Var` 才会被登记**为这个模型的一部分。"登记"意味着它们会
  出现在 `parameters()` 和 `state_dict()` 里。下一节看一个不登记的写法。

## 2. 参数在哪

```{code-cell} ipython3
print("参数名（带点号路径）:")
for name, value in model.named_parameters():
    print("  %-14s %s" % (name, value.shape))

print("参数个数:", len(list(model.parameters())))
print("state_dict 的键:", sorted(model.state_dict().keys()))
```

三个方法，用途不同：

| 方法 | 给出什么 | 什么时候用 |
| --- | --- | --- |
| `parameters()` | 只有张量 | 交给优化器 |
| `named_parameters()` | `(名字, 张量)` | 想按名字处理某几层时 |
| `state_dict()` | `{名字: 张量}` | 存盘 / 载入检查点 |

### 放进普通 Python 容器里的层不会被登记

这是个安静的坑：

```{code-cell} ipython3
class WithList(nn.Module):
    def __init__(self):
        self.layers = [nn.Linear(2, 2), nn.Linear(2, 2)]   # 普通 list
        self.head = nn.Linear(2, 2)

    def execute(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


listing = WithList()
print("登记到的参数:", [n for n, _ in listing.named_parameters()])
```

`layers` 里的两层**一个都没登记**——`parameters()` 里没有它们，优化器也就永远不会更新
它们，而前向照样能跑。层多了以后这种错误极难发现。

需要循环堆叠用 `nn.Sequential`，或者用 `nn.ModuleList`：

```{code-cell} ipython3
print("nn.ModuleList 是否存在:", hasattr(nn, "ModuleList"))


class WithModuleList(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.head = nn.Linear(2, 2)

    def execute(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


listed = WithModuleList()
print("登记到的参数:", [n for n, _ in listed.named_parameters()])
```

## 3. 损失与优化器

```{code-cell} ipython3
rng = np.random.default_rng(0)
xs = rng.uniform(-1.0, 1.0, size=(64, 1)).astype("float32")
ys = (xs ** 2).astype("float32")          # 目标：拟合 y = x^2
X = jt.array(xs)
Y = jt.array(ys)

jt.set_global_seed(0)
model = Model()
optimizer = jt.optim.Adam(model.parameters(), lr=0.1)

loss = nn.mse_loss(model(X), Y)
print("初始损失:", float(loss.item()))
```

* 损失函数在 `jt.nn` 下：`nn.mse_loss` / `nn.l1_loss` / `nn.cross_entropy_loss`，
  也有对应的类形式 `nn.MSELoss()`。
* 优化器在 `jt.optim` 下：`SGD` / `Adam` / `AdamW` / `RMSprop` 等。
* **优化器要在模型建好之后创建**，因为它拿到的是一份参数列表。之后新建的层不会被它
  管理。

## 4. 训练循环

Jittor 的训练循环比 PyTorch 短一截——**一步到位，不需要手动清零梯度**：

```{code-cell} ipython3
for step in range(200):
    loss = nn.mse_loss(model(X), Y)
    optimizer.step(loss)                   # 求导 + 更新，一步完成
    if step % 50 == 0:
        print("step %3d  loss %.6f" % (step, float(loss.item())))

print("最终损失:", float(loss.item()))
```

`optimizer.step(loss)` 做的是：对 `loss` 求所有参数的梯度，按优化器规则更新参数，
然后把梯度清掉，准备下一步。**清零是它自己做掉的**（`post_step` 里调用 `zero_grad`），
所以循环里不需要写 `optimizer.zero_grad()`。

> 如果忘了这一步的语义、以为要自己清零，结果就是每个参数各自累加梯度——不会报错，
> 只是训练变得很怪。

## 5. 和 PyTorch 写法对照

PyTorch 那三步拆开的写法在 Jittor 里同样能用，而且**结果完全一样**：

```{code-cell} ipython3
jt.set_global_seed(0)
model_b = Model()
optimizer_b = jt.optim.Adam(model_b.parameters(), lr=0.1)

for step in range(200):
    optimizer_b.zero_grad()                # 显式清零
    loss_b = nn.mse_loss(model_b(X), Y)
    optimizer_b.backward(loss_b)           # 求梯度
    optimizer_b.step()                     # 只更新

print("两种写法的最终损失:")
print("  step(loss) 一次到位: %.8f" % float(loss.item()))
print("  zero_grad/backward/step: %.8f" % float(loss_b.item()))
print("  相同:", abs(float(loss.item()) - float(loss_b.item())) < 1e-8)
```

对照速记：

| | Jittor | PyTorch |
| --- | --- | --- |
| 前向方法名 | `execute` | `forward` |
| 训练一步 | `optim.step(loss)` | `zero_grad()` → `backward(loss)` → `step()` |
| 梯度累加 | `optim.backward(loss)` 累加，不自动清零 | 同 |

`backward` 是**累加**的，所以拆开写时必须自己调 `zero_grad()`；用一步到位的
`step(loss)` 时不需要。

## 6. 检查结果

损失降下来了，还要确认**预测值真的对**：损失函数只能说明平均误差，不能说明模型学到了
我们要的关系。

```{code-cell} ipython3
model.eval()                               # 切到推理模式
print("is_training:", model.is_training())

with jt.no_grad():                         # 不建计算图，省内存也更快
    prediction = model(X)

prediction_np = prediction.numpy().ravel()
max_error = np.abs(prediction_np - ys.ravel()).max()
correlation = np.corrcoef(prediction_np, ys.ravel())[0, 1]

print("最大绝对误差: %.4f" % max_error)
print("与目标的相关系数: %.6f" % correlation)
print("前 5 个预测:", np.round(prediction_np[:5], 3))
print("前 5 个真实:", np.round(ys.ravel()[:5], 3))
```

* `model.eval()` 切到推理模式（`is_training()` 变成 `False`），`model.train()` 切回来。
  对 Dropout、BatchNorm 这类层有影响。
* `jt.no_grad()` 里的计算**不建计算图**。做推理、算指标时都应该用它——否则每算一次就多
  一张图，显存会一直涨。

## 7. 常见错误清单

- 前向方法写成 `forward` 了吗？必须是 `execute`。
- 层放进普通 `list`/`dict` 了吗？那样不会被登记，优化器也管不到；用
  `nn.Sequential` 或 `nn.ModuleList`。
- 优化器是在所有层都建好之后创建的吗？晚建的层不在它的参数表里。
- 拆开写循环时调用 `zero_grad()` 了吗？`backward` 会累加梯度。
- 推理时用了 `jt.no_grad()` 吗？不用会一直建图。
- 只看损失就下结论了吗？损失小不等于预测对，像上面那样比一次数值。
- 想量训练速度和显存，见 [性能与显存](profiler.md)；想存下来接着训，见
  [检查点与断点续训](checkpointing.md)。

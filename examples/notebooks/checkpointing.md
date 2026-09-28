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

# 检查点与断点续训

训练跑到一半被抢占、或者你要调一个跑几小时的模型，都需要把"训练到哪了"存下来，之后
接着跑。这一篇讲清楚：

1. 训练的**状态**到底由哪几部分组成；
2. `jt.save` / `jt.load` 和 `Module.save` / `Module.load` 的分工；
3. 为什么只存模型权重会**静默地**毁掉续训；
4. `load_state_dict` 不报错并不代表载入成功。

全篇 CPU 上几秒跑完。

## 1. 训练状态有哪几部分

模型参数只是其中一部分。优化器自己也带状态——动量、步数计数器。看一个最小的例子：

```{code-cell} ipython3
import os
import tempfile

import numpy as np
import jittor as jt

jt.flags.use_cuda = 0
from jittor import nn

WORK = tempfile.mkdtemp(prefix="ckpt-tutorial-")


def new_model():
    jt.set_global_seed(0)                # 让每次新建的初始权重一致
    return nn.Linear(2, 3)


def new_optim(model):
    return jt.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


model = new_model()
optim = new_optim(model)

print("模型 state_dict 的键:", sorted(model.state_dict().keys()))

state = optim.state_dict()
print("优化器 state_dict 的顶层键:", sorted(state.keys()))
group = state["defaults"]["param_groups"][0]
print("参数组里的键:", sorted(k for k in group))
print("已走步数 n_step:", state["defaults"]["n_step"])
```

三个要点：

* `model.state_dict()` 是**有序字典**，键是带点号的参数名（`trunk.weight`、`head.bias`），
  值是 `jt.Var`。
* `optim.state_dict()` 只有一个顶层键 `defaults`，里面装着优化器的全部公开属性。
* 真正重要的是 `defaults["param_groups"]`：那里有 `values`——**动量缓冲**就在里面。
  还没走过步时只有 `params` 和 `values`；第一次 `step` 之后还会多出 `grads`。
  动量不在模型里，也不在 `model.state_dict()` 里——只在优化器里。

## 2. 存与读

`jt.save` / `jt.load` 存的是**任意字典**，所以它同时装得下模型和优化器：

```{code-cell} ipython3
path = os.path.join(WORK, "step2.pkl")


def train_steps(model, optim, x, y, count):
    for _ in range(count):
        loss = ((model(x) - y) ** 2).mean()
        optim.step(loss)
    return float(loss.item())


rng = np.random.default_rng(0)
x = jt.array(rng.normal(size=(8, 2)).astype("float32"))
y = jt.array(rng.normal(size=(8, 3)).astype("float32"))

train_steps(model, optim, x, y, 2)
jt.save({"model": model.state_dict(), "optim": optim.state_dict()}, path)

print("文件大小: %d 字节" % os.path.getsize(path))
loaded = jt.load(path)
print("读回来的顶层键:", sorted(loaded.keys()))
print("读回来的类型:", type(loaded["model"]["weight"]).__name__)
```

`jt.save` 直接 pickle 你给的字典（`Var` 也一起存），并在末尾附一个 **sha1 校验和**和
一个魔数标记。`jt.load` 会先验证它们——文件损坏时报
`ValueError: Pickle checksum does not match!`，而不是给你一堆错乱的数字。
这一点和下一节的静默失败正好相反。

所以读回来的还是 `jt.Var`，不是 numpy：这是 Jittor 自己的序列化格式，不要指望用别的工具
解开。要跨框架交换权重，用下面 `Module.save` 存出的纯权重，或者直接存成 `.pth`——
`jt.load` 和 `Module.load` 都能读 PyTorch 的 `.pth`。

只想存权重、不打算续训时（比如要发布预训练模型），用 `Module.save` / `Module.load` 更省事：

```{code-cell} ipython3
weights_only = os.path.join(WORK, "weights.pkl")
model.save(weights_only)

fresh = new_model()
fresh.load(weights_only)
print("只存权重的文件大小: %d 字节" % os.path.getsize(weights_only))
print("权重一致:", np.allclose(model.weight.numpy(), fresh.weight.numpy()))
```

## 3. 断点续训：一次对照实验

下面三组实验跑的是**同一段训练**，唯一变量是"续训时恢复了什么"。先跑一条不中断的基线：

```{code-cell} ipython3
def final_weights(model):
    return [round(float(v), 8) for v in model.weight.numpy().ravel()]


def run_continuous(steps):
    m = new_model()
    o = new_optim(m)
    train_steps(m, o, x, y, steps)
    return final_weights(m)


baseline = run_continuous(3)
print("连续 3 步      :", baseline)
```

第一组：第 2 步后存档，然后**重新建**模型和优化器，把两份状态都读回来，再走第 3 步。

```{code-cell} ipython3
def save_after(steps, path):
    m = new_model()
    o = new_optim(m)
    train_steps(m, o, x, y, steps)
    jt.save({"model": m.state_dict(), "optim": o.state_dict()}, path)


save_after(2, path)
state = jt.load(path)

m2 = new_model()
o2 = new_optim(m2)
m2.load_state_dict(state["model"])
o2.load_state_dict(state["optim"])
print("恢复后优化器的 n_step:", o2.n_step)
train_steps(m2, o2, x, y, 1)

resumed_both = final_weights(m2)
print("恢复模型+优化器 :", resumed_both)
print("与连续训练一致  :", np.allclose(baseline, resumed_both, atol=1e-6))
```

第二组：同样从第 2 步恢复，但**只读回模型权重**，优化器用全新的。

```{code-cell} ipython3
m3 = new_model()
o3 = new_optim(m3)                       # 全新优化器：动量为 0、n_step 为 0
m3.load_state_dict(jt.load(path)["model"])
print("没恢复优化器时的 n_step:", o3.n_step)
train_steps(m3, o3, x, y, 1)

resumed_model_only = final_weights(m3)
print("只恢复模型权重  :", resumed_model_only)
print("与连续训练一致  :", np.allclose(baseline, resumed_model_only, atol=1e-6))
print("差了多少        :", ["%.4f" % (a - b) for a, b in
                            zip(resumed_model_only, baseline)])
```

这就是这一篇最重要的一条结论：

> **只保存模型权重，不足以续训。** 优化器的动量和步数计数器是训练状态的一部分，
> 丢了它们，第 3 步走出来的方向就和原来不一样了——而程序**不会报任何错**，
> 它只是安静地训出一个不同的模型。

所以检查点里要放的东西，除了模型和优化器，通常还包括：当前 epoch / step、学习率调度器的
状态、随机数状态、以及你用来复现的实验配置。

## 4. 静默失败：`load_state_dict` 不报错

比"忘记存"更危险的是"以为读成功了"。`load_state_dict` 在参数对不上时**不会抛异常**：

```{code-cell} ipython3
class TwoLayer(nn.Module):
    def __init__(self):
        self.trunk = nn.Linear(2, 4)
        self.head = nn.Linear(4, 3)

    def execute(self, value):
        return self.head(nn.relu(self.trunk(value)))


reference = TwoLayer()
full = reference.state_dict()
print("完整 state_dict 的键:", sorted(full.keys()))

# 只给一部分参数
partial = {k: v for k, v in full.items() if k.startswith("trunk.")}
other = TwoLayer()
result = other.load_state_dict(partial)

print("load_state_dict 的返回值:", result)
print("没被覆盖的 head.weight 还是初始值:",
      not np.allclose(other.head.weight.numpy(), reference.head.weight.numpy()))
```

注意 `load_state_dict` 的返回值是 `None`——它**不**像 PyTorch 那样回报 `missing_keys`。
只载入了一部分，另一部分保持初始化时的随机值，没有任何异常，没有任何返回值可以检查。

形状对不上也一样，只是往日志里写一行：

```{code-cell} ipython3
wrong_shape = dict(full)
wrong_shape["trunk.weight"] = jt.randn((5, 5))       # 形状是错的：应是 [4, 2]

victim = TwoLayer()
victim.load_state_dict(wrong_shape)
print("形状不匹配也没有抛异常；上面那行 [e] 日志就是全部提示")
print("trunk.weight 仍是初始值:",
      not np.allclose(victim.trunk.weight.numpy(), reference.trunk.weight.numpy()))
```

**所以"没报错"不能当作"载入成功"。** 要确认，就得自己比：

```{code-cell} ipython3
def check_loaded(model, expected, atol=1e-6):
    """载入后自己核对一遍：每个键都在，且数值对得上。"""
    actual = model.state_dict()
    missing = [k for k in expected if k not in actual]
    mismatched = [k for k in expected
                  if k in actual
                  and not np.allclose(np.asarray(expected[k].numpy(),
                                                 dtype="float32"),
                                      actual[k].numpy(), atol=atol)]
    return missing, mismatched


missing, mismatched = check_loaded(other, {k: v for k, v in full.items()
                                           if k.startswith("trunk.")})
print("只载入 trunk 后：缺失", missing, "不一致", mismatched)

missing, mismatched = check_loaded(other, full)
print("拿完整存档去核对：缺失", missing, "不一致", mismatched)
```

最后一行说明 `other` 的 `head` 部分从来没被写进去——而这正是"**存完之后改了模型结构**"
的典型症状：

```{code-cell} ipython3
class Renamed(TwoLayer):
    def __init__(self):
        self.body = nn.Linear(2, 4)      # 原来是 trunk
        self.head = nn.Linear(4, 3)


renamed = Renamed()
renamed.load_state_dict(full)            # 用改名前的存档载入：不报错

missing, mismatched = check_loaded(renamed, full)
print("改名后：缺失", missing)
print("        不一致", mismatched)
```

`trunk.*` 在 `Renamed` 里根本不存在，被静默忽略；`head.*` 名字还在、形状也对，正常载入。
模型照跑，指标照算，只有 `body` 一直是随机初始化的——这就是"改了名字再载入旧检查点"
为什么会悄悄毁掉一次实验。

## 5. 检查清单

- 检查点里放了**优化器状态**吗？只放权重会让续训悄悄跑偏。
- 还缺什么？epoch / step、学习率调度器、随机数状态、实验配置——凡是"重跑会对不上"的都要存。
- 载入后核对过吗？`load_state_dict` 不报错、返回 `None`，形状不符只写日志。
  用上面 `check_loaded` 那样自己比一遍。
- 存完改过模型定义吗？改了名字或形状，对应参数会被静默跳过。
- 文件损坏能发现吗？`jt.save` 写了 sha1 校验和，`jt.load` 会校验；
  自己用 `pickle` 存的东西没有这层保护。

性能与显存怎么量，见 [性能与显存](profiler.md)。

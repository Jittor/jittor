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

# 从零实现扩散模型

扩散模型的教程通常用图像演示，结果真正的机制被几小时的 GPU 训练掩盖了。这一篇改用
一个**二维分布**训练完整的 DDPM：前向加噪、训练目标、采样器都和图像版本一样，但在 CPU 上
二十多秒就能跑完，而且结果**可以直接用数值检查**。

这一篇会：

* 写出前向过程的**一步到位**恒等式，并确认第 100 步时信号真的没了；
* 训练一个预测噪声的网络，并说明时间步嵌入为什么不能省；
* 采样出两个高斯团，用**三个读数**检验（均值方差、模式占比、到最近中心的距离）；
* 做两次消融，看到**两种不同的失败方式**——以及其中一个会让某个指标**变得更好看**；
* 给出一个判断原则：**哪个读数会被骗，就别只信它**。

> 全篇 CPU 可跑，两次训练合计约 **25 秒**。

```{code-cell} ipython3
import math
import time

import numpy as np

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
jt.set_global_seed(0)
rng = np.random.RandomState(0)

CENTERS = np.array([[-2.0, -2.0], [2.0, 2.0]], dtype="float32")
```

## 1. 数据：为什么用二维

两个分得很开的高斯团。任何**真的学到分布**的模型都必须把样本同时放到两个团上，
而且**方差也要对**。

用二维而不是图像，是因为这样每一个检验都是**数值**的——不需要看图说话。

```{code-cell} ipython3
def sample_data(n):
    which = rng.randint(0, len(CENTERS), size=n)
    return (CENTERS[which] + 0.3 * rng.randn(n, 2)).astype("float32")


data = sample_data(2000)
print("真实数据 mean", np.round(data.mean(0), 3), " std", np.round(data.std(0), 3))
```

## 2. 前向过程：一步到位

扩散分 `T` 步不断加高斯噪声。关键恒等式是**不需要真的一步步迭代**——记 `alpha_bar[t]`
为 `1 - beta` 的累积乘积，任意时刻的带噪样本可以一步得到：

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$$

```{code-cell} ipython3
STEPS = 100
betas = np.linspace(1e-4, 0.2, STEPS).astype("float32")
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas).astype("float32")

sqrt_ab = jt.array(np.sqrt(alpha_bar))
sqrt_1mab = jt.array(np.sqrt(1.0 - alpha_bar))

print("alpha_bar: 起点 %.4f  中点 %.4f  终点 %.6f"
      % (alpha_bar[0], alpha_bar[STEPS // 2], alpha_bar[-1]))
print("最后一步 sqrt(alpha_bar) = %.4f   sqrt(1-alpha_bar) = %.4f"
      % (math.sqrt(alpha_bar[-1]), math.sqrt(1 - alpha_bar[-1])))
print("=> 信号那项几乎归零，剩下的就是标准正态噪声")

x0 = jt.array(data[:4])
for t in (0, STEPS // 2, STEPS - 1):
    noisy = sqrt_ab[t] * x0 + sqrt_1mab[t] * jt.randn(x0.shape)
    print("t=%3d  样本 %s" % (t, np.round(noisy.numpy()[0], 3)))
print("\n这很重要：采样时不需要真的从噪声一步步「扩散回去」，")
print("只要从标准正态出发、按公式倒着走就行。")
```

## 3. 模型：预测噪声，且必须知道时间步

网络预测的是**被加上去的噪声**，而不是干净样本。它必须知道**该预期多少噪声**——
第 1 步和第 99 步的噪声强度完全不同——所以时间步用 Transformer 那套正弦特征编码后拼到
输入上。

```{code-cell} ipython3
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = jt.exp(-math.log(10000.0) * jt.arange(half).float() / half)
    angles = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    return jt.concat([jt.sin(angles), jt.cos(angles)], dim=-1)


class NoisePredictor(nn.Module):
    def __init__(self, dim=128, time_dim=32, use_time=True):
        super().__init__()
        self.time_dim = time_dim
        self.use_time = use_time
        in_dim = 2 + (time_dim if use_time else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, 2),
        )

    def execute(self, x, t):
        if not self.use_time:                       # 消融用：假装不知道时间步
            return self.net(x)
        return self.net(jt.concat([x, timestep_embedding(t, self.time_dim)], dim=-1))


print("参数量:", sum(int(np.prod(p.shape)) for p in NoisePredictor().parameters()))
```

## 4. 训练：目标只有一行

随机抽一个时间步，按该步的强度给样本加噪，让网络说出加的是什么噪声。

```{code-cell} ipython3
def train(use_time=True, steps=1501, batch=256, lr=2e-3):
    model = NoisePredictor(use_time=use_time)
    optimizer = nn.Adam(model.parameters(), lr=lr)
    start = time.time()
    for step in range(steps):
        clean = jt.array(sample_data(batch))
        t = jt.array(rng.randint(0, STEPS, size=batch).astype("int32"))
        noise = jt.randn(clean.shape)
        noisy = sqrt_ab[t].reshape(-1, 1) * clean + sqrt_1mab[t].reshape(-1, 1) * noise
        loss = ((model(noisy, t) - noise) ** 2).mean()
        optimizer.step(loss)
        if step % 500 == 0:
            print("  step %4d  loss %.4f" % (step, float(loss.numpy().ravel()[0])))
    return model, time.time() - start, float(loss.numpy().ravel()[0])


model, train_seconds, final_loss = train(use_time=True)
print("\n训练 %.1fs  最终 loss %.4f" % (train_seconds, final_loss))
```

**整个目标就是这么一行。** 没有对抗博弈（对比 [WGAN-GP](wgan_gp.md)），也不用像
[VAE](vae.md) 那样计算似然下界——扩散的训练稳定性正来自这里。

## 5. 采样：倒着走，并且每一步都注入噪声

采样是把链条反向跑一遍。每一步先减去预测出的噪声得到**后验均值**，然后**除最后一步外
再加回新的噪声**。这次注入是让采样器产出**分布**、而不是塌缩到一个点的关键。

```{code-cell} ipython3
@jt.no_grad()
def sample(model, n, inject_noise=True):
    x = jt.randn(n, 2)                              # 从标准正态出发
    for step in range(STEPS - 1, -1, -1):
        t = jt.array(np.full(n, step, dtype="int32"))
        predicted = model(x, t)
        mean = x - betas[step] / math.sqrt(1.0 - alpha_bar[step]) * predicted
        mean = mean / math.sqrt(alphas[step])
        if step == 0 or not inject_noise:
            x = mean
        else:
            x = mean + math.sqrt(betas[step]) * jt.randn(x.shape)
    return x.numpy()


def readings(name, generated):
    """三个读数：均值/方差、模式占比、到最近中心的距离。"""
    assign = np.argmin(((generated[:, None, :] - CENTERS[None]) ** 2).sum(-1), axis=1)
    distance = np.linalg.norm(generated - CENTERS[assign], axis=1)
    fraction = [(assign == 0).mean(), (assign == 1).mean()]
    spread = float(generated.std(0).mean())
    print("%-12s mean %s  std %.2f  模式占比 %s  到最近中心 %.3f"
          % (name, np.round(generated.mean(0), 2), spread,
             np.round(fraction, 3), distance.mean()))
    return fraction, distance.mean(), spread


print("真实数据就是这几个数该有的样子：std ≈ 2.0，模式占比 ≈ 0.5，距离 ≈ 0.37\n")
readings("真实数据", data)
fraction, distance, spread = readings("采样", sample(model, 2000))
assert min(fraction) > 0.35, "两个模式都要有样本"
assert 0.25 < distance < 0.6, "样本应当落在真实中心附近"
print("三种读数同时合理：模式均衡、位置正确、方差合适")
```

## 6. 消融一：去掉采样时的噪声注入

把 `inject_noise` 关掉，其它全部不动——**模型是同一个，只是采样器不注噪声了**。

```{code-cell} ipython3
print("%-12s %-22s %-9s %-16s %s" % ("", "mean", "std", "模式占比", "到最近中心"))
readings("真实数据", data)
injected_fraction, injected_distance, injected_spread = readings("正常采样", sample(model, 2000))
plain_fraction, plain_distance, plain_spread = readings(
    "不注噪声", sample(model, 2000, inject_noise=False))

print("\n模式占比变化：%s  ->  %s" % (np.round(injected_fraction, 3),
                                        np.round(plain_fraction, 3)))
print("标准差变化  ：%.2f  ->  %.2f" % (injected_spread, plain_spread))
print("到最近中心  ：%.3f ->  %.3f  ← 这一组反而更小" % (injected_distance, plain_distance))
assert min(plain_fraction) < 0.1, "不注噪声应当塌缩到一个模式"
```

**塌缩了。** 96% 的样本落到同一个模式，标准差从 2.0 掉到 0.6 附近——分布退化成一个点。

但这里有一个**必须指出的陷阱**：这一组的**「到最近中心的距离」反而更小**（对比上面两行）！
因为样本全挤在某个中心附近，离它当然近。

**如果只用「到最近中心的距离」这一个指标，这一组看起来更好。** 这就是为什么
[VAE](vae.md) 那一篇说「单看一个数会选错模型」——**判断生成质量必须同时看多样性与位置**。

## 7. 消融二：去掉时间步嵌入

时间步嵌入是模型知道「该预期多少噪声」的唯一途径。去掉它再训一遍。

```{code-cell} ipython3
model_no_time, seconds_no_time, loss_no_time = train(use_time=False)
print("\n去时间嵌入 训练 %.1fs  最终 loss %.4f" % (seconds_no_time, loss_no_time))
print("对照：正常训练最终 loss %.4f" % final_loss)

print("\n%-12s %-22s %-9s %-16s %s" % ("", "mean", "std", "模式占比", "到最近中心"))
readings("真实数据", data)
readings("正常采样", sample(model, 2000))
no_time_fraction, no_time_distance, no_time_spread = readings(
    "无时间嵌入", sample(model_no_time, 2000))

assert loss_no_time > final_loss * 1.5, "去时间嵌入的 loss 应当明显更高"
assert no_time_distance > distance * 1.5, "样本应当明显偏离真实中心"
print("\nloss: %.4f -> %.4f（差 %.1f 倍）"
      % (final_loss, loss_no_time, loss_no_time / final_loss))
```

**失败方式不一样。** 模式占比仍然是 `0.5 / 0.5`——两个模式都有人，方差也不算离谱——
但**到最近中心的距离涨了一倍多**（对比上面两行）。样本被**抹开**了，落在了两个中心
之间的空地上。

这正是「两个模式的中点放一个大团」那种最容易被统计量掩盖的失败：
均值和方差都可能看着没问题，只有**逐样本到最近中心的距离**能看出来。

## 8. 两种失败方式对照

把三次测量放在一起看，每个数都是本次运行实际测出来的：

```{code-cell} ipython3
rows = [
    ("正常", final_loss, injected_fraction, injected_distance, "——"),
    ("不注噪声", final_loss, plain_fraction, plain_distance, "距离指标被骗"),
    ("去时间嵌入", loss_no_time, no_time_fraction, no_time_distance, "模式占比被骗"),
]
print("%-12s %-8s %-14s %-12s %s" % ("消融", "loss", "模式占比", "到最近中心", "被掩盖的读数"))
for name, loss, frac, dist, fooled in rows:
    print("%-12s %-8.4f %-14s %-12.3f %s"
          % (name, loss, "%.2f / %.2f" % (frac[0], frac[1]), dist, fooled))
```

**两次消融各骗过了不同的一个读数**，这就是对照表要放在一起看的原因：

* 「不注噪声」那一组的**模式占比垮了**（约 `0.96 / 0.04`），但它的距离反而更小；
* 「去时间嵌入」那一组的**距离涨到两倍**，但它的模式占比完好。

所以判断生成模型至少要**同时**看：

1. **模式占比**——有没有丢模式；
2. **到最近中心的距离**——样本有没有落在真实位置上；
3. **标准差**——分布有没有退化。

**任何单个指标都能被某种失败方式骗过。** 这和 [VAE](vae.md) 里「KL = 0 时采样图看着
不错」是同一类纪律，只是这里的两个失败方式恰好各骗一个——更说明不能只看一个数。

## 9. 同一件事，用 PyTorch API 再写一遍

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

torch.manual_seed(0)

net_t = tnn.Sequential(tnn.Linear(34, 128), tnn.SiLU(), tnn.Linear(128, 2))
opt_t = torch.optim.Adam(net_t.parameters(), lr=2e-3)

clean_t = torch.tensor(sample_data(64))
t_t = torch.randint(0, STEPS, (64,))
noise_t = torch.randn(64, 2)
sqrt_ab_t = torch.tensor(np.sqrt(alpha_bar))
sqrt_1mab_t = torch.tensor(np.sqrt(1 - alpha_bar))
angles = t_t.float().unsqueeze(1) * torch.exp(
    -math.log(10000.0) * torch.arange(16).float() / 16).unsqueeze(0)
time_feat = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

noisy_t = sqrt_ab_t[t_t].unsqueeze(1) * clean_t + sqrt_1mab_t[t_t].unsqueeze(1) * noise_t
opt_t.zero_grad()
loss_t = ((net_t(torch.cat([noisy_t, time_feat], dim=-1)) - noise_t) ** 2).mean()
loss_t.backward()
opt_t.step()

print("torch 侧 loss %.4f" % float(loss_t))
print("梯度回传了吗:", net_t[0].weight.grad is not None)
print("no_grad 可用:", hasattr(torch, "no_grad"))
```

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 无梯度采样 | `@jt.no_grad()` 装饰器 | `with torch.no_grad():` |
| 时间步索引 | `sqrt_ab[t]` 直接高级索引 | 同左 |
| 训练一步 | `optim.step(loss)` | `zero_grad()` / `backward()` / `step()` |

## 10. 检查清单

```{code-block} text
1. 前向不必迭代             —— 用 alpha_bar 一步得到任意时刻的带噪样本
2. 时间步必须进模型         —— 去掉它 loss 涨一倍，样本被抹到两个中心之间
3. 采样每一步要注入噪声      —— 去掉它分布塌缩到一个模式（96/4）
4. 至少同时看三个读数        —— 模式占比、到最近中心距离、标准差
5. 某个指标变好要先怀疑      —— 「不注噪声」那组距离更小，但它是塌缩的
6. 判断要有常数基线          —— 这里是真实数据本身的三个读数
7. 二维不是玩具退化          —— 这里跑的是图像版本完全相同的目标与采样器
```

第 4 条和第 5 条是这一篇最实用的收获。**两个消融各骗过一个指标**这件事说明：
面对生成模型，**先问「哪种失败方式会让这个数变好看」，再决定要不要信它。**

## 接下来

* [VAE](vae.md)：另一条生成路线（似然下界），以及它自己的指标陷阱。
* [WGAN-GP](wgan_gp.md)：对抗路线的失败模式与可读的距离估计。
* [视觉 Transformer 训练](vit_training.md)：把 `NoisePredictor` 换成 UNet 就能上图像。
* 回到 [教程总目录](README.md)。

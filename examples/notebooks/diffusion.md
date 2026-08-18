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

# Denoising diffusion from scratch

# 从零实现扩散模型

Diffusion models are usually demonstrated on images, which hides the actual
mechanism behind hours of GPU time. This tutorial trains a complete DDPM on a
two-dimensional distribution instead: the same forward process, the same
training objective, the same sampler, but it converges on a CPU in under a
minute and you can check the result by looking at the numbers.

扩散模型的教程通常用图像演示，结果真正的机制被几小时的 GPU 训练掩盖了。本教程
改用一个二维分布来训练一个完整的 DDPM：前向加噪过程、训练目标、采样器都和图像
版本完全一样，但在 CPU 上一分钟内就能收敛，而且可以直接用数值检查结果。

## The data

## 数据

Two well-separated Gaussian blobs. Any model that has genuinely learned the
distribution must place its samples on both, with the right spread.

两个分得很开的高斯团。真正学到分布的模型必须把样本同时放到两个团上，而且方差
也要对。

```{code-cell} ipython3
import math

import numpy as np

import jittor as jt
from jittor import nn

jt.set_global_seed(0)
rng = np.random.RandomState(0)

CENTERS = np.array([[-2.0, -2.0], [2.0, 2.0]], dtype="float32")


def sample_data(n):
    which = rng.randint(0, len(CENTERS), size=n)
    return (CENTERS[which] + 0.3 * rng.randn(n, 2)).astype("float32")


data = sample_data(2000)
print("mean", np.round(data.mean(0), 3), "std", np.round(data.std(0), 3))
```

## The forward process

## 前向加噪过程

Diffusion adds Gaussian noise over `T` steps. The useful identity is that you
never have to iterate: with `alpha_bar[t]` the cumulative product of
`1 - beta`, any noisy sample is available in one shot as

扩散过程分 `T` 步不断加高斯噪声。关键恒等式是不需要真的一步步迭代：记 `alpha_bar[t]`
为 `1 - beta` 的累积乘积，任意时刻的带噪样本可以一步得到

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$$

```{code-cell} ipython3
STEPS = 100
betas = np.linspace(1e-4, 0.2, STEPS).astype("float32")
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas).astype("float32")

sqrt_ab = jt.array(np.sqrt(alpha_bar))
sqrt_1mab = jt.array(np.sqrt(1.0 - alpha_bar))

x0 = jt.array(data[:4])
for t in (0, STEPS // 2, STEPS - 1):
    noisy = sqrt_ab[t] * x0 + sqrt_1mab[t] * jt.randn(x0.shape)
    print("t=%3d  alpha_bar=%.4f  sample=%s"
          % (t, alpha_bar[t], np.round(noisy.numpy()[0], 3)))
```

By the last step the signal is gone and what remains is standard normal noise --
which is exactly what sampling will start from.

到最后一步，信号已经消失，剩下的就是标准正态噪声——这也正是采样的起点。

## The model

## 模型

The network predicts the noise that was added, not the clean sample. It has to
know *how much* noise to expect, so the timestep is embedded with the same
sinusoidal features a Transformer uses for position and concatenated to the
input.

网络预测的是被加上去的噪声，而不是干净样本。它必须知道该预期多少噪声，所以时间步
用 Transformer 里那套正弦特征编码后拼到输入上。

```{code-cell} ipython3
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = jt.exp(-math.log(10000.0) * jt.arange(half).float() / half)
    angles = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
    return jt.concat([jt.sin(angles), jt.cos(angles)], dim=-1)


class NoisePredictor(nn.Module):
    def __init__(self, dim=128, time_dim=32):
        super().__init__()
        self.time_dim = time_dim
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, 2),
        )

    def execute(self, x, t):
        return self.net(jt.concat([x, timestep_embedding(t, self.time_dim)], dim=-1))


model = NoisePredictor()
print("parameters:", sum(int(np.prod(p.shape)) for p in model.parameters()))
```

## Training

## 训练

The entire objective is one line: draw a random timestep, corrupt the sample by
that amount, and ask the network to name the noise. No adversarial game, no
likelihood bound to evaluate.

整个训练目标就是一行：随机抽一个时间步，按该步的强度给样本加噪，让网络说出加的
是什么噪声。没有对抗博弈，也不用计算似然下界。

```{code-cell} ipython3
BATCH = 256
optimizer = nn.Adam(model.parameters(), lr=2e-3)

for step in range(1501):
    x0 = jt.array(sample_data(BATCH))
    t = jt.array(rng.randint(0, STEPS, size=BATCH).astype("int32"))
    noise = jt.randn(x0.shape)
    noisy = sqrt_ab[t].reshape(-1, 1) * x0 + sqrt_1mab[t].reshape(-1, 1) * noise
    loss = ((model(noisy, t) - noise) ** 2).mean()
    optimizer.step(loss)
    if step % 500 == 0:
        print("step %4d  loss %.4f" % (step, float(loss.numpy().reshape(-1)[0])))
```

## Sampling

## 采样

Sampling runs the chain backwards. At each step the predicted noise is removed
to get the posterior mean, and fresh noise is added back for every step except
the last -- that injection is what makes the sampler produce a distribution
rather than collapsing to a single point.

采样是把链条反向跑一遍。每一步先减去预测出的噪声得到后验均值，然后除最后一步外
再加回新的噪声——正是这次注入让采样器产出一个分布，而不是塌缩到一个点。

```{code-cell} ipython3
@jt.no_grad()
def sample(n):
    x = jt.randn(n, 2)
    for step in range(STEPS - 1, -1, -1):
        t = jt.array(np.full(n, step, dtype="int32"))
        predicted = model(x, t)
        mean = (x - betas[step] / math.sqrt(1.0 - alpha_bar[step]) * predicted)
        mean = mean / math.sqrt(alphas[step])
        x = mean if step == 0 else mean + math.sqrt(betas[step]) * jt.randn(x.shape)
    return x.numpy()


generated = sample(2000)
print("generated mean", np.round(generated.mean(0), 3),
      " std", np.round(generated.std(0), 3))
print("real      mean", np.round(data.mean(0), 3),
      " std", np.round(data.std(0), 3))
```

The means and spreads agree, but a two-mode distribution can also be matched by
one wide blob in the middle. The honest check is whether both modes are
populated and whether samples land near a real center.

均值和方差对上了，但一个居中的大团也能凑出同样的统计量。真正该检查的是两个模态
是否都有样本，以及样本是否落在真实中心附近。

```{code-cell} ipython3
assignments = np.argmin(
    ((generated[:, None, :] - CENTERS[None]) ** 2).sum(-1), axis=1
)
distance = np.linalg.norm(generated - CENTERS[assignments], axis=1)
print("fraction in mode 0: %.3f" % (assignments == 0).mean())
print("fraction in mode 1: %.3f" % (assignments == 1).mean())
print("mean distance to the nearest true center: %.3f" % distance.mean())
```

## What to try next

## 接下来可以尝试

* Swap `NoisePredictor` for a small UNet and the 2-D data for MNIST; the
  training loop and sampler above stay unchanged.
* Try a cosine `beta` schedule instead of the linear one and compare how far the
  samples sit from their nearest centre.
* Predict `x0` instead of the noise and re-derive the posterior mean -- the two
  parameterizations are equivalent but behave differently early in training.

* 把 `NoisePredictor` 换成小 UNet，把二维数据换成 MNIST；上面的训练循环和采样器
  完全不用改。
* 把线性 `beta` 调度换成 cosine 调度，比较样本到最近中心的距离变化。
* 改成预测 `x0` 而不是噪声，再重新推导后验均值——两种参数化等价，但训练早期表现
  并不一样。

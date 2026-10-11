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

# 变分自编码器（VAE）：重建得好，不等于能生成

[WGAN-GP](wgan_gp.md) 用对抗的方式学一个生成模型。这一篇换一条路：不用判别器，
只要一个**编码器**和一个**解码器**，加上一项把隐空间「压成标准正态」的惩罚。

先说清楚它要解决什么问题。普通的自编码器（AE）只做一件事：把输入压成隐向量再还原，
目标是还原得越准越好。它在这个目标上可以做得非常好——但我们想要的是**生成**：
从隐空间里随便采一个点，解码器要能画出一张合理的新图。AE 做不到这件事，
因为**没有任何东西规定隐向量该长什么样**：

* 编码器可以把「横条」编码到 `(30, 2)`，把「竖条」编码到 `(31, 1.5)`——
  两个点在隐空间里几乎重合，中间那一大片区域则从未被训练过；
* 你在原点附近采一个点（正态分布的常驻区），解码器只会给出没训过的垃圾。

VAE 的做法是给隐向量**指定一个分布**：让编码器输出 `N(μ, σ²)`，并惩罚它偏离标准正态
`N(0, I)`。这样整个隐空间都被「用起来」了，采样的点才有意义。

这一篇会：

* 讲清楚 VAE 的损失为什么是「重构 + β·KL」，以及 KL 那一项的闭式解；
* 讲清楚**重参数化技巧**——为什么采样这一步可以求导；
* 用一个 β 阶梯（0 → 0.005 → 0.02 → 0.1 → 1.0）看到三种不同的失败方式；
* 用**三个读数**判断一个 VAE 到底有没有在正常工作，以及**只看重构质量会选错模型**。

> 这一篇全程用 MLP，不用卷积，CPU 上一分钟以内跑完。

## 1. 数据

还是那 4 类 16×16 模板（横条、竖条、方块、十字），像素在 `[-1, 1]`。
这一篇的读数需要知道每张图**属于哪个模板**，所以采样函数把标签一起返回。

```{code-cell} ipython3
import time

import numpy as np
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0        # 本篇用 CPU；有显卡时可以改成 1
jt.set_global_seed(0)
np.random.seed(0)

IMG = 16          # 图像边长
DIM = IMG * IMG   # 展平后的维度
LATENT = 16       # 隐向量维度
BATCH = 32
STEPS = 300

TEMPLATES = np.full((4, IMG, IMG), -1.0, dtype="float32")
TEMPLATES[0, 6:10, :] = 1.0            # 横条
TEMPLATES[1, :, 6:10] = 1.0            # 竖条
TEMPLATES[2, 5:11, 5:11] = 1.0         # 方块
TEMPLATES[3, 6:10, 2:14] = 1.0         # 十字：一横……
TEMPLATES[3, 2:14, 6:10] = 1.0         # ……加一竖
FLAT_TEMPLATES = TEMPLATES.reshape(4, -1)


def sample_real(batch):
    """返回 (标签, 图像)；图像是 (batch, IMG, IMG) 的 numpy 数组。"""
    which = np.random.randint(0, TEMPLATES.shape[0], size=batch)
    images = TEMPLATES[which] + np.random.normal(0, 0.05, (batch, IMG, IMG)).astype("float32")
    return which, images


labels, images = sample_real(4)
print("形状:", images.shape, " 标签:", labels)
print("取值:", float(images.min()), "~", float(images.max()))
```

## 2. 两个网络

编码器输出**两个**向量：均值 `μ` 和**对数方差** `logvar`。为什么不直接输出标准差 σ？

* σ 必须是正数，得再加一层约束（`softplus`、`exp`）；`logvar` 是任意实数，没有约束；
* 后面要用到 `σ²` 和 `log σ²`，直接给 `logvar` 就省掉了取对数这一步。

解码器就是一个普通的 MLP，最后一层用 `Tanh` 把输出压回 `[-1, 1]`。

```{code-cell} ipython3
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIM, 128), nn.ReLU())
        self.mu = nn.Linear(128, LATENT)         # 均值
        self.logvar = nn.Linear(128, LATENT)     # 对数方差

    def execute(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, DIM),
        )

    def execute(self, z):
        return jt.tanh(self.net(z))          # 压回 [-1, 1]


encoder, decoder = Encoder(), Decoder()
print("编码器参数量:", sum(p.numel() for p in encoder.parameters()))
print("解码器参数量:", sum(p.numel() for p in decoder.parameters()))

# 注意编码器返回的是两个东西
mu, logvar = encoder(jt.array(images.reshape(4, -1)))
print("mu:", mu.shape, " logvar:", logvar.shape)
```

## 3. 损失：重构 + β·KL

VAE 的损失有两项：

```{code-block} text
损失 = 重构误差 + β · KL( N(μ, σ²) ‖ N(0, 1) )
```

**重构项**要它还原得准；**KL 项**要编码器输出的分布贴近标准正态。两项的作用正好相反：
重构项希望每个样本都有自己的位置（方差越小越好），KL 项希望所有分布都挤回原点附近。

### KL 项有闭式解

对两个高斯分布，KL 散度可以写成一行：

```{code-block} text
KL = 0.5 · Σ ( μ² + σ² − 1 − log σ² )
```

不用采样估计，也不用手推。用 `logvar` 表示就是 `σ² = exp(logvar)`、`log σ² = logvar`：

```{code-block} text
KL = 0.5 · Σ ( mu² + exp(logvar) − 1 − logvar )
```

这个公式值得验一遍——用蒙特卡洛估计 `E_q[log q(z|x) − log p(z)]` 和它对比：

```{code-cell} ipython3
def kl_closed_form(mu, logvar):
    """KL(N(mu, exp(logvar)) ‖ N(0,1))，逐样本求和后对批取平均。"""
    return (0.5 * (mu * mu + jt.exp(logvar) - 1.0 - logvar).sum(dim=1)).mean()


def kl_monte_carlo(mu, logvar, samples=2000):
    """按定义估：E_q[log q(z|x) − log p(z)]。"""
    sigma = jt.exp(0.5 * logvar)
    z = mu + sigma * jt.randn((samples,) + tuple(mu.shape))   # (samples, batch, LATENT)
    log_q = (-0.5 * (((z - mu) / sigma) ** 2 + logvar + np.log(2 * np.pi)).sum(dim=2))
    log_p = -0.5 * (z * z + np.log(2 * np.pi)).sum(dim=2)
    return (log_q - log_p).mean()


jt.set_global_seed(0)
mu_demo, logvar_demo = jt.randn((32, LATENT)), jt.randn((32, LATENT))
print("闭式解      : %.4f" % float(kl_closed_form(mu_demo, logvar_demo).item()))
print("蒙特卡洛估计: %.4f" % float(kl_monte_carlo(mu_demo, logvar_demo).item()))
```

两个数在采样误差内一致。**训练时用闭式解**——它精确、便宜，而且不需要采样。

## 4. 重参数化技巧

问题来了：损失里要对 `z` 求导，而 `z` 是从 `N(μ, σ²)` 里**采样**出来的。
采样的那一步没有导数，梯度传不回去。

重参数化技巧把它拆成两步：

```{code-block} text
z = μ + σ · ε,    ε ~ N(0, 1)        σ = exp(0.5 · logvar)
```

随机性被挪到了 `ε` 上，`ε` 与网络参数无关；`μ` 和 `σ` 都是网络的确定性输出。
于是 `z` 对参数是可导的，`jt.grad` 能正常穿过这一步。**这就是 VAE 能端到端训练的全部秘密。**

```{code-cell} ipython3
def encode(encoder, flat):
    """编码并重参数化采样，返回 (z, mu, logvar)。"""
    mu, logvar = encoder(flat)
    sigma = jt.exp(0.5 * logvar)
    eps = jt.randn(mu.shape)
    return mu + sigma * eps, mu, logvar


z, mu_demo, logvar_demo = encode(encoder, jt.array(images.reshape(4, -1)))
print("z:", z.shape, " 它由 mu、exp(0.5*logvar) 和 eps 三部分算出")
print("重建形状:", decoder(z).shape)
```

## 5. 训练

`β` 是这一篇的主角，所以把它做成参数。`β=1` 是原论文的默认值；后面会看到，
在这个任务上它太大。

```{code-cell} ipython3
def train(beta=0.02, steps=STEPS, lr=1e-3, seed=0, report=100):
    """训练一个 VAE，返回 ((编码器, 解码器), 诊断记录, 用时秒数)。

    记录里每项是 (步数, 重构误差, KL, 平均 logvar)。
    """
    jt.set_global_seed(seed)
    np.random.seed(seed)
    enc, dec = Encoder(), Decoder()
    optim = jt.optim.Adam(enc.parameters() + dec.parameters(), lr=lr)

    history = []
    start = time.perf_counter()
    for step in range(steps):
        _, batch = sample_real(BATCH)
        flat = jt.array(batch.reshape(BATCH, -1))
        z, mu, logvar = encode(enc, flat)
        recon = ((dec(z) - flat) ** 2).mean()
        kl = kl_closed_form(mu, logvar)
        optim.step(recon + beta * kl)
        if step % report == 0 or step == steps - 1:
            history.append((step, float(recon.item()), float(kl.item()),
                            float(logvar.mean().item())))
    return (enc, dec), history, time.perf_counter() - start
```

## 6. 三个读数，缺一不可

怎么判断一个 VAE 训好了？这里要用三个读数，**只看其中一个都会被骗**：

| 读数 | 怎么算 | 它回答什么 |
| --- | --- | --- |
| 重构误差 | 还原图和原图的均方误差 | 还原准不准 |
| **模板命中率** | 还原图最近的模板，是不是原图自己的模板 | 还原是不是**真的**还原了（而不是画了个平均脸） |
| KL | 训练时那一项 | 编码器到底往隐空间里传了多少信息 |
| 先验采样 | 从 `N(0, 1)` 采样解码，看落在哪些模板上 | 能不能生成 |

```{code-cell} ipython3
def diagnose(model, samples=128):
    """返回 (重构MSE, 模板命中率, 先验采样的各模式计数, 采样的中位距离)。"""
    enc, dec = model
    labels, batch = sample_real(samples)
    flat = jt.array(batch.reshape(samples, -1))
    recon = dec(enc(flat)[0]).numpy()
    prior = dec(jt.randn((samples, LATENT))).numpy()

    def distance_to_templates(fakes):
        return ((fakes[:, None, :] - FLAT_TEMPLATES[None, :, :]) ** 2).mean(axis=2)

    recon_distance = distance_to_templates(recon)
    recon_mse = float(((recon - flat.numpy()) ** 2).mean())
    label_acc = float((recon_distance.argmin(axis=1) == labels).mean())

    prior_distance = distance_to_templates(prior)
    nearest = prior_distance.argmin(axis=1)
    close = prior_distance.min(axis=1) < 0.3
    histogram = {int(m): int((nearest[close] == m).sum()) for m in range(4)}
    return recon_mse, label_acc, histogram, float(np.median(prior_distance.min(axis=1)))


print("（这一格只定义函数，下一步开始训练）")
```

先看一个正常工作的 VAE（`β=0.02`）：

```{code-cell} ipython3
(vae_encoder, vae_decoder), history, seconds = train(beta=0.02)
print("训练用时 %.0f s" % seconds)
print("步数   重构误差      KL      平均logvar")
for step, recon, kl, logvar_mean in history:
    print("%4d   %.4f    %7.2f   %+8.2f" % (step, recon, kl, logvar_mean))

recon_mse, label_acc, histogram, prior_median = diagnose((vae_encoder, vae_decoder))
print()
print("重构误差    : %.4f" % recon_mse)
print("模板命中率  : %.2f" % label_acc)
print("先验采样    : 各模式 %s，中位距离 %.3f" % (histogram, prior_median))
```

三行读数都健康：重构几乎无损（`模板命中率 = 1.00`），KL 停在 3.6 左右（既不塌缩到 0，
也没有大到失控），先验采样四个模式都有。

## 7. β 阶梯：三种失败方式

现在把 β 从 0 扫到 1。**每一行只有 β 不同**，其它一切相同。

```{code-cell} ipython3
print("  β        重构MSE   模板命中率   KL      采样中位距离   采样各模式")
for beta in (0.0, 0.005, 0.02, 0.1, 1.0):
    model, ladder_history, _ = train(beta=beta)
    recon_mse, label_acc, histogram, prior_median = diagnose(model)
    print("  %-6s   %.4f    %.2f        %7.2f   %.3f          %s"
          % (beta, recon_mse, label_acc, ladder_history[-1][2], prior_median, histogram))
```

实测结果如下——表里每一列都在讲同一件事的不同侧面：

| β | 重构MSE | 模板命中率 | KL | 采样中位距离 | 采样各模式 |
| --- | --- | --- | --- | --- | --- |
| **0.0**（纯自编码器） | **0.0025** | 1.00 | **407.80** | 0.329 | `{0:11, 1:13, 2:11, 3:8}` |
| 0.005 | 0.0026 | 1.00 | 5.33 | **0.048** | `{0:24, 1:30, 2:31, 3:43}` |
| 0.02 | 0.0026 | 1.00 | 3.57 | 0.053 | `{0:23, 1:25, 2:33, 3:47}` |
| 0.1 | 0.0503 | 1.00 | 1.20 | 0.130 | `{0:26, 1:31, 2:28, 3:43}` |
| **1.0** | 0.3994 | **0.27** | **0.00** | 0.240 | `{3:128}` |

三种失败方式，从上往下看：

**β=0：重构最好，采样最差。** 重构 MSE 0.0025 是全表最低，模板命中率 1.00——单看
重构质量它会赢。但它的 KL 是 **407.80**：隐向量的分布离 `N(0, 1)` 极远
（平均 `logvar = −3.67`，相当于 σ ≈ 0.16，几乎退化成确定性编码）。
结果是采样中位距离 0.329，比任何一个 VAE 都差 6 倍以上。
**重建得最好，却最不能生成——这就是只看重构质量会选错模型的原因。**

**β=1：后验塌缩（posterior collapse）。** KL 精确停在 **0.00**，平均 `logvar = 0`——
编码器发现「什么都不传」最省事，于是把所有输入都映射到原点。解码器只好忽略 `z`，
改成一个「平均模板」的生成器：模板命中率掉到 **0.27**，而先验采样 128 张
**全部**落进模板 3。注意这里的陷阱：KL=0 时采样图看起来「有模有样」，
`采样中位距离` 也有 0.240——**不看 KL 和模式计数就发现不了它已经塌了**。

**中间三行才是 VAE。** KL 在 1.2 ~ 5.3 之间，重构依然完美，而采样中位距离降到
0.048 ~ 0.130，比 AE 好一个数量级。β 越大，KL 越小、采样越保守（0.005 → 0.1 时
采样距离从 0.048 升到 0.130），**β 就是这个权衡的旋钮**。

## 8. 看图：重建、采样、插值

```{code-cell} ipython3
import matplotlib.pyplot as plt

labels, batch = sample_real(8)
flat = jt.array(batch.reshape(8, -1))
with jt.no_grad():
    recon = vae_decoder(vae_encoder(flat)[0]).numpy().reshape(8, IMG, IMG)
    prior = vae_decoder(jt.randn((8, LATENT))).numpy().reshape(8, IMG, IMG)

fig, axes = plt.subplots(3, 8, figsize=(11, 4.5))
for column in range(8):
    axes[0, column].imshow(batch[column], cmap="gray", vmin=-1, vmax=1)
    axes[1, column].imshow(recon[column], cmap="gray", vmin=-1, vmax=1)
    axes[2, column].imshow(prior[column], cmap="gray", vmin=-1, vmax=1)
    for row in range(3):
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])
axes[0, 0].set_ylabel("真实", fontsize=9)
axes[1, 0].set_ylabel("重建", fontsize=9)
axes[2, 0].set_ylabel("先验采样", fontsize=9)
```

中间一行和第一行几乎一样（模板命中率 1.00 说的就是这件事），下面一行是**从没见过的**
样本——它们不是任何一张训练图，但都是合理的模板。

再看隐空间的连续性：在标准正态里取两个点，线性插值后解码。

```{code-cell} ipython3
with jt.no_grad():
    z_start, z_end = jt.randn((1, LATENT)), jt.randn((1, LATENT))
    steps = 8
    weights = jt.array(np.linspace(0.0, 1.0, steps, dtype="float32").reshape(steps, 1))
    line = z_start + (z_end - z_start) * weights
    morph = vae_decoder(line).numpy().reshape(steps, IMG, IMG)

fig, axes = plt.subplots(1, steps, figsize=(11, 1.8))
for column in range(steps):
    axes[column].imshow(morph[column], cmap="gray", vmin=-1, vmax=1)
    axes[column].set_xticks([])
    axes[column].set_yticks([])
axes[0].set_ylabel("插值", fontsize=9)
```

形状是连续变化的，没有突然跳变。**这是 KL 项买来的性质**：隐空间被压成了一个连贯的
区域，而不是一堆互不相干的孤岛。普通自编码器没有这个保证。

## 9. 怎么读这三个数

| 看到的现象 | 含义 | 该怎么办 |
| --- | --- | --- |
| 重构好、KL 很大（几百） | 这是自编码器，不是 VAE | 提高 β |
| KL 精确为 0，平均 logvar = 0 | **后验塌缩**，编码器什么都不传 | 降低 β |
| 重构变差、KL 很小但不为 0 | KL 挤得太紧 | 降低 β |
| 采样模式计数全挤在一两类 | 生成塌缩（和 GAN 同一类问题） | 检查 KL 和 β |
| 三个读数都健康 | KL 在个位数到几十，重构无损，采样四种模式都有 | 可以往下用了 |

核心那条经验：**「看起来像」不等于「能生成」。** β=1 那一行采样图的观感并不差，
但它的 KL 是 0、模式计数是 `{3:128}`。判断一个 VAE 必须同时看
**重构质量、KL、采样的分布**——任意一个单独看都会给出错误结论。
这和 [WGAN-GP](wgan_gp.md) 里「W 为负时不要相信它的数值」是同一类纪律。

## 10. 同一件事，用 PyTorch API 再写一遍

VAE 的损失里没有任何需要二次求导的地方，所以迁移起来比 WGAN-GP 简单得多。
唯一要留意的还是那两步确定性操作：`exp(0.5 * logvar)` 和采样。

激活必须在**创建任何张量之前**完成，所以这一节自成一段，不要和上面的原生代码混着写
（细节见 [用 PyTorch API 写 Jittor](torch_compat.md)）。

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

torch.manual_seed(0)

encoder_t = tnn.Sequential(tnn.Linear(DIM, 128), tnn.ReLU())
head_mu, head_logvar = tnn.Linear(128, LATENT), tnn.Linear(128, LATENT)
decoder_t = tnn.Sequential(
    tnn.Linear(LATENT, 64), tnn.ReLU(),
    tnn.Linear(64, 128), tnn.ReLU(),
    tnn.Linear(128, DIM), tnn.Tanh(),
)

flat_t = torch.tensor(batch.reshape(8, -1))
hidden = encoder_t(flat_t)
mu_out, logvar_out = head_mu(hidden), head_logvar(hidden)
# 和原生写法逐字对应：exp(0.5 * logvar) 是 sigma，randn_like 是 eps 的分布
z_t = mu_out + torch.exp(0.5 * logvar_out) * torch.randn_like(logvar_out)
out_t = decoder_t(z_t)
recon_t = ((out_t - flat_t) ** 2).mean()
kl_t = (0.5 * (mu_out * mu_out + torch.exp(logvar_out) - 1.0 - logvar_out).sum(dim=1)).mean()
loss_t = recon_t + 0.02 * kl_t
loss_t.backward()
print("torch 侧的重构误差: %.4f" % float(recon_t))
print("torch 侧的 KL      : %.4f" % float(kl_t))
print("梯度回传到编码器了吗:", head_mu.weight.grad is not None)
```

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 采样噪声 | `jt.randn(mu.shape)` | `torch.randn_like(logvar)` |
| 从 logvar 得 σ | `jt.exp(0.5 * logvar)` | `torch.exp(0.5 * logvar)` |
| 一步训练 | `optim.step(loss)` | `zero_grad()` / `backward()` / `step()` |
| 求导 | 自动，`optim.step` 内部完成 | 显式 `loss.backward()` |

## 11. 检查清单

```{code-block} text
1. 编码器输出 mu 和 logvar 两个向量        —— 不是 sigma
2. 采样写成 z = mu + exp(0.5*logvar) * eps  —— 重参数化，不能直接采样
3. KL 用闭式解，不用采样估计                 —— 已验证与蒙特卡洛估计一致
4. 同时看重构质量、KL、采样分布               —— 任一单独看都会被骗
5. KL 精确为 0 就是后验塌缩                  —— 降 beta
6. KL 上百就是普通自编码器                    —— 升 beta
7. 采样噪声只在训练时用；检查生成时用先验采样   —— N(0, 1)
```

到这里，生成模型这一组的三条路都走过了：对抗（`conditional_gan.md`、`dcgan.md`）、
距离（`wgan_gp.md`）、似然下界（本篇）。三条路的失败模式不一样，但判断它们的方法是一样的：
**先想清楚「什么数能证明它学会了」，再去看那个数。**

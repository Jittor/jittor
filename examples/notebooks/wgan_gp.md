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

# WGAN-GP：把「判别器有多强」换成一个能读的数字

[深度卷积生成对抗网络](dcgan.md) 的第 7 节留下了一个让人不安的结论：判别器的准确率
**两个极端都是坏消息**，所以它只能当趋势看，不能当读数用。第 9 节又证明更糟的事——
判别器坏掉时，`G损失` 反而更低。

问题的根源是损失函数本身。GAN 用二元交叉熵，判别器学的是「真 / 假」这个**分类问题**。
分类问题的损失衡量的是分类器的好坏，**不是生成样本的好坏**：判别器可以一边把分类做得更准，
一边让生成器更没方向。这两件事在交叉熵里纠缠在一起，分不开。

WGAN 换掉了这把尺子：让判别器（改叫 **critic**）输出一个**没有范围限制的实数**，
用「真样本的平均分 − 假样本的平均分」当作两堆分布之间的距离估计。这个量叫
**Wasserstein 估计**，简称 W。它的好处是：**当 critic 训练得足够好时，W 本身就是一个
距离**，于是「生成器变好」等价于「W 变小」——训练的好坏第一次有了直接读数。

这一篇会：

* 讲清楚 W 是什么，以及**为什么必须给 critic 加 Lipschitz 约束**才能让它有意义；
* 用 `jt.grad` 把梯度惩罚（gradient penalty）写出来，并指出一个**会把惩罚项悄悄变成
  常数**的写法陷阱；
* 跑通 WGAN-GP，并看到 `|∇|` 被拉到 1 附近——约束真的在起作用；
* 用一次只改 critic 学习率的对照实验，看到 **W 的符号**能正确区分「学到了」和「没学到」；
* 顺带看到「识别率高但模式不均衡」这种单个数字掩盖不了的情况。

> 这一篇和 `dcgan.md` 用同一套网络、同一份数据，只换损失。整篇在 CPU 上约五分钟。
> 卷积算子的即时编译占其中一大部分（原因见 [JIT、算子融合与异步执行](jit_and_async.md) 第 5 节）。

## 1. 换掉的三样东西

| | GAN（`dcgan.md`） | WGAN-GP（本篇） |
| --- | --- | --- |
| 判别器输出 | 「是真的」的概率（logit） | **任意实数**，不接任何激活 |
| 损失 | 二元交叉熵 | `E[critic(假)] − E[critic(真)]`，越大越好（critic 视角） |
| 名字 | discriminator | **critic**（它不再做分类，只打分） |
| 约束 | 无 | **Lipschitz 连续**：梯度的模长处处不超过 1 |
| 每步更新 | 1 次 critic + 1 次生成器 | **n_critic 次** critic + 1 次生成器 |
| 网络里的 BatchNorm | critic 每层都有 | **critic 里全部去掉** |

最后一行是连锁反应：梯度惩罚是对**每一个样本**分别算梯度模长的，而 `BatchNorm` 让一个
样本的输出依赖同一批里其它样本，逐样本的约束就失去了意义。所以 WGAN-GP 的 critic 不能用
`BatchNorm`——这正是 `dcgan.md` 第 1 节表里那句「后来很多工作（WGAN-GP、谱归一化等）正是
去掉了其中的 `BatchNorm`」的具体所指。

## 2. 数据与两个网络

数据和 `dcgan.md` 完全一样：4 类 16×16 的灰度图，每类是一个白图案加噪声。
像素取 `[-1, 1]`，和生成器最后的 `Tanh` 对齐。

```{code-cell} ipython3
import time

import numpy as np
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0        # 本篇用 CPU；有显卡时可以改成 1
jt.set_global_seed(0)
np.random.seed(0)

IMG = 16          # 图像边长
LATENT = 16       # 噪声向量长度
BATCH = 32
STEPS = 150       # 两种设置各训这么多步

TEMPLATES = np.full((4, IMG, IMG), -1.0, dtype="float32")
TEMPLATES[0, 6:10, :] = 1.0            # 横条
TEMPLATES[1, :, 6:10] = 1.0            # 竖条
TEMPLATES[2, 5:11, 5:11] = 1.0         # 方块
TEMPLATES[3, 6:10, 2:14] = 1.0         # 十字：一横……
TEMPLATES[3, 2:14, 6:10] = 1.0         # ……加一竖
FLAT_TEMPLATES = TEMPLATES.reshape(4, -1)


def sample_real(batch):
    """造一批真实样本：随机挑一个模板，加一点噪声。"""
    which = np.random.randint(0, TEMPLATES.shape[0], size=batch)
    images = TEMPLATES[which] + np.random.normal(0, 0.05, (batch, IMG, IMG)).astype("float32")
    return jt.array(images[:, None])       # (batch, 1, IMG, IMG)


real = sample_real(4)
print("形状:", real.shape, " 取值:", float(real.min().item()), "~", float(real.max().item()))
```

critic 是 `dcgan.md` 里判别器的卷积主干，**删掉两个 `BatchNorm`**、**删掉所有收尾激活**。
生成器一字不改。

```{code-cell} ipython3
class Critic(nn.Module):
    """只打分，不分类：输出是一个没有范围限制的实数。"""

    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 8, 4, stride=2, padding=1)      # 16 -> 8
        self.down2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)     # 8 -> 4
        self.out = nn.Conv2d(16, 1, 4)                            # 4 -> 1

    def execute(self, x):
        h = nn.leaky_relu(self.down1(x), 0.2)
        h = nn.leaky_relu(self.down2(h), 0.2)
        return self.out(h).reshape((-1,))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT, 32 * 4 * 4)          # 噪声 -> 4x4x32
        self.bn0 = nn.BatchNorm2d(32)
        self.up1 = nn.ConvTranspose(32, 16, 4, stride=2, padding=1)   # 4 -> 8
        self.bn1 = nn.BatchNorm2d(16)
        self.up2 = nn.ConvTranspose(16, 8, 4, stride=2, padding=1)    # 8 -> 16
        self.bn2 = nn.BatchNorm2d(8)
        self.out = nn.Conv2d(8, 1, 3, padding=1)

    def execute(self, z):
        h = nn.relu(self.bn0(self.fc(z).reshape((-1, 32, 4, 4))))
        h = nn.relu(self.bn1(self.up1(h)))
        h = nn.relu(self.bn2(self.up2(h)))
        return jt.tanh(self.out(h))


critic, generator = Critic(), Generator()
print("critic 参数量  :", sum(p.numel() for p in critic.parameters()))
print("生成器参数量  :", sum(p.numel() for p in generator.parameters()))
print("critic 对真样本的打分:", critic(real).numpy())
```

注意最后一行：分数有正有负，**没有上下界**。这正是它和 sigmoid 概率的区别。

## 3. W 估计：一个应该有符号的数

```{code-block} text
W = E[critic(真样本)] − E[critic(假样本)]
```

读法只有两句话：

* **critic 想让 W 越大越好**——把真样本的分数抬高、假样本的分数压低；
* **生成器想让 W 越小越好**——让假样本的分数追上来。

这就是一个二人零和博弈，而 `dcgan.md` 里的交叉熵不是（那是一个分类问题外加一个欺骗目标）。
当 critic 训练得足够好、且满足 Lipschitz 约束时，W 逼近真实的 Wasserstein 距离，
于是**它同时是损失和指标**。

```{code-cell} ipython3
def wasserstein_estimate(critic, generator, samples=256):
    """W 估计 = 真样本的平均分 − 假样本的平均分。"""
    return float(critic(sample_real(samples)).mean().item()
                 - critic(generator(jt.randn((samples, LATENT)))).mean().item())


print("还没训练时: W 估计 = %+.3f" % wasserstein_estimate(critic, generator))
```

还没训练时接近 0：critic 是随机的，对真假一视同仁。下面做两件事——先看**不给约束会怎样**，
再把约束写出来。

## 4. 不给约束会怎样

W 是「critic 能拉开的差距」。如果对 critic 不做任何限制，它可以把 W 拉到任意大：
只要把输出尺度放大、让梯度变陡就行——这与数据分布毫无关系。所以 W 会变成一串没有意义的
大数。

Lipschitz 约束就是用来堵住这条路的。它要求 critic 在**任何输入点上**的梯度模长都不超过 1，
这样 critic 就不能靠「变陡」来拉大差距，而只能靠「真的把真假分开」。

下面把生成器冻住，只训 critic，跑两种设置：不加惩罚（`lam=0`）和加惩罚（`lam=10`）。
冻结生成器是为了让 `|∇|` 的变化**只**反映惩罚项的作用。

```{code-cell} ipython3
def penalty_norm(critic, real, fake):
    """在真假样本连线上取点，算 critic 在该点的梯度模长。"""
    alpha = jt.rand((real.shape[0], 1, 1, 1))
    mixed = alpha * real + (1 - alpha) * fake
    grad = jt.grad(critic(mixed).sum(), mixed)
    return jt.sqrt((grad.reshape((real.shape[0], -1)) ** 2).sum(dim=1) + 1e-12)


def critic_only(lam, steps=60, c_lr=1e-3):
    """只训 critic（生成器冻结），返回每一步的 (步数, 平均|∇|, W 估计)。"""
    jt.set_global_seed(0)
    np.random.seed(0)
    net, frozen = Critic(), Generator()
    optim = jt.optim.Adam(net.parameters(), lr=c_lr, betas=(0.5, 0.9))
    trace = []
    for step in range(steps):
        batch_real = sample_real(BATCH)
        batch_fake = frozen(jt.randn((BATCH, LATENT))).stop_grad()
        norm = penalty_norm(net, batch_real, batch_fake)
        loss = (net(batch_fake).mean() - net(batch_real).mean()
                + lam * ((norm - 1) ** 2).mean())
        optim.step(loss)
        if step % 10 == 0 or step == steps - 1:
            trace.append((step, float(norm.mean().item()),
                          float(net(batch_real).mean().item()
                                - net(batch_fake).mean().item())))
    return trace


print("步数   不给约束: |∇| / W          加了惩罚(λ=10): |∇| / W")
for (step, g_no, w_no), (_, g_yes, w_yes) in zip(critic_only(0.0), critic_only(10.0)):
    print("%4d   %5.2f / %+7.2f            %5.2f / %+7.2f"
          % (step, g_no, w_no, g_yes, w_yes))
```

两列对比非常干脆：

| 步数 | 不给约束 `\|∇\|` / W | 加了惩罚 `\|∇\|` / W |
| --- | --- | --- |
| 0 | 0.10 / +0.04 | 0.10 / +0.01 |
| 30 | 0.60 / +5.24 | 0.80 / +2.27 |
| 50 | 1.80 / +16.54 | 1.18 / +6.41 |
| 59 | **2.62 / +24.42** | **1.25 / +8.63** |

左边 `|∇|` 一路涨到 2.62 还在涨，W 跟着涨到 +24.4；右边 `|∇|` 爬到 1 附近就**停住了**，
W 也稳定在同一个量级。**惩罚项做的事情就是给 `|∇|` 装一个弹簧，把它按在 1 上。**

## 5. 梯度惩罚怎么写

WGAN-GP 不直接约束每一处的梯度（那做不到），而是在**真假样本的连线上随机取点**，
惩罚那些点上梯度模长偏离 1 的部分：

```{code-block} text
x̂  = ε·真 + (1 − ε)·假，  ε ~ U(0,1)          在连线上取点
惩罚 = λ · mean( (‖∇_x̂ critic(x̂)‖₂ − 1)² )      把它按到 1
```

λ 取 10 是原论文的默认值。剩下的问题是：**这个 `∇_x̂` 在 Jittor 里怎么算？**
`jt.grad` 返回的就是一个 Var，而且它自己也在计算图上，所以可以直接进损失。

但这里有一个非常容易踩的坑——**`stop_grad` 放在哪一层**：

```{code-cell} ipython3
def cut_at(text, cut="fake"):
    """两种切断梯度的地方，看 jt.grad 分别拿到什么。"""
    jt.set_global_seed(0)
    net = Critic()
    batch_real, batch_fake = sample_real(BATCH), sample_real(BATCH)
    alpha = jt.rand((BATCH, 1, 1, 1))
    if cut == "mixed":
        mixed = (alpha * batch_real + (1 - alpha) * batch_fake).stop_grad()
    else:
        mixed = alpha * batch_real + (1 - alpha) * batch_fake.stop_grad()
    grad = jt.grad(net(mixed).sum(), mixed)
    length = jt.sqrt((grad.reshape((BATCH, -1)) ** 2).sum(dim=1) + 1e-12)
    print("%s: 平均 |∇| = %.4f" % (text, float(length.mean().item())))


cut_at("stop_grad 放在混合点上", cut="mixed")
cut_at("stop_grad 放在假样本上", cut="fake")
```

第一行永远是 `0.0000`，第二行是一个正常的非零值（这里约 0.1，具体大小看 critic 的初始权重）。
原因和 [自动求导](autograd_details.md) 第 2 节讲的是同一件事：
**`jt.grad` 对不在计算图上的目标不报错，它返回全 0。** 一旦 `|∇|` 恒为 0，
惩罚项就退化成常数 `λ·(0−1)² = 10`——损失看起来正常，critic 却完全没被约束。

那为什么要切在**假样本**上，而不是干脆什么都不切？因为不切的话，惩罚项的梯度会顺着
「连线 → 假样本 → 生成器」一路流回去，等于让惩罚项也在训练生成器。
我们只要用它约束 critic，所以把生成器那一路切断：**切假样本，不切混合点。**

> 这里也是 PyTorch 兼容层和原生 Jittor 的一个真实差别：同样把 `stop_grad` / `detach` 放错
> 位置，原生 `jt.grad` 安静地返回全 0，而 `torch.autograd.grad` 会直接报
> `One of the differentiated Tensors appears to not have been used in the graph`。
> 第 10 节会把同一段惩罚用 PyTorch API 再写一遍。

## 6. 训练：n_critic 次 critic + 1 次生成器

WGAN 要求 critic 大致接近最优，W 才近似一个距离，所以每更新一次生成器要先把 critic
多训几步。原论文用 `n_critic=5`，这里用小模型，取 3。

训练时要看的第二个数是「生成得像不像」，先把它定义出来。看图只能看个大概，而数据是模板
生成的，所以可以**直接量**——把每张生成图和四个模板比距离，取最近的那个。这里记录两件事：
多少张落在某个模板附近，以及**命中的模板各有多少张**。第二个数比第一个重要，
只看总数会漏掉「识别率很高但只会画一类」。

```{code-cell} ipython3
def inspect_generator(model, samples=128):
    """返回 (像某个模板的样本数, 中位距离, 各模式的样本数)。"""
    fakes = model(jt.randn((samples, LATENT))).numpy()[:, 0].reshape(samples, -1)
    distance = ((fakes[:, None, :] - FLAT_TEMPLATES[None, :, :]) ** 2).mean(axis=2)
    nearest = distance.argmin(axis=1)
    close = distance.min(axis=1) < 0.3
    counts = {int(m): int((nearest[close] == m).sum())
              for m in sorted(set(nearest[close].tolist()))}
    return int(close.sum()), float(np.median(distance.min(axis=1))), counts


LAMBDA = 10.0
N_CRITIC = 3


def train(steps=STEPS, c_lr=2e-3, g_lr=2e-2, n_critic=N_CRITIC, lam=LAMBDA, seed=0):
    """训一个 WGAN-GP，返回 (生成器, critic, 诊断记录, 用时秒数)。

    记录里每项是 (步数, W估计, 平均|∇|, 像某个模板的样本数, 中位距离, 各模式计数)。
    """
    jt.set_global_seed(seed)
    np.random.seed(seed)
    g, c = Generator(), Critic()
    g_optim = jt.optim.Adam(g.parameters(), lr=g_lr, betas=(0.5, 0.9))
    c_optim = jt.optim.Adam(c.parameters(), lr=c_lr, betas=(0.5, 0.9))

    history = []
    start = time.perf_counter()
    for step in range(steps):
        for _ in range(n_critic):
            batch_real = sample_real(BATCH)
            batch_fake = g(jt.randn((BATCH, LATENT))).stop_grad()
            norm = penalty_norm(c, batch_real, batch_fake)
            c_loss = (c(batch_fake).mean() - c(batch_real).mean()
                      + lam * ((norm - 1) ** 2).mean())
            c_optim.step(c_loss)

        # 生成器要让 W 变小：抬高假样本的分数
        g_loss = -c(g(jt.randn((BATCH, LATENT)))).mean()
        g_optim.step(g_loss)

        if step % 25 == 0 or step == steps - 1:
            history.append((step, wasserstein_estimate(c, g),
                            float(norm.mean().item())) + inspect_generator(g))
    return g, c, history, time.perf_counter() - start
```

## 7. 基线：W 转正，`|∇|` 被按在 1 上

```{code-cell} ipython3
generator, critic, baseline, baseline_seconds = train()
print("基线用时 %.0f s" % baseline_seconds)
print()
print("步数    W估计     识别      中位距离   平均|∇|   各模式的样本数")
for step, w_est, grad_mean, hit, median, mode_counts in baseline:
    print("%4d   %+7.3f   %3d/128   %7.3f   %5.2f    %s"
          % (step, w_est, hit, median, grad_mean, mode_counts))
```

这一格要等一两分钟（卷积算子的即时编译都在这里）。三列一起看：

* **`|∇|` 从 0.14 爬到 1.06 后就稳住了**——第 4 节看到的那个弹簧在工作；
* **W 估计从 −0.06 一路转正，稳定在 +1 到 +2.6**；critic 学会了把真假分开；
* **中位距离从 0.695 降到 0.253**，识别数从 0 涨到 75/128。

还有一处值得单独看：**最后三行的四个模式都在，但计数并不均匀**
（`{0: 27, 1: 30, 2: 6, 3: 12}`）。第 2 行更极端——66/128 里有 60 张全是模板 1。
如果只报「识别 66/128」这一个数字，你完全看不出它当时几乎只会画竖条。
**这就是为什么要连着计数一起看。**

## 8. 生成结果检验

```{code-cell} ipython3
recognised, median, counts = inspect_generator(generator)
grad_mean = baseline[-1][2]
w_estimate = baseline[-1][1]
print("像某个模板的样本: %d/128" % recognised)
print("中位距离        : %.3f" % median)
print("各模式的样本数  : %s（共 4 种）" % counts)
print("W 估计          : %+.3f" % w_estimate)
print("平均 |∇|        : %.2f" % grad_mean)

assert w_estimate > 0, "W 估计没有转正，critic 没学会区分真假"
assert recognised >= 128 // 3, "生成器没学出可辨认的形状"
assert len(counts) >= 2, "疑似模式塌缩：只生成了一两类"
assert 0.7 <= grad_mean <= 1.4, "Lipschitz 约束没有被满足"
```

这一格重新采了一批噪声（128 张），所以识别数会和训练记录里最后一格略有出入——
同一个生成器每次采样都不同。

再看一眼图：

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 8, figsize=(11, 3))
with jt.no_grad():
    fakes = generator(jt.randn((8, LATENT))).numpy()
for column in range(8):
    axes[0, column].imshow(TEMPLATES[column % 4], cmap="gray", vmin=-1, vmax=1)
    axes[1, column].imshow(fakes[column, 0], cmap="gray", vmin=-1, vmax=1)
    for row in (0, 1):
        axes[row, column].set_xticks([])
        axes[row, column].set_yticks([])
axes[0, 0].set_ylabel("真实模板", fontsize=9)
axes[1, 0].set_ylabel("生成", fontsize=9)
```

网格状伪影比 `dcgan.md` 里轻一些——同样的转置卷积结构，但训练目标不同，critic 对高频伪影
不像交叉熵判别器那样敏感。

## 9. 对照实验：把 critic 调慢

现在做那个对照：**同样的一切，只把 critic 的学习率降到 `2e-4`**（比基线小 10 倍）。
生成器的学习率、步数、种子都不动。

```{code-cell} ipython3
weak_generator, weak_critic, weak_history, weak_seconds = train(c_lr=2e-4)
print("对照用时 %.0f s" % weak_seconds)
print()
print("步数    W估计     识别      中位距离   平均|∇|   各模式的样本数")
for step, w_est, grad_mean, hit, median, mode_counts in weak_history:
    print("%4d   %+7.3f   %3d/128   %7.3f   %5.2f    %s"
          % (step, w_est, hit, median, grad_mean, mode_counts))
```

```{code-cell} ipython3
weak_recognised, _, weak_counts = inspect_generator(weak_generator)
weak_w = weak_history[-1][1]
print("判别器学习率         W估计     识别       各模式")
print("2e-3（基线）        %+7.3f   %3d/128   %s"
      % (baseline[-1][1], recognised, counts))
print("2e-4（小 10 倍）    %+7.3f   %3d/128   %s"
      % (weak_w, weak_recognised, weak_counts))

assert weak_w < 0, "对照组的 W 估计没有落到负值，这次对照没成立"
assert weak_recognised < recognised, "对照组反而更好，说明对照变量没起作用"
```

两列读数的差别不在大小，而在**符号**：

* 基线的 W 从第 25 步起一直是**正的**（+1 到 +2.6）；
* 对照组的 W **整场都是负的**（−0.13 → −2.04 → −0.76），识别始终是 0。

W 是负的意味着 `critic(假) > critic(真)`——critic 不但没学会区分，还把顺序搞反了。
这时候它的数字**不含任何关于生成质量的信息**，所以「W 是负的」这条读数的用处是：
**它先告诉你这把尺子现在不可信，而不是给你一个错误的小数。**

还有一个容易忽略的细节：对照组的 `|∇|` 最后还是爬到了 0.96。也就是说
**惩罚项是可以被满足的，但它不保证 critic 有用**——一个几乎不学习的 critic 也能把
梯度模长压到 1 附近。Lipschitz 约束是必要条件，不是充分条件；critic 的学习率还得够大。

## 10. 怎么读这个数

| 看到的现象 | 含义 | 该做什么 |
| --- | --- | --- |
| W 为正且稳定 | critic 能分开真假，这个数可以信 | 看它的**趋势**：生成器变好时它应该变小 |
| W 长期为负 | critic 还没学会，尺子不可信 | 提高 critic 学习率，或增大 n_critic |
| `\|∇\|` 远离 1 且一直涨 | 惩罚没生效 | 检查 `stop_grad` 的位置、λ 是否为 0 |
| `\|∇\|` ≈ 1 但样本很差 | 约束满足了，critic 却没用 | 这是第 9 节的情况：约束不等于有效 |
| 识别率高但模式计数偏斜 | 模式不均衡 / 塌缩 | 别只看总数，看每个模式各几张 |

最后放一张和 `dcgan.md` 的对照表——同一个问题的两种读法：

| | `dcgan.md`（交叉熵） | 本篇（WGAN-GP） |
| --- | --- | --- |
| 判别器的数 | 准确率，越高越可疑 | W，为正才可信 |
| 判别器坏掉时 | `G损失` 反而更低 | W 变负，明确报警 |
| 需要额外约束吗 | 不需要 | 需要，且要验证 `\|∇\|` 真的接近 1 |
| 主要麻烦 | 模式塌缩、梯度消失 | 惩罚写法容易悄悄失效 |

## 11. 同一件事，用 PyTorch API 再写一遍

惩罚项是全篇唯一一处需要「对梯度再求一次梯度」的地方，也是两种 API 差别最大的一处。
原生 Jittor 是函数式的 `jt.grad`；PyTorch 侧需要在 `torch.autograd.grad` 上加
`create_graph=True`，并且**把插值点声明成一个需要梯度的叶子**。

激活必须在**创建任何张量之前**完成，所以这一节自成一段，不要和上面的原生代码混着写
（细节见 [用 PyTorch API 写 Jittor](torch_compat.md)）。

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

torch.manual_seed(0)

# 主干用 Sequential，输出前手工 reshape——和前面几篇一样，只是为了避开
# 兼容层里可能没有的 Flatten。
trunk_t = tnn.Sequential(
    tnn.Conv2d(1, 8, 4, stride=2, padding=1),
    tnn.LeakyReLU(0.2),
    tnn.Conv2d(8, 16, 4, stride=2, padding=1),
    tnn.LeakyReLU(0.2),
)
head_t = tnn.Linear(16 * 4 * 4, 1)
real_t = torch.randn(8, 1, IMG, IMG)
fake_t = torch.randn(8, 1, IMG, IMG)

alpha = torch.rand(8, 1, 1, 1)
# 注意 requires_grad_：torch.randn 造出来的张量默认不需要梯度，
# 不声明的话 autograd.grad 会报「没有出现在图里」。
mixed_t = (alpha * real_t + (1 - alpha) * fake_t.detach()).requires_grad_(True)

scores = head_t(trunk_t(mixed_t).reshape(8, -1)).reshape(-1)
grad_t = torch.autograd.grad(scores.sum(), mixed_t, create_graph=True)[0]
length_t = grad_t.reshape(8, -1).norm(dim=1)
print("torch 侧的平均 |∇| : %.4f" % float(length_t.mean()))
print("梯度自己还需要梯度:", grad_t.requires_grad)

params_t = list(trunk_t.parameters()) + list(head_t.parameters())
penalty_t = LAMBDA * ((length_t - 1) ** 2).mean()
optim_t = torch.optim.Adam(params_t, lr=2e-4, betas=(0.5, 0.9))
before = [float(p.detach().abs().sum()) for p in params_t]
optim_t.zero_grad()
penalty_t.backward()
optim_t.step()
after = [float(p.detach().abs().sum()) for p in params_t]
print("惩罚项能反传到 critic 参数:", before[0] != after[0])
```

两种写法的对照：

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 一阶梯度 | `jt.grad(loss, x)` | `torch.autograd.grad(loss, x)[0]` |
| 要再求一次梯度 | 直接用它返回的 Var | 必须写 `create_graph=True` |
| 插值点 | `jt.rand(...)` 造出来就在图上 | 要显式 `.requires_grad_(True)` |
| 切断生成器 | `x.stop_grad()` | `x.detach()` |
| 放错位置的后果 | 静默返回全 0 | 报错 |

最后一行值得记住：**同一处错误，一边安静一边吵。** 安静的那一边要靠自己去查 `|∇|` 是不是 0。

## 12. 检查清单

```{code-block} text
1. critic 里没有 BatchNorm                —— 逐样本的约束不允许批内耦合
2. critic 输出不接激活                     —— 它要能输出任意实数
3. stop_grad 切在假样本上，不切混合点        —— 切错会静默拿到全 0
4. 每次都打 |∇|                            —— 它是惩罚项唯一可见的证据
5. W 为负时不要相信它的数值                 —— 先修 critic 的学习率
6. 看每个模式的计数，不只看到总数            —— 总数掩盖不均衡
7. 生成器和 critic 用 Adam(betas=(0.5, 0.9)) —— WGAN-GP 不用默认的 0.999
```

想接着看下一类生成模型，可以读变分自编码器（VAE）：它不用对抗，而是一个显式的似然下界，
读数是另一个样子。

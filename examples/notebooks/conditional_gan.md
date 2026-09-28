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

# 条件生成对抗网络（Conditional GAN）

普通的 GAN 能生成逼真的样本，但你无法告诉它**生成什么**：给它一个噪声向量，它给你
一张图，至于这张图是什么，由训练数据分布随机决定。

Conditional GAN（CGAN）解决了这一点：**在生成器和判别器上都额外接一个条件 `y`**，
于是你可以指定「生成一张写着 3 的图」，而不是听天由命。

这一篇会：

* 讲清楚条件是怎么接进网络的，以及**为什么判别器也必须拿到条件**；
* 用一个 8×8 的合成数据集，**在 CPU 上约一分钟**训练一个能用的条件生成器；
* 用一个可量化的指标验证「条件真的生效了」，而不是只看图；
* 用 **PyTorch 兼容层**把同一件事再写一遍，对照两种 API；
* 给出放大到 MNIST 的完整思路，以及把损失换成最小二乘（LSGAN）的改法。

全程离线、CPU 可跑，不需要显卡，也不需要下载数据集。

## 1. 条件从哪里进来

CGAN 的结构改动只有一处：把 `y` 同时喂给 `G` 和 `D`。

```{code-block} text
        噪声 z ─┐                      ┌─ 图像 x ─┐
               ├─► G(z, y) ─► 假图像 x̂ ┤          ├─► D(x, y) ─► 真/假
   条件 y ─────┘                      └─ 条件 y ──┘
```

* **生成器**：`G(z, y)` —— 噪声提供随机性，`y` 决定生成哪一类。
* **判别器**：`D(x, y)` —— 判断「在条件 `y` 下，`x` 像不像真实样本」。

判别器必须看到 `y`，这一点容易被忽略。如果 `D` 只看图像，那么 `G` 完全可以无视 `y`
去生成任意一类真实样本，`D` 依然会判它为真——条件就成了摆设。`D` 拿到 `y` 之后，
它才能指出「你要的是 3，可你画的是 7」，梯度才会把 `G` 推向正确的方向。

实现上，`y` 是一个类别编号。把编号做一次嵌入（`nn.Embedding`）变成一个向量，再和别的
输入拼接起来，就是最经典的 CGAN 做法。下面都采用这种方式。

## 2. 数据：一个能「查对错」的条件生成任务

真实图像一般是 MNIST 这类数据集，但本篇故意不用它，因为我们要的是一条**离线、一分钟
内跑完、而且能精确检验条件是否生效**的路径。MNIST 的完整版本放在第 10 节。

这里的合成数据是 8×8 的灰度图，共 8 个条件：

* 「第 `y` 列的像素为 +1，其余列为 -1」，再叠加一点噪声。

也就是说，图像的内容完全由一个整数决定。这样一个任务是**可检验的**：生成一张图之后，
只要看哪一列最亮，就知道生成器有没有听条件的话。

像素取 `[-1, 1]` 而不是 `[0, 1]`，是为了和后面生成器最后一层的 `Tanh` 对齐——生成器的
输出范围必须和真实数据一致。

```{code-cell} ipython3
import jittor as jt
from jittor import nn
import numpy as np

# 固定随机种子，让本文的每个数字都可以复现
jt.set_global_seed(0)
np.random.seed(0)

N_CLASSES = 8      # 条件（类别）数量
IMG = 8            # 图像边长
LATENT = 16        # 噪声向量长度
BATCH = 64
STEPS = 400
```

```{code-cell} ipython3
def sample_real(batch, labels=None):
    """造一批真实样本：第 labels 列为 +1，其余列为 -1。"""
    if labels is None:
        labels = np.random.randint(0, N_CLASSES, size=batch)
    images = np.full((batch, 1, IMG, IMG), -1.0, dtype="float32")
    images[np.arange(batch), 0, :, labels] = 1.0
    images += np.random.normal(0, 0.05, images.shape).astype("float32")
    return jt.array(images), jt.array(labels).int32()


images, labels = sample_real(3)
print("图像形状:", images.shape)
print("这一批的条件:", labels.numpy())
```

`sample_real` 返回 `(图像, 条件)` 一对。注意 `labels` 被转成 `int32`：它要作为
`nn.Embedding` 的下标，而不是一个数值。

## 3. 生成器：`G(z, y)`

生成器把「噪声」和「条件」拼成一个向量，再经过几层全连接映射成一张图。

最后一层用 `Tanh`，把输出压到 `[-1, 1]`——这正是真实数据的取值范围（亮列 +1，背景
-1），生成器的输出必须落在同一个区间里。

```{code-cell} ipython3
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(N_CLASSES, LATENT)
        self.net = nn.Sequential(
            nn.Linear(LATENT * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, IMG * IMG),
            nn.Tanh(),
        )

    def execute(self, noise, labels):
        condition = self.embed(labels)
        hidden = self.net(jt.concat((noise, condition), dim=1))
        return hidden.reshape((hidden.shape[0], 1, IMG, IMG))
```

`nn.Embedding(N_CLASSES, LATENT)` 把 0..7 的编号查成 `LATENT` 维向量，于是
`jt.concat((noise, condition), dim=1)` 得到 `LATENT * 2` 维的输入。生成器的参数量只有
一万多，CPU 训练绰绰有余。

## 4. 判别器：`D(x, y)`

判别器把图像展平，和条件的嵌入向量拼接，输出**一个 logit**：越大表示越像真的。

```{code-cell} ipython3
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(N_CLASSES, IMG * IMG)
        self.net = nn.Sequential(
            nn.Linear(IMG * IMG * 2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )

    def execute(self, images, labels):
        flat = images.reshape((images.shape[0], -1))
        condition = self.embed(labels)
        return self.net(jt.concat((flat, condition), dim=1))
```

注意两个网络的 `execute` 签名都多了一个 `labels` 参数，这就是条件的入口。

把两个网络实例化，准备训练：

```{code-cell} ipython3
generator = Generator()
discriminator = Discriminator()
print("生成器参数量:", sum(parameter.numel() for parameter in generator.parameters()))
```

## 5. 损失函数

GAN 的原始目标是一个极小极大博弈：

$$ \min_G \max_D \; \mathbb{E}_{x}\big[\log D(x \mid y)\big]
   + \mathbb{E}_{z}\big[\log(1 - D(G(z \mid y) \mid y))\big] $$

落到代码里就是两个二分类问题：

| 谁 | 希望 | 损失 |
| --- | --- | --- |
| 判别器 `D` | 真样本判 1，假样本判 0 | `bce(d_real, 1) + bce(d_fake, 0)` |
| 生成器 `G` | 假样本被判 1 | `bce(d_fake, 1)` |

判别器每步**最小化**自己的损失，生成器也**最小化**自己的损失——后者正是「骗过判别器」。
两个网络各用一个 Adam，学习率相同但优化器独立。`betas=(0.5, 0.999)` 是 GAN 的常用值：
默认的 `(0.9, 0.999)` 动量太大，在对抗训练里容易震荡。

```{code-cell} ipython3
bce = nn.BCEWithLogitsLoss()
g_optim = jt.optim.Adam(generator.parameters(), lr=2e-3, betas=(0.5, 0.999))
d_optim = jt.optim.Adam(discriminator.parameters(), lr=2e-3, betas=(0.5, 0.999))
```

用 `BCEWithLogitsLoss` 而不是「先 `sigmoid` 再 `BCELoss`」，是因为前者把两步合并成一个
数值稳定的算子；上面的网络因此直接输出 logit，不带 `Sigmoid`。

## 6. 训练循环

每一步做两次更新。第一次更新判别器，第二次更新生成器。

```{code-cell} ipython3
for step in range(STEPS):
    real_images, real_labels = sample_real(BATCH)

    # --- 更新判别器：真样本判 1，假样本判 0
    noise = jt.randn((BATCH, LATENT))
    fake_images = generator(noise, real_labels)
    d_real = discriminator(real_images, real_labels)
    d_fake = discriminator(fake_images.stop_grad(), real_labels)
    d_loss = bce(d_real, jt.ones_like(d_real)) + bce(d_fake, jt.zeros_like(d_fake))
    d_optim.step(d_loss)

    # --- 更新生成器：让判别器把假样本判成 1
    noise = jt.randn((BATCH, LATENT))
    fake_images = generator(noise, real_labels)
    d_fake = discriminator(fake_images, real_labels)
    g_loss = bce(d_fake, jt.ones_like(d_fake))
    g_optim.step(g_loss)

    if step % 50 == 0:
        print("step %4d  D损失=%.3f  G损失=%.3f" % (step, d_loss.item(), g_loss.item()))
```

有三处值得说明：

* **`fake_images.stop_grad()`**：更新判别器时把生成器这一路的梯度切断。判别器要学的
  是「分辨真假」，不是把生成器一起往下拖；不加这一刀，两张网络会互相干扰。
* **生成器前向做了两次**：判别器更新用的那张假图是在生成器更新**之前**算的，所以第二
  次必须重新前向，才能得到当前参数下的图和新梯度。
* **`optim.step(loss)`**：原生 Jittor 里优化器一步到位——传入损失，反传和参数更新一起
  完成，不需要 `zero_grad()` 和 `backward()`。这一点和 PyTorch 不同，第 9 节会对照。

损失值在训练中上下跳动是正常的：两个网络此消彼长，D 变强则 G 的损失上升，反之亦然。
**不要用损失大小判断 GAN 训得好不好**，要看生成结果——这正是下一节要做的事。

## 7. 条件生效了吗？

看图只能看个大概。既然数据是合成的，我们可以直接量化：对每个条件生成若干张图，取
**每一列在整幅图上的平均值**，最亮的那一列应该正好是要求的编号。

对列取平均而不是逐像素取最大值，是因为信号是「整列都亮」，逐像素会被噪声带偏。

```{code-cell} ipython3
def conditional_accuracy(samples=4):
    correct = 0
    for label in range(N_CLASSES):
        noise = jt.randn((samples, LATENT))
        requested = jt.array(np.full(samples, label)).int32()
        generated = generator(noise, requested).numpy()
        # 对每张图，逐列求平均 -> 形状 (samples, IMG)，再取最亮的列
        predicted = generated[:, 0, :, :].mean(axis=1).argmax(axis=1)
        correct += int((predicted == label).sum())
    return correct, samples * N_CLASSES


hit, total = conditional_accuracy()
print("条件准确率: {}/{}".format(hit, total))
```

32/32 表示：对全部 8 个条件各生成 4 张图，每一张的最亮列都等于要求的编号。换一台机器
或换用显卡时具体数值可能不同，但训练正常时应当接近满分。

## 8. 看图

把真实样本和生成样本并排画出来，最直观。

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, N_CLASSES, figsize=(N_CLASSES * 1.1, 2.8))
for label in range(N_CLASSES):
    real, _ = sample_real(1, np.array([label]))
    axes[0, label].imshow(real.numpy()[0, 0], cmap="gray", vmin=-1, vmax=1)
    axes[0, label].set_title("y=%d" % label, fontsize=9)

    noise = jt.randn((1, LATENT))
    fake = generator(noise, jt.array([label]).int32())
    axes[1, label].imshow(fake.numpy()[0, 0], cmap="gray", vmin=-1, vmax=1)

    for row in (0, 1):
        axes[row, label].set_xticks([])
        axes[row, label].set_yticks([])
axes[0, 0].set_ylabel("real", fontsize=10)
axes[1, 0].set_ylabel("generated", fontsize=10)
plt.show()
```

上图每一列对应一个条件，亮带的位置随条件从左到右移动；下面一行是生成结果，位置和真实
数据完全一致。生成结果反而比真实样本干净——真实样本里那点噪点是造数据时故意加的，生成器
学到的是它背后的规律，而不是某一次采样的噪声。

## 9. 同一件事，用 PyTorch API 再写一遍

如果你更熟悉 PyTorch，可以不用换写法：Jittor 的兼容层让 `torch` 的 API 由 Jittor 实现。
激活之后，同一套 `torch.nn` / `torch.optim` 代码直接跑在 Jittor 的运行时上。

激活必须在**创建任何张量之前**完成，所以下面几格自成一段，不要和上面的原生代码混着
写（细节见 [用 PyTorch API 写 Jittor](torch_compat.md)）。

```{code-cell} ipython3
from jittor.compat.shim import activate
activate()

import torch
import torch.nn as tnn

print("torch 来自:", type(torch).__module__)
print("torch.__version__ =", torch.__version__)
```

模型的定义方式换成 `forward` 和 `torch.cat`，其余思路完全一样（这一节重新种一次随机
种子，让结果同样可复现）：

```{code-cell} ipython3
torch.manual_seed(0)
np.random.seed(0)


def sample_batch(batch):
    labels = np.random.randint(0, N_CLASSES, size=batch)
    images = np.zeros((batch, IMG, IMG), dtype="float32")
    images[np.arange(batch), :, labels] = 1.0
    images += np.random.normal(0, 0.05, images.shape).astype("float32")
    return images, labels


class TorchGenerator(tnn.Module):
    def __init__(self):
        super().__init__()
        self.embed = tnn.Embedding(N_CLASSES, LATENT)
        self.net = tnn.Sequential(
            tnn.Linear(LATENT * 2, 64),
            tnn.ReLU(),
            tnn.Linear(64, 128),
            tnn.ReLU(),
            tnn.Linear(128, IMG * IMG),
            tnn.Tanh(),
        )

    def forward(self, noise, labels):
        hidden = self.net(torch.cat((noise, self.embed(labels)), dim=1))
        return hidden.reshape(hidden.shape[0], 1, IMG, IMG)


class TorchDiscriminator(tnn.Module):
    def __init__(self):
        super().__init__()
        self.embed = tnn.Embedding(N_CLASSES, IMG * IMG)
        self.net = tnn.Sequential(
            tnn.Linear(IMG * IMG * 2, 128),
            tnn.LeakyReLU(0.2),
            tnn.Linear(128, 64),
            tnn.LeakyReLU(0.2),
            tnn.Linear(64, 1),
        )

    def forward(self, images, labels):
        flat = images.reshape(images.shape[0], -1)
        return self.net(torch.cat((flat, self.embed(labels)), dim=1))
```

训练循环换成 PyTorch 的三个动作：`zero_grad()` → `backward()` → `step()`，
分离梯度用 `detach()`。

```{code-cell} ipython3
generator_t = TorchGenerator()
discriminator_t = TorchDiscriminator()
bce_t = tnn.BCEWithLogitsLoss()
g_optim_t = torch.optim.Adam(generator_t.parameters(), lr=2e-3, betas=(0.5, 0.999))
d_optim_t = torch.optim.Adam(discriminator_t.parameters(), lr=2e-3, betas=(0.5, 0.999))

for step in range(STEPS):
    images_np, labels_np = sample_batch(BATCH)
    real_images = torch.tensor(images_np)
    real_labels = torch.tensor(labels_np)

    noise = torch.randn(BATCH, LATENT)
    fake_images = generator_t(noise, real_labels)
    d_real = discriminator_t(real_images, real_labels)
    d_fake = discriminator_t(fake_images.detach(), real_labels)
    d_loss = bce_t(d_real, torch.ones_like(d_real)) + bce_t(d_fake, torch.zeros_like(d_fake))
    d_optim_t.zero_grad()
    d_loss.backward()
    d_optim_t.step()

    noise = torch.randn(BATCH, LATENT)
    fake_images = generator_t(noise, real_labels)
    d_fake = discriminator_t(fake_images, real_labels)
    g_loss = bce_t(d_fake, torch.ones_like(d_fake))
    g_optim_t.zero_grad()
    g_loss.backward()
    g_optim_t.step()

    if step % 100 == 0:
        print("step %4d  D损失=%.3f  G损失=%.3f" % (step, float(d_loss), float(g_loss)))
```

用同一个指标检验：

```{code-cell} ipython3
correct = 0
for label in range(N_CLASSES):
    noise = torch.randn(4, LATENT)
    requested = torch.tensor(np.full(4, label))
    generated = generator_t(noise, requested).detach().numpy()
    predicted = generated[:, 0, :, :].mean(axis=1).argmax(axis=1)
    correct += int((predicted == label).sum())
print("条件准确率: %d/%d" % (correct, 4 * N_CLASSES))
```

两种写法只差这几处：

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 前向方法名 | `execute` | `forward` |
| 一步训练 | `optim.step(loss)` | `zero_grad()` / `backward()` / `step()` |
| 拼接 | `jt.concat((a, b), dim=1)` | `torch.cat((a, b), dim=1)` |
| 切断梯度 | `x.stop_grad()` | `x.detach()` |
| 从数组建张量 | `jt.array(x)` | `torch.tensor(x)` |

迁移一份现成脚本时要注意的其它差异（设备、dtype、`argmax` 的返回形状等），见
[把已有 PyTorch 脚本迁到 Jittor](torch_compat_migration.md)。

## 10. 放大到 MNIST

上面的任务规模是刻意压小的。换成真实数据集时，训练循环的**结构**完全不变（仍然是一步
判别器、一步生成器），要改的是两处：数据的读入方式，以及把全连接换成卷积。

```{code-cell} ipython3
:tags: [gan, network, long-running, skip-execution]

from jittor.dataset.mnist import MNIST
import jittor.transform as transform

# MNIST 是 28x28，这里 resize 到 32x32 方便用 2 的幂次做卷积
train_loader = MNIST(train=True, transform=transform.Compose([
    transform.Resize(32),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),   # -> [-1, 1]
])).set_attrs(batch_size=64, shuffle=True)


class ConvGenerator(nn.Module):
    """条件仍然走嵌入向量，和噪声拼接后映射成 8x8 的特征图，再上采样到 32x32。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 32)
        self.fc = nn.Linear(100 + 32, 128 * 8 * 8)
        self.net = nn.Sequential(
            nn.ConvTranspose(128, 64, 4, 2, 1), nn.BatchNorm(64), nn.ReLU(),
            nn.ConvTranspose(64, 1, 4, 2, 1), nn.Tanh(),
        )

    def execute(self, noise, labels):
        hidden = self.fc(jt.concat((noise, self.embed(labels)), dim=1))
        return self.net(hidden.reshape((hidden.shape[0], 128, 8, 8)))


class ConvDiscriminator(nn.Module):
    """换个方式注入条件：把条件铺成和图像同尺寸的一张图，当作额外通道拼上去。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 32 * 32)
        self.net = nn.Sequential(
            nn.Conv(2, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv(64, 128, 4, 2, 1), nn.BatchNorm(128), nn.LeakyReLU(0.2),
            nn.Conv(128, 1, 8),
        )

    def execute(self, images, labels):
        condition = self.embed(labels).reshape((images.shape[0], 1, 32, 32))
        inputs = jt.concat((images, condition), dim=1)
        return self.net(inputs).reshape((images.shape[0], 1))


G = ConvGenerator()
D = ConvDiscriminator()
g_optim = jt.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optim = jt.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(20):
    for real_images, real_labels in train_loader:
        batch = real_images.shape[0]
        noise = jt.randn((batch, 100))
        fake_images = G(noise, real_labels)
        d_loss = (bce(D(real_images, real_labels), jt.ones((batch, 1)))
                  + bce(D(fake_images.stop_grad(), real_labels), jt.zeros((batch, 1))))
        d_optim.step(d_loss)

        noise = jt.randn((batch, 100))
        d_fake = D(G(noise, real_labels), real_labels)
        g_optim.step(bce(d_fake, jt.ones((batch, 1))))
    print("epoch %d done" % epoch)
```

这一格被标记为不参与离线冒烟：它要下载 MNIST，而且即使只跑几十步，在 CPU 上也要比上面
的合成任务慢上两个数量级。在带显卡的机器上把 `jt.flags.use_cuda = 1` 打开会快得多。

## 11. 损失函数的变体：把 BCE 换成最小二乘

CGAN 的「条件」和「损失」是两件独立的事，上面那套损失可以直接替换。

原始 GAN 用交叉熵（就是本篇的 BCE）。LSGAN 的出发点是：真样本被判别器判得**足够真**
之后，交叉熵给出的梯度就很小，生成器学得慢；改用最小二乘，把「离目标还有多远」一直量化
成平方误差，梯度不会提前消失。

要改的只有损失这两行，其余一字不动：

```{code-block} python
# 判别器：真样本靠近 1，假样本靠近 0
d_loss = (nn.mse_loss(d_real, jt.ones_like(d_real))
          + nn.mse_loss(d_fake, jt.zeros_like(d_fake)))

# 生成器：假样本靠近 1
g_loss = nn.mse_loss(d_fake, jt.ones_like(d_fake))
```

判别器本来就输出一个不带 `Sigmoid` 的分数——这正是上面用 `BCEWithLogitsLoss` 的原因——
换成最小二乘时它不用动；生成器最后一层的 `Tanh` 也保持原样，它的值域 `[-1, 1]` 正好
对齐真实数据的取值范围。LSGAN 论文见参考文献 [3]。

## 参考文献

1. Goodfellow, Ian, et al. "Generative adversarial nets." *NeurIPS*, 2014.
2. Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets."
   *arXiv:1411.1784*, 2014.
3. Mao, Xudong, et al. "Least squares generative adversarial networks." *ICCV*, 2017.

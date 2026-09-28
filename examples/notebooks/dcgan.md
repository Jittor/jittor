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

# 深度卷积生成对抗网络（DCGAN）

[条件生成对抗网络](conditional_gan.md) 里的生成器是一串全连接层：把噪声映射成一个向量，
再 reshape 成一张 8×8 的图。它在那篇里够用，因为数据是"某一列亮"这种**位置固定**的模式——
全连接层只需要记住"第几列该亮"。

换成人脸、物体照片这类自然图像就不行了。自然图像有几个全连接层表达起来很费劲的性质：

* **局部性**：一个像素只和邻近像素强相关；
* **平移不变性**：猫在左上角和猫在右下角是同一只猫；
* **层次性**：边缘 → 纹理 → 部件 → 物体。

卷积正好编码了前两条。DCGAN 就是把 GAN 里的全连接换成一整套卷积约定，并给了几条
让训练不崩的规则。这一篇会：

* 讲清楚 **`ConvTranspose` 是怎么把特征图放大回去的**，以及那个容易算错输出尺寸的公式；
* 在 CPU 上跑一个**真的用卷积**的 GAN（16×16 的合成形状），并做一次对照实验；
* 用一个可量化的指标检验生成结果，并看到**模式覆盖**；
* 把判别器的强弱用数字看出来——这是 GAN 训练最常见的失败模式。

如果还没读过，建议先看 [条件生成对抗网络](conditional_gan.md)（对抗训练的基本概念）和
[模型定义与训练](example.md)（`Module`、优化器）。仓库里没有单独的卷积教程，所以卷积在
Jittor 里怎么写（形状阶梯、转置卷积的尺寸）会在本篇第 3、4 节补上。

> **这一篇比其它教程重**：整篇在 CPU 上约五分钟，其中一大半是卷积算子的即时编译而不是训练
> （实测 4 分 49 秒，含编译）。为什么第一格那么慢，见
> [JIT、算子融合与异步执行](jit_and_async.md) 第 5 节。有显卡时会快得多。

## 1. DCGAN 的几条约定

2016 年的 DCGAN 论文没有提出新损失，它给的是**架构上的经验规则**。这些规则至今仍是
卷积 GAN 的默认起点：

| 位置 | 做法 | 为什么 |
| --- | --- | --- |
| 上采样 | 用 `ConvTranspose`（可学习的），不用池化 | 池化没有参数，放大方式是死的 |
| 下采样 | 用 stride=2 的卷积，不用池化 | 同上，让网络自己学怎么缩小 |
| 生成器 | 每层 `BatchNorm` + `ReLU` | 稳定深层训练 |
| 生成器**输出层** | **不加** `BatchNorm`，用 `Tanh` | 输出要落到数据范围 `[-1, 1]`，BN 会把它拉走 |
| 判别器 | 每层 `BatchNorm` + `LeakyReLU(0.2)` | 防止梯度在负半轴归零 |
| 判别器**输入层** | **不加** `BatchNorm` | 让判别器保留对真实分布的原始看法 |
| 全连接 | 生成器只留输入那一层，判别器只留输出 | 空间结构交给卷积 |

这些是**起点而不是定理**：后来很多工作（WGAN-GP、谱归一化等）正是去掉了其中的 `BatchNorm`。
但先按这套规则跑通一遍，再谈改动。

## 2. 数据：能「查对错」的卷积任务

真实数据集在离线环境里跑不动，所以还是用合成数据。4 类 16×16 的灰度图，每类是一个白图案
加噪声：

* 0：横条　1：竖条　2：方块　3：十字

用 16×16 而不是上一篇的 8×8，是为了有足够空间做两次 2 倍上采样（4 → 8 → 16）。

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
STEPS = 200       # 训练步数（两次实验都用这么多）

TEMPLATES = np.full((4, IMG, IMG), -1.0, dtype="float32")
TEMPLATES[0, 6:10, :] = 1.0            # 横条：中间几行全亮
TEMPLATES[1, :, 6:10] = 1.0            # 竖条：中间几列全亮
TEMPLATES[2, 5:11, 5:11] = 1.0         # 方块：中间一块亮
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

## 3. 生成器：噪声 → 特征图 → 逐级上采样

生成器分两段：

1. **全连接**把噪声映射成一个小的特征图 `(32, 4, 4)`——这一步决定了"生成什么"；
2. **两次 `ConvTranspose`** 把它放大到 `(8, 8)`、`(16, 16)`——这一步把"什么"铺成像素。

第一段的 reshape 是这套结构的关键：**噪声不是"一张小图"，而是"一张特征图上的所有通道"**。
空间信息由后续的转置卷积一层层长出来。

```{code-cell} ipython3
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT, 32 * 4 * 4)          # 噪声 -> 4x4x32
        self.bn0 = nn.BatchNorm2d(32)
        self.up1 = nn.ConvTranspose(32, 16, 4, stride=2, padding=1)   # 4 -> 8
        self.bn1 = nn.BatchNorm2d(16)
        self.up2 = nn.ConvTranspose(16, 8, 4, stride=2, padding=1)    # 8 -> 16
        self.bn2 = nn.BatchNorm2d(8)
        self.out = nn.Conv2d(8, 1, 3, padding=1)          # 通道压到 1，尺寸不变

    def execute(self, z):
        h = nn.relu(self.bn0(self.fc(z).reshape((-1, 32, 4, 4))))
        h = nn.relu(self.bn1(self.up1(h)))
        h = nn.relu(self.bn2(self.up2(h)))
        return jt.tanh(self.out(h))          # 压到 [-1, 1]


generator = Generator()
print("生成器参数量:", sum(p.numel() for p in generator.parameters()))

# 把每一步的形状打出来——这种"形状阶梯"是调试卷积网络最有效的手段
z = jt.randn((2, LATENT))
stage1 = generator.fc(z).reshape((-1, 32, 4, 4))
print("阶梯:", z.shape, "->", stage1.shape, "->", generator.up1(stage1).shape,
      "->", generator.up2(generator.up1(stage1)).shape)
print("最终输出:", generator(z).shape)
```

## 4. 转置卷积的输出尺寸

`ConvTranspose` 是这一篇最容易算错的地方。它的输出边长是：

```{code-block} text
H_out = (H_in - 1) * stride - 2 * padding + (kernel_size - 1) + output_padding + 1
```

记住两件事就够用了：

* 最常见的配置 `kernel_size=4, stride=2, padding=1` 正好让尺寸**翻倍**（`H_out = 2 * H_in`）；
* 想让尺寸翻倍又不想记公式，就用这一组参数。

```{code-cell} ipython3
for h_in, stride, kernel, padding, out_pad in (
        (8, 2, 4, 1, 0),        # 最常用的一组：正好翻倍
        (7, 2, 3, 1, 1),        # 奇数边长想翻倍时的写法
        (16, 2, 4, 0, 0)):      # 故意不补 padding，看它长什么样
    layer = nn.ConvTranspose(1, 1, kernel, stride=stride,
                             padding=padding, output_padding=out_pad)
    shape = layer(jt.randn((1, 1, h_in, h_in))).shape
    formula = (h_in - 1) * stride - 2 * padding + (kernel - 1) + out_pad + 1
    print("H=%2d s=%d k=%d p=%d op=%d -> %2d（公式 %d）%s"
          % (h_in, stride, kernel, padding, out_pad, shape[2], formula,
             "OK" if shape[2] == formula else "不一致"))
```

## 5. 判别器：卷积下采样成一个 logit

判别器是生成器的镜像：两次 stride=2 的卷积把 16×16 降到 4×4，最后用一个 4×4 的卷积核
把每个样本压成**一个数**（logit）。

没有 `Sigmoid`——损失用 `BCEWithLogitsLoss`，它把 sigmoid 和交叉熵合成了一个数值稳定的算子。

```{code-cell} ipython3
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 8, 4, stride=2, padding=1)      # 16 -> 8
        self.down2 = nn.Conv2d(8, 16, 4, stride=2, padding=1)     # 8 -> 4
        self.bn2 = nn.BatchNorm2d(16)
        self.out = nn.Conv2d(16, 1, 4)                            # 4 -> 1

    def execute(self, x):
        h = nn.leaky_relu(self.down1(x), 0.2)
        h = nn.leaky_relu(self.bn2(self.down2(h)), 0.2)
        return self.out(h).reshape((-1,))         # (batch,)


discriminator = Discriminator()
print("判别器参数量:", sum(p.numel() for p in discriminator.parameters()))
print("输入", real.shape, "-> 输出", discriminator(real).shape)
```

这里有个权重布局上的不一致，很容易写错：

```{code-cell} ipython3
print("生成器的 up1（ConvTranspose）:", generator.up1.weight.shape, "（in, out, kH, kW）")
print("判别器的 down1（Conv2d）     :", discriminator.down1.weight.shape, "（out, in, kH, kW）")
```

两者都是 PyTorch 的布局，所以照着 PyTorch 抄不会出错；但**在 Jittor 里手写初始化或做
权重迁移时**要记住 `ConvTranspose` 的通道顺序是反的。

注意判别器的参数量比生成器小得多（几千对几万）。这不奇怪：**判别器的任务更简单**，
而且 GAN 的常见失败模式恰恰是"判别器太强"。

## 6. 训练：把它写成一个函数

损失和训练循环与 [条件生成对抗网络](conditional_gan.md) 完全一样——DCGAN 换掉的只是网络。
每一步两次更新：先判别器，再生成器。

因为后面要做对照实验，这里把训练包成一个函数。**除了学习率，两次实验的每一行都相同**——
对照组只该改一个变量，否则比出来的差异说明不了任何问题。

```{code-cell} ipython3
bce = nn.BCEWithLogitsLoss()


def train(steps=STEPS, d_lr=2e-3, g_lr=2e-3, seed=0):
    """训一个 GAN，返回 (生成器, 诊断记录, 用时秒数)。

    记录里每项是 (step, D损失, G损失, D准确率)；D准确率是"判别器把真样本判正、
    假样本判负"的比例，在每四分之一处测一次。
    """
    jt.set_global_seed(seed)
    np.random.seed(seed)
    g, d = Generator(), Discriminator()
    g_optim = jt.optim.Adam(g.parameters(), lr=g_lr, betas=(0.5, 0.999))
    d_optim = jt.optim.Adam(d.parameters(), lr=d_lr, betas=(0.5, 0.999))

    history = []
    start = time.perf_counter()
    for step in range(steps):
        # --- 更新判别器：真样本判 1，假样本判 0
        fake = g(jt.randn((BATCH, LATENT)))
        d_loss = (bce(d(sample_real(BATCH)), jt.ones(BATCH))
                  + bce(d(fake.stop_grad()), jt.zeros(BATCH)))
        d_optim.step(d_loss)

        # --- 更新生成器：让判别器把假样本判成 1
        g_loss = bce(d(g(jt.randn((BATCH, LATENT)))), jt.ones(BATCH))
        g_optim.step(g_loss)

        if step % max(1, steps // 4) == 0 or step == steps - 1:
            probe = 64
            correct = (d(sample_real(probe)) > 0).float() \
                * (d(g(jt.randn((probe, LATENT)))) < 0).float()
            history.append((step, float(d_loss.item()), float(g_loss.item()),
                            float(correct.mean().item())))
    return g, history, time.perf_counter() - start


generator, baseline, baseline_seconds = train()
print("基线用时 %.0f s" % baseline_seconds)
for step, d_loss, g_loss, accuracy in baseline:
    print("step %3d  D损失=%.3f  G损失=%.3f  D准确率=%.2f" % (step, d_loss, g_loss, accuracy))
```

这一格要等一两分钟：卷积算子的**即时编译**都发生在这里（原因见
[JIT、算子融合与异步执行](jit_and_async.md)）。后面的对照实验用的是同一批算子，不再重复编译。

## 7. 一条诊断曲线：判别器有多强

把 `D准确率` 那一列单独看，就是 GAN 训练最要紧的一条诊断：

```{code-cell} ipython3
print("step  D准确率")
for step, _, _, accuracy in baseline:
    print("%4d   %.2f  %s" % (step, accuracy, "#" * int(accuracy * 40)))
```

这条曲线量的是「判别器把真样本判正、假样本判负的比例」。它的**两个极端都是坏消息**：

| 读数 | 含义 | 后果 |
| --- | --- | --- |
| 长期 ≈ 1.0 | 判别器太强，真假一眼可分 | 生成器的梯度被压到接近 0，学不动 |
| 长期 ≈ 0.0 | 判别器太弱，或者已经崩了 | 它给不出有用的方向，生成器一样学不动 |
| 0.5 ~ 0.9 之间来回 | 两边在拉锯 | 这是想要的状态 |

**判别器的准确率不是越高越好**——这一点和普通分类任务正好相反。你要的不是一个好判别器，
而是一个骗得过它的生成器。

基线这一次的读数是：起步 0.16，中途冲到 1.00，之后回落到 0.5 附近——**中间那次冲到 1.00
并不代表训练失败**，它只是判别器一时占了上风，随后生成器追了上来。看这条曲线要看的是
趋势，不是某一次读数。

下一节把判别器**故意调弱**，看这条曲线会怎么变、生成结果又变成什么样。

## 8. 生成结果检验

看图只能看个大概。既然数据是模板生成的，就能直接量：把每张生成图**和四个模板比距离**，
取最近的那个。

这样能回答两个不同的问题：

* **到底像不像**：有多少生成样本落在某个模板附近（识别率）；
* **塌缩了没有**：命中的模板覆盖了几种。

只看第一个数会漏掉模式塌缩——如果生成器只会画"横条"，识别率可以很高，但它只会一类。

```{code-cell} ipython3
def inspect_generator(model, samples=128):
    """返回 (像某个模板的样本数, 总数, 覆盖到的模板, 取值铺满 [-1,1] 的比例)。"""
    fakes = model(jt.randn((samples, LATENT))).numpy()[:, 0].reshape(samples, -1)
    distance = ((fakes[:, None, :] - FLAT_TEMPLATES[None, :, :]) ** 2).mean(axis=2)
    nearest = distance.argmin(axis=1)
    close = distance.min(axis=1) < 0.3
    contrast = float(((fakes.min(axis=1) < -0.5) & (fakes.max(axis=1) > 0.5)).mean())
    return int(close.sum()), samples, sorted(set(nearest[close].tolist())), contrast


recognised, total, modes, contrast = inspect_generator(generator)
print("像某个模板的样本: %d/%d" % (recognised, total))
print("覆盖到的模板    : %s（共 4 种）" % modes)
print("取值铺满 [-1,1] 的比例: %.2f" % contrast)

assert recognised >= total // 2, "生成器没学出可辨认的形状"
assert len(modes) >= 2, "疑似模式塌缩：只生成了一两类"
assert contrast == 1.0, "生成器的输出没有铺满 [-1, 1]"
```

最后那个 `contrast` 检查的是 `Tanh` 有没有真的被用起来。如果生成器把所有像素都推到一个
狭窄的区间（比如全在 0 附近），图会是一片灰——`BatchNorm` 加在输出层上就会这样，
所以第 1 节的表里特意说输出层不能加它。

换机器或换显卡时具体数字会变（CPU 和 GPU 的浮点顺序不同），但训练正常时三个断言都该通过。

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

生成样本通常有明显的网格状伪影，这是转置卷积的已知副作用（相邻输出像素的卷积核重叠次数
不均匀）。想彻底去掉它，可以把上采样换成"最近邻插值 + 普通卷积"，代价是参数更多。

## 9. 对照实验：把判别器调弱

现在做那个对照：**同样的种子、同样的生成器结构**，只把判别器的学习率降到 `3e-4`
（比生成器小约 7 倍）。判别器这一路从第一步起就分不出真假，所以跑 100 步就足够看清楚。

```{code-cell} ipython3
weak_generator, weak_history, weak_seconds = train(steps=100, d_lr=3e-4)
print("对照实验用时 %.0f s" % weak_seconds)
for step, d_loss, g_loss, accuracy in weak_history:
    print("step %3d  D损失=%.3f  G损失=%.3f  D准确率=%.2f" % (step, d_loss, g_loss, accuracy))


weak_recognised, _, weak_modes, _ = inspect_generator(weak_generator)
print()
print("%-20s %-9s %s" % ("判别器学习率", "识别", "覆盖模式"))
print("%-20s %-9s %s" % ("2e-3（与生成器相同）", "%d/%d" % (recognised, total), modes))
print("%-20s %-9s %s" % ("3e-4（小 7 倍）", "%d/%d" % (weak_recognised, total), weak_modes))
```

这一次的结果值得仔细读，因为它**不是**一个"把学习率调小就变好"的故事。三件事同时发生了：

* **判别器彻底失效**：准确率一直贴着 `0.00`，而且它的损失不降反升（1.4 ~ 2.0）——
  它不是"还没学好"，是**自信地判错**：把真的判成假、假的判成真；
* **生成器很轻松**：`G损失` 一直很低（0.3 ~ 0.7），因为它随便画点什么都能骗过判别器；
* **生成结果什么都不是**：识别率掉到 `0/128`，连一个模板都没命中。

最后一条是这一节的重点：**`G损失` 低不等于生成得好**。生成器「骗过了判别器」，但它学到的
不是数据分布，而是判别器当前的弱点。第 7 节那张表里的**两个极端都要避免**——判别器太强，
生成器没有梯度；判别器太弱，生成器没有方向。

所以判断 GAN 训得好不好，只能看**生成结果本身**（第 8 节的指标），损失曲线和判别器准确率
都只是线索。

> 学习率只是调节判别器强弱的旋钮之一。另外两个常被提到的做法是给判别器的输入加噪声
> （标签平滑、实例噪声），以及换一个不依赖判别器"有多自信"的损失——后者就是 WGAN 系列
> 要解决的问题。它们改的都是同一件事：**别让判别器太早、太彻底地赢**。

## 10. 放大到真实数据集

上面的网络和训练循环放大到 MNIST/CIFAR 时，**结构不用改**，改的是三处：

* 输入从 16×16 提到 32×32——多加一层 `ConvTranspose`（32 → 16 → 8 → 4 的阶梯反过来用）；
* 通道数从 8/16 提到 64/128；
* 数据换成 `jt.dataset` 的 loader，并按标准做法把像素归一到 `[-1, 1]`。

```{code-cell} ipython3
:tags: [gan, network, long-running, skip-execution]

from jittor.dataset.mnist import MNIST
import jittor.transform as transform

train_loader = MNIST(train=True, transform=transform.Compose([
    transform.Resize(32),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),   # -> [-1, 1]
])).set_attrs(batch_size=64, shuffle=True)


class BigGenerator(nn.Module):
    """32 -> 16 -> 8 -> 4 的上采样阶梯，通道数按 1/2 递减。"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 256 * 4 * 4)
        self.bn0 = nn.BatchNorm2d(256)
        self.up1 = nn.ConvTranspose(256, 128, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.up2 = nn.ConvTranspose(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.up3 = nn.ConvTranspose(64, 32, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def execute(self, z):
        h = nn.relu(self.bn0(self.fc(z).reshape((-1, 256, 4, 4))))
        h = nn.relu(self.bn1(self.up1(h)))
        h = nn.relu(self.bn2(self.up2(h)))
        h = nn.relu(self.bn3(self.up3(h)))
        return jt.tanh(self.out(h))       # (N, 1, 32, 32)


big_generator = BigGenerator()
big_discriminator = Discriminator()
print("生成器输出:", big_generator(jt.randn((2, 100))).shape)
```

判别器也要同步加深一层（32 → 16 → 8 → 4 → 1）。真实训练通常要几万步，这里只是把形状
走通，确认换成真实数据后维度对得上。

## 11. 检查清单

- 输出尺寸算对了吗？`k=4, s=2, p=1` 翻倍；其它组合套第 4 节的公式。
- `ConvTranspose` 的权重是 `(in, out, kH, kW)`，和 `Conv2d` 的 `(out, in, kH, kW)` 相反。
- 生成器输出层加 `BatchNorm` 了吗？加了就会得到一片灰。
- 判别器的**第一层**加 `BatchNorm` 了吗？不该加。
- 判别器准确率长期贴着 1.0 吗？那是判别器太强，不是训练得好。
- 只看识别率够不够？还要看**覆盖了几个模板**——否则会漏掉模式塌缩。
- 训练曲线：`D损失` 和 `G损失` 一起上下摆是正常的，不要指望它们单调下降。

相关主题：对抗训练的基本概念见 [条件生成对抗网络](conditional_gan.md)；为什么第一格那么慢见
[JIT、算子融合与异步执行](jit_and_async.md)；数据管道见 [数据加载与预处理](data_pipeline.md)。

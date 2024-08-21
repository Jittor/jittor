# 图像生成之LSGAN

生成对抗网络（GAN, Generative Adversarial Networks ）是一种深度学习模型，是近年来复杂分布上无监督学习最具前景的方法之一。GAN模型由生成器（Generator）和判别器（Discriminator）两个部分组成。在训练过程中，生成器的目标就是尽量生成真实的图片去欺骗判别器。而判别器的目标就是尽量把生成器生成的图片和真实的图片分别开来。这样，生成器和判别器构成了一个动态的“博弈过程”。许多相关的研究工作表明GAN能够产生效果非常真实的生成效果。

本教程使用Jittor框架实现了一种经典GAN模型LSGAN 。LSGAN将GAN的目标函数由交叉熵损失替换成最小二乘损失，以此拒绝了标准GAN生成的图片质量不高以及训练过程不稳定这两个缺陷。本教程通过LSGAN的实现介绍了Jittor数据加载、模型定义、模型训练的使用方法。

LSGAN论文：<https://arxiv.org/abs/1611.04076>

## 1.数据集准备

本教程使用两种数据集进行LSGAN的训练，分别是Jittor自带的数据集MNIST，和用户构建的数据集CelebA。

如果要使用CelebA数据集进行训练，可以通过以下链接下载CelebA数据集。

- CelebA 数据集： <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>

将下载的训练数据和验证数据分别存储在`./data/celebA_train/imgs/`和`./data/celebA_eval/imgs/`中

最终数据集的文件组织如下。

```
# 文件组织
根目录
|----data
     |----celebA_train
     |    |----imgs
     |----celebA_eval
     |    |----imgs
```

## 2.模型定义

本教程使用LSGAN进行图像生成，其网络结构由生成器和别器。生成器网络输入一个`1024`维的向量，生成分辨率为`112*112`的图像；判别器网络输入`112*112`的图像，输出一个数字表示输入图像为真实图像的可信程度。

下面分别定义生成器和判别器

```python
import jittor as jt
from jittor import nn, Module, init
from jittor.dataset.mnist import MNIST
from jittor.dataset.dataset import ImageFolder
import jittor.transform as transform
import os
import numpy as np
import matplotlib.pyplot as plt

# 如果有CUDA，则通过use_cuda设置在GPU上进行训练
if jt.has_cuda:
    jt.flags.use_cuda = 1

class generator(Module):
    def __init__(self, dim=3):
        super(generator, self).__init__()
        self.fc = nn.Linear(1024, 7*7*256)
        self.fc_bn = nn.BatchNorm(256)
        self.deconv1 = nn.ConvTranspose(256, 256, 3, 2, 1, 1)
        self.deconv1_bn = nn.BatchNorm(256)
        self.deconv2 = nn.ConvTranspose(256, 256, 3, 1, 1)
        self.deconv2_bn = nn.BatchNorm(256)
        self.deconv3 = nn.ConvTranspose(256, 256, 3, 2, 1, 1)
        self.deconv3_bn = nn.BatchNorm(256)
        self.deconv4 = nn.ConvTranspose(256, 256, 3, 1, 1)
        self.deconv4_bn = nn.BatchNorm(256)
        self.deconv5 = nn.ConvTranspose(256, 128, 3, 2, 1, 1)
        self.deconv5_bn = nn.BatchNorm(128)
        self.deconv6 = nn.ConvTranspose(128, 64, 3, 2, 1, 1)
        self.deconv6_bn = nn.BatchNorm(64)
        self.deconv7 = nn.ConvTranspose(64 , dim, 3, 1, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def execute(self, input):
        x = self.fc(input).reshape((input.shape[0], 256, 7, 7))
        x = self.relu(self.fc_bn(x))
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        x = self.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.relu(self.deconv4_bn(self.deconv4(x)))
        x = self.relu(self.deconv5_bn(self.deconv5(x)))
        x = self.relu(self.deconv6_bn(self.deconv6(x)))
        x = self.tanh(self.deconv7(x))
        return x


class discriminator(nn.Module):
    def __init__(self, dim=3):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv(dim, 64, 5, 2, 2)
        self.conv2 = nn.Conv(64, 128, 5, 2, 2)
        self.conv2_bn = nn.BatchNorm(128)
        self.conv3 = nn.Conv(128, 256, 5, 2, 2)
        self.conv3_bn = nn.BatchNorm(256)
        self.conv4 = nn.Conv(256, 512, 5, 2, 2)
        self.conv4_bn = nn.BatchNorm(512)
        self.fc = nn.Linear(512*7*7, 1)
        self.leaky_relu = nn.Leaky_relu()

    def execute(self, input):
        x = self.leaky_relu(self.conv1(input), 0.2)
        x = self.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = self.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = x.reshape((x.shape[0], 512*7*7))
        x = self.fc(x)
        return x
```

损失函数采用最小二乘损失函数。具体实现如下，`x`为生成器的输出值，`b`表示该图像是否希望被判别为真。

```python
def ls_loss(x, b):
    mini_batch = x.shape[0]
    y_real_ = jt.ones((mini_batch,))
    y_fake_ = jt.zeros((mini_batch,))
    if b:
        return (x-y_real_).sqr().mean()
    else:
        return (x-y_fake_).sqr().mean()
```

## 3.模型训练

参数设定如下：

```python
# 使用 MNIST 或者 CelebA数据集进行训练
task = "MNIST"
# task = "CelebA"
# 批大小
batch_size = 128
# 学习率
lr = 0.0002
# 训练轮数
train_epoch = 20 if task=="MNIST" else 50
# 训练图像标准大小
img_size = 112
# Adam优化器参数
betas = (0.5,0.999)
# 数据集图像通道数，MNIST为1，CelebA为3
dim = 1 if task=="MNIST" else 3
# 结果图片存储路径
save_path = "./results_img"
```

分别声明生成器和判别器，并使用Adam作为优化器。

```python
G = generator (dim)
D = discriminator (dim)
G_optim = nn.Adam(G.parameters(), lr, betas=betas)
D_optim = nn.Adam(D.parameters(), lr, betas=betas)
```

jittor自带有MNIST数据集。使用`jittor.transform`可以进行数据归一化及数据增强，这里本教程通过`transform`将图片归一化到指定区间，并resize到标准大小`112*112`。。通过`set_attrs`函数可以修改数据集的相关参数，如`batch_size`、`shuffle`及`transform`等。

如果使用自己构建CelebA数据集进行训练，可以通过通用数据加载器`jittor.dataset.dataset.ImageFolder`，输入数据集路径即可构建用户数据集。

构建数据集代码如下：

```python
if task=="MNIST":
    transform = transform.Compose([
        transform.Resize(size=img_size),
        transform.Gray(),
        transform.ImageNormalize(mean=[0.5], std=[0.5]),
    ])
    train_loader = MNIST(train=True, transform=transform).set_attrs(batch_size=batch_size, shuffle=True)
    eval_loader = MNIST(train=False, transform = transform).set_attrs(batch_size=batch_size, shuffle=True)
elif task=="CelebA":
    transform = transform.Compose([
        transform.Resize(size=img_size),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_dir = './data/celebA_train'
    train_loader = ImageFolder(train_dir).set_attrs(transform=transform, batch_size=batch_size, shuffle=True)
    eval_dir = './data/celebA_eval'
    eval_loader = ImageFolder(eval_dir).set_attrs(transform=transform, batch_size=batch_size, shuffle=True)
```

训练和验证代码如下：

```python
def train(epoch):
    for batch_idx, (x_, target) in enumerate(train_loader):
        mini_batch = x_.shape[0]
        # train discriminator
        D_result = D(x_)
        D_real_loss = ls_loss(D_result, True)
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_result_ = D(G_result)
        D_fake_loss = ls_loss(D_result_, False)
        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.sync()
        D_optim.step(D_train_loss)

        # train generator
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_result = D(G_result)
        G_train_loss = ls_loss(D_result, True)
        G_train_loss.sync()
        G_optim.step(G_train_loss)
        if (batch_idx%100==0):
            print("train: batch_idx",batch_idx,"epoch",epoch)
            print('  D training loss =', D_train_loss.data.mean())
            print('  G training loss =', G_train_loss.data.mean())

def validate(epoch):
    D_losses = []
    G_losses = []
    G.eval()
    D.eval()
    for batch_idx, (x_, target) in enumerate(eval_loader):
        mini_batch = x_.shape[0]
        
        # calculation discriminator loss
        D_result = D(x_)
        D_real_loss = ls_loss(D_result, True)
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_result_ = D(G_result)
        D_fake_loss = ls_loss(D_result_, False)
        D_train_loss = D_real_loss + D_fake_loss
        D_losses.append(D_train_loss.data.mean())

        # calculation generator loss
        z_ = jt.init.gauss((mini_batch, 1024), 'float')
        G_result = G(z_)
        D_result = D(G_result)
        G_train_loss = ls_loss(D_result, True)
        G_losses.append(G_train_loss.data.mean())
    G.train()
    D.train()
    print("validate: epoch",epoch)
    print('  D validate loss =', np.array(D_losses).mean())
    print('  G validate loss =', np.array(G_losses).mean())
```

使用每个epoch的生成器通过固定向量生成图片，将图片显示并存储在`./results_img/`中

```python
if not os.path.exists(save_path):
    os.mkdir(save_path)
fixed_z_ = jt.init.gauss((5 * 5, 1024), 'float')
def save_result(num_epoch, G , path = 'result.png'):
    """Use the current generator to generate 5*5 pictures and store them.

    Args:
        num_epoch(int): current epoch
        G(generator): current generator
        path(string): storage path of result image
    """

    z_ = fixed_z_
    G.eval()
    test_images = G(z_)
    G.train()
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i in range(size_figure_grid):
        for j in range(size_figure_grid):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        if task=="MNIST":
            ax[i, j].imshow((test_images[k, 0].data+1)/2, cmap='gray')
        else:
            ax[i, j].imshow((test_images[k].data.transpose(1, 2, 0)+1)/2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)
    plt.show()
```

现在，让我们训练一番试试！

```python
for epoch in range(train_epoch):
    print ('number of epochs', epoch)
    train(epoch)
    validate(epoch)
    result_img_path = './results_img/' + task + str(epoch) + '.png'
    save_result(epoch, G, path=result_img_path)
```


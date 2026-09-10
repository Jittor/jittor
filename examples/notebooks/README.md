# Jittor 教程

这里是可执行教程的唯一权威来源。每篇都是 MyST Markdown，由 Jupytext 生成 notebook；
**生成物不进仓库**。

下面是一条有顺序的学习路径。每篇都注明它教什么、需要什么前置，以及是否需要显卡。

## 从哪开始

有两条入门线，覆盖的内容有重叠，按你的背景选一条即可：

* **零基础**（没写过深度学习代码）→ 走「60 分钟入门」四篇，从安装一路到 MNIST。
* **有 PyTorch 等框架经验**→ 直接走下面的「主线」，它假设你懂张量和训练循环，
  只讲 Jittor 与它们的差异。

两条线都走完不会浪费，但没有必要。

### 60 分钟入门（零基础，中文）

| 教程 | 教什么 |
| --- | --- |
| [0 介绍与安装](60分钟快速入门Jittor/计图入门教程%200%20---%20介绍与安装.md) | 装好、跑通第一行代码 |
| [1 基本概念](60分钟快速入门Jittor/计图入门教程%201%20---%20基本概念.md) | Var 的创建、形状、运算 |
| [2 训练一个线性回归](60分钟快速入门Jittor/计图入门教程%202%20---%20如何训练一个简单线性回归.md) | 完整训练循环的最小形态 |
| [3 解决一个实际问题](60分钟快速入门Jittor/计图入门教程%203%20---%20尝试解决一个实际问题.md) | MNIST 手写数字识别 |

## 主线

### 一、基础

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [算子与 Var](basics.md) | Jittor 的两个核心概念；异步计算与 `sync` | 否 |
| [设备与驻留](device_placement.md) | 数据到底在哪；全局 `use_cuda` 与 per-tensor 的差异；**读一个设备张量会把它搬走** | 部分小节需要 |
| [模型定义与训练](example.md) | `Module`、参数、优化器、完整训练循环 | 否 |

**建议按上面的顺序读。**「设备与驻留」放在训练之前是有意的：Jittor 的设备模型与
PyTorch 不同，不先弄清楚，后面的性能问题会归因到错误的地方。

### 二、Jittor 的核心机制

这两篇讲的是 Jittor 区别于其它框架的地方。想理解它为什么快、或者想扩展它，读这两篇。

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [元算子](meta_op.md) | 用三个元算子拼出卷积——Jittor 的算子表达方式 | 否 |
| [自定义 C++ 与 CUDA 算子](custom_op.md) | 写一个自己的算子并即时编译 | 部分小节需要 |

### 三、训练实战

按从简到繁排列。每篇都是可以跑起来的完整模型。

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [残差网络训练](resnet_training.md) | 图像分类的标准训练流程 | 否（小规模） |
| [从零实现 Transformer](transformer.md) | 注意力与编码器的实现细节 | 否（小规模） |
| [视觉 Transformer 训练](vit_training.md) | 把 Transformer 用在图像上 | 否（小规模） |
| [训练 GPT 风格语言模型](gpt2_training.md) | 自回归语言建模与生成 | 否（小规模） |
| [从零实现去噪扩散模型](diffusion.md) | 扩散过程的前向与反向 | 否（小规模） |
| [LoRA 参数高效微调](lora.md) | 只训练低秩增量而不动主干权重 | 否（小规模） |

### 四、性能

模型跑通之后再读这两篇。先正确，再快。

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [性能分析器](profiler.md) | 找出时间花在哪个算子上 | 否 |
| [混合精度](mixed_precision.md) | 以图策略的方式控制精度 | 部分小节需要 |

### 五、生成模型

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [Conditional GAN](ConditionGAN.md) | 条件生成对抗网络 | 部分小节需要 |
| [LSGAN 图像生成](LSGAN.md) | 最小二乘 GAN | 部分小节需要 |

配套的可运行 Web 示例在 [`examples/gan/`](../gan/README.md)。

## 在本地打开

```bash
python -m pip install -r requirements/examples.txt
```

notebook 生成到仓库外，避免在工作区里留下产物：

```bash
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
mkdir -p "$STATE"
python -m jupytext --to ipynb \
  --output "$STATE/basics.ipynb" examples/notebooks/basics.md
python -m notebook --ServerApp.root_dir="$STATE"
```

校验全部源文件并执行离线 CPU 冒烟教程：

```bash
python -m nox -s tutorials
```

## 写教程的约定

notebook 单元格用这些执行标签：

- `network`：需要下载数据或其它远程资源。
- `cuda`：需要或显式启用 CUDA。
- `gan`：属于 GAN 工作流。
- `long-running`：训练或其它昂贵操作。
- `interactive`：产生交互式或帮助类输出，不适合冒烟运行。
- `skip-execution`：不纳入仓库维护的 CPU 冒烟测试。

生成的 notebook 不得带有已保存的输出或执行计数。MyST 源里不得出现机器相关的
缓存、home 或环境路径。

**新增一篇教程时，必须同时把它加进上面的学习路径。**
`tests/integration/test_notebooks.py` 会检查这一点：目录里有而路径里没有的教程会让
门禁变红。一篇没人知道该在什么时候读的教程，和一篇不存在的教程价值相同。

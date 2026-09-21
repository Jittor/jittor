# Jittor 教程

这里是可执行教程的唯一权威来源。每篇都是 **MyST Markdown**，由 Jupytext 生成
notebook；**生成物（`.ipynb`）不进仓库**，一律写到仓库之外。

下面是一条有顺序的学习路径。每篇都注明它教什么、需要什么前置、是否需要显卡。

---

## 从哪开始

**只有一个入口**：[从零开始：装好、跑通、训练第一个模型](getting_started.md)。
它从安装讲到训练，全程 CPU 可跑完，不需要任何深度学习背景。

跑完这一篇之后，按你的背景选一条主线：

* **继续用原生 Jittor**（默认）→ 走下面的「第一线：原生 Jittor」。
* **手上已有 PyTorch 代码** → 走「第二线：PyTorch 兼容」，让你的 `torch` 代码不改一行
  跑在 Jittor 上。

两条线可以都走，但**先走完一条**再换，混着读容易把两套设备模型和优化器语义记串。

---

## 第一线：原生 Jittor

### 一、基础

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [算子与 Var](basics.md) | 两个核心概念；异步计算与 `sync` | 否 |
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
| [条件生成对抗网络](conditional_gan.md) | 条件是怎么接进生成器和判别器的；**原生 Jittor 与 PyTorch 兼容两种写法**；怎么用指标检验条件真的生效 | 否 |

这一篇不需要显卡、不需要下载数据，CPU 上一分钟跑完。它同时站在两条线的交叉点上：前半段
是原生 Jittor，后半段用 PyTorch 兼容层把同一个模型再写一遍——想快速看清两种 API 的差别，
这一篇是最短的入口。

配套的可运行 Web 示例在 [`examples/gan/`](../gan/README.md)。

---

## 第二线：PyTorch 兼容

这条路讲的是「不重写代码」：激活兼容层之后，PyTorch 的 API 由 Jittor 实现。

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [用 PyTorch API 写 Jittor](torch_compat.md) | 怎么激活；激活后张量/模型/自动求导怎么用；设备与混合精度 | 部分小节需要 |
| [把已有 PyTorch 脚本迁到 Jittor](torch_compat_migration.md) | 同一计算两种写法的数值对照；迁移时必须改的四处 | 否 |

原生写法与兼容写法的对照速记：

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 前向方法名 | `execute` | `forward` |
| 训练一步 | `optim.step(loss)` | `zero_grad()` / `backward()` / `step()` |
| 设备 | 全局 `jt.flags.use_cuda` | 全局，或按 PyTorch 语义 `.cuda()` |
| 入口 | `import jittor as jt` | 先 `activate()` 再 `import torch` |

生态库（transformers / diffusers / PEFT 等）的接入方式与已验证结论，见维护者文档
`docs/compatibility/torch.md`。

---

## 在本地打开

### 1. 装教程工具链

```bash
python -m pip install -r requirements/examples.txt
```

### 2. 把 Markdown 源生成 notebook

生成物写到仓库之外，避免在工作区留下 `.ipynb`：

```bash
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
mkdir -p "$STATE"

# 只生成一篇
python -m jupytext --to ipynb \
  --output "$STATE/getting_started.ipynb" examples/notebooks/getting_started.md

# 或者一次生成全部（保留目录结构）
cp -R examples/notebooks/. "$STATE/"
find "$STATE" -type f -name '*.md' ! -name README.md -print0 \
  | xargs -0 -n1 python -m jupytext --to ipynb
```

### 3. 启动 Jupyter

```bash
python -m notebook --ServerApp.root_dir="$STATE"
```

在浏览器里打开刚生成的 `.ipynb` 即可。第一次执行代码单元会比之后慢——那是 Jittor 在
即时编译算子；第二次就快了。

### 4. 离线校验（维护者）

生成、静态契约与离线 CPU 冒烟一次跑完：

```bash
python -m nox -s tutorials
```

---

## 写教程的约定

notebook 单元格用这些执行标签：

- `network`：需要下载数据或其它远程资源。
- `cuda`：需要或显式启用 CUDA。
- `gan`：属于 GAN 工作流。
- `long-running`：训练或其它昂贵操作。
- `interactive`：产生交互式或帮助类输出，不适合冒烟运行。
- `skip-execution`：不纳入仓库维护的 CPU 冒烟测试。

带 `network`/`cuda`/`long-running` 的单元格**必须同时带 `skip-execution`**；带
`skip-execution` 的单元格**必须**给出一个原因标签。

生成的 notebook 不得带有已保存的输出或执行计数。MyST 源里不得出现机器相关的
缓存、home 或环境路径。

**新增一篇教程时，必须同时把它加进上面的学习路径。**
`tests/integration/test_notebooks.py` 会检查这一点：目录里有而路径里没有的教程会让
门禁变红。一篇没人知道该在什么时候读的教程，和一篇不存在的教程价值相同。

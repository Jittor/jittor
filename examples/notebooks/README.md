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
| [数据加载与预处理](data_pipeline.md) | 自定义 `Dataset`；`len(loader)`/`shuffle`/`num_workers` 的真实语义；`jt.transform` 与 torchvision 的差别 | 否 |
| [检查点与断点续训](checkpointing.md) | 训练状态包含哪些部分；为什么只存权重会静默毁掉续训；`load_state_dict` 为什么不报错 | 否 |

**建议按上面的顺序读。**「设备与驻留」放在训练之前是有意的：Jittor 的设备模型与
PyTorch 不同，不先弄清楚，后面的性能问题会归因到错误的地方。

### 二、执行与求导机制

这两篇讲 Jittor 内部怎么跑：一篇是执行（先建图、后融合、异步），一篇是求导。它们解释的
是 Jittor 与 PyTorch 在语义上的几处关键差异——读懂之后再写训练循环，能省掉很多
「结果对不上但又不报错」的排查。

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [JIT、算子融合与异步执行](jit_and_async.md) | 惰性图是怎么执行的；一串小算子如何被融合成一个核；打断融合的代价；编译缓存命中什么 | 否 |
| [自动求导](autograd_details.md) | `jt.grad` 把梯度当返回值；什么时候会静默拿到全 0；梯度存在优化器里；累加与 `zero_grad` | 否 |

### 三、扩展 Jittor

想理解它为什么能这样表达算子、或者想加一个自己的算子，读这两篇。

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [元算子](meta_op.md) | 用三个元算子拼出卷积——Jittor 的算子表达方式；**与 `nn.Conv2d` 对拍并看融合后的内核** | 否 |
| [自定义 C++ 与 CUDA 算子](custom_op.md) | 三条路怎么选；`jt.code` 的内核源码；**为什么反向必须自己写**；双后端写法 | 部分小节需要 |

### 四、训练实战

按从简到繁排列。每篇都是可以跑起来的完整模型。

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [残差网络训练](resnet_training.md) | 完整的分类训练流程（划分、逐 epoch 评估、`eval()` 模式）；**残差 vs 普通卷积的对照实验**；一个与教科书预期相反的读数 | 否（约 2.5 分钟） |
| [循环网络](rnn.md) | 隐藏状态与 BPTT；**普通 RNN 与 LSTM 在长依赖上的实测差距**；截断 BPTT 剪断了什么 | 否（小规模） |
| [从零实现 Transformer](transformer.md) | 注意力的实现细节；**缩放系数与因果掩码的实测**；在一个已知满分的复制任务上训练到 1.000 | 否（小规模） |
| [视觉 Transformer 训练](vit_training.md) | patch embedding、class token 与位置编码；**可学习位置 vs 正弦的对照**；注意力图为什么不能过度解读 | 否（小规模） |
| [训练 GPT 风格语言模型](gpt2_training.md) | 自回归语言建模与生成 | 否（小规模） |
| [从零实现去噪扩散模型](diffusion.md) | 扩散过程的前向与反向 | 否（小规模） |
| [LoRA 参数高效微调](lora.md) | 只训练低秩增量；冻结、恒等初始化、合并回权重三个性质都要实测；**rank 阶梯与它的前提** | 否（小规模） |

`rnn.md` 是这一组里唯一处理**变长输入**的一篇，也是唯一把「为什么这个结构不行」当成
主线的一篇：它用一个能测的记忆任务，把「普通 RNN 学不会长依赖」从经验说法变成一个
可以复现的数字；`transformer.md` 接着给出绕开这条路的方法。

### 五、性能

模型跑通之后再读这两篇。先正确，再快。

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [性能分析器](profiler.md) | 找出时间花在哪个算子上 | 否 |
| [混合精度](mixed_precision.md) | 用**一个整数**控制整张图的精度策略；十个算子的实测 dtype 表；为什么 `exp` 被单独留住；与 `autocast` 的关系 | 部分小节需要 |

### 六、生成模型

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [条件生成对抗网络](conditional_gan.md) | 条件是怎么接进生成器和判别器的；**原生 Jittor 与 PyTorch 兼容两种写法**；怎么用指标检验条件真的生效 | 否 |
| [深度卷积生成对抗网络](dcgan.md) | 用卷积替换全连接；`ConvTranspose` 的尺寸与权重布局；**判别器强弱的对照实验**；模式塌缩怎么发现 | 否 |
| [WGAN-GP](wgan_gp.md) | 把「判别器有多强」换成一个能读的距离估计；Lipschitz 约束与梯度惩罚；**一个会把惩罚悄悄变成常数的写法陷阱** | 否 |
| [变分自编码器](vae.md) | 重参数化技巧；KL 项的闭式解；β 阶梯上的三种失败方式；**为什么「重建得好」会选错模型** | 否 |

这四篇不需要显卡、不需要下载数据。`conditional_gan.md` 同时站在两条线的交叉点上：前半段是
原生 Jittor，后半段用 PyTorch 兼容层把同一个模型再写一遍——想快速看清两种 API 的差别，
这一篇是最短的入口。`dcgan.md` 是它的续篇：把网络换成真正的卷积结构，并用一次对照实验说明
判别器过强和过弱分别会发生什么。`wgan_gp.md` 再往前一步，换掉损失函数本身，让训练的好坏
第一次有一个能读的数；`vae.md` 走另一条路（似然下界），重点在「怎么判断它学会了」。
**`dcgan.md` 和 `wgan_gp.md` 明显更重**（CPU 上各约五分钟，大半是卷积算子的即时编译）；
`vae.md` 是纯 MLP，一分钟以内。

配套的可运行 Web 示例在 [`examples/gan/`](../gan/README.md)。

---

## 第二线：PyTorch 兼容

这条路讲的是「不重写代码」：激活兼容层之后，PyTorch 的 API 由 Jittor 实现。

| 教程 | 教什么 | 需要显卡 |
| --- | --- | --- |
| [用 PyTorch API 写 Jittor](torch_compat.md) | 怎么激活；激活后张量/模型/自动求导怎么用；设备与混合精度 | 部分小节需要 |
| [把已有 PyTorch 脚本迁到 Jittor](torch_compat_migration.md) | 同一计算两种写法的数值对照；哪些**不用改**；跨前端存档的静默陷阱；兼容层的实测开销 | 否 |
| [把真实的生态库跑在 Jittor 上](torch_ecosystem.md) | 别人写的库（transformers/diffusers/PEFT/MMCV）怎么带起来；**哪些库根本带不动**；怎么验证算得对 | 否 |

三篇的顺序是「从自己的代码，到别人的代码」：`torch_compat.md` 教你把自己的代码
改成 `import torch`；`torch_compat_migration.md` 讲迁移时真正要动的那几处（以及哪些
**不用**动）；`torch_ecosystem.md` 处理最省事也最容易踩空的一类——从 pip 装进来的
第三方库，它的源码你改不了，也不该改。

原生写法与兼容写法的对照速记：

| | 原生 Jittor | PyTorch 兼容层 |
| --- | --- | --- |
| 前向方法名 | `execute` | `forward` |
| 训练一步 | `optim.step(loss)` | `zero_grad()` / `backward()` / `step()` |
| 设备 | 全局 `jt.flags.use_cuda` | 全局，或按 PyTorch 语义 `.cuda()` |
| 入口 | `import jittor as jt` | 先 `activate()` 再 `import torch` |

生态库（transformers / diffusers / PEFT 等）的接入方式与「哪些库根本带不动」见
[把真实的生态库跑在 Jittor 上](torch_ecosystem.md)；完整的已验证模型清单与门禁状态见
维护者文档 `docs/compatibility/torch.md`。

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

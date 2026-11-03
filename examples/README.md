# Jittor Examples

Examples are source-distribution assets and are intentionally excluded from the
runtime wheel. They are not a Python package and importing an example must not
download data, initialize a device, open a port, or create a Jittor cache.

## What is here

| 目录 | 内容 |
| --- | --- |
| [`notebooks/`](notebooks/README.md) | 可执行教程。MyST Markdown 是唯一权威来源，notebook 由 Jupytext 生成 |
| [`gan/`](gan/README.md) | 可直接运行的 Web 应用示例（启动时自训练，不依赖外部权重） |
| 生态库 | 下游集成已迁到各自的仓库（见下节）；本目录只保留 [`minimax-h3/`](minimax-h3/README.md) 与 [`flash-attention/`](flash-attention/README.md) 两个指针页 |

**教程的完整阅读顺序在 [`notebooks/README.md`](notebooks/README.md)**，那里有两条主线：

* **原生 Jittor**：从 [`notebooks/getting_started.md`](notebooks/getting_started.md) 起步，
  一路到训练、性能与生成模型。
* **PyTorch 兼容**：从 [`notebooks/torch_compat.md`](notebooks/torch_compat.md) 起步，
  让已有的 `torch` 代码不改一行跑在 Jittor 上。

## 生态库（15 个独立仓库）

主仓库只保留 Jittor 本体与 `jittor.compat.torch` 这一层共享前端；下游集成从 2026 年
10 月起全部迁到 [JittorEcosystem](https://github.com/JittorEcosystem) 下的独立仓库。
每个仓库负责同一组事：**pin**（`compatibility.json` 里的上游版本与 `jittor-compat` 版本）、
**adapter**（需要时通过入口点 `jittor.module_patches` 注册，不需要就不带）、**验证记录**
（官方测试子集、示例，以及哪些结论来自真实设备）。布局也是同一套：`README`、`AGENTS.md`、
`compatibility.json`、`docs/`、`examples/`、`tests/`、`support/`。

| 分组 | 仓库 |
| --- | --- |
| 模型与服务框架 | [transformers](https://github.com/JittorEcosystem/jittor-transformers) · [vllm](https://github.com/JittorEcosystem/jittor-vllm) · [deepspeed](https://github.com/JittorEcosystem/jittor-deepspeed) · [accelerate](https://github.com/JittorEcosystem/jittor-accelerate) · [ray](https://github.com/JittorEcosystem/jittor-ray) · [verl](https://github.com/JittorEcosystem/jittor-verl) · [trl](https://github.com/JittorEcosystem/jittor-trl) · [ms-swift](https://github.com/JittorEcosystem/jittor-ms-swift) |
| 数据与格式 | [datasets](https://github.com/JittorEcosystem/jittor-datasets) · [tokenizers](https://github.com/JittorEcosystem/jittor-tokenizers) · [safetensors](https://github.com/JittorEcosystem/jittor-safetensors) · [tensordict](https://github.com/JittorEcosystem/jittor-tensordict) |
| 指标 | [torchmetrics](https://github.com/JittorEcosystem/jittor-torchmetrics) |
| 内核与服务 | [flash-attention](https://github.com/JittorEcosystem/jittor-flash-attn) · [MiniMax-H3](https://github.com/JittorEcosystem/jittor-minimax-h3) |

这 15 个是**并列关系**：flash-attention 和 MiniMax-H3 与 transformers、vllm 一样，
都有独立的版本 pin、独立的主机契约和各自的验证边界。逐个库的版本、用法，以及
「出问题该改哪一层」的判据，见教程
[把真实的生态库跑在 Jittor 上](notebooks/torch_ecosystem.md#6-生态库是并列的一个库一个仓库)。

不在表里的库（diffusers、PEFT、mmcv/mmengine 等）直接落在 `jittor.compat.torch` 的公开
API 上，没有单独的补丁仓库，也就没有独立的版本 pin：跟着 `jittor-compat` 的版本走。

## 怎么跑教程

### 1. 装工具链

教程需要 Jupyter 与 Jupytext。用 Python 3.11 环境装全：

```bash
python -m pip install -r requirements/examples.txt
```

（`requirements/examples.txt` 是给教程用的；只想跑 Jittor 本体的话不需要它。）

### 2. 生成 notebook

仓库里**只有 `.md` 源**，`.ipynb` 是生成物，写到仓库之外，避免在工作区里留产物：

```bash
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/notebooks"
mkdir -p "$STATE"

# 生成一篇
python -m jupytext --to ipynb \
  --output "$STATE/getting_started.ipynb" examples/notebooks/getting_started.md

# 生成全部
cp -R examples/notebooks/. "$STATE/"
find "$STATE" -type f -name '*.md' ! -name README.md -print0 \
  | xargs -0 -n1 python -m jupytext --to ipynb
```

### 3. 打开 Jupyter

```bash
python -m notebook --ServerApp.root_dir="$STATE"
```

第一次执行代码单元会慢一些，因为 Jittor 在即时编译算子；第二次就快了。

### 4. 离线校验（维护者）

```bash
python -m nox -s tutorials
```

它会校验源文件的静态契约（学习路径是否列全、标签是否规范），并用 nbclient 离线执行
冒烟教程。

## 跑 Web 应用示例

`gan/` 下的示例是独立脚本，按各自 README 的说明运行，例如：

```bash
python examples/gan/simple_cgan.py --no-server \
  --output "$JITTOR_LAB_ROOT/_state/examples/cgan.png"
```

`simple_cgan.py` 在启动时训练一个小模型（CPU 上约一分钟），不下载任何权重，训练完再把
生成器接到命令行或 Web 界面上。

它们同样要求「导入不产生副作用」：训练、下载数据、开端口都发生在脚本执行阶段，
而不是 import 阶段。

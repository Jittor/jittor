# Jittor Examples

Examples are source-distribution assets and are intentionally excluded from the
runtime wheel. They are not a Python package and importing an example must not
download data, initialize a device, open a port, or create a Jittor cache.

## What is here

| 目录 | 内容 |
| --- | --- |
| [`notebooks/`](notebooks/README.md) | 可执行教程。MyST Markdown 是唯一权威来源，notebook 由 Jupytext 生成 |
| [`gan/`](gan/README.md) | 可直接运行的 Web 应用示例 |

**教程的完整阅读顺序在 [`notebooks/README.md`](notebooks/README.md)**，那里有两条主线：

* **原生 Jittor**：从 [`notebooks/getting_started.md`](notebooks/getting_started.md) 起步，
  一路到训练、性能与生成模型。
* **PyTorch 兼容**：从 [`notebooks/torch_compat.md`](notebooks/torch_compat.md) 起步，
  让已有的 `torch` 代码不改一行跑在 Jittor 上。

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
python examples/gan/simple_cgan.py
```

它们同样要求「导入不产生副作用」：需要下载数据或开端口的地方都写在脚本执行阶段，
而不是 import 阶段。

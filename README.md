# Jittor：基于即时编译的深度学习框架 / 计图

![Jittor Logo](https://cg.cs.tsinghua.edu.cn/jittor/favicon_package_v0/JittorLogo_Final1220.svg)

[快速开始](#quickstart) | [安装](#install) |
[教程](#tutorials) | [贡献](#contributing)

计图（Jittor）是一个基于即时编译和元算子的高性能深度学习框架。前端使用 Python
动态图接口，后端通过 C++ 和 CUDA 为实际工作负载编译、调优算子。

- [官网](https://cg.cs.tsinghua.edu.cn/jittor/)
- [教程](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
- [模型库](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
- [API 文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
- [论坛](https://discuss.jittor.org/)
- [Awesome Jittor（中文）](docs/community/AWESOME-JITTOR-LIST.md) |
  [Awesome Jittor（English）](docs/community/AWESOME-JITTOR-LIST.en.md)

<a id="quickstart"></a>
## 快速开始

下面的示例定义并训练一个两层网络，同时展示当前版本推荐的张量构造方式。

```python
import numpy as np

import jittor as jt
from jittor import Module, nn


class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.Relu()
        self.layer2 = nn.Linear(10, 1)

    def execute(self, x):
        return self.layer2(self.relu(self.layer1(x)))


def get_data(steps, batch_size):
    for _ in range(steps):
        x = np.random.rand(batch_size, 1).astype("float32")
        yield jt.array(x), jt.array(x * x)


model = Model()
optimizer = nn.SGD(model.parameters(), lr=0.1)

for step, (x, y) in enumerate(get_data(1000, 50)):
    prediction = model(x)
    loss = ((prediction - y) ** 2).mean()
    optimizer.step(loss)
    print("step {}, loss = {}".format(step, loss.item()))
```

Jittor 算子通常同时提供函数和 `Var` 方法；读取 `.data`、`.numpy()` 或 `.item()`
会同步尚未完成的计算。

```python
import jittor as jt

a = jt.array([1, 2, 3], dtype="float32")
b = jt.array([4, 5, 6], dtype="float32")
c = a * b
print(c.numpy())
print(c.max(), jt.max(c))
```

<a id="install"></a>
## 安装

安装要求以 [`pyproject.toml`](pyproject.toml) 为准。Jittor 支持 Python 3.7
及以上版本；本地即时编译需要 C++ 编译器和 OpenMP，使用加速设备时还需要对应驱动与工具链。

| 平台 | 最低开发要求 |
| --- | --- |
| Linux | Python >= 3.7, g++ >= 5.4, OpenMP |
| Windows 10/11 | Python >= 3.8；NVIDIA GPU 需要 CUDA >= 10.2 |
| macOS >= 10.14 | Python >= 3.7, clang >= 8.0, `libomp` |

### 从 PyPI 安装

```bash
python -m pip install jittor
python -m jittor.selftest
```

### 使用 uv 配置开发环境

仓库包含 `uv.lock`，以及面向当前维护的 Python 3.11 测试环境的 `dev` 依赖组。
`uv sync` 会以可编辑方式安装当前源码；虚拟环境应放在仓库外的 lab 状态目录中。

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
export UV_PROJECT_ENVIRONMENT="$JITTOR_LAB_ROOT/_state/uv/venv"
uv python install 3.11
uv venv --python 3.11
uv sync --locked --group dev
uv run --locked python -m jittor.selftest
```

使用仓库门禁使用的同一个 fail-closed 独立 runner 分别运行 CPU 和 CUDA 测试。
CUDA 命令要求已安装 CUDA 工具链并能找到 `nvcc`，只有 NVIDIA 驱动还不够。

```bash
uv run --locked python tools/run_test_suite.py --tier core --backend cpu

export nvcc_path="$(command -v nvcc)"
CUDA_VISIBLE_DEVICES=0 uv run --locked \
  python tools/run_test_suite.py --tier core --backend cuda
```

对于采用 conda `targets/x86_64-linux` 布局的工具链，请使用一个包含
`bin/nvcc`、`bin/g++`、`include/`、`lib64/` 和 `nvvm/` 的兼容 CUDA 根目录：

```bash
export CUDA_COMPAT_ROOT=/path/to/cuda-compat
export CUDA_VISIBLE_DEVICES=2
export nvcc_path="$CUDA_COMPAT_ROOT/bin/nvcc"
export cc_path="$CUDA_COMPAT_ROOT/bin/g++"
export CUDA_HOME="$CUDA_COMPAT_ROOT"
export PATH="$CUDA_COMPAT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_COMPAT_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run --locked python tools/run_test_suite.py --tier core --backend cuda
```

短 tier 通过后，去掉 `--tier core` 即可运行完整测试树：

```bash
uv run --locked python tools/run_test_suite.py --backend cpu
CUDA_VISIBLE_DEVICES=0 uv run --locked \
  python tools/run_test_suite.py --backend cuda
```

Debian 或 Ubuntu 需要先安装编译依赖：

```bash
sudo apt install python3-dev libomp-dev g++ build-essential
```

macOS 请先通过 Homebrew 安装 OpenMP：

```bash
brew install libomp
python -m pip install jittor
python -m jittor.selftest
```

Windows 的 conda 环境可能还需要 `pywin32`：

```bash
conda install pywin32
python -m pip install jittor
python -m jittor.selftest
```

### 首次运行会发生什么

Jittor 是即时编译的，所以 `import jittor` 不只是导入一个库：第一次会编译 C++
内核，之后每个算子在被用到时再编译。具体代价如下。

- **耗时。** 首次 import 要编译内核，通常几分钟，核心数少的机器更久；
  之后的 import 走缓存，约一两秒。
- **联网。** 首次会下载少量第三方归档（MKL/oneDNN、cub、cutt，
  几十 MB）。无网络的机器见下面的「离线安装」。
- **磁盘。** 缓存放内核与已编译的算子，量级 1–2 GB，默认在
  `~/.cache/jittor`，可用 `JITTOR_HOME` 改位置。
- **不会自动下载 CUDA。** Jittor **不会**自己下载
  CUDA 工具链。有驱动但没有 `nvcc` 时它会明确告知并按 CPU 构建；请自行安装工具链，
  或设 `nvcc_path=""` 明确选择 CPU。
- **缓存目录取决于工具链。** 目录名包含 Jittor / 编译器 /
  Python 版本、平台、CPU 与构建配置（`cc_flags`、`nvcc_flags`、`cuda_archs`、
  `enable_lto`、`nvcc_path`）；其中任何一项变化都会写进新目录而不是覆盖旧的，
  不同的源码目录也各有各的目录。两个不能共用缓存的运行要设不同的 `cache_name`
  （或不同的 `JITTOR_HOME`）。

报构建问题之前先跑一次前置条件检查。它一次列出所有缺失项，并逐条说明是 Jittor
能自己解决还是需要你动手：

```bash
python -m jittor_utils.preflight
```

要回收磁盘空间，可以按组或整体清理缓存：

```bash
python -m jittor_utils.clean_cache help    # list the groups / 列出可清理的组
python -m jittor_utils.clean_cache core    # compiled products only / 只清编译产物
python -m jittor_utils.clean_cache all     # everything / 全部
```

### 离线安装

无网络的机器上，先在别处把第三方归档准备好再指过去。在一台有网络、且已检出本仓库的
机器上：

```bash
nox -s prefetch          # fills a mirror directory / 填充镜像目录
```

把它填充的目录拷到离线机器，然后设置：

```bash
export JITTOR_OFFLINE_PATH=/path/to/that/directory
```

Jittor 会从那里拷贝而不是下载。`python -m jittor_utils.preflight` 会告诉你还缺什么。

### CUDA 12 组件包

Linux x86_64 可使用 `cuda12` extra 安装固定版本的 CUDA 12.2 运行时，cuDNN 取
8.9.7 及以上（含 cuDNN 9，因此可与钉了自己那份 cuDNN 9 的现代 torch 共存）；
JIT 编译仍需要 `nvcc`，且 Jittor 不会替你下载：请安装 CUDA 工具链并把
`nvcc` 放进 PATH，或用 `nvcc_path` 指过去。

```bash
python -m pip install "jittor[cuda12]"
use_cuda=1 python -m jittor.selftest
```

仅安装运行时依赖时，对应的 uv 命令为：

```bash
uv sync --locked --no-default-groups --extra cuda12
```

设置 `JITTOR_CUDA_WHEEL_STRICT=1` 可拒绝不完整或不匹配的组件栈；设置
`JITTOR_CUDA_WHEEL_DISABLE=1` 则只使用系统或 JTCUDA 提供的库。

### 从源码安装

```bash
git clone https://github.com/Jittor/jittor.git
cd jittor
python -m pip install -e .
python -m jittor.selftest
```

C++ 与 CUDA 源码与 Python 包并列，不在包内：内核是 [`src/`](src/)，每个后端是
`backends/<名字>/`（例如 [`backends/cuda/`](backends/cuda/)）。源码 checkout 里
Jittor 在 `python/jittor/` 的同级目录查找它们——用 `src/core/common.h` 认出内核，
用带 `__init__.py` 的目录认出后端；wheel 把同样的目录装成 `jittor/src/` 与
`jittor/backends/<名字>/`。改动其中任何源码都会在下次使用时重新编译。显式选择 CUDA：

```bash
export JT_BUILD_NVCC_PATH=/usr/local/cuda/bin/nvcc
JT_USE_CUDA=1 python -m jittor.selftest
```

`JT_BUILD_*` 决定编译产物，`JT_*` 决定运行行为；`python -m jittor_utils.env_manifest`
列出全部变量。旧的无前缀名（`nvcc_path`、`use_cuda` 等）仍然生效，启动时会提示一次。

在 Python 中，导入 Jittor 后可通过 `jt.flags.use_cuda = 1` 启用 CUDA。

### Docker 容器

维护的 CPU 与 CUDA 镜像共用同一份 [`Dockerfile`](Dockerfile)：

```bash
docker build -t jittor/jittor .
docker build -t jittor/jittor-cuda \
  --build-arg FROM_IMAGE=nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 .

docker run -it --network host jittor/jittor
docker run -it --network host --gpus all jittor/jittor-cuda
```

<a id="tutorials"></a>
## 教程

维护的教程以 MyST Markdown 作为版本化源文件；执行和验证时由 Jupytext 临时生成
`.ipynb`，生成的 notebook 不进入仓库。

按顺序编排的完整课程（读什么、什么顺序、每篇需要什么前置）维护在唯一一处：
**[examples/notebooks/README.md](examples/notebooks/README.md)**。

零基础从 `getting_started` 开始，它从安装一路讲到训练，全程 CPU 可跑完。之后按背景选
一条主线：继续用原生 Jittor，或走 PyTorch 兼容线（`torch_compat`，再读
`torch_compat_migration`），让已有的 `torch` 代码不改一行跑在 Jittor 上。

```bash
python -m pip install -r requirements/examples.txt
python -m nox -s tutorials
```

<a id="contributing"></a>
## 贡献

欢迎提交缺陷报告、测试、文档、算子、性能改进和模型。发起合并请求前，请阅读
[贡献指南](CONTRIBUTING.md)、[行为准则](CODE_OF_CONDUCT.md)和
[治理文档](GOVERNANCE.md)。

## 引用

```bibtex
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Science China Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```

## 许可证

Jittor 使用 Apache License 2.0，详见 [`LICENSE.txt`](LICENSE.txt)。

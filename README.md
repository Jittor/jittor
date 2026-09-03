# Jittor: a Just-in-time Deep Learning Framework / 计图

![Jittor Logo](https://cg.cs.tsinghua.edu.cn/jittor/favicon_package_v0/JittorLogo_Final1220.svg)

[Quickstart / 快速开始](#quickstart) | [Install / 安装](#install) |
[Tutorials / 教程](#tutorials) | [Contributing / 贡献](#contributing)

Jittor is a high-performance deep learning framework based on just-in-time
compilation and meta-operators. Its Python frontend uses dynamic graph execution,
while its C++ and CUDA backend compiles and tunes operators for each workload.

计图（Jittor）是一个基于即时编译和元算子的高性能深度学习框架。前端使用 Python
动态图接口，后端通过 C++ 和 CUDA 为实际工作负载编译、调优算子。

- [Website / 官网](https://cg.cs.tsinghua.edu.cn/jittor/)
- [Tutorials / 教程](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
- [Models / 模型库](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
- [API documentation / API 文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
- [Forum / 论坛](https://discuss.jittor.org/)
- [Awesome Jittor (English)](AWESOME-JITTOR-LIST.md) |
  [Awesome Jittor (中文)](AWESOME-JITTOR-LIST.cn.md)

<a id="quickstart"></a>
## Quickstart / 快速开始

The following example defines and trains a small two-layer network. It also
shows the canonical tensor-construction API used by current Jittor releases.

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

Jittor operations are available as both functions and `Var` methods. Accessing
`.data`, `.numpy()`, or `.item()` synchronizes pending computation.

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
## Install / 安装

The package metadata in [`pyproject.toml`](pyproject.toml) is authoritative.
Jittor supports Python 3.7 or newer. A C++ compiler and OpenMP runtime are
required for local JIT compilation; accelerator support additionally requires
the corresponding driver and toolchain.

安装要求以 [`pyproject.toml`](pyproject.toml) 为准。Jittor 支持 Python 3.7
及以上版本；本地即时编译需要 C++ 编译器和 OpenMP，使用加速设备时还需要对应驱动与工具链。

| Platform / 平台 | Minimum development requirements / 最低开发要求 |
| --- | --- |
| Linux | Python >= 3.7, g++ >= 5.4, OpenMP |
| Windows 10/11 | Python >= 3.8; CUDA >= 10.2 for NVIDIA GPU |
| macOS >= 10.14 | Python >= 3.7, clang >= 8.0, `libomp` |

### Install from PyPI / 从 PyPI 安装

```bash
python -m pip install jittor
python -m jittor.selftest
```

On Debian or Ubuntu, install the compiler dependencies first:

Debian 或 Ubuntu 需要先安装编译依赖：

```bash
sudo apt install python3-dev libomp-dev g++ build-essential
```

On macOS, install OpenMP with Homebrew before installing Jittor:

macOS 请先通过 Homebrew 安装 OpenMP：

```bash
brew install libomp
python -m pip install jittor
python -m jittor.selftest
```

On Windows, a conda environment may additionally need `pywin32`:

Windows 的 conda 环境可能还需要 `pywin32`：

```bash
conda install pywin32
python -m pip install jittor
python -m jittor.selftest
```

### What the first run does / 首次运行会发生什么

Jittor compiles at runtime, so `import jittor` is not only a library import: the
first one builds the C++ core and then compiles each operator as your program
reaches it. This is what to expect.

Jittor 是即时编译的，所以 `import jittor` 不只是导入一个库：第一次会编译 C++
内核，之后每个算子在被用到时再编译。具体代价如下。

- **Time / 耗时.** The first import compiles the core: minutes on a typical
  machine, and longer on few-core ones. Later imports reuse the cache and take
  a couple of seconds. 首次 import 要编译内核，通常几分钟，核心数少的机器更久；
  之后的 import 走缓存，约一两秒。
- **Network / 联网.** Jittor downloads a small set of third-party archives once
  (MKL/oneDNN, cub, cutt — tens of MB). See *Offline install* below if the
  machine has no network. 首次会下载少量第三方归档（MKL/oneDNN、cub、cutt，
  几十 MB）。无网络的机器见下面的「离线安装」。
- **Disk / 磁盘.** The cache holds the compiled core and every operator built so
  far, on the order of 1–2 GB. It lives under `~/.cache/jittor`; set
  `JITTOR_HOME` to move it. 缓存放内核与已编译的算子，量级 1–2 GB，默认在
  `~/.cache/jittor`，可用 `JITTOR_HOME` 改位置。
- **No automatic CUDA toolkit / 不会自动下载 CUDA.** Jittor does **not**
  download a CUDA toolkit by itself. If the machine has an NVIDIA driver but no
  `nvcc`, Jittor says so and builds for CPU; install a toolkit, or set
  `nvcc_path=""` to make the CPU-only build explicit. Jittor **不会**自己下载
  CUDA 工具链。有驱动但没有 `nvcc` 时它会明确告知并按 CPU 构建；请自行安装工具链，
  或设 `nvcc_path=""` 明确选择 CPU。
- **The cache directory depends on your toolchain / 缓存目录取决于工具链.** Its
  name includes the Jittor, compiler and Python versions, the platform, the CPU,
  and the build configuration (`cc_flags`, `nvcc_flags`, `cuda_archs`,
  `enable_lto`, `nvcc_path`). Changing any of them builds into a new directory
  rather than overwriting the old one, and different checkouts get different
  directories. Two runs that must not share a cache need different
  `cache_name` values (or different `JITTOR_HOME`s). 目录名包含 Jittor / 编译器 /
  Python 版本、平台、CPU 与构建配置；其中任何一项变化都会写进新目录而不是覆盖旧的，
  不同的源码目录也各有各的目录。两个不能共用缓存的运行要设不同的 `cache_name`
  （或不同的 `JITTOR_HOME`）。

Before reporting a build problem, run the preconditions check. It reports
everything that is missing at once, and says for each item whether Jittor can
resolve it or you have to:

报构建问题之前先跑一次前置条件检查。它一次列出所有缺失项，并逐条说明是 Jittor
能自己解决还是需要你动手：

```bash
python -m jittor_utils.preflight
```

To reclaim disk space, remove part or all of the cache:

要回收磁盘空间，可以按组或整体清理缓存：

```bash
python -m jittor_utils.clean_cache help    # list the groups / 列出可清理的组
python -m jittor_utils.clean_cache core    # compiled products only / 只清编译产物
python -m jittor_utils.clean_cache all     # everything / 全部
```

### Offline install / 离线安装

On a machine without a network, fetch the third-party archives elsewhere and
point Jittor at them. On a connected machine with this repository checked out:

无网络的机器上，先在别处把第三方归档准备好再指过去。在一台有网络、且已检出本仓库的
机器上：

```bash
nox -s prefetch          # fills a mirror directory / 填充镜像目录
```

Copy the directory it fills to the offline machine and set:

把它填充的目录拷到离线机器，然后设置：

```bash
export JITTOR_OFFLINE_PATH=/path/to/that/directory
```

Jittor copies from there instead of downloading. `python -m
jittor_utils.preflight` reports whether anything is still missing.

Jittor 会从那里拷贝而不是下载。`python -m jittor_utils.preflight` 会告诉你还缺什么。

### CUDA 12 component wheels / CUDA 12 组件包

On Linux x86_64, the `cuda12` extra installs a pinned CUDA 12.2 runtime
stack with cuDNN 8.9.7 or newer (cuDNN 9 included, so the extra can coexist
with a modern torch, which pins its own cuDNN 9). Jittor still needs an `nvcc`
compiler, and does not download one for you: install a CUDA toolkit and put
`nvcc` on PATH, or set `nvcc_path` to it.

Linux x86_64 可使用 `cuda12` extra 安装固定版本的 CUDA 12.2 运行时，cuDNN 取
8.9.7 及以上（含 cuDNN 9，因此可与钉了自己那份 cuDNN 9 的现代 torch 共存）；
JIT 编译仍需要 `nvcc`，且 Jittor 不会替你下载：请安装 CUDA 工具链并把
`nvcc` 放进 PATH，或用 `nvcc_path` 指过去。

```bash
python -m pip install "jittor[cuda12]"
use_cuda=1 python -m jittor.selftest
```

Set `JITTOR_CUDA_WHEEL_STRICT=1` to reject an incomplete or mismatched component
stack. Set `JITTOR_CUDA_WHEEL_DISABLE=1` to use only the system/JTCUDA libraries.

### Install from source / 从源码安装

```bash
git clone https://github.com/Jittor/jittor.git
cd jittor
python -m pip install -e .
python -m jittor.selftest
```

C++ and CUDA source changes under `python/jittor/src/` are rebuilt on their next
use. To select CUDA explicitly:

`python/jittor/src/` 下的 C++、CUDA 源码会在下次使用时自动重编译。显式选择 CUDA：

```bash
export nvcc_path=/usr/local/cuda/bin/nvcc
use_cuda=1 python -m jittor.selftest
```

Inside Python, enable CUDA with `jt.flags.use_cuda = 1` after importing Jittor.

### Docker / 容器

The maintained CPU and CUDA images share one [`Dockerfile`](Dockerfile):

维护的 CPU 与 CUDA 镜像共用同一份 [`Dockerfile`](Dockerfile)：

```bash
docker build -t jittor/jittor .
docker build -t jittor/jittor-cuda \
  --build-arg FROM_IMAGE=nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 .

docker run -it --network host jittor/jittor
docker run -it --network host --gpus all jittor/jittor-cuda
```

<a id="tutorials"></a>
## Tutorials / 教程

The maintained notebook sources are versioned as MyST Markdown. Jupytext creates
temporary `.ipynb` files for execution and validation; generated notebooks are
not repository sources.

维护的教程以 MyST Markdown 作为版本化源文件；执行和验证时由 Jupytext 临时生成
`.ipynb`，生成的 notebook 不进入仓库。

- [Model definition and training / 模型定义与训练](examples/notebooks/example.md)
- [Ops and Vars / 算子与 Var](examples/notebooks/basics.md)
- [Meta-operators / 元算子](examples/notebooks/meta_op.md)
- [Custom C++ and CUDA operators / 自定义 C++ 与 CUDA 算子](examples/notebooks/custom_op.md)
- [Profiler / 性能分析](examples/notebooks/profiler.md)
- [Residual network training / 残差网络训练](examples/notebooks/resnet_training.md)
- [60-minute Chinese introduction / 60 分钟中文入门](examples/notebooks/60分钟快速入门Jittor/README.md)

```bash
python -m pip install -r requirements/examples.txt
python -m notebook --ServerApp.root_dir="$PWD/examples/notebooks"
```

<a id="contributing"></a>
## Contributing / 贡献

Bug reports, tests, documentation, operators, performance improvements, and
model contributions are welcome. Read the [contributing guide](CONTRIBUTING.md),
[code of conduct](CODE_OF_CONDUCT.md), and [governance document](GOVERNANCE.md)
before opening a pull request.

欢迎提交缺陷报告、测试、文档、算子、性能改进和模型。发起合并请求前，请阅读
[贡献指南](CONTRIBUTING.md)、[行为准则](CODE_OF_CONDUCT.md)和
[治理文档](GOVERNANCE.md)。

## Citation / 引用

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

## License / 许可证

Jittor is licensed under Apache License 2.0. See [`LICENSE.txt`](LICENSE.txt).

Jittor 使用 Apache License 2.0，详见 [`LICENSE.txt`](LICENSE.txt)。

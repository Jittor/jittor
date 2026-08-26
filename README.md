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

### CUDA 12 component wheels / CUDA 12 组件包

On Linux x86_64, the `cuda12` extra installs a pinned CUDA 12.2 and cuDNN 8
runtime stack. Jittor still needs an `nvcc` compiler, supplied by the system or
its automatic JTCUDA fallback.

Linux x86_64 可使用 `cuda12` extra 安装固定版本的 CUDA 12.2 与 cuDNN 8
运行时；JIT 编译仍需要系统 `nvcc`，或使用 Jittor 的自动 JTCUDA 回退。

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

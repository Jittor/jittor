# Jittor: a Just-in-time Deep Learning Framework

[中文](README.md) | English

![Jittor Logo](https://cg.cs.tsinghua.edu.cn/jittor/favicon_package_v0/JittorLogo_Final1220.svg)

[Quickstart](#quickstart) | [Install](#install) |
[Tutorials](#tutorials) | [Contributing](#contributing)

Jittor is a high-performance deep learning framework based on just-in-time
compilation and meta-operators. Its Python frontend uses dynamic graph execution,
while its C++ and CUDA backend compiles and tunes operators for each workload.

- [Website](https://cg.cs.tsinghua.edu.cn/jittor/)
- [Tutorials](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
- [Models](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
- [API documentation](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
- [Forum](https://discuss.jittor.org/)
- [Awesome Jittor (English)](docs/community/AWESOME-JITTOR-LIST.en.md) |
  [Awesome Jittor (中文)](docs/community/AWESOME-JITTOR-LIST.md)

<a id="quickstart"></a>
## Quickstart

The following example defines and trains a small two-layer network. It also
shows the canonical tensor-construction API used by current Jittor releases.

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

```python
import jittor as jt

a = jt.array([1, 2, 3], dtype="float32")
b = jt.array([4, 5, 6], dtype="float32")
c = a * b
print(c.numpy())
print(c.max(), jt.max(c))
```

<a id="install"></a>
## Install

The package metadata in [`pyproject.toml`](pyproject.toml) is authoritative.
Jittor supports Python 3.7 or newer. A C++ compiler and OpenMP runtime are
required for local JIT compilation; accelerator support additionally requires
the corresponding driver and toolchain.

| Platform | Minimum development requirements |
| --- | --- |
| Linux | Python >= 3.7, g++ >= 5.4, OpenMP |
| Windows 10/11 | Python >= 3.8; CUDA >= 10.2 for NVIDIA GPU |
| macOS >= 10.14 | Python >= 3.7, clang >= 8.0, `libomp` |

### Install from PyPI

```bash
python -m pip install jittor
python -m jittor.selftest
```

### uv development environment

The repository includes a `uv.lock` and a pinned `dev` dependency group for
the maintained Python 3.11 test environment. `uv sync` installs this checkout
in editable mode. Keep the virtual environment in the repository's external
lab state directory.

```bash
export JITTOR_LAB_ROOT="${JITTOR_LAB_ROOT:-$(cd .. && pwd)/jittor-lab}"
export UV_PROJECT_ENVIRONMENT="$JITTOR_LAB_ROOT/_state/uv/venv"
uv python install 3.11
uv venv --python 3.11
uv sync --locked --group dev
uv run --locked python -m jittor.selftest
```

Run the CPU and CUDA suites through the same fail-closed standalone runner used
by the repository gates. The CUDA command requires a working CUDA toolkit with
`nvcc` on `PATH`; an NVIDIA driver alone is not enough.

```bash
uv run --locked python tools/run_test_suite.py --tier core --backend cpu

export nvcc_path="$(command -v nvcc)"
CUDA_VISIBLE_DEVICES=0 uv run --locked \
  python tools/run_test_suite.py --tier core --backend cuda
```

For a conda `targets/x86_64-linux` toolkit, point Jittor at a compatibility
root that exposes `bin/nvcc`, `bin/g++`, `include/`, `lib64/`, and `nvvm/`:

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

For the complete trees, omit `--tier core` after the short tier is green:

```bash
uv run --locked python tools/run_test_suite.py --backend cpu
CUDA_VISIBLE_DEVICES=0 uv run --locked \
  python tools/run_test_suite.py --backend cuda
```

On Debian or Ubuntu, install the compiler dependencies first:

```bash
sudo apt install python3-dev libomp-dev g++ build-essential
```

On macOS, install OpenMP with Homebrew before installing Jittor:

```bash
brew install libomp
python -m pip install jittor
python -m jittor.selftest
```

On Windows, a conda environment may additionally need `pywin32`:

```bash
conda install pywin32
python -m pip install jittor
python -m jittor.selftest
```

### What the first run does

Jittor compiles at runtime, so `import jittor` is not only a library import: the
first one builds the C++ core and then compiles each operator as your program
reaches it. This is what to expect.

- **Time.** The first import compiles the core: minutes on a typical
  machine, and longer on few-core ones. Later imports reuse the cache and take
  a couple of seconds.
- **Network.** Jittor downloads a small set of third-party archives once
  (MKL/oneDNN, cub, cutt — tens of MB). See *Offline install* below if the
  machine has no network.
- **Disk.** The cache holds the compiled core and every operator built so
  far, on the order of 1–2 GB. It lives under `~/.cache/jittor`; set
  `JITTOR_HOME` to move it.
- **No automatic CUDA toolkit.** Jittor does **not**
  download a CUDA toolkit by itself. If the machine has an NVIDIA driver but no
  `nvcc`, Jittor says so and builds for CPU; install a toolkit, or set
  `nvcc_path=""` to make the CPU-only build explicit.
- **The cache directory depends on your toolchain.** Its
  name includes the Jittor, compiler and Python versions, the platform, the CPU,
  and the build configuration (`cc_flags`, `nvcc_flags`, `cuda_archs`,
  `enable_lto`, `nvcc_path`). Changing any of them builds into a new directory
  rather than overwriting the old one, and different checkouts get different
  directories. Two runs that must not share a cache need different
  `cache_name` values (or different `JITTOR_HOME`s).

Before reporting a build problem, run the preconditions check. It reports
everything that is missing at once, and says for each item whether Jittor can
resolve it or you have to:

```bash
python -m jittor_utils.preflight
```

To reclaim disk space, remove part or all of the cache:

```bash
python -m jittor_utils.clean_cache help    # list the groups / 列出可清理的组
python -m jittor_utils.clean_cache core    # compiled products only / 只清编译产物
python -m jittor_utils.clean_cache all     # everything / 全部
```

### Offline install

On a machine without a network, fetch the third-party archives elsewhere and
point Jittor at them. On a connected machine with this repository checked out:

```bash
nox -s prefetch          # fills a mirror directory / 填充镜像目录
```

Copy the directory it fills to the offline machine and set:

```bash
export JITTOR_OFFLINE_PATH=/path/to/that/directory
```

Jittor copies from there instead of downloading. `python -m
jittor_utils.preflight` reports whether anything is still missing.

### CUDA 12 component wheels

On Linux x86_64, the `cuda12` extra installs a pinned CUDA 12.2 runtime
stack with cuDNN 8.9.7 or newer (cuDNN 9 included, so the extra can coexist
with a modern torch, which pins its own cuDNN 9). Jittor still needs an `nvcc`
compiler, and does not download one for you: install a CUDA toolkit and put
`nvcc` on PATH, or set `nvcc_path` to it.

```bash
python -m pip install "jittor[cuda12]"
use_cuda=1 python -m jittor.selftest
```

The equivalent uv command for a runtime-only environment is:

```bash
uv sync --locked --no-default-groups --extra cuda12
```

Set `JITTOR_CUDA_WHEEL_STRICT=1` to reject an incomplete or mismatched component
stack. Set `JITTOR_CUDA_WHEEL_DISABLE=1` to use only the system/JTCUDA libraries.

### Install from source

```bash
git clone https://github.com/Jittor/jittor.git
cd jittor
python -m pip install -e .
python -m jittor.selftest
```

The C++ and CUDA sources live beside the Python package, not inside it: the
core is [`src/`](src/) and each backend is `backends/<name>/` (for example
[`backends/cuda/`](backends/cuda/)). In a checkout Jittor finds them by looking
next to `python/jittor/` -- `src/core/common.h` identifies the core, and a
backend is the directory with its `__init__.py`; a wheel ships the same trees as
`jittor/src/` and `jittor/backends/<name>/`. Editing any of those sources
rebuilds it on its next use. To select CUDA explicitly:

```bash
export JT_BUILD_NVCC_PATH=/usr/local/cuda/bin/nvcc
JT_USE_CUDA=1 python -m jittor.selftest
```

`JT_BUILD_*` variables decide what gets compiled, `JT_*` variables decide what
the compiled core does; `python -m jittor_utils.env_manifest` lists every one of
them. The older unprefixed names (`nvcc_path`, `use_cuda`, ...) still work and
report themselves once at startup.

Inside Python, enable CUDA with `jt.flags.use_cuda = 1` after importing Jittor.

### Docker

The maintained CPU and CUDA images share one [`Dockerfile`](Dockerfile):

```bash
docker build -t jittor/jittor .
docker build -t jittor/jittor-cuda \
  --build-arg FROM_IMAGE=nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 .

docker run -it --network host jittor/jittor
docker run -it --network host --gpus all jittor/jittor-cuda
```

<a id="tutorials"></a>
## Tutorials

The maintained notebook sources are versioned as MyST Markdown. Jupytext creates
temporary `.ipynb` files for execution and validation; generated notebooks are
not repository sources.

The ordered curriculum -- what to read, in what order, and what each tutorial
needs -- is maintained in one place:
**[examples/notebooks/README.md](examples/notebooks/README.md)**.

Beginners start with `getting_started`, which goes from installation to training on
CPU alone. From there, pick one of two tracks: keep writing native Jittor, or use
the PyTorch-compatible front end (`torch_compat`, then `torch_compat_migration`)
so existing `torch` code runs unchanged.

```bash
python -m pip install -r requirements/examples.txt
python -m nox -s tutorials
```

<a id="contributing"></a>
## Contributing

Bug reports, tests, documentation, operators, performance improvements, and
model contributions are welcome. Read the [contributing guide](CONTRIBUTING.en.md),
[code of conduct](CODE_OF_CONDUCT.en.md), and [governance document](GOVERNANCE.en.md)
before opening a pull request.

## Citation

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

## License

Jittor is licensed under Apache License 2.0. See [`LICENSE.txt`](LICENSE.txt).

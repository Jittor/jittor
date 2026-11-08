# Jittor

[中文](README.md) | English

![Jittor Logo](https://cg.cs.tsinghua.edu.cn/jittor/favicon_package_v0/JittorLogo_Final1220.svg)

**A deep learning framework built on just-in-time compilation and meta-operators.**

Jittor provides Python dynamic graphs, automatic differentiation, and neural network APIs,
compiling operators for the computation at hand. Use the native Jittor API to build models,
or use Torch-style interfaces through `jittor-compat`, with computation executed by Jittor.

[Quickstart](#quickstart) · [Install](#install) · [Torch compatibility and ecosystem](#ecosystem) ·
[Documentation and tutorials](#tutorials) · [Contributing](#contributing)

> This page describes the **`2.0-refactor` development branch**. The PyPI release may differ
> from this branch. Install from source below to use the implementation documented here.

## Project components

| Component | Purpose | Start here |
| --- | --- | --- |
| Jittor core | Tensors, automatic differentiation, training, JIT compilation, and device execution | [Quickstart](docs/getting-started/quickstart.en.md) |
| `jittor-compat` | Optional Torch-compatible frontend sharing the core runtime | [Compatibility guide](docs/compatibility/index.en.md) |
| JittorEcosystem | Per-library versions, adapters, upstream examples, and feature/device validation | [Ecosystem repositories](https://github.com/JittorEcosystem) |

Coverage differs across CPU, CUDA, Ascend NPU, and other backends. See
[platform support](docs/guides/hardware/platform-support.en.md) for validation evidence by
OS, Python version, device, and distributed capability, and review the
[known limitations](docs/guides/known-limitations.en.md).

<a id="install"></a>
## Install

Jittor compiles kernels locally and needs **Python, a C++ compiler, and OpenMP**.
Package requirements are declared in [`pyproject.toml`](pyproject.toml); repository maintenance
uses Python 3.11. Use a dedicated virtual environment. See the
[installation guide](docs/getting-started/installation.en.md) for Linux, macOS, and compat setup.

### Install this development branch

```bash
git clone --branch 2.0-refactor https://github.com/Jittor/jittor.git
cd jittor
python -m pip install --config-settings editable_mode=compat -e .
python -m jittor.selftest
```

For the published PyPI release instead:

```bash
python -m pip install jittor
python -m jittor.selftest
```

### Use CUDA

Install the NVIDIA driver and a complete CUDA toolkit suitable for your device. In an
environment with Jittor installed:

```bash
export JT_BUILD_NVCC_PATH="$(command -v nvcc)"
JT_USE_CUDA=1 python -m jittor.selftest
```

Set `jt.flags.use_cuda = 1` in model code. The toolkit must include the math libraries your
model uses; a driver or runtime components alone do not provide the JIT build environment.
Install the CUDA toolkit explicitly: importing Jittor **does not download one**.
`JT_BUILD_NVCC_PATH` selects the compiler; its legacy name `nvcc_path` is still recognized.
For Ascend, see the [910B guide](docs/guides/hardware/ascend-910b.en.md) and
[310P guide](docs/guides/hardware/ascend-310p.en.md).

### What the first run does

The first import and first use of operators trigger compilation. Time depends on the machine
and workload; subsequent runs reuse the cache. Depending on the enabled backend, the build
may download third-party dependencies. Reserve disk space as the cache grows with models and
build configurations.

- `JITTOR_HOME`: choose the cache root; isolate parallel validation with separate roots or `cache_name` values.
- `python -m jittor_utils.preflight`: check compilers, dependencies, and runtime prerequisites.
- `python -m jittor_utils.clean_cache help`: list cache cleanup options.

**Offline install:** run `python -m nox -s prefetch` in an online source development environment,
copy the prepared dependency directory to the target machine, and point `JITTOR_OFFLINE_PATH`
at it. Run `preflight` again to check for missing dependencies.

<a id="quickstart"></a>
## Quickstart

Train a linear model on CPU, including forward computation, backward computation, and parameter
updates. No dataset download is needed.

```python
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 0
jt.set_global_seed(0)

x = jt.array([[0.0], [1.0], [2.0], [3.0]])
y = 2 * x + 1
model = nn.Linear(1, 1)
optimizer = jt.optim.SGD(model.parameters(), lr=0.05)

for step in range(100):
    loss = ((model(x) - y) ** 2).mean()
    optimizer.step(loss)
    if step % 20 == 0:
        print(step, loss.item())

print(model(x).numpy())
```

Predictions approach `1, 3, 5, 7`. Native training uses `optimizer.step(loss)` for backward
computation and parameter updates; reading `.item()` or `.numpy()` synchronizes computation.
Continue with the [complete introductory tutorial](examples/notebooks/getting_started.md).

<a id="ecosystem"></a>
## Torch compatibility and ecosystem

`jittor-compat` is a separate distribution, with source under [`compat/`](compat/).
After installing the core from this branch in a dedicated environment, run:

```bash
python -m pip install -e ./compat
```

Follow the [Torch compatibility guide](docs/compatibility/torch.en.md) for a training example.
For existing applications that directly `import torch`, see
[installation and deployment](docs/compatibility/torch-shim.en.md).
Use separate environments for real PyTorch and Jittor's Torch entry point.

Downstream integrations such as vLLM, Transformers, and verl are maintained independently in
**[JittorEcosystem](https://github.com/JittorEcosystem)**. Each repository owns its upstream
versions, adapters, tests, tutorial links, and feature/device support scope.
Use the target repository's pinned versions and validation records: a successful import does
not establish support for every training, inference, or distributed path.

<a id="tutorials"></a>
## Documentation and Tutorials

| Task | Start here |
| --- | --- |
| Install and train a first model | [Getting started](docs/getting-started/index.en.md) |
| Learn native Jittor or Torch-compatible APIs | [Tutorial catalog](examples/notebooks/README.md) |
| Look up APIs, devices, precision, and execution | [API reference](docs/api/index.en.md) · [Core concepts](docs/notes/index.en.md) |
| Configure distributed training or debug errors | [User guides](docs/guides/index.en.md) |
| Investigate speed and memory | [Performance](docs/performance/index.en.md) · [Benchmark tools](benchmarks/README.md) |
| Find runnable examples and community models | [Examples](examples/README.md) · [Community projects](docs/project/community/projects.en.md) |

Full entry point: [branch documentation](docs/index.en.md). Tutorials are maintained as Markdown;
see the [examples guide](examples/README.md) to generate notebooks. Website resources and historical
tutorials are available on the [Jittor website](https://cg.cs.tsinghua.edu.cn/jittor/).

<a id="contributing"></a>
## Contributing

Contributions of reproductions, code, tests, documentation, and performance improvements are welcome.
Read the [contribution guide](CONTRIBUTING.en.md), then use the
[development documentation](docs/development/index.en.md) to find the relevant modules and checks.
Report issues with versions, device details, a minimal script, and the full error output on
[GitHub Issues](https://github.com/Jittor/jittor/issues). For usage discussions, visit the
[forum](https://discuss.jittor.org/).

| Directory | Contents |
| --- | --- |
| `python/`, `src/`, `backends/` | Python interfaces, C++ core, and device backends |
| `compat/` | Optional Torch-compatible distribution and its tests |
| `tests/` | Core tests and repository contracts |
| `docs/`, `examples/` | Documentation and examples |
| `tools/`, `benchmarks/` | Maintenance tools and performance tests |

See [repository layout](docs/development/repository-layout.en.md) for directory and packaging rules.
Participation follows the [code of conduct](CODE_OF_CONDUCT.en.md) and
[governance rules](docs/project/community/governance.en.md).

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

[Apache License 2.0](LICENSE.txt).

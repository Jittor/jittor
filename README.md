# 计图 Jittor

中文 | [English](README.en.md)

![Jittor Logo](https://cg.cs.tsinghua.edu.cn/jittor/favicon_package_v0/JittorLogo_Final1220.svg)

**基于即时编译与元算子的深度学习框架。**

Jittor 提供 Python 动态图、自动微分和神经网络接口，按实际计算需求编译算子。
你可以用原生 Jittor API 编写模型，也可以通过 `jittor-compat` 使用 Torch 风格的接口，
计算由 Jittor 运行时执行。

[快速开始](#quickstart) · [安装](#install) · [Torch 兼容与生态](#ecosystem) ·
[文档与教程](#tutorials) · [贡献](#contributing)

> 本页对应 **`2.0-refactor` 开发分支**。PyPI 发布版与本分支可能不同；
> 需要本分支功能时，请按下面的源码安装步骤操作。

## 项目组成

| 部分 | 用途 | 入口 |
| --- | --- | --- |
| Jittor 核心 | 张量、自动微分、模型训练、JIT 编译与设备执行 | [快速开始](docs/getting-started/quickstart.md) |
| `jittor-compat` | 可选的 Torch 兼容前端，与核心共享计算运行时 | [兼容指南](docs/compatibility/index.md) |
| JittorEcosystem | 各下游库的版本组合、适配、官方用例与功能/芯片验证 | [生态仓库](https://github.com/JittorEcosystem) |

CPU、CUDA、昇腾 NPU 与其它后端的覆盖范围不同。操作系统、Python、设备和分布式能力的
验证依据见[平台支持](docs/guides/hardware/platform-support.md)；使用前也请查看
[已知限制](docs/guides/known-limitations.md)。

<a id="install"></a>
## 安装

Jittor 在本机编译内核，需要 **Python、C++ 编译器与 OpenMP**。
包的版本要求以 [`pyproject.toml`](pyproject.toml) 为准，仓库维护环境使用 Python 3.11。
建议使用独立虚拟环境；Linux、macOS 与兼容包的准备步骤见
[安装指南](docs/getting-started/installation.md)。

### 安装当前开发分支

```bash
git clone --branch 2.0-refactor https://github.com/Jittor/jittor.git
cd jittor
python -m pip install --config-settings editable_mode=compat -e .
python -m jittor.selftest
```

只需要 PyPI 发布版时：

```bash
python -m pip install jittor
python -m jittor.selftest
```

### 使用 CUDA

先准备与设备匹配的 NVIDIA 驱动及完整 CUDA 工具链。在已安装 Jittor 的环境中：

```bash
export JT_BUILD_NVCC_PATH="$(command -v nvcc)"
JT_USE_CUDA=1 python -m jittor.selftest
```

运行模型时可设置 `jt.flags.use_cuda = 1`。CUDA 工具链需要包含模型所用的数学库；
仅有驱动或运行时组件不等于具备 JIT 编译条件。Jittor 在导入时**不会自动下载** CUDA。
`JT_BUILD_NVCC_PATH` 是编译器路径配置，旧名 `nvcc_path` 仍可识别。
昇腾配置见 [910B 指南](docs/guides/hardware/ascend-910b.md)与
[310P 指南](docs/guides/hardware/ascend-310p.md)。

### 首次运行与缓存

首次导入和首次使用算子时会编译，耗时取决于机器与工作负载；后续运行复用缓存。
按启用的后端，构建过程可能下载第三方依赖。缓存也会随模型和编译配置增长，
请为它预留磁盘空间。

- `JITTOR_HOME`：调整缓存根位置；并行验证可用不同目录或 `cache_name` 隔离。
- `python -m jittor_utils.preflight`：检查编译器、依赖与运行条件。
- `python -m jittor_utils.clean_cache help`：查看缓存清理选项。

**离线安装：** 在联网的源码开发环境运行 `python -m nox -s prefetch`，
将准备好的依赖目录复制到目标机器，并设置 `JITTOR_OFFLINE_PATH` 指向它。
再运行 `preflight` 确认依赖齐全。

<a id="quickstart"></a>
## 快速开始

下面用 CPU 训练一个线性模型，包含前向、反向和参数更新，不需要下载数据。

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

预测值会逐步接近 `1、3、5、7`。原生训练用 `optimizer.step(loss)` 完成反向和更新；
读取 `.item()` 或 `.numpy()` 会同步计算。继续阅读
[完整入门教程](examples/notebooks/getting_started.md)。

<a id="ecosystem"></a>
## Torch 兼容与生态

`jittor-compat` 是独立发行物，源码在本仓库的 [`compat/`](compat/)。
使用当前分支时，在安装好核心的专用环境中执行：

```bash
python -m pip install -e ./compat
```

随后按[Torch 兼容指南](docs/compatibility/torch.md)运行训练示例；让已有应用直接
`import torch` 的部署方式见[安装与部署](docs/compatibility/torch-shim.md)。
真实 PyTorch 与 Jittor 的 Torch 入口应使用不同环境。

vLLM、Transformers、verl 等下游集成在
**[JittorEcosystem](https://github.com/JittorEcosystem)** 独立维护。
每个仓库负责上游版本、适配代码、测试、教程入口，以及功能和芯片支持范围。
请以目标仓库的固定版本与验收记录为准：导入成功不代表训练、推理和分布式路径均可用。

<a id="tutorials"></a>
## 文档与教程

| 你要做什么 | 从这里开始 |
| --- | --- |
| 安装并训练第一个模型 | [入门指南](docs/getting-started/index.md) |
| 系统学习原生 Jittor 或 Torch 兼容接口 | [教程目录](examples/notebooks/README.md) |
| 查 API、设备、精度与执行机制 | [API 参考](docs/api/index.md) · [核心概念](docs/notes/index.md) |
| 配置分布式训练或排查错误 | [使用指南](docs/guides/index.md) |
| 分析性能和显存 | [性能优化](docs/performance/index.md) · [基准工具](benchmarks/README.md) |
| 查看可运行示例和社区模型 | [示例](examples/README.md) · [社区项目](docs/project/community/projects.md) |

完整入口：[当前分支文档](docs/index.md)。教程以 Markdown 维护，生成 notebook 的方法见
[示例说明](examples/README.md#怎么跑教程)。官网资源与历史教程见
[Jittor 官网](https://cg.cs.tsinghua.edu.cn/jittor/)。

<a id="contributing"></a>
## 贡献

欢迎提交缺陷复现、代码、测试、文档和性能改进。先阅读[贡献指南](CONTRIBUTING.md)，
再按[开发文档](docs/development/index.md)选择相关模块与验证方式。
问题报告请附版本、设备、最小脚本及完整错误信息，提交到
[GitHub Issues](https://github.com/Jittor/jittor/issues)；使用交流可到[论坛](https://discuss.jittor.org/)。

| 目录 | 内容 |
| --- | --- |
| `python/`、`src/`、`backends/` | Python 接口、C++ 核心与设备后端 |
| `compat/` | 可选 Torch 兼容发行物及其测试 |
| `tests/` | 核心与仓库契约测试 |
| `docs/`、`examples/` | 文档与示例 |
| `tools/`、`benchmarks/` | 维护工具与性能测试 |

目录和打包规则见[仓库布局](docs/development/repository-layout.md)。参与项目请遵守
[行为准则](CODE_OF_CONDUCT.md)与[治理规则](docs/project/community/governance.md)。

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

[Apache License 2.0](LICENSE.txt)。

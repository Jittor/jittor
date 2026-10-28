# 为 Jittor 做贡献

Jittor 欢迎缺陷修复、新功能、算子、测试、文档和性能优化。参与项目即表示同意遵守
[行为准则](CODE_OF_CONDUCT.md)与[治理规则](GOVERNANCE.md)。

## 开始之前

- 提交新 issue 前先搜索[已有问题](https://github.com/Jittor/jittor/issues)；行为变更应先说明预期契约。
- 一个合并请求只处理一组相关改动，不提交生成文件、运行缓存、模型权重或无关格式化。
- 修复前先复现，并记录最小命令、设备、输入、期望结果和实际结果。
- 仅能导入不代表支持加速设备；必须在所声明的每个后端执行目标操作。

## 开发环境

运行时兼容范围以 [`pyproject.toml`](pyproject.toml) 为准。包支持 Python 3.7
及以上版本；仓库工具使用固定的 Python 3.11 环境，并用真实 Python 3.7 解释器编译检查全树。

```bash
git clone https://github.com/<your-account>/jittor.git
cd jittor
python -m pip install -e . -e ./compat
python -m pip install -r requirements/dev-tools.txt
python -m jittor.selftest
```

Jittor 会按需编译 C++ 与 CUDA。每个工作树或并行任务应使用独立缓存：

```bash
export JITTOR_HOME="${JITTOR_HOME:-$HOME/.cache/jittor-dev}"
export cache_name="${cache_name:-local-dev}"
```

使用 CUDA 时，若 `nvcc` 不在 `PATH` 中，请设置 `nvcc_path`。各后端的 CI 要求见
[`agent/manuals/environment.md`](agent/manuals/environment.md)。

## 仓库结构

```text
.
├── src/                 # C++ core: core/ type/ mem/ codegen/ ops/ runtime/ bindings/
├── backends/            # one directory per backend: cpu/ cuda/ acl/ rocm/ corex/ comm/
├── python/jittor/       # the Python runtime package (pure Python)
├── python/jittor_utils/ # bootstrap for the build utilities in jittor/build/utils/
├── compat/              # the optional `jittor-torch` distribution: torch shim, FSDP2, triton
├── adapters/            # the optional `jittor-torch-adapters` distribution
├── tests/               # repository pytest suite; not shipped in wheels
├── examples/            # runnable examples and MyST notebook sources
├── benchmarks/          # ASV regressions, Torch comparisons, and ACL measurements
├── tools/               # build, docs, lint, release and maintenance commands
├── docs/                # the single user- and contributor-facing documentation tree
├── agent/               # operational manuals and skills (manuals/ and skills/ only)
├── pyproject.toml       # authoritative package, tool, and pytest configuration
└── noxfile.py           # reproducible local and CI sessions
```

C++ 与 CUDA 源码在 Python 包**旁边**，不在包内：`src/` 是内核，每个
`backends/<name>/` 是一个后端。源码 checkout 在 `python/jittor/` 同级解析它们；
wheel 把同样的目录装成 `jittor/src/` 与 `jittor/backends/<name>/`。这些路径在构建时
按路径加载，并有打包与编译器契约，`python/jittor/contrib/math_util/src/` 和
`compat/shim/cpp_extension/{include,src}/` 也是如此。移动它们之前请阅读
[仓库布局决策](docs/development/repository-layout.md)与
[源码架构](docs/development/source-architecture.md)。

## 修改代码

修改代码时遵循对应子系统已有风格。除非同一变更明确调整项目元数据，否则新增 Python
代码必须兼容 Python 3.7。

### Python

- 使用四个空格和描述性的名字。
- 周边模块有类型标注时，公开 API 也要保持文档和类型标注。
- 不要用宽泛的异常处理隐藏导入或安装失败。
- 每个修正的行为都要加一条定向回归测试。

### C++ 与 CUDA

- 遵循附近的命名、格式与生成代码约定。
- 修改 CUDA 路径时保持 CPU 行为不变，反之亦然。
- 把 dtype 推导、广播、空张量、非连续布局、梯度和设备派发视为算子契约的一部分。
- 共享缓存的首次扩展/JIT 编译必须串行执行。

## 测试

仓库测试统一位于根目录 [`tests/`](tests/)，由 pytest 收集；已有
`unittest.TestCase` 无需改写。测试配置和合法 marker 均在
[`pyproject.toml`](pyproject.toml) 中定义。

把测试放在对应领域，模块命名为 `test_*.py`。使用最窄的后端或生命周期 marker
（`structure`、`cpu`、`cuda`、`rocm`、`npu`、`mpi`、`slow`、`network` 或 `manual`）。
跳过与预期失败必须写明具体不支持的契约；不要把失败改成无条件 skip。

```bash
# Focused test
python -m pytest tests/nn/test_nn_capabilities.py -v

# One test case
python -m pytest \
  tests/nn/test_nn_capabilities.py::TestAttentionCapabilities::test_layout_lengths_and_cumulative_cache \
  -v

# Collection contract
python -m pytest --collect-only -q tests
```

### Nox 门禁

[`noxfile.py`](noxfile.py) 是统一命令入口；不带参数的 `python -m nox` 运行默认门禁：
`lint`、`format`、`typing`、`structure`、`cpu`、`packaging`、`py37`、`py312`、`py313`。

```bash
python -m nox
python -m nox -s structure
python -m nox -s smoke        # the pull-request CPU tier
python -m nox -s cpu
python -m nox -s cuda
python -m nox -s npu
python -m nox -s rocm
python -m nox -s mpi
python -m nox -s ecosystem    # nightly: downstream libraries against real PyTorch
python -m nox -s benchmark
python -m nox -s docs
python -m nox -s docs_links
python -m nox -s tutorials
```

两次编辑之间用 [`tools/run_test_suite.py`](tools/run_test_suite.py) 比跑门禁便宜。
Torch 模式是进程级的，所以它把原生测试与 Torch 兼容测试分进程、分编译缓存执行，
最后给一个合并结论：

```bash
python tools/run_test_suite.py --tier core    # after an edit
python tools/run_test_suite.py --tier smoke   # before opening a pull request
python tools/run_test_suite.py                # the whole tree, on demand
```

`structure` 会检查仓库布局、源码包和 wheel 内容、从源码包生成的 wheel，以及安装后的
`jittor.selftest`。硬件 session 需要预先配置对应后端，并可在 `--` 后传入 pytest 目标；
`docs` 严格构建文档（警告即错误），`docs_links` 验证文档链接，
`tutorials` 在临时目录生成 notebook 并执行维护的教程冒烟测试。

```bash
python -m nox -s cuda -- tests/nn/test_nn_capabilities.py
```

涉及共享语义的改动，先跑定向测试，再跑相关的 CPU 与加速器门禁。并行任务使用不同的
`JITTOR_HOME` 或 `cache_name`；不要让基准与测试共用同一份缓存。

## 文档

`docs/` 是唯一的文档树，其源语言是中文，每一页只属于一个栏目：

| 栏目 | 内容 |
| --- | --- |
| [`docs/guides/`](docs/guides/) | 任务指南：硬件、调试、显存、平台支持与限制 |
| [`docs/notes/`](docs/notes/) | 长期有效的框架机制（精度、放置、异步执行） |
| [`docs/compatibility/`](docs/compatibility/) | Torch 兼容层及其验收规则 |
| [`docs/development/`](docs/development/) | 仓库布局、源码架构、测试体系、已知问题说明 |
| [`docs/research/`](docs/research/) | 尚未定案的研究提案 |
| [`docs/results/`](docs/results/) | 可复现的维护者结论，每条带状态、提交、范围、维护者与复查条件 |
| [`docs/releases/`](docs/releases/) | 发布说明 |

- 安装或首次使用发生变化时更新 [`README.md`](README.md)。它是唯一权威的根 README，
  中文版为 `README.md`，英文版为 `README.en.md`；不要再加按语言复制或生成的第三份。
- 原始日志、缓存与基准产物不进入文档树。
- 文档之间使用相对链接；指向 `docs/` 之外的仓库文件时使用
  `https://github.com/Jittor/jittor/blob/master/<路径>`，两种形式都会被检查。
- 文档改动后运行 `python -m nox -s docs` 与 `python -m nox -s docs_links`，
  移动文件后再跑结构门禁。

## 合并请求

一个合并请求应包含：

- 问题与预期行为；
- 实现边界与兼容性影响；
- 实际运行的测试及其后端/环境；
- 已知限制、skip 或推迟的后续工作；
- 用户受影响时的文档与发布说明变更。

提交评审前检查暂存区，避免 `git add -A`，确保只暂存属于本次改动的文件。评审会优先关注
正确性、公开兼容性、后端行为、打包边界和回归覆盖，而不是表面的格式一致性。

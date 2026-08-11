# Contributing to Jittor / 为 Jittor 做贡献

Jittor accepts bug fixes, features, operators, tests, documentation, and
performance work. By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md) and the project's
[governance rules](GOVERNANCE.md).

Jittor 欢迎缺陷修复、新功能、算子、测试、文档和性能优化。参与项目即表示同意遵守
[行为准则](CODE_OF_CONDUCT.md)与[治理规则](GOVERNANCE.md)。

## Before you start / 开始之前

- Search [existing issues](https://github.com/Jittor/jittor/issues) before filing
  a new report. For behavior changes, describe the intended contract before
  implementation.
- Keep one pull request focused on one coherent change. Avoid generated files,
  runtime caches, model weights, and unrelated formatting churn.
- Reproduce a bug before changing code. Record the smallest command, device,
  input, expected result, and actual result.
- Do not claim accelerator support from import success. Exercise the operation
  on every backend the change advertises.

- 提交新 issue 前先搜索[已有问题](https://github.com/Jittor/jittor/issues)；行为变更应先说明预期契约。
- 一个合并请求只处理一组相关改动，不提交生成文件、运行缓存、模型权重或无关格式化。
- 修复前先复现，并记录最小命令、设备、输入、期望结果和实际结果。
- 仅能导入不代表支持加速设备；必须在所声明的每个后端执行目标操作。

## Development setup / 开发环境

Runtime compatibility is declared in [`pyproject.toml`](pyproject.toml). The
package supports Python 3.7 or newer; repository tooling uses pinned Python 3.11
environments and also compiles the tree with a real Python 3.7 interpreter.

运行时兼容范围以 [`pyproject.toml`](pyproject.toml) 为准。包支持 Python 3.7
及以上版本；仓库工具使用固定的 Python 3.11 环境，并用真实 Python 3.7 解释器编译检查全树。

```bash
git clone https://github.com/<your-account>/jittor.git
cd jittor
python -m pip install -e .
python -m pip install -r requirements/dev-tools.txt
python -m jittor.selftest
```

Jittor compiles C++ and CUDA code on demand. Use an isolated cache for each
checkout or concurrent job:

Jittor 会按需编译 C++ 与 CUDA。每个工作树或并行任务应使用独立缓存：

```bash
export JITTOR_HOME="${JITTOR_HOME:-$HOME/.cache/jittor-dev}"
export cache_name="${cache_name:-local-dev}"
```

For CUDA, set `nvcc_path` when `nvcc` is not already on `PATH`. Backend-specific
CI requirements are documented in [`agent/manuals/environment.md`](agent/manuals/environment.md).

## Repository map / 仓库结构

```text
.
├── python/jittor/       # runtime package and JIT C++/CUDA sources
├── python/jittor_utils/ # installation and compiler utilities
├── tests/               # repository pytest suite; not shipped in wheels
├── examples/            # runnable examples and MyST notebook sources
├── benchmarks/          # ASV performance suite
├── tools/               # build, release, install, and maintenance commands
├── docs/                # durable architecture, development, and release docs
├── agent/               # maintainer/agent workflow and verification reports
├── pyproject.toml       # authoritative package, tool, and pytest configuration
└── noxfile.py           # reproducible local and CI sessions
```

Runtime files loaded by path, especially `python/jittor/src/` and
`python/jittor/extern/`, have packaging and compiler contracts. Read the
[repository layout decision](docs/architecture/repository-layout.md) before
moving them.

## Making a change / 修改代码

Follow the style already used by the subsystem you edit. New Python code must
remain Python 3.7 compatible unless project metadata changes in the same
reviewed change.

修改代码时遵循对应子系统已有风格。除非同一变更明确调整项目元数据，否则新增 Python
代码必须兼容 Python 3.7。

### Python

- Use four spaces and descriptive names.
- Keep public APIs documented and typed where the surrounding module is typed.
- Do not hide import or installation failures with broad exception handlers.
- Add a focused regression test for every corrected behavior.

### C++ and CUDA

- Follow nearby naming, formatting, and generated-code conventions.
- Preserve CPU behavior when changing a CUDA path and vice versa.
- Treat dtype inference, broadcasting, empty tensors, non-contiguous layouts,
  gradients, and device dispatch as part of an operator's contract.
- First-time extension/JIT builds sharing a cache must run serially.

## Tests / 测试

The repository suite lives at root [`tests/`](tests/) and is collected by
pytest. Existing `unittest.TestCase` tests are valid pytest tests. Test
configuration and registered markers are in [`pyproject.toml`](pyproject.toml).

仓库测试统一位于根目录 [`tests/`](tests/)，由 pytest 收集；已有
`unittest.TestCase` 无需改写。测试配置和合法 marker 均在
[`pyproject.toml`](pyproject.toml) 中定义。

Put tests in the matching domain and name modules `test_*.py`. Apply the narrowest
backend or lifecycle marker (`structure`, `cpu`, `cuda`, `rocm`, `npu`, `mpi`,
`slow`, `network`, or `manual`). Skips and expected failures must state the exact
unsupported contract; do not turn failures into unconditional skips.

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

### Nox gates / Nox 门禁

[`noxfile.py`](noxfile.py) is the canonical command surface. The default gate is
`lint`, `format`, `typing`, `structure`, and `py37`.

[`noxfile.py`](noxfile.py) 是统一命令入口；默认门禁包括 `lint`、`format`、
`typing`、`structure` 与 `py37`。

```bash
python -m nox
python -m nox -s structure
python -m nox -s cpu
python -m nox -s cuda
python -m nox -s npu
python -m nox -s rocm
python -m nox -s mpi
python -m nox -s benchmark
python -m nox -s docs
python -m nox -s docs_zh
python -m nox -s docs_links
python -m nox -s tutorials
```

The `structure` session checks repository layout, source and wheel contents, a
wheel built from the sdist, and the installed `jittor.selftest`. Hardware
sessions require a provisioned backend and accept pytest targets after `--`.
`docs` and `docs_zh` build the English and Chinese documentation with warnings
treated as errors, `docs_links` validates documentation links, and `tutorials`
materializes MyST sources in temporary storage and executes the maintained
notebook smoke tests.

`structure` 会检查仓库布局、源码包和 wheel 内容、从源码包生成的 wheel，以及安装后的
`jittor.selftest`。硬件 session 需要预先配置对应后端，并可在 `--` 后传入 pytest 目标；
`docs` 与 `docs_zh` 分别严格构建英文和中文文档，`docs_links` 验证文档链接，
`tutorials` 在临时目录生成 notebook 并执行维护的教程冒烟测试。

```bash
python -m nox -s cuda -- tests/nn/test_nn_capabilities.py
```

For changes spanning shared semantics, run the focused test first, then the
relevant CPU and accelerator gates. Use separate `JITTOR_HOME` or `cache_name`
values for concurrent jobs; do not run benchmarks and tests against one cache.

## Documentation / 文档

- Update [`README.md`](README.md) for installation or first-use changes. It is
  the single authoritative root README and keeps English and Chinese together.
- Put durable decisions under `docs/architecture/`, testing contracts under
  `docs/testing/`, development guidance under `docs/development/`, and research
  proposals under `docs/research/`.
- Put reproducible maintainer evidence under `agent/results/`; keep raw logs and
  generated artifacts outside the source checkout.
- Use relative links for repository files and run the structure gate after moves.

- 安装和首次使用发生变化时更新 [`README.md`](README.md)；它是唯一权威的根 README，
  中英文共同维护。
- 长期决策、测试契约、开发指南和研究提案分别放入 `docs/architecture/`、
  `docs/testing/`、`docs/development/` 与 `docs/research/`。
- 可复现的维护者结论放入 `agent/results/`；原始日志与生成产物不进入源码仓库。
- 仓库内文件使用相对链接，移动后执行结构门禁。

## Pull requests / 合并请求

A pull request should include:

- the problem and intended behavior;
- the implementation boundary and compatibility impact;
- exact tests run and their backend/environment;
- known limitations, skips, or deferred follow-up;
- documentation and release-note changes when users are affected.

Before requesting review, inspect the staged diff, avoid `git add -A`, and ensure
that only files belonging to the change are staged. Reviewers prioritize
correctness, public compatibility, backend behavior, packaging, and regression
coverage over cosmetic consistency.

提交评审前检查暂存区，只加入本次改动涉及的文件。评审会优先关注正确性、公开兼容性、
后端行为、打包边界和回归覆盖。

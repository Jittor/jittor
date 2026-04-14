# Contributing to Jittor / 为 Jittor 做贡献

[English](#contributing-to-jittor) | [中文](#为-jittor-做贡献)

Thank you for your interest in contributing to Jittor! Jittor is an open-source deep learning framework maintained by the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/). We welcome contributions from the community, whether it's fixing bugs, adding new features, improving documentation, or suggesting ideas.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Environment Setup](#development-environment-setup)
- [Codebase Structure](#codebase-structure)
- [Making Changes](#making-changes)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Review Process](#review-process)
- [Writing Documentation](#writing-documentation)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Ways to Contribute

There are many ways to contribute to Jittor:

- **Reporting Bugs**: File bug reports using our [issue templates](https://github.com/Jittor/jittor/issues/new/choose).
- **Fixing Bugs**: Look for issues labeled with [`bug`](https://github.com/Jittor/jittor/labels/bug) or [`good first issue`](https://github.com/Jittor/jittor/labels/good%20first%20issue).
- **Implementing New Features**: Propose new features via [Feature Request](https://github.com/Jittor/jittor/issues/new/choose) issues.
- **Improving Documentation**: Fix typos, add examples, or improve existing documentation.
- **Adding Test Cases**: Help improve test coverage.
- **Implementing New Operators**: Contribute new operators or optimize existing ones.
- **Contributing Model Libraries**: Add new model implementations to the Jittor ecosystem.
- **Citing Jittor**: Cite Jittor in your academic papers.
- **Answering Questions**: Help other users in [Issues](https://github.com/Jittor/jittor/issues) or the [Jittor Forum](https://discuss.jittor.org/).

## Getting Started

### Finding an Issue to Work On

- **New contributors**: Look for issues labeled [`good first issue`](https://github.com/Jittor/jittor/labels/good%20first%20issue). These are tasks suitable for newcomers, such as documentation fixes, simple operator implementations, or test case additions.
- **Experienced contributors**: Check issues labeled [`enhancement`](https://github.com/Jittor/jittor/labels/enhancement) or [`bug`](https://github.com/Jittor/jittor/labels/bug) for more challenging tasks.

Before starting work, please comment on the issue to let others know you're working on it, to avoid duplicated effort.

## Development Environment Setup

Jittor is a JIT-compiled deep learning framework with a Python frontend and C++/CUDA backend. Setting up the development environment requires a few additional steps compared to a regular installation.

### Prerequisites

| OS | CPU | Python | Compiler | (Optional) GPU |
|---|---|---|---|---|
| Linux (Ubuntu, CentOS, Arch, etc.) | x86 / x86_64 / ARM / loongson | >= 3.7 | g++ >= 5.4 | NVIDIA CUDA >= 10.0, cuDNN / AMD ROCm >= 4.0 / Hygon DCU DTK >= 22.04 |
| Windows 10 & 11 | x86_64 | >= 3.8 | - | NVIDIA CUDA >= 10.2, cuDNN |
| macOS (>= 10.14) | Intel / Apple Silicon | >= 3.7 | clang >= 8.0 | - |

### Step 1: Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/<your-username>/jittor.git
cd jittor

# Add upstream remote
git remote add upstream https://github.com/Jittor/jittor.git
```

### Step 2: Install Dependencies

**Linux (Ubuntu):**
```bash
sudo apt install python3-dev libomp-dev g++ build-essential
```

**macOS:**
```bash
brew install libomp
```

**Windows:**
Ensure Python >= 3.8 is installed. If using conda:
```bash
conda install pywin32
```

### Step 3: Install Jittor in Development Mode

```bash
# Install from local source
pip install -e .

# Verify the installation
python -m jittor.test.test_example
```

> **Note**: Since Jittor uses JIT compilation, you don't need to recompile after modifying Python source files. However, changes to C++/CUDA header files in `python/jittor/src/` will be automatically recompiled by the JIT compiler on next use.

### Step 4: Enable CUDA (Optional)

```bash
# Set the nvcc path
export nvcc_path="/usr/local/cuda/bin/nvcc"

# Test CUDA support
python -m jittor.test.test_cuda
```

Or let Jittor install CUDA automatically (Windows):
```bash
python -m jittor_utils.install_cuda
```

## Codebase Structure

```
jittor/
├── python/                    # Python source code
│   ├── jittor/               # Main Jittor package
│   │   ├── __init__.py       # Package initialization, version info
│   │   ├── nn.py             # Neural network modules
│   │   ├── optim.py          # Optimizers
│   │   ├── dataset/          # Dataset utilities
│   │   ├── models/           # Pre-built model implementations
│   │   ├── test/             # Python test suite
│   │   ├── utils/            # Utility functions
│   │   ├── notebook/         # Jupyter notebook tutorials
│   │   └── src/              # C++/CUDA source files (JIT compiled)
│   │       ├── ops/          # Operator implementations
│   │       ├── executor/     # Execution engine
│   │       ├── mem/          # Memory management
│   │       └── optimizer/    # Graph optimization passes
│   └── jittor_utils/         # Utility package
├── doc/                      # Documentation (Sphinx)
├── .github/                  # GitHub templates and CI
│   ├── ISSUE_TEMPLATE/       # Issue templates
│   └── workflows/            # CI workflows
├── setup.py                  # Package setup
├── Dockerfile                # Docker configuration
└── README.md                 # Project README
```

### Key Concepts

- **JIT Compilation**: Jittor compiles C++/CUDA code at runtime. Source files in `python/jittor/src/` are compiled on-the-fly.
- **Meta-operators**: Jittor uses meta-operators to generate specialized code for different hardware.
- **Var**: The basic data type in Jittor for representing tensors.
- **Module**: Base class for neural network modules, using `execute()` as the forward computation method.

## Making Changes

### Branch Naming Convention

Create a new branch from `master` for your changes:

```bash
git checkout master
git pull upstream master
git checkout -b <branch-type>/<short-description>
```

Branch type prefixes:
- `fix/` - Bug fixes (e.g., `fix/conv-padding-error`)
- `feature/` - New features (e.g., `feature/add-groupnorm`)
- `docs/` - Documentation changes (e.g., `docs/update-install-guide`)
- `test/` - Test additions/improvements (e.g., `test/add-batchnorm-tests`)

### Commit Message Convention

Write clear, descriptive commit messages:

```
<type>: <short summary>

<optional body>

<optional footer>
```

**Types**: `fix`, `feat`, `docs`, `test`, `refactor`, `perf`, `chore`

**Examples:**
```
fix: correct padding calculation in conv2d operator

The padding was incorrectly computed when using 'same' mode with
stride > 1, causing output shape mismatch.

Fixes #123
```

## Code Style Guidelines

### Python Code

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide.
- Use 4 spaces for indentation (no tabs).
- Maximum line length: 120 characters.
- Use descriptive variable and function names.
- Add docstrings to public functions and classes.

### C++/CUDA Code

- Use 4 spaces for indentation.
- Follow the existing code style in `python/jittor/src/`.
- Use descriptive variable names.
- Add comments for complex logic.

## Testing

### Running Tests

```bash
# Run the full test suite
python -m jittor.test -v

# Run a specific test file
python -m jittor.test.test_nn

# Run a specific test case
python -m pytest python/jittor/test/test_nn.py::TestNN::test_conv2d -v
```

### Writing Tests

- Place test files in `python/jittor/test/`.
- Test file names should follow the pattern `test_*.py`.
- Test class names should start with `Test`.
- Test method names should start with `test_`.

Example:
```python
import jittor as jt
import unittest
import numpy as np

class TestMyFeature(unittest.TestCase):
    def test_basic_functionality(self):
        """Test basic functionality of my feature."""
        x = jt.array([1.0, 2.0, 3.0])
        result = my_feature(x)
        expected = np.array([...])
        np.testing.assert_allclose(result.data, expected, rtol=1e-5)

    def test_cuda(self):
        """Test CUDA support if available."""
        if not jt.has_cuda:
            return
        jt.flags.use_cuda = 1
        # ... test with CUDA
        jt.flags.use_cuda = 0

if __name__ == '__main__':
    unittest.main()
```

### Test Requirements for Pull Requests

- All new features **must** include unit tests.
- Bug fixes **should** include regression tests.
- All existing tests must pass before submitting a PR.
- If your PR includes CUDA code, include both CPU and CUDA tests.

## Submitting a Pull Request

### Before Submitting

1. **Sync with upstream**: Ensure your branch is up to date.
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```
2. **Run tests**: Ensure all tests pass locally.
3. **Check code style**: Follow the code style guidelines.
4. **Write/update documentation**: If your changes affect user-facing functionality.

### PR Submission Checklist

When submitting a PR, please use the PR template and ensure:

- [ ] The PR description clearly explains the changes and motivation.
- [ ] Related issue(s) are linked (use `Fixes #<issue-number>`).
- [ ] New tests are added for new features or bug fixes.
- [ ] All existing tests pass.
- [ ] Documentation is updated if needed.
- [ ] Code follows the project's style guidelines.

## Review Process

1. **Automated CI**: After submitting a PR, automated tests will run. Please ensure they pass.
2. **Code Review**: A maintainer will review your PR. They may request changes or suggest improvements.
3. **Iteration**: Address review comments by pushing additional commits to your branch.
4. **Merge**: Once approved, a maintainer will merge your PR.

**Review Timeline:**
- We aim to provide initial feedback within **1 week**.
- If you haven't received a response after 1 week, feel free to leave a polite comment on your PR.

## Writing Documentation

Jittor uses [Sphinx](https://www.sphinx-doc.org/) for documentation generation.

- API documentation is written in Markdown format in the `doc/source/` directory.
- Include docstrings in your Python code for automatic API documentation generation.
- When adding new features, update the relevant documentation.

### Building Documentation

```bash
cd doc
bash build_doc.sh
```

## Issue Guidelines

### Reporting Bugs

Please use the [Bug Report template](https://github.com/Jittor/jittor/issues/new?template=bug_report.yml) and provide:

- A clear description of the bug.
- Minimal reproduction steps/code.
- Full error logs.
- Environment information (OS, Python version, Jittor version, GPU info).

### Feature Requests

Please use the [Feature Request template](https://github.com/Jittor/jittor/issues/new?template=feature_request.yml) and describe:

- The feature you'd like to see.
- The use case / motivation.
- Any alternatives you've considered.

## Contact

- **Email**: jittor@qq.com
- **QQ Group**: 761222083
- **Jittor Forum**: https://discuss.jittor.org/
- **GitHub Issues**: https://github.com/Jittor/jittor/issues

---

---

# 为 Jittor 做贡献

感谢你对 Jittor 项目的关注！Jittor 是由[清华大学计算机图形学组](https://cg.cs.tsinghua.edu.cn/)维护的开源深度学习框架。我们欢迎社区的贡献，无论是修复 Bug、增加新功能、改进文档，还是提出建议。

## 目录

- [行为准则](#行为准则)
- [贡献方式](#贡献方式)
- [快速入门](#快速入门)
- [开发环境搭建](#开发环境搭建)
- [代码结构](#代码结构)
- [进行修改](#进行修改)
- [代码风格指南](#代码风格指南)
- [测试](#测试)
- [提交 Pull Request](#提交-pull-request)
- [审查流程](#审查流程)
- [编写文档](#编写文档)
- [Issue 指南](#issue-指南)

## 行为准则

参与本项目即表示你同意遵守我们的[行为准则](CODE_OF_CONDUCT.md)。请在贡献前阅读。

## 贡献方式

你可以通过多种方式为 Jittor 做贡献：

- **报告 Bug**：使用我们的 [Issue 模板](https://github.com/Jittor/jittor/issues/new/choose)提交 Bug 报告。
- **修复 Bug**：查找标记为 [`bug`](https://github.com/Jittor/jittor/labels/bug) 或 [`good first issue`](https://github.com/Jittor/jittor/labels/good%20first%20issue) 的 Issue。
- **实现新功能**：通过[功能建议](https://github.com/Jittor/jittor/issues/new/choose)提交你的想法。
- **改进文档**：修复文档错误、添加示例或改进现有文档。
- **添加测试用例**：帮助提高测试覆盖率。
- **实现新算子**：贡献新的算子实现或优化现有算子。
- **贡献模型库**：为 Jittor 生态添加新的模型实现。
- **引用 Jittor**：在你的学术论文中引用 Jittor。
- **回答问题**：在 [Issues](https://github.com/Jittor/jittor/issues) 或 [Jittor 论坛](https://discuss.jittor.org/)中帮助其他用户。

## 快速入门

### 寻找可参与的 Issue

- **新手贡献者**：请查找标记为 [`good first issue`](https://github.com/Jittor/jittor/labels/good%20first%20issue) 的 Issue。这些任务适合新手，例如文档修复、简单算子实现或测试用例补充。
- **有经验的贡献者**：查看标记为 [`enhancement`](https://github.com/Jittor/jittor/labels/enhancement) 或 [`bug`](https://github.com/Jittor/jittor/labels/bug) 的 Issue。

在开始工作之前，请在 Issue 下评论告知你正在处理该问题，以避免重复工作。

## 开发环境搭建

Jittor 是一个基于 JIT 编译的深度学习框架，前端为 Python，后端为 C++/CUDA。相比普通安装，搭建开发环境需要一些额外步骤。

### 系统要求

| 操作系统 | CPU | Python | 编译器 | GPU（可选） |
|---|---|---|---|---|
| Linux (Ubuntu, CentOS, Arch 等) | x86 / x86_64 / ARM / loongson | >= 3.7 | g++ >= 5.4 | NVIDIA CUDA >= 10.0 / AMD ROCm >= 4.0 / Hygon DCU DTK >= 22.04 |
| Windows 10 & 11 | x86_64 | >= 3.8 | - | NVIDIA CUDA >= 10.2 |
| macOS (>= 10.14) | Intel / Apple Silicon | >= 3.7 | clang >= 8.0 | - |

### 第一步：Fork 并克隆仓库

```bash
# 在 GitHub 上 Fork 仓库，然后：
git clone https://github.com/<你的用户名>/jittor.git
cd jittor

# 添加上游远程仓库
git remote add upstream https://github.com/Jittor/jittor.git
```

### 第二步：安装依赖

**Linux (Ubuntu)：**
```bash
sudo apt install python3-dev libomp-dev g++ build-essential
```

**macOS：**
```bash
brew install libomp
```

**Windows：**
确保已安装 Python >= 3.8。如果使用 conda：
```bash
conda install pywin32
```

### 第三步：以开发模式安装 Jittor

```bash
# 从本地源码安装
pip install -e .

# 验证安装
python -m jittor.test.test_example
```

> **提示**：由于 Jittor 使用 JIT 编译，修改 Python 源文件后无需重新编译。但对 `python/jittor/src/` 中 C++/CUDA 头文件的修改会在下次使用时由 JIT 编译器自动重新编译。

### 第四步：启用 CUDA（可选）

```bash
# 设置 nvcc 路径
export nvcc_path="/usr/local/cuda/bin/nvcc"

# 测试 CUDA 支持
python -m jittor.test.test_cuda
```

或让 Jittor 自动安装 CUDA（Windows）：
```bash
python -m jittor_utils.install_cuda
```

## 代码结构

```
jittor/
├── python/                    # Python 源代码
│   ├── jittor/               # Jittor 主包
│   │   ├── __init__.py       # 包初始化，版本信息
│   │   ├── nn.py             # 神经网络模块
│   │   ├── optim.py          # 优化器
│   │   ├── dataset/          # 数据集工具
│   │   ├── models/           # 预置模型实现
│   │   ├── test/             # Python 测试套件
│   │   ├── utils/            # 工具函数
│   │   ├── notebook/         # Jupyter notebook 教程
│   │   └── src/              # C++/CUDA 源文件（JIT 编译）
│   │       ├── ops/          # 算子实现
│   │       ├── executor/     # 执行引擎
│   │       ├── mem/          # 内存管理
│   │       └── optimizer/    # 图优化
│   └── jittor_utils/         # 工具包
├── doc/                      # 文档（Sphinx）
├── .github/                  # GitHub 模板和 CI
│   ├── ISSUE_TEMPLATE/       # Issue 模板
│   └── workflows/            # CI 工作流
├── setup.py                  # 包安装配置
├── Dockerfile                # Docker 配置
└── README.md                 # 项目 README
```

### 核心概念

- **JIT 编译**：Jittor 在运行时编译 C++/CUDA 代码，`python/jittor/src/` 中的源文件会即时编译。
- **元算子**：Jittor 使用元算子为不同硬件生成专门的代码。
- **Var**：Jittor 的基本数据类型，用于表示张量。
- **Module**：神经网络模块的基类，使用 `execute()` 作为前向计算方法。

## 进行修改

### 分支命名规范

从 `master` 分支创建新分支：

```bash
git checkout master
git pull upstream master
git checkout -b <分支类型>/<简短描述>
```

分支类型前缀：
- `fix/` - Bug 修复（如 `fix/conv-padding-error`）
- `feature/` - 新功能（如 `feature/add-groupnorm`）
- `docs/` - 文档修改（如 `docs/update-install-guide`）
- `test/` - 测试添加/改进（如 `test/add-batchnorm-tests`）

### 提交信息规范

请编写清晰的提交信息：

```
<类型>: <简短摘要>

<可选的详细描述>

<可选的关联信息>
```

**类型**：`fix`、`feat`、`docs`、`test`、`refactor`、`perf`、`chore`

**示例：**
```
fix: 修复 conv2d 算子的 padding 计算错误

当使用 'same' 模式且 stride > 1 时，padding 计算不正确，
导致输出形状不匹配。

Fixes #123
```

## 代码风格指南

### Python 代码

- 遵循 [PEP 8](https://peps.python.org/pep-0008/) 风格指南
- 使用 4 个空格缩进（不使用 Tab）
- 每行最大长度：120 个字符
- 使用描述性的变量和函数命名
- 为公共函数和类添加文档字符串

### C++/CUDA 代码

- 使用 4 个空格缩进
- 遵循 `python/jittor/src/` 中现有的代码风格
- 使用描述性的变量命名
- 为复杂逻辑添加注释

## 测试

### 运行测试

```bash
# 运行完整测试套件
python -m jittor.test -v

# 运行特定测试文件
python -m jittor.test.test_nn

# 运行特定测试用例
python -m pytest python/jittor/test/test_nn.py::TestNN::test_conv2d -v
```

### 编写测试

- 将测试文件放在 `python/jittor/test/` 目录下
- 测试文件名应遵循 `test_*.py` 模式
- 测试类名应以 `Test` 开头
- 测试方法名应以 `test_` 开头

示例：
```python
import jittor as jt
import unittest
import numpy as np

class TestMyFeature(unittest.TestCase):
    def test_basic_functionality(self):
        """Test basic functionality of my feature."""
        x = jt.array([1.0, 2.0, 3.0])
        result = my_feature(x)
        expected = np.array([...])
        np.testing.assert_allclose(result.data, expected, rtol=1e-5)

    def test_cuda(self):
        """Test CUDA support if available."""
        if not jt.has_cuda:
            return
        jt.flags.use_cuda = 1
        # ... test with CUDA
        jt.flags.use_cuda = 0

if __name__ == '__main__':
    unittest.main()
```

### PR 的测试要求

- 所有新功能**必须**包含单元测试
- Bug 修复**应该**包含回归测试
- 提交 PR 前所有现有测试必须通过
- 如果 PR 包含 CUDA 代码，需要同时包含 CPU 和 CUDA 测试

## 提交 Pull Request

### 提交前检查

1. **同步上游代码**：确保你的分支是最新的
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```
2. **运行测试**：确保所有测试在本地通过
3. **检查代码风格**：遵循代码风格指南
4. **编写/更新文档**：如果修改影响用户可见的功能

### PR 提交检查清单

提交 PR 时，请使用 PR 模板并确保：

- [ ] PR 描述清晰说明了修改内容和动机
- [ ] 关联了相关 Issue（使用 `Fixes #<issue编号>`）
- [ ] 为新功能或 Bug 修复添加了测试
- [ ] 所有现有测试通过
- [ ] 按需更新了文档
- [ ] 代码遵循项目的风格指南

## 审查流程

1. **自动化 CI**：提交 PR 后，自动化测试将运行，请确保测试通过。
2. **代码审查**：维护者将审查你的 PR，可能会要求修改或提出改进建议。
3. **迭代修改**：通过向分支推送额外提交来回应审查意见。
4. **合并**：审查通过后，维护者将合并你的 PR。

**审查时间线：**
- 我们会在 **1 周内**提供初始反馈。
- 如果 1 周后未收到回复，请在 PR 中留下礼貌的评论作为提醒。

## 编写文档

Jittor 使用 [Sphinx](https://www.sphinx-doc.org/) 生成文档。

- API 文档以 Markdown 格式编写，位于 `doc/source/` 目录
- 在 Python 代码中添加文档字符串以自动生成 API 文档
- 添加新功能时，请更新相关文档

### 构建文档

```bash
cd doc
bash build_doc.sh
```

## Issue 指南

### 报告 Bug

请使用 [Bug 报告模板](https://github.com/Jittor/jittor/issues/new?template=bug_report.yml)并提供：

- 清晰的 Bug 描述
- 最小复现步骤/代码
- 完整的错误日志
- 环境信息（操作系统、Python 版本、Jittor 版本、GPU 信息）

### 功能建议

请使用[功能建议模板](https://github.com/Jittor/jittor/issues/new?template=feature_request.yml)并描述：

- 你希望看到的功能
- 使用场景/动机
- 你考虑过的替代方案

## 联系方式

- **邮箱**：jittor@qq.com
- **QQ 群**：761222083
- **Jittor 论坛**：https://discuss.jittor.org/
- **GitHub Issues**：https://github.com/Jittor/jittor/issues

---

Thank you for contributing to Jittor! / 感谢你为 Jittor 做贡献！

# 仓库布局

- 状态：已接受
- 日期：2026-08-11
- 上次复查：2026-09-09
- 基线：`f5e8e944` 加本文所述的边界文档改动
- 范围：Jittor 2.0 的仓库结构、打包、兼容层归属、测试、工具与文档
- 取代：长期的 `facade.py + _private/` 目标，以及把 `jittor.test` 作为公开包发布的决定

## 背景

本仓库由上游源码树逐步累积兼容层、后端适配、测试、发布脚本和生成文档而成，由此产生
三个系统性问题：

1. **打包不权威。** 一份简短的显式包列表加上递归的 `package_data` 通配，恰好把未声明的
   Python 包放进了 wheel，同时更深层的运行时资源却可能被漏掉。
2. **领域模块采用"公开文件 + 平行 `_domain/` 实现包"的形态**，于是需要运行时代理和被
   改写的 `__module__` 元数据来绕开这条人为边界。
3. **框架能力、可复用的打补丁机制、Jittor 语义修复和下游项目粘合代码混在 Torch 兼容层里。**

本文定义终态。此前的 facade 拆分仍是有用的迁移工作，但它那种私有包形态是**过渡态而非
目标**。

## 目标

- 让安装产物**只由一份可审计的包与资源清单**推导出来；
- 使用常规的 Python 领域包，**物理模块路径即表达归属**；
- 把可复用的 Torch 兼容与项目专属集成分开；
- 把测试、示例、工具、基准和文档移出运行时包，除非安装后确实需要；
- 把正确性、打包、lint 与性能要求变成**自动门禁**；
- 迁移全程保持 Jittor 的 JIT 运行时资源路径和公开导入兼容。

## 非目标

- 用 PyTorch 内部实现替换 Jittor 的元算子或统一图设计；
- 仅仅为了让源码树看起来整齐而搬动运行时 C++/CUDA 资源；
- 仅凭导入或结构测试就宣称后端能力；
- 永远保留每一条历史反射路径。凡是刻意做出的公开元数据变更，旧的序列化产物会获得
  一条显式兼容路径和一个迁移测试。

## 硬路径不变量

`python/jittor` 是运行时的 `jittor_path`，不只是一个 Python 包。下列路径在**编译器契约
未先行变更并独立验证**之前保持物理稳定：

- `src/**`，安装为 `jittor/src`，含由编译器排序规则选中的源文件；
- `backends/**`，映射到安装后的 `jittor.backends` 各包；
- `python/jittor/build/{dlink_compiler.py,dumpdef.py}` 与 `python/jittor/tools/tracer.py`
  （tracer 由绑定层通过其显式模块名访问）；
- `python/jittor/contrib/math_util/src/*.h`；
- `compat/shim/cpp_extension/{include,src}/**`，由可选的 `jittor-torch` 发行物**独占
  拥有**，作为扩展的已安装 ABI 资源边界；
- `python/jittor_utils` 作为 `python/jittor` 的同级目录；
- `python/jittor/__init__.py` 里字面量形式的 `__version__ = '...'` 赋值——只要发布与
  缓存读取方仍直接解析它。

这些不变量按**运行时契约**检查。它们**不是打包完整性的豁免**：这些契约所需的每一个
安装资源都必须出现在 wheel 里。

## 文档归属

对外文档只有 `docs/` 一棵树，面向使用者与贡献者：入口、教程、API 参考、使用指南、
Torch 兼容、机制说明（`docs/notes/`）、性能、发布说明、社区，以及贡献者文档
（`docs/development/`）。**源语言是中文。**

整改期的过程材料在顶层 `refactor-wip/`：任务看板、计划、交接、分工、迁移期的 API owner
对照表、后端与安装器的内部契约，以及 `refactor-wip/results/` 下的历次验证报告。
**看板是当前任务状态的唯一来源。** 整改收口后 `refactor-wip/` 整个目录删除，其中内容
不迁回 `docs/`。

活跃索引与入链必须可解析；已归档的结论及其原始测试证据**不被改写成当前的成功**。
文档检查维持这条"活跃/归档"边界。

`agent/` 只包含 `manuals/` 与 `skills/`；仓库检查脚本一律在 `tools/`。工作流入口是
[`agent/manuals/agent-index.md`](../../agent/manuals/agent-index.md)。社区项目列表在
`docs/community/`；ASV 源配置在 `benchmarks/asv.conf.json`——nox 从该配置推导绝对的输入
与外部输出路径，不把基准运行状态搬进工作树。

## 工具与包资源归属

仓库命令位于 `tools/`：构建检查在 `tools/build/`，文档治理在 `tools/docs/`，发行物检查在
`tools/release/`，布局入口是 `tools/check_repo_layout.sh`。`python/jittor/tools/` 保留
面向用户的 NVTX、jtune、tracing 和 `jt.benchmark` 实现——**它不是第二个维护者工具目录**。

每个发行物在自己的 `pyproject.toml` 的 `[tool.setuptools.package-data]` 里声明运行时
资源。核心把只含数据的 `jittor.src` 命名空间映射到 checkout 的 `src/`，setuptools 据此
复制文件，**不需要在 setup.py 里再列一份后缀清单**。后端的 package-dir 映射与同一张资源
表保持其安装后的 `jittor/backends/` 路径。独立的 compat 项目拥有自己的 shim 资源与 Torch
entry point。

`tools/build/generate_manifest.py` 从这些表加上 `[tool.jittor.sdist]` 的"仅源码"选择项
推导出两份 MANIFEST.in。生成的是**精确路径**，以规避 MANIFEST 中 `**` 非递归的行为；
含空格的源文件名使用锚定的单字符模式，因为 MANIFEST 没有空白引用语法。枚举前会先移除
缓存与生成输出目录。**新增仅源码文件或改动声明后要重跑生成器**；`--check` 由打包会话与
结构门禁强制执行。两个 sdist 都能用 setuptools 独立构建，不导入运行时，也不依赖兄弟
发行物去读资源列表。

## 目标布局

下面这棵树是 2026-09-02 决定的目的地。理由、逐项的源到目的地对照表、打包耦合关系与
排序，见 [`refactor-wip/architecture/target-layout.md`](../../refactor-wip/architecture/target-layout.md)。

```text
.
├── pyproject.toml  README.md  LICENSE.txt  AGENTS.md  CONTRIBUTING.md  noxfile.py
├── src/                      # C++ 核心，与 Python 包并列，不是包数据
│   ├── core/                 # node var op graph grad executor fused_op
│   ├── type/  mem/  codegen/  ops/  runtime/  bindings/  third_party/  tests/
├── backends/                 # 每个后端一种形态：构建片段 + kernels/ + 注册项
│   ├── cpu/  cuda/  acl/  rocm/  corex/
│   └── comm/                 # mpi nccl hccl
├── python/jittor/            # 纯 Python
│   ├── _core/                # var module function flags hooks
│   ├── build/                # compiler compile_extern pyjt_compiler cuda_wheel install_cuda + jittor_utils
│   ├── ops/                  # 原 misc/，tensor_ops 按领域拆分
│   ├── nn/  optim/  autograd/  fft/  sparse/  dataset/  transform/  models/
│   ├── linalg/  distributions/  init/
│   ├── distributed/  contrib/  tools/
├── compat/                   # 独立发行物（jittor-torch）：torch shim、fsdp2、triton；vLLM 属于 adapters/
├── tools/  tests/  docs/  examples/  benchmarks/
└── agent/                    # 只有 manuals/ 与 skills/
```

旧树没有说明的三条规则：

- **C++ 核心与后端不是 Python 包数据。** 它们与包并列，由显式的构建步骤产出，而不是由
  `import jittor` 产出。
- **一个概念，一个位置。** CUDA kernel 无论被哪一层 Python 调用都住在 `backends/cuda/`；
  构建工具住在 `jittor/build/`；张量运算住在 `jittor/ops/`。
- **布局搬动是每个重构阶段的最后一步，绝不是第一步。** 只有当被搬的代码已经收敛成一种
  形态时，搬动才有意义。

公开的 1.x 导入路径（`jittor.ccl.ccl_2d`、`jittor.pool.AvgPool2d`、`jittor.misc.*`）
在每次搬动后都以**弃用的转发模块**形式存活一个大版本。

## 决定 1：领域包

- 领域包的 `__init__.py` 是它的**组合与再导出面**。
- 实现放在有意义的路径下，例如 `nn.modules.conv`、`nn.functional.padding`。
- 实现可以导入它**确实依赖**的、已初始化的领域接口。终态里**不存在**重复的
  `_JittorRuntimeProxy` 服务定位器。
- 可调用对象的元数据报告**真实的实现模块**，与主流 Python 库的做法一致。新代码**不再
  递归改写 `__module__` 来隐藏归属**。
- `from jittor import nn` 等历史导入保持有效。
- **pickle 兼容性用搬动之前制作的 fixture 验证。** 只有当旧产物确实无法以其它方式加载时，
  才保留兼容别名或范围极窄的反序列化映射。
- 后端的后处理与 monkeypatch 点变成**显式注册表或有文档的扩展钩子**，而不是对 facade
  全局变量的偶然修改。

Stage 3 的四对 facade/私有包以及后来的 FSDP2 迁移脚手架都已不存在。维护中的 Torch API
在 `jittor.compat.torch`，FSDP2 支持在 `jittor.compat.fsdp2`，部署与导入打补丁在
`jittor.compat.shim`。**任何新领域都不得采用 facade/私有包配对。**

运行时根目录还由一条精确的结构契约封闭：其余 Python 文件仅限于运行时组合、编译器/设备
引导、原生的 `distributions`/`init`/`linalg` 领域，以及安装后的自检。`contrib.py`、
`weightnorm.py`、`lr_scheduler.py`、`sparse.py`、`torch_fsdp2_compat.py` 这类兼容文件
**不是合法的根目录归属者**。

## 规范导入与历史导入

| 规范实现 | 兼容入口 |
| --- | --- |
| `jittor.nn` 包 | 既有的 `from jittor import nn` |
| `jittor.autograd` | `jittor.gradfunctional` |
| `jittor.fft` | Torch 模式下的 `torch.fft` |
| `jittor.misc.concatenation` 与 `jittor.misc.indexing` | `jittor.contrib` |
| `jittor.nn.backends.softmax_cuda` | `jittor.other.code_softmax` |
| `jittor.nn.utils.weight_norm` | `jittor.weightnorm` |
| `jittor.optim.legacy_schedulers` | `jittor.lr_scheduler` |
| `jittor.sparse.convolution` | `jittor.nn.sparse` |
| `jittor.compat.torch` | `jittor.torch_compat` |
| `jittor.compat.shim` | `jittor.torch_shim` 与部署的 `torch` 别名 |
| `jittor.compat.fsdp2` | `jittor.torch_fsdp2_compat` |
| `jittor.compat.triton` | `jittor.triton_shim` |

**兼容路径必须解析到规范对象，不得拥有第二份实现或安装器。** 历史的
`jittor.torch_compat` 入口可以幂等地激活规范安装器，因为普通的 Jittor 启动现在刻意保持
原生语义。在调用方今天能观察到 `sys.modules` 标识的地方，迁移保留**同一个共享模块对象**，
而不是状态被复制过的平行 facade。

结构门禁扫描同文件内被遮蔽的定义和跨文件完全相同的实现体。有一小组经过评审的例外，
因为它们的独立性是真实契约：仅涉及文件系统的部署命令、ACL 代码生成模板、架构局部的
模型助手，以及新旧两版 PyTorch checkpoint 读取器。**新出现的重复组在被合并或在该门禁中
被显式说明之前一律失败。**

## 决定 2：四层兼容

Torch 兼容按**归属**而不是按下游项目划分：

1. **框架能力**属于 Jittor 各领域。例如参数化注意力、RoPE、归一化、mesh 和生产级稀疏
   卷积运算。
2. **可复用机制**属于 `jittor.compat`。基于注册表的模块 patcher 拥有导入时机与还原；
   参数化的外部后端加载器拥有源发现、清单、构建和缓存标识。
3. **Jittor 的根因修复**属于核心语义。张量梯度状态、`Parameter` 标识和形状可哈希性
   **不得在每个下游库里各糊一遍**。
4. **项目/版本粘合**放在可选的集成发行物里，例如 `jittor-trellis`、`jittor-gs`、
   `jittor-hf-compat`，通过 entry point 注册。**主线 Jittor 不为每个进程安装常驻的
   TRELLIS 或 Gaussian Splatting finder。**

vLLM 的粘合代码及其专属测试由主仓库的 `adapters/jittor_adapters/vllm` 与
`adapters/tests/vllm` 拥有，随既有的 `jittor-torch-adapters` 发行物分发，**不需要单独的
仓库或发布**。核心与 `jittor-torch` 都不打包该实现，也不打包已废弃的 `jittor.compat.vllm`。
可选的 Torch 阶段只发现名为 `jittor_vllm` 的模块补丁 entry point；不存在时保持可选。
该适配器用共享的、owner 感知的回滚机制保存导入前的扩展设置与导入后的补丁。

**安装器必须独立报告每一次尝试的补丁。** 禁止在一串补丁外面套一个宽泛的
`try/except Exception: pass`——否则一次失败会把之后所有兼容工作都掩盖掉。

## 打包边界

- `pyproject.toml` 是权威的项目元数据与构建系统入口。
- 包发现包含 `python/` 下每个含 `__init__.py` 的目录；一个结构测试比对文件系统与构建
  元数据。
- 运行时的非 Python 资源通过 `MANIFEST.in` 与 package-data 配置**显式声明**。
  **不使用递归的兜底通配把未声明的包偷渡进 wheel。**
- 除非有文档记载的已安装命令需要，测试、演示、notebook、发布工具和基准**不是 wheel
  载荷**。
- `jittor.selftest` 是供安装脚本与容器健康检查使用的小型已安装冒烟测试。
- 每次打包变更都用**成员 SHA-256 加精确路径**的 wheel 基线核对。新增、内容变化和删除
  都需要经过评审的哈希或路径豁免，而且**每条豁免都必须被候选 wheel 消费掉**——这样一个
  陈旧的 wheel 就无法靠回退一次已批准的转换而静默通过。

## 测试边界

仓库测试套件位于顶层 `tests/`，由 pytest 收集，**不是已安装的 `jittor` 子包**。既有的
`unittest.TestCase` 测试在 pytest 下继续有效，没有仅仅为了搬动而重写。

- 共享测试工具在 `tests/_helpers/`；测试模块**不互相导入**作为隐式 helper API。
- 结构、CPU、CUDA、NPU 套件是**分开的 CI 层**。
- 测试选择使用名字与 marker，**绝不使用 `listdir()` 的位置**。
- C++ 的 `test.h` 资源与其编译器 include 消费者一起放在 `python/jittor/src/utils/`。
- 已安装环境的检查用 `jittor.selftest`，而不是随包发布一个测试包。

## 工具与示例边界

仓库运维命令在顶层 `tools/`，教学与可运行样例在 `examples/`。两棵树都**不含
`__init__.py`**，也不参与运行时包发现。它们包含在源码发行物中供维护者使用，但直接构建的
wheel 和从 sdist 推导的 wheel 都排除它们。

Stage 6 作为刻意的 Jittor 2.0 破坏性变更，退役了已发布的 `jittor.vcompiler` 包：它在
import 期间编译一个私有 C++ 扩展、在仓库内没有受支持的消费者，且**没有直接替代品**；
自定义算子作者应改用维护中的 `compile_custom_op` 与 `compile_custom_ops`。包内的
`version` 文件同样退役，因为项目元数据和 `jittor.__version__` 才是版本权威。

Stage 6 同时退役了旧的 LLVM 对齐 pass 和未被使用的 `compiler.compile_extern()` 构建钩子：
活跃的编译器启动路径从不调用该钩子，而保留一个绑定 LLVM 编译器内部的私有 pass，只会
增加不受支持的发行载荷，对运行时 JIT 路径毫无影响。

## 工具链与交付

- **Ruff** 提供格式化与 lint；**mypy** 以显式、逐步收紧的范围引入，而不是一份无法维护的
  一次性豁免清单。
- **pre-commit** 跑快速确定性检查，**nox** 是本地与 CI 的规范命令入口。
- `.github/ci-baseline.env` 是宿主 runner、Python 版本、CPU CI 镜像、CUDA 要求、硬件
  runner 标签、容器矩阵和发布验证矩阵的**唯一维护定义**。可复用的 `_ci-baseline.yml`
  工作流导出这些值，各工作流消费它的输出，**而不是各自重复声明版本或 runner 标签**。
- CPU 测试与打包在该基线的 CPU 镜像中运行。CUDA 与 NPU 作业是容器一致性的**正式例外**：
  它们跑在打了标签的自托管主机上，因为设备、驱动、工具包和厂商运行时是主机能力。
  macOS 与 Windows 的发布检查是对应的原生平台例外。这些作业仍然消费适用于它们的公共
  Python、工具包、runner 标签或平台矩阵值。
- 发布使用 `python -m build` 与 PyPI 可信发布。Jittor 分发的是 Python 加 JIT 源码资源，
  而不是预编译的平台扩展，因此它唯一的规范 wheel 是 `py3-none-any`。发布只构建一次该
  wheel，并在 Linux、macOS、Windows 上安装同一个产物。cibuildwheel 检查断言的是**不产生**
  平台 wheel，而不是用三个平台专属 wheel 替换规范 wheel。
- **ASV** 按提交跟踪速度与内存，使用 `--python=same`、选定的修订、与单元测试隔离的 Jittor
  缓存，以及外部存储的结果与 HTML 报告。
- **文档**使用 Sphinx 与 MyST，源语言为中文，教程由 jupytext 管理。

## 阶段门槛

每个阶段在被认为完成之前，必须提供下列全部适用证据：

- 与已接受基线的**成员哈希加精确路径** wheel 比对；新增、内容变化与删除都需要精确且被
  完全消费的白名单，并与改变发行边界的那个阶段绑定；
- 从隔离的 wheel 安装冷启动导入；
- 受影响的 CPU 矩阵无回归；
- **至少一次真实加速器回归**；在某能力被描述为"支持 NPU"之前，必须先有 NPU 证据；
- `tools/check_repo_layout.sh` 与仓库结构检查；
- 记录命令、结果、已知 skip 和未验证的后端；
- 一次**聚焦的提交**，不夹带无关的用户改动。

兼容层的抽离还额外要求：TRELLIS 与 Gaussian Splatting 的端到端路径在改由可选集成包安装
之后，仍保持其既定的数值与性能结果。

## 后果

迁移期间会**同时存在新旧两种形态**，一部分公开可调用对象最终会报告更具体的实现模块。
换来的是：包内容可审计、依赖方向在文件系统上可见、第三方集成成本变成可选项，以及正确性
与性能要求**从文档变成可强制执行的门禁**。

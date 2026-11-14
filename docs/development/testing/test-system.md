# 测试体系

[本栏目目录](../index.md) · [English](test-system.en.md)

从这里选择测试层级，并在正确的进程模式下运行。pytest 执行仓库测试，nox 提供可复现环境；运行时 wheel 不安装测试。

- 编写断言：[数值与覆盖证据](test-evidence.md)。
- 判断失败、跳过和崩溃：[隔离与完成判据](test-isolation.md)。
- 排查测试耗时：[JIT 缓存](jit-cache.md)。

前提：按[贡献指南](../contributing.md)准备开发 checkout。下表耗时是历史测量，不是每台机器的运行时保证。

## 三层：core / smoke / full

一次全量跑完是十几分钟起步,拿它当「我刚才改的东西有没有坏」的答案太贵,于是有三层,
回答的是三个不同的问题:

| 层 | 问题 | 选择方式 | 实测 |
| --- | --- | --- | --- |
| `core` | 改完一处,有没有立刻坏掉 | `tiers.CORE_FILES` 的**白名单**,每个基本面一个文件 | 44 s(双进程模式,串行) |
| `smoke` | 一个 PR 该等多久 | 整棵树**减去** `tiers.SLOW_FILES` | 约 390 s(4 worker) |
| `full` | 这棵树到底是什么状态 | 整棵树 | 十几分钟起 |

只有 `full` 是关于这棵树的陈述,另外两层是关于时间的陈述 —— 这也是为什么 `core`
用白名单而 `smoke` 用黑名单:`smoke` 漏掉一个文件是**覆盖的洞**,所以默认必须是包含;
`core` 漏掉一个文件只是**发现得晚一点**,后面两层照样会红,所以可以是选择。

`core` 刻意串行:它小到 worker 买不到什么,而且这样在没装 pytest-xdist 的检出上照样能跑。
它的预算由 `tests/structure/test_gate_tiers.py` **按算术**校验(把每个文件的实测秒数加起来
和预算比),不是去断言墙上时钟 —— 后者在忙机器上失败的样子和真回归一模一样。

```bash
python tools/run_test_suite.py --tier core    # 改完就跑这个
python tools/run_test_suite.py --tier smoke   # 提 PR 前
python tools/run_test_suite.py                # 全量,按需
```

往 `core` 里加文件,就是在花所有人的每一次编辑的时间;条目里要写清**实测秒数**和
**它守的是哪个基本面**,两样都由结构测试检查。

## 命令

```bash
# 三层,以及仅原生收集、单个模块
python tools/run_test_suite.py --tier core
python tools/run_test_suite.py
python -m pytest --collect-only -q tests
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -v tests/ops/test_ops.py

# 选择某个运算或后端 marker
JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=cpu PYTHONPATH=python \
    python -m pytest tests/ops/test_ops.py -k exp
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -m structure tests/structure

# 兼容层测试：走符号链接的那条路径，不要直接指 compat/
JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
    python -m pytest python/jittor/compat/tests/torch

# 可复现门禁
python -m nox -s structure
python -m nox -s cpu
python -m nox -s optional
python -m nox -s cuda
python -m nox -s npu
python -m nox -s rocm
python -m nox -s mpi
python -m nox -s nccl
```

nox 会话会建立隔离的状态与缓存。**直接并发运行时也必须使用不同的 `JITTOR_HOME` 或
`cache_name`；新 JIT 运算或扩展的首次构建应当串行执行。**

兼容层那条命令为什么要绕一下：`compat/` 自己带 `__init__.py`，直接
`pytest compat/tests/torch` 会让 pytest 把它当**顶层**包 `compat` 导入，而
`compat/_aliases.py` 的 `from .._runtime import ...` 要求自己是 `jittor.compat`，
于是每一个用例都以
`ImportError: attempted relative import beyond top-level package` 报错——收集阶段
看起来是好的（1888 collected），错误全落在 setup，所以很容易误判成"收集不了"。
`python/jittor/compat` 是指向 `compat/` 的符号链接，从它进去包名就解析成
`jittor.compat.tests.torch`，相对导入成立。门禁里没有这个问题是因为
`_run_pytest_once` 会先 `pip install -e compat/`，把 `jittor-compat` 装成真包。

## 目录布局

主仓有两个测试根，共用一份 pytest 策略：

```text
tests/                    # 原生测试根
├── conftest.py           # 薄适配层：注册 _helpers.pytest_policy
├── _fixtures/            # 版本化的测试数据
├── _helpers/             # 显式的共享测试工具（进程模式、门禁范围、子进程、能力探测等）
├── opinfo/               # 算子元数据、样本、参考、skip 策略
├── ops/                  # 通用 OpInfo 前向与梯度批测
├── autograd/ core/ type/ mem/ runtime/ bindings/   # 原生图、dtype、内存、运行时与绑定
├── codegen/              # 生成代码、pass 与 tuner
├── build/                # Python 构建与编译工具
├── nn/ optim/ init/ linalg/ distributions/ serialization/ data/ contrib/ models/
├── backends/             # 镜像 backends/{cpu,cuda,acl,rocm,corex,comm}，外加 parity/ 跨设备一致性
├── distributed/          # 原生启动模式下的分布式行为
├── debug/ utils/ tools/  # 诊断、通用工具与仓库工具
├── integration/          # 示例、notebook 与跨组件工作流
└── structure/            # 仓库、打包与静态契约
compat/tests/             # 可选 jittor-compat 项目的测试根
├── torch/                # Torch 模式下真跑张量的行为测试
├── structure/            # 名字归属与命名空间发布的结构契约
├── fsdp2/                # FSDP2
└── triton/               # Triton 桥
```

两个根都经由薄 conftest 加载 `tests/_helpers/pytest_policy.py`。helper 是 monorepo 的
开发依赖：**Jittor 与 compat 的 wheel 都不安装任何测试包**，跑 `compat/tests` 需要带着
`tests/_helpers` 的完整 checkout；独立的 adapter sdist 在缺少这个目录时仍能跑它自己的
主机测试。

结构契约按源码域放在 `tests/structure/<源码域>`，跨域的仓库治理留在 `tests/structure`
根上；compat 拥有的结构测试在 `compat/tests/structure`。ACL dtype 记录器
（`tests/structure/backends/acl/test_acl_dtype_preservation.py`）是 CPU mock 契约，带显式
的原生进程例外，**不继承**真 ACL 硬件测试的 fallback fixture 与 `npu` marker。
`tests/core/test_setitem.py` 有意保留在这个路径。

**进程模式属于每个文件，不属于它所在的目录。** 权威选择在
`tests/_helpers/process_modes.py`，完整 runner 经 `tests/_helpers/gate_scope.py` 从中推出
原生与 Torch 两个选择集；没有按 argv 激活的机制。硬件 marker 由路径推出：路径含 `acl`
的带 `npu`，含 `distributed`、`comm` 或 `fsdp2` 的带 `mpi`，`structure` 下的带
`structure`。

`nox -s full` 或 `tools/run_test_suite.py` 跑双进程完整套件；`nox -s structure` 包含各
结构根的契约；下游专属测试在独立生态仓库维护，并单独运行原生 CPU 记录器。直接选择用例时用物理路径。

**生成输入的确定性跨越了布局搬迁。** `tests/_helpers/layout_seed_paths.json` 记录搬迁前的
旧路径；`layout_seed.seed_nodeid` 只在输入生成器对键做哈希之前把新路径换回旧前缀（保留
类、方法与参数化后缀），所以搬过家的用例生成与搬迁前相同的输入。新文件与新方法用新的
规范路径；复用旧方法名的新类、以及名字恰好与旧生成器前缀相同的新显式方法，都算新用例。
pytest nodeid、日志、选择器、失败报告与覆盖率**从不**被这张映射改写。

测试模块可以经由为套件配置的 pytest Python 路径导入 `tests/_helpers` 和 `tests/opinfo`。
**不得把另一个测试模块当作隐式的 helper API 导入。** 共享工具若含有非平凡的比较或设备
逻辑，需要有自己的定向测试。

## 进程模式隔离

Torch 兼容的安装是**进程级**的，会改变公开方法、dtype 提升、归约默认值和惰性执行。
因此**原生测试与面向 Torch 的测试跑在不同的 pytest 进程里**。

`tests/_helpers/process_modes.py` 拥有 `TORCH_MODE_PATHS` 列表；`tests/conftest.py`
在广泛的原生收集中忽略这些路径，并在显式选择且模式不匹配时拒绝收集；调用者必须显式设置 `JITTOR_TORCH_SHIM=1`。共享的 OpInfo
与设备一致性套件使用面向 Torch 的签名，因此属于 Torch 进程。

`tools/run_test_suite.py` 是**完整套件的入口**：它以各自独立的状态、缓存和模式变量运行
原生与 Torch 两个会话，再汇总结果。直接执行 `python -m pytest tests` **有意只是原生
会话**，不得作为全套件覆盖来报告。

一个断言 `result_type`、Torch cast 别名、带类型的张量名或 Torch 特有默认值的测试属于
Torch 会话——即使该文件同时携带底层契约并且位于 `tests/core/` 下。

进程内的独立 PyTorch oracle 要求 `REAL_TORCH_SITE` 指向一个含 PyTorch 二进制 `_C`
扩展的 site-packages 目录，pytest 会在 Jittor 之前预加载该实现。**没有这个显式 oracle
时，测试会跳过可选的 PyTorch 对比**，即使能发现一个已部署的、由 Jittor 支撑的 `torch`
stub 也一样——**该 stub 绝不能被当作独立参考**。

测试子进程同样通过公共 child helper 传递 `REAL_TORCH_SITE` 的模块搜索路径；
只在父进程修改 `sys.path` 不足以固定子进程的参考来源。该路径位于继承的 shim
搜索路径之前，并保留当前 Jittor checkout 的优先级。
显式清除原生参考并请求兼容模式的子进程，还会移除继承搜索路径中由父进程拥有的
原生参考目录，其它搜索路径保持原有顺序。

## CI 支持矩阵

工作流状态是**显式声明**的，因为"某个 nox 会话能跑"并不等于 CI 拥有所需的硬件或依赖。
"手动"意味着维护者必须在已配置的机器上运行指定的 fail-closed 会话，之后才能声称该接口
已验证——**不得把它描述成自动检查**。

| 会话 | CI 状态 | Runner 与触发 |
| --- | --- | --- |
| `cuda` | 自动 | 推送与每周计划使用声明的 RTX 4090 / CUDA 12.2 runner。维护者可给 PR 打 `ci:cuda` 标签；打标签、重开和之后的 synchronize 事件都会跑同一门禁。 |
| `optional` | 手动 | 未声明依赖完备的 CUDA runner。请在下述预置环境中运行 `python -m nox -s optional`。 |
| `rocm` | 手动 | 未声明 AMD/ROCm runner。请在真实受支持的 AMD GPU 上运行 `python -m nox -s rocm`。 |
| `mpi` | 手动 | 未声明多进程 MPI runner。请在有可用启动器与编译器包装器的环境运行 `python -m nox -s mpi`。 |
| `nccl` | 手动 | 声明的 CUDA runner 只保证一块 RTX 4090，而该门禁需要两块可见 GPU。请在多卡主机上运行 `python -m nox -s nccl`。 |

**该矩阵描述的是当前调度，不是后端支持程度。** 手动结果必须记录被测提交、工具链、
设备拓扑、命令和 pytest 结果；**runner 不可用不是通过。**

维护中的 CUDA 会话跑完整的 CUDA 后端目录、dtype 覆盖、CPU/CUDA 设备一致性、Torch TF32
控制项和严格的 CUDA OpInfo 套件。

`nccl` 会话默认需要两块可见的 NVIDIA GPU。它先为每个 rank 串行预热一个隔离的 JIT 缓存，
再用 `jittor.distributed.launch` 验证真实的扁平 FSDP2 参数分片、NCCL all-gather 与
reduce-scatter，以及针对独立 NumPy 结果的分片优化器更新。设置 `JITTOR_NCCL_WORLD_SIZE`
并暴露至少那么多设备，可以跑多于两个 rank。

`optional` 会话是一个 fail-closed 的离线 CUDA 门禁，针对预置的 TorchMetrics、
mmcv-lite/MMEngine、PEFT、Safetensors、TensorDict 和已部署的 FlashAttention 适配器。
它在 pytest 之前逐个探测每个包、显式启用 Jittor Torch shim，并把 **PEFT 导入失败当作
错误而不是可选 skip**。当 `JITTOR_FLASH_ATTN_JITTOR_SRC` 指向官方 FlashAttention
checkout 时，该会话分两阶段：常规可选测试用已部署的 math 适配器运行，然后一个
**要求原生实现**的阶段跑融合 fp16 前向、稠密/变长/打包反向、dropout RNG 重放、GQA 和
float32 opt-in 测试。原生阶段默认 head 维度 32 与 fp16；FlashAttention 的能力环境变量
是在这个基础集上**扩展**而非替换。**原生阶段不能靠回退来满足。**

## 目标

- 用**独立参考**比对前向行为；
- 数值化验证解析梯度，并跨设备验证；
- 让结构、CPU 与硬件要求保持显式；
- 保留有价值的历史边界用例，同时把跨测试导入和临时发现替换为共享 helper 与注册表；
- 通过**具体的 skip 或严格的预期失败**让不支持的行为可见；
- 让收集阶段不产生编译、下载和硬件副作用。

## 验收

一次测试体系改动在满足下列条件时算完成：收集过程无设备副作用、新的 harness 逻辑有定向
自测、结构门禁通过、至少有一个强制 CPU 用例真正执行，并且每个硬件结果都区分了通过、
框架失败与环境不可用。

**数量本身不是验收标准**——证据必须真正检验所声称的语义。

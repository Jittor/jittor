# 测试体系

- 状态：已接受
- 上次复查：2026-09-03
- 基线：`6b8fb594` 加任务 10.03
- Owner：测试基础设施维护者
- 复查触发：收集根目录、进程模式归属、marker、OpInfo 契约或后端门禁发生变化时

Jittor 用 pytest 作为仓库的测试运行器，同时保留兼容的 `unittest.TestCase` 测试。
套件位于顶层 `tests/`，**不属于已安装的包**。
[`pyproject.toml`](https://github.com/Jittor/jittor/blob/master/pyproject.toml)
是收集与 marker 配置的权威来源；
[`noxfile.py`](https://github.com/Jittor/jittor/blob/master/noxfile.py)
是可复现的命令面。

## 目标

- 用**独立参考**比对前向行为；
- 数值化验证解析梯度，并跨设备验证；
- 让结构、CPU 与硬件要求保持显式；
- 保留有价值的历史边界用例，同时把跨测试导入和临时发现替换为共享 helper 与注册表；
- 通过**具体的 skip 或严格的预期失败**让不支持的行为可见；
- 让收集阶段不产生编译、下载和硬件副作用。

## 目录布局

```text
tests/
├── _fixtures/            # 版本化的测试数据
├── _helpers/             # 显式的共享测试工具
├── opinfo/               # 算子元数据、样本、参考、skip 策略
├── ops/                  # 通用 OpInfo 前向与梯度批测
├── backends/             # CPU/CUDA/ROCm/NPU 与跨设备一致性
├── compiler/             # JIT/编译器行为与 kernel 陷阱
├── core/                 # 张量、dtype、图与自动微分契约
├── nn/                   # 神经网络算子与模块
├── optim/                # 优化器契约
├── distributed/          # MPI 与分布式行为
├── compat/               # 兼容接口与导入行为
├── integration/          # notebook 与跨组件工作流
├── structure/            # 仓库、打包与静态契约
├── models/               # 维护中的模型级测试
└── system/               # 进程/环境集成测试
```

测试模块可以经由为套件配置的 pytest Python 路径导入 `tests/_helpers` 和 `tests/opinfo`。
**不得把另一个测试模块当作隐式的 helper API 导入。** 共享工具若含有非平凡的比较或设备
逻辑，需要有自己的定向测试。

## 进程模式隔离

Torch 兼容的安装是**进程级**的，会改变公开方法、dtype 提升、归约默认值和惰性执行。
因此**原生测试与面向 Torch 的测试跑在不同的 pytest 进程里**。

`tests/_helpers/process_modes.py` 拥有 `TORCH_MODE_PATHS` 列表；`tests/conftest.py`
在广泛的原生收集中忽略这些路径，并在显式选中其中之一时激活 Torch 模式。共享的 OpInfo
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

## 算子证据的三层

### 1. 独立前向参考

每个 `OpInfo` 描述可调用对象、样本构造器、dtype、自动微分支持、容差，以及定向的
skip/xfail 策略。`tests/ops/test_ops.py` 把这个数据库在所请求的设备上展开，并与独立的
NumPy 或数学参考比对。

**参考实现不得调用被测的那个 Jittor 运算。** 样本构造器要覆盖有意义的形状、轴、广播、
可选参数和容易出错的 dtype，而不是**在没有语义多样性的前提下堆数量**。

### 2. 数值梯度

可导的 OpInfo 条目跑 float64 的 CPU `gradcheck`；支持二阶微分的运算还跑 `gradgradcheck`。
标注 `supports_autograd=False` 或 `supports_gradgrad=False` 是**契约决定，不是把失败的
测试弄绿的手段**，必须在定义中或问题总账里给出理由。

数值检查在 CPU 上证明导数公式。**它不证明加速器正确执行了同一个反向 kernel。**

### 3. 设备一致性

[`tests/backends/parity/test_device_parity.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/parity/test_device_parity.py)
在 CPU 和可用的加速器上跑相同的输入与余切，用全局与逐元素两种误差度量比较前向输出和
梯度。这一层捕获设备特有的编译失败、丢失的梯度贡献，以及静默的 kernel 偏差。

> **CPU 只有在被前向层与数值梯度层独立钉死之后，才是有效的一致性参考。** 后端环境
> 失败要与框架缺陷分开报告。

## 覆盖面：三层之外的那个问题

上面三层回答的是**「这个算子对不对」**。它们不回答**「我们到底问了哪些算子、
哪些 dtype」**——而缺陷正是从没被问到的地方冒出来的。

三个已经证实的形态：

**算子面远大于数据库。** 公开可调用面有 1294 个名字，OpInfo 覆盖 179 个算子。
差集不是"还没来得及"，它是缺陷的住所。

**"被覆盖"不是二值的。** `bitwise_not` 对 bool 恒返回 `True`（与输入无关），
全程有 OpInfo 条目。条目声明 `dtypes=_INT`（`integral_types()` 不含 bool），
样本生成器又写着 `dt = _int_dtype(dtype)` 把传入 dtype 强制转成整型——于是没有
任何测试**能**喂给它 bool。算子读起来是被覆盖的。另注意 `OpInfo.dtypes` 缺省为
`floating_types()`：忘记声明的条目静默只测浮点。

**按文本统计覆盖率是自欺。** 1294 个名字每一个都在 `tests/` 下某处出现过
（import、注释、无关标识符），那个口径给出约 96%。

对应的三件工具：

| 工具 | 回答 |
| --- | --- |
| [`tests/structure/public_api_manifest.json`](https://github.com/Jittor/jittor/blob/master/tests/structure/public_api_manifest.json) 与其门禁 | 这 1294 个名字**还在不在** |
| [`tests/_helpers/api_coverage.py`](https://github.com/Jittor/jittor/blob/master/tests/_helpers/api_coverage.py)（`JITTOR_API_COVERAGE=1`） | 一次运行**真的调用**了哪些入口 |
| [`tools/opinfo_dtype_gaps.py`](https://github.com/Jittor/jittor/blob/master/tools/opinfo_dtype_gaps.py) | 哪些 (算子, dtype) **跑得通却没声明** |

覆盖测量默认关闭：包装每个公开入口是诊断，不是常态。它已用开/关对照验证不改变
被测系统。无法包装的入口单独记账而不是丢弃——分母里少算一个会美化结果。
dtype 探测按 (算子, dtype) 逐个试编译，属于 nightly 级诊断而非 PR 门禁。

## 跨切面契约

有些语义不属于任何单个算子，因而不属于上面任何一层：驻留与迁移、别名与就地写、
惰性与实体化时机。它们没有归属就没有测试。

[`tests/core/test_var_residency_contract.py`](https://github.com/Jittor/jittor/blob/master/tests/core/test_var_residency_contract.py)
是这一类的第一个：未计算的 Var 没有驻留；读取 device Var 的数据会把存储真的迁回
主机；`device_id` 跨 `cpu()` 保留源设备而 `device`/`location()` 跟随数据。它整体
在 CPU 上运行，CUDA 用例逐条 skip——**去掉加速器只会收窄它而不是清空它**。

这一点是有来历的：Var/Module 的设备方法契约此前只有四条用例，且全部要求两张
CUDA 卡，于是在单卡与 CPU 机器上整体跳过，报告里和四条通过长得一模一样。现在
"设备数量不够"由 `insufficient-devices` 单独计数并在汇总里点名，与"没有加速器"
区分开——前者是硬件在场却仍丢掉的覆盖，不是环境事实。

## 测试分类与 marker

Marker 在 [`pyproject.toml`](https://github.com/Jittor/jittor/blob/master/pyproject.toml)
中注册：

| Marker | 契约 |
| --- | --- |
| `structure` | 不执行设备代码；布局、打包与静态检查 |
| `cpu` | 维护中的 CPU 行为 |
| `cuda` | 需要 NVIDIA CUDA 环境 |
| `rocm` | 需要 AMD ROCm 环境 |
| `npu` | 需要昇腾 CANN 环境 |
| `mpi` | 需要 MPI 启动器或多进程 |
| `slow` | 排除在快速 PR 门禁之外 |
| `network` | 需要外部网络访问 |
| `manual` | 显式选择；**绝不进入自动的默认运行** |

**使用适用范围最窄的 marker。** 硬件测试要探测真实运算，**不得在 CPU 上静默通过**。
network 与 manual 测试要在模块 docstring 里说明其外部要求。

## Skip 与已知失败

- **skip** 表示前置条件不可用，或某个契约有意不被支持；它的 reason 要指明**确切的
  前置条件或限制**。
- **预期失败** 表示一个已复现的框架缺陷。pytest 使用严格 xfail，因此修复会产生 XPASS
  并**强制清理总账**。
- **不要**在测试体外面捕获任意异常再转成 skip。要在执行前**窄范围地**探测可选环境。
- 每个持久的预期失败都要列进
  [问题总账](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)，
  带 owner 和退出条件。

## 命令

```bash
# 完整双进程套件、仅原生收集、或单个模块
python tools/run_test_suite.py
python -m pytest --collect-only -q tests
python -m pytest -v tests/ops/test_ops.py

# 选择某个运算或后端 marker
JITTOR_TEST_DEVICES=cpu python -m pytest tests/ops/test_ops.py -k exp
python -m pytest -m structure tests/structure

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

## 增加覆盖

当一个运算可以共享标准的样本、参考、dtype、梯度和设备一致性机制时，用 **OpInfo 定义**。
当契约涉及状态、变更、序列化、错误行为、模块生命周期、导入顺序、分布式协调，或某个
无法用 OpInfo 表达的具体回归时，用**定向测试**。

一个新运算通常需要：

1. 独立的前向样本与参考；
2. 适用的 dtype、形状、轴、空张量、广播和非连续用例；
3. 有测试支撑的梯度与二阶梯度声明；
4. 对**每一个宣称支持的加速器**做设备一致性；
5. 显式的错误契约测试；
6. 一条 OpInfo 报告条目或定向测试名，使**缺失的覆盖可被发现**。

## 验收

一次测试体系改动在满足下列条件时算完成：收集过程无设备副作用、新的 harness 逻辑有定向
自测、结构门禁通过、至少有一个强制 CPU 用例真正执行，并且每个硬件结果都区分了通过、
框架失败与环境不可用。

**数量本身不是验收标准**——证据必须真正检验所声称的语义。

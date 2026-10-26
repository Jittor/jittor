# Metric 状态转换与真实 TensorDict oracle

状态：已就选定的 tiny 组件在 CPU 与 CUDA 上验证；不声称整库、注册表覆盖、
内存或性能。维护者：Jittor Torch 兼容性维护者。当所列上游版本、Module/张量转换所有者、
runner 精度策略或选定操作发生变化时复查。

实现为 `cf601b9a88d959eb06e1e6e9e298359d27540497`。正式组件测量使用干净的合并基线
`5ad8a1929f2e4a88db39b4f0e570e4901615487d`，其中包含远端
`f02e7aec0b43daf1b275a97eab340de7c4e917d3` 与该实现。这保留了父级的完整
constraint 发布修复与测试。修复前/登记前的原始产物仍按其原始基线存放在外部；
它们不被重新标注。

## 现已验证的内容

新增的构建器位于 `compat/tests/torch/_ecosystem_cases.py`，使用真实的上游包，
没有针对具体用例的 adapter。结果由公共 `verify_repo.py` 驱动、共享
`_ecosystem_runner.py` 和既有 `training_probe.py` 产生。
本验证环境未安装任何 `jittor-torch-adapters` 分发或 `jittor.module_patches` 入口点。

| 组件 | 确切选定范围 | 每次基线/扰动检查的张量数 | 协议证据 |
| --- | --- | --- | --- |
| `torchmetrics_regression` | 真实 MSE 与 MAE 子模块；FP32 预测/目标 `(4, 8)` | 输出向量与两个输入梯度：3 个数组，0 个参数梯度 | CPU/CUDA L2 组件协议；无可训练参数 |
| `tensordict_index_and_stack` | 真实 TensorDict batch `[2, 3]`，输入 `(2, 3, 4)`，整型/bool 索引，惰性 stack，expand，Linear(4,3) | 输出、2 个参数梯度、1 个输入梯度：4 个数组 | 含 3 次显式 SGD 步骤的 CPU/CUDA L3 组件协议 |

每个用例从 seed 0 的独立参考初始权重运行，随后两个运行时使用原始权重重跑 seed 7 输入。
参考输出在扰动下发生变化。期望的梯度名从实际参考构建器中枚举，并在**两个**输出归档中
检查；两个缺失的梯度不能算作一致。所有比较值都是有限值，候选方报告了请求的真实设备、
`fallback_policy=error` 以及零后备。

| 组件/设备 | 基线或扰动的最差绝对误差 | 最差缩放相对误差 | 请求/已检查参数梯度 | 请求/已检查输入梯度 |
| --- | ---: | ---: | --- | --- |
| TorchMetrics CPU | 0 | 0 | 0/0 | 2/2 |
| TorchMetrics CUDA | 0 | 0 | 0/0 | 2/2 |
| TensorDict CPU | 9.536743e-7 | 6.690514e-8 | 2/2 | 1/1 |
| TensorDict CUDA | 1.595855e-3 | 1.021639e-4 | 2/2 | 1/1 |

既有比较在绝对误差至多 `2e-3` **或**字段缩放相对误差至多 `1e-2` 时接受每个张量。
这些是下述实际精度设置下的 FP32 tiny 组件结果，而非 TensorDict CUDA 的精确一致
或 strict-TF32-off 一致。

TensorDict 的三步轨迹使用学习率 `1e-3`、seed 0、两侧相同的完整初始权重以及 train
模式。全部 **21** 个数组一致：每一步的输出、标量 loss、两个参数梯度、输入梯度和两个
更新后的参数。可训练列表恰为 `projection.bias`、`projection.weight`；冻结列表为空。
参数确实发生变化。更新是共享探针中的显式 SGD，不验证 Torch 优化器类、Adam、
上游 Trainer 或分布式训练。

| 设备 | 轨迹数组数 | 最差绝对误差 | 最差缩放相对误差 | 参考 loss 历史 | 候选 loss 历史 |
| --- | ---: | ---: | ---: | --- | --- |
| CPU | 21 | 5.960464e-8 | 1.111221e-8 | 0.041279454, -0.167627022, -0.210601747 | 0.041279420, -0.167627022, -0.210601762 |
| CUDA | 21 | 1.291275e-3 | 2.406770e-4 | 0.041469362, -0.167696729, -0.210582942 | 0.041279450, -0.167627007, -0.210601747 |

对同一个三步探针发出的真实 TorchMetrics 请求以
`Training case has no trainable parameters` 失败，且不创建轨迹归档。
其输入反向成功并不等于 L3 训练。在该测量基线上，scorecard 工具中两个库都还没有
自动注册表分母；那四条记录保留 `registry_verified_level=null` 和
`registry_coverage=null`。仅登记一个手工 CASES 条目不会增加 pytest 生态 CI 方法，
也不会授予注册表覆盖。

## 实际平台与上游标识

- Linux 6.1.188 x86_64、AMD EPYC 9554、Python 3.11、g++ 11.5.0、Jittor 2.0.0。
- 独立的真实 Torch `2.11.0+cu128`，包含其编译的 `_C` 扩展；
  它不是 Jittor 前端。两侧运行时/线程环境计数均为 2。
- CPU 显式屏蔽 CUDA 并设置 `use_cuda=0`；实际 TF32 matmul/cuDNN 被禁用，
  matmul 精度为 `highest`。
- CUDA 在一块分配的 RTX 4090 上执行，驱动 580.178.04。Jittor 使用真实的
  NVCC 12.2.140 工具链和 cuDNN 9.19；oracle Torch 使用 CUDA 12.8。
- CUDA JSON 记录实际 TF32 matmul/cuDNN **启用**，matmul 精度为 **high**，
  cuDNN benchmark 禁用。每组配对的完整 `runtime_conditions` 相等；
  平台块与实际测量 JSON 一致。Kernel 数学为 `strict`，
  MKL/NCCL/MPI/CUTT 以及并行 op 编译被禁用。
- 单元测试与组件测量的缓存根相互独立；CPU 和 CUDA 各有自己的缓存和临时输出。
  首次 JIT 构建未与其他验证 worker 共享。

| 上游 | 固定 release | 干净检出 SHA | 已安装受追踪 Python 源码一致性 |
| --- | --- | --- | --- |
| TorchMetrics | 1.7.4 | `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca` | 341/341 文件字节一致 |
| TensorDict | 0.10.0 | `c34264a58b12471dc3e2913ea586b32431872d4c` | 36/36 文件字节一致 |

一致性检查把已安装 wheel 的 Python 文件绑定到干净的 release 检出；
这不声称对生成文件或可选原生扩展也一致。两个运行时的结果文档都记录已安装包版本与来源。

## 最小复现失败与所有者修复

TorchMetrics 的失败可拆分为两个独立、与框架无关的 API 契约：

1. `Module.to/float` 直接遍历已注册张量，绕过了自定义的 `_apply` 覆盖。
   Metric 未注册的张量/列表累加器留在了旧设备上。
   前端现在递归调用子模块覆盖，对直接的子模块别名去重，并在保留参数对象和梯度的同时
   转换自己的参数/buffer。原生 Jittor 所有者未改动。
2. 增强张量运算符继承了原生的停梯度策略。此外，通过 `detach().clone()` 完成的状态重置
   在 `assign` 把其停梯度标志复制进替换对象时，可能永久丢失 RHS 计算图。
   前端的四个增强算术运算符现在共享既有的 in-place 所有者；当该所有者获得可微的 RHS
   时，它会通过原生 `_update` 重新绑定一个已停止的非视图持有者。no-grad 写入仍被冻结。
   这不确立新的视图梯度行为，也不确立 Torch 未来的 swap/set-data 策略。

TensorDict 此前在导入时失败，因为前端克隆 `ModuleType` 子类时漏掉了定义在类上的
constraints。父提交 `9d9310e3c9a4af737702e12d2d6d47062fa015cd` 在保留原生所有者的
同时发布了类和实例成员。本批次消费该远端实现，而不是保留一个更旧的本地副本。
未创建任何额外的 TensorDict adapter。

## 回归检查与可复现性

在提交实现之前，打过补丁的 `db23449f8` 工作树通过了：

- CPU core：**322 通过、42 跳过、1 xfailed**（原生 98；Torch 模式 224）。
- CUDA autograd/module/梯度管理邻居：**193 通过**。
- 最终新增 API 用例，含子模块别名处理：CPU 由 **43** 项定向检查覆盖；
  CUDA **7 通过**。同样的七个通用 API 用例在 CPU 和 CUDA 上独立对照真实 Torch
  通过：**14 通过**。
- 完整结构：**1430 通过、6 跳过、2 失败**。两处失败都是既有下游结果页缺少共享
  toctree/MANIFEST 登记，由父级集成负责。这不表示该工作树中完整结构已通过。

在同步远端 `f02e7aec0` 之后，干净测量基线 `5ad8a1929` 通过了 **48** 项
CPU API/constraint/audit/训练注册表/probe 检查以及 **8** 项 CUDA API/constraint 检查。
在 runbook/报告更新之后的最终完整结构再次为 **1430 通过、6 跳过、2 失败**：
同样的 toctree/MANIFEST 契约现在同时指出 10 月 14 日的 scorecard 页面和本新报告，
等待共享父级集成。仓库布局与空白检查通过；两个 skill 均通过 skill 校验。
旧的 GradScaler 回归现在按真实 Torch 的要求在 `unscale_` 之前初始化 `scale`，
保留原有的条目计数与 Inf 跳过断言；独立的 CPU/CUDA 契约也经过了检查。

请使用仓库 `agent/skills/` 路径下更新后的
[TorchMetrics runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torchmetrics-torch-compat/SKILL.md)
和 [TensorDict runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/tensordict-torch-compat/SKILL.md)。
公共入口为：

```bash
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py --repo torchmetrics --list-only
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py --repo tensordict --list-only
```

完整命令、原始 NPZ 文件、元数据、运行时精度、期望梯度检查和外部驱动源码哈希绑定在
`$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/component-evidence-20261015.json` 中。
其 SHA-256 为
`6d26ff702720c75526da445053e326cc592d940a980cb1d70339903b8371efd0`。
已安装源码一致性 JSON 的 SHA-256 为
`40840b7ac74a71f5046299d614cd9d4dc0b6cbc2fe7cfbd97d3976418396498a`。
原始数组、日志、wheel、缓存以及本地平台的物理设备选择留在仓库之外。
诊断用的 runner 计时是在没有受控性能协议的情况下收集的；本报告不引用任何速度或内存比。

## 既有 oracle 准入与有界 API 清单

更晚的[共享 scorecard 报告](2026-10-15-metrics-tensordict-scorecards.md)把全新的
干净 `90ee18a` 测量绑定到这些有界维度上。本页较早的组件数组保留其原始基线，
不被重新标注。

基于 `b2aeec10874136c1b4b16c1598241252a5f92c87` 的后续工作为两个已注册构建器增加了
真实的共享 oracle pytest 方法。它们要求 TorchMetrics 的输入梯度以及 TensorDict 的
全部参数/输入梯度。既有 CUDA 子类继承这两个测试。仅源码的准入审计现在在选择器中
保留外层类，因此其报告的路径指向可执行的 pytest 节点，而不是未限定类的 unittest
方法名。

两个 CPU 节点通过，随后两个真实 CUDA 节点通过，零跳过。
真实 oracle 仍为 Torch 2.11.0+cu128；两侧都使用 TorchMetrics
1.7.4 和 TensorDict 0.10.0。CPU 显式隐藏了 CUDA。这些新门禁运行使用
TF32 禁用、cuDNN benchmark 禁用和 strict 原生 kernel 数学；它们不会重新标注更早
启用 TF32 的 CUDA 产物。每次共享比较检查实际设备、相同的精度条件和零后备。一次初始
CPU 尝试失败，因为私有解释器暴露了 SciPy 桩；在其前面加上已有的真实 SciPy 安装修复了
环境。原始失败保留在外部日志中，本次准入后续工作未改动任何包/数学实现。

`registry_inventory.py` 还从顶层包源码 `__all__` 中的每个名称推导出 `public_api` 维度，
不导入任何一个库：

| 仓库 | 完整上游 SHA | 有界公共 API 分母 |
| --- | --- | ---: |
| TorchMetrics 1.7.4 | `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca` | 105 |
| TensorDict 0.10.0 | `c34264a58b12471dc3e2913ea586b32431872d4c` | 60 |

清单保留完整的导出名称列表、源码哈希、真实版本溯源以及全部受追踪 Python 包哈希。
未知的变更、别名或动态导出构造会以失败关闭方式处理。该维度覆盖顶层声明的命名空间；
嵌套 API、类方法和任务变体不在其中。它不是模型族或整库分母。仅凭新清单和门禁日志
不授予任何注册表分子；上文的历史组件记录在另行用完整共享 schema 证据转换之前，
保留其原始源码与范围。

对于两个 CPU 节点，运行既有的 ecosystem pytest 文件，选择
`EcosystemParity::test_torchmetrics_regression` 和
`EcosystemParity::test_tensordict_index_and_stack`，并设置独立的
`REAL_TORCH_PYTHON`、实际的包安装位置、`JITTOR_REQUIRE_REAL_TORCH=1`、
`JITTOR_REQUIRE_TORCHMETRICS=1` 和 `JITTOR_REQUIRE_TENSORDICT=1`。显式隐藏
CUDA 并设置 `JT_USE_CUDA=0`。对于 CUDA，暴露已分配的 GPU，设置
`JT_USE_CUDA=1`，并在 `EcosystemParityCUDA` 上选择相同的方法。使用既有的清单 CLI，
传入 `--repo torchmetrics` 或 `--repo tensordict` 以及相应的干净上游检出。
新的未纳入版本控制的证据位于 `$JITTOR_LAB_ROOT/_state/metric-tensordict-gates/`；
`oracle-cpu-real-scipy.log` 和 `oracle-cuda.log` 包含实际的四次执行，两个
`*-inventory.json` 文件绑定推导出的分母。

后续回归验证：core 322 通过、42 跳过和一项既有预期失败；结构加工具 1690 通过、
六项跳过。后者包含修正后的类限定 Longformer 选择器契约。仓库布局、改动文件 Ruff 以及
空白检查通过。最初的陈旧选择器断言失败保留在 `structure-tools.log` 中；
通过的重跑为同一外部证据目录下的 `structure-tools-corrected.log`。

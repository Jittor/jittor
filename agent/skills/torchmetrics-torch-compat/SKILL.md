---
name: torchmetrics-torch-compat
description: 在 Jittor Torch frontend 上验证真实 TorchMetrics 的回归指标、输入梯度和状态迁移，使用独立 PyTorch 双解释器；区分无参数指标的 L2 组件证据与 adapter 单测。
---

# TorchMetrics × Jittor Torch frontend

使用本 skill 时，同时遵循 [下游接入分流](../downstream-library-adaptation/SKILL.md) 与
[公共验证协议](../torch-compat-repo-runbook/SKILL.md)。先复现纯 Torch API 断点，再决定 owner。

## 当前可运行范围

`compat/tests/torch/_ecosystem_cases.py` 注册了 `torchmetrics_regression`：真实
`MeanSquaredError`、`MeanAbsoluteError` 两个子模块，prediction/target 均为 FP32 `(4, 8)`。
公共 runner 比较两个指标组成的输出及 **prediction、target 两组输入梯度**，共 3 个数组；
不是只比较写死的期望常数。该 case 没有参数，最高只形成 **L2 组件协议证据**，不能授予 L3。

2026-10-15 在 clean 基线 `5ad8a1929f2e4a88db39b4f0e570e4901615487d`，
TorchMetrics **1.7.4** 与独立 Torch **2.11.0+cu128** 完成真实 CPU/CUDA 基线和固定权重
seed=7 输入扰动；所有输出和两组梯度均精确一致。该环境没有安装
`jittor-torch-adapters`，未发现 `jittor.module_patches` entry point；builder 不启用 adapter。
完整环境、上游 SHA、哈希及范围见
[维护者报告](../../../docs/results/2026-10-15-metric-state-and-tensordict-oracles.md)。

该组件尚无自动生成的 TorchMetrics 公共 registry 分母，未计入 registry scorecard。
它在 `CASES` 中可由公共命令执行；本次没有新增 pytest 生态 CI 方法。
分类、聚合、排名指标、MetricCollection、多进程同步和其他 dtype 需要各自证据。

## 复验命令

先确认 oracle 不是 shim，再列出实际注册 case：

```bash
<oracle-python> -c 'import torch; assert hasattr(torch,"_C") and not hasattr(torch,"_torch_compat_install_context"); print(torch.__version__)'
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py --repo torchmetrics --list-only
```

依赖必须在两个解释器中真实可用。运行公共双解释器入口，先 CPU，随后在配置好真实 CUDA
工具链的环境中将 `--device` 与产物目录改为 `cuda`；GPU 选择使用当前会话分配的环境变量。

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 nvcc_path='' JT_BUILD_NVCC_PATH='' \
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo torchmetrics --cases torchmetrics_regression --device cpu \
  --oracle-python <oracle-python> --shim-python <shim-python> \
  --repeats 3 --no-memory --out "$JITTOR_LAB_ROOT/_state/torchmetrics-validation/cpu"
```

`verify_repo.py` 是单次 forward/backward 比较，本身不构成完整 L2 验收。输入扰动按公共
runbook 使用 `_ecosystem_runner.py --seed 7 --weights <seed0-reference.weights.npz>`，
两侧共用原始权重；验证输出确实随输入变化且两组梯度都存在。不能用“两个解释器都没输出梯度”
作为一致。CPU 必须 `use_cuda=0`；CUDA 结果必须记录实际 device 和零 fallback。

CUDA 的精度配置取实际 JSON 的 `tf32` 与 `runtime_conditions.precision`。本次为
matmul/cuDNN TF32 **开启**、matmul precision **high**；CPU 为关闭/highest。
公共命令的诊断计时没有形成速度排名、L4 或 L5 结论，`--no-memory` 不提供显存证据。

## 断点归属与 adapter 边界

| 纯 API 断点 | owner 与回归 |
| --- | --- |
| `Module.to/float` 没调用子类 `_apply`，Metric 自有状态未迁移 | `compat/torch/installers/nn/module_methods.py`；`test_module_custom_state.py` 含递归、child alias 去重、整数 buffer 和参数 identity |
| detached/cloned reset 状态 `+=` 后输入梯度丢失 | tensor installer 的 augmented operator 与共享 `_ip` owner；同文件验证四种 augmented 运算、reset state 和 no_grad |
| 基础 Var/Op 算子数值错误 | Jittor 核心；不能塞进 TorchMetrics adapter |

已有 `adapters/jittor_adapters/torchmetrics.py` 仅处理 1.7.4 的有界 helper policy
（`_bincount`、`dim_zero_cat`、`_safe_divide`），不承担通用 Module/梯度语义。
`compat/tests/torch/test_torchmetrics_compat.py` 是另一条带硬编码期望值及 adapter 标记断言的
可选路径。本轮没有以它证明分类、排名或 adapter 门禁；不要将其直接交给原生 Torch 解释器。
历史“未安装/没有 ecosystem case”的结果已被上述固定版本的组件证据替代，不能扩大成整库支持。

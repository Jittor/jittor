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

后续接入 `EcosystemParity::test_torchmetrics_regression` 与继承的 CUDA 节点，
两组输入梯度均为必需字段；真实 CPU/CUDA 各一项通过。中央 `registry_inventory.py`
可用 `--repo torchmetrics` 从固定上游 `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca`
的顶层 `__all__` 自动枚举 105 个 `public_api` 名称。该轴只覆盖顶层声明的 API，
不是全部指标变体或整库分母；新分母和节点本身不授予 registry scorecard 分子。
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

CUDA 的精度配置取实际 JSON 的 `tf32` 与 `runtime_conditions.precision`。历史 `5ad` 测量为
matmul/cuDNN TF32 **开启**、matmul precision **high**；CPU 为关闭/highest。
公共命令的诊断计时没有形成速度排名、L4 或 L5 结论，`--no-memory` 不提供显存证据。

## 最新固定源码 scorecard

2026-10-15 在干净测量源码 `90ee18a9916acda9b5e86ca665f09a231a41ced1`，
重新运行真实 CPU→CUDA、seed=0 原始权重和 seed=7 固定权重扰动，Metrics 两端均 L2，2/105 顶层 API，零参数，输出与两组输入梯度精确一致。
两侧实际 TF32 关闭、matmul precision highest、fallback error/count0；
共享 auditor 真正接受四条 CPU/CUDA 记录。该结论仅限登记的组件场景，
不扩展为整库支持，也不改标旧 `5ad` 数组。见
[当前 scorecard 报告](../../../docs/results/2026-10-15-metrics-tensordict-scorecards.md)。

## 可复用完整 scorecard 采集

公共 [`component_measure.py`](../torch-compat-repo-runbook/scripts/component_measure.py)
可包装 `verify_repo.py` 启动的两个独立解释器，以及共享 runner 的 seed=7 扰动进程。
每场测量从干净 upstream checkout 再生 inventory，核对实际包版本与全部
`package_sources`，记录完整安装目录文件哈希、真实 Torch 二进制身份和初始权重哈希。
使用任务专属 executable launcher 转发 `python component_measure.py --root <checkout>
--repo torchmetrics --checkout <clean-upstream> -- "$@"`，不要修改共享解释器。

[`collect_component_scorecard.py`](../torch-compat-repo-runbook/scripts/collect_component_scorecard.py)
接受 `--upstream torchmetrics=<clean-upstream>`、每设备一个
`--run torchmetrics=<measurement-directory>`、`--platform cpu=<platform-json>` /
`--platform cuda=<platform-json>`、实际 `--baseline`、`--report`、完整 `--command` 与
`--output`。目录包含公共 `verify-report.json`、`perturb-comparison.json` 以及
runner 原始 NPZ/JSON；扰动文件 stem 为 `torchmetrics_regression.perturb`。
collector 重新比较全部数组后调用共享 `audit_coverage.scorecard`，只引用实际调用的
MeanAbsoluteError、MeanSquaredError 顶层 API，最高 L2；任何版本/源码、梯度、权重、
precision/device/fallback 断点拒绝授级。原始数组不可从旧基线改标为新提交。

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

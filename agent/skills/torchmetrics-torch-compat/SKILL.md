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

## 当前严格来源 scorecard

2026-10-16 在干净测量源码 `91242395f555f3c24f60df665c3ea66e13d01caf`，
以四个新私有计算环境重新完成 CPU→CUDA、seed 0 完整初始权重和 seed 7 固定权重扰动。
Metrics 每设备 6 个完整数组精确相同，输出和两组输入梯度均检查，零参数，最高 **L2**；
分子为 **2/105 @ upstream 986eb900b8bc660ff8174c95ec8f702e0e5cb6ca**。
TensorDict 另有独立 L3 行；共享 auditor 实际接受共四条记录、零拒绝。
两侧 TF32/benchmark 关闭、highest、fallback error/count0，CUDA 有实际指针/UUID/驻留证据。
新报告：[固定官方安装来源与完整协议](../../../docs/results/2026-10-16-metrics-tensordict-installed-source.md)。

`90ee` 的原始数组和当时结果保持不动；当前严格 guard 因缺固定官方 wheel/实际
METADATA/RECORD 证明拒绝其四条记录，不能从今天的安装目录回填旧证据。
新测量绑定全部 341 个上游 Python 文件、346 个完整官方 wheel 包文件和实际 pip
METADATA/RECORD，包含非 Python 资源；不是仅版本字符串或 Python 子集相等。
这仍不证明其余 103 个导出、其他变体或 hosted nightly 实际执行。

## 可复用完整 scorecard 采集

公共 [`component_measure.py`](../torch-compat-repo-runbook/scripts/component_measure.py)
包装公共 runner 和 `training_probe.py`，不复制模型数学。先从干净 upstream 再生 inventory，
用 [`_metric_td_installation.py`](../torch-compat-repo-runbook/scripts/_metric_td_installation.py)
固定的官方 PyPI wheel；实际安装内容、METADATA、RECORD 和 wheel 的完整包载荷必须一致。
任务专属 executable launcher 转发：

```bash
<runtime-python> agent/skills/torch-compat-repo-runbook/scripts/component_measure.py \
  --root <measurement-checkout> --repo torchmetrics --checkout <clean-upstream> \
  --wheel <fixed-official-torchmetrics-wheel> -- "$@"
```

launcher 在通用 runner 清理环境后恢复每侧独立 HOME/JITTOR_HOME/TMPDIR/XDG_CACHE_HOME；
CPU 显式隐藏 CUDA，CUDA 两侧恢复实际 `JT_BUILD_NVCC_PATH`，不得猜测编译器或回填平台。
`verify_repo.py` 用 seed 0；runner 的 seed 7 扰动加载同一完整初始权重，输出必须改变。

[`collect_component_scorecard.py`](../torch-compat-repo-runbook/scripts/collect_component_scorecard.py)
需 `--upstream torchmetrics=<clean-upstream>`、`--wheel torchmetrics=<official-wheel>`、
每设备的 `--run torchmetrics=<measurement-directory>`、`--platform cpu=<json>` /
`--platform cuda=<json>`、实际测量 `--baseline`、`--report`、完整 `--command` 和 `--output`。
目录含 `verify-report.json`、`perturb-comparison.json` 及原始 NPZ/JSON；扰动 stem 为
`torchmetrics_regression.perturb`。collector 比较全部数组，再实际调用共享
`audit_coverage.scorecard`；registry_refs 仅 MeanAbsoluteError、MeanSquaredError。
完整命令、平台、负例与原始 seal 在新报告中。缺梯度、安装来源、平台、原始权重或零 fallback
证明均不得授级；缺 CPU 配对不得计入加速卡覆盖分子。不重标旧数组，不授 L3/L4/L5。

## 断点归属与 adapter 边界

完整 CPU maintained 入口为 `nox -s ecosystem_metric_tensordict_source`，同时要求两库原协议，
不能选择更少 case、梯度、步数或放宽预算。可用
`python tools/provision_ecosystem_nightly.py --library metric_tensordict --state-root <new-external-state> --run`
创建三个独立环境、完整解析依赖并执行该入口。新 profile 保留实际 pip RECORD，使用固定官方 wheel，
CPU 显式隐藏 CUDA；两库共 10 个 worker、35 对数组须全部被公共 collector 验收。
实际 clean `c47033ed0a6bb692922efa98ecbbf7896bfe867f` 完整 fresh CPU profile 已通过：
两库共 10 workers / 35 pairs，三独立环境完整解析与 `pip check` 均通过，公共 auditor 两条 CPU
记录全部接受；其中 Metrics 六对数组、完整输入梯度为 L2，分母仍为 2/105。
完整官方 wheel 载荷和实际 METADATA/pip RECORD 已独立复核，原始 seal 与命令见
[专用报告](../../../docs/results/2026-10-16-metrics-tensordict-maintained-cpu-nightly.md)。
工作流接线本身不证明 hosted schedule 已运行；历史 `912` CPU/CUDA 测量不重标。

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

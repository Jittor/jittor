---
name: tensordict-torch-compat
description: 用独立 PyTorch 对拍真实 TensorDict 的整数与布尔索引、lazy stack 和 Linear 输入/参数梯度，执行固定权重 CPU/CUDA 与三步 SGD 组件验证；不推断整库支持。
---

# TensorDict × Jittor Torch frontend

遵循 [下游接入分流](../downstream-library-adaptation/SKILL.md) 与
[公共验证协议](../torch-compat-repo-runbook/SKILL.md)。TensorDict 私有接口问题与通用
Torch API 语义要分别复现，不能仅凭安装或补丁标记判通过。

## 当前组件范围

`compat/tests/torch/_ecosystem_cases.py` 注册 `tensordict_index_and_stack`：真实
`TensorDict`、`lazy_stack`，FP32 sample `(2, 3, 4)`、batch size `[2, 3]`，含整数顺序索引、
布尔 mask、lazy stack、expand 及 `Linear(4, 3)`。公共 runner 检查输出、两个 Linear
参数梯度和 sample 输入梯度，共 4 个数组。

2026-10-15 clean 基线 `5ad8a1929f2e4a88db39b4f0e570e4901615487d`，
TensorDict **0.10.0** 与独立 Torch **2.11.0+cu128** 的 CPU/CUDA 基线、固定权重 seed=7
输入扰动和三步 SGD 均通过。每步检查输出、loss、两个参数梯度、输入梯度和更新后的两个参数，
共 21 个轨迹数组；两个 trainable 参数、零 frozen 参数。本次没有启用额外 adapter。
详见 [维护者报告](../../../docs/results/2026-10-15-metric-state-and-tensordict-oracles.md)。

这是绑定原测量基线的 **L3 组件协议证据**。后续加入
`EcosystemParity::test_tensordict_index_and_stack` 与继承的 CUDA 节点，显式要求全部
参数和输入梯度，真实 CPU/CUDA 各一项通过。中央 `registry_inventory.py --repo tensordict`
从上游 `c34264a58b12471dc3e2913ea586b32431872d4c` 的顶层 `__all__` 自动枚举 60 个
`public_api` 名称。该轴不是全部容器方法或整库分母；新增分母/节点不自动授予
registry scorecard 分子，也不推断分布式、编译、持久化与其他 dtype 支持。

## 公共入口

```bash
<oracle-python> -c 'import torch; assert hasattr(torch,"_C") and not hasattr(torch,"_torch_compat_install_context"); print(torch.__version__)'
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py --repo tensordict --list-only
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 nvcc_path='' JT_BUILD_NVCC_PATH='' \
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo tensordict --cases tensordict_index_and_stack --device cpu \
  --oracle-python <oracle-python> --shim-python <shim-python> --repeats 3 \
  --no-memory --out "$JITTOR_LAB_ROOT/_state/tensordict-validation/cpu"
```

`verify_repo.py` 单次比较之后仍需固定权重输入扰动和多步轨迹。扰动由同一个
`_ecosystem_runner.py` 入口使用 seed=7、两侧读取 seed=0 原生权重；检查实际输出已改变。

三步轨迹使用公共 `training_probe.py`。示例先在显式禁 CUDA 的环境中运行；CUDA 档须配置
真实工具链、会话分配的 GPU 和独立缓存，再将两次调用的 device/output 同时改为 CUDA。

```bash
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case tensordict_index_and_stack --runtime torch --device cpu --steps 3 \
  --output "$JITTOR_LAB_ROOT/_state/tensordict-validation/cpu/td.train.ref.npz"
PYTHONPATH=python JITTOR_TORCH_SHIM=1 \
<shim-python> agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case tensordict_index_and_stack --runtime jittor --device cpu --steps 3 \
  --weights "$JITTOR_LAB_ROOT/_state/tensordict-validation/cpu/td.train.ref.weights.npz" \
  --output "$JITTOR_LAB_ROOT/_state/tensordict-validation/cpu/td.train.shim.npz"
```

调用两次并不自动验收。按公共 runbook 比较每个 NPZ key、两侧 metadata、初始权重哈希、
完整 trainable/frozen 名单、实际 device 与零 fallback，确认参数发生更新且所有轨迹有限。
该工具显式执行 SGD 更新，不证明 Adam、上游 Trainer 或分布式训练路径。
历史 `5ad` CUDA 实际 TF32 为开启/high，CPU 为关闭/highest。未做 L4/L5、整进程显存或速度排名。

## 最新固定源码 scorecard

2026-10-15 在干净测量源码 `90ee18a9916acda9b5e86ca665f09a231a41ced1`，
重新运行真实 CPU→CUDA、seed=0 原始权重和 seed=7 固定权重扰动，TensorDict 两端均 L3，2/60 顶层 API，完整梯度和三步 SGD 轨迹通过。
两侧实际 TF32 关闭、matmul precision highest、fallback error/count0；
共享 auditor 真正接受四条 CPU/CUDA 记录。该结论仅限登记的组件场景，
不扩展为整库支持，也不改标旧 `5ad` 数组。见
[当前 scorecard 报告](../../../docs/results/2026-10-15-metrics-tensordict-scorecards.md)。

## 可复用完整 scorecard 采集

复用 TorchMetrics skill 描述的公共
[`component_measure.py`](../torch-compat-repo-runbook/scripts/component_measure.py) 与
[`collect_component_scorecard.py`](../torch-compat-repo-runbook/scripts/collect_component_scorecard.py)，
将 repo 改为 `tensordict`。公共 base 输出 stem 为 `tensordict_index_and_stack`，
固定初始权重的 seed=7 扰动 stem 为 `tensordict_index_and_stack.perturb`。
还须通过同一个 measurement wrapper 执行 `training_probe.py`，保存
`tensordict_index_and_stack.train.ref.npz/.json`、`.train.shim.npz/.json` 与原始权重。
collector 实际比较三个步骤的全部 21 个字段，并核对两个 trainable 参数、零 frozen、
sample 输入梯度、SGD 学习率 0.001 和真实参数更新。registry_refs 仅 TensorDict 与
lazy_stack，表示该 tiny 组合场景的 L3，不扩大为全部容器方法或训练设施。

## 断点归属与旧门禁

| 断点 | 实际 owner |
| --- | --- |
| `torch.distributions.constraints` 空 namespace，导入 TensorDict 失败 | `compat/torch/distribution_frontend.py`；ModuleType 子类上定义的 constraint 属性需和实例成员共同发布，保留 native owner；修复提交 `9d9310e3c9a4af737702e12d2d6d47062fa015cd` |
| TensorDict/LazyStackedTensorDict 对 Var 索引 | 既有 `compat/torch/installers/autograd.py` 的 TensorDict compatibility owner；本轮未新增 patch |
| `torch._C._nn._parse_to`、通用 tensor 索引/梯度语义 | `jittor.compat.torch`，先做无 TensorDict 的最小复现 |
| native 算子或真实设备计算错误 | Jittor 核心 |

`adapters/` 没有 TensorDict adapter。已有 `compat/tests/torch/test_tensordict_compat.py` 是
另一条可选 CUDA 单解释器门禁，含 CPU conversion、构造/update/index/lazy stack；它有依赖和
设备 skip 条件。`_jittor_index_compat=True` 和 import 成功都不能替代真实行为与梯度证据。
本轮正式双解释器 case 通过不等于旧 optional 文件已重跑。历史 constraints 导入失败已由
真实 owner 修复和本次 CPU/CUDA 执行闭环替代；不能用旧缺包结论否认当前固定版本结果。

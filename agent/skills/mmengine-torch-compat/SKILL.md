---
name: mmengine-torch-compat
description: 验证真实 MMEngine CPU/CUDA 数值、完整梯度与下游拥有的多步训练；运行独立 PyTorch oracle、正式门禁和带固定上游 SHA 分母的覆盖审计。
---

# MMEngine 与独立 PyTorch 对拍

先按 [downstream-library-adaptation](../downstream-library-adaptation/SKILL.md) 分流，
并阅读配套 [mmcv runbook](../mmcv-torch-compat/SKILL.md)。

## 用途

`mmengine_train_step` 验证真实 `BaseModel.train_step`、默认 `BaseDataPreprocessor`、注册表构建的 `OptimWrapper`、梯度裁剪与 momentum SGD。模型为 LayerNorm(6) → Linear(6,8) → SiLU → Linear(8,2)，输入 `(4,6)`，全部 6 个参数和浮点输入参与梯度检查。
另有原有 `mmengine_base_module` 基础模型 case；
它的普通 backward 或通用 SGD probe 不能证明 MMEngine 自己的 `train_step`。

这里的 CPU/CUDA 小模型正确性不代表完整 Runner/hook/数据集训练、分布式训练、
全库注册项支持或真实尺寸性能。`mmcv.ops` 的编译扩展没有被替换为 Python stub。

## 两侧环境

两个独立解释器分别承载二进制 PyTorch 与 Jittor shim。先断言 oracle 的
`hasattr(torch, '_C')` 为真、`_torch_compat_install_context` 不存在。
候选侧先 `import jittor` 再 `import torch`，必须具有 shim 标记。

专用 `ecosystem_openmmlab` session 使用预装的 MMEngine **0.10.7**、MMCV **2.2.0**
和 NumPy **1.x**，不升级已有生态环境。可直接使用干净的上游 release checkout；
两侧实际 import 的版本与 origin 必须一致。`mmcv.cnn` 可由不含编译 `_ext` 的源码提供，
这不建立任何 `mmcv.ops` 覆盖。所有环境、原始日志和缓存位于外置 lab state。

## 在 shim 上跑

以下命令是实际下游训练入口；oracle 先保存完整初始权重，候选侧复用它们：

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> \
  agent/skills/torch-compat-repo-runbook/scripts/openmmlab_training_probe.py \
  --case mmengine_train_step --runtime jittor --device cpu \
  --steps 3 --learning-rate 0.01 \
  --weights "$JITTOR_LAB_ROOT/_state/openmmlab/ref.weights.npz" \
  --output "$JITTOR_LAB_ROOT/_state/openmmlab/shim.npz"
```

## 在原生 torch 上跑

```bash
<real-torch-python> \
  agent/skills/torch-compat-repo-runbook/scripts/openmmlab_training_probe.py \
  --case mmengine_train_step --runtime torch --device cpu \
  --steps 3 --learning-rate 0.01 \
  --output "$JITTOR_LAB_ROOT/_state/openmmlab/ref.npz"
```

`--device cpu` 在导入框架前明确禁用 CUDA。CUDA 在 CPU 通过后另跑，显式分配可见设备、
配置真实工具链；观察到的 device 必须与请求一致，候选侧禁止 backend fallback。

## 对拍

```bash
REAL_TORCH_PYTHON=<real-torch-python> JITTOR_CI_PYTHON=<jittor-python> \
  JITTOR_TEST_DEVICES=cpu <tools-python> -m nox -s ecosystem_openmmlab
# 在已分配真实 CUDA 设备的独立缓存中，另执行 JITTOR_TEST_DEVICES=cuda。
```

专用 session 同时执行两个真实 MM case 和 BatchNorm counter 回归。门禁必须启用
`JITTOR_ECOSYSTEM_OPENMMLAB=1`、逐库 `JITTOR_REQUIRE_MMENGINE/MMCV=1`、
`JITTOR_REQUIRE_REAL_TORCH=1`、`JITTOR_TEST_REQUIRE_EXECUTION=1`，速度比例门限为空。
手工只跑单库 test 时，该库缺失同样直接失败，不能把 skip 视作通过。

门禁先对比普通 forward/backward 的全部命名梯度，再逐张量检查真实 `train_step`
三步的输出、scalar/list loss 解析、输入梯度、裁剪后/SGD 更新前的全部参数梯度、
更新后参数、momentum state 和 buffers。每步必须恰有一次 loss forward 和优化器 step。
两侧 tensor key 集合与 shape 必须完全相同、数值有限；固定输入和 targets 必须逐元素一致。
浮点梯度按现有 harness 的场尺度 floor 判断，不能用广播或 reshape 隐藏形状差异。

L3 scorecard 还须另绑定固定平台、初始权重、7/19 两次输入扰动、实际三步 raw/metadata、
重新生成的上游 inventory 与执行门禁 node。CUDA TF32 状态按实测记录；tiny case 不授 L4/L5。

## 断点与分流

- `torch.nn.BatchNorm*.num_batches_tracked` 的 scalar/int64/persistent 契约归
  **compat NN 构造前端**；保持原生 Jittor 的 checkpoint 格式。逐设备检查三步计数与
  `state_dict` 保存、加载后的继续更新。
- 真正计算/梯度问题先去掉 MM 依赖复现，再按 core/compat 归属修复。
- 上游 registry、训练 owner 和编译扩展不能由 adapter、mock 或 no-op 替代。

## 坑与假绿

源码注册分母由 `registry_inventory.py --repo mmengine --checkout <clean-release>`
自动提取。MMEngine 显式声明轴为 MODELS / OPTIM_WRAPPERS / LOOPS /
WEIGHT_INITIALIZERS；动态遍历 `torch.optim` 与可选优化器另行披露，不伪造完整优化器分母。
MMCV 扫描整个包，保留条件分支、alias、注册循环和依赖扩展的声明，不能只取已能 import
的 `cnn` 子集缩小分母。动态声明无法解析时必须拒绝该有界轴。
这些是固定源码的**声明轴**，不是运行环境注册表或整个库的 API 完整性。

## 证据

模型与完整命名梯度契约见
[`_ecosystem_openmmlab_cases.py`](../../../compat/tests/torch/_ecosystem_openmmlab_cases.py)，
真实训练 probe 见
[`openmmlab_training_probe.py`](../torch-compat-repo-runbook/scripts/openmmlab_training_probe.py)，
正式 node 见
[`test_ecosystem_openmmlab.py`](../../../compat/tests/torch/test_ecosystem_openmmlab.py)。
可复现覆盖与平台协议遵循
[共享 runbook](../torch-compat-repo-runbook/SKILL.md) 与
[scorecard 协议](../../../docs/development/downstream-scorecards.md)。
具体通过级别只取带 source SHA、逐设备 raw hashes 的结果报告；环境缺库记录与旧 tiny
速度不能转移到新版本、新平台或新训练路径。

固定 clean `2aa27ffe1ce1699c888575f3b5fb3092b1fc6eff` 的 CPU/CUDA 实测、三步
真实 owner 轨迹、源码分母及 precision 否定证据见
[OpenMMLab 结果报告](../../../docs/results/2026-10-15-openmmlab-training-oracles.md)。
本场景两设备授 L3；具体 registry refs、全部可训练参数与未覆盖轴以该报告和
原始 scorecard 为准，不外推完整 Runner、编译 ops 或性能级别。

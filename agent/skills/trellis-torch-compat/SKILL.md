---
name: trellis-torch-compat
description: 验证真实 pinned TRELLIS.2 导出模型的独立 CPU/CUDA 数值、全梯度及三步训练轨迹，生成自动注册表分母的 L3 scorecard。完整预训练 3D pipeline、稀疏 CUDA 扩展和性能仍须单独验收。
---

# TRELLIS.2 真实模型对拍

## 当前实测范围

2026-10-15，干净上游源码 `75fbf0183001ed9876c8dbb35de6b68552ee08bd` 上的
`SparseStructureFlowModel`、`SparseStructureEncoder` 和 `SparseStructureDecoder`
已分别通过独立 CPU、CUDA 的 L3 对拍，候选 fallback 均为零。
结果为 **models 3 / 9 @ upstream 75fbf0183001ed9876c8dbb35de6b68552ee08bd**，
两个 device 分别计数。它们是真实上游类的 tiny 配置，dense attention 选择上游支持的
`sdpa`；全部 FP32，TF32 和 cuDNN benchmark 关闭。

可复查结论、完整基线、误差、环境差异和原始证据入口见
[`2026-10-15-trellis-model-validation.md`](../../../docs/results/2026-10-15-trellis-model-validation.md)。
维护者：下游兼容性维护者。Review when：上游声明/模型源码、compat attention/Conv3d/
normalization/autograd/optimizer、device/precision 检查、所选 CUDA 二进制或门禁协议变化。

没有验证完整 4B 生成 pipeline、预训练权重、mesh 数值、CuMesh/FlexGEMM/nvdiffrast/
o-voxel/FlashAttention 扩展、FP16/BF16、NPU/ROCm 或速度/整进程峰值显存。
历史报告派生的 pipeline 比值 `1.0926x` 未在这里复现，性能门禁仍开放。
L3 加零 fallback 不能升级成 L4，不能外推整库支持。

## 先确认环境和真实断点

按 [`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)
分流：缺能力到 core、Torch 签名/语义到 compat、库私有细节才考虑 adapter。
本次三个模型路径直接通过，**无需 adapter 或 framework 修补**。

- 两个进程使用同一干净的官方 Microsoft TRELLIS.2 checkout，钉完整 SHA。
  本源码没有发布版 version 字段；`source@<SHA>` 仅表示源码身份。
- oracle 必须是真实 binary Torch；门禁断言 `_C` 是共享库且未导入 Jittor。
  本轮实际 Torch `2.11.0+cu128`、Python 3.11.16、NumPy 1.26.4、einops 0.8.2、
  easydict 1.13。上游 setup 推荐 Torch 2.6.0，本轮不证明该另一个版本组合。
- 默认 dense Flow 前向在独立 Torch 上因缺少 `flash_attn` 失败；上游 `sdpa`
  选项使这三个 dense 模型可测。**不要给 sparse attention 伪造一个 CPU 实现**。
- 实际 `SLatFlowModel.forward` 和 `SparseConv3d` 构造分别停在缺少 `flash_attn`、
  `flex_gemm`；另有模型/完整 pipeline 导入停在 `cumesh`、`trimesh`。
  这些是当前环境的首个依赖断点，不是整个库 CPU 不可能运行的结论。
- CPU 在 import 前显式 `JT_USE_CUDA=0 use_cuda=0 JT_BUILD_NVCC_PATH='' nvcc_path=''`。
  CUDA 只给显式分配的一张可见卡，用独立缓存并串行首次编译。
- 核对**实际二进制**，不只看 wheel metadata。本轮 cuDNN metadata 9.26.0.51，
  所选实际库 API 返回 9.24.0；平台证据同时保存两者和库哈希，不能把 metadata 写成实测版。

## 运行已维护的门禁

```bash
export JITTOR_TRELLIS_CHECKOUT="$JITTOR_LAB_ROOT/trellis-validation/TRELLIS.2"
export REAL_TORCH_PYTHON=<binary-torch-python>
export JITTOR_REQUIRE_TRELLIS=1 JITTOR_REQUIRE_REAL_TORCH=1
export JITTOR_TORCH_SHIM=1 PYTHONPATH=python
export JITTOR_TEST_DEVICES=cpu
export JT_USE_CUDA=0 use_cuda=0 JT_BUILD_NVCC_PATH='' nvcc_path=''
export JITTOR_TRELLIS_EVIDENCE_DIR="$JITTOR_LAB_ROOT/_state/trellis-validation/fresh-cpu"
<jittor-python> -m pytest -q -s --timeout=1800 compat/tests/torch/test_ecosystem_trellis.py
```

CUDA 设置 `JITTOR_TEST_DEVICES=cuda JT_BACKEND=cuda JT_USE_CUDA=1 use_cuda=1`，
提供完整受支持的 CUDA 编译器/组件栈、显式 `CUDA_VISIBLE_DEVICES=<allocated-gpu>`、
独立 `JITTOR_HOME` 和全新 evidence 目录。候选的 device location/ordinal 与 error/zero
fallback 检查覆盖所有观测张量。原始输出目录拒绝复用，避免旧结果被当作新执行。
缺 checkout/oracle 时普通环境诚实 skip；require 标志使其成为配置错误。

中央 nox 入口为 `ecosystem_trellis`，通过 `JITTOR_CI_PYTHON` 选择候选解释器，
并读取同样的 checkout/oracle/device 环境变量：

```bash
JITTOR_CI_PYTHON=<jittor-python> nox -s ecosystem_trellis
```

实际运行器为
[`_ecosystem_trellis_runner.py`](../../../compat/tests/torch/_ecosystem_trellis_runner.py)，
门禁节点为 `test_real_trellis_models_training[<public-model-name>]`。
原始权重从 oracle 转给候选，绕开 Flow 默认全零输出的假绿。
每个模型检查全部参数与输入梯度、三步 loss/参数/SGD momentum 轨迹，并逐一扰动每个输入。
比较器沿用共享 forward `2e-3`、backward `1e-2` 判据，不按当前误差放宽。

## 自动分母与 scorecard

`registry_inventory.py --repo trellis` 重新读取全部 literal lazy export maps、sampler
导出、模块和 package loader，按 namespace 分轴。保留 abstract `Sampler` 和 8 个
声明但无定义的导出项；没有安装某个扩展也不能删分母。动态/空注册表显式失败。
本轮其它轴为零已验证成员，不能凭调用 trace 把模型通过拆成组件通过。

[`collect_scorecard.py`](scripts/collect_scorecard.py) 从正式 gate 产物、实际平台 JSON 和
重新生成的上游 inventory 构造 L3 records，再调用现有 `audit_coverage.scorecard` 校验。
它检查源身份、全部 93 个 Python 源文件哈希（含 package 根 __init__.py）、基线、
完整原始产物 manifest 和重复场景；所有证据仍留在仓外。
已接受 scorecard 为六个 L3 records、零拒绝；749 旧记录缺根文件身份，被拒后保留原样，
正式矩阵在 6d397 修正后重跑，不能补写旧 JSON 冒充已验证。
平台 JSON 须包含真实硬件、库/驱动/编译器、完整 flags、precision 和 binary oracle 身份，
不能复制另一台机器的表头。

```bash
<jittor-python> agent/skills/trellis-torch-compat/scripts/collect_scorecard.py \
  --checkout "$JITTOR_TRELLIS_CHECKOUT" \
  --run "$JITTOR_LAB_ROOT/_state/trellis-validation/formal-source-complete-cpu" \
  --run "$JITTOR_LAB_ROOT/_state/trellis-validation/formal-source-complete-cuda" \
  --platform "cpu=$JITTOR_LAB_ROOT/_state/trellis-validation/cpu-platform.json" \
  --platform "cuda=$JITTOR_LAB_ROOT/_state/trellis-validation/cuda-platform.json" \
  --baseline 6d39728fffc37d4618c1cb91a9caa1d67beec311 \
  --report docs/results/2026-10-15-trellis-model-validation.md \
  --command '<exact validation command and environment>' \
  --output "$JITTOR_LAB_ROOT/_state/trellis-validation/scorecard-source-complete-final.json"
```

更换基线后重新跑 gate，再传新 SHA；历史记录始终显式标成历史基线，不改原始 metadata。

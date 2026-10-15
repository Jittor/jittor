---
name: flash-attention-torch-compat
description: 验证 FlashAttention 公开 API 在 Jittor Torch 前端上的真实融合执行、完整梯度与三步训练，用隔离的真实 PyTorch CUDA 对拍并检查源码、产物和平台证据。
---

# FlashAttention 公开 API 验证

先按 [`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)
分流。FlashAttention 的公开 facade、扩展桥接和类型边界属于 compat；随机状态和
计算属于 native owner，不为这些问题新建 adapter。

## 源码与覆盖边界

训练探针固定官方 FlashAttention **2.7.4.post1**，提交
`5231d95fe13733fb534c01895f7ea88c6a6c7793`，来自
`https://github.com/Dao-AILab/flash-attention.git`。两侧使用同一份干净源码；
上游或 ABI 改变后重新构建和验证。

用 [registry_inventory.py](../torch-compat-repo-runbook/scripts/registry_inventory.py)
重新枚举 `flash-attention` 的公开 `attention_api` 分母。该固定提交导出七项，
训练探针接受其中六个 dense/packed/varlen 入口；`flash_attn_with_kvcache` 不在本探针
范围内，不能从六项训练结果推断其覆盖。head dimension、dtype、causal、输入扰动是
另列的场景轴，不用它们改变 API 分母。

[flash_training_probe.py](scripts/flash_training_probe.py) 实际调用公开函数，使用三个
FP32 Q/K/V 参数和一个可微输入；输入转换为真实 fp16/bf16，执行三次真实
`torch.optim.SGD` 更新。每步保留输出、loss、全部参数/输入梯度及更新后参数，共
30 个数组，并检查 CUDA、dtype、有限值、参数更新及候选端零 fallback。
默认 head dimension 32，也支持 64；可分别选择 causal 和输入扰动。输入来自固定
NumPy seed，属于 tiny 组件训练，不是完整模型、生产尺寸或速度验收。

该探针只执行真实 CUDA。上游 CUDA 扩展不能提供 CPU 对拍结果；CPU 核心和 facade
类型契约单独验证，不能当作上游 FlashAttention CPU 数值证据。CUDA 候选端跑通也
不是独立 oracle 一致性结论；需要下面的比较器验收实际两侧产物。

## 隔离环境

- 候选端加载当前源码的 Jittor/Torch shim。部署的 `flash_attn` facade 必须逐 Python
  文件匹配 `compat/shim/resources/stubs/flash_attn/`；探针会核验，避免测试旧副本。
- 参考端是独立 binary PyTorch，不能导入 Jittor 或带 shim 标记。其 Python
  `flash_attn` 必须来自指定的上游 checkout，`flash_attn_2_cuda` 必须是真实二进制扩展。
  按该解释器的 Torch/CUDA ABI 构建扩展；先检查依赖 pin，不让安装过程替换 oracle。
- oracle 环境清除 Jittor 的 source/shim 路径和库路径污染，配置自己的 `CUDA_HOME`
  与 `CXX`，让工具记录可核查的编译器版本。候选端记录实际 Jittor compiler 配置。
- 每侧、每 device、单测与 benchmark 分别使用独立缓存；原始日志、NPZ、JSON、
  编译缓存和工具链放在 `$JITTOR_LAB_ROOT` 外置状态目录。第一次编译串行。
  编译无输出时检查子进程与锁；观察超时不代表进程终止。

候选端设置 `JITTOR_TORCH_SHIM=1`、源码 `PYTHONPATH`、实际 CUDA、私有 `JITTOR_HOME`
与 `TMPDIR`，以及：

```bash
export JITTOR_FLASH_ATTN_JITTOR_SRC=<pinned-upstream-checkout>
export JITTOR_FLASH_ATTN_HEAD_DIMS=32,64
export JITTOR_FLASH_ATTN_DTYPES=fp16,bf16
export JITTOR_FLASH_ATTN_JITTOR_REQUIRED=1
export JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES=0
export JITTOR_FLASH_ATTN_MIN_SCORES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

两侧均显式选择分配的 `CUDA_VISIBLE_DEVICES`。探针自动进入 `--jittor-root` 对应目录，
使扩展缓存命名空间与所验证的项目一致。参数默认指向工具所在 checkout；候选端
运行的 Jittor 必须来自这个源码目录。正式验收要求提交干净；`--allow-dirty` 只供诊断，
比较器会拒绝这种产物。

## 执行和比较

以下两条命令分别在各自配置好的独立环境中运行。替换 `float16`、API、head dimension，
并增加 `--causal` 或 `--perturbation 0.125` 来执行其他场景；两侧参数必须一致。

```bash
<oracle-python> agent/skills/flash-attention-torch-compat/scripts/flash_training_probe.py \
  --runtime torch --upstream <pinned-upstream-checkout> \
  --api flash_attn_func --dtype float16 --head-dim 32 \
  --output <external-state>/reference.npz
<jittor-python> agent/skills/flash-attention-torch-compat/scripts/flash_training_probe.py \
  --runtime jittor --upstream <pinned-upstream-checkout> \
  --api flash_attn_func --dtype float16 --head-dim 32 \
  --output <external-state>/candidate.npz
python agent/skills/flash-attention-torch-compat/scripts/compare_training.py \
  <external-state>/reference.npz <external-state>/candidate.npz \
  --output <external-state>/comparison.json
```

比较器检查版本、源提交、执行脚本 hash、输入/精度协议、完整 30 数组、产物 hash、
有限值和每次参数更新。参数本身很大时也不能隐藏缺失的小更新。误差按各 tensor 的
参考最大绝对值归一化，floor 为 `1e-8`；fp16 前向 `3e-3`、梯度/更新 `6e-3`，
bf16 均 `3e-2`。这些判据在首次两侧比较前固定，禁止看结果后放宽。初始参数要求
精确相同。修改精度协议后重新采集；不拿不同协议的产物直接比较。

平台记录包含真实 GPU/driver 清单、可见卡、CPU、Torch/CUDA、编译器、NVIDIA wheel、
源码与二进制 hash、关键 flag。`use_acl` 是 native `use_cuda` 的遗留别名，单独读它
不能证明运行了 NPU；实际 CUDA dispatch 和张量设备另行断言。工具不输出 L4 速度结论。
通过比较器后，再按[统一 scorecard 协议](../../../docs/development/downstream-scorecards.md)
登记实测范围与覆盖分母；工具存在、单侧完成或合成验证器测试都不算库的已验分子。

## 回归与证据

- `compat/tests/torch/test_flash_attn_frontend.py` 检查公开入口保留 Torch Tensor 类型，
  以及原生调用者和异常后的作用域恢复。
- `compat/tests/torch/test_torch_compat_attention.py` 包含 native-required CUDA 融合、
  Q/K/V 反向和 dropout 种子重放；`nox -s optional` 的 math/native 阶段各自验收。
- `tests/tools/test_flash_training_comparison.py` 用合成产物验证缺失的小梯度/更新、
  不同协议、脏源码、损坏产物和禁用断言不能假绿。
- [CUDA 扩展 Philox 与融合训练回归](../../../docs/results/2026-10-15-cuda-extension-philox.md)
  是已提交的 native CUDA 证据，明确没有将其等同于独立 PyTorch FlashAttention 对拍。
- [公开训练 API 独立对拍](../../../docs/results/2026-10-15-flash-attention-training-oracles.md)
  记录干净 `71c5be74e` 的六入口、48 条真实 CUDA 三步轨迹，保留原阈值、完整梯度和
  单独更新检查；统一 scorecard、固定 causality 的独立扰动门禁及 KV cache 仍待验。

Owner：native CUDA RNG/计算与 Torch 兼容维护者。上游版本、扩展 ABI、公开 facade、
Tensor 类型/梯度边界、RNG owner 或验证协议变化时重新检查证据。

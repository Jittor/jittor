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
`https://github.com/Dao-AILab/flash-attention.git`。官方参考实现和 Jittor facade/原生
扩展桥接是两个不同实现，分别核验实际安装来源；不能把 facade 冒充官方 Python 包。
两侧扩展引用固定官方 CUDA 源码；上游或 ABI 改变后重新构建和验证。

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
  `flash_attn` 必须来自固定上游源码构建的真实安装 wheel，`flash_attn_2_cuda` 必须是该
  wheel 中的真实二进制扩展。完整 RECORD、安装 payload、源码与构建记录逐项核对。
  本地源码构建 wheel 不称为 PyPI 发布的官方二进制。复用既有扩展时明确原构建身份，
  不称为本轮重新编译。按该解释器的 Torch/CUDA ABI 构建扩展；先检查依赖 pin，
  不让安装过程替换 oracle。
- oracle 环境清除 Jittor 的 source/shim 路径和库路径污染，配置自己的 `CUDA_HOME`
  与 `CXX`，让工具记录可核查的编译器版本。候选端记录实际 Jittor compiler 配置。
- 候选 C++ 扩展需要真实 pybind11 头文件；在私有候选解释器安装构建依赖（本轮实际
  `pybind11==2.13.6`），不要修改共享环境或拿参考端包目录补假安装元数据。
  来源证明核实实际 `cpp_extension.cfg()` 的 Python/CUDA/pybind include 字节、真实
  pybind distribution 的 METADATA/RECORD 和安装文件；缺头文件时立即报实际加载错误。
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

维护的 `nox -s ecosystem_flash_attention` 还要求
`JITTOR_FLASH_ATTN_SOURCE_PROOF=<external-source-configuration.json>`。该文件只给真实
产物路径，不填写假造的安装元数据：

```json
{
  "upstream": "<clean-pinned-official-checkout>",
  "wheel": "<actual-locally-built-flash-attn-wheel>",
  "oracle_build": "<original-native-build-manifest.json>",
  "torch_wheel_proof": "<complete-binary-torch-wheel-proof.json>"
}
```

原生构建 manifest 必须包含原先实际成功构建的源码、二进制 SHA256 和固定上游提交。
Torch wheel proof 包含实际 wheel 路径/哈希、实际安装前缀和完整 payload 哈希。
该固定协议同时校验已验证的本地 Flash wheel 完整 artifact SHA256；它不是 PyPI 发布的
Flash 二进制。同步重写 wheel、安装目录和 RECORD 也不能替换该 artifact；其他合法
构建需要审查 pin 并重新采集来源绑定证据。
当前固定 native 构建的独立参考 ABI 是 Linux CPython 3.11、官方 binary Torch
`2.11.0+cu128`；完整 wheel 哈希与 RECORD 在每个进程重新核验。更换 ABI 或 wheel
需要重新构建并验收，不能只更改配置中的版本字符串。
这些文件及所有安装字节都由每个实际 tensor 进程重新核验，不能事后补写旧 raw。
probe 的 `--source-proof` 与 collector 的同名选项使用该配置。候选端另核实真实
`jittor-torch` carrier 的 METADATA/RECORD、递归 facade 文件、实际生成的 C++/CUDA
桥接源码和二进制；API 兼容版本不替代 carrier 发行版本。

维护 session 自动执行全部十二格（六 API × fp16/bf16），每格 base/受控扰动各跑两个
独立解释器，随后自动复算全部三步数值和共同 scorecard。新记录使用严格的
`flash-attention-training` protocol，`axes=[]`，保留独立的完整七 API inventory。
缺真实同规格 CPU 对拍时，独立 CUDA L3 可通过，但共同结论必须为 blocked，CUDA
覆盖分子为零；不能修改普通 registry 安装证明规则来接纳这种不同实现。
历史 checkout-only raw 不会因新工具出现而被重新授级。
平台 NVIDIA 依赖另记实际首选 METADATA、真实已加载库路径/哈希及 CUDA runtime
版本；RECORD 存在则绑定，不存在则明确记录缺席，不冒称完整 NVIDIA wheel 安装证明。
这不放宽 Torch/FlashAttention 实现各自完整 RECORD 和 payload 的必需校验。
session 用已有公开 `jittor.compat.shim.deploy --target` 和 `--check` 在私有持久目录部署
候选 facade；只把该目录加入候选 Python 路径。它不写入共享解释器或参考端环境。
部署产生的 FlashAttention METADATA 按仓内 owned compatibility resource 逐字节核验，
明确标成兼容 API 元数据，不能当作安装了官方 wheel 的证据。

`JITTOR_FLASH_ATTN_EVIDENCE_DIR` 可给绝对外置输出根；各 case 目录独占创建，已有目录
显式拒绝，失败保留原始文件。未给定时 nox 在 scratch 之外创建持久输出，并在成功比较
全部 JSON/NPZ 后运行 collector。临时 pytest 用例仍自动清理临时输出。

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
  单独更新检查；另记录 `06ed62ad7` 的 12 个正式 CUDA 门禁场景与固定 causality 输入扰动。

## 维护门禁与统一 scorecard

`compat/tests/torch/test_ecosystem_flash_attention.py` 对六入口和两种低精度运行 12 个
场景。每个场景固定 head dimension 32、noncausal，分别在两个独立解释器执行原输入
与单独扰动 noise 的三步训练；其他输入哈希必须不变，两侧输出都必须实际改变。
要求 `JITTOR_TEST_DEVICES=cuda`、`JITTOR_REQUIRE_FLASH_ATTENTION=1`、
`JITTOR_REQUIRE_REAL_TORCH=1`，并配置前述上游路径及 `REAL_TORCH_PYTHON`、
`JITTOR_FLASH_ATTN_ORACLE_EXTENSION`、`JITTOR_FLASH_ATTN_ORACLE_CUDA_HOME`。
`JITTOR_FLASH_ATTN_EVIDENCE_DIR` 指定全新仓外目录；已有场景目录会被拒绝。
维护入口为 `nox -s ecosystem_flash_attention`；session 缺 CUDA、独立 oracle 或扩展路径
时会在执行前失败，并隔离候选与参考端环境。

[collect_scorecard.py](scripts/collect_scorecard.py) 重验原始比较、输入扰动、完整产物、
真实平台和上游源文件身份，再调用共享 `audit_coverage.scorecard`，不另写准入判据：

```bash
<jittor-python> agent/skills/flash-attention-torch-compat/scripts/collect_scorecard.py \
  --root "$PWD" --upstream <pinned-upstream-checkout> \
  --evidence-dir <immutable-gate-evidence> --output-dir <fresh-scorecard-directory> \
  --baseline <full-measured-jittor-sha>
```

`06ed62ad7` 的 12 条 CUDA L3 记录通过统一 schema，实际执行六项公开 API。
上游 FlashAttention 没有对应 CPU 执行，因此共享广度仍为 **0 / 7**；这与六项 CUDA
API 的实测执行范围分开报告，不放宽 CPU 阶梯要求。KV cache、速度及其他形状仍未验。

[独立安装来源与完整训练复验](../../../docs/results/2026-10-16-flash-attention-installed-source-training.md)
记录 `c33627faa` 的原十二格 / 48 worker，以及独立 host artifact guard
`be4827ef3`；两份 source/manifest 分开，CPU 缺失仍 blocked、CUDA 广度不增加。

Owner：native CUDA RNG/计算与 Torch 兼容维护者。上游版本、扩展 ABI、公开 facade、
Tensor 类型/梯度边界、RNG owner 或验证协议变化时重新检查证据。

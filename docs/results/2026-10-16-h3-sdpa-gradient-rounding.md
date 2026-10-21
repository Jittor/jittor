# H3 BF16 attention：保留原 1% 梯度合同

Status: 短序列 attention 原语通过原预算；完整 H3 前向通过，完整 AD 仍失败；不授予 H3 L2/L3、CPU 或 CUDA 广度。

Date: 2026-10-16。Owner: Torch/CUDA attention 维护者。
Baseline commit: `aa9e669f514432a3aae77840d1604458c1b9f3ff`，依赖前向批次
`67969d159e4f9033bac2ffc864a66a1b2a6a41c1`。本批先整合远端
`bc4dc0bc01a3930d3fad22326da0f9b8c70d444f`，再在验证终态后整合
`00ceb26fd395707fb33649cc43709af045110a9a`；正式测量只使用干净的 `aa9e669f5`。

Review when: 原生 attention/opmath、CuDNN frontend、Torch SDPA 路由、Function 的保存值语义、
Omni 私有 CUDA/Triton 运算、安装来源校验或固定上游版本变化。

## 问题与修复范围

前一批 CuDNN 前向消除了 H3 默认 Flash oracle 的前向差异，但四组真实 attention fixture 的
12 个输入梯度中仅 11 个满足原 scaled-relative `.01`：seed 19 refiner 的 dQ 为
`0.01818181818181818`。旧受保护 math 路径为 9/12，最大误差
`0.32462121212121214`。旧数据与报告保持原身份，不以本批结果替换。

独立二进制 Torch 在同卡上强制 CuDNN 也复现同一 dQ 差异。本批实际编译的 CuDNN frontend
图将 scale 放在 dS 的 BF16 舍入之前；外置逐阶段真实 CUDA 对照中，先舍入 BF16 dS、
在 FP32 Q/K GEMM 后缩放的组合能逐字节重现这组默认 Flash 梯度参考。
这项对照仅用于归因，完整 H3 oracle 始终请求默认 attention，没有改成 MATH 或 CuDNN。

新分支保留真实 CuDNN 前向，使用共享原生 attention 梯度组合：FP32 probabilities/dP，
从舍入后的输出计算 row term，dS 舍入到 BF16，Q/K 的 FP32 product 后缩放，再返回 BF16
梯度；dV 使用 BF16 舍入后的 probabilities。**这是 CuDNN 前向加原生组合反向，不是 fused
CuDNN backward，也没有性能授级。** 分流仅落在前一批已经限定的短 BF16、无 mask、
非 causal、零 dropout、非 GQA 准入域，没有增加新的 shape 准入。

默认原生路径保留原概率及梯度公式和运算顺序；独立 AST 折叠对照与实际 CPU/CUDA 回归均验证
这一点。非 BF16 路径和拒绝域继续原路线。原始输入通过独立可微 holder 保存，覆盖 q/k/v
分别重绑定及 sequence-major 布局。

## 真实 CUDA 结果

平台为 RTX 4090 D / sm89，driver 580.178.04。参考为真实 Torch `2.13.0+cu130`，
实载 CUDART `13000`、cuDNN `92000`；候选为 Jittor Torch facade，实载 CUDART `12020`、
cuDNN `92400`，实际 nvcc `12.2.140` / GCC `11.5`，frontend `1.30.0`。
候选的 Torch API/媒体 metadata 是兼容元数据，不是安装了真实 Torch 的声明。
两侧实际依赖栈分别绑定，没有宣称相同 CUDA/cuDNN 发行栈。

| 验证 | 实测结果 |
| --- | --- |
| 四组原 fixture，seed 7/19，sequence 2/6，head_dim 8 | 新路径 4 outputs + 12 input gradients 全部 byte equal；原 `.002/.01` 全过 |
| 原路径 before/after | 旧 math 9/12 → 新路径 12/12 gradient arrays；没有旧通过变成新失败 |
| 12 组边界，seed 7/19，head_dim 8/16/24，query 1/64，key 64 | 48/48 arrays 过原 `.002/.01`；46 byte equal |
| 边界非零误差 | seed19/d16/q64 的 dQ `0.002336448598130841`、dK `0.0005081300813008131`，均低于原 `.01`；其余 byte equal |
| 实际 provider | CuDNN `_forward` 执行、原生共享 gradient helper 执行，CuDNN `_backward` 不执行；真实 BF16 CUDA residency，fallback delta 0 |

原有 NumPy 边界测试的 `.02` 仍仅是 provider 合同，不能代替以上公开 `.01` 对照，也不能
代替完整 H3 的 `.002`。mask、causal、dropout、GQA、long sequence、single-key 拒绝域
保留原路线和原预算。

## 干净提交上的完整 H3

固定 Omni `446c2b5474dabffccf9a0034836eeab77c62b4b8`、vLLM
`98dff2a81d747d1dba01a47f939f48c3526d4206`。实际 `MiniMaxH3DiTModel` 保留
1 主干 / 1 refiner / hidden32、33268 scalars、36 parameters / 37 initial states、initial
seed 3、input seeds 7/19、原默认 attention。完整安装来源 guard 与固定 C6 source-built
Omni wheel 原样启用；没有把该 wheel 称为 PyPI 官方发行。

两侧 CUDA 未观测前向各产生 6 个数组，全部 byte equal，原
`max(abs(diff))/max(1e-8,max(abs(reference))) <= .002` 全过。37 个初始 state 与原始
参考逐字节相同。独立观测与未观测的每侧 6 outputs / 37 states 全相同，16 对内部
Q/K/V/output 全相同。实际 4 次调用分别命中真实 Flash 和真实 CuDNN layout 7；
参考 22 个、候选 92 个实际 mapped library/JIT 文件逐一重哈希。完整模型 43 个 state/output
pointer 记录验证真实 CUDA，候选 fallback count 0。

两侧 eval/train 的 4 个 AD worker 均写出各 14 个真实数组后严格 rc1。每个 seed 都有
36 个参数及 4 个输入请求梯度，却缺 **32/36 参数梯度和全部 4 输入梯度**；仅 4 个最终输出头
参数有梯度。8 行完整 expected/observed ledger 保留该失败，不能用部分 head gradients 或
相同的缺失状态授 L2/L3。CPU genuine 在默认模型构造时报告 `No CUDA GPUs are available`；
候选在真实 Gloo 可用性检查处拒绝。两侧 CPU installation/platform packet 均保留，
没有 CPU counterpart，不计 CUDA 广度。

## 回归与复核

最终私有安装修复后：core 322 passed / 42 skipped / 1 xfailed；CPU attention
19 passed / 2 CUDA-only skipped；新增真实 CUDA suite 8 passed / 0 skipped；structure +
compat structure + Omni/platform host 1701 passed / 4 skipped；Ruff、layout、diff check 通过。
同一生产数学文件此前另有 41 项 CUDA-configured native 邻近测试（包含显式 CPU HVP 检查）、
7 项兼容层和 3 项 FP32 CUDA owner 测试通过。默认 native 的 `.002/.01`、FP16/FP32、
mask/GQA、重绑定与 FP64 HVP 合同均未放宽。
单测在提交前冻结的相同文件字节上执行，其实际 HEAD/dirty 状态保存在原 command packet；
完整 H3 与最终生产对拍在 clean `aa9e669f5` 上执行，没有改写旧单测的提交身份。

此前 3 个结构失败单列保留：新 grad 实现缺 reference 登记，以及 2 个 nested xdist 子进程
未显式加载插件。窄补真实 CUDA/NumPy 测试登记并修外置 launcher 后，原失败节点 28 passed，
再跑完整结构通过。另一个裸 import-spec 问题来自本任务私有 compat/adapters 的 editable
安装方式，修复后裸 import 实际落到真实 Torch binary；与上述 3 个失败分别记录。
SHA 参数手误和 CPU 预期错误文本检查的两次控制面失败也保留，没有改写实际 worker。

维护回归使用本提交的 `compat/tests/torch/test_sdpa_bf16_backend_precision.py`。先按环境手册准备
当前源码的独立 candidate prefix、真实 CuDNN/frontend 和已部署 facade，再在独立目录运行：

```bash
h3_unit="${JITTOR_LAB_ROOT:?}/_state/h3-sdpa-recheck/unit"
mkdir -p "$h3_unit/home" "$h3_unit/jittor-home" "$h3_unit/tmp"
env HOME="$h3_unit/home" JITTOR_HOME="$h3_unit/jittor-home" TMPDIR="$h3_unit/tmp" \
  CUDA_VISIBLE_DEVICES="${H3_CUDA_VISIBLE_DEVICE:?}" \
  JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICE=cuda JITTOR_TEST_DEVICES=cuda \
  JT_BACKEND=cuda JITTOR_CUDA_WHEEL_STRICT=1 JT_CUDA_KERNEL_MATH=strict \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_PLUGINS=pytest_timeout,xdist.plugin \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="python:${H3_FACADE:?}:adapters" \
  "${H3_CANDIDATE_PYTHON:?}" -m pytest -q \
  compat/tests/torch/test_sdpa_bf16_backend_precision.py
```

原始证据**未版本化**，位于 `$JITTOR_LAB_ROOT/_state/omni-backward-precision/20261016/`。
`public-gradient-production-v1/`、`boundary-production-v1/`、`formal-forward-audit-v1.json`、
`formal-gradient-ledger-v1.json`、`formal-observer-audit-v1.json` 分别保存上述真实数值和独立复算。
完整 source/wheel/METADATA/RECORD/payload 校验绑定 1607 个 framework files；完整 Git seal
绑定 2912 个 mode/blob/content 条目，包括 symlink target 文本。

最终 `evidence-manifest-v1.json` SHA256：
`13c906638fe36a8e24884e2740241fdecf6595b56646133b527a5bd625f119d0`，
9352 artifact paths / 5,763,671,636 bytes。seal 不包含自身仍打开的 stdout log。
独立全量重哈希与脚本保存在
`$JITTOR_LAB_ROOT/_state/omni-backward-seal-review/20261016/`；复核时复制该脚本到新的外置目录，
输出写新目录，不覆盖封存结果。原 `67969d` 的 seal、源码与 raw 始终保持原身份。

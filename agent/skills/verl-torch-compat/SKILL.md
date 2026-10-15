---
name: verl-torch-compat
description: 在真实 VERL 注册算法组件上对拍 Jittor Torch shim 与独立 PyTorch。包含 CPU/CUDA 的 GAE+vanilla PPO、packed k3 蒸馏、完整梯度和三步 SGD 的可运行门禁，以及注册表分母和未验证训练入口的边界。
---

# VERL 组件的双解释器对拍

## 完整安装来源复验

最新来源绑定复验见[安装来源与完整组件训练](../../../docs/results/2026-10-15-verl-installed-source-training.md)。
真实测量固定 `73333a677dae984a3f577afb7d1586aecac5f490`；旧 `b367` 证据保持原身份。
同一干净上游的自构 wheel（不是官方 PyPI wheel）安装到双解释器：完整 360 Python 源文件、
413 wheel 包文件逐阶段绑定。源码/module `0.10.0.dev` 与实际 METADATA `0.10.0.dev0`
均原样保留；仅通过严格 PEP440 相等和完整 wheel proof 处理规范化，不修改版本字符串。

新的 opt-in `ecosystem_verl_training` 实际 CPU/CUDA 各 2 passed、0 skipped，包含下述两个
组件的种子 0/7/19、全部参数/输入梯度与原三步 SGD。共同 scorecard 为四行 L3，144 对数组，
没有 Ray/rollout/Hydra/full trainer 或性能授级。首次编译串行，unit/benchmark 缓存独立。

```bash
JITTOR_CI_PYTHON="$SHIM_PY" REAL_TORCH_PYTHON="$ORACLE_PY" \
  JITTOR_VERL_UPSTREAM="$VERL_CHECKOUT" JITTOR_VERL_WHEEL="$VERL_WHEEL" \
  JITTOR_ECOSYSTEM_PACKAGE_SITE="$VERL_SITE" \
  JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE="$VERL_SITE" \
  JITTOR_TEST_DEVICES=cpu python -m nox -s ecosystem_verl_training
# CUDA 用独立缓存、显式分配 CUDA_VISIBLE_DEVICES 和真实 nvcc_path。
```

实际节点为 `test_ecosystem_verl_training.py::EcosystemVERLTraining` 两个 `test_verl_*` 方法，
收集器为 runbook scripts 的 `collect_verl_scorecard.py`。它重新检查全部原始数组、实际
物理设备、完整来源与原阶段预算；不能只凭进程退出码或基础 seed 授级。缺依赖/来源证明
按 required 条件失败，不把 skip 当证据。新门禁不等于已经进入 nightly。

## 当前验收范围

历史协议：CPU/CUDA 的两个 tiny 组件协议已复验；报告见
[真实 VERL 注册组件证据](../../../docs/results/2026-10-15-verl-registered-component-oracles.md)。
上游固定 `e5a627cf55c94d85406556854c2c1a22db4ff5e7`（`0.10.0.dev`），
测量基线 `b367a25ff89b2496d4c34c57fe0a4b129b781dcd`。重新运行或换上游版本后重新生成分母和证据。

| CASES 入口 | 实际执行的上游注册项 | 输出与梯度 | 训练范围 |
| --- | --- | --- | --- |
| `verl_ppo_gae_vanilla` | `get_adv_estimator_fn("gae")`、`get_policy_loss_fn("vanilla")` | policy loss、全部 GAE advantages/returns、Linear 全部参数梯度、policy 输入梯度 | 三步 SGD，仅 `policy.weight/bias` |
| `verl_distillation_k3` | `get_distillation_loss_fn("k3")` | packed token k3 loss、student 全部参数梯度、features 与 teacher 输入梯度 | 三步 SGD，仅 `student.weight/bias` |

GAE 按上游实现保持 `no_grad`；它的数值和 returns 进入输出对拍，输入梯度来自实际
policy 支路。k3 的无公共 batch 维 TensorDict 是 packed 组件输入；没有运行 rollout、Ray
worker、Hydra training main、FSDP2、完整 PPO 或模型级蒸馏。ActorConfig 的 strategy 字段不是
这些路径执行过的证据。最高结论是所列场景的 L3 组件训练协议，不授予 L4/L5 或全库等级。

## 环境和依赖

当前上游声明 NumPy `>=2`、Transformers `>=5.5.3,!=5.6.0,<5.13`；使用独立环境，
不要升级既有 NumPy 1 / Transformers 4 的 `ecosystem` 会话。实测使用 NumPy 2.2.6、
Transformers 5.5.3、TensorDict 0.10.0、Ray 2.54.1；oracle 是真实 PyTorch 2.11.0+cu128。
两侧使用同一份干净 VERL checkout 和匹配版本的真实依赖。

先按[下游接入流程](../downstream-library-adaptation/SKILL.md)验证 oracle 身份、安装 pin、
源码 SHA 与离线条件，再裸跑。此次没有 adapter 或上游补丁。真实断点
`ProfilerActivity.__members__` 缺失归 `jittor.compat.torch` 的 profiler 枚举契约；
修复使用真实枚举成员、name/value 和动作身份，不代表所有枚举中的设备都能被 profile。

```bash
"$ORACLE_PY" -c 'import torch; assert hasattr(torch,"_C"); assert not hasattr(torch,"_torch_compat_install_context"); print(torch.__version__)'
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py --repo verl --list-only
python agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py \
  --repo verl --checkout "$VERL_CHECKOUT" --output "$STATE/verl-inventory.json"
```

## 门禁

专用 opt-in 会话使用预先配置好的两解释器，不安装或下载 VERL/模型。CPU 显式关闭 CUDA；
CUDA 必须显式分配设备和真实 nvcc。会话复用项目环境与 pytest gate，使用独立缓存，
不设微型协议的速度门槛。

```bash
JITTOR_CI_PYTHON="$SHIM_PY" REAL_TORCH_PYTHON="$ORACLE_PY" \
  JITTOR_TEST_DEVICES=cpu python -m nox -s ecosystem_verl
JITTOR_CI_PYTHON="$SHIM_PY" REAL_TORCH_PYTHON="$ORACLE_PY" \
  JITTOR_TEST_DEVICES=cuda CUDA_VISIBLE_DEVICES=<allocated-device> \
  nvcc_path="$CUDA_TOOLCHAIN/bin/nvcc" python -m nox -s ecosystem_verl
```

实际节点是 `compat/tests/torch/test_ecosystem_verl.py::EcosystemVERL`。两方法显式列出
oracle 与 candidate 均不能缺少的梯度键。共享 harness 拒绝非零退出、错误 case marker、
两侧键不一致、可广播但不同的 shape，以及任意非有限数组。专用 gate 缺依赖会失败；共享 harness 同时检查
显式 package site 和解释器实际路径，逐库 REQUIRE 只约束该缺失库。多 checkout 环境保留
各自的真实安装路径；可配置 `JITTOR_ECOSYSTEM_PACKAGE_SITE` 与独立的 reference site，
不得把缺依赖或 skipped 当作已验证。

## 完整证据

共享 runner 负责同权重同输入、实际 device、依赖版本、精度和零 fallback。基线输入之外，
至少再运行两组固定权重的输入扰动，并对拍全部输入/参数梯度。三步轨迹使用公共工具：

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 "$ORACLE_PY" agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case verl_distillation_k3 --runtime torch --device cpu \
  --output "$STATE/k3.train.ref.npz" --steps 3 --learning-rate 0.001 --seed 0
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 "$SHIM_PY" agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case verl_distillation_k3 --runtime jittor --device cpu \
  --weights "$STATE/k3.train.ref.weights.npz" \
  --output "$STATE/k3.train.shim.npz" --steps 3 --learning-rate 0.001 --seed 0
```

PPO 使用另一 CASES 名称，CUDA 在独立进程和缓存中复验。完整命令、原始数组和输入扰动的
权重哈希绑定见报告及外置证据 manifest。CUDA 实测 TF32 为 ON/high，CPU 为 OFF/highest；
两侧必须一致。没有 whole-process memory 或真实尺寸速度证据，不报告性能等级。

注册分母从干净上游自动生成：14 advantage estimators、12 policy losses、8 distillation
losses、5 Hydra entrypoints。此轮仅所列 `gae`、`vanilla`、`k3` 各计一项；entrypoint
分子保持 0。NPU/ROCm/Corex 本轮未验证；早期 NPU 结论仅按
[历史 Ascend 报告](../../../docs/results/2026-09-02-verl-ascend-core-algorithms.md)自己的版本和边界解读。

# CPU 与 CUDA 上的真实 VERL registry 组件

- 状态：复现了 tiny 组件正确性与三步训练轨迹；无 trainer 或性能等级。
- 测量基线：`b367a25ff89b2496d4c34c57fe0a4b129b781dcd`；代码含 profiler 枚举、必需梯度断言、真实多站点依赖发现、逐库缺失依赖强制以及 fail-closed 的 runner/数组校验。
- 上游：干净 VERL `e5a627cf55c94d85406556854c2c1a22db4ff5e7`，版本 `0.10.0.dev`。
- 维护者：Jittor 核心 / Torch 兼容层维护者。
- 复查条件：VERL registry/config/loss 布局、依赖版本、case builder、精度策略或设备后端变化。

本报告在独立的 PyTorch 与 Jittor 解释器中，用同一套 builder 执行固定上游源码中的
真实 registry 函数。没有任何 adapter、替换 registry、
mock、复制算法、下游源码补丁或模型下载参与测量。
源码 checkout 干净，其 runtime import 来源与 inventory 核对。

## 确切范围与 registry 覆盖

| Case | 实际上游组件 | 输出 / 参数梯度 / 输入梯度 | 三步可训练范围 |
| --- | --- | --- | --- |
| `verl_ppo_gae_vanilla` | `get_adv_estimator_fn("gae")` + `get_policy_loss_fn("vanilla")` | 49 个值（loss + 全部 advantages/returns），2 个梯度，`features` 梯度 | `policy.weight`、`policy.bias`；21 个数组 |
| `verl_distillation_k3` | `get_distillation_loss_fn("k3")` | masked 4×6 token loss，2 个 student 梯度，`features` 与 `teacher` 梯度 | `student.weight`、`student.bias`；24 个数组 |

PPO 使用 3→1 Linear policy、4×6×3 feature 张量与变化的 response-mask 长度。GAE 保留其
上游 `no_grad` 契约：rewards/values 被 detach，advantages 与 returns 进行数值
比较，完整 feature 梯度来自实际 policy 分支。这不表示
GAE 可微。packed k3 case 使用 28×3 student features 与 28×1 teacher logits，
由上游 packed-loss 路径转换为四个六 token 的 response。TensorDict 没有共同
batch 维，因为 packed token 与逐序列元数据的前导维不同。
这是组件协议，不是已验证的 trainer batch 契约。student 与 teacher 的输入
梯度都比较。`DistillationConfig(enabled=False)` 避免校验或启动蒸馏引擎；直接调用已登记的 k3 loss，并强制要求其非零 student/teacher 梯度。

scorecard 在每个 `verified_scenarios` 行中保留 case、完整 registry 引用、tiny shape 与可训练参数名。
该矩阵分别适用于 CPU 与 CUDA：

| 上游轴 | 已验证成员 | 已验证 / 自动分母 |
| --- | --- | --- |
| `advantage_estimator` | `gae`，在上述 GAE+vanilla policy 场景中 | 1 / 14 |
| `policy_loss` | `vanilla`，同一场景 | 1 / 12 |
| `distillation_loss` | `k3`，在 packed student/teacher 场景中 | 1 / 8 |
| Hydra `entrypoint` | 无 | 0 / 5 |

这些分母由所述 SHA 上的全部上游登记推导，而非来自 CASES。
同一函数的其他别名不会自动计入。已确立的最高级别为
命名 tiny 组件训练场景内的 L3。Ray worker、rollout、Hydra 训练入口、
实际 FSDP2 执行、完整 PPO、模型级蒸馏、NPU、ROCm 与 Corex 在此未验证。
ActorConfig 的 `strategy="fsdp2"` 值只是配置所选 loss；它不确立
分布式引擎的执行。由此不产生 L4/L5 或整库声明。

## 固定平台与协议

| 字段 | CPU | CUDA |
| --- | --- | --- |
| OS / Python | Linux 6.1, glibc 2.34 / Python 3.11.16 | 相同 |
| CPU / 加速器 | AMD EPYC 9554；加速器显式遮蔽 | 一张已分配的 NVIDIA RTX 4090 |
| 独立 oracle | PyTorch 2.11.0+cu128，真实二进制 `_C` | 相同，CUDA 12.8 |
| 候选 | Jittor 2.0.0 + 源码 Torch 兼容层 | 相同，真实 NVCC 12.2.140 / 驱动 580.178.04 |
| 依赖 | NumPy 2.2.6, Transformers 5.5.3, TensorDict 0.10.0, Ray 2.54.1 | 相同的真实包来源 |
| 精度 | TF32 OFF, matmul `highest`, cuDNN benchmark OFF | TF32 matmul/cuDNN ON, matmul `high`, benchmark OFF |
| 线程 / fallback | 两个 runtime 线程，affinity 一致；策略 `error`，计数 0 | 相同策略；观测到 CUDA 分发，计数 0 |

VERL 修订声明 NumPy >=2 与 Transformers >=5.5.3, !=5.6.0, <5.13。独立环境
遵守这些要求；已确立的 NumPy 1 / Transformers 4 生态 session 保持独立。
原始平台产物包含编译器输出、可见包元数据、真实 oracle 二进制来源、
已分配设备映射、实际精度、依赖版本与 runtime 条件。

输入离线，base seed 0。额外 seed 7 与 19 复用完整 base 参考权重；
输出必须改变，两个 runtime 必须产生每个被请求的参数/输入梯度。
三步协议使用公开 `training_probe.py`、seed 0、SGD 学习率 0.001 与新的固定
参考初始权重。步输入在两个 runtime 上以相同顺序使用确定性 seed 1、2、3；这比较的是更新轨迹，不是收敛。每步保存输出、标量 loss、完整梯度与更新后
参数；两个可训练参数都改变，且没有冻结参数。所有被比较数组
均有限。正确性容差为绝对 2e-3 / 相对 1e-2；下面的实际误差远小于此。

## 数值与门禁结果

| 设备 / case | 输出 + 全部梯度，seed 0/7/19 上最大绝对差 | 三步数组 / 最大绝对差 |
| --- | --- | --- |
| CPU / `verl_ppo_gae_vanilla` | 2.38418579e-07 | 21 / 2.38418579e-07 |
| CPU / `verl_distillation_k3` | 5.96046448e-07 | 24 / 7.15255737e-07 |
| CUDA / `verl_ppo_gae_vanilla` | 1.1920929e-07 | 21 / 2.38418579e-07 |
| CUDA / `verl_distillation_k3` | 2.98023224e-07 | 24 / 1.1920929e-07 |

专用 pytest 类在每台设备上实际执行两个 VERL 节点，并带显式必需的
梯度键。CPU 运行还执行两项 profiler 枚举契约；CUDA 运行另外
执行四项真实 profiler/allocator 检查。JUnit 证据拒绝任何被 skip 的 VERL 节点。测试
直接在完整准备好的解释器中运行；nox 包装 session 本身未执行。
该 nox session 的命令选择、必需标志、依赖版本预检、CPU 遮蔽与
显式 CUDA 分配由 host 契约覆盖。

- 公共 scorecard：四条被接受记录（两个组件 × CPU/CUDA），零 rejected；每条保留 tiny shape、完整 registry 引用与命名可训练范围。两台设备都产生上述矩阵。
- 实际专用对拍/profiler 节点：CPU **4 passed**，CUDA **8 passed**，零 skip。两次运行都设置 `JITTOR_REQUIRE_VERL=1`，要求独立真实 Torch oracle 与实际执行；保存的命令证据包含这些标志。JUnit 确认两个 VERL 测试方法都已执行。
- 共享依赖、runner 协议、数组与必需梯度契约（含四个既有 noisy-marker case）：**54 passed**。最终代码基线上的 CPU core：**322 passed, 42 skipped, 1 xfailed**。Lint、差异、布局与内部 Markdown 链接检查也通过。
- 一次负向审计使用最终记录/平台声明的副本，把 CUDA TF32 ON/high 虚标为 OFF/highest，并重新生成副本的平台产物哈希。两条 CUDA 记录以 `Platform/measurement precision mismatch` 被拒绝；两条未改动的 CPU 记录仍被接受。原始 raw 数组与测量元数据未变。

## 真实断点与修复

1. 上游 profiler 配置枚举 `torch.profiler.ProfilerActivity.__members__`。旧的
   兼容常量缺少枚举发现。修复属于 profiler facade，暴露
   匹配的名称/值与 ProfilerAction 身份；枚举不是对每个
   所列后端的 capable 声明。
2. 当 `PACKAGE_SITE` 存在时，共享 harness 将发现限制在 Transformers 目录，
   排除了其他源码 checkout/site 中的真实依赖。它现在搜索该显式 site 与
   解释器的实际 import 路径。缺失依赖只在自身 import 名称的
   REQUIRE 标志设置时报错；要求可用的 Transformers 不会因缺少可选 PEFT 而失败。
   专用 VERL 门禁另外拒绝任何缺失依赖，而不是跳过。
3. 共享比较现在要求两侧都声明梯度键，因此相同的遗漏
   不能产生一致。一个 host 负向测试覆盖仅 oracle、仅候选与共同遗漏。
   它还会在删除临时目录前关闭两个 NPZ 归档，避免 NFS 打开文件
   清理失败。

4. 共享 harness 现在在解析成功标记前拒绝非零进程退出，拒绝属于其他或缺失 case 的标记，并要求两侧张量键对称、shape 精确且值有限。修复之前，host 反例证明了非零退出、多余候选梯度、可广播 shape 与 NaN 梯度会造成虚假一致。两个 runtime 分支与畸形数组变体都被覆盖；数值阈值与数学未变。

## 复现与产物绑定

环境前置与 `nox -s ecosystem_verl` 见 [VERL runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/verl-torch-compat/SKILL.md)。
纯源码 inventory 与固定平台 scorecard 契约记录在
[下游 scorecard](../development/downstream-scorecards.md) 中。

```bash
python agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py \
  --repo verl --checkout "$VERL_CHECKOUT" --output "$STATE/verl-inventory.json"
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 "$SHIM_PY" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo verl --device cpu --oracle-python "$ORACLE_PY" --shim-python "$SHIM_PY" \
  --seed 0 --repeats 3 --no-memory --out "$STATE/cpu"
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 JITTOR_TORCH_SHIM=1 REAL_TORCH_PYTHON="$ORACLE_PY" JITTOR_TEST_DEVICES=cpu \
  JITTOR_REQUIRE_VERL=1 JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_TEST_REQUIRE_EXECUTION=1 "$SHIM_PY" -m pytest -q \
  compat/tests/torch/test_ecosystem_verl.py
python agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py scorecard \
  --upstream "verl=$VERL_CHECKOUT" --evidence "$STATE/scorecard-evidence.json" \
  --baseline b367a25ff89b2496d4c34c57fe0a4b129b781dcd --output "$STATE/scorecard.json"
```

此处 `STATE` 是 `$JITTOR_LAB_ROOT/_state/` 下的外部目录；每台设备使用自己的
编译缓存，unit 门禁与数值 probe 使用不同缓存。CUDA 在独立进程中重复同一协议，
使用显式分配的可见设备与真实工具链。manifest 保留两个扰动 seed 的确切命令、
每条公开三步训练命令、源码 inventory、驱动/源码哈希、raw 数组、平台与门禁 JUnit 文件。

外部运行根为 `$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/verl-components-20261015-final/`。
以下 SHA256 标识其最终证据；manifest 对每个单独 raw 产物取哈希。

| 外部产物 | SHA256 |
| --- | --- |
| `scorecard-evidence.json` | `bc15b70a10ab9950b3eae49beb76ef9b0f5b652f39aedd2142778b721501c0f6` |
| `scorecard.json` | `92ffe78afb57475d06cca47f58905792ad04392b63cf681da49a743ce00093fe` |
| `verl-inventory.json` | `83065316c7b8d7c8883876321d5a33d743bcdedb43181776efb8818c9eb9eba8` |
| `platform-cpu.json` | `3a2ddd74eab7dc872e601f5a3dc8e689327b248de7ac6cb098cefb03fbc391d4` |
| `platform-cuda.json` | `1d8145ee5775021e6d47e7582f17ca5e2c3b9b73f6d59c25775b51702c6f82f7` |
| `negative-precision-scorecard.json` | `0d75d3bab17e6368e334dae29a37833a6776713fbb1416a9ea8ad6a3262406ad` |

原始数据、日志、已安装依赖与编译器缓存保留在仓库之外。中间
原型、缓存引导失败与中断的基线单独归档，不是被接受的 scorecard 记录。
tiny 计时样本与被 skip 的显存检查不确立速度或整进程显存等级；
本报告不作性能声明。

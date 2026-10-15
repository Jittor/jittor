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

## 当前严格来源 scorecard

2026-10-16 在干净测量源码 `91242395f555f3c24f60df665c3ea66e13d01caf`，
真实 CPU→CUDA 双解释器原规格重新通过，TensorDict 每设备 **29 个数组、L3**，
包括 seed 0/7 完整参数/输入梯度及三步 SGD 的全部 21 个字段；两参数均真实更新。
分子为 **2/60 @ upstream c34264a58b12471dc3e2913ea586b32431872d4c**，仅 TensorDict 和实际
lazy_stack 组合，不扩大为完整容器、分布式或训练设施。CPU 最大绝对误差 9.54e-7，
CUDA 7.15e-7；原 .002/.01 预算不变，实际 TF32/benchmark 关闭、highest、F0。
与 Metrics 共四行被共享 auditor 实际接受、零拒绝。见
[严格官方来源报告](../../../docs/results/2026-10-16-metrics-tensordict-installed-source.md)。

`90ee` 原数组保留；当前 guard 拒绝其缺少固定官方 wheel/METADATA/RECORD 的记录，
不补造历史来源。新测量绑定 36 个上游 Python 文件及完整 42 个官方 wheel 包文件。
生成 `_version.py` 的源码声明为 setuptools-scm `version_file`；官方产物的真实
commit_id 为 **None**，不伪装成源码 SHA。固定 whole-wheel hash 同时保护生成文件与 `_C.so`；
载荷存在不代表其全部 C++ kernel 已执行，也不证明 hosted nightly 实际执行。

## 可复用完整 scorecard 采集

复用 TorchMetrics skill 的公共 wrapper/collector 和原 `_ecosystem_runner.py`，
repo 为 `tensordict`。wrapper 必须提供 `--wheel <fixed-official-tensordict-wheel>`；
collector 对应 `--wheel tensordict=<wheel>`。先获取官方 `v0.10.0` tag 并核对其 peeled commit；
只有 SHA、没有真实 release tag 的 checkout 如实枚举为 source@SHA，不能强行改成 release。

每侧恢复独立 HOME/JITTOR_HOME/TMPDIR/XDG_CACHE_HOME，CPU 显式隐藏 CUDA；
CUDA 两侧恢复真实 `JT_BUILD_NVCC_PATH`。base stem 为 `tensordict_index_and_stack`，
固定初始权重的 seed 7 扰动 stem 为 `tensordict_index_and_stack.perturb`。
再经同一 wrapper 执行 `training_probe.py --seed 0 --steps 3 --learning-rate .001`，
加载原 seed 0 完整 weights，保存 `.train.ref.npz/.json` 和 `.train.shim.npz/.json`。
collector 检查两个 trainable 参数、零 frozen、sample 梯度、完整轨迹和真实更新，再实际调用
共享 auditor。平台、安装 payload、实际 METADATA/RECORD、源码、权重或梯度缺失均拒绝；
原始命令、完整平台、负例和 seal 见新报告，不重标旧数组。

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

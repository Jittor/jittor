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

这是 **L3 组件协议证据**。TensorDict 尚无本工具自动生成的公共 registry 分母，不能计入
registry scorecard 或推断其他容器、分布式、编译、持久化与 dtype 支持。CASES 注册提供手动
公共入口；本轮没有新增 pytest 生态 CI 方法。

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
本次 CUDA 实际 TF32 为开启/high，CPU 为关闭/highest。未做 L4/L5、整进程显存或速度排名。

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

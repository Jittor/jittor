---
name: deepspeed-torch-compat
description: 复验源码钉死的 DeepSpeed 真实 Engine 训练、优化器分区及 checkpoint 续训；使用独立 binary Torch 对拍，按下游适配判据区分 native、compat 与库私有 adapter。
---

# DeepSpeed Engine 验证

## 范围

先遵守 [`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)。
目标版本为官方 `v0.19.7`，完整源码提交
`f5af15c2b4da10411e340850c00ba0cb5b14fd87`。adapter 检查源码版本和依赖的源文件
SHA256，未知版本、源码变更或导入后激活均拒绝。

正式入口是 `nox -s ecosystem_deepspeed`，对应
`compat/tests/torch/test_ecosystem_deepspeed.py`。它调用真实
`deepspeed.initialize` / `DeepSpeedEngine.backward` / `step` / `save_checkpoint` /
`load_checkpoint`，并在新 Engine 中恢复后继续训练。
普通模型 runner 的外部 SGD 不能代替这个入口；本门禁没有伪造共享 `CASES` 项。

当前单 rank FP32 矩阵为 ZeRO 0/1/2 × Adam/AdamW × 两组输入扰动，CPU 与 CUDA
分别执行。模型为确定权重的 residual Linear/GELU/LayerNorm/Linear，batch 2、hidden 16、
两个可微输入；六个 microstep、GAS 2、三次更新，第四个 microstep 后保存并恢复续训。
每个 microstep 通过真实 Tensor hook 验证全部参数的 incoming gradient；ZeRO 2 的
公开 full gradient getter 只在 GAS 边界使用。输出、loss、全部 input/parameter gradient、
更新后的参数及优化器矩状态/step 必须与独立 oracle 对齐。

**单 rank 不证明多 rank 分区或 NCCL 归约。** bf16、实际双 rank 训练/分区/恢复另行验收。
该 tiny 模型不授予性能级别，也不代表全部 DeepSpeed optimizer/model API。

## 环境与执行

将下游 checkout、wheel 和原始产物放在 `$JITTOR_LAB_ROOT/deepspeed/` 与
`$JITTOR_LAB_ROOT/_state/deepspeed/`，不写进主仓库。两端使用同一源码版本的真实
DeepSpeed 发行包；构建包时使用 binary Torch，不让 pip 换掉 oracle。
Jittor 端独立安装 `adapters/` 发行包，不能靠 source `sys.path` 假装 entry point 已安装。

准备两套 Python 环境后执行：

```bash
export JITTOR_CI_PYTHON=<jittor-python>
export REAL_TORCH_PYTHON=<independent-binary-torch-python>
export JITTOR_TEST_DEVICES=cpu
python -m nox -s ecosystem_deepspeed

export JITTOR_TEST_DEVICES=cuda
export CUDA_VISIBLE_DEVICES=<allocated-device>
export JT_BUILD_NVCC_PATH=<real-nvcc>
python -m nox -s ecosystem_deepspeed
```

分离安装的下游包可通过 `JITTOR_ECOSYSTEM_PACKAGE_SITE` 和
`JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE` 指定。独立 oracle 必须已经能导入 binary
Torch；门禁清除 Jittor 的 source/shim 路径，不向它注入本仓 Torch facade。
已有 oneDNN 安装可用 `JT_BUILD_MKL_INCLUDE_PATH` 与 `JT_BUILD_MKL_LIB_PATH` 显式指定，
避免首次源码下载；原生编译层验证这两个路径，不用关闭数学库来掩盖依赖问题。
会话强制 `JITTOR_REQUIRE_REAL_TORCH=1`、`JITTOR_REQUIRE_DEEPSPEED=1` 和实际执行，
缺库不能变成通过。CPU/CUDA 使用不同的编译缓存；第一次编译串行。

CUDA runner 在错误 fallback 策略下检查 count 为零，同时物化并检查输入、参数、输出
与梯度的真实 residency。输入、版本、TF32 策略和线程设置也要在两端一致。

## 归属与边界

| 断点 | 所有者与处理 |
| --- | --- |
| Module 实际注册 storage | native Module 单一 live view；compat 薄转发 |
| 叶子与非叶 Tensor hook | native 完整梯度形成后执行；不插 identity 节点改变叶子身份 |
| post-accumulate 与最终 callback | 复用 compat 已有累积/native 求导；FIFO 最终 callback |
| flat parameter 替换/重排 | 现有 optimizer state owner 按参数身份重绑定；复用 native Adam 算法 |
| DeepSpeed 按版本选择 private AccumulateGrad | 钉源码 adapter 选择真实 public post-accumulate API |
| DeepSpeed private worker-node 计数 | public `jittor.compat.torch.gradient_task.participating_tensors`，只在 post-accumulate 阶段查询实际 native 结果 |
| DeepCompile/elastic/CPU Torch-ABI shm 扩展 | 不伪造 Torch private module；明确拒绝前两者，shm 按 upstream capability 返回 None 后走真实 public collective 路径 |

post-accumulate observers 在全部现有叶子/optimizer storage 发布之后运行，跨叶子的调度
顺序不等同于 PyTorch worker engine，fidelity 为 approximate。公开参与集合不提供
worker graph ID，也不通过 stale `.grad` 推断当前图。高级 reentrant/compiled autograd
需单独验收。GradBucket/Adagrad 只允许导入，构造明确拒绝，不计作实现。

原生叶子闭包与 removable handle 的生命周期有真实回归；捕获前向 graph 的 hook
闭包循环在独立 PyTorch 中也能保留 graph，不承诺自动回收，应显式移除 handle。

Review when: DeepSpeed 六个钉死源码文件、native autograd/holder、Module 注册/序列化、
optimizer state、compat public `gradient_task.participating_tensors`、`owned_runtime_hook`、
`transaction` 或 `module_patcher` 改变。详细的维护者结论在
[`docs/results`](../../../docs/results/index.md)，原始日志不版本化。

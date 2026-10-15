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

**单 rank 不证明多 rank 分区或 NCCL 归约。** 双 rank BF16 使用独立入口
`nox -s ecosystem_deepspeed_multirank`，对应
`compat/tests/torch/test_ecosystem_deepspeed_multirank.py`：ZeRO 1/2 × AdamW ×
输入扰动 0/.125，两个 rank 使用不同输入且实际分区 FP32 master optimizer storage。
它复用原 Engine runner、数值判据与独立 oracle；不把 world-one FP32 CPU 记录
充作 world-two BF16 的 CPU 阶梯，也不计入缺少同规格 CPU 场景的加速覆盖分子。
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

双卡独立矩阵要求明确分配两张卡和新的仓外产物目录：

```bash
export JITTOR_TEST_DEVICES=cuda
export CUDA_VISIBLE_DEVICES=<allocated-device-a>,<allocated-device-b>
export JITTOR_DEEPSPEED_MULTIRANK_ROOT="$JITTOR_LAB_ROOT/_state/deepspeed/<new-run>"
python -m nox -s ecosystem_deepspeed_multirank
```

每格路径包含 stage、optimizer、precision、world size 和扰动；已有 case 目录直接
拒绝，成功或失败后的原始证据都不能覆盖。每 rank 有独立 HOME/JITTOR_HOME/TMPDIR；
首次 core/NCCL 扩展预热串行，之后同时启动真实训练 rank。可用
`JITTOR_DEEPSPEED_MULTIRANK_CACHE_ROOT` 指定本任务独占缓存根，不能与其他活动任务
或 benchmark 共用。任一 rank 失败或超时会终止并回收其同伴。
同一 case 的两 rank 共享 communicator rendezvous/heartbeat 目录；它属于控制面，
不能跟随 HOME/JITTOR_HOME/TMPDIR 按 rank 拆开。各 case 的控制面目录彼此隔离。
进程组使用每个 case/runtime 独占的 FileStore，单 rank 使用独占临时 FileStore；
不先释放一个空闲 TCP 端口再尝试监听，避免与其他任务抢端口。显式 CLI rendezvous
仍可供独立复现使用。FileStore 只负责 bootstrap，CUDA 归约仍须实际使用 NCCL。

`comparison.json` 记录原始 NPZ/log/protocol/source manifest 的哈希。门禁检查全量
incoming/full/input gradient，以及保存前后和新 Engine 续训后的 master、optimizer
moments/steps、输出、loss、梯度与模型参数逐位一致；不能仅用恢复后的输出近似通过。
源码 manifest 覆盖 native、compat、backends、adapters 与 nox，测量结束复核内容和
commit 未变。独立 oracle 的二进制身份、实际 optimizer 类、dtype、world size、分区大小
均在两端元数据中复核，CUDA fallback 必须为零。
逐个梯度字段记录实际 dtype 并跨运行时严格相等；`safe_get_full_grad` 在 ZeRO1
BF16_Optimizer 存有 `param.grad` 时可能直接返回 BF16，而 ZeRO2 的分区 getter
返回本规格的 FP32 accumulation storage，不能把两条上游路径都断言为 FP32。

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
[单 rank FP32 Engine 验收报告](../../../docs/results/2026-10-15-deepspeed-zero-engine.md)。
报告绑定原始计算基线、独立 CPU/CUDA oracle 与主线整合验证。双 rank BF16 的独立
源码及硬件边界见[多卡验证报告](../../../docs/results/2026-10-15-deepspeed-bfloat16-multirank.md)。
维护入口、CPU Adam 舍入边界、FileStore 控制面及最终 clean-source 四格复验见
[维护门禁验收报告](../../../docs/results/2026-10-15-deepspeed-maintained-gate.md)。
原始日志不版本化。

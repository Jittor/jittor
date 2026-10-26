# DeepSpeed Engine：有界 FP32 训练与 checkpoint 验收

- Status: 单 rank FP32 Engine 路径已验证；**R14 仍不完整**，尚待
  实际多 rank bf16 的分区/归约与 checkpoint/续训验收。
- Date: 2026-10-15。
- Baselines: owner 修复 `4bf21c9fe` / `a0c667538`，adapter/gate `67712827a`，
  同步后的树 `e082cf0a38aa195dc07c860231a454a097d02a68`
  （含远端 `2b16004e3`），加上本报告的提交。
- Owner: native autograd/Module、Torch 兼容与下游 adapter 维护者。
- Review when: native 梯度形成/holder 归属、Module 注册、
  optimizer 参数身份、DeepSpeed 固定源码，或 compat 公开的
  `gradient_task.participating_tensors`、`owned_runtime_hook`、`transaction` 或
  `module_patcher` 变化。

这是一个专用的真实 `DeepSpeedEngine` 门禁，而不是使用外部 optimizer 的普通模型
runner。官方 DeepSpeed `v0.19.7` 固定在
`f5af15c2b4da10411e340850c00ba0cb5b14fd87`，来自
[上游仓库](https://github.com/deepspeedai/DeepSpeed)。
两个解释器都使用同一份源码构建的
`deepspeed-0.19.7+f5af15c-py3-none-any.whl`，SHA256
`467622416c9819162ad853ea517805b5eb28e59ba3d43a3641add97668286c48`。
oracle 是真实的二进制 PyTorch 2.11.0+cu128，其 `_C` 扩展已校验，
不是 shim。adapter 准入检查源码版本与六个依赖文件
哈希；未知版本、被修改的源码与导入之后的激活都会 fail closed。

## 已验证协议

矩阵是 **CPU 上 12 / 12 个显式 Engine 配置，CUDA 上 12 / 12，@ 上游 `f5af15c2b4da10411e340850c00ba0cb5b14fd87`**：
ZeRO 0/1/2 × Adam/AdamW × 输入扰动 0/0.125。该分母是
已声明的 Engine 测试矩阵，而不是全部 DeepSpeed API 或模型的比例。
每个配置都在所请求的设备上运行两个独立解释器。

确定性残差模型使用 Linear(8,16)、GELU、Linear(16,8)、残差
相加、LayerNorm(8) 与 Linear(8,4)，权重相同且显式。
Batch 2、FP32、GAS 2 与 clipping 1.0 从六个 microstep 产生三次 optimizer 更新。
实际的 Engine 在第二次更新后调用 `backward`、`step`、`save_checkpoint` 与
`load_checkpoint` 到新 Engine。续训的 microstep 必须在每个 runtime 内
以 `rtol=2e-6`、`atol=2e-7` 复现未中断的输出/loss/参数轨迹，
包括全局 step 与 client 状态恢复。

全部八个参数的传入梯度与两个输入梯度在
每个 microstep 上都是必需的。累加的完整参数梯度也会比较；
ZeRO 2 只在 GAS 边界暴露其公开的完整梯度 getter。缺失
梯度、shape/schema 差异、非有限观测、残留的 Engine
图注册、optimizer 组、恢复的 moments/step、版本/设备或
执行条件不匹配都会失败。共享比较对
forward/state 与梯度使用分开的参考下限，归一化容差为 `2e-3`
与 `1e-2`；这些是正确性阈值，不是性能预算。

CUDA 在 error fallback 策略下物化并检查实际的输入/参数/输出/梯度驻留，
后端 fallback 尝试为零。CPU 隐藏所有 GPU。每个 runtime 初始化其声明的
Gloo/NCCL world-size-one 进程组；singleton 操作不测试与另一个 rank 的通信。
world-size-one ZeRO **不**证明跨 rank 的分区或归约。

## 修正与 adapter 边界

较早的 leaf-hook 反例已在 native 梯度 owner 中解决。
Hook 现在在 Var 的完整梯度形成之后运行，保留叶子身份，
校验替换的 shape/dtype/device，在参数数据重绑定后仍然有效，并
遵守可移除句柄的生命周期。独立 Torch 测试覆盖叶/非叶
目标、loss hook、被删除的中间包装、未使用的输入、重复
反向、失败与简单的闭合叶子闭包回收。捕获前向图的 hook 闭包
可能在该图在两个 runtime 中都保留；请显式移除其
句柄。这不是对一般图环回收的主张。

公开的 post-accumulate 入口复用既有的梯度/optimizer 存储。
其 fidelity 仍为近似：所有梯度存储在观察者运行之前就已发布，
因此跨叶子观察者的调度与 PyTorch worker 的交错不同。final callback
在累加之后按 FIFO 排空。公开的
参与查询只在 post-accumulate 执行内部报告实际当前的 native 反向参与者；
它不暴露伪造的 worker 图 ID，也不从过期的 `.grad` 推断参与者。optimizer
参数替换/重排会重新绑定既有的 Adam 状态 owner，而不新增另一个更新算法。
活跃的 native Module 注册视图仍是唯一的子节点存储 owner。

可重定位 adapter 只处理固定的库私有导入与选择：
DeepCompile 的 eager 导入被隔离，compile/elastic 请求显式
拒绝；hook 选择使用真实的公开 post-accumulate 行为，
库的私有 worker 节点计数使用公开的参与查询。一个 native Jittor 无法使用的
CPU Torch-ABI 共享内存扩展返回上游的
能力值 `None`，从而选择真实的公开 Gloo 通信。二进制 Torch 的
CPU 参考构建并运行其实际扩展。没有提供伪造的 Inductor/SymPy/私有
Torch 命名空间或 no-op 训练/通信实现。
`GradBucket` 与 `Adagrad` 仍可导入，但构造时显式拒绝；
它们不计为已实现的功能。

## 复现与回归

可复用的 [DeepSpeed runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/deepspeed-torch-compat/SKILL.md)
描述了隔离的包安装与专用 `nox -s
ecosystem_deepspeed` 入口。它要求独立的 oracle 与实际依赖
执行；缺失依赖不能变成验收跳过。这个专用
Engine 门禁不是在共享普通模型 `CASES` 计数中伪造的条目。
定期的 nightly 登记是另一项整合步骤；本报告
不声称它已自动运行。

平台：Linux x86_64、Python 3.11.16、NumPy 1.26.4、pytest 7.4.4、g++ 11.5，
RTX 4090/SM89、NVIDIA 驱动 580.178.04。native CUDA 使用实际 nvcc 12.2.140
与 pip-cu122；二进制 Torch 使用 cu128。TF32 与线程设置由
门禁匹配。其他设备与 Torch/DeepSpeed 版本尚未在此验收。

原始日志与数组未版本化，位于 lab 的
`_state/refactor-goal/deepspeed/`：

| 验证 | 结果 | 原始日志 |
| --- | --- | --- |
| 正式 CPU Engine 矩阵，两个解释器 | 12 passed，0 skipped | `cpu-formal-engine-gate-v1.log` |
| 正式 CUDA Engine 矩阵，两个解释器 | 12 passed，0 skipped | `cuda-formal-engine-gate-v1.log` |
| 实际专用 nox CPU CLI，独立 editable 安装 | 12 passed，0 skipped | `nox-engine-cpu-v3.log` |
| 实际专用 nox CUDA CLI，真实 pip-cu122 构建 | 12 passed，0 skipped | `nox-engine-cuda-v2.log` |
| Native/公开 owner 测试，CPU 与 CUDA | 42 passed，0 skipped | `cuda-hooks-owners-v2.log` |
| 独立二进制 Torch owner 契约 | 19 个方法在 CPU 通过；CUDA 19 个 | `oracle-owners-final-{cpu,cuda}.log` |
| 远端 initializer 修复后扩大的 CUDA optimizer/autograd | 54 passed，0 skipped | `cuda-optimizer-autograd-regression-after-sync.log` |
| hook 变更后的 CPU 官方 core | 322 passed, 42 skipped, 1 existing xfail | `cpu-core-after-hooks.log` |
| Adapter 源码/重定位/生命周期契约与既有 vLLM 契约 | 20 passed，0 skipped | `adapter-contract-v3.log` |
| 专用 nox 入口隔离/配置负向 | 6 passed，0 skipped | `nox-engine-negative-v3.log` |
| 既有 nox/环境与 pytest 契约，配共享 host recorder | 57 passed，0 skipped | `nox-contract-v2.log` |
| 同步 owner 基线处的完整结构 | 1457 passed，6 skipped；2 项共享文档登记失败 | `structure-final-v3.log` |
| 仓库布局 | 通过 | `layout-final-v3.log` |

其余两项结构失败是两份新 DeepSpeed 报告的文档可达性与
生成的打包 manifest 检查。它们的共享索引/manifest 登记属于
整合维护者；本报告不把那次未完整的扫描标为全绿。第一次扫描还
发现一个新门禁用例导入了另一个测试模块；其仅 host 的 recorder 被
移到 `_helpers`，重复扫描确认该契约已修复。

第一次隔离的 nox CPU 运行触发了缓慢的 oneDNN 源码下载；该会话
现在接受两个显式的既有 oneDNN 安装路径。其下一次运行正确地
拒绝了不匹配的执行条件：Jittor 有四个线程，而二进制
Torch 遵守继承的 128 线程 MKL 预算。该会话现在把
已声明的预算应用到每个共享线程池环境名。元数据检查
被保留，实际的 CPU CLI 随后通过全部十二个用例。失败与
中断的原始日志保留可用，而不是替换记录。

私有 CUDA 门禁环境最初在固定组件目标之前解析了较新的 CUDA 元数据，
并选择了系统 SDK。那次运行在出现任何矩阵结果之前
就被停止。优先使用私有的固定组件
目标后解析出 `pipcu122_a89fa4fed08c`；重复的实际 CUDA CLI 编译了
该 provider 并通过全部十二个用例。独立的二进制 oracle 包
与环境未变。oracle 此前不完整的头文件
从精确的官方 PyTorch 2.11.0+cu128 wheel 恢复，并按
其逐文件 RECORD 哈希校验；不主张整 wheel 摘要校验。

扩大的 CUDA 回归原本有 53 通过，1 项在冻结的
in-place initializer 上失败。独立的干净前置 hook 基线复现了它；远端
owner 修复 `236709174` 阻止随机初始化获得 grad。
在保留原始失败日志并同步该修复之后，全部 54
通过。该回归没有被跳过或削弱。

本报告未接受：bf16、双 rank ZeRO/NCCL、ZeRO 3、模型/张量/pipeline
并行、offload/融合 Adam/CPUAdam 扩展、compiled/reentrant autograd、
DeepCompile、弹性启动、真实规模性能，或整个 optimizer
与模型命名空间。下一次 R14 验收必须运行实际的多 rank bf16
训练、分区/归约与保存/新 Engine 续训，并对照二进制 Torch。

## 合并主树整合验证

整合树从 `0cddc76749e4f881d587a1f9ee865b9129464883` 开始，
包含上述 owner/Engine 提交、native ZeRO1 `0700dde342`，
以及 OpenMMLab `99ee01f7c8` / `2aa27ffe1c` / `cc7e09ff94`。
未提交的测试源码由
`$JITTOR_LAB_ROOT/_state/deepspeed-integration/combined-source.json` 绑定；包含本节
的提交记录整合后的源码。上面的原始数值
测量保留其原始基线。

| 整合验证 | 结果 | `_state/deepspeed-integration/` 下的原始日志 |
| --- | --- | --- |
| CPU core，native 与 Torch 模式 | 322 passed, 42 skipped, 1 existing xfail | `core-combined.log` |
| 梯度 hook、callback/参与查询、Module 注册、optimizer 重绑定与 BatchNorm 状态，CPU | 32 passed，零跳过 | `owners-cpu.log` |
| 同样的 owner 回归在实际 CUDA/pip-cu122 上 | 32 passed，零跳过 | `cuda/owners.log` |
| Native ZeRO 与分片 checkpoint 的 CPU 回归 | 45 passed，零跳过 | `native-zero-checkpoint.log` |
| 覆盖审计、必需库跳过策略与 OpenMMLab host 契约 | 61 passed，零跳过 | `host-combined.log` |
| 完整 Torch 模式结构门禁 | 1460 passed，6 skipped | `structure-combined.log` |

中心报告索引与生成的 manifest 现已登记，关闭了本次整合
较早的文档失败。六项结构跳过仍已披露：两项加速器检查、
两项已声明检查，以及两项因孤立解释器没有已安装 Jittor
发行版而无法执行的检查。CUDA 运行在上述同一 SM89 平台上
使用单独的编译缓存；并发作业排除了任何性能结论。最终的显式导入
再导出与这条整合说明不改变被测量的张量算法。
多 rank bf16 DeepSpeed 仍是另一项未完成的验收任务。

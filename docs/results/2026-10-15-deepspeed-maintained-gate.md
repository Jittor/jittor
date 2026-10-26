# DeepSpeed：维护中的双 rank BF16 Engine 门禁

- Status: 有界的真实 Engine 验收；整库覆盖仍待完成。
- Date: 2026-10-15
- Baseline: `f666c3f7190aeed39d7bb747dcf7f0753afaced8`（干净最终矩阵）；
  CPU 算术修正 `a441034299beb141785bbbffe7afaf03c99638c2`。
- Owner: Jittor 核心/兼容维护者。
- Review when: native Adam 状态算术、view/存储发布、autograd hook、
  optimizer 加载、native 集合通信/Store、runner/launcher，或固定版本的
  DeepSpeed adapter/源码变化。面向 adapter 的公开契约包括
  `gradient_task.participating_tensors`、`owned_runtime_hook`、`transaction` 与
  `module_patcher`；这些变化同样需要重新验证。

## 范围与协议

维护入口是 `nox -s ecosystem_deepspeed_multirank`，调用
`test_two_rank_bfloat16_engine_training_and_resume`。它在两个实际 CUDA rank 上运行
ZeRO 1/2 × AdamW × 输入扰动 0/.125。每个 rank 的输入不同；
实际的 FP32 主分区包含模型 332 个参数中的 166 个。这是在新 Engine 中执行
DeepSpeedEngine.backward/step/save_checkpoint/load_checkpoint，
六个 microstep、GAS 2、三次更新、在第二次更新后保存并续训两个 microstep。
这个极小的残差 Linear/GELU/LayerNorm/Linear 模型不是速度门禁。

两个 runtime 都使用官方 DeepSpeed v0.19.7 于
`f5af15c2b4da10411e340850c00ba0cb5b14fd87`；独立构建的源码 wheel 的
SHA256 为 `467622416c9819162ad853ea517805b5eb28e59ba3d43a3641add97668286c48`。
oracle 是二进制 Torch 2.11.0+cu128，带实际 `_C` 扩展且无 shim 标记。
adapter 仍单独安装，并固定上游源码哈希。没有普通模型 runner 或外部 SGD 替代 Engine。

数值比较复用原有的共享 harness：forward/state 容差 2e-3、全梯度容差 1e-2、
既有的类别下限 1e-3 乘以参考最大量级，以及最小缩放 1e-6。这些阈值未变。
每个传入参数梯度、两个输入梯度与公开可用的完整梯度都被检查。保存/恢复与
最终/续训的 optimizer 状态、主分区以及全部续训输出/loss/梯度/参数字段，
在每个 runtime 内逐位检查，独立于跨 runtime 的数值容差。

观测到的 dtype 很重要：ZeRO1 的 BF16_Optimizer 可能从 safe_get_full_grad 返回既有
的 BF16 param.grad；ZeRO2 的 partition getter 返回 FP32 累加存储。
原始元数据记录每个观测梯度的 dtype；两个 runtime 与续训字段必须一致。初始标志
不能替代张量驻留检查。所有请求的张量都在 CUDA 上物化，fallback 被禁止且计数为 0。

## 复现与产物安全

在 native 环境中安装经校验的源码包与独立 adapter；使用独立的二进制 Torch 解释器。
依赖隔离遵循
[runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/deepspeed-torch-compat/SKILL.md)。

```bash
export JITTOR_CI_PYTHON=<native-python>
export REAL_TORCH_PYTHON=<independent-binary-torch-python>
export JITTOR_TEST_DEVICES=cuda
export CUDA_VISIBLE_DEVICES=<allocated-device-a>,<allocated-device-b>
export JITTOR_DEEPSPEED_MULTIRANK_ROOT="$JITTOR_LAB_ROOT/_state/deepspeed/<fresh-run>"
python -m nox -s ecosystem_deepspeed_multirank
```

各用例在其路径中包含 stage/optimizer/精度/world size/扰动。预先存在的用例目录，
包括空目录，都会被拒绝。失败的证据保留。每个 runtime/rank 拥有私有的
HOME/JITTOR_HOME/TMPDIR/缓存目录。两个 native rank 共享其用例的通信器
heartbeat/control 目录；该目录不得与 rank 缓存混用。冷 core/NCCL 预热按顺序运行。
失败或超时的 rank 会终止并回收其活跃的对端。源码哈希与当前提交在每个用例前后检查；
comparison.json 绑定 NPZ/日志/协议/源码哈希。任何原始产物都不版本化。

## 验证

| 检查 | 结果 |
| --- | --- |
| 最终维护双 rank BF16 CUDA 矩阵，干净 `f666c3f719` | 4 passed，0 skipped |
| FileStore 修复后的维护 world-one FP32 CPU Engine 矩阵 | 12 passed，0 skipped |
| FileStore 修复后的维护 world-one FP32 CUDA Engine 矩阵 | 12 passed，0 skipped |
| Native Adam 状态/获取顺序回归 | 8 passed |
| Core，native 与 Torch 模式 | 322 passed, 42 skipped, 1 existing xfail |
| 完整结构加 DeepSpeed 协议/跳过策略测试 | 1,507 passed，6 skipped |
| 定向 launcher/nox host 契约 | 23 passed |
| 仓库布局 | 通过 |

完整的 CPU/CUDA world-one 门禁在算术与 FileStore 变更之后重跑。其源码是干净 `f666`
未改动的祖先；随后的远端合并加入 metric/TensorDict 门禁、清单与文档，
未改变 DeepSpeed runner 或 native Adam。最终四个 CUDA 单元在干净 `f666` 上运行。
热缓存四单元 pytest 运行耗时 83.84 s；这只是执行证据，不作比较性的速度或吞吐主张。

只读复验检查四个用例/两个 rank 上的全部 1,566 个受跟踪 runtime 源码哈希与 1,792 个
数值数组。全部 1,024 个梯度数组与独立 Torch 逐位一致。最大归一化 forward/state
误差为 1.7584651928227423e-7；在未变的比较器下梯度误差为零。每个 rank 都更新全部
166 个 FP32 主分区元素。Stage1 每 rank/runtime 检查 64 个严格续训字段，Stage2 检查
56 个，包括恢复的 optimizer 状态与继续训练路径。两个 native rank 报告的 fallback
计数均为 0。

未版本化原始证据位于
`$JITTOR_LAB_ROOT/_state/refactor-goal/deepspeed/multirank/`：

- `maintained-formal-v3/`：独占的原始用例目录、rank NPZ/日志、
  checkpoint、FileStore、协议/源码 manifest 与比较结果。
- `maintained-deepspeed-v3-evidence.json`：1,566 个源码哈希与 128 个证据
  哈希，包括全部 108 个最终原始用例文件；SHA256
  `425e8b405647c7aef75b88c28742598b43a60b87e508a690e829d8101b0da9df`。
- `maintained-nox-formal-v3.log`、`maintained-matrix-summary-v3.json` 与
  `maintained-matrix-audit-v3.log`：最终执行与重算的比较。
- `maintained-filestore-nox-{cpu,cuda}-v1.log`、
  `maintained-postsync-core-cpu-v1.log`、`maintained-filestore-structure-v1.log`、
  `maintained-filestore-host-v1.log` 与 `maintained-filestore-layout-v1.log`：
  相邻门禁。六项结构/环境跳过不构成张量执行的证明；实际 Engine 矩阵没有跳过。
- `upstream-protocol-inventory-v2.json`：有界的上游协议声明、
  官方源码身份与四个源码文件哈希。

代码依赖顺序为 `f4c1a00b7794`（维护门禁）、`5da8ee2410a5`
（共享控制面目录与观测到的 dtype）、`a441034299be`（native CPU
Adam 边界），然后是 `0f1bc0de239a`（独占 FileStore 引导）。算术
修正在其两文件提交中可独立审查。没有源码哈希或基线
被重标为后续的整合提交。

实际平台：Linux x86_64、RTX4090/SM89、NVIDIA 驱动 580.178.04、
Jittor pip-cu122 配 nvcc 12.2.140 与 GCC 11.5；独立 Torch 使用其 CUDA12.8
二进制发行版。Matmul/cuDNN TF32 启用，matmul precision 为 high，
cuDNN benchmark 为 false，每进程四个 CPU 线程。这不是 TF32 关闭的主张。

## CPU 状态舍入修正

整合到 `26be435076cfd24ff9d0611fbe55f3a04085c298`（包含 native
RNG 修复）后，独立重新哈希全部 128 个证据文件与 1,566 个源码哈希，
并重算四个原始 CUDA 比较。在合并后的 worktree 上，
CPU 与实际 CUDA 的 world-one Engine 门禁各通过全部 12 个用例且无
跳过；native Adam 的八项定向回归也通过。Core 通过 322
项检查，42 项跳过与 1 项既有预期失败。结构与工具
通过 1,707 项检查，含六项已声明/环境跳过。四个仅格式化的
后续修改保留其源码 AST。整合日志与复现
环境未版本化，位于
`$JITTOR_LAB_ROOT/_state/deepspeed-integration-26be/`。

第一次整合的 CPU 尝试在训练之前的独立 oracle 中失败，
因为其已安装的 Ninja 可执行文件不在 PATH 中。加入
既有的私有依赖 bin 目录后，单个严格续训用例与
全套 12 用例重跑通过。原始失败日志保留；该环境修复
没有改变任何算术或阈值。

未改动的维护前 runner 在独立的干净 `4a4a480e6` 上复现了
CPU ZeRO1 AdamW 参数 fresh-resume 发散：31/128 个 up.weight 元素，
最大绝对差 1.4901161e-8。保存/恢复的 master 与 moments、续训的前向/loss，
以及全部传入/完整/输入梯度都逐位相等。第三步 Adam 收到了
相同的参数、梯度、一阶/二阶 moments、标量超参数与
step 计数；最终 master 与 moments 相差一个 ulp。

一个不含 DeepSpeed 的确定性回归通过在返回参数之前获取更新后的
moments（而不是先获取参数）使这一点可观测。
干净基线的最终回归在 86/332 个 FP32 参数上不同，最大
2.9802322e-8。由 state view/AD 保留的中间输出允许 CPU 收缩
跨越不同的边界。只保留最终 moment 输出可修正旧有的
参数断言，但在新的全状态严格
检查下暴露 moment 差异；那次失败的尝试被保留，而不是作为验收呈现。

只有 CPU 的 torch_math=True 对 optimizer 梯度、每个状态
加数与更新后的 moments 使用显式的惰性舍入边界。共享的 native Adam 公式
拥有该变更；默认 native 数学与 CUDA 的既有路径保持其策略。
没有全图同步、全局编译器策略变更或 adapter 数学。
这些边界可能增加 CPU 物化；吞吐未测量。
八项回归覆盖 Adam/AdamW × true FP32/FP64 与 state/梯度获取顺序。
独立二进制 Torch 在 CPU 与 CUDA 上检查实际的同 dtype 算术；
跨 runtime 算术保持数值接近而非逐位相同。
true-FP64 补充 oracle 探针仍观察到至多约 2.19e-8 的
归一化残差，来自既有的标量系数精度策略；这些
回归断言同 runtime 舍入稳定性，而不是完整的 FP64 oracle 保真度。

## 边界与保留的失败

第一次维护尝试诚实地失败：两个 Stage1 oracle 单元暴露了
错误的“全完整梯度为 FP32”断言；两个 Stage2 native 单元使用分开的 heartbeat
rendezvous 目录，watchdog 中止。dtype 观测与共享
控制面设置修复了这些问题；watchdog 没有被禁用。
稍后的一次单 rank CUDA 尝试有 11 项数值通过，1 项独立 Torch
TCPStore EADDRINUSE 发生在训练之前；随后按 runtime 独占的 FileStore 引导
通过了两套完整的 12 单元 world-one 矩阵与最终的四单元 world-two 矩阵。
FileStore 是控制面引导；两个 CUDA rank 仍执行真实的 NCCL
训练集合通信。所有失败日志与较早的原始矩阵都与最终运行保持分离。

较早的 native 58 节点 view 运行没有全局启用 CUDA。其未设作用域的
分配节点在 CPU 上运行。完全启用 CUDA 的运行存在独立
基线复现的 KI-CODEGEN-005 全和分配失败；该门禁不
改变该阈值，也不声称分配检查在 CUDA 上通过。

world-one FP32 CPU 的 12 单元 Engine 门禁是独立证据。它不是
world-two BF16 的 CPU 阶梯。匹配的 world-two BF16 CPU 结果尚未
运行；当前维护中的
launcher 是 CUDA 专用的，这并不表明实际的 CPU API 不可能。
需要一项独立的同模型/权重/规格的 CPU API 探针。公共
记分卡准入仍待完成，该 CUDA 结果不授予加速器
覆盖分子。只读的上游清单把四个唯一 ZeRO
阶段（max_stage 是别名）、三个 offload 设备值与九个已声明的
DeepSpeed optimizer 名称记录为独立的有界轴。动态 torch.optim 查找
与用户 optimizer 实例/可调用对象使这九个名称的声明不能
被视为完整的 optimizer API 分母。该清单绑定完整的上游
SHA 与四个源码文件哈希；中心记分卡接线是另一项工作。

未验证：ZeRO3、CPU/NVMe offload、world-two BF16 中的其他 optimizer 类、
FP16 多 rank、多机、pipeline/tensor/expert 并行、DeepCompile、
弹性启动、性能、Windows CUDA、ROCm 或 NPU。既有的固定版本
adapter 仍显式拒绝不受支持的 compile/elastic 路径。四个极小的
CUDA 单元不构成整库支持。

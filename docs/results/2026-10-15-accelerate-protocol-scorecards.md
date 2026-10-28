# Accelerate checkpoint 协议记分卡

- Status: 独立的 CPU world-one 与 CUDA world-one/two 正确性已验证；
  共享准入接受 world-one 的 CPU/CUDA，而 CUDA world-two 仍因缺少匹配的 CPU FSDP2 协议
  而阻塞。
- Measured source: `b7e5a07b32f300e6f348a2147e7fafd5f4704c96`。
- Synchronized upstream Jittor baseline: `0751e425b0f197b8e7b388f218613378c71f701a`。
- Maintainer: Jittor 核心/兼容维护者。Reviewed: 2026-10-15。
- Recheck when: Accelerate、二进制 Torch、FSDP2、optimizer/checkpoint/RNG 归属、
  launcher、精度策略或协议校验变化。

## 范围与实际执行

维护中的 gate 在两个独立解释器中调用现有 Accelerate 测试。它增加观测与来源信息，
不改变 model 或 optimizer 的数学。这里 Accelerate 没有 model registry 分母：
记录显式使用 `record_kind=protocol`、`axes=[]`、`registry_refs=[]`。
这些结果不构成整个库的覆盖证明。

两个解释器都执行 Accelerate **1.10.1**，来自官方上游提交
`623cc0ba588d192c77758939a6e340adf07edd4f`（tag `v1.10.1`）。每个被测量的
进程都把全部 **83 个被跟踪的包文件** 与全部 **83 个已安装包文件** 绑定到 SHA256。
独立 oracle 是实际的二进制 Torch **2.11.0+cu128**，记录了其扩展路径/哈希与解释器 PID。
候选侧识别的是 worktree 的 Jittor frontend，而不是借用 oracle 的张量实现。

CPU 显式隐藏 CUDA，并把规范/legacy CUDA 标志置零。
机器报告 AMD EPYC 9554、Linux x86_64 与 GCC 11.5。CUDA 在两块 RTX 4090 上执行，
驱动 580.178.04；候选 CUDA 是固定版本的 12.2 wheel/工具链路径，NVCC 12.2.140。
oracle 的 CUDA 构建是 12.8。
已安装 wheel 元数据单独记录，不构成“每个已安装 CUDA 库都被加载”的主张。实际的
CUDA UUID、逻辑序号与张量指针设备在把观测复制到 host 数组之前被检查。
其他用户有并发的 GPU 负载，因此这些极小的正确性运行不作吞吐、延迟、内存峰值或
L4 结论。

每种精度/输入 seed 使用新的 launcher，串行暖机并使用各自的 rank 编译缓存。候选
world-two 运行使用公开的 `python -m jittor.distributed.launch --standalone --backend=nccl`
入口。它与 `pyproject.toml` 中注册为 `jtrun = "jittor.distributed.launch:main"`
的公开 owner 相同；这些测量保留实际的模块命令与 `[jtrun]` 日志，而不是声称执行过
某个控制台 argv。没有使用独立的 `mpirun` 训练 launcher。oracle rank 使用
`torch.distributed.run`；两个 CUDA runtime 都报告实际的 NCCL。

## 固定协议与独立结果

所有权重都被显式覆盖为维护中的确定性初始值。seed zero 提供基线；seed seven 用
NumPy 生成的高斯噪声扰动实际输入，同时保留初始权重。AdamW 有两组，学习率
0.01/0.015，betas (0.7, 0.93)，epsilon 0.0001，weight decay 0.02。world-one 另外
检查 StepLR。每个进程在保存前执行三次更新、两次预期更新与加载后的两次更新。所有
输入/输出/loss、完整输入/参数梯度、model 值、Adam moments/steps 与声明的超参数
都被捕获。两个续训步骤在每个 runtime 内都必须与预期步骤精确相等。Python、NumPy、
CPU Torch 与设备 Torch 的 RNG 流也必须精确恢复。

独立比较保留原始逐元素预算
`abs(candidate-reference) <= 2e-6 + 2e-5 * abs(reference)`。它比较每个捕获的
数组；没有新增 BF16 专用放宽。下面的归一化误差为
`max(abs(error)/(abs(reference)+0.1))`，与该预算一致。

| 实际协议 | 数组对 | 参数 / 输入梯度张量 | 最大绝对误差 | 最大归一化误差 | 独立深度 | 配对准入 |
| --- | ---: | --- | ---: | ---: | --- | --- |
| CPU world1 FP32，Linear 3→2 | 438 | 28/28; 14/14 | 5.960465e-8 | 1.774503e-7 | L3 | 接受 |
| CUDA world1 FP32，Linear 3→2 | 438 | 28/28; 14/14 | 2.980232e-8 | 1.986478e-7 | L3 | 接受 |
| CUDA world2 FSDP2 FULL FP32，Linear 4→3 | 820 | 56/56; 28/28 | 1.490116e-8 | 8.923406e-8 | L3 | 阻塞，level null |
| CUDA world2 FSDP2 FULL BF16，Linear 4→3 | 820 | 56/56; 28/28 | 1.490116e-8 | 6.489249e-8 | L3 | 阻塞，level null |

这些计数包含两个 seed 与每个 rank，包括重复的续训观测。两个 runtime 各自通过
十项 CUDA rank 测试与两项 CPU 测试：**24 个实际通过的测试**，没有跳过的必需节点。
所有候选记录回读到的 fallback 策略为 `error`，fallback 计数为零。

BF16 观察者在 Accelerate 的 FP32 输出包装之前于 Linear 内执行：每个进程的全部
七次前向都在 CUDA 上观察到实际的 BF16 输入、权重、bias 与输出。FP32 主参数、完整
梯度与 Adam moments 在 CUDA 上保持 FP32。native 与 Torch 的本地分片布局可能不同；
校验器检查每个 rank 拥有的元素数与完整的全局权重/bias 覆盖，而不是把不同的合法
布局视为相等。这是真实的 FSDP2 参数分片，而不是仅做 I/O 的 checkpoint 划分。

## CPU 配对与保留的失败

同源 CPU world-two 诊断在两个 rank 上初始化了真实的 Torch/Gloo，但 Accelerate
没有选择 FSDP2：两个 rank 都报告
`CPU world-two Accelerator did not use FSDP2`。MULTI_CPU/DDP 不能替代该协议。
另一个候选直接 init 诊断停在其缺失的动态 bootstrap；它不构成 native MPI provider
失败的证明，也没有测试完整的候选 CPU 双 rank 训练运行。

因此 CPU world1 只与 CUDA world1 配对。共享 auditor 实际验证全部四个测量行，
接受 world-one 配对，并把独立验证的 world-two 行放入 `blocked`，保留
`independent_verified_level=3`，但把准入的 `level` 与 `verified_level` 置为 null。
没有发明 registry 分子或分母。

更早的不可变证据仍位于 `formal-ee51` 与 `formal-297`。
第一次暖机使用基于尺寸的 `torch.ones(requires_grad=True)`，其候选输入梯度缺失，
而维护中的 checkpoint 测试显式使用 `torch.tensor` 输入并确实获得梯度。这是既有的
KI-COMPAT-004 边界；工具现在使用维护中的 tensor 工厂并保留显式的 leaf/梯度断言。
它没有修复也没有隐藏 ones 用例。

297 的暖机 worker 还把分配给 rank-one 的缓存重置为 rank zero。
两个 FP32 launcher 因此等待仍在编译的 rank 而超时。
修复传入显式的暖机缓存 rank，而不假装初始化分布式通信器。新的干净源码 CPU/CUDA
测量完全重新进行；较早的原始观测不重标。校验器在分布式保存的训练观测与其仅含
model/optimizer 的即时恢复快照之间的不匹配，通过精确比较这些恢复字段修复；所有
完整的续训与 RNG 检查保持严格且独立。

## 复现与原始绑定

使用带有匹配 Accelerate 与固定官方 checkout 的私有解释器；不要向共享环境安装任何
东西。从干净的测量源码出发，准备 CUDA 工具链/库环境，然后运行：

```bash
python tools/run_accelerate_gate.py --device cpu \
  --jittor-python <private-candidate-cpu-python> --oracle-python <private-binary-torch-python> \
  --upstream "$JITTOR_LAB_ROOT/accelerate-validation/upstream" \
  --output "$JITTOR_LAB_ROOT/_state/accelerate-protocol/<run>/cpu" \
  --cache-root <external-cpu-benchmark-cache>
python tools/run_accelerate_gate.py --device cuda --devices <gpu0>,<gpu1> \
  --jittor-python <private-candidate-cuda-python> --oracle-python <private-binary-torch-python> \
  --upstream "$JITTOR_LAB_ROOT/accelerate-validation/upstream" \
  --output "$JITTOR_LAB_ROOT/_state/accelerate-protocol/<run>/cuda" \
  --cache-root <external-cuda-benchmark-cache>
python agent/skills/torch-compat-repo-runbook/scripts/collect_accelerate_scorecards.py \
  --run "$JITTOR_LAB_ROOT/_state/accelerate-protocol/<run>/cpu" \
  --run "$JITTOR_LAB_ROOT/_state/accelerate-protocol/<run>/cuda" \
  --upstream "$JITTOR_LAB_ROOT/accelerate-validation/upstream" \
  --baseline b7e5a07b32f300e6f348a2147e7fafd5f4704c96 \
  --report docs/results/2026-10-15-accelerate-protocol-scorecards.md \
  --command '<exact CPU and CUDA measurement commands>' --output <external-scorecard-directory>
```

维护中的节点是 `test_optimizer_checkpoint_state.py` 中的 Accelerate 方法，以及
`test_accelerate_distributed_integration.py` 中的
`test_fsdp2_full_model_optimizer_checkpoint_resume[no|bf16]`。`nox -s ecosystem_accelerate`
转发 gate 参数；`benchmark_accelerate` 仍是独立的性能 gate。

未版本化证据位于
`$JITTOR_LAB_ROOT/_state/accelerate-protocol/20261015/`：
`formal-b7e5/{cpu,cuda}/execution.json`、逐 rank 的 JSON/NPZ/JUnit 与日志、
`formal-b7e5/cpu-provider/`、`scorecard-b7e5/`、完整的原始 SHA256 清单以及
`registry-backward-compatibility-b7e5.json`。后者把 **52 条历史 registry 记录**
与协议之前的 auditor 比较，完整输出一致。
`source-b7e5.bundle` 保留被测量的源码提交；其前置条件是已发布的
`0751e425b0f197b8e7b388f218613378c71f701a` 与
`282783a3dfcd3521400869beadf56c6f0fe9a0f9` 提交。`raw-bindings-b7e5.json`
绑定测量、实际命令、collector 输出、日志与本 bundle。
每个记分卡都会重算干净的上游来源与实际数值比较。历史的八 rank checkpoint JSON
不被重标或修复。

这验证的是恢复进同一 prepared model/optimizer 对象；新的 launcher 不能证明
新对象或新进程的 checkpoint 恢复。维护中的节点在同一个 worker 内保存、训练、加载
并续训。它没有独立的 producer/consumer 启动阶段：后续的 fresh-process 检查必须
持久化实际 checkpoint、终止 producer rank、在新的 consumer rank 中构造新的
model/optimizer 对象，并比较完整的续训轨迹。本批次不关闭该训练续训要求。
多机、SHARDED/DCP planner、runtime 之间的 checkpoint 互换、任意模型、完整
Accelerate 覆盖、ROCm 与 NPU 都不在本结果范围内。

## 仓库验证

定向的协议/registry 负向测试与子进程契约通过
**93 项测试**。CPU `tools/run_test_suite.py --tier core --backend cpu` 在
benchmark 进程终止后使用独立单测缓存，通过 **322 项测试**，跳过 42 项，
1 项预期失败。布局、Ruff 与空白检查通过。

最终的完整结构检查记录为 **1503 passed, 6 skipped, 2 failed**。
两项失败分别是该新报告在中心 results toctree 中的条目，以及为该报告/工具生成的
`MANIFEST.in` 条目；这些文件由整合维护者负责，故意留给整合处理。本孤立 worktree
不作完整结构全绿的声明。

## 主树整合验证

整合到同步后的 `d404c7027317c0200d648cb7b517bf578f9ec0f4`。
原始干净测量仍为 `b7e5a07b32f300e6f348a2147e7fafd5f4704c96`。
一项独立审计重新哈希全部 138 个封存的原始产物，检查十二个交付的源码/报告哈希，
并从归档交付与整合后的树中分别重跑 collector。两者都重算出全部 2,516 对数组，
给出两 accepted、两 blocked、零 rejected 的记录，且没有 registry 覆盖。
被阻塞的 world-two 记录尽管有独立 L3 数值一致，仍保留为 null 的准入层级。

整合后的树通过 111 个 scorecard/registry/Diffusers 工具契约、
实际 CPU checkpoint 契约（七个节点，零跳过）、实际 CUDA checkpoint 契约
（七个节点，零跳过）以及完整结构（1,505 passed，六 skipped）。
CPU checkpoint 与最终的协议工具运行合计通过 39 个节点。改动 Python 文件的 Ruff
与仓库布局通过。中心报告与 manifest 的登记解决了孤立 worktree 的两项结构错误。

整合还修正了可复用 runner，使每个 runtime/设备/rank 拥有自己的外部 `HOME`，
与既有的独立 `JITTOR_HOME` 和 `TMPDIR` 并列；暖机 rank 契约会验证这些目录。
原始 b7 runner 继承了 `HOME`，其封存观测没有捕获该值。因此其实际的历史 `HOME`
隔离未经验证，也不做追溯性填充。原始源码与数值身份保持不变；runner 修正是面向
后续测量的整合变更。

未版本化的独立与整合日志位于
`$JITTOR_LAB_ROOT/_state/accelerate-protocol/20261015/`，使用
`root-independent-*` 与 `root-main-*` 前缀。这些检查不扩展已记录的同对象恢复范围，
也不提供缺失的 CPU FSDP2 伙伴。

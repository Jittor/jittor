# 原生 DDP 真实训练与独立 PyTorch 对比（2026-10-12）

状态：两类真实数据 FP32 训练及单/双/四卡、梯度累积、实际 P2P 和持久状态恢复通过；小型 Transformer 性能仍有差距。维护者：native distributed 维护组。复查条件：Dataset、DDP、优化器状态、CUDA 数值策略或 NCCL 传输配置改变时，复验受影响的真实训练任务。首期范围保持单机 NCCL 数据并行。

## 基线与产物

本轮开始与恢复工作时均 fetch 目标分支，`HEAD` 与 `origin/cgq_transformers` 为
`e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`；fetch 同时记录的
`origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`。
实验在目标分支及本轮工作区修复上执行，关键源码 SHA256 为
`dataset.py: aab40248b1cccd063012859beba8e5941ac888c753f9b30075ab7bf4356195ab`、
`launch.py: 581e06e4ea0981eb621dbf6b10bb706b26dbd9b830d62f4f3b8a0e451c215366`。
未整合后一个 refactor tip，不能把本报告
解释为对该 tip 的兼容验收。原有 launcher P2P 策略修改与其他未提交工作均保留。

未版本化的脚本、资源与汇总位于
`$JITTOR_LAB_ROOT/jittor-native-multicard/real-training-20261012/`，下文称 `EXPERIMENT`；
原始日志、进程采样与编译缓存位于
`$JITTOR_LAB_ROOT/_state/native-ddp-real-training/20261012/`，下文称 `STATE`。
队列 JSON 记录每个任务的返回码；`execution.json` 记录启动命令、环境、脚本 hash
与工作区状态。`success.json` 只在 launcher 返回 0 且结果可解析之后生成，绑定
结果 hash；复用与汇总均要求此标记。早期成功任务按保留队列中的显式 `rc=0`
补标记，单凭结果文件存在不能记为成功。失败目录和相应队列快照保留。

## 拓扑和实际 P2P

所选四卡为 NVIDIA A800 80GB PCIe。rank 0 位于 NUMA 0，其余 rank 位于 NUMA 1；
rank 0 到其他卡的系统拓扑为 `SYS`，其余三卡之间为 `PXB/PIX`。具体物理映射
随实验记录保存在 `execution.json`，不把这组卡当作同 NUMA 四卡。

`nvidia-smi topo -p2p r/w/p` 能力矩阵为 `OK`；更进一步，所选四卡全部 12 个方向的
`cudaDeviceCanAccessPeer=1`，peer enable 成功，1 MiB `cudaMemcpyPeer` 逐字节检查通过。
证据为 `STATE/topology/{m,r,w,p}.txt` 与 `cuda_peer_copy.json`。

未设置 `NCCL_P2P_DISABLE` 和 `NCCL_P2P_LEVEL` 时，NCCL 对同 NUMA 链路选择
`P2P/CUMEM`，跨 NUMA 链路选择 `SHM/direct/direct`。另设
`NCCL_P2P_LEVEL=SYS` 的四 rank 原生 smoke 返回 0，报告
`NATIVE_NCCL_SMOKE_OK checks=20 nccl_version_code=22105 bf16_tested=True`；
实际已建立的通道有 14 条 `P2P/CUMEM` 日志、0 条 SHM 日志，包含跨 NUMA 通道。
证据为 `STATE/p2p-sys-smoke/`。这说明此拓扑允许实际 P2P，同时 NCCL 默认会按距离
选择传输方式；`SYS` 是专门的实验设置。

## 比较协议

Jittor 路径为原生 `distributed.launch`、NCCL ProcessGroup 与
`nn.parallel.DistributedDataParallel`，执行 `optimizer.backward(loss)` 和
`optimizer.step()`，禁用 MPI 构建，并断言没有导入 `compat.torch`。独立参考为
真实 CUDA PyTorch 2.6.0+cu124，使用 `torch.distributed.run` 和 PyTorch DDP。
两侧 CUDA 12.4、NCCL 2.21.5，FP32、关闭 TF32；Jittor 另设
`use_tensorcore=0` 和最高 FP32 matmul precision。卷积算法策略仍有差异：Torch
显式 `cudnn.benchmark=False`，原生卷积 backward 使用 benchmark 算法选择并未过滤
非确定性算法，因此速度与重复性比较包含这一配置差异，不宣称逐步轨迹或逐位等价。

两侧模型结构、NumPy 初值、数据划分、增强、全局 batch 和优化器超参数一致，
每 rank 使用两个 CPU data workers。原生 Dataset 按已完成 worker 通知读取批次，
可能改变批次交付顺序；完整 epoch 逐 rank 审计样本覆盖并记录顺序是否一致。
覆盖一致与顺序一致分别记录，不能因覆盖全部样本而声称每一步输入相同。

每 20 个 optimizer update 同步实际 CUDA device，计时取各 rank 最大 elapsed。
统计归约、打印及 block 边界 barrier 位于 block 计时区外，另记录 epoch wall。
首次 JIT 在独立预热中串行执行，预热结束后恢复同一初值并重建优化器。
预热与编译不计入稳定训练耗时；编译缓存与单测缓存隔离，两个框架顺序运行。

显存使用 `nvidia-smi` 采样本训练 rank PID 的进程显存；训练、预热、验证和
checkpoint 分开统计。恢复检查会同时持有两个模型，不能将其峰值计入训练显存。
采样请求间隔 0.5 秒，真实间隔与 rank 峰值、并发进程总峰值写入汇总 JSON。
进程采样峰值可能漏掉短暂分配，也不等同于框架 allocator 的精确峰值。

默认传输四卡训练和显式 `NCCL_P2P_LEVEL=SYS` 的训练分别记录；ResNet 先完成默认
策略和 N GPU/accumulation 矩阵，再做两侧三轮实际跨 NUMA P2P 对照，文本四卡任务
也显式使用 SYS。此配置留在实验 supervisor，launcher 公共默认仍由 NCCL 决定。

所选四卡在实验期间出现外部训练和推理任务，保留这些进程并记录其 PID、显存及
采样出现比例。PyTorch 空闲样本与共享负载样本分别保留；共享负载下按顺序执行
仍不保证两个测量窗口资源竞争完全相同，速度数字仅代表本机当时的观测。

## 模型和真实数据

ResNet-18 为 CIFAR 结构：3×3 stem、无 maxpool、`[2,2,2,2]` BasicBlocks，
11,173,962 个参数。CIFAR-10 train 50,000/test 10,000；确定性 crop/flip，
全局 batch 256、drop-last，每 epoch 49,920 图、195 update；两侧 SyncBN。
SGD lr 0.1、momentum 0.9、weight decay 5e-4，三轮恒定学习率。
初值 NPZ SHA256 为 `cc127cb99d45a6e865d2a2dac756f0337a847acc239e6f1ac0072a42e38445ab`；
CIFAR tar SHA256 为 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`。

文本模型为四层 causal decoder：hidden 256、8 heads、FFN 1024、sequence 128、
byte vocabulary 256，共 3,323,648 参数。pre-LN eps 1e-5、exact GELU、dropout 0，
两侧显式 QK/mask/softmax/AV attention，embedding 与输出权重不绑定。
AdamW lr 3e-4、weight decay 0.01、betas (0.9,0.999)、eps 1e-8。
真实 tiny Shakespeare 语料按连续 90/10 划分，每轮 12,800 个训练窗口，
全局 batch 64、200 update；验证 1,024 窗口、131,072 token。
语料 1,115,394 bytes，SHA256 为
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`，
下载来源为 `karpathy/char-rnn` 的 tinyshakespeare `input.txt`，按 hash 冻结。
初值 hash 为 `35841c66311a41fd4f5f2336a18a4311bc1c53407a97e189d0eb7caf5406bc2c`。
这是原生小型 Transformer 训练证据，不等同于大型预训练模型、AMP 或 fused attention 验收。

## 真实训练发现与修复

`dataset.py` 的 worker seed 来自 NumPy uint32，但原生 RNG 接口接受 signed int32；
部分 rank 首次启动 worker 即因溢出失败。现在把结果限制到非负 int32 范围，
rank 已进入 SeedSequence，不再通过 legacy MPI 再叠加一次。回归用真实双 worker
读取此前会溢出的 rank，并检查实际分片结果。

epoch reset 还复现了通知队列竞态：worker 已提交 batch，却在满 idqueue 的 push
上等待，父 reset 停止队列后，worker 将 `RuntimeError("stop")` 当错误退出，父
一直等待该 worker 报 idle。现在仅在 idqueue 与 batch buffer 都确实被停止时
处理这次 reset 通知，其他异常继续抛出；父 reset 等待期间轮询 worker traceback
和退出状态，真正死亡及时报告。用真实小 RingBuffer 强制第二个通知阻塞，修复前
超时，修复后返回 `RESET_OK`。另覆盖 reset 时 Python 异常、exit code 3，以及
提前结束迭代后连续三个新 epoch，避免只验证正常全量迭代。

## 保存与恢复检查

三轮 ResNet 训练最初在恢复一步的旧 `atol=1e-6` 判据上失败，最大参数差异约
`5.98e-6`，训练与验证记录仍完整保留在 `STATE/resnet-jittor4/`。没有将此失败目录
改为成功；最终全流程使用独立 `resnet-jittor4-final` 目的地。

对保留 checkpoint 做四 rank 独立短 control，`STATE/verify-resnet-resume-control4/`
返回 0，各 rank 保存自己的 JSON。源文件有 62 个参数与 40 个持久 BN buffer；
恢复前对模型状态、optimizer 张量/计数/超参数逐项检查 shape、dtype 和精确数值，
并确认 optimizer 的 params 引用新模型参数，全部通过。BN 还有 20 个非持久
`num_batches_tracked`，不包含在 native checkpoint；报告单独记录这些计数，
不把它们混入“持久状态精确恢复”的结论。当前固定 BN momentum 不依赖这些计数。

将同一模型恢复到同一份 NumPy 深拷贝的内存状态，重复同一个真实训练 batch，
两次重复的参数 max abs 分别为 `5.9884e-6`、`2.2426e-6`；新模型文件恢复的
参数 max abs 为 `2.2352e-6`，更新量 relative L2 为 `1.4164e-4`，小于本次
无文件重复测到的 `2.6701e-4`。差异在已平均梯度阶段已经出现；绕过卷积、BN 的
已知常数梯度 control 中，文件与内存恢复的平均梯度、momentum、最终参数逐位相同。
输入和 label 的 SHA、实际 CUDA device、每层误差均写入原始 control JSON。

这些结果支持本 checkpoint 状态恢复正确，真实 batch 续训存在计算波动。
源码中 cuDNN 卷积 backward 的算法选择没有提供严格确定性保证，但本实验未逐 kernel
归因，不能把全部差异直接称为某个 cuDNN kernel 的误差。最终训练 probe 先严格
检查持久模型和 optimizer 状态，再要求 Jittor 参数 max abs ≤1e-5（Torch ≤1e-6）、
参数 relative L2 <1e-5、更新量 relative L2 <1e-3，并检查恢复一步 loss。

## 结果

两组主比较都在四卡实际 P2P SYS 下完成，launcher 返回 0，持久状态/optimizer 恢复与一步续训检查通过；每轮副本最大参数差异均为 0。主/短任务、默认策略与负载补跑合计 22 个成功训练任务，另有预热和恢复 control；队列已全部退出，没有遗留本轮训练进程。

| 任务 | 框架 | 三轮 train loss | 最终 val loss | 最终验证指标 | 平均训练秒/轮 | 吞吐 | 最大 rank 训练显存采样峰值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet | PyTorch | 2.088477 → 1.523741 → 1.246076 | 1.315751 | accuracy 53.74% | 60.080 | 830.9 samples/s | 1.449 GiB |
| ResNet | Jittor | 2.081556 → 1.532326 → 1.298037 | 1.240946 | accuracy 55.90% | 60.097 | 830.7 samples/s | 1.414 GiB |
| Transformer | PyTorch | 2.693796 → 2.113439 → 1.803922 | 1.871896 | perplexity 6.500610; token accuracy 44.6518% | 10.494 | 156121.2 tokens/s | 0.990 GiB |
| Transformer | Jittor | 2.693797 → 2.113439 → 1.803923 | 1.871899 | perplexity 6.500629; token accuracy 44.6510% | 15.419 | 106256.5 tokens/s | 0.941 GiB |

ResNet 的两个主任务均在外部负载出现比例 100% 的窗口运行，平均时间为 60.080 与 60.097 秒/轮；该次测量近似相同。后两轮分别为 57.412 与 61.961 秒/轮，说明逐轮负载波动仍可改变比较。两侧 loss 总体下降、验证准确率上升；batch 交付顺序及卷积算法选择存在差异，不能据前三轮数值判定框架最终精度高低。两侧主任务 NCCL 通道均为 14 条 P2P/CUMEM、0 条 SHM/NET。

文本主比较使用外部进程采样比例均为 0 的 `text-torch4-idle` 与 `text-jittor4`。最初 `text-torch4` 仍有外部进程，因此保留为参考并单独补跑空闲 PyTorch。三轮平均 Jittor 耗时为 PyTorch 的 1.469 倍，吞吐约为其 68.1%；后两轮 14.733 对 10.436 秒，约 1.412 倍。两侧最终验证 loss 相差 2.86e-6，perplexity 分别约 6.500610/6.500629；这组曲线非常接近，但仍不称逐位相同。文本主任务都实际使用 P2P/CUMEM，训练样本覆盖与 rank 0 顺序审计全部通过。

默认 NCCL 传输策略的 ResNet 同样完整训练：`resnet-torch4-shared` 与 `resnet-jittor4-final` 为 60.278/50.819 秒/轮，峰值 1.449/2.100 GiB。空闲 PyTorch `resnet-torch4` 为 33.381 秒/轮。由于外部进程和调度窗口不同，默认策略与 SYS 的这些结果不能直接用于计算 P2P 固有加速比；首次预热与 allocator 保留状态也会改变不同运行的进程显存。

下面用相同全局 batch、前 60 update 的短检查比较 N GPU 功能和实测耗时。ResNet 的四卡行取单独 SYS 短任务；Transformer 的四卡 PyTorch 行取空闲补跑前 60 update。短任务不等同于三轮收敛验收。

| 任务 | GPU 数 | Jittor 前 60 update 秒 | PyTorch 前 60 update 秒 | Jittor 峰值 GiB/rank | PyTorch 峰值 GiB/rank |
| --- | --- | --- | --- | --- | --- |
| ResNet | 1 | 6.570 | 6.356 | 2.643 | 3.006 |
| ResNet | 2 | 9.744 | 6.248 | 1.854 | 2.201 |
| ResNet | 4 | 21.947 | 22.431 | 2.100 | 1.445 |
| Transformer | 1 | 1.793 | 1.386 | 1.283 | 1.713 |
| Transformer | 2 | 3.025 | 1.174 | 1.102 | 1.195 |
| Transformer | 4 | 6.569 | 3.413 | 0.941 | 0.990 |

单/双/四卡任务的输出均有限，已完成 launcher 清理及模型副本检查。固定全局 batch 的这些小模型没有呈现四卡线性加速：ResNet 四卡受跨 NUMA 和外部负载影响；无外部进程的文本模型也表现出多卡通信及 host 开销。当前 DDP 采用同步 NCCL group bucket（`defer_join=False`），仍在同步 backward 检查结构签名；本轮未做 profiler 归因，也未引入通信计算重叠，不能把时间差直接归因某一个环节。性能仍是后续工作。

两类模型两侧的 `accumulation=2` 各完成 20 update，原生使用 `no_sync()`，loss 按两次 microbatch 缩放；副本一致、持久状态与 SGD/AdamW 恢复检查均通过。ResNet 的短累积任务仅训练开头 20 update，验证 loss 仍高且波动明显（PyTorch 35.93、Jittor 10.61），因此只接受功能和恢复证据，不把这项记为收敛通过。Transformer 对应验证 loss 两侧约 3.3826。

四个主比较任务的实际显存采样中位间隔约 0.79–0.82 秒，超过请求的 0.5 秒；采样最大间隔约 1.05–1.52 秒。主比较的四 rank 并发训练进程显存总峰值：ResNet PyTorch/Jittor 为 5.770/5.633 GiB，Transformer 为 3.938/3.742 GiB。短暂分配可能漏采，数字描述的是进程采样峰值。

所有成功结果、逐 epoch 统计、checkpoint 数值界限和各 phase 显存位于 `EXPERIMENT/comparison/summary.json`；可导出的图为 `resnet_comparison.png/.svg` 和 `text_comparison.png/.svg`。图的 ResNet 使用 SYS 三轮主任务，文本使用空闲主任务；完整短任务数值保留在 `summary.md`，原始 rank 日志仍在 STATE。`final_verification.json` 校验 22 个成功标记、全部 epoch loss 有限、replica 为零及主任务实际 P2P，记录导出文件 hash；最早两个默认策略 PyTorch 任务尚未加入样本覆盖审计，已明确单列，四个主比较任务均有该审计。

## 回归与边界

Dataset 定向回归为 **32 passed、4 skipped、1 deselected**；跳过项包括 CUDA、MPI
与网络用例。两项 reset failure 定向复验为 **2 passed、12 deselected**。
CPU native core 为 **98 passed、41 skipped、1 xfailed**。日志分别为
`STATE/data-regression-final.log`、`reset-errors-final.log`、`core-native-final.log`。
最终 native DDP、ProcessGroup 和 launcher 定向回归为 **35 passed**，日志为
`STATE/distributed-regression-final.log`。

全 core 的 Torch 模式为 **217 passed、3 skipped、4 failed**：独立前端 Linear
反序列化无参构造、boolean mask RHS gradient、arange 默认 dtype、types.py 行数预算。
这些失败位于本轮未修改的 compat 文件，不能将全 core 写为通过。完整 structure
为 **1392 passed、6 skipped、2 failed**：开工时已有的未跟踪
`docs/development/2.0-refactor-onboarding.md` 缺 toctree，并导致生成 manifest 与当前
文件列表不一致。没有把该文档纳入本轮训练修复。最终文档、child-process 与测试布局
结构定向复验为 **33 passed、1 deselected**，只排除了已记录的 onboarding orphan
检查，日志为 `STATE/structure-affected-final.log`。布局与 `git diff --check` 通过。
原始日志为 `STATE/core-regression.log` 和
`structure-final.log`。

本轮证明的范围是实际 FP32 单机训练、普通 SGD/AdamW、Dataset 多 worker、多 epoch、
DDP 同步、评估及保存状态续训。多机、FSDP、弹性、分片 checkpoint 不在目标内。
收敛比较仅覆盖这两个真实数据任务的前三轮，不能据此声称最终精度、完整训练
性能、所有下游模型或 cuDNN 逐位确定性已经验收。

## 复现入口

在 `EXPERIMENT` 中使用本轮同一代码快照和已保留资源执行：

```bash
python supervise.py
python supervise_p2p.py
python supervise_text.py
python summarize.py
```

supervisor 只跳过已存在且成功的结果，遇到不完整目录会拒绝覆盖；重跑前显式换
任务名称或保留原目录后使用新目的地。四卡 P2P 专项任务传入
`NCCL_P2P_LEVEL=SYS`，默认任务清除调用环境中的 P2P 强制配置以便按默认策略比较。
复现原生路径需确保 `PYTHONPATH` 命中目标 checkout，并预加载环境所需系统
`libstdc++`；默认环境里的 CPU PyTorch 不能作为本 CUDA 对比的参考。

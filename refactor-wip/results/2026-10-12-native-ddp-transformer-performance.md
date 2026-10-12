# 原生 DDP Transformer 分桶与训练性能核查（2026-10-12）

状态：连续梯度块、结构检查和优化器覆盖修复已通过真实 CUDA/NCCL 与训练验证；
显式融合 AdamW 的四卡文本训练已接近独立 PyTorch，普通 AdamW 仍有差距。
维护者：native distributed 维护组。复查条件：DDP、梯度存储、优化器更新、CUDA
执行流、Dataset 或实验的数学/精度设置改变时，重跑受影响门禁。范围为单机 NCCL
数据并行，不增加多机、FSDP 或 Torch frontend 依赖。

## 基线与复现入口

开工先检查 dirty state、正在运行的测试并 fetch。目标
`origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`
已是本地 `661e98f728f33a14803f4e9978cf5cacb4e799e5` 的祖先，无新的目标分支提交
需要整合。本轮实现随本报告所在提交。fetch 同时记录
`origin/2.0-refactor@7a18abf295668d9b19da5fa1657f5606e84b65a0`，本轮未整合该 tip，
结果不等同于其兼容验收。用户原有未跟踪文档和 skill 保持独立。

以下位置未版本化；`LAB` 指仓库同级 `jittor-lab` 或实际配置的 `JITTOR_LAB_ROOT`：

- `PERF=$LAB/jittor-native-multicard/performance-20261012/`：固定输入诊断、顺序 supervisor、
  融合 AdamW 真实训练入口和实验流水线。
- `STATE=$LAB/_state/native-ddp-performance/20261012/`：每任务命令、源码 hash、rank 日志、
  GPU 采样、loss、checkpoint、完成记录。
- `REAL=$LAB/jittor-native-multicard/real-training-20261012/`：已冻结真实数据、同初值
  ResNet/decoder 及共用训练/恢复协议。
- `TEST_STATE=$LAB/_state/native-ddp-real-training/20261012/test-distributed/`：独立 CPU
  审核的解析测试和优化器回归日志。

可复用真实 NCCL 梯度门禁已沉淀到
[`verify_native_ddp_gradients.py`](../../agent/skills/jittor-distributed-verification/scripts/verify_native_ddp_gradients.py)，
运行方法见[分布式验证 skill](../../agent/skills/jittor-distributed-verification/SKILL.md)。
固定输入与真实训练实验分别使用 `PERF/run_probe.py` 和 `PERF/run_validation.py`；
每次指定新的结果目录，失败目录不能覆盖或改写成通过。原训练脚本也保留原基线。

实现的 SHA256 为：

| 文件 | SHA256 |
| --- | --- |
| `python/jittor/nn/parallel/distributed_data_parallel.py` | `17e86ce54fe1e00f6bc07215704ac7f06dfcf0b0eaf204b14ce63bdaf45e48d5` |
| `backends/cuda/kernels/misc/ddp_bucket.py` | `6f3162f45d91d216fbfd0e690ae9315130c60d8c8fa0d3bacf19205e63eca1b9` |

## 已证实的开销与 PyTorch 参考

旧默认 WORLD 的 `_backend_kind=None`，通过 `_get_backend_name()` 才能查询到 NCCL。
旧 DDP 只比较私有字段，因此真实训练没有进入源码中已有的 group 分支，而是每个
参数各做一次 mean all-reduce。此模型有 54 个参数张量，单步还额外做结构签名归约
并 `.numpy()` 读回；host profile 中五步共 275 次 all-reduce。这更正了
[前次报告](2026-10-12-native-ddp-real-training-comparison.md)关于 WORLD 已启用 bucket
的判断；不据此否定其参数一致性和训练数值证据。

按固定输入消融，仅取消每步设备到主机的签名 round trip，双卡每 20 步从约
0.878 秒降到约 0.640 秒；之后正确使用 group、连续打包和批量拆包继续减少通信和
host 构图。profiler 的 Python、编译查找和 launch 分项用于归因，profile 本身会
增加时间，不能混入稳态表。此轮未获取设备 timeline，不把 host profile 推论成
某个 GPU kernel 的确定瓶颈。

参考本机独立 PyTorch 2.6 的 `torch/nn/parallel/distributed.py` 中构造验证、
`_compute_bucket_assignment_by_size`、默认 25 MiB bucket 和 reducer 到梯度的映射。
PyTorch 的参数结构约束也要求在 DDP 构造后不增删/更换参数。Jittor 保留自己的
`optimizer.backward/step` 契约，通过本地检查代替每步跨 rank 签名读回，按 dtype、
设备和容量规划连续块；当前没有 PyTorch 的 autograd readiness hook、反向途中
通信重叠或持久 `gradient_as_bucket_view`。每块至多 64 个输入，单个超容量参数独立
通信；固定模型的 12.68 MiB FP32 梯度合并为一次 data all-reduce。

旧 group 分支还在 scope 内执行 mean 的除法 consumer，违反 NCCL group 的输出
使用契约。新实现 scope 内只同步 raw sum sinks，退出 group、join 之后才平均和
拆包。旧 WORLD 没有走该分支，因此这里不能称为前次 WORLD 训练已出现数值错误。
CUDA pack/unpack 在 compute stream 上执行 device copies，多输出拥有独立存储并
脱离梯度图；梯度 holder 身份保留，不在 group 内消费未提交的通信结果。

## 计时与训练协议

固定输入模型与真实数据模型均为 4 层 byte causal decoder：hidden 256、8 heads、
FFN 1024、sequence 128、vocab 256、3,323,648 参数。FP32、两侧 TF32 关闭，
显式 math attention、exact GELU、dropout 0；不换用低精度或 fused attention。
global batch 64，双卡 local 32、四卡 local 16。AdamW lr 3e-4、weight decay
0.01、betas (0.9,0.999)、eps 1e-8。普通与 `fused=True` 分开报告，两侧都显式融合
时才作为融合配置的公平对照；本轮没有改变 AdamW 的默认选择。

固定输入使用 4 个不同的预分配 slot，各 rank 输入不同；20 步预热后计时 3 个
20 步 block，保留每步 loss，真实执行全部 optimizer updates，block 末设备同步。
以最慢 rank 的每 block 时间取平均；末尾重新 backward 检查所有参数梯度有限且非零，
并检查各 rank 全部参数。native post_step 已清零旧梯度，不能拿清零后的 holder 做
“梯度非零”判据。保存的参数可作框架/实现间交叉数值核对。

真实训练复用 tiny Shakespeare 90/10 连续划分及冻结初值，双 worker、三轮，每轮
12,800 窗口、200 次 update、1,638,400 token；每 20 步记录 loss/accuracy/time，
验证 131,072 token。预热 20 步后恢复相同初值和新 optimizer，训练时间包括取 batch、
H2D、训练和设备统计，排除验证与 checkpoint 阶段；另外保存 epoch wall time。
完整精度、数据 hash 与保存恢复容差见前次真实训练报告。

GPU 为此前验证过 P2P 的四卡跨 NUMA 组合，本轮使用调用方显式的
`NCCL_P2P_LEVEL=SYS`，没有设置 `NCCL_P2P_DISABLE=1`。GPU 和首次 JIT 任务顺序执行，
CPU unittest 使用独立缓存。外部训练/推理进程持续存在，NVML 每 0.5 秒记录其 PID、
负载与本任务显存；共享机器的顺序测量不能保证完全相同的竞争强度，不能把不同日期
的秒数直接作为改动收益。第一组最终训练窗口中另有独立 CPU 回归的 oneDNN 编译；
随后反转框架执行顺序复测融合配置，等待该编译结束后才计入复测结论。

## 固定输入结果

| 配置 | Jittor 毫秒/步 | 独立 PyTorch 毫秒/步 | 结论 |
| --- | --- | --- | --- |
| 双卡普通 AdamW，旧 reducer 重测 | 42.96 | — | 同日旧实现基线 |
| 双卡普通 AdamW，最终 reducer | 27.71 | 23.86 | 对旧 reducer 约减少 35.5%；仍有普通更新开销 |
| 双卡两侧融合 AdamW | 20.76 | 21.07 | 同一量级 |
| 四卡普通 AdamW，最终 reducer，首轮 | 54.30 | 36.05 | 普通更新仍较慢 |
| 四卡两侧融合 AdamW，首轮 | 35.82 | 36.72–39.63 | 负载使 block 和测量窗口波动 |

编译干扰结束后保留旧 reducer 快照并再次按顺序测量，四卡普通 AdamW 为旧实现
163.75、最终实现 54.19 毫秒/步；两者执行相同的 80 次更新且副本一致。显式融合
AdamW 下，旧 reducer 为 131.13、最终为 29.82 毫秒/步。双卡与四卡均支持归因到
reducer 的额外同步/逐参数通信；四卡收益更大，但共享负载下这一对窗口不是独占
机器的稳态速度保证。原始目录为 `baseline4-final`、`final4-repeat`、
`baseline-fused4-final` 与 `final-fused4-repeat`。

双卡相同 80 次更新后的最终参数，对旧 reducer 的 max abs 为 1.8621e-5、relative
L2 为 1.5079e-6；融合 Jittor 对融合 PyTorch，双卡分别为 2.1606e-5/2.0155e-6，
四卡为 2.0931e-5/2.5179e-6。全部 gradient norm 有限且非零，各任务 replica error
均为 0；不要求不同实现的归约/更新路径逐位相同。不要把带 profile 多跑 5 次更新的
目录与无 profile 的参数作同一步比较。

## 真实训练与资源

最终普通配置的四卡三轮训练平均为 Jittor 14.007 秒/轮、PyTorch 10.618 秒/轮，
约 1.319 倍。最终融合配置第一组为 Jittor 9.469 秒/轮、PyTorch 9.385 秒/轮，
约 1.009 倍；这一结果必须结合随后反向顺序复测与共享负载边界阅读。

独立 CPU 编译结束后反向顺序复测，PyTorch 先运行、Jittor 后运行，仍为相同初值的
三轮完整文本训练和文件恢复。PyTorch 三轮为 9.338/9.394/9.310 秒，平均 9.347；
Jittor 为 10.109/9.521/9.423 秒，平均 9.685，约 1.036 倍。后两轮分别为
9.352/9.472 秒，约 1.013 倍。两组表明此配置实际训练耗时已在相近范围，不能据
0.9% 与 3.6% 区分更小的框架速度差。复测目录为 `text-fused-repeat-torch4` 和
`text-fused-repeat-jittor4`；所有 epoch replica error 0、状态恢复和一步续训通过。

两组实际训练的 NCCL 日志均有 14 条 P2P/CUMEM、0 条 SHM/NET。复测的训练显存采样
仍为 Jittor 944 MiB、PyTorch 1004 MiB。完整 loss/时间/显存图位于未版本化的
`PERF/plots/transformer-fused-training.png`，可导出的矢量版本为同名 `.svg`。

| 最终配置第一组 | 三轮训练秒 | 三轮 train loss | 最终 val loss | 最大 rank 训练显存采样值 |
| --- | --- | --- | --- | --- |
| Jittor 普通 AdamW | 14.371 / 13.819 / 13.832 | 2.693797 / 2.113439 / 1.803923 | 1.871899 | 962 MiB |
| PyTorch 普通 AdamW | 11.229 / 10.487 / 10.138 | 2.693796 / 2.113439 / 1.803922 | 1.871896 | 1016 MiB |
| Jittor 融合 AdamW | 9.844 / 9.518 / 9.046 | 2.693797 / 2.113439 / 1.803922 | 1.871900 | 944 MiB |
| PyTorch 融合 AdamW | 9.417 / 9.310 / 9.428 | 2.693797 / 2.113439 / 1.803922 | 1.871901 | 1004 MiB |

两侧数据覆盖与交付顺序均通过，epoch replica error 均为 0。模型持久状态和
optimizer tensor/count/hyperparameter 精确恢复，optimizer params 引用新模型；
Jittor 文件恢复后一更新的 max abs 在 1.49e-6–1.53e-6，更新量 relative L2 约
5.6e-5–6.1e-5，满足已声明恢复容差。最后一轮 token accuracy 约 44.651%，验证
perplexity 约 6.5006。NVML 数字仅为进程级采样峰值，不包含外部任务显存，可能
漏掉采样间隔内的瞬时峰值，不代替 allocator 的严格峰值。

ResNet 受 DDP 公共变化影响，另跑四卡真实 CIFAR-10 两轮各 20 update 的回归，
global batch 256、两次 microbatch 累积、原生 SyncBN 与 SGD momentum，双 worker。
两轮训练 11.359/11.164 秒，replica error 0，数据短跑与模型/optimizer 恢复通过，
恢复一步 max parameter abs 4.36e-7，训练采样最大 rank 为 1718 MiB。这是公共变化
的定向回归，完整三轮 ResNet 精度证据仍属于前次报告。

## 正确性和维护门禁

真实解析 CUDA/NCCL 四卡 40 步检查 rank 不同梯度、FP32/FP64 多桶、超容量参数、
optimizer 顺序不同、`no_sync`、unused/rank-only/empty 参数及 SGD momentum：
max gradient abs 1.52588e-6，参数符合 NumPy 独立期望；双卡未分组 mean reference
也通过。CUDA pack/unpack 五项真实测试覆盖 FP16/BF16/FP32/FP64、非连续输入、
空输出与全空 bucket，确认 shape/dtype、detach 和独立存储。低精度打包通过不等同
于低精度全训练验收。

单 rank 的参数同 metadata 替换、新增、删除、requires_grad 改变，以及 optimizer
漏参、全部移除、重复参数，七个独立 launcher 在约 4.1–4.3 秒非零退出，并在
rank 0 日志指出原因、停止同伴。本地结构检查不再让每个 rank 通过额外 collective
同步报错；错误处理依赖 launcher fail-stop。不能在异常后用 barrier 等待同伴。
构造时跨 rank 全结构验证保留；单卡和 `no_sync` 也执行本地结构/覆盖检查。

optimizer 的私有 reducer 登记解决“删除全部 DDP 参数后 grouped 为空绕过检查”。
新增参数组的 SGD/Adam/AdamW/RMSprop/Adan 都调用基类登记；正常 state_dict/pickle
不会序列化 wrapper/process group，新模型恢复不会绑定旧模型。先建 optimizer 后
包装 DDP 可在 backward/zero_grad 发现绑定；同 wrapper 多 optimizer 共享 pending
标记的更广泛协调没有作为本轮支持面。

| 门禁 | 最终结果/日志 |
| --- | --- |
| DDP + 基础 optimizer + save/load + zero_grad | 25 passed；`TEST_STATE/frozen-ddp-optimizer-final.log`，另一个 oracle 项该次为环境 error |
| 独立 PyTorch 2.6 AdamW 对照 | 1 passed、0 skipped；`TEST_STATE/frozen-adamw-venv-final.log`，已解决 oracle preload/SymPy 环境错误 |
| CUDA bucket | 5 passed、0 skipped；`STATE/gates/cuda-bucket.log` |
| 四卡原生 NCCL smoke | 每 rank 20 项通过；NCCL 2.21.5、BF16 collective 实测、P2P/CUMEM；`STATE/native-smoke-final4/` |
| native ProcessGroup / launcher | 28 passed；`STATE/gates/process-launch.log` |
| 最终 native core | 98 passed、41 skipped、1 xfailed；`STATE/gates/core-native-final.log` |
| 完整 core 两模式 | native 同上；Torch 217 passed、3 skipped、4 failed；`STATE/gates/core.log` |
| 完整 structure（Torch 模式） | 1392 passed、6 skipped、2 failed；`STATE/gates/structure.log` |
| 受影响 packaging / backend ownership / dispatch structure | 24 passed、1 deselected（已有 manifest 项）；`STATE/gates/affected-structure.log` |
| 最终 layout | 通过；`STATE/gates/layout-final.log` |

Torch core 的 4 项与前次相同：独立前端 Tensor 安装、boolean mask 写入梯度、arange
dtype、compat module 行数。structure 的 2 项仍由用户未跟踪 onboarding 文档缺
toctree、生成 manifest 缺该页引起；当前 manifest 差异仅这一个用户文档。没有为本
性能任务提交或改动它。此前若干诊断目录失败于 harness 或 JIT 源码构造，均保留失败
状态，成功结果取独立重跑目录。没有声称全仓测试通过。

## 验收边界与后续

在本次小型 decoder、FP32 math attention、global batch 64 和显式融合 AdamW 配置
下，原生 DDP 已达到接近 PyTorch 的训练耗时，并保留梯度正确性和持久状态恢复。
普通 AdamW 仍有可见 host 构图/更新成本；指南给出 `fused=True` 的显式配置，不把
融合实验当作未改训练配置的默认结果。未验证大模型、AMP、长序列、冷缓存完整训练、
图捕获中 DDP 或计算通信重叠；NPU/ROCm 硬件也不属于本轮结论。大型模型若通信
占主导，需要基于实际 timeline 再设计 readiness hooks/overlap，不能只打开
`defer_join` 就宣布已完成。

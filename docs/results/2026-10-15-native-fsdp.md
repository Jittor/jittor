# 原生 FSDP CUDA 训练与全新进程续跑

- 状态：已验证有界的 FP32 正确性；BF16 通过固定协议与精确续跑，
  但存在下文的小梯度数值限制。
- 被测源码：`ee24fac2be1e81ab6a2a14f32c869d6d17b600b1`，在最终独立 oracle 和
  全部三个启动器阶段中保持干净。
- 同步基线：远端 `cf8d2b6a961090c2b6ffd9f2139a6d21294108fc`，在源码提交之前
  集成为 `7277b26f0a30ec43423c2914cd648a6d39a025d6`。
- 日期：2026-10-15。维护者：Jittor 核心维护者。
- 复查时机：FSDP 存储、Module 引用、原生反向、checkpoint、集合通信、RNG、
  Linear/BF16 或精度策略归属发生变化时。
- 范围：CUDA/NCCL WORLD、两个真实 rank、固定 world size、原生 SGD/AdamW、
  FP32 master 分片以及 FP32 或 BF16 前向。这并未关闭每一项原生 FSDP 要求，
  也不确立性能或其他后端。

[实现说明](../notes/native-fsdp.md)描述了所有权和 API 限制。
原生执行既不导入 Torch 也不导入 `jittor.compat`；兼容 FSDP 把其既有的分片数学
委托给同一个原生所有者。

## Fixture、平台与设备证明

fixture 是一个真实的原生 `Linear(5, 7) -> tanh -> Dropout -> Linear(7, 3)`
模块，带一个持久 gain buffer、四个参数和 66 个完整参数元素。NumPy 提供固定权重以及
依赖 rank 的 `(3, 5)` 输入和 `(3, 3)` 目标。每个 rank 拥有 35 个 FP32 参数元素，
其中包括集合通信填充；checkpoint 区间只包含跨 rank 的 66 个有效元素。
SGD 使用 momentum 0.9；AdamW 使用 betas `(0.85, 0.95)` 和 epsilon `1e-8`。
两个参数组使用学习率 0.013/0.007 和衰减 0.02/0。

原生环境使用 Python 3.11.16、NumPy 1.26.4、CUDA 12.2.140、
NCCL 2.18.3 和两块 NVIDIA RTX 4090 设备。独立解释器在 CUDA 上使用真实的二进制
Torch 2.11.0+cu128，而非 shim。其元数据记录实际的 `_C` 二进制、版本和原始数据哈希。
Torch TF32 被禁用，matmul 精度为 `highest`；原生有效 matmul 与 cuDNN 级别都被断言
并记录为 `highest`。CUDA kernel 编译使用 strict 数学。

每个被观测的算术/集合通信输出都会同步，并且在请求其设备指针之前必须已驻留在 CUDA 上。
随后驱动指针属性验证实际的逻辑 CUDA 设备。前后观察器探针确立了：显式的 `.numpy()`
跟踪 I/O 可以把同一 Var 的存储移到主机，而不会改变 CUDA 分派。只有携带该 Var 自身
跟踪回读标签的集合通信输入才可能被再次上传；无法解释的主机驻留会失败。
这不授予 CPU 算子后备。所有最终轨迹都记录零后端后备且无前端导入。主机上既有无关的 GPU
占用意味着本次运行仅支持正确性结论。

每一步都观测实际的 Linear 输入和 gathered 权重 dtype。
BF16 运行使用 BF16 all-gather 和前向操作数、FP32 reduce-scatter 输出、
FP32 本地梯度、master 和 moment。前向策略不改变全局 autocast，也不复制优化器更新算法。

## 验证结果

| 检查 | 结果 | 边界 |
| --- | --- | --- |
| 原生 FSDP + 邻近 TP + ZeRO CPU 测试 | 51 通过，零跳过 | 干净被测源码；18 项 FSDP 契约 |
| CPU core，原生与 Torch 模式 | 322 通过、42 跳过、1 项预期失败 | 提交前运行时字节；显式禁用 CUDA |
| 完整结构门禁 | 1472 通过、4 跳过 | 最终跟踪断言之前的同步运行时；零失败 |
| 原生 CUDA 对等 | 每 rank 7 通过，零跳过 | 四条优化器/精度轨迹和三项分布式反例 |
| 原生 CUDA 保存 + 在线续跑 | 每 rank 4 通过，零跳过 | 三步、checkpoint、带真实 dropout 的再两步 |
| 全新启动器恢复 | 每 rank 4 通过，零跳过 | 独立进程组；最后两步精确相等 |
| 兼容 FSDP CPU 所有权/扁平策略/数学 | 24 通过 | 既有邻近 CPU 覆盖 |
| 既有兼容 FSDP CUDA 套件 | 每 rank 6 通过、1 跳过、2 失败 | 两处失败在这些改动之前也可复现 |
| Ruff、空白和仓库布局 | 通过 | 自有文件；无中央注册表编辑 |

完整结构运行包含一个已声明的依赖加速器文件，其收集用例为零，另有四项已声明/后端
跳过；这不是加速器验收。兼容 CUDA 跳过要求至少四个 rank，不在本次双 rank 运行的
覆盖之内。

独立 oracle 为五步计算稠密双 rank 平均 loss、每个参数梯度、各 rank 特定的输入梯度、
输出、loss、更新后的参数和优化器 moment。最终产物审计还对照该 oracle 检查确切的期望
键集合。四份参考包含 456 个数组；
24 个候选 NPZ 文件在三个阶段中共包含 1,968 个数组。

保存/恢复 fixture 启用概率 0.25 的真实 dropout 并记录其输入和输出。每个阶段都验证
存在非平凡的丢弃值。新的启动器重建模型和优化器对象、加载 checkpoint 并执行第四步和
第五步。全部 **368** 个结果数组都与不间断训练精确相等，包括 dropout 输出、输入和
参数梯度、参数、moment 以及每 rank 的 Python、NumPy、原生 CPU 和 CUDA RNG 流。
RNG 流在各 rank 之间不同，因此在每个 rank 上复现 rank zero 会失败。
这比在同一进程中加载到另一个模型更强。
带插桩的阶段在两个 rank 上共执行 384 次 all-gather 和 384 次 reduce-scatter 本地调用。

checkpoint 清单描述真实的模型/moment 分片区间；gain buffer 保持复制。
模型/优化器/布局没有被包进整树的 `RankLocal`。分布式反例拒绝不一致的初始参数、
优化器控件以及在一个 rank 上被改动的复制 gain buffer。
CPU 反例覆盖陈旧/部分优化器、图/生命周期误用、非法布局以及以失败而非跳过方式处理的
required-CUDA。

## 固定数值协议与 BF16 限制

所有比较都要求精确形状和有限值。FP32 的逐元素容差为 `atol=2e-6, rtol=2e-5`，
BF16 为 `atol=4e-4, rtol=3e-2`。每参数步长增量另外要求 L2 误差至多
`2e-6 + 2e-5 * oracle_norm`（FP32）或 `1e-6 + 0.03 * oracle_norm`（BF16）。
不大于十个绝对下限的 oracle 更新会被判为无信息而拒绝。
这些阈值在共享 Linear 数值修复之前就已固定，且在该比较中必须保持不变。

每个实际参数/步都注入一个零梯度与一个零更新，并要求被拒绝：
**160 个梯度反例和 160 个更新反例**，各在 FP32 与 BF16 之间均分。
两项宿主测试另外仅把一个参数置零而保持另一个正确，并拒绝小到无法与绝对下限区分的更新。
所有逐步范数、误差、下限和预算都予以保留。

| 优化器 / 计算 | 最小 oracle 更新 L2 范数 | 最大绝对下限占比 | 最大更新相对 L2 误差 | 最大梯度相对 L2 误差 |
| --- | ---: | ---: | ---: | ---: |
| SGD / FP32 | 1.41889e-4 | 1.410% | 0 | 1.542e-7 |
| SGD / BF16 | 1.41218e-4 | 0.7082% | 0.5229% | 0.8657% |
| AdamW / FP32 | 4.45890e-3 | 0.04486% | 1.121e-6 | 1.104e-6 |
| AdamW / BF16 | 4.46419e-3 | 0.02241% | 0.5804% | **11.2242%** |

BF16 AdamW 的最大值出现在 rank zero 的 `step/4/gradient/fc1.bias`：
oracle L2 范数 `0.001057407702319324`，差异 L2 范数
`0.00011868552246596664`。其较小的坐标通过所述的逐元素绝对容差。
这**不是精确的梯度一致**，尽管门禁通过且同一运行时的续跑精确一致。
BF16 AdamW 参数的最大绝对差异约为 `2.7663e-4`。
在把 BF16 数值一致视为最终结论之前，必须用这些相同的输入和阈值重放共享的
Linear/Lt 精度调查。

## 既有兼容性失败与保留的对照

原始兼容套件在 `test_mesh_reordered_shards_custom_adam_and_lifetime` 中未能通过其
即时的旧分片弱生命周期断言，此时数值检查已经通过。
`test_nested_sharding_and_full_state_reload` 用一个原生 Module 到达
`set_model_state_dict`，并因原生 `load_state_dict` 不接受 Torch 的 `strict` 关键字而抛出。

在临时反转本批次的**全部九处受追踪的 runtime/test/skill 改动**、只保留同步后的改动前
运行时时，两处失败在两个 CUDA rank 上都得到复现。同样的两个节点在两个 rank 上都失败，
日志位于 `logs/compat-prechange`。重新应用保存的补丁后恢复了 FSDP 工作。
它们在该源码中仍是被识别出的既有失败；后续修复必须保留改动前的证据，
并验证共享 Module/加载所有者。

更早的一次结构运行有五个失败：降级改动引入的一个重复 helper、两个缺少 xdist 插件的
环境失败，以及两处 TP 中央文档/清单登记。该 helper 现在改为委托，环境显式加载 xdist，
同步远端登记后消除了后两项。上文的后续完整结构运行已全绿。新的 checkpoint 反例中
此前的断言 API 与正则错误保留在失败日志中，并在不改变原生张量 API 或 checkpoint 校验
的前提下得到修正。

## 复现与产物绑定

使用[分布式验证 skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/SKILL.md)，
配备独立原生与二进制 Torch 解释器、私有的每 rank 缓存和串行首次编译。
CPU 验证显式隐藏 CUDA。可复用的 oracle 工具为：

```bash
"$TORCH_PYTHON" agent/skills/jittor-distributed-verification/native_fsdp_oracle.py \
  --source-root "$PWD" --device cuda --kind sgd --dtype float32 \
  --output-dir "$JITTOR_LAB_ROOT/_state/native-fsdp/oracle"
```

生成两个优化器和两种 dtype 的结果。原生既有测试节点为
`tests/backends/comm/nccl/test_native_fsdp_nccl.py`。设置
`JITTOR_REQUIRE_NATIVE_FSDP=1`、`JITTOR_TEST_REQUIRE_EXECUTION=1`、
`JITTOR_NATIVE_FSDP_TEST_DIR` 和 `JITTOR_NATIVE_FSDP_ORACLE_DIR`。
在两块分配的 GPU 上以三次不同的调用运行原生启动器，先
`JITTOR_NATIVE_FSDP_PHASE=parity`，然后 `baseline`，最后 `resume`；
后两个阶段选择 `-k test_oracle_and_fresh_launcher_resume`。每个子进程在导入 Jittor 之前
必须设置其私有的 `HOME`、`TMPDIR` 和 `JITTOR_HOME`。

原始数据和确切的 rank 选择器/环境位于外部
`$JITTOR_LAB_ROOT/_state/native-fsdp/20261015/`。最终产物目录为
`artifacts/clean-ee24fac2b`；`SOURCE.json` 绑定全部 22 个改动源码文件。
`AUDIT.json` 独立复核完整键集合、数值指标、零注入、分片角色、dtype/设备元数据和
精确续跑。`SHA256SUMS.json` 绑定 236 个产物文件、13 份日志和五个 runner/环境文件。
其 SHA-256 为：

```text
9901d1ed7dd834dae4d11b4afdb501a37d2cdc3bc692019daad2d6a0c27c9a27
```

更早的 `formal-dirty-v3`、观察器探针、失败日志和改动前对照单独保留。
最终原始张量数据没有为修正元数据而被重写。
未来的运行必须创建新的产物目录和源码绑定。

## 主集成检查

在 `2c0d2681a7db089aa2e996f0ad8e15057fb8e35e` 之上的集成重新哈希了全部
236 个产物、13 份日志、五个 runner/环境文件和 22 文件被测源码绑定。
二十个文件与该源码完全一致。两处小的集成改动影响宿主逻辑：优化器受管参数标识只合并
一次，而不是为每个参数重建；FSDP checkpoint 的张量列表长度在编码/解码可能用 `zip`
截断它们之前就被检查。没有改变任何分片、梯度归约或优化器更新数学。

列表长度缺陷用六个真实畸形 checkpoint 复现，覆盖普通模型与分片模型的梯度、一阶矩和
二阶矩。修复之前六个都被错误接受。修复之后，六次加载都在改动目标参数、标志、优化器
学习率/步数或原生 CPU RNG 之前抛出。另有三例畸形 FSDP 保存在没有完整清单时也被拒绝。
上文的原始 CUDA 证据仍绑定到 `ee24fac2b`；它不会被重新标注为本次宿主校验改动的新设备
测量。

主集成 CPU 所有者测试无跳过通过 78 个用例。九项损坏检查单独通过（其中六项与那 78 项
重叠）。CPU core 通过 322 项测试，42 跳过和一项既有 xfail；邻近的 FSDP 数学/扁平策略/
结构检查通过 24 项测试。完整 Torch 模式结构通过 1470 项测试，六项跳过。
相对于更早报告多出的两项跳过，是在没有安装 Jittor 分发的隔离解释器中进行的包解析检查。
文档索引、生成的清单、布局和 diff 检查也已完成。原始集成证据未纳入版本控制，位于
`$JITTOR_LAB_ROOT/_state/native-fsdp-integration/`，包括 `artifact-audit.json`、
`source-equivalence.json`、`extra-list-before.log`、`extra-list-after.log`、
`cpu-owners.log`、`core.log`、`compat-owners.log` 和 `structure.log`。

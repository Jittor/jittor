# TorchMetrics 与 TensorDict 组件 scorecard

- 状态：已测量 CPU/CUDA 组件；共享审计器的结果见下文。
- 评审日期：2026-10-15。
- 维护者：Torch 兼容性与下游验证维护者。
- 同步基线：`ab006244a34c550a5f7235b1f65ec7d4165bde9d`。
- 干净测量源码：`90ee18a9916acda9b5e86ca665f09a231a41ced1`。
- 复查时机：既有用例构建器、Module/张量梯度、索引、包版本/源码标识、runner/训练探针、精度策略、RNG 播种、设备放置或后备归属发生变化时。

这是对既有已注册用例的一次新执行，而不是对历史 `5ad8a1929f2e4a88db39b4f0e570e4901615487d`
数组的重新标注。此前的[组件报告](2026-10-15-metric-state-and-tensordict-oracles.md)保留其
原始基线。本批次未改动任何框架、RNG、adapter、中央清单或共享审计器源码。

## 确切范围与分母

每次收集都会从干净的 release 检出重新推导清单。TorchMetrics 1.7.4 为
`986eb900b8bc660ff8174c95ec8f702e0e5cb6ca`，位于
`git@github.com:Lightning-AI/torchmetrics.git`；其完整顶层 `__all__` 维度有 105 个成员。
TensorDict 0.10.0 为 `c34264a58b12471dc3e2913ea586b32431872d4c`，位于
`git@github.com:pytorch/tensordict.git`；其顶层维度有 60 个成员。这些是有界的导出分母，
不是全部 metric 变体、方法、分布式设施或整库支持百分比。

| 既有用例 | 实际 API 与规格 | CPU | CUDA |
| --- | --- | --- | --- |
| `torchmetrics_regression` | MeanAbsoluteError + MeanSquaredError，FP32 预测/目标 `(4, 8)`；输出 + 两个输入梯度；零参数 | L2：6 个比较数组，精确 | L2：6 个比较数组，精确 |
| `tensordict_index_and_stack` | TensorDict + lazy_stack，batch `[2,3]`，FP32 样本 `(2,3,4)`，整型/bool 索引，惰性 stack/expand 和 Linear(4,3)；两个参数梯度 + 输入梯度 | L3：29 个比较数组，最大绝对误差 9.54e-7 | L3：29 个比较数组，最大绝对误差 7.15e-7 |

Metrics 仅为 L2，因为没有可训练参数。TensorDict 另加恰好三次学习率 0.001 的显式 SGD
步骤：每一步的输出、loss、两个参数梯度、样本梯度和两个更新后的参数（21 个数组），
包含真实的参数更新。这证明所声明的 tiny 组件组合，而不是 Adam、上游 Trainer、
分布式同步、持久化或编译支持。采样为核心 0、broad 4 个场景/设备行、tail 0。
每个设备的一个组件场景引用同样这两个 API 成员；无关导出不获得分子。
共享审计器接受了全部四行，零拒绝：**CPU 和 CUDA 上各有 2/105 个 Metrics API 达到 L2**，
以及 **CPU 和 CUDA 上各有 2/60 个 TensorDict API 达到 L3**。每个设备上其余 103/58 个
导出仍未验证；导出维度之外的 API 方法和变体不属于这些分数。

## 执行与标识

两个任务自有的 Python 3.11 解释器运行共享的 `verify_repo.py`、`_ecosystem_runner.py`
和 `training_probe.py`。oracle 是真实的二进制 Torch 2.11.0+cu128，`_C` SHA256 为
`981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`；它显式拒绝
shim 标识。两侧都加载真实的 TorchMetrics 1.7.4 和 TensorDict 0.10.0。
真实的 SciPy 1.13.1 依赖目录在私有环境中优先；共享安装保持不变。

新增的 `component_measure.py` 包装器在不替换其计算的前提下运行既有探针。
在每个实际测量解释器内部，它校验每个上游包的 Python 源码（341 个 Metrics 文件、
36 个 TensorDict 文件），记录除字节码缓存外的全部已安装包文件（346 个 Metrics 文件
和 42 个 TensorDict 文件），哈希真实的 oracle 二进制，并绑定初始权重、源码提交、
探针/用例/包装器哈希、运行时标志、可执行文件/PID 和原生工作树路径。收集器拒绝缺失
或变化的绑定。全部 20 个原始测量进程具有不同的 PID；最终跨阶段检查确认每个仓库的
完整已安装内容在两个设备和所有阶段之间都相同。两个运行时必须有完全相同的完整已安装
包哈希；这也绑定了上游 Python 子集之外的已安装生成文件和原生包二进制。

CPU 显式隐藏所有 CUDA 设备，把规范/遗留 CUDA 标志置零并清空 nvcc 路径。CUDA 在 CPU
之后运行，使用一块分配的物理 GPU，带有真实的驻留设备断言并禁止后端后备。两侧都使用
TF32 关闭、cuDNN benchmark 关闭以及 float32 matmul 精度 `highest`；实际的 JSON 回读、
线程设置和亲和性必须一致。设备专用的 benchmark 缓存与独立的单元缓存为任务自有；
首次 JIT 串行执行。基线 seed 0 导出参考方的完整初始权重。两个 seed-7 扰动进程加载那些
确切的原始权重，且输出必须变化。TensorDict 训练在两侧也加载同一个 seed-0 权重产物。
任何诊断计时或分配器数值都不用于授予 L4/L5、速度或内存结论。

CPU 平台：Linux 6.1.188/glibc 2.34、AMD EPYC 9554 64-Core Processor、g++ 11.5.0、
零可见显卡。运行时线程与 OMP/MKL/OpenBLAS 均为 2。原生 auto_graph_replay 为 1，
保留预算 268435456 字节；这是默认策略，这些带梯度的正确性探针不作出任何推理重放
性能主张。实际 CPU 的 TF32 与 cuDNN benchmark 回读为 false，matmul 精度 highest，
use_cuda/use_acl 为零。CUDA 平台：一块 NVIDIA GeForce RTX 4090，驱动 580.178.04，
Jittor strict nvcc/CUDA 12.2.140 wheel 栈与独立的 Torch CUDA 12.8。
两个 TF32 开关和 cuDNN benchmark 均为 false，matmul 精度 highest，
use_cuda=1/use_acl=0。原始平台包含分配的物理到可见映射、UUID、两个解释器选定的
NVIDIA 分发元数据、编译器版本和实际标志。只读的 `/proc/PID/maps` 快照佐证了实际测量
PID 加载的 CUDA 库；它们是补充性的软件标识，不能替代驻留张量检查。

最终的共享 schema 产物为 `final-scorecard.json`，SHA256 为
`41a0826b495d63b9a989db7c04ecdc5239df3fed6557b24187721e6f0f48ae80`：
四条记录被接受，零拒绝，70 个比较字段，58 个唯一原始产物路径/哈希绑定。
每条记录还绑定两阶段基线/扰动元数据、完整梯度、平台、原始权重和 TensorDict 训练
元数据。源码/框架内容与同步的 `ab006` 基线相比未变；`measurement-source.json`
保留了该比较。可执行的增加项是测量/收集工具及其宿主测试。

## 复现

使用两个私有环境并安装相匹配的真实包。为每个仓库/运行时使用一个可执行启动器，
通过以下方式转发到原始解释器：

```bash
<runtime-python> agent/skills/torch-compat-repo-runbook/scripts/component_measure.py \
  --root <jittor-checkout> --repo <torchmetrics-or-tensordict> \
  --checkout <clean-release-checkout> -- "$@"
```

把这些启动器分别作为 `verify_repo.py --oracle-python` 和 `--shim-python` 传入，
选择既有用例并加上 `--seed 0 --repeats 3 --no-memory`。再通过每个启动器以
`--seed 7 --weights <original-reference.weights.npz>` 运行既有 runner，保存
`<case>.perturb.ref.npz` 和 `.perturb.shim.npz`。独立的比较 JSON 记录两个原始结果
以及输出变化检查。对 TensorDict，通过两个启动器运行 `training_probe.py`，
参数为 `--seed 0 --steps 3 --learning-rate .001 --weights <original-reference.weights.npz>`，
输出主名为 `<case>.train.ref` / `.train.shim`。先运行 CPU，设置
`CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 nvcc_path='' JT_BUILD_NVCC_PATH=''`；
为 CUDA 配置分配的 GPU 和独立缓存。两个阶段都设置
`JITTOR_ECOSYSTEM_TF32=0 JITTOR_ECOSYSTEM_CUDNN_BENCHMARK=0`、离线标志和相同的
线程设置。

使用可复用的 skill 工具进行收集，它会重新枚举两个上游并调用实际共享的
`audit_coverage.scorecard`：

```bash
python agent/skills/torch-compat-repo-runbook/scripts/collect_component_scorecard.py \
  --upstream torchmetrics=<clean-metrics-checkout> \
  --upstream tensordict=<clean-tensordict-checkout> \
  --run torchmetrics=<cpu-metrics-run> --run torchmetrics=<cuda-metrics-run> \
  --run tensordict=<cpu-tensordict-run> --run tensordict=<cuda-tensordict-run> \
  --platform cpu=<actual-cpu-platform.json> --platform cuda=<actual-cuda-platform.json> \
  --baseline 90ee18a9916acda9b5e86ca665f09a231a41ced1 \
  --report docs/results/2026-10-15-metrics-tensordict-scorecards.md \
  --command '<complete two-runtime commands or retained task driver invocation>' \
  --output <outside-checkout-scorecard.json>
```

首次 CUDA 尝试在候选计算之前停止：严格的 wheel 校验发现 cu128 运行时元数据领先于
所需的 cu122 栈。它完整保留在 `attempts/cuda-wheel-mismatch/` 下。一个独立的任务自有
CUDA shim 解释器把既有的 cu122 目录放在其私有搜索路径首位，从而解决了环境问题；
严格校验保持启用。CPU 解释器与全部 CPU 测量被保留。

可复用工具的验证：8 项定向宿主测试；在把新测试的收集期加载器移入 fixture 修复之后，
相关契约/工具运行通过 42 项测试，完整结构运行通过 1486 项、六项跳过。
布局、空白和 Ruff 通过。没有放宽任何允许列表。原始结构失败予以保留。
在加入本报告之后，定向的文档/工具运行通过 29 项测试；余下两处失败是本报告的中央
results-toctree 登记和重新生成的根清单。集成所有者维护这些共享文件。
因此完整的 1486/6 结果描述的是加入本报告之前的工具源码，并不表示未登记的文档树已全绿。

原始数组、包含失败尝试在内的原始日志、完整命令/环境配方、平台 JSON、清单和哈希
未纳入版本控制，位于 `$JITTOR_LAB_ROOT/_state/metric-tensordict-scorecards/20261015/`。
不重写任何旧原始文件。

## 主分支集成验证

集成同步了 `02c14ff7edde1203f6bb54230e6d0f8f0686e5e9`。独立评审重新哈希了全部
139 个封存原始文件和六个交付源码文件，重算了全部 70 次数值比较，检查了 20 个测量进程
和完整已安装包绑定，并重跑了共享审计器：四条记录被接受，零拒绝。
历史测量基线仍为 `90ee18a9916acda9b5e86ca665f09a231a41ced1`。

一项针对 TensorDict 产物副本的反例实验暴露了收集器的缺口：两个训练运行时可能就一份
与基线不同的替换初始权重快照达成一致。收集器现在要求每个阶段的实际权重文件 SHA256
等于其元数据和原始基线快照。同一反例实验在修复后被拒绝；对全部四条原始正向记录的副本
重新收集仍得到四条被接受、零拒绝。139 个封存原件保持不变。四个定向回归用例覆盖有效、
替换、修改和缺失的快照；相关宿主/契约套件通过 99 项测试。

在集成后的主工作树上，既有 Metrics 与 TensorDict 门禁在真实 CPU（2 通过，零跳过）
和 CUDA（2 通过，零跳过）上通过。这些检查验证与更新主分支改动的组合；
它们不改变更早 scorecard 测量的源码标识。集成日志与独立审计结果未纳入版本控制，
位于 `$JITTOR_LAB_ROOT/_state/metric-scorecard-integration-02c/`。

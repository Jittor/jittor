# CPU 与 CUDA 上真实的 32768 token Longformer

- 状态：复现了全上下文编码器正确性与三步全参数训练；tiny 宽度，无性能评级。
- 测量基线：`cbe154c3b81b3f2cca65363e0bdc6bdcec60c191`，在所有数值与 pytest 子进程
  执行中保持干净；包含 Longformer 构建器、显式完整梯度契约和需显式开启的测试节点。
- 上游：干净检出 Transformers `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc`，release `4.56.2`。
- 维护者：Jittor 核心 / Torch 兼容性维护者。
- 复查时机：Longformer attention/位置实现、模型参数注册、下游依赖、兼容张量语义
  或精度策略发生变化时。

## 确切的模型与 attention 范围

同一个构建器在独立的二进制 PyTorch 与 Jittor 解释器中执行真实的上游
`LongformerModel`。它有一层、hidden size 32、四个 head、intermediate size 64、
词表 64、batch 1、local attention 窗口 256、最大位置嵌入 32770、dropout 为零。
输入有 **32768 个位置**，全部有效，且第一个 token 具有真实的全局 attention。
未使用序列截短、模型下载、adapter、复制的 attention 算法、mock、no-op 或下游源码补丁。

公共 `get_input_embeddings()` 先计算词嵌入，然后加入一个按 0.01 缩放的浮点噪声张量，
并通过真实的 `inputs_embeds` 路径传入。这同时保留了词嵌入参数梯度和一个可微输入。
独立的 oracle 观测确认实际位置 ID 为 2 到 32769，以及一个 `[1,1,32]` 的全局 query 输入；
只读钩子保持了上游计算不变。完整的 `[1,32768,32]` 隐藏状态与真实的 `[1,32]` pooler
都参与 loss；返回的展平输出包含 1048608 个值。local 与 global 的 query/key/value 参数
以及 pooler 参数被显式纳入完整梯度契约。

每次比较要求输出、**全部 29 个参数梯度**和 `embedding_noise` 输入梯度：
两侧各 31 个精确形状的有限数组。共有 29 个可训练参数，无冻结参数。
尽管具有真实的 32768 token 长度，模型宽度/层数采样仍为 `shape_kind=tiny`。
已确立的范围是该全参数编码器场景，而不是稠密 32K attention、生成、
生产规模的预训练 checkpoint 或每一个 Longformer 任务 head/API。

## 固定平台与协议

| 字段 | CPU | CUDA |
| --- | --- | --- |
| Oracle | 真实二进制 PyTorch 2.11.0+cu128 | 相同，CUDA 12.8 |
| 候选方 | Jittor 2.0.0 Torch 兼容层 | 相同，NVCC 12.2.140 |
| 依赖 | NumPy 1.26.4；来自固定源码的 Transformers 4.56.2 | 相同 |
| 硬件 | AMD EPYC 9554，加速器可见性为空 | 一块分配的 NVIDIA RTX 4090，驱动 580.178.04 |
| 精度 | TF32 matmul/cuDNN OFF，matmul highest | TF32 matmul/cuDNN ON，matmul high |
| 其他控制 | float32，cuDNN benchmark OFF；两个运行时线程 | 相同；真实 CUDA 分派，零后备 |

两个解释器加载同一个固定的上游检出。基线 seed 0 导出完整的参考权重。
独立的输入 seed 7 和 19 在两个运行时上复用同一权重哈希；
输出必须变化，且所有梯度数组必须保持存在。另一次公共 `training_probe.py` 运行使用
全新的 seed-0 参考权重、SGD 学习率 0.001 以及三个确定性输入 seed
1、2、3。每一步保存输出、标量 loss、全部梯度以及每一个更新后的参数：
共 **183 个数组**。这比较的是更新轨迹，而不是收敛。

既有的生态 harness 把归一化偏差定义为
`maximum_absolute_error / max(field_maximum_absolute_reference, 1e-3 * block_maximum_absolute_reference, 1e-6)`；
梯度块包含全部请求的梯度，输出有其自己的参考字段。报告同时保留绝对误差和该确切的
归一化协议。CPU 限值为输出 2e-3、梯度 1e-2；CUDA 限值为 5e-3 和 2e-2。
更新后的参数/loss 使用输出限值及其自身的参考数组下限。这些是既有的协议阈值，
不是事后调校的容差。

## 结果与证据

| 设备 | 每个 seed 的 seed 0/7/19 数组数 | 最大绝对 / 归一化误差 | 三步数组数 | 最大绝对 / 归一化误差 |
| --- | --- | --- | --- | --- |
| CPU | 31 | 0.0166015625 / 8.67267123e-06 | 183 | 9.53674316e-07 / 9.84836796e-06 |
| CUDA | 31 | 0.078125 / 2.50184426e-05 | 183 | 1.43647194e-05 / 2.73647853e-05 |

全部比较均为有限值且形状精确，键集合一致，请求的梯度完整。
参考方与候选方都在三步中更新了 28 个参数。
`key_global.weight` 在该协议中保持相同的 float32 值；其完整梯度和每个参数快照仍被检查。
数学上接近零的 attention-key bias 梯度也被保留，而不是从契约中移除。参考标量 loss
轨迹为 CPU：-0.000152084525、-0.00200377125、-0.000821642869；CUDA：-0.000152087901、-0.00200377079、-0.00082164153。完整的候选 loss 与逐数组误差存储在原始比较中。

- 实际需显式开启的门禁：CPU **1 通过**，CUDA **1 通过**，零跳过。CPU/CUDA 的 JUnit
  节点时间为 297.888/454.044 秒，包含在不同单元缓存中的首次编译。
- 公共 scorecard：**两条被接受的 L3 记录，零拒绝**；84 个 SHA256 产物绑定经独立检查。
- 宿主用例/审计契约：**37 通过**。CPU core：**322 通过、42 跳过、1 xfailed**。
- 加入本报告之前的完整结构：**1451 通过、6 跳过、零失败**。Lint、diff 和布局通过。
- 伪造的 CUDA OFF/highest 标签会以 `Platform/measurement precision mismatch` 被拒绝；
  CPU 仍被接受。

资源诊断来自更早的相同全上下文原型，并保留其原始源码基线。子进程 rusage 最大 RSS
约为 Torch/Jittor CPU 的 1.93/2.22 GiB，CUDA 根进程采样到的 GPU 值为
Torch/Jittor 的 1950/2334 MiB。这些是诊断采样和子进程 rusage，不是连续的整进程
GPU 峰值。正式的 CUDA 分配采样还显示存在其他进程。单次重复的 benchmark 时间仅保留在
原始验证 JSON 中；无论采样还是 tiny 宽度计时都不构成 L4/L5 性能或资源验收。

需显式开启的节点为
`compat/tests/torch/test_ecosystem_longformer.py::EcosystemLongformer::test_transformers_longformer_32768`。
它要求真实执行、独立的 Torch oracle、Transformers 以及单一的显式设备。
在缺少依赖/oracle 或混用 CPU/CUDA 配置时选择该节点会失败。速度比请求会被拒绝，
因为这一 tiny 宽度用例不能确立性能。常规的未选择运行会可见地跳过这一昂贵用例。
正式运行使用独立的单元测试缓存；benchmark 缓存只由 benchmark/probe 工具复用。
每个真实节点都直接在 pytest 下运行；不声称使用 nox 包装器或定时 CI 调用。

## 自动分母与限制

公共清单从干净固定的 SHA 重新生成 Transformers 模型族与任务注册表。
共有 **374 个模型族**和 **46 个任务注册**。scorecard 在 **L3** 接受一个具名的
Longformer 全参数 CPU 场景及其匹配的 CUDA 场景：在模型族维度上分别为 **1/374**；
在任务维度上为 **0/46**。每一行已验证记录都保留具体用例、完整注册表引用、tiny 形状
以及全部可训练参数名。本报告不确立任何其他库/模型族、NPU、ROCm、Corex、L4 速度或
L5 资源结果。

原始证据位于外部路径
`$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/longformer-32768-20261015/formal-cbe154/`。
其 schema-2 证据文档对每个原始数组、真实平台声明、比较、训练元数据、命令、JUnit 结果、
清单和源码标识进行哈希。一项反例审计复制 CUDA 平台标签，把实际的 TF32 ON/high 伪造成
OFF/highest，并重新计算该副本的产物哈希。CUDA 记录必须被拒绝，而未被改动的 CPU 记录
被接受；原始测量元数据和张量保持不变。

关键的封存产物摘要：

| 外部产物 | SHA256 |
| --- | --- |
| `scorecard-evidence.json` | `bd4f0a68f578be23bb6e2ce4016aa14b26b307ea7a07e035e1d06ea4c8406274` |
| `scorecard.json` | `a8a77219533559a151efe6cbe62a3634dff6d32196f4402394dbd6f8622f131c` |
| `transformers-inventory.json` | `f4efa7d5e6437ef219e69fa90287b88c4d1d7bee50228eff7df191251a5dc8da` |
| `negative-precision-scorecard.json` | `eb3534f437fa85c1adcbe183216f5d8b19215e120535a3664c4bcf73b317777f` |

## 复现

使用独立准备好的 NumPy-1/Transformers-4 解释器以及独立的 state/缓存根：

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 \
python agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --cases transformers_longformer_32768 \
  --oracle-python "$REAL_TORCH_PYTHON" --shim-python "$SHIM_PYTHON" \
  --device cpu --seed 0 --repeats 1 --no-memory --out "$RESULT_ROOT/cpu"
```

在已分配的 CUDA 上以 `--device cuda` 重复。CPU 必须显式屏蔽加速器并禁用 CUDA。
两个运行时的实际精度/版本/导入来源块必须一致。对于扰动，在 seed 7 和 19 处调用公共
`_ecosystem_runner.py`，并把基线 `.ref.weights.npz` 传给两个运行时。以
`--steps 3 --learning-rate 0.001` 调用公共 `training_probe.py`，把全新的训练参考权重
传给候选方。保存的外部命令产物包含完整的实际调用与参数。

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 JITTOR_HOME="$UNIT_JITTOR_HOME" \
JITTOR_TORCH_CACHE_ROOT="$UNIT_SHIM_CACHE_ROOT" \
JITTOR_ECOSYSTEM_LONGFORMER_32K=1 JITTOR_REQUIRE_TRANSFORMERS=1 \
JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_TEST_REQUIRE_EXECUTION=1 \
JITTOR_TEST_DEVICES=cpu JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
"$SHIM_PYTHON" -m pytest -q compat/tests/torch/test_ecosystem_longformer.py

python agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py scorecard \
  --upstream "transformers=$TRANSFORMERS_CHECKOUT" \
  --evidence "$RESULT_ROOT/scorecard-evidence.json" \
  --baseline cbe154c3b81b3f2cca65363e0bdc6bdcec60c191 \
  --output "$RESULT_ROOT/scorecard.json"
```

对 CUDA 节点，设置那一块已分配设备和 `JITTOR_TEST_DEVICES=cuda`；使用相同的真实
oracle、显式必需标志和独立的单元测试缓存。保存的源码哈希与完整提交把本报告与其被测
实现绑定。

## 主分支集成

在父提交 `23670917490c36f790e1bc24ec136cb1a5cb2230` 上集成，包含结果索引和生成的
清单登记。数值记录保留其干净的 `cbe154c3` 测量基线。独立集成检查重新哈希了全部
84 个产物绑定，通过 37 项宿主覆盖/准入测试，core 322 通过 / 42 跳过 / 1 项既有
xfail，完整结构 1453 通过 / 6 跳过 / 零失败。此前两处中央文档登记失败已解决。
集成日志未纳入版本控制，位于
`$JITTOR_LAB_ROOT/_state/frozen-view-fix/longformer-*.log`。

# 四库真实 CPU/CUDA oracle 与三步训练 scorecard

- 状态：四个指定样例的 CPU、CUDA 数值与三步训练实测通过；六条注册表 L3 记录通过机器审计，另有两条 Diffusers **组件**训练证据，不计 pipeline 覆盖。全部结果均为手动验收。此前手工平台快照误标 CUDA TF32 关闭的 schema v1 scorecard 已撤回；本报告使用逐阶段原始元数据绑定的更正 schema v2。
- 日期：2026-10-14（Asia/Shanghai）。
- Baseline commit：`230ba7a0305091fb64ec81a679f32dcaefcea518`；该提交在同步远端后的 `45d08301f3357bf462d88f9be543bba85fb84063` 上修复训练 probe 的参数快照变量。
- 审计器提交：`822632d097ab848d3263ff69272aaf993a95dc64`，schema v2；计算提交仍为上述 baseline。
- 范围：Transformers GPT-2、Diffusers UNet2DModel、PEFT LoRA/Llama、ms-swift LoRA/Llama；tiny FP32，独立 PyTorch 二进制 oracle 与 Jittor shim 分进程执行。
- Owner：Torch 兼容层与下游生态维护者。
- Review when：上述提交涉及的 compat/核心梯度语义、共享 ecosystem builder/runner、训练 probe、下游版本或平台/精度配置发生变化。

## 固定平台与真实上游

CPU 为 AMD EPYC 9554 64-Core Processor，Linux 6.1.188，Python 3.11.16、NumPy 1.26.4、GCC 11.5.0。双方计算线程及 OMP/MKL/OpenBLAS 线程均固定为 2，进程 CPU affinity 相同。CPU 进程显式设置 `CUDA_VISIBLE_DEVICES=''`、`use_cuda=0`、`JT_USE_CUDA=0` 并禁用 nvcc；实测 Jittor `use_cuda=false`、`use_acl=false`。

CUDA 使用一张 NVIDIA GeForce RTX 4090，驱动 580.178.04，分配的物理卡映射为进程内 `cuda:0`。Jittor 工具链 NVCC 12.2.140，实际 cuBLAS API 返回 12.2.5、cuDNN API 返回 91900；双方实际 **matmul/cuDNN TF32 均允许，`matmul_precision=high`**；Jittor 原生 HF32 关闭，native `cuda_kernel_math=strict` 标志单独记录，不能将其解释为 Torch frontend 的 matmul 是 strict。默认 graph regime，MKL/NCCL/MPI/CUTT 和并行 op compiler 均关闭。oracle 是真实 `torch._C` 二进制的 PyTorch **2.11.0+cu128**，无 shim 标记；其构建 CUDA 版本与 Jittor 编译工具链分别记录，不能写成双方都由 CUDA 12.2 构建。

每一侧使用独立解释器及运行环境，CPU、CUDA、结构测试分别使用独立 JIT 缓存。两侧加载下表同一份干净 release 源码；固定版本的源码注册表也从该 checkout 自动枚举，没有沿用开发分支更大的分母。

| 库 | 实际版本 | 完整 upstream SHA |
| --- | --- | --- |
| Transformers | 4.56.2 | `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc` |
| Diffusers | 0.35.1 | `0f252be0ed42006c125ef4429156cb13ae6c1d60` |
| PEFT | 0.17.1 | `53c25fe4fdd7c6aa4b40db0560815ea570d32303` |
| ms-swift | 4.5.2 | `ff5128777dd3dc5538f1a95fbcc29442c327d7d2` |

ms-swift 真实依赖另含 ModelScope 1.33.0、Datasets 3.6.0、PyArrow 19.0.1 与 pandas 2.2.3。依赖恢复只写外置私有环境：既有环境缺失的 Python 源文件以其原有 CPython 3.11 bytecode 私有镜像恢复；cuDNN 缺失的 CNN/ADV 库和头文件从官方 **9.19.0.56** wheel 提取，并逐成员核验 wheel RECORD 的 SHA256。未修改共享安装、下游源码或框架语义以绕过验证。实际库解析路径、wheel 来源、成员哈希和失败尝试原始日志均保留在外置产物中。

## 计算协议与结果

首先复用[共享 runner](../../compat/tests/torch/_ecosystem_runner.py)执行输出、全部请求的参数梯度及浮点输入梯度对拍。两侧从 oracle 保存的完整同名参数与 buffer 初始化。随后固定这份权重，仅将输入随机种子从 0 改为 7，验证输出确实随输入改变，并重新比较全梯度。整数 token 不请求输入梯度。

训练使用[已提交的 training probe](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/scripts/training_probe.py)，`model.train()`、3 步、固定 SGD 更新 `p -= 0.001 * grad`，每步输入种子为 1、2、3，固定随机 loss 权重并取 mean；逐步保存输出、loss、全部请求梯度及所有参数。先物化所有旧梯度再更新 lazy Var；保留上游原始冻结状态，每一步检查冻结参数不变，且训练后至少一个可训练参数发生变化。LoRA 的数值梯度测试刻意请求全部 29 个参数的梯度；真正训练则仅更新上游声明的 8 个 adapter 参数，21 个基座参数冻结。此 probe 不验证 Adam、Trainer、调度器或分布式训练。

| 样例 | 每次数值对拍张量数 | 参数梯度 / 输入梯度 | 训练可更新 / 冻结参数 | 三步轨迹数组数 | CPU / CUDA |
| --- | ---: | ---: | ---: | ---: | --- |
| `transformers_gpt2` | 29 | 28 / 0 | 28 / 0 | 174 | 全部通过 |
| `diffusers_unet2d` | 146 | 144 / 1 | 144 / 0 | 873 | 全部通过 |
| `peft_lora_llama` | 30 | 29 / 0 | 8 / 21 | 117 | 全部通过 |
| `ms_swift_lora_llama` | 30 | 29 / 0 | 8 / 21 | 117 | 全部通过 |

比较沿用本次 `verify_repo.py` 的 FP32 判据：每个张量的最大绝对误差不超过 `2e-3`，**或**相对同类 reference field 尺度的误差不超过 `1e-2`；不是逐元素 `np.allclose` 的相对误差。全部张量必须 finite，无缺失梯度。下面前两项取基线与固定权重输入扰动两轮中的最坏值，轨迹项覆盖三个完整更新步骤：

| Device / 样例 | 数值最大绝对误差 | 数值最大 field-scaled 相对误差 | 三步轨迹最大绝对误差 |
| --- | ---: | ---: | ---: |
| CPU / GPT-2 | 2.67029e-5 | 2.83476e-7 | 2.38419e-7 |
| CPU / UNet2D | 4.67300e-5 | 8.86536e-7 | 1.37091e-6 |
| CPU / PEFT LoRA | 1.43051e-5 | 3.08378e-7 | 2.38419e-7 |
| CPU / Swift LoRA | 1.43051e-5 | 3.08378e-7 | 2.38419e-7 |
| CUDA / GPT-2 | 9.65500e-3 | 1.34945e-4 | 1.68085e-4 |
| CUDA / UNet2D | 1.25256e-2 | 2.37624e-4 | 3.41415e-4 |
| CUDA / PEFT LoRA | 1.03002e-2 | 2.21989e-4 | 5.96279e-7 |
| CUDA / Swift LoRA | 1.03002e-2 | 2.21989e-4 | 5.96279e-7 |

CUDA 四例在基线、扰动和训练三个阶段，双方实际 device 都是 `cuda`；Jittor fallback policy 为 `error`，计数增量为 **0**。CPU 四例同样实测关闭 CUDA。CUDA 的部分原始梯度绝对差超过 `2e-3`，通过的是明确记录的 field-scaled 相对判据；本报告没有宣称全部绝对误差小于该阈值。

## 注册分母、授级与门禁边界

[scorecard 审计器](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py)重新枚举上游完整注册表，逐条检查本次 Git SHA、上游 SHA/版本、平台、真实二进制 oracle、请求/实际 device、完整梯度、输入扰动、多步轨迹、原始产物哈希及现存具体 gate 节点。CPU 与 CUDA 各接受 GPT-2、PEFT LoRA、Swift LoRA 三条 **broad/L3** 记录，共六条；每个 CUDA 记录都有对应的独立 CPU 对拍。

| 库 / 注册轴 | CPU 已验成员 / 完整分母 | CUDA 已验成员 / 完整分母 | 本次实际证据 |
| --- | ---: | ---: | --- |
| Transformers / model family | 2 / 374 | 2 / 374 | GPT-2 全参 SGD；Llama **仅 PEFT/Swift LoRA**，未验 dense Llama 全参训练 |
| Transformers / task | 1 / 46 | 1 / 46 | causal LM mapping |
| PEFT / method | 1 / 26 | 1 / 26 | LORA |
| PEFT / task | 1 / 6 | 1 / 6 | CAUSAL_LM |
| ms-swift / model family | 0 / 222 | 0 / 222 | 没有调用 Swift 的注册模型加载器 |
| ms-swift / method | 1 / 11 | 1 / 11 | Swift 原生 LoRA tuner |
| Diffusers / pipeline module family | 0 / 79 | 0 / 79 | 本次只执行 UNet2DModel 组件 |
| Diffusers / pipeline class | 0 / 249 | 0 / 249 | 未执行完整 pipeline |
| Diffusers / auto pipeline family | 0 / 37 | 0 / 37 | 未执行 auto pipeline |
| Diffusers / auto pipeline task | 0 / 3 | 0 / 3 | 未执行 auto pipeline |

**UNet2DModel 的 CPU/CUDA 证据满足 L3 的计算和训练条件，但不属于现有 inventory 的 pipeline 注册轴。** 两条组件证据单独保存，未伪造 DDPM pipeline 引用、未缩小分母，也未送入注册覆盖接受清单。Swift builder 直接构造 Transformers Llama 后交给 Swift tuner，因此计入 Swift LORA 方法与 Transformers Llama，不能据此计入 Swift 自己的 222 个注册模型家族。

上述 2/374 是带方法上下文的家族触达计数，**不是两个 dense 模型的全参 L3 验收**。GPT-2 样例更新全部 28 个参数；Llama 样例只通过 PEFT/Swift LORA 方法更新 8 个 adapter 参数。原始 accepted 记录同时保留 Transformers 家族与对应 LoRA 方法的 registry_refs，阅读平铺 X/N 时必须一同查看这些场景，不能外推 dense Llama、其他 tuners 或同家族全部 API。

既有具体门禁位于 `test_ecosystem_parity.py::EcosystemParity::test_<case>` 及其 CUDA 子类；该门禁覆盖共享的一步数值计算。本次固定权重扰动与三步训练由手动 probe 补充，**不是已经接入每次 CI 的 L3 门禁**。没有执行其他九库的本轮独立 CPU/CUDA oracle；未执行 TorchMetrics、TensorDict、MMCV、MMEngine、FlashAttention、vLLM、vLLM-Omni、VERL、TRELLIS。已有其他历史结果的范围请查逐库 runbook，不能合并成本次新授级。

本次无 real-size 性能验收，没有完整进程显存口径，也未比较 train/eval 或 graph 的其他 regime，因此 **不授予 L4 或 L5**。零回退是已执行 CUDA 样例的直接事实，不绕过 L4 的累积条件。短样例 `seconds` 只保留为诊断数据，训练 probe 的 NumPy 物化与传输也不用于性能结论。

## 复现与产物

实际 release checkout 位于 `$JITTOR_LAB_ROOT/downstream-coverage/releases/<repo>`；原始数据位于 `$JITTOR_LAB_ROOT/_state/downstream-evidence/real-runs/`，包括独立解释器、运行条件、实际 dependency origin、每轮 NPZ/JSON/log、完整源注册 inventory、平台及 CUDA provider map。原始数据和环境绝对路径不进入 Git。

先按上述版本准备真实 PyTorch oracle 与 shim 两个独立环境，设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 与线程环境，CPU 显式禁 CUDA。CUDA 显式设置 `JITTOR_ECOSYSTEM_TF32=1`、`JITTOR_ECOSYSTEM_CUDNN_BENCHMARK=0` 以复现本次实际 high regime；改为 strict 是另一组需要重跑的测量。基线数值计算示例：

```bash
<shim-python> agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --cases transformers_gpt2 --device cpu \
  --oracle-python <oracle-python> --shim-python <shim-python> \
  --no-memory --repeats 3 --out <outside-checkout>/cpu
```

三步训练单侧执行命令如下；CUDA 将两个 `--device cpu` 改为 `cuda`，使用分配的卡及完整工具链，独立于 CPU 编译缓存。相同命令依次用于表中四个 case，两个 NPZ 的全部数组必须按上面的阈值逐一比较，不能仅比较 `TRAINING_RESULT` 标记：

```bash
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case transformers_gpt2 --runtime torch --device cpu --steps 3 \
  --output <outside-checkout>/cpu/transformers_gpt2.train.ref.npz
<shim-python> agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case transformers_gpt2 --runtime jittor --device cpu --steps 3 \
  --weights <outside-checkout>/cpu/transformers_gpt2.train.ref.weights.npz \
  --output <outside-checkout>/cpu/transformers_gpt2.train.shim.npz
```

固定权重扰动使用共享 `_ecosystem_runner.py`，两侧均 `--seed 7 --weights <base-reference-weights.npz>`；完整命令、各阶段比较明细和原始哈希均随外置 evidence JSON 保存。重新审计本次接受清单：

```bash
python -S agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py scorecard \
  --baseline 230ba7a0305091fb64ec81a679f32dcaefcea518 \
  --upstream transformers=<releases>/transformers \
  --upstream peft=<releases>/peft \
  --upstream diffusers=<releases>/diffusers \
  --upstream ms-swift=<releases>/ms-swift \
  --evidence <real-runs>/scorecard-evidence-v2.json \
  --output <outside-checkout>/scorecard-v2.json
```

命令通过完整 `--baseline` 明确重新审计测量提交 `230ba7a0305091fb64ec81a679f32dcaefcea518` 的历史证据；输出同时记录新审计器提交。默认不传该参数时仍拒绝不同 HEAD 的旧证据，不能为获得通过而改写原始测量基线。显式历史审计不证明当前树已经重跑。审计输出六条 accepted、零 rejected；两条 Diffusers 组件记录在 `component-evidence-v2.json` 单列。原始阶段标记中的 `scorecard_eligible=false` 仍保留：单次 `verify_repo.py` 本来不具备扰动、平台和训练证据，本报告组合独立阶段产物后才建立 L3。

该次真实执行还复现并修复了训练 probe 的变量名覆盖错误：首次正式 oracle 在首步报 `TypeError` 并停止，失败日志保留；修复后在本报告基线上完整重跑了四库两 device 的全部阶段。审计与训练 probe 的 29 个主机侧测试全部通过，相关 pytest/child-process 契约及工具定向复验 **71 passed**；Ruff 0.15.22 通过；布局及本报告和初始审计报告的内部链接检查通过。本基线完整结构测试实际为 **1425 passed、6 skipped、3 failed**：旧 `DropoutACL` 梯度清单不一致，以及新报告尚未加入 results toctree、打包 MANIFEST 两项集成缺口。本任务未修改共享索引或 MANIFEST；不能据此宣称完整结构门禁全绿。

本轮 metadata 更正保留原始 base/perturb/training NPZ 与比较 JSON，不改计算基线或原始数值。错误来源是外置平台采集器把 TF32 标签硬编码为 false，而 CUDA 驱动未显式设置 `JITTOR_ECOSYSTEM_TF32=0`，共享 runner 的真实默认为 ON/high。v2 直接读取三个阶段双方的 `tf32`、`runtime_conditions`、actual device，并与平台快照逐一比对；训练另绑定实际 trainable/frozen 参数名。旧 schema v1 及错误平台快照保留于外置 `history/schema-v1/`，明确撤回。更正来源及各原始测量哈希见 `precision-correction-provenance.json`。

同一组真实 raw 的反例也已执行：给三个 CUDA 行重新提供 hash 正确、标签为 OFF/highest 的平台快照，v2 审计实际返回非零，三条均以 `Platform/measurement precision mismatch` 拒绝；三个 CPU 行不受影响。正确 ON/high 平台重审则六条接受、零拒绝。coverage 每行还保留 `verified_scenarios` 的 case、repo、L、完整 registry_refs、tiny/real、实际 trainable/frozen 参数名与计数，保留 Llama+LoRA 的方法边界。

本次外置总产物的 SHA256 如下；逐个原始 NPZ、平台、比较与初始权重文件的哈希在 evidence 的 artifacts 中继续展开。

| 外置产物 | SHA256 |
| --- | --- |
| `scorecard-evidence-v2.json` | `d351f0669721b7d033c6c1a006181219e6ab68096c6f424d6dd7b621654ed2b4` |
| `component-evidence-v2.json` | `0dcba24e0102dfd920dc07980da3afb679d3ad1c7dc459c617636c4e75d78ad9` |
| `scorecard-v2.json` | `5c1fc98b1445cf0f55480ccf9c9b29384a4f77d1df3dc14d210622c456109fde` |
| `precision-correction-provenance.json` | `8e3ffa63924cfeb0b2a346cfa71a0daff2e244448bb3b3d316ebf5ca7f4a2be1` |
| `audit-validation-v2.json` | `126660351d4a17cd1ce40b0061f3bf3538948a76d5a8cd7fc53c29dc643d7a39` |

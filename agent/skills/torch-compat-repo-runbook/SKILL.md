---
name: torch-compat-repo-runbook
description: 为下游 Torch 生态库建立带上游 registry 分母的覆盖矩阵、独立 oracle 证据与固定平台/速度 scorecard，审计 skill、门禁和结果报告。用于新增一个库、复核已有 runbook、或判断一次对拍是否真的有证据。
---

# 下游仓库 torch-compat Runbook 模板与四轴验收

**用途**：给一个下游库建立可信的"shim 跑 / 原生 torch 跑 / 两者对拍"文档，并用
四轴证据签收。`transformers`、`diffusers`、`vllm`、`mmcv` 等各自的具体断点在它们
自己的 `*-torch-compat` skill 里；本 skill 只管**格式、协议和验收标准**。

**不覆盖**：断点本身该修 core 还是 compat 还是 adapter——那走
[`downstream-library-adaptation`](../downstream-library-adaptation/SKILL.md)。
数值 harness 细节见 [`jittor-torch-diff`](../jittor-torch-diff/SKILL.md)。

## 一、Runbook 骨架（一个仓库一个 skill）

目录 `agent/skills/<lib>-torch-compat/SKILL.md`，章节固定为：

| 章节 | 必须回答 |
| --- | --- |
| `## 用途` | 这个 runbook 能做什么，**以及明确不覆盖什么** |
| `## 两侧环境` | shim / oracle 两个解释器、库版本、离线开关、缓存隔离、oracle 断言 |
| `## 在 shim 上跑` | 可直接复制执行的命令 |
| `## 在原生 torch 上跑` | 同上 |
| `## 对拍` | 同权重同输入；精度口径；速度口径；device 规则 |
| `## 断点与分流` | 已知断点，每条标注 core / `jittor.compat.torch` / adapter |
| `## 坑与假绿` | 本库特有的坑，以及**看起来像成功**的失败 |
| `## 证据` | 结果落在哪，以及**实测 vs 仅报告推导**的分界 |

## 二、两个解释器

真实 PyTorch 和 jittor shim 都抢占 `torch` 这个名字，所以对拍必须**两个解释器**：
oracle 产出权重和参考值，shim 从同一份权重重算。

- **oracle 断言（看任何数字之前必须过）**：
  `assert not hasattr(torch, '_torch_compat_install_context')`。
  不过就是拿 shim 对自己，什么都证明不了。
- **环境变量契约**以
  [`_ecosystem_harness.py`](../../../compat/tests/torch/_ecosystem_harness.py)
  的模块 docstring 为准（`REAL_TORCH_PYTHON`、`JITTOR_ECOSYSTEM_PACKAGE_SITE`、
  `JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE`、`JITTOR_ECOSYSTEM_SPEED_RATIO`、
  `JITTOR_ECOSYSTEM_TF32`、`JITTOR_ECOSYSTEM_CUDNN_BENCHMARK` 等）。**不要另造一套。**
- **两侧下游库版本必须一致**，否则是在比两个不同的库。
- **离线固定输入**：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，上游 checkout 钉
  commit SHA。
- 缓存隔离：每个并行任务一个 `JITTOR_HOME`（或不同 `cache_name`）；首次 JIT/扩展编译
  串行跑。

## 三、四轴验收（签收标准）

只跑通不算接入。四轴都要有数：

| 轴 | 怎么量 | 通过标准 | 记录 |
| --- | --- | --- | --- |
| **带分母的覆盖矩阵** | 从固定上游 SHA 的 registry 自动枚举模型族/方法/任务轴，再绑定数值证据 | L1/import/登记不算覆盖；设备③须有CPU①；核心矩阵至少L3 | X/N @ SHA、代表规格、门禁node、实际L级 |
| **精度** | 同权重、同输入、同 device，两个解释器；逐输出/逐梯度比 | 误差不超过门禁容差；**相对误差用全场最大量级做 floor** | worst abs / worst rel |
| **显存** | L4 两侧用整进程峰值：CUDA NVML、NPU npu-smi、CPU RSS；allocator live/pool 仅作独立诊断 | 不把进程峰值、allocated、reserved 混成一列；首步与稳态分开 | jPeak/tPeak、采样来源/间隔、live/pool 附列 |
| **速度** | 同卡同规格、预热后至少10次采样；两侧使用同一已声明统计量 | 固定 eager/default 图重放模式，无并发；speed 旁必须有 agreement；PR 只报告 | 时间、ratio/上限、agreement、完整 regime |

`verify_repo.py` 的 peak/pool 两列都取自运行时自己的 high-water（jittor 侧
`get_peak_device_used_memory`，`f1a498c7` 起）。**不**是 python 线程采样出来的最大值：
采样线程只在解释器释放 GIL 时才跑，快 step 上两列都会少报（peak 实测 3406.5 对 7272 MiB，
小探针的 pool 实测 512 对 1536 MiB），而且会把同一个 step 的「峰值」读成两种不同的值。

三条容易踩的口径：

1. **精度的相对误差别按数组自己的量级除。** 一个接近 0 的梯度会报出 ~1 的相对误差，
   而绝对误差其实只有 1e-5——把正确结果读成失败。用整个场的最大量级做分母。
2. **allocator 诊断必须同口径比，两侧都问运行时自己。** 这不能替代 L4 的整进程峰值。
   外部采样必须按进程与设备映射；做不到时 L4 缺证据，不能拿 allocator 代填。
   例如
   `nvidia-smi --query-compute-apps` 在容器里不列出该进程（pid 映射），按 GPU 的
   `memory.used` 又含同租户。两侧都问运行时自己：torch 用 `max_memory_allocated` /
   `max_memory_reserved`；shim 侧 `torch.cuda.max_memory_allocated()` 恒为 0（实测），
   要用 `jt.core.device_memory_used(N)` / `device_memory_reserved(N)`。
   **不要**用 `jt.get_mem_info().total_cuda_used`：它是 used+cached-free 且对所有设备求和
   （`mem_info.cc`），跟 `max_memory_allocated` 比是 reserved 对 allocated，会把 jittor
   凭空放大——2026-09-19 的 transformers runbook 就是这么写的，见其「显存列是旧口径」。
   同样**不要**用 `jt.core.get_peak_allocator_used_memory()` 顶替：名字像 live high-water，
   实际是**主机+设备**（`memory_profiler.cc` 不过滤 CUDA；实测 512 MiB CPU Var 推 512、
   再加 256 MiB CUDA Var 推到 768），且没有复位接口。峰值要问
   `jt.core.get_peak_device_used_memory(N)`——它只累加该设备各池的 `used`，是
   `max_memory_allocated(N)` 的真正对应物。**不要**在 python 里轮询
   `device_memory_used` 当峰值：采样线程只在解释器释放 GIL 时（step 内即 device wait
   时）才跑，快 step 上只能抓到中间值（8 层 llama 实测采样 3406.5 MiB、运行时自己的
   high-water 7272 MiB、torch 6655 MiB）。
3. **whole-run 峰值会掩盖「第一步瞬态」。** 两侧都是 high-water，jittor 的 pool 还不归还
   CUDA。`large_transformers_bert` 修完 grad 身份 bug 后，稳态每步只差 1.11x，而 whole-run
   是 2.11x——差额全在第一次 backward / 第一个训练步。报显存时把稳态与首步分开写，
   否则预算按 whole-run 留、实际训练循环用不到；反之亦然。逐阶段量法见
   [历史记录](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-09-19-torch-compat-runbook-verification.md)。

## 四、跑四轴

[`scripts/verify_repo.py`](scripts/verify_repo.py) 把单 case 执行委托给项目自己的
[`_ecosystem_runner.py`](../../../compat/tests/torch/_ecosystem_runner.py)，因此数字和
生态门禁同源，不是私有实现：

```bash
source <lab>/env-jittor.sh
REAL_TORCH_PYTHON=<real-torch-python> <jittor-python> \
  agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo transformers --device cuda --repeats 5 \
  --out "$JITTOR_LAB_ROOT/_state/<topic>/verify/transformers"
```

- 先 `--list-only` 看该库的 case 清单，不必起进程。
- 输出 `verify-report.json`，每个 case 一行：精度、allocator 显存、诊断时间、fallback、device 一致性。
  非零退出（即使先打印成功 marker）、非有限张量、形状/依赖版本/设备/精度或线程配置不符均失败，
  不生成可比较 ratio；失败会让命令返回非零。原始双方报告与日志保留供审计。
- 这是单次前反向探针，**不是自动 L3/L4/L5 签收器**；缺平台块、输入扰动检查、多步轨迹或
  整进程峰值时，保留 `scorecard_eligible=false`，不能从 `passed` 推出这些更高级别。
- case 名用下划线而发行版用连字符（`ms-swift` → `ms_swift_lora_llama`），脚本已归一化。

### 多步训练证据

[`scripts/training_probe.py`](scripts/training_probe.py) 复用同一份已准入 ecosystem builder，
在独立解释器中执行默认三步、固定学习率的 SGD；不是新模型清单。先运行 CPU①，再运行
实际加速设备③。两侧用相同完整初始参数与 buffer，逐步保存输出、loss、全部请求的参数/
输入梯度和更新后参数。它保留上游 tuner 的冻结状态，并逐步断言冻结参数未改动；LoRA
训练不能因为单步数值探针打开了全部梯度而偷偷变成全参训练。

```bash
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case peft_lora_llama --runtime torch --device cpu \
  --output "$JITTOR_LAB_ROOT/_state/<topic>/train/ref.npz"
PYTHONPATH=python JITTOR_TORCH_SHIM=1 <jittor-python> \
  agent/skills/torch-compat-repo-runbook/scripts/training_probe.py \
  --case peft_lora_llama --runtime jittor --device cpu \
  --weights "$JITTOR_LAB_ROOT/_state/<topic>/train/ref.weights.npz" \
  --output "$JITTOR_LAB_ROOT/_state/<topic>/train/candidate.npz"
```

每个 `.npz` 附带 `.json`：实际 device、运行条件、依赖来源、训练/冻结参数清单、
初始权重 SHA256、探针 SHA256、基线及零 fallback 计数。比较双方全部数组，不能只比最后
一个 loss；固定权重的输入扰动还需另跑并核实输出确实改变。参数更新前会物化全部旧梯度，
快照包含 host 传输，**不作为计时器**。这些训练产物连同独立 L2 数值证据可以支持精确
规格的 L3；完整 scorecard 仍需平台块与注册项绑定，不能直接扩大到整库或 L4/L5。

## 五、验证环境没有该库时

`pip` 可用的机器上，把下游依赖装到一个**独立目录**当包站，再让两侧都指过去
（`JITTOR_ECOSYSTEM_PACKAGE_SITE` / `JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE`）：

```bash
pip install --target <site> <lib>==<version>
```

CPython minor 版本不同时默认拒绝共享包站，确实只走稳定 ABI 才显式加
`JITTOR_ECOSYSTEM_PACKAGE_SITE_CROSS_ABI=1`。

**绝不要把下游库装进 shim 解释器本身**：pip 可能把 jittor 的 `torch` 换成真 PyTorch，
那一次安装会作废之后所有对拍。

仓库自带两条门禁，优先复用而不是重写：

- `nox -s ecosystem`：十二个下游 case 的双解释器 parity + speed；要求
  `REAL_TORCH_PYTHON`，且是 fail-closed 的。
- `nox -s optional`：真 CUDA 上的可选兼容门禁，覆盖
  `OPTIONAL_COMPAT_PACKAGES`（`torchmetrics`、`mmcv`、`mmengine`、`peft`、
  `safetensors`、`tensordict`、`flash_attn`）。

## 六、假绿清单

1. **shim 对自己**：缺 oracle 时 case 自我 skip；用 `JITTOR_REQUIRE_REAL_TORCH=1`
   把这类 skip 变成失败。
2. **"跑 CPU" 其实不是 CPU**：jittor 没有 per-tensor device，有卡时 CUDA 默认就是
   开的，必须显式关。
3. **两侧版本不同**：比的是两个库，不是两个 runtime。
4. **签名齐全的 no-op**：见 [`torch-shim-noop-audit`](../torch-shim-noop-audit/SKILL.md)。
5. **只报告不测试**：runbook 里的命令没实际跑过，就标"未验证"，别写成既成事实。

## 七、落盘

- 可复现结论写进相关 GitHub issue/PR：Status / 日期 / Baseline commit /
  Owner / Review when，以及环境、命令、结果、边界。
- 原始日志、包站、缓存、`verify-report.json` 留在 `$JITTOR_LAB_ROOT/_state/<topic>/`，
  不进主仓库。
- 加新 runbook 时，同时在该库 skill 的 `## 证据` 里写清"实测"还是"报告推导"。

## MMEngine / MMCV 的下游训练 owner

OpenMMLab 的三步证据使用
[`openmmlab_training_probe.py`](scripts/openmmlab_training_probe.py)，保留真实
`BaseModel.train_step → OPTIM_WRAPPERS.build(OptimWrapper) → torch.optim.SGD`
调用链；通用 `training_probe.py` 的手工 SGD 不能代替它。`RecordingSGD` 仅在真实 step
前记录裁剪后的梯度，随后执行真实父实现；逐步检查 loss forward/optimizer update 次数、
scalar/list loss 解析、输入与参数梯度、参数/buffer/momentum state。

`nox -s ecosystem_openmmlab` 是预装 MMEngine 0.10.7 / MMCV 2.2.0 / NumPy 1.x 的独立
CPU/CUDA opt-in 门禁。它要求独立 oracle、逐库 REQUIRE 与实际执行；缺包、设备不符、
任何 fallback 或省略必要 tensor 都失败。这里只验证 tiny 训练规格，不授速度级别。

`registry_inventory.py` 支持两个库的有界源码声明轴。MMCV 整包条件/扩展注册仍进分母，
MMEngine 的动态优化器枚举则明确不在该声明轴内；每条保留源码位置与条件，未解析的
有界注册直接报错。不要把声明轴写成全库分母或已验证的 backend 能力。

## 八、统一覆盖审计与 scorecard

连续验证不同提交时，可显式设置 `JITTOR_ECOSYSTEM_CACHE_ROOT` 为当前任务专属的
benchmark 缓存根目录。`verify_repo.py` 在其下分开 CPU/CUDA/NPU 编译缓存，
结果仍写到每次独立的 `--out` 目录，避免仅移动结果目录就重编整个核心。
不设置时沿用结果目录内的缓存。不要与 unittest 或并行验证任务共用此根目录；
首轮编译仍串行预热。

先运行无框架依赖的事实审计：

```bash
python -S agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py audit
python -S agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py audit --json
```

审计展示 **有 skill / 注册 case / 实际共享门禁 / 存在与失效的报告链接 / 提及的 L 级**，
不给存在性证据授予“通过”。完整上游分母的派生、core/broad/tail 抽样规则和反作弊项见
[适配流程的覆盖度章节](../downstream-library-adaptation/SKILL.md)。

平台与速度元数据是所有下游结论的固定表头，不再仅限 NPU。每行都记录设备型号/数量/映射、
CUDA/driver 或 CANN、nvcc/gcc、nvidia wheel、Jittor SHA/flags、独立 oracle 身份和版本、
上游 SHA/version/remote、离线种子及 tiny/real 规格。L4 另需同卡时间/ratio/验收上限、
整进程峰值、warmup/iters、统计量、PURE eager/default 图重放模式、无并发声明与 agreement。
具体 schema、可执行命令与可复制的结果报告模板以
[统一下游 scorecard 协议](../../../docs/development/testing/downstream-scorecards.md) 为唯一入口。

`scorecard` 子命令每次重新读取 registry，不接受调用者提供小分母；版本/提交不符、
丢失平台块、skip/xfail/全skip、缺CPU①、非零fallback、缺梯度、假gate节点、产物哈希不符、
用tiny规格声称L4等均不计分，并使含被拒行的命令非零退出。旧 Markdown 报告需补齐并
亲自复验后才能进入分子；没有新证据的格子写“未验证”，不能拿本库 CPU/device 自比补空。

Release 版本的注册表 L2+ 也要求逐阶段、逐 runtime 的实际安装源码绑定，覆盖记录引用的
每个库；绑定 wheel 时核对完整包内容而非仅 Python 文件。具体字段与历史记录处理遵循
上述统一协议，不能用版本相同代替源码来源证据。

## PEFT 完整 LoRA 训练门禁

`nox -s ecosystem_peft` 调用 `tools/run_peft_gate.py`，复用现有完整 tiny Llama/LoRA
case、runner 和三步训练 probe。它串行执行 base、seed 7 扰动及三步训练的双 runtime，
自动独立检查全部 177 数组字段、冻结参数、实际设备与完整安装来源后才成功。
`--preflight` 只验证 base，不授 L3；nox 不接受缩小阶段范围的参数。

准备固定的 PEFT/Transformers clean checkout、官方 wheel、独立双解释器及私有状态目录；
先显式 CPU，再执行真实 CUDA。完整环境变量、collector 命令及范围见
[PEFT 来源与训练结果](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-10-15-peft-installed-source-training.md)。
普通 ecosystem pytest 的一步 parity 与这个完整 L3 gate 分开；nightly 未选择新 session
时不能声称已有定期执行。历史 raw 缺来源字段时重新运行，不能反向补证。

## Accelerate checkpoint 协议

完整单 rank CPU nightly 可用 `python tools/provision_ecosystem_nightly.py --library accelerate --state-root <new-external-state> --run`。
它完整配置三个独立环境并执行原 438 数组/8 阶段 worker 协议，要求原参数/输入梯度、三步保存前训练及两步新进程续训，
各 prefix 完整依赖解析与 `pip check`、实际官方 wheel/METADATA/pip RECORD 和公共 L3 准入均为必需。
`JITTOR_ACCELERATE_COMPLETE_CPU=1` 才启用 fixed CPU 模式，不改变普通 CPU/CUDA 调用。
实际执行状态和边界见[专用报告](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-10-16-accelerate-maintained-cpu-nightly.md)，
不将本地执行或工作流接线称为 hosted schedule 已运行，不用 CPU world1 配对 CUDA world2。

Accelerate 没有与模型族同义的注册表。`tools/run_accelerate_gate.py` 直接调用维护中的
单进程 checkpoint 与两 rank FSDP2 FULL 测试，`nox -s ecosystem_accelerate` 是正确性入口；
`benchmark_accelerate` 仍只负责性能。使用预装且版本相同的私有双解释器，设置
`REAL_TORCH_PYTHON`、`JITTOR_ACCELERATE_CHECKOUT`、`JITTOR_TEST_DEVICES=cpu|cuda` 和
`JITTOR_ACCELERATE_WHEEL`（官方 Accelerate 1.10.1 wheel）。CUDA 额外要求显式分配两张
`CUDA_VISIBLE_DEVICES`。工具不安装或升级包，也不从版本字符串推断安装来源。

```bash
python tools/run_accelerate_gate.py --fresh-process --device cpu \
  --jittor-python <jittor-python> --oracle-python <real-torch-python> \
  --upstream "$JITTOR_LAB_ROOT/accelerate-validation/upstream" \
  --wheel "$JITTOR_ACCELERATE_WHEEL" \
  --cache-root "$JITTOR_LAB_ROOT/_state/accelerate-protocol/benchmark-cache" \
  --output "$JITTOR_LAB_ROOT/_state/accelerate-protocol/cpu"
python tools/run_accelerate_gate.py --fresh-process --device cuda --devices <gpu0>,<gpu1> \
  --jittor-python <jittor-python> --oracle-python <real-torch-python> \
  --upstream "$JITTOR_LAB_ROOT/accelerate-validation/upstream" \
  --wheel "$JITTOR_ACCELERATE_WHEEL" \
  --cache-root "$JITTOR_LAB_ROOT/_state/accelerate-protocol/benchmark-cache" \
  --output "$JITTOR_LAB_ROOT/_state/accelerate-protocol/cuda"
python agent/skills/torch-compat-repo-runbook/scripts/collect_accelerate_scorecards.py \
  --run "$JITTOR_LAB_ROOT/_state/accelerate-protocol/cpu" \
  --run "$JITTOR_LAB_ROOT/_state/accelerate-protocol/cuda" \
  --upstream "$JITTOR_LAB_ROOT/accelerate-validation/upstream" \
  --baseline <full-measured-source-SHA> --report <commit-pinned-historical-report-URL> \
  --command '<exact measurement commands>' \
  --output "$JITTOR_LAB_ROOT/_state/accelerate-protocol/scorecard"
```

先冻结并提交工具源码，再采集。CPU 显式隐藏 CUDA；两 rank CUDA 每种精度/输入种子独立
launcher，rank 缓存逐个预热，benchmark 与单测缓存隔离。实际进程绑定完整上游包源码、
实际安装包所有文件、二进制 Torch 身份、平台、精度开关、所有输入/参数梯度、Adam moments、
逐步轨迹和四路 RNG。采集不覆盖旧产物。

每个真实 producer/consumer 绑定官方 wheel 的固定 SHA、完整安装包文件、同侧实际
METADATA/RECORD 原始字节和导入根；auditor 重读并逐项核验。保留原 wheel 和私有安装
前缀以便复核。旧 raw 缺少这些实际字段时重新测量，不能补写、仅添加同名字段或改版本
字符串冒充新的安装来源证明。

共享 schema 用 `record_kind=protocol`、显式 `axes=[]/registry_refs=[]`，单独展示协议及
world/precision/rank，不产生注册表分子或分母。CPU world1 只匹配 CUDA world1；两 rank
CUDA 缺 CPU world2 对应时保留独立数值结论，但进入 `blocked`，准入 `level=null`，
不进入 `accepted`。own-runtime 精确恢复不能代替独立跨 runtime 数值
一致；保留原 `atol=2e-6, rtol=2e-5`，失败行 `level=null`，不得靠放宽预算授级。

多次采集使用不同 `--output`；可用 `--cache-root <external-benchmark-root>` 复用任务自己的
设备/rank benchmark 缓存，不能指向单测缓存。协议 dispatcher 只允许明确的 repo/id；
专用 validator 尚未存在或 scope 不匹配时拒绝。配对比较完整设备无关协议描述，不能将
不同 world、precision、ZeRO stage 或训练规格配成一对；失败/阻塞不填 verified level。

### Full-checkpoint TRL SFT probe

`scripts/trl_sft_probe.py` observes the official TRL 0.21.0 `SFTTrainer.train()`
with Transformers 4.56.2 and Accelerate 1.10.1. Use a pinned, local
`Qwen/Qwen3-0.6B` checkpoint, including its tokenizer, in independent runtimes.
All parameters remain trainable; the default is three FP32 SGD updates at
`1e-4`, batch one, eager attention, and at most 32 tokens. One fixed text record
is repeated so sampler differences cannot change the training inputs. This is
a bounded compatibility experiment, not a convergence or throughput benchmark.

Prepare the normal isolated environments first. Run CPU before the accelerator;
use separate device caches and serialize first compilation. With
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, run the oracle first to create the
shared nonpersistent buffer artifact (the checkpoint omits RoPE frequencies):

```bash
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/trl_sft_probe.py run \
  --root "$PWD" --model "$JITTOR_LAB_ROOT/<checkpoint>" \
  --upstream "$JITTOR_LAB_ROOT/<trl-checkout>" \
  --initial-buffers "$JITTOR_LAB_ROOT/_state/<run>/buffers.npz" \
  --runtime torch --device cpu --steps 3 \
  --output "$JITTOR_LAB_ROOT/_state/<run>/torch-cpu"

<jittor-python> agent/skills/torch-compat-repo-runbook/scripts/trl_sft_probe.py run \
  --root "$PWD" --model "$JITTOR_LAB_ROOT/<checkpoint>" \
  --upstream "$JITTOR_LAB_ROOT/<trl-checkout>" \
  --initial-buffers "$JITTOR_LAB_ROOT/_state/<run>/buffers.npz" \
  --runtime jittor --device cpu --steps 3 \
  --output "$JITTOR_LAB_ROOT/_state/<run>/jittor-cpu"

<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/trl_sft_probe.py compare \
  --reference "$JITTOR_LAB_ROOT/_state/<run>/torch-cpu" \
  --candidate "$JITTOR_LAB_ROOT/_state/<run>/jittor-cpu" \
  --output "$JITTOR_LAB_ROOT/_state/<run>/comparison.json"
```

Pin the official TRL checkout to `46d09bd2408f17605409fb3ee8ba12705add7faa`.
Providing `--upstream` verifies the installed TRL package against that clean
source. Reuse the exact buffer artifact and model for CUDA, with new output
paths and `--device cuda`. The output directory must not already exist.

The probe writes every parameter gradient and updated parameter one tensor at
a time, plus actual model inputs, logits, loss and an embedding-output gradient.
Three updates require roughly 14 GB per runtime. It checks exact initial state,
input dtypes, tensor coverage, device residency, hashes and zero fallback;
missing gradients or changed artifacts fail closed. Host snapshots add I/O and
synchronization, so reported elapsed times are diagnostic only. The probe does
not test streaming datasets, checkpoint resume, AdamW, mixed precision, LoRA or
distributed training, and does not grant registry coverage or nightly admission.

### Full-checkpoint TRL DPO and GRPO probes

`scripts/trl_rl_probe.py` observes official TRL 0.21.0 DPO/GRPO training with the
same full Qwen3-0.6B checkpoint, downstream pins and independent environments
as SFT. Prepare the shared oracle-created `initial-buffers.npz` first; this
probe requires it to exist. Each formal run uses ten distinct examples and ten
FP32 SGD updates at `1e-4`, with all 310 parameter tensors trainable.

Use private CPU/CUDA caches and offline environments. Set `OMP_NUM_THREADS=4`,
`MKL_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=4`, `DISABLE_MULTIPROCESSING=1`,
`JT_USE_PARALLEL_OP_COMPILER=0` and `JT_CUDA_KERNEL_MATH=strict` for the recorded
configuration. CPU must hide accelerators and disable CUDA/NVCC; CUDA must
explicitly select the actual device/toolkit. First JIT compilation stays serial.

From the checkout, run the oracle before its matching candidate:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=0 \
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/trl_rl_probe.py \
  --root "$PWD" --algorithm dpo --device cpu --runtime torch --steps 10 --threads 4 \
  --model "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/model" \
  --initial-buffers "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/initial-buffers.npz" \
  --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/dpo-torch-cpu"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
<jittor-python> agent/skills/torch-compat-repo-runbook/scripts/trl_rl_probe.py \
  --root "$PWD" --algorithm dpo --device cpu --runtime jittor --steps 10 --threads 4 \
  --model "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/model" \
  --initial-buffers "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/initial-buffers.npz" \
  --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --reference "$JITTOR_LAB_ROOT/_state/<new-run>/dpo-torch-cpu" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/dpo-jittor-cpu"
```

Repeat with `--algorithm grpo` and distinct `grpo-*` output/reference paths.
After CPU, repeat both pairs with `--device cuda`, a separate CUDA cache and
distinct `*-cuda` paths. Never pair different devices or reuse an existing
output directory. Keep probe sources unchanged across each pair; their hashes
are checked. Read both `status` and `numerical_passed`: execution can complete
while numerical comparisons fail. This probe does not use exit status alone
to signal numerical acceptance failure.

DPO uses one preference pair per update, beta `0.1` and sigmoid loss. Official
reference log probabilities are precomputed for **both** train and evaluation
data before updates. Evaluation uses the same ten training pairs, not held-out
quality data.

GRPO uses one prompt/two completions per update, beta `0.04`, explicit `grpo`
loss, a resident reference model and sixteen-token completion limit. The oracle
samples real completions and records effective configuration with
`use_model_defaults=False`; the candidate verifies/replays those exact tokens.
Official reward normalization, KL, policy loss and gradients still execute.
Batch rows are canonically reordered together to remove backend RNG ordering
differences. Actual character/length rewards and nonzero reward-group counts
are recorded; this is not semantic accuracy or independent online sampling
parity. Final fixed-completion evaluation compares each group's own pre-update
log probabilities with the final policy, so before values come from different
policy states. Separate native generation smoke needs its own provenance and
does not replace controlled numerical comparison.

The training probe compares every step's complete parameter gradients, final
weights, losses and method-specific observations. It does not collect full
model logits, input/embedding gradients or per-step updated-weight snapshots.
Budgets remain `atol=1e-5, rtol=1e-4` with the documented field-scale floor.
Report failed elements alongside category relative L2; final weight error
should also be divided by the actual update norm. Additional trained-model
logit comparisons must load and verify each runtime's saved final weights and
use identical tokenized prompts, not independently sampled answers.

Oracle traces retain roughly 26 GB each; candidates stream-compare gradients
and retain final weights plus hashes/statistics. These asymmetric I/O costs
rule out a speed comparison. Keep `scorecard_eligible=false`: full model size
and ten completed steps do not grant an L level or library-wide compatibility.
Protocol, results and boundaries are in the
DPO/GRPO report (historical validation record; see PR evidence).

### Compare trained DPO/GRPO model logits

`scripts/trl_final_logits_probe.py` reads the completed training traces above,
loads each runtime's own full final parameter set, verifies all saved-array
hashes and reads parameters back after copying to verify exact loaded bytes.
The complete 310-parameter inventory, shared buffers, model/tokenizer artifacts,
training protocol and runtime/device identities must match. It does not retrain
or substitute the original checkpoint weights for trained weights.

Run genuine Torch first, then Jittor, in the same isolated environments as
training. The following CPU example uses the DPO trace paths above. Existing
output directories are rejected:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=0 \
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/trl_final_logits_probe.py \
  --root "$PWD" --runtime torch --device cpu --threads 4 \
  --model "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/model" \
  --initial-buffers "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/initial-buffers.npz" \
  --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --training-trace "$JITTOR_LAB_ROOT/_state/<new-run>/dpo-torch-cpu" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/dpo-logits-torch-cpu"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
<jittor-python> agent/skills/torch-compat-repo-runbook/scripts/trl_final_logits_probe.py \
  --root "$PWD" --runtime jittor --device cpu --threads 4 \
  --model "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/model" \
  --initial-buffers "$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/initial-buffers.npz" \
  --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --training-trace "$JITTOR_LAB_ROOT/_state/<new-run>/dpo-jittor-cpu" \
  --reference "$JITTOR_LAB_ROOT/_state/<new-run>/dpo-logits-torch-cpu" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/dpo-logits-jittor-cpu"
```

There is no `--algorithm` argument: the method comes from `--training-trace`.
`--reference` is the independent oracle **logit** output, not its training trace.
Directories or explicit `manifest.json` paths are accepted for both. Repeat for
GRPO by changing the trace/output paths; repeat on CUDA using `--device cuda`
and the completed same-device training traces. Preserve separate device caches
and serialize first compilation as before.

For all ten identical tokenized prompts, the probe saves complete `[1,T,V]`
logits and next-token `[V]` vectors, with `V=151936`, in FP32/eager eval mode,
no sampling and no cache. It checks actual physical input/output residency and
zero fallback. Candidate manifests report maximum absolute error, RMSE,
relative L2, strict-budget failed fields/elements, top-1 agreement and FP64
softmax `KL(P_Torch || P_Jittor)`. Read `comparison.logits` for all prompt
positions and `comparison.next_logits` for the ten next-token decisions;
their top-1 agreement denominators differ.

The resulting discrepancy combines **training and inference** differences,
because the two models load their own trained weights. It is not a same-weight
operator test or a claim about sampled-answer quality. `status=complete` does
not imply zero strict-budget failures, and `scorecard_eligible` stays false.


### TRL trainer inventory and bounded breadth probe

`scripts/trl_inventory.py` derives all public Trainer exports, including
conditional exports, from the clean pinned TRL checkout without importing it.
Its default selection is the original eleven-method language-model subset.
The maintained gate below explicitly selects sixteen text Trainers from the
full 18-Trainer denominator. Inventory/selection counts alone are not numerical
support counts; the two excluded conditional exports are the deprecated
diffusion/image Trainers `AlignPropTrainer` and `DDPOTrainer`.

```bash
python agent/skills/torch-compat-repo-runbook/scripts/trl_inventory.py \
  --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/inventory.json"
```

`scripts/trl_mainstream_probe.py` executes official Trainer training on a
randomly initialized two-layer Qwen3 (hidden size 32, vocabulary 256), with a
local synthetic tokenizer and ten distinct input sequences. It does not load
the pretrained 0.6B checkpoint. Supported experiment selectors are `sft`,
`dpo`, `kto`, `orpo`, `cpo`, `reward`, `ppo`, `rloo`, `gkd`, `online_dpo`, `grpo`.
A selector being available is not evidence that its runtime comparison passes.

Use the same isolated interpreter/offline/device environment as the full-model
probes. First run the genuine oracle, then the candidate, using new output
paths for every method/device pair:

```bash
<oracle-python> agent/skills/torch-compat-repo-runbook/scripts/trl_mainstream_probe.py \
  --root "$PWD" --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --runtime torch --device cpu --algorithm kto --steps 10 \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/kto-torch-cpu"

<jittor-python> agent/skills/torch-compat-repo-runbook/scripts/trl_mainstream_probe.py \
  --root "$PWD" --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --runtime jittor --device cpu --algorithm kto --steps 10 \
  --reference "$JITTOR_LAB_ROOT/_state/<new-run>/kto-torch-cpu" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/kto-jittor-cpu"
```

Repeat on CUDA only with a same-device genuine oracle and explicit accelerator
environment. Serialize each cache's users and initial compilation. Initial
parameters and buffers for every model role are transferred from the oracle
and read back byte-for-byte. Every optimizer step records all trained parameter
gradients, retained embedding-output gradients and updated parameters; fixed
model state, observed outputs, final evaluation outputs and real device dispatch
are checked. PPO trains its policy and value models. Reward uses a scalar head,
so its output is not a vocabulary logit vector. Per-method metadata records
actual batch size, rollouts and sample exposure; ten updates need not mean ten
examples processed (KTO uses batch two).

Online methods run local generation forwards and scores. During generation,
candidate multinomial token choices are replayed from the oracle to keep later
training inputs identical. This is controlled-token numerical testing, not
independent sampling parity or model quality evidence. Source/initial-state
identities must match; changing a measured probe requires a new pair. Completion
and numerical acceptance are separate manifest fields: inspect
`status`, `optimizer_steps`, `fallback_count`, `numerical_passed`, and individual
`comparison` failures. A process exit of zero alone is not numerical acceptance.
The report keeps tiny breadth and full-checkpoint evidence separate:
TRL mainstream experiments (historical validation record; see PR evidence).


### Maintained sixteen-text-Trainer gate and extra five selectors

[`tools/run_trl_gate.py`](../../../tools/run_trl_gate.py) is the maintained
entry; [the CPU workflow](../../../.github/workflows/trl-compat-cpu.yml) schedules
it nightly, supports manual dispatch and scopes PR runs to relevant paths.
The canonical local wrapper is `nox -s ecosystem_trl`: set
`JITTOR_CI_PYTHON`, `REAL_TORCH_PYTHON`, `JITTOR_TRL_CHECKOUT` and exactly
`JITTOR_TEST_DEVICES=cpu` or `cuda`. It fixes the complete sixteen-method,
three-update matrix; only fresh output/exclusive cache paths may be overridden.
`registry_inventory.py --repo trl` reuses the same full AST denominator, and
`audit_coverage.py audit` discovers this dedicated CLI without granting an L
level or pretending it is a model-builder case.
Provision two independent Python 3.11 virtual environments using
[`requirements/trl-compat.txt`](../../../requirements/trl-compat.txt): TRL
0.21.0, Transformers 4.56.2, Accelerate 1.10.1, datasets 3.6.0, PEFT 0.17.1 and
NumPy 1.26.4. The oracle must import genuine Torch 2.8.0 with its own native
binary. Install core editable with `--config-settings editable_mode=compat`
and the separate `compat` package only in the candidate; do not activate its
shim in the oracle. The tool neither installs packages nor downloads weights.

```bash
<oracle-python> tools/run_trl_gate.py \
  --jittor-python <jittor-python> --oracle-python <oracle-python> \
  --upstream "$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream" \
  --output "$JITTOR_LAB_ROOT/_state/<new-run>/trl-gate-cpu" \
  --jittor-home "$JITTOR_LAB_ROOT/_state/<leased-cpu-cache>" \
  --device cpu --steps 3
```

The upstream must be clean official TRL at
`46d09bd2408f17605409fb3ee8ba12705add7faa`; output must be fresh and outside the
checkout. Default is CPU and three actual updates. All sixteen genuine-Torch
references are generated afresh before the source-frozen candidate batch.
The gate serially leases the optional cache, checks private environments and
pins, zero fallback, actual optimizer calls, every trained gradient/parameter
and embedding-gradient role at every step, original initialization payloads,
final prompts, retained array hashes and recomputed original-budget comparisons.
Two identically incomplete traces cannot pass. Unused Jittor runtime-context
changes do not invalidate an otherwise independent Torch reference; candidate
runtime sources are frozen and checked separately. Only pre-measurement
`jit_utils` rebuild exit 3 permits a documented warm-up restart; training
failures are preserved, not retried into an existing trace.

Inspect `summary.json`, `inventory.json`, environment/binary proofs,
`producer-sources.json`, `runtime-sources.json`, per-command receipts and
`audits/`. Success requires sixteen paired strict passes, rather than sixteen
successful imports. A CUDA run additionally requires `--device cuda
--cuda-device <allocated-gpu>` and an exclusive CUDA cache; the scheduled
workflow covers CPU only. All of this remains random tiny FP32 text coverage,
not pretrained full-model, mixed-precision, speed or diffusion acceptance.

The five additional text selectors live in
[`trl_tail_probe.py`](scripts/trl_tail_probe.py): `prm`, `bco`, `iterative_sft`,
`nash_md`, `xpo`. To diagnose one separately, use the same upstream/environment
and oracle-first pattern as `trl_mainstream_probe.py`, replacing the script
and selector, for example:

```bash
trl_scripts="$PWD/agent/skills/torch-compat-repo-runbook/scripts"
trl_state="$JITTOR_LAB_ROOT/_state/<new-run>"
trl_upstream="$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream"
<oracle-python> "$trl_scripts/trl_tail_probe.py" \
  --root "$PWD" --upstream "$trl_upstream" --runtime torch --device cpu \
  --algorithm nash_md --steps 10 --output "$trl_state/nash-torch/trace"
<jittor-python> "$trl_scripts/trl_tail_probe.py" \
  --root "$PWD" --upstream "$trl_upstream" --runtime jittor --device cpu \
  --algorithm nash_md --steps 10 --reference "$trl_state/nash-torch/trace" \
  --output "$trl_state/nash-jittor/trace"
```

PRM trains token-level reward outputs, not vocabulary logits.
NashMD/XPO execute two generation calls per update, including NashMD's actual
geometric-mixture forward; only discrete choices are replayed. This does not
establish independent sampling parity or every optional BCO/online setting.


### Diagnose full-model numerical discrepancies

Use `scripts/trl_rounding_probe.py` for a same-checkpoint, archived step-zero
DPO/GRPO diagnostic, then `scripts/trl_rounding_compare.py` to hash-check and
compare observations. Start with FP32 CPU/CUDA, and use `--precision fp64` as
a precision control: it removes Qwen's internal FP32 casts and explicitly
types constants to avoid KI-DTYPE-003. These diagnostic overrides do not grant
production FP64 support. `--eager-rms` is a process-local fusion ablation.

Preserve common reference probabilities, original input/weight hashes, zero
fallback and actual devices; do not substitute an unrelated tiny model. The
probe retains full gradients of selected blocks/norms and samples the rest.
Do not describe those samples as all gradient elements. Read the comparison's
unmatched fields and distinguish full/sampled gradient summaries. Use the
existing offline interpreter/cache isolation and new output directories.
Measured commands, limitations and current conclusions are in the
rounding diagnosis (historical validation record; see PR evidence).

### Full-checkpoint TRL method matrix with bounded trace storage

[`trl_full_probe.py`](scripts/trl_full_probe.py), with
[`trl_full_cases.py`](scripts/trl_full_cases.py), runs official trainers for
`kto`, `orpo`, `cpo`, `reward`, `ppo`, `rloo`, `gkd`, and `online_dpo` on the full
local Qwen3-0.6B checkpoint. Keep the same pinned downstream stack, offline
settings, independent interpreters, source binding, shared buffers and explicit
CPU/CUDA selection described above. New output directories are mandatory.
CPU precedes CUDA; serialize users of each JIT cache and allocate devices
explicitly. These are FP32 SGD probes, not LoRA/AdamW probes.

From the repository root, set task-specific paths (the shared buffers must
already exist), then run each method's oracle before its candidate:

```bash
trl_scripts="$PWD/agent/skills/torch-compat-repo-runbook/scripts"
trl_state="$JITTOR_LAB_ROOT/_state/<new-run>"
trl_model="$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/model"
trl_buffers="$JITTOR_LAB_ROOT/_state/trl-qwen3-sft/initial-buffers.npz"
trl_upstream="$JITTOR_LAB_ROOT/trl-qwen3-sft/upstream"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=0 \
<oracle-python> "$trl_scripts/trl_full_probe.py" \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --algorithm orpo --steps 10 --threads 4 \
  --learning-rate 1e-4 --runtime torch --device cpu \
  --output "$trl_state/orpo-torch-cpu/trace"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
<jittor-python> "$trl_scripts/trl_full_probe.py" \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --algorithm orpo --steps 10 --threads 4 \
  --learning-rate 1e-4 --runtime jittor --device cpu \
  --reference "$trl_state/orpo-torch-cpu/trace" \
  --output "$trl_state/orpo-jittor-cpu/trace"

<oracle-python> "$trl_scripts/trl_full_collect.py" \
  --reference "$trl_state/orpo-torch-cpu/trace" \
  --candidate "$trl_state/orpo-jittor-cpu/trace" \
  --output "$trl_state/orpo-cpu-audit.json"
```

Replace the algorithm and all corresponding paths together. For CUDA repeat
the pair with `--device cuda`, separate output paths and the allocated CUDA
cache/device. The formal protocol uses ten updates; `--steps 1` is only a
smoke test. Ten distinct records do not imply ten total exposures: inspect
per-method batch/rollout metadata. Reward/value heads and GKD's perturbed
teacher are synchronized diagnostic fixtures, not independently trained
quality models. Online methods replay oracle multinomial choices while
executing local generation forwards and scores; this does not establish
independent sampling parity. Multi-model memory requirements can block a
method before training; such a run is not a passed matrix cell.

This storage protocol samples intermediate gradients and updated parameters
at stride 4096, while checking complete gradient finiteness/norms. It compares
all final-step parameter gradients and final parameters; only the oracle
retains those bulk payloads. Candidate manifests retain hashes/comparison
statistics for them. It is **not all-elements, all-steps gradient coverage**.
The collector verifies original payloads or completed lossless archives and
reconciles recorded comparisons; it cannot independently reconstruct omitted
candidate payloads from their hashes. Inspect completed status, real optimizer
updates, device/fallback checks and numerical failures separately. Keep the
original `atol=1e-5, rtol=1e-4`; the tools do not award registry coverage,
nightly admission, quality/convergence claims or throughput results.

### Full-model PPO with policy/value LoRA

[`trl_ppo_lora_probe.py`](scripts/trl_ppo_lora_probe.py), with
[`trl_ppo_lora_case.py`](scripts/trl_ppo_lora_case.py), defines a separate
FP32 AdamW profile using three complete Qwen3-0.6B bases. Official PPO disables
policy adapters for the reference instead of loading a fourth model. Policy
and value use rank-8 q/v LoRA; the value score head is also trained, for 225
tensors/2,294,784 parameters. The reward head is a synchronized diagnostic
fixture. This does not replace the full-parameter PPO capacity result.

Use the path variables, isolated pinned interpreters and offline environment
from the full-checkpoint example. First execute the one-update CPU prerequisite:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=0 \
<oracle-python> "$trl_scripts/trl_ppo_lora_probe.py" \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --algorithm ppo --steps 1 --threads 4 \
  --runtime torch --device cpu --output "$trl_state/ppo-lora-torch-cpu/trace"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
<jittor-python> "$trl_scripts/trl_ppo_lora_probe.py" \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --algorithm ppo --steps 1 --threads 4 \
  --runtime jittor --device cpu --reference "$trl_state/ppo-lora-torch-cpu/trace" \
  --output "$trl_state/ppo-lora-jittor-cpu/trace"
```

After checking that pair, repeat both commands with `--device cuda --steps 10`,
fresh CUDA output/reference paths, the same explicitly allocated physical GPU
and an exclusively leased CUDA `JITTOR_HOME`. Use the selected toolkit's explicit
compiler path consistently. The CPU control visits two prompts once; CUDA uses
five two-prompt rollouts and two minibatches, hence ten actual optimizer calls.
All initial adapters/heads are exported by the oracle under `initial/`; the
candidate verifies and loads them, without assuming matching random generators.

Every trained gradient, parameter and Adam state is retained at every actual
call. Native generation executes in both runtimes; only sampled discrete tokens
are replayed. Scalar losses are observed at official `accelerator.backward`.
The candidate manifest's `comparison` and `numerical_passed` distinguish execution
completion from strict parity; do not turn exit 0 into a numerical pass. Keep the
original tolerance. CPU's one-minibatch upstream `val/ratio_var` is undefined,
while observed loss/gradients/state must remain finite. This probe does not test
checkpoint resume, model quality or throughput. Exact measured sources and
results are described in the PPO/loss report (historical validation record; see PR evidence).

### CUDA generation index diagnostics

[`trl_generation_index_probe.py`](scripts/trl_generation_index_probe.py) observes
boolean-mask gathers in the native generation path. Run `mask` first in the
independent Torch environment, then with `--runtime jittor --reference` pointing
to that output, using the same `--batch-size` and `--repetitions`. Both commands
require `--root`, `--device cuda` and a new external `--output` directory.
`--observe metadata` records shape/dtype/device; `--observe values` also
synchronizes and retains the actual index values. Observation can change race
timing and therefore cannot establish that an intermittent failure is absent.

`--boundary-stress --require-async` varies bounded allocations before the mask
operation and rejects a candidate run unless the actual async executor runs.
It changes no runtime flags and generates only valid indices. Keep its samples,
source hashes, device checks and zero-fallback result together. `ppo` additionally
requires `--upstream`, an explicit observation level and the maintained official
PPO producer; its original numerical contracts remain unchanged. A passing
stress run is supporting evidence, not attribution of a historical CUDA fault.

### Practical SFT: LoRA, AdamW and fresh-process checkpoint resume

[`trl_practical_probe.py`](scripts/trl_practical_probe.py) uses official
`SFTTrainer` and PEFT 0.17.1 on the complete Qwen3-0.6B base, with only rank-8
`q_proj`/`v_proj` adapters trainable (alpha 16, dropout 0). It runs ten distinct
short texts, batch one, maximum length 32, AdamW learning rate `1e-4`, weight
decay `0.01` and a ten-step linear schedule. The oracle creates the shared
initial-adapter NPZ; both sides verify identical base/adapters/buffers and
frozen base parameters. Inspect PEFT dependencies before installation; never
let dependency resolution replace the shim with a genuine Torch wheel.

Using the path variables above, run the oracle first, then the candidate:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=0 \
<oracle-python> "$trl_scripts/trl_practical_probe.py" run \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --initial-adapters "$trl_state/adapters.npz" \
  --runtime torch --device cpu --precision fp32 \
  --output "$trl_state/practical-torch-cpu"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
<jittor-python> "$trl_scripts/trl_practical_probe.py" run \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --initial-adapters "$trl_state/adapters.npz" \
  --runtime jittor --device cpu --precision fp32 \
  --output "$trl_state/practical-jittor-cpu"

<oracle-python> "$trl_scripts/trl_practical_compare.py" \
  --reference "$trl_state/practical-torch-cpu" \
  --candidate "$trl_state/practical-jittor-cpu" \
  --output "$trl_state/practical-cpu-comparison.json"
```

The trace captures every trainable gradient, adapter update and Adam moment/
step, fixed-prompt full-vocabulary logits, scheduler/scaler state and RNG
observations. The posthoc comparator independently requires all 112 adapters
(1,146,880 trainable parameters), 5,649 full-run or 2,829 resumed-run fields,
all actual optimizer steps and nonempty per-step state. Two equally incomplete
manifests cannot pass. Resume also binds and rehashes the reference's actual
checkpoint-five files; comparing a directory with itself is rejected. Collector
source hashes are recorded separately from the measured producer hashes.
Preserve archived producer versions when hardening posthoc comparison; this
is a re-audit of existing traces, not a new training measurement. Cross-runtime comparison normalizes **only scalar integer Adam
step-counter values** to float64 for exact value comparison; original dtype
mismatches remain explicit and prevent strict API parity. Other dtypes and
numerical budgets are unchanged. A same-runtime uninterrupted repeat control
uses `trl_practical_compare.py --repeat` with two separate completed runs.

For each runtime, launch a **new process** with the same run arguments and a
new output directory, adding its own saved checkpoint path. For example:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
<jittor-python> "$trl_scripts/trl_practical_probe.py" run \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --initial-adapters "$trl_state/adapters.npz" \
  --runtime jittor --device cpu --precision fp32 \
  --resume "$trl_state/practical-jittor-cpu/trainer/checkpoint-5" \
  --output "$trl_state/practical-jittor-cpu-resume"

<oracle-python> "$trl_scripts/trl_practical_compare.py" --resume \
  --reference "$trl_state/practical-jittor-cpu" \
  --candidate "$trl_state/practical-jittor-cpu-resume" \
  --output "$trl_state/practical-jittor-cpu-resume-comparison.json"
```

Repeat for Torch with its own checkpoint. Both uninterrupted and resumed
training retain `max_steps=10`; never shorten the first schedule to five.
Resume must reproduce steps 6–10, including inputs, Adam state, scheduler,
scaler and RNG; same-runtime comparison requires exact values. This does not
claim cross-runtime optimizer-checkpoint interoperability. If CUDA resume
is inexact, compare fresh uninterrupted repeats before attributing the
entire difference to checkpoint restoration.

After CPU FP32, repeat in CUDA with `--precision fp32`, `fp16`, or `bf16`, new
paths and matching precision in both runtimes and their resumes. FP16 uses
`--initial-loss-scale 65536` by default. A diagnostic alternative such as
`--initial-loss-scale 1024` must be set identically and recorded as a different
protocol, preserving the original failed/skipped-step run. Count actual
optimizer updates, not merely Trainer global steps: overflow can skip updates.
Record module output dtypes/physical devices, not just configuration flags.
Native BF16 acceleration and functional CUDA BF16 support are different
capabilities; test the latter directly and report both. No multi-GPU,
DeepSpeed, quantized-LoRA or full-parameter AdamW coverage follows from these
single-device adapter experiments.


### Expanded full-model LoRA methods

[`trl_lora_probe.py`](scripts/trl_lora_probe.py) extends the practical observer
to official `sft`, `dpo` and `grpo` Trainers. With `trl_scripts`, `trl_state`,
`trl_model`, `trl_buffers` and `trl_upstream` set as in the full-checkpoint
example, run an independent pair with identical adapters and arguments:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=0 \
<oracle-python> "$trl_scripts/trl_lora_probe.py" run \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --initial-adapters "$trl_state/adapters.npz" \
  --algorithm sft --gradient-accumulation 2 --max-length 128 \
  --runtime torch --device cpu --precision fp32 \
  --output "$trl_state/lora-sft-torch"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
<jittor-python> "$trl_scripts/trl_lora_probe.py" run \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --initial-adapters "$trl_state/adapters.npz" \
  --algorithm sft --gradient-accumulation 2 --max-length 128 \
  --runtime jittor --device cpu --precision fp32 \
  --reference "$trl_state/lora-sft-torch" --output "$trl_state/lora-sft-jittor"
<oracle-python> "$trl_scripts/trl_lora_compare.py" \
  --reference "$trl_state/lora-sft-torch" --candidate "$trl_state/lora-sft-jittor" \
  --initial-adapters "$trl_state/adapters.npz" --output "$trl_state/lora-sft-comparison.json"
```

The oracle creates the adapter archive if absent; both sides verify it. The
protocol always uses ten updates and checkpoint 5. `--gradient-accumulation`
accepts 1/2 and `--max-length` 32/128; unequal SFT microbatch lengths exercise
upstream token-loss handling without replacement normalization. DPO/GRPO use
their actual official objectives, with checked discrete-token replay for GRPO.
For CUDA allocate its device/cache explicitly and use matching `--device cuda`
and `--precision fp32|fp16|bf16`; FP16's default initial scale here is 1024.
Fresh-process own-checkpoint continuation adds `--resume
<own-run>/trainer/checkpoint-5` to the unchanged run arguments and a fresh output;
compare it against that runtime's full run with `trl_lora_compare.py --resume`.
`--repeat` audits separate uninterrupted repeats. Inspect strict failures,
full adapter gradients/state and actual updates, not only completed status.

The 2026-10-25 report (historical validation record; see PR evidence)
separates tiny sixteen-method strict passes from full-model FP32 and AMP
cross-runtime strict failures. After the nested-SiLU and frozen mixed-RMS
repairs, measured FP16/BF16 own-runtime resume is exact in 2,829 fields each;
full cross-runtime training remains open. An actual first-layer FP32 RMS
replay is within three ULPs of the FP64 ideal in both runtimes; it does not
explain every final ten-update discrepancy.

### Two-rank CUDA DDP and checkpoint continuation

[`trl_ddp_probe.py`](scripts/trl_ddp_probe.py) launches a genuine Torch oracle
and candidate on exactly two explicitly allocated GPUs. Lease two distinct,
existing external rank caches first; the launcher warms them serially before
measurement. Its fixed scope is complete Qwen3-0.6B FP32 LoRA SFT, ten updates,
rank-distinct inputs, real DDP/NCCL and fresh checkpoint-5 continuation.

```bash
<oracle-python> "$trl_scripts/trl_ddp_probe.py" launch \
  --root "$PWD" --state "$trl_state/ddp" --model "$trl_model" \
  --upstream "$trl_upstream" --initial-buffers "$trl_buffers" \
  --initial-adapters "$trl_state/adapters.npz" \
  --oracle-python <oracle-python> --jittor-python <jittor-python> \
  --devices '<allocated-gpu-a>,<allocated-gpu-b>' --nvcc '<nvcc-path>' \
  --rank-cache-home "$JITTOR_LAB_ROOT/_state/<leased-rank0-cache>" \
  --rank-cache-home "$JITTOR_LAB_ROOT/_state/<leased-rank1-cache>"
```

`--runtime torch|jittor|both` defaults to both. `--append-runtime` is only for
adding the missing runtime to the same preserved state, not retrying existing
measurements. Read `measurements/<runtime>/{full,resume}/rank*/manifest.json`,
`<runtime>-resume-comparison.json` and `cross-comparison.json`. The collector
checks two actual ranks, full adapter/gradient/moment/counter inventories,
rank-consistent trained parameters, distinct inputs, physical devices, zero
fallback and exact own-runtime resume. Launch completion alone does not imply
cross-runtime strict acceptance. To re-audit without training, use `compare
--reference <full-run> --candidate <other-run> --mode cross|resume --output
<new-json>`. This is not DeepSpeed/FSDP, arbitrary world-size or distributed
mixed-precision coverage.

### Matched SFT throughput, separate from correctness

[`trl_throughput_probe.py`](scripts/trl_throughput_probe.py) uses official
full-model LoRA SFT with accumulation two and the fixed unequal 128/39–40-token
sequence schedule. Reserve one otherwise idle physical GPU and one exclusive
candidate cache. Prime all profiles before timing:

```bash
<oracle-python> "$trl_scripts/trl_throughput_probe.py" driver \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --initial-adapters "$trl_state/adapters.npz" \
  --oracle-python <oracle-python> --jittor-python <jittor-python> \
  --gpu '<allocated-gpu>' --nvcc '<nvcc-path>' \
  --jittor-home "$JITTOR_LAB_ROOT/_state/<leased-throughput-cache>" \
  --stage prime --state "$trl_state/throughput"
```

Repeat the identical command with `--stage measure`. The driver runs FP32,
FP16 and BF16, three paired repeats in alternating runtime order. Each measured
row has five warm-up and twenty timed actual optimizer calls; timings reject
new/changed JIT binaries and skipped updates. Token rates distinguish input
nonpadding tokens from causal prediction targets. Source/device/fallback,
frozen base and all 112 adapter gradients/states are checked outside the timed
interval. Inspect the completed `summary.json` and per-row `result.json`,
including hardware, versions and cache provenance. Existing-cache setup is
explicitly not a genuine cold-start measurement. Throughput does not grant
cross-runtime numerical parity, model-quality or allocator-memory claims;
retain the separate full correctness and continuation comparisons.

### Replay actual AMP backward boundaries

[`trl_backward_boundary_probe.py`](scripts/trl_backward_boundary_probe.py) captures
one genuine-Torch backward from the full practical SFT profile and intentionally
stops before the first optimizer update. Use a complete matching formal oracle
trace; FP16 capture uses loss scale 1024. Capture requires every adapter gradient
to match that formal reference exactly.

```bash
JITTOR_TORCH_SHIM=0 <oracle-python> "$trl_scripts/trl_backward_boundary_probe.py" capture \
  --root "$PWD" --model "$trl_model" --upstream "$trl_upstream" \
  --initial-buffers "$trl_buffers" --initial-adapters "$trl_state/adapters.npz" \
  --reference "$trl_state/formal-torch-fp16/trace" --precision fp16 \
  --output "$trl_state/backward-fp16-capture"
<jittor-python> "$trl_scripts/trl_backward_boundary_probe.py" replay \
  --root "$PWD" --runtime jittor --source "$trl_state/backward-fp16-capture" \
  --output "$trl_state/backward-fp16-jittor"
```

Select one allocated CUDA device and an exclusive cache through the environment,
as for the practical producer. Replay also accepts `--runtime torch` for the
independent control and `--kinds lora_B rms lm_head` to select boundaries. It uses
identical captured activations/weights/cotangents and attaches no native module
hooks. A complete diagnostic is not complete training or quality acceptance.

### Isolate CUDA loss and local DDP components

[`trl_ddp_local_probe.py`](scripts/trl_ddp_local_probe.py) separates a repeated
cross-entropy calculation from full-model local backward replay. Its measured
`ce` route supports CPU/CUDA and retains loss, full input gradients and native
saved LSE; use independent runtimes on the same explicitly allocated device:

```bash
<oracle-python> "$trl_scripts/trl_ddp_local_probe.py" ce --root "$PWD" \
  --runtime torch --device cuda --rows 16 --vocab 151936 --repetitions 32 \
  --output "$trl_state/ce-torch"
<jittor-python> "$trl_scripts/trl_ddp_local_probe.py" ce --root "$PWD" \
  --runtime jittor --device cuda --rows 16 --vocab 151936 --repetitions 32 \
  --output "$trl_state/ce-jittor"
<oracle-python> "$trl_scripts/trl_ddp_local_probe.py" compare \
  --reference "$trl_state/ce-torch" --candidate "$trl_state/ce-jittor" \
  --output "$trl_state/ce-comparison.json"
```

`local` additionally requires the explicit model, upstream, initial buffers,
adapters and rank trace. A noninitial `--step` must load retained preceding-step
parameters; an optional `--parameter-trace` supplies rank-zero payloads for a
rank-one input. Every loaded parameter must match the selected rank's recorded
SHA, shape and dtype. This unmeasured local-model mode is a diagnostic aid, not
evidence that collectives or a full distributed training trajectory pass.

### Isolate AMP precision-policy differences

[`trl_amp_precision_probe.py`](scripts/trl_amp_precision_probe.py) runs a small
CUDA RMSNorm arithmetic chain and FP32 elementwise multiplication under both
FP16 and BF16 autocast. It records each node's actual dtype/device, verifies
shared inputs and preserves the original dtype even when snapshots require a
float32 NumPy representation. It has no `--device` flag: this is a CUDA-only
microdiagnostic, not a substitute for full-model training.

```bash
JITTOR_TORCH_SHIM=0 <oracle-python> "$trl_scripts/trl_amp_precision_probe.py" run \
  --root "$PWD" --runtime torch --output "$trl_state/amp-torch"
JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
<jittor-python> "$trl_scripts/trl_amp_precision_probe.py" run \
  --root "$PWD" --runtime jittor --output "$trl_state/amp-jittor"
<oracle-python> "$trl_scripts/trl_amp_precision_probe.py" compare \
  --reference "$trl_state/amp-torch" --candidate "$trl_state/amp-jittor" \
  --output "$trl_state/amp-comparison.json"
```

Allocate the CUDA device/cache before these runs. Distinguish observed dtype
policy changes from ordinary FP32 reduction rounding; this microprobe alone
cannot quantify how much of a full training discrepancy each operator causes.

### Preserve full-trace evidence with lossless archives

[`trl_trace_archive.py`](scripts/trl_trace_archive.py) defaults to **dry-run**.
Only explicit `--archive` compresses completed, zero-fallback Torch-oracle
`final_gradient`/`final_parameter` NPY payloads using the external `zstd`
executable. It leaves manifests and all other observations unchanged:

```bash
<oracle-python> "$trl_scripts/trl_trace_archive.py" \
  "$trl_state/orpo-torch-cpu/trace"
<oracle-python> "$trl_scripts/trl_trace_archive.py" \
  "$trl_state/orpo-torch-cpu/trace" --archive
```

Every sibling `.npy.zst` must reconstruct the **exact original NPY bytes**:
raw SHA256/length and array SHA256/dtype/shape/finiteness are checked. The
archive, recovery journal and containing directories are fsynced before
removing the redundant uncompressed copy. Exclusive journal/completion files
are stored beside the trace directory and sealed read-only on completion;
they bind the unchanged original manifest and complete bulk inventory.
Existing receipts or archives are never overwritten. An interrupted archive
leaves recovery evidence and fails closed; do not erase it to force a retry.

The collector automatically reads completed archives through
`read_archived_array`/`ArchiveReader`, checking recovery metadata and payloads.
The full-checkpoint producer still reads its reference with `np.load`. Before
reusing an archived reference for another training run, restore ordinary NPY
files into a **new, separate** trace directory:

```bash
<oracle-python> "$trl_scripts/trl_trace_archive.py" \
  "$trl_state/orpo-torch-cpu/trace" \
  --restore-to "$trl_state/restored-orpo-torch-cpu/trace"
```

Recovery copies the unchanged manifest, initial extras and small observations,
then checks each decoded payload's original NPY SHA256/length and array
metadata. `restore-journal.jsonl` and `restore-complete.json` live beside the
new trace. Use the new trace as `--reference` only after completion. Source
archives are untouched, existing destinations are rejected, and failed partial
destinations are retained without a completion receipt. Put the destination on
persistent storage when restoring a temporary-memory reference.

Keep the journal, completion receipt, original manifest and compressed arrays
together. RAM-backed traces require persistent copies of **all** these files
before that temporary storage disappears. Compression preserves the data;
retaining hashes alone does not. This workflow does not authorize discarding
raw evidence. The collector's separate `--prune-oracle-bulk` operation destroys
recoverability and is not part of these commands or a substitute for archival.

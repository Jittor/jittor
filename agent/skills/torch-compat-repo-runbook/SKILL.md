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
   `docs/results/2026-09-19-torch-compat-runbook-verification.md`。

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

- 可复现结论写 `docs/results/YYYY-MM-DD-<topic>.md`：Status / 日期 / Baseline commit /
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
[统一下游 scorecard 协议](../../../docs/development/downstream-scorecards.md) 为唯一入口。

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
[PEFT 来源与训练结果](../../../docs/results/2026-10-15-peft-installed-source-training.md)。
普通 ecosystem pytest 的一步 parity 与这个完整 L3 gate 分开；nightly 未选择新 session
时不能声称已有定期执行。历史 raw 缺来源字段时重新运行，不能反向补证。

## Accelerate checkpoint 协议

完整单 rank CPU nightly 可用 `python tools/provision_ecosystem_nightly.py --library accelerate --state-root <new-external-state> --run`。
它完整配置三个独立环境并执行原 438 数组/8 阶段 worker 协议，要求原参数/输入梯度、三步保存前训练及两步新进程续训，
各 prefix 完整依赖解析与 `pip check`、实际官方 wheel/METADATA/pip RECORD 和公共 L3 准入均为必需。
`JITTOR_ACCELERATE_COMPLETE_CPU=1` 才启用 fixed CPU 模式，不改变普通 CPU/CUDA 调用。
实际执行状态和边界见[专用报告](../../../docs/results/2026-10-16-accelerate-maintained-cpu-nightly.md)，
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
  --baseline <full-measured-source-SHA> --report docs/results/<dated-report>.md \
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

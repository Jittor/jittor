# 下游覆盖矩阵与 scorecard 协议

- 状态：维护者验收协议；模板占位符不是验证结果。
- 复查日期：2026-10-14。
- 基线：`cded3102037edc3663b8f2054a6ebc06da6ebfd2`，加本文件所在提交。
- Owner：Torch 兼容层与下游生态维护者。
- Review when：上游注册表、生态 runner、设备回退、图重放、计时或进程内存口径变化。

覆盖广度与 L1–L5 深度正交。一个场景的最高级别必须有相应证据，多个 import 成功不能
拼成训练支持。本协议适用于所有下游，NPU 特有的驱动/算子排查继续使用
[NPU 模板](npu-validation-templates.md)；计时细节沿用
[基准规范](../performance/benchmarking.md)。

## 上游分母

入口是 [`registry_inventory.py`](../../agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py)，
只读 Git 跟踪的源文件，不导入 Torch 或下游库。必须是干净的 checkout，记录完整 SHA、
remote、源码版本与注册源文件 SHA256；每次 scorecard 重新枚举，不接受手写分母。

| 仓库 | 自动来源与广度轴 | 不能外推的范围 |
| --- | --- | --- |
| Transformers | `MODEL_MAPPING_NAMES` 模型族；`MODEL_*_MAPPING_NAMES` 任务映射 | 一个配置/任务不证明该族所有 checkpoint、MoE、多模态或长上下文 |
| PEFT | `PeftType` 方法；`TaskType` 任务 | 类型登记不证明实现；DoRA 等配置变体须另列具体规格，模型族另引用 Transformers 分母 |
| Diffusers | `pipelines._import_structure` 的全部可选分支：pipeline 模块族与 API；AutoPipeline 家族/任务另列 | UNet 单模块不证明整 pipeline；AutoPipeline 子集不冒充全库 |
| ms-swift | `register_model(ModelMeta)` 模型族；`SWIFT_MAPPING` tuner；普通/Megatron trainer 与 CLI；参数 `task_type`/`rlhf_type` 的 Literal | 普通与 Megatron 轴分开；一个 tiny LoRA 不证明全量/分布式/所有 trainer 方法 |
| verl | advantage-estimator、policy-loss、distillation-loss 注册；Hydra 入口 | 入口包括训练、评测与生成；登记不证明完整 PPO/GRPO 流水或全部模型 |

这里枚举的是**源码声明**，包含可选依赖的条件分支。没有安装某个依赖不应缩小分母。
上游增加新的动态注册方式或解析器遇到未知布局时失败，先更新解析器与负向测试，再给
覆盖数；不得把不能解析的项静默丢掉。

```bash
scripts=agent/skills/torch-compat-repo-runbook/scripts
python -S "$scripts/registry_inventory.py" --repo transformers \
  --checkout "$JITTOR_LAB_ROOT/downstream-coverage/upstream/transformers" \
  --output "$JITTOR_LAB_ROOT/_state/downstream-coverage/transformers-registry.json"
python -S "$scripts/audit_coverage.py" audit --json
python -S "$scripts/audit_coverage.py" scorecard \
  --upstream "transformers=$JITTOR_LAB_ROOT/downstream-coverage/upstream/transformers" \
  --upstream "peft=$JITTOR_LAB_ROOT/downstream-coverage/upstream/peft" \
  --evidence "$JITTOR_LAB_ROOT/_state/downstream-coverage/evidence.json" \
  --output "$JITTOR_LAB_ROOT/_state/downstream-coverage/scorecard.json"
```

未传 `--evidence` 可生成**零已验分子**的完整清单，表示本次没有可接受的新证据，不能
解释成“历史上从没跑通过”。各轴分别报告 `X / N @ upstream SHA`，不把模型数、任务数、
方法数相加。core/broad/tail 的抽样规则按
[适配流程](../../agent/skills/downstream-library-adaptation/SKILL.md) 执行；报告必须列每层
场景数及未验证格，core 矩阵至少 L3。

审计中的 `gate_cases` 记录共享模型 CASES；`dedicated_gates` 单独记录 DeepSpeed Engine
这类专用协议入口及 nox session，不将其伪装为普通模型 case。两者只证明门禁已接入，
`verified_level` 仍为空；L 级必须由当前 scorecard 的实际执行证据授予。

## 每行必填的平台块

| 字段 | 固定内容 |
| --- | --- |
| 身份 | Jittor 完整 commit；上游完整 commit/version/remote；两侧实际下游版本必须等于该源码版本 |
| 主机/设备 | OS、CPU、卡型号、卡数、可见 index 映射；CPU 明写零卡，并显式关闭 CUDA/ACL |
| 软件栈 | driver、CUDA 或 CANN、nvcc、gcc、nvidia wheel 版本；不适用的字段明写 `not-applicable` |
| 实际 flag | `use_cuda`、`use_acl`、graph replay、TF32、HF32；NPU/ROCm 等可能仍使用 Jittor 的 `use_cuda` 调度开关，须记录真实值 |
| oracle | torch/torch_npu 版本、二进制扩展来源、`is_shim=false`；“另一个 Python”本身不够 |
| 输入 | seed、`HF_HUB_OFFLINE=1`、tiny/real；模型、batch、sequence、dtype、precision、train/infer/decode |
| 来源 | 已入树 gate node、dated results 报告、完整复现命令；双方输出、比较结果与平台快照的外部路径及 SHA256 |

同卡对拍用两个独立进程，不使用 Jittor CPU/device 自比补 oracle。CPU①必须显式禁 CUDA；
设备③必须证明驻留与 `fallback_policy=error,count=0`。缺失、skip、xfail、环境阻塞与失败
各自记录，全部 skip 不能作为通过。平台快照用于复核身份，**不能替代计算所在设备的断言**。

## L4 的速度块

L4 累计要求 L2 数值与 L3 多步 loss/参数轨迹；只跑 forward/backward 的探针不能升为 L3。
真实尺寸且同卡、同模型规格、同精度；两侧预热、计时次数与统计量一致。至少预热一轮并
计时十轮，生产结论应按基准规范增加样本、交错重跑并记录机器负载。

每行必须有 `jittor_seconds`、`oracle_seconds`、重新计算的 ratio、预先声明的 `ratio_limit`、
**紧挨速度的 agreement 及 tolerance**；PURE eager/default 图重放模式不可混比，平台
flag 与测量 regime 必须一致。没有并发负载，记录 warmup/iterations/statistic。
CUDA 显存为两侧整进程 NVML 峰值，NPU 用进程级 npu-smi，CPU 用进程 RSS；缺少对应
采样就缺 L4 证据，不用 allocator 的 allocated/reserved 代填。allocator live/pool 可作为
附列，来源与采样间隔在原始快照说明。L5 还需真实加速卡零回退。

PR 只报告时间；速度上限断言用于固定硬件的 nightly/验收。诊断测量可以报告原始时间，
但不能从缺平台/缺数值的时间表得出“性能达标”。

## 机器可检的证据结构

`evidence.json` 为 `{"schema_version": 2, "records": [...]}`。以下是 L2 的**结构模板**，
尖括号必须由实际执行填写。输入扰动检查要在固定权重下改变输入并重新对拍，不能把
“重复跑同一个输入”写成已验证。整数 token 不请求 input gradient 时明写 requested=0；
所有被请求的参数与浮点输入梯度都必须检查。

```json
{
  "repo": "transformers", "case": "<真实case>", "status": "passed", "level": 2, "tier": "broad",
  "jittor_commit": "<40位SHA>",
  "upstream": {"commit": "<40位SHA>", "version": "<源码版本>", "remote": "<实际上游URL>"},
  "downstream_versions": {"candidate": "<源码版本>", "oracle": "<源码版本>"},
  "registry_refs": [{"repo": "transformers", "axis": "model_family", "member": "<自动registry中的键>"}],
  "device": "cpu", "observed_device": "cpu",
  "platform": {
    "os": "<OS>", "cpu": "<型号>", "accelerator_model": "not-applicable", "card_count": 0, "card_indices": [],
    "driver": "not-applicable", "cuda": "not-applicable", "cann": "not-applicable", "nvcc": "not-applicable",
    "gcc": "<版本>", "nvidia_wheels": {},
    "precision": {"matmul": false, "cudnn": false, "cudnn_benchmark": false, "matmul_precision": "highest"},
    "jittor_flags": {"use_cuda": 0, "use_acl": 0, "graph_replay": "eager", "tf32": false, "hf32": false},
    "oracle": {"is_shim": false, "torch_version": "<版本>", "torch_npu_version": "not-applicable", "binary_extension": "<真实_C.so>"}
  },
  "inputs": {"seed": 0, "hf_hub_offline": true, "shape_kind": "tiny"},
  "numerics": {
    "passed": true, "finite": true, "checked_tensors": 1, "missing_tensors": [], "input_perturbation_verified": true,
    "max_abs": 0.0, "max_scaled_relative": 0.0, "atol": 0.0001, "rtol": 0.001,
    "parameter_gradients": {"requested": 1, "checked": 1}, "input_gradients": {"requested": 0, "checked": 0}
  },
  "evidence": {
    "command": "<完整命令>", "report": "docs/results/<dated-report>.md", "gate_nodeid": "<文件>::<真实测试节点>",
    "artifacts": [
      {"role": "reference", "path": "<checkout外原始输出>", "sha256": "<SHA256>"},
      {"role": "candidate", "path": "<checkout外原始输出>", "sha256": "<SHA256>"},
      {"role": "comparison", "path": "<checkout外数值比较>", "sha256": "<SHA256>"},
      {"role": "platform", "path": "<checkout外平台快照>", "sha256": "<SHA256>"},
      {"role": "base-comparison", "path": "<verify_repo的原始JSON报告>", "sha256": "<SHA256>"},
      {"role": "perturb-comparison", "path": "<固定权重输入扰动比较JSON>", "sha256": "<SHA256>"}
    ]
  }
}
```

L3 追加 `training: {steps, loss_trajectory_passed, parameter_trajectory_passed, trainable, frozen, trainable_parameters, frozen_parameters}`，后两项是实际参数名列表；artifacts 另需 `training-reference-metadata` 与 `training-candidate-metadata`，指向两侧 training probe 的原始 JSON。L4 追加
`speed`（上一节字段，加 `model/batch/sequence/dtype/precision/phase`、`graph_regime`、
`no_concurrent_work: true`、`memory: {source, jittor_peak_bytes, oracle_peak_bytes}`）；
设备证据追加 `fallback: {policy: "error", count: 0}`。机器校验负责结构、SHA、成员、
门禁节点存在性、数值/级别前置和采样口径，**不替代维护者检查原始结果是否真的来自该命令**。

schema v2 逐阶段读取带哈希的真实测量 JSON：`base-comparison` 的 `cases[].oracle_result/jittor_result`、`perturb-comparison` 的 `reference_result/candidate_result`，以及 L3 两侧训练元数据。扰动 JSON 顶层必须含 `case`、`status=passed`、`input_perturbation_verified=true`；两侧结果沿用共享 runner 的字段，不能换成手工 precision 标签。所有阶段的 actual device、TF32 的 matmul/cuDNN 开关、`matmul_precision`、runtime conditions 与平台 `precision` 必须相同；平台 JSON 必须与记录平台块完全一致。L3 再核对实际 Git SHA、步骤数及 trainable/frozen 参数名与计数。每个 artifact role 唯一。只重写报告标签，或者给错误平台标签重新算 SHA，都无法通过这些校验。

`coverage[].verified_scenarios` 保留 case、所属 repo、已验 L 级、完整的跨库 registry_refs、tiny/real 和实际 trainable/frozen 参数范围。家族分子必须连同场景阅读：Llama+LoRA 的 L3 不能写成 dense Llama 全参 L3；没有训练证明的 L2 场景，参数更新范围明确为 null。schema v1 不再用于新授级，须从原始测量重新生成 v2 绑定。

默认仍将证据绑定当前 HEAD。审计历史报告可显式传 `--baseline <完整40位Jittor SHA>`，该 SHA 必须是本地 Git 中的真实 commit；输出记录 `baseline_policy=explicit-historical`、测量 baseline 与当前 `auditor_commit`。这只重新审计旧提交的证据，不能证明当前代码也通过。不得将旧测量中的 SHA 改成当前 HEAD；当前树的新验收须重新执行计算。

现有 [`verify_repo.py`](../../agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py)
复用生态 runner，可快速生成双方原始元数据和数值诊断。它不执行输入扰动、多步优化器
轨迹或完整平台/进程峰值采集，因此输出明确标 `scorecard_eligible=false`；不要把单 case
的 `passed` 自动转换为完整 scorecard 行。

多步补证可运行 [`training_probe.py`](../../agent/skills/torch-compat-repo-runbook/scripts/training_probe.py)：
同一个已注册 builder，独立 oracle 保存完整初始权重，candidate 加载后运行相同 SGD。
工具保留 upstream tuner 的 trainable/frozen 划分，输出每步 loss、参数/输入梯度与全部参数
轨迹，冻结参数发生变化、缺请求梯度、非有限值、device 不一致都会直接失败。它对原始
梯度与参数做 host 快照以固定 lazy update 的语义；没有性能结论。具体双解释器命令见
[runbook](../../agent/skills/torch-compat-repo-runbook/SKILL.md#多步训练证据)。

## Results 报告模板

```text
# <repo> <model/method> coverage scorecard
Status: measured / blocked / partial
Date: <date and timezone>
Baseline commit: <full SHA>
Owner: <maintainer role>
Review when: <upstream/compat/runtime/driver/measurement changes>

Upstream: <remote, full SHA, source+installed version, source hashes>
Platform: <固定平台块；仅引用外部文件还不够，关键值必须在报告可读>
Sampling: core <x> scenarios / broad <y> / tail <z>; selection rationale

| Model family / method / exact spec | CPU① level | Device③ level | Speed ratio / agreement | Status / evidence |
| ... | ... | ... | ... | ... |

Breadth: <X / N @ upstream SHA, 分轴分设备；未验证/未抽查数量>
Commands: <完整重现命令，包括明确CPU禁CUDA和独立缓存>
Artifacts: <lab相对路径、SHA256、未版本化>
Limits: <未覆盖的API/精度/设备/级别；缺硬件和环境错误分别写>
```

没有可接受证据时保留未验证的格子。本协议的负向测试见
[`test_downstream_coverage.py`](../../tests/tools/test_downstream_coverage.py)。

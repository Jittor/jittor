---
name: deepspeed-torch-compat
description: DeepSpeed 0.17.6 显式 adapter 的 Ascend NPU 单机双卡 ZeRO 适配、独立 PyTorch 对拍与 checkpoint 恢复复现入口。
---

# DeepSpeed torch compat 开发入口

## 用途

2026-09-26 当前状态：97b9aab4455b145ca1fb431af3504d705ac1edf3加保留修改与公共FP32 ACL RMS guard；Qwen3-0.6B、单机两NPU、FP32 AdamW eps=1e-8、每rank batch1/sequence12/GAS1、无offload/裁剪：Stage 1/2/3各自限定累计L0–L4。各Stage三步全部310参数梯度/更新、logits/loss与输入梯度3744跨runtime/3720rank检查，rank差0、fallback0；原5e-5+5e-5×同一步同类全场oracle量级容差未放宽。严格汇总`R/rms-guard-stage123-current-l4-summary.json`；历史eps=1e-6各Stage L4保留，不迁移新配置。公共RMS真实NPU契约2测试/16组合、CPU core321/42skip/1xfail及structure1394/4skip保持已验。详见[开发报告](../../../docs/results/2026-09-22-deepspeed-torch-compat.md)。

Stage 2 曾在 rank1 等待子组 rootinfo 文件时超过默认120秒；诊断重试配置300秒后通过，但文件生成实测89秒，不能据此认定增大超时解决了根因。恢复探针可用 DS_TRACE_HANG=1 在模型加载前启动栈采样并记录阶段、rank、会合路径与实际超时；历史75aa默认120秒双卡恢复通过，Stage 2 WORLD→PG1文件间隔约29秒；只完成一轮，稳定性问题仍保留，原始日志已归档。整库、其他配置、性能不声明 L4/L5。老师暂缓多机，当前只做单机多卡。

先在同一配置验证 L0–L3，再单独验证 L4 生成、恢复与随机采样；不同模型、配置或提交的结果不能拼成更高累计等级。
固定小模型Stage 1/2及Stage 3覆盖均保留原基线；历史75aa真实Qwen的eps=1e-6三Stage各自限定L4见下文。当前97b9真实Qwen按Stage 1/2/3独立复验FP32、eps=1e-8、单机两NPU、每卡batch 1、sequence 12、GAS 1、无offload/裁剪；正式比较与后续层级分别验收。其他模型/配置、多机、混合精度、其他优化器及性能未验证。
任务采用负责人提供的新版严格累计 L0–L5：仅 import 成功不等于完整 L0。

先遵循 [下游接入规则](../downstream-library-adaptation/SKILL.md)。L0 已进入专用必需门禁及生态入口；
固定小模型 Stage 1 结论只适用于两卡 HCCL、FP32 AdamW、三步玩具网络；真实 Qwen3 的限定 L4 证据按下文单独固定 checkpoint、设备和配置。速度、显存未验证。

## 最新可复用工具（2026-09-27维护）

按[eps1e8复现说明](references/eps1e8-reproduction.md)选择正式入口。Stage2/3训练、权重往返、优化器恢复四个新探针逐字节保留已验证仓外副本的SHA；旧eps1e6工具保持原用途。Stage1训练仍用 `scripts/qwen3_training_probe.py`，它硬编码Stage1，不能仅设置环境变量冒充Stage2。

新版 `scripts/qwen3_l3_source_compare.py` 在独立PyTorch解释器中读取指定Stage的已保存产物，核对来源报告SHA、配置/设备/参数清单、四份manifest以及1240个保存权重与对应训练末步的一致性，并拒绝非有限权重。loss/logits往返项核对原报告及diagnostic，不重新执行前向。本轮工具验证为CPU只读重放已有NPU证据，不是新一轮NPU训练。

新版构造入口是 `scripts/qwen3_construct_manifest_evidence.py` 与 `scripts/qwen3_construct_compare.py`；增强清单记录依赖、独立oracle身份、全部311张量同步后的placement和fallback。旧清单缺少依赖版本时，比较器明确返回partial，不替历史记录补字段。requested_zero_stage仅表示构造用例标签，不证明分片Engine已初始化。统一双rank配对启动器与完整L0–L4严格汇总仍需整理；当前不是一键完整复现入口。历史站点启动器和原始结果继续保留为未版本化产物。L0–L5声明采用团队最新累计定义，不能与旧skill中的不同分级混用。

## 当前97b9 eps=1e-8各Stage复验

97b9aab4455b145ca1fb431af3504d705ac1edf3加保留修改与公共FP32 ACL RMS guard；Qwen3-0.6B、单机两NPU、FP32 AdamW eps=1e-8、每rank batch1/sequence12/GAS1、无offload/裁剪：Stage 1/2/3各自限定累计L0–L4。各Stage三步全部310参数梯度/更新、logits/loss与输入梯度3744跨runtime/3720rank检查，rank差0、fallback0；原5e-5+5e-5×同一步同类全场oracle量级容差未放宽。

| 层 | Stage 1：R内路径 | Stage 2：R内路径 | Stage 3：R内路径 |
| --- | --- | --- | --- |
| L0 | `qwen-stage1-l0-97b9-rms-guard-v2/comparison.json` | `qwen-stage2-l0-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-l0-97b9-rms-guard-v1/comparison.json` |
| L1/L2 | `qwen-stage1-eps1e8-97b9-rms-guard-v1/comparison.json` | `qwen-stage2-training-97b9-rms-guard-v2/comparison.json` | `qwen-stage3-training-97b9-rms-guard-v1/comparison.json` |
| L3 | `qwen-stage1-l3-97b9-rms-guard-v2/comparison.json` | `qwen-stage2-l3-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-l3-97b9-rms-guard-v1/comparison.json` |
| L4生成 | `qwen-stage1-gen-97b9-rms-guard-v1/comparison.json` | `qwen-stage2-gen-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-gen-97b9-rms-guard-v1/comparison.json` |
| L4采样 | `qwen-stage1-sample-97b9-rms-guard-v1/comparison.json` | `qwen-stage2-sample-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-sample-97b9-rms-guard-v1/comparison.json` |
| L4恢复 | `qwen-stage1-resume-97b9-rms-guard-eps1e8-v1/comparison.json` | `qwen-stage2-resume-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-resume-97b9-rms-guard-v1/comparison.json` |

| Stage | L3自身loss roundtrip | gen prefill | gen decode | sort values | top20 values | top-p probability | 全排序索引不同/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.536743164e-07 | 2.002716064e-05 | 2.217292786e-05 | 1.788139343e-05 | 1.239776611e-05 | 9.401785285e-07 | 12610 |
| 2 | 9.536743164e-07 | 2.002716064e-05 | 2.217292786e-05 | 1.788139343e-05 | 1.239776611e-05 | 9.401785285e-07 | 12610 |
| 3 | 9.536743164e-07 | 1.966953278e-05 | 1.811981201e-05 | 1.966953278e-05 | 9.536743164e-06 | 2.667238848e-06 | 14178 |

| Stage | 恢复loss | logits | 输入梯度 | 更新权重 | oracle重放/非逐位字段数 | shim重放/非逐位字段数 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1.907348633e-06 | 2.241134644e-05 | 8.964538574e-05 | 1.888908446e-05 | 9.536743164e-07 / 1 | 0.000000000e+00 / 0 |
| 2 | 9.536743164e-07 | 2.241134644e-05 | 8.964538574e-05 | 1.888908446e-05 | 0.000000000e+00 / 0 | 0.000000000e+00 / 0 |
| 3 | 2.861022949e-06 | 3.719329834e-05 | 7.677078247e-05 | 1.825205982e-05 | 9.536743164e-07 / 1 | 0.000000000e+00 / 0 |

已完成L3的各Stage四份source_training_report_sha256绑定自己的正式训练，1240个实际safetensors张量与末步精确一致，权重roundtrip0、logits自身roundtrip0、五token一致；任务loss差如表，未把数值容差通过称逐位一致。生成14字段、cache本侧差0、cached/uncached greedy与beam逐token一致；默认top-k20/top-p0.95/temperature0.6采样四status/errors空，四新token/top20索引/16候选一致。全排序索引差异如表，不宣称全部索引相同。 每Stage独立恢复1256字段passed，四model_restored_exact true、rank同步0、shim fallback0；第4步同runtime重放差与非逐位字段如表，原阈值通过。314字段/phase/rank为310 updated权重加input_ids/loss/logits/input_grad，不含全部参数梯度/optimizer-state hash。checkpoint_sha256实际config摘要；内部逐参数模型hash断言以model_restored_exact记录，公开load_optimizer_states=True及第4步数值重放不等于优化器hash一致。

Stage 2训练v1实际zero_stage=1，S/stage2-v1-invalid-evidence.json标为无效Stage2证据、保留原件。v2真实Stage2完整比较passed；外层shell在等待比较期间被补强编辑，随后语法退出1，S/stage2-training-result-check.json独立核对comparison SHA/status passed。此为外层执行记录异常，不是数值失败，原日志保留，未重跑数学或调宽阈值。 Stage 2/3仓外S/qwen3_training_stage{2,3}_eps1e8.py分别来自版本化eps1e6训练探针和Stage 3专用探针，只改optimizer_config eps=1e-6为1e-8；来源SHA及唯一替换见S/stage{2,3}-eps1e8-provenance.json。Stage 3保留初始化前完整shape，通过公开full-grad/full-FP32-param访问器取全量张量，不能把分片占位shape当完整参数。L0为普通模型310参数/1 buffer及requested stage，不单凭manifest证明分片Engine。

公共RMS guard仅FP32 ACL grad-enabled保留第三方forward；eval/frozen输入或weight和额外可训练参数仍按实际公式保留autograd图，避免融合替代其forward。no_grad标准模块融合保留，BF16三旧算子仅仓外默认device修正后通过，不声明DeepSpeed混合精度。Stage1 canonical/L3 eps前置/resume eps-only来源与首轮依赖失败原件继续保留于C；143源码冻结非提交清单。证据各自绑定本Stage同次训练与保存权重，不借用旧提交/其他Stage补层。生成与采样来自普通重载训练后模型，不是DeepSpeed inference Engine或分片Engine内generate。其他eps/checkpoint/world配置、BF16全训练、offload、多机及L5未验证；KI001裸FX和KI003偶发PG1仍开放。

#### 当前97b9 Stage 2/3 eps=1e-8复现入口

S为`$JITTOR_LAB_ROOT/state/deepspeed-20260926/stage23-rms-guard`；R为对应日期state，C为R/candidate-97b9。全部脚本/report/数组/checkpoint为仓外未版本化产物。

入口S/run-stage{2,3}-l0-pair.sh、run-stage{2,3}-training-pair.sh、run-stage{2,3}-resume-pair.sh、run-stage{2,3}-dependent.sh；dependent串行执行本Stage l3/gen/sample pair。训练Stage2必须使用v2，v1实际Stage1标invalid。Stage3使用专用full-grad/full-param探针，不能只改普通canonical环境变量。只改eps的来源见S/stage{2,3}-eps1e8-provenance.json，launchers冻结见S/launcher-freeze.json。L3/resume副本沿用C的eps1e8 provenance，oracle-site固定依赖与训练一致。

比较入口S/compare-stage{2,3}-l0.py、compare-stage{2,3}-l3.py、compare-stage{2,3}-generation.py；正式训练与恢复复用版本化完整比较器。S/collect-stage{2,3}.py只读严格汇总为R/rms-guard-stage{2,3}-current-l4-summary.json，S/collect-stage123.py生成R/rms-guard-stage123-current-l4-summary.json。先核对真实stage/配置/独立oracle身份/设备/源码SHA，再检查六JSON status/errors及训练gaps，不用脚本存在或单边passed充层级。

复跑前替换实际有效分配作业、正确工作树、全新输出目录与独立rank/runtime缓存；WORLD_SIZE=2，两runtime串行、每runtime两rank同时，首次JIT串行。不要复用旧job/并发缓存；公共文档不写个人home/host/固定卡号。生成为普通重载模型，1256恢复字段和hash范围见§2.3。

## 历史75aa三Stage复现与证据边界

仓外根R为 $JITTOR_LAB_ROOT/state/deepspeed-20260926。各s=1/2/3独立证据为qwen-stage{s}-l0-75aa-v1/、qwen-stage{s}-eps1e6-75aa-formal-v1/、qwen-stage{s}-l3-75aa-v1/、qwen-stage{s}-gen-75aa-v1/、qwen-stage{s}-sample-75aa-v1/、qwen-stage{s}-resume-default120-v1/内comparison.json；stage123-current-l4-summary.json记录层级、实测误差及这些JSON的SHA256。站点入口、日志、权重未版本化，不进入主仓库；模板完整smoke/ecosystem/四轴与正式L5速度显存未跑，不能写通过。

Stage1/3本轮站点入口为run-qwen-stage{1,3}-training-pair-75aa-v1.sh、相应L0/L3/L4包装与resume-default120.sh；用有效Slurm分配替换旧job773，并使用全新输出目录。NPU两runtime串行、每runtime两rank同时；WORLD_SIZE=2、各rank独立JITTOR_HOME及匹配可见卡，默认会合预算120秒。训练探针分别qwen3_training_probe_eps1e6.py和qwen3_training_probe_stage3.py，比较器qwen3_training_compare.py --zero-stage 1或3 --steps 3；不得把Stage3分片或parameter.grad=None当成完整梯度。

L0普通模型清单在Engine初始化前采集；requested_zero_stage字段不证明分片Engine构造。Engine能力来自真实训练与恢复记录。L3保存模型绑定各自正式训练report SHA256，再逐张量核对末步权重；CPU读取safetensors需同版oracle-site，禁止缺包时skip充通过。L4的DS_GEN_MODEL必须指向本Stage、同runtime/rank的L3 saved-model。

恢复比较各1,256字段覆盖310更新参数及输入token/loss/logits/输入梯度的首次与重放两阶段；不包含全部参数梯度或优化器状态逐项哈希。公开load_optimizer_states=True后第四步数值重放通过、模型恢复哈希内部精确断言。恢复report checkpoint_sha256字段实际为源config.json摘要，不是原始model.safetensors摘要；两种摘要不可混称。

默认采样top_k20/top_p0.95/temperature0.6：三个Stage top20索引、16个top-p候选和四新token一致；完整词表排序索引每rank分别10279/11215/11680处跨runtime差异。排序数值在固定容差内，不声明完整排序索引精确一致。恢复同runtime重放也不是全部逐位相同，具体误差/非精确字段见报告。

所有比较器结果必须另断言JSON status=passed、errors为空；训练还需evidence_gaps为空，采样另三status均passed。进程exit0不等于验证通过。原eps=1e-8输入梯度超差保留为修前历史，当前RMS guard的指定配置复验见前文；Stage2 PG1偶发超时仍未定位，默认120秒单轮成功不能关闭稳定性缺陷。

## 两侧环境

- Python 3.11、DeepSpeed 0.17.6、NumPy 1.26.4，两侧相同。
- oracle：独立 PyTorch 2.7.1+cpu；NPU 加 torch_npu 2.7.1.post4、匹配 CANN。
- shim：当前 Jittor 源码，`JITTOR_TORCH_SHIM=1`。
- 每个 runtime/device 单独 JITTOR_HOME、TMPDIR、XDG_CACHE_HOME。
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`，`JT_BACKEND_FALLBACK=error`。
- oracle 起手断言 `assert not hasattr(torch, '_torch_compat_install_context')`。
- 在调度器分配内执行，CPU 先行；真实 NPU 对真 torch_npu；禁用有已知问题的算子并行编译。

以下命令假设 `ORACLE` 和 `PY` 为已安装依赖的绝对解释器路径，`RUN_ROOT` 为仓外产物目录。
按设备设置独立进程环境；NPU 使用 `JT_BACKEND=acl`，CPU 使用 `JT_BACKEND=cpu`。

## 正式 L0 入口

范围：单 Ascend NPU、FP32、真实 rank0/size1 HCCL、eager ZeRO Stage 0、显式默认 AdamW。
CPU 只验 import/config/model，CPU Engine 明确失败。不激活 adapter 的原版裸导入仍未通过。
独立安装 `adapters/` 发行包；在任何 DeepSpeed 导入之前执行：

```python
from jittor_adapters.deepspeed import activate
activate(device="npu")
import deepspeed
```

采用共享版本检查与固定源码/构建清单；未校验 wheel 直接拒绝。
不修改已安装 DeepSpeed 文件，不加载 PyTorch ABI 二进制。两边依赖按
`requirements/deepspeed-l0.txt` 固定，版本记录到 JSON 并逐项比较。

```bash
# 先 CPU；换设备时换进程与仓外状态目录
JT_BACKEND=cpu JITTOR_TEST_DEVICES=cpu JITTOR_REQUIRE_DEEPSPEED=1 \
  REAL_TORCH_PYTHON="$ORACLE" JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_TORCH_SHIM=1 \
  JITTOR_DEEPSPEED_L0_OUT="$RUN_ROOT/results" "$PY" -m pytest -q compat/tests/torch/test_deepspeed_l0.py
# 真 NPU 环境中把 JT_BACKEND 改为 acl、JITTOR_TEST_DEVICES 改为 npu。
# 测试为 shim 子进程初始化真实单 rank HCCL，再检查公开原生 WORLD 状态。
JITTOR_DEEPSPEED_WHEEL="$WHEEL" REAL_TORCH_PYTHON="$ORACLE" python -m nox -s deepspeed_l0
```

`l0_probe.py` 比较 2 配置、4 参数和 3 非空 buffer（含非持久 buffer）；NPU 额外构造 Engine。
非法配置错误类型对拍；范围外 optimizer/配置拒绝。零 fallback、真二进制 oracle、依赖一致都是硬断言。
`.github/workflows/deepspeed-l0.yml` 固定原包 SHA，关闭扩展构建，在独立 oracle 构建同一 CPU wheel 供两侧使用。
缺库不能算通过。CI 服务中的 scheduled run 尚未执行，硬件验收看 dated report。


## 两卡 ZeRO Stage 1 L2 入口

范围：DeepSpeed 0.17.6、单机两张 Ascend NPU、HCCL WORLD size 2、FP32、公开 AdamW、
`gradient_accumulation_steps=1`、无 offload、无梯度裁剪。默认连续梯度与显式
`contiguous_gradients=False` 均有回归覆盖。adapter 保留固定源码 SHA 和唯一锚点检查；
未验证的 ZeRO 选项由 `scope.py` 明确拒绝。

`stage1_probe.py` 必须在两个独立 rank 中同时运行。站点 launcher 需为每个进程设置：

- `DS_RUNTIME=oracle|shim`、`RANK=0|1`、`WORLD_SIZE=2`、`LOCAL_RANK=0`；
- `MASTER_ADDR`、两侧互不冲突的 `MASTER_PORT`、共同的 `DS_MULTI_OUT`；
- 每个 rank 只暴露一张物理 NPU；shim 额外设置匹配的 `JT_HCCL_WORLD_SIZE=2`、
  `JT_HCCL_RANK`、`JT_HCCL_LOCAL_RANK=0`、共享 `JT_HCCL_ROOTINFO_FILE`；
- 每个 rank 独立 `JITTOR_HOME`、`TMPDIR`、`XDG_CACHE_HOME`；
- shim 设置 `JITTOR_TORCH_SHIM=1 JT_BACKEND=acl JT_BACKEND_FALLBACK=error`，
  oracle 设置 `DS_ACCELERATOR=npu` 并确保没有 shim 标记。

每个 rank 的执行入口相同：

```bash
# 默认连续梯度路径：oracle 与 shim 各启动两个 rank
DS_RUNTIME=oracle DS_MULTI_OUT="$RUN_ROOT/results" \
  "$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_probe.py
DS_RUNTIME=shim DS_MULTI_OUT="$RUN_ROOT/results" \
  "$PY" agent/skills/deepspeed-torch-compat/scripts/stage1_probe.py

# 非连续梯度路径：同样分别启动 oracle 与 shim 的两个 rank
DS_CONTIGUOUS_GRADIENTS=false DS_RUNTIME=oracle DS_MULTI_OUT="$RUN_ROOT/results" \
  "$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_probe.py
DS_CONTIGUOUS_GRADIENTS=false DS_RUNTIME=shim DS_MULTI_OUT="$RUN_ROOT/results" \
  "$PY" agent/skills/deepspeed-torch-compat/scripts/stage1_probe.py

# 两侧四个进程结束后对拍
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_compare.py \
  --root "$RUN_ROOT/results/stage1-formal" \
  --out "$RUN_ROOT/results/stage1-formal/comparison.json"
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_compare.py \
  --root "$RUN_ROOT/results/stage1-formal-noncontig" \
  --out "$RUN_ROOT/results/stage1-formal-noncontig/comparison.json"
```

探针固定 4 个参数的两层 MLP，各 rank 使用不同输入，连续训练 3 步；记录输出、loss、
输入梯度、优化器分片梯度和每步参数。对拍硬断言真 PyTorch oracle、两卡参数完全同步、
shim fallback 为 0，并按 `2e-5 + 2e-4 * reference_scale` 验收。
2026-09-23 实测 worst abs：loss `0`、输出 `1.49e-8`、输入梯度 `5.59e-9`、
分片梯度 `1.49e-8`、更新参数 `7.45e-9`；两侧参数跨 rank 最大差均为 `0`。

## 两卡 ZeRO Stage 2 L2 入口

范围与 Stage 1 相同：DeepSpeed 0.17.6、单机两张 Ascend NPU、HCCL WORLD size 2、FP32 AdamW、`gradient_accumulation_steps=1`、无 offload、无梯度裁剪。默认连续梯度和 `contiguous_gradients=False` 均已验证。

沿用 Stage 1 的两 rank launcher 环境，将入口换为 `stage2_probe.py`：

```bash
# 默认连续梯度：oracle 与 shim 各启动两个 rank
DS_RUNTIME=oracle DS_MULTI_OUT="$RUN_ROOT/results" \
  "$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage2_probe.py
DS_RUNTIME=shim DS_MULTI_OUT="$RUN_ROOT/results" \
  "$PY" agent/skills/deepspeed-torch-compat/scripts/stage2_probe.py

# 非连续梯度：同样各启动两个 rank
DS_CONTIGUOUS_GRADIENTS=false DS_RUNTIME=oracle DS_MULTI_OUT="$RUN_ROOT/results" \
  "$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage2_probe.py
DS_CONTIGUOUS_GRADIENTS=false DS_RUNTIME=shim DS_MULTI_OUT="$RUN_ROOT/results" \
  "$PY" agent/skills/deepspeed-torch-compat/scripts/stage2_probe.py

"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_compare.py \
  --root "$RUN_ROOT/results/stage2-formal" \
  --out "$RUN_ROOT/results/stage2-formal/comparison.json"
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_compare.py \
  --root "$RUN_ROOT/results/stage2-formal-noncontig" \
  --out "$RUN_ROOT/results/stage2-formal-noncontig/comparison.json"
```

比较器从结果 JSON 读取 `zero_stage`，对 Stage 1/2/3 使用相同的 shape、dtype、有限值、容差、跨 rank 同步和 fallback 断言。2026-09-23 两条 Stage 2 路径实测 worst abs：loss `0`、输出 `1.49e-8`、输入梯度 `5.59e-9`、分片梯度 `1.49e-8`、更新参数 `7.45e-9`；两侧跨 rank 参数差均为 `0`。

## 两卡 ZeRO Stage 3 L2 入口

范围：DeepSpeed 0.17.6、单机两张 Ascend NPU、HCCL WORLD size 2、FP32 AdamW、默认连续梯度、`gradient_accumulation_steps=1`、无 offload、无梯度裁剪。Stage 3 会在前向和反向之间释放、重新聚合并重绑完整参数，因此探针同时检查输入梯度、优化器持有的 FP32 分片梯度及每步更新参数。显式 `contiguous_gradients` 选项未验证，adapter 与探针均拒绝。

沿用 Stage 1 的两 rank launcher 环境，将入口换为 `stage3_probe.py`：

```bash
# oracle 与 shim 各同时启动两个 rank；Stage 3 只允许默认连续梯度
DS_RUNTIME=oracle DS_MULTI_OUT="$RUN_ROOT/results" \
  "$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage3_probe.py
DS_RUNTIME=shim DS_MULTI_OUT="$RUN_ROOT/results" \
  "$PY" agent/skills/deepspeed-torch-compat/scripts/stage3_probe.py

"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_compare.py \
  --root "$RUN_ROOT/results/stage3-formal" \
  --out "$RUN_ROOT/results/stage3-formal/comparison.json"
```

2026-09-24 正式 adapter、无诊断旁路实测 worst abs：loss `0`、输出 `1.49e-8`、输入梯度 `5.59e-9`、FP32 分片梯度 `1.49e-8`、更新参数 `7.45e-9`；两侧跨 rank 参数差均为 `0`，shim fallback 为 `0`。HCCL 当前通过 all-reduce 加本 rank 切片实现公开 reduce-scatter 语义，结果留在 device，但通信量高于原生 reduce-scatter；该结果不构成性能声明。

## Stage 1 checkpoint round-trip 前置验证

这一项验证固定两层 MLP 在两卡 ZeRO Stage 1 下训练 3 步、保存模型与 AdamW 状态、故意再前进一步、恢复 checkpoint，并从恢复点重复第 4 步。它是当前 L3 的分布式恢复补充证据；单凭这个玩具 checkpoint 不能证明真实权重与正式任务，L3 结论来自下一节的 Qwen3-0.6B 验证。

沿用上一节的两 rank 启动环境，并额外为两侧设置不同的可信仓外 checkpoint 根目录：

```bash
# oracle 与 shim 各启动两个 rank；同一 runtime 的两个 rank 共享 checkpoint 根目录
DS_CHECKPOINT_ROOT="$RUN_ROOT/checkpoints" DS_RUNTIME=oracle DS_MULTI_OUT="$RUN_ROOT/results" \
  "$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_checkpoint_probe.py
DS_CHECKPOINT_ROOT="$RUN_ROOT/checkpoints" DS_RUNTIME=shim DS_MULTI_OUT="$RUN_ROOT/results" \
  "$PY" agent/skills/deepspeed-torch-compat/scripts/stage1_checkpoint_probe.py

# 两侧四个进程结束后严格对拍保存前 3 步、恢复权重和恢复后的第 4 步
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_compare.py \
  --root "$RUN_ROOT/results/stage1-checkpoint" \
  --out "$RUN_ROOT/results/stage1-checkpoint/comparison.json"
```

探针要求 `torch.load` 恢复模型和优化器状态，恢复权重先做逐元素精确断言，再比较恢复后的输出、loss、输入梯度和更新参数。2026-09-23 实测：恢复权重与真 PyTorch最坏绝对误差 `7.45e-9`；恢复后第 4 步输出 `1.49e-8`、loss `0`、输入梯度 `3.73e-9`、更新参数 `7.45e-9`；两侧跨 rank 参数差均为 `0`，shim fallback 为 `0`。

## 历史 Qwen3-0.6B Stage 0 权重与任务子项

范围：DeepSpeed 0.17.6 显式 adapter、Transformers 4.56.2、Qwen3-0.6B、FP32、单机两张 Ascend NPU、HCCL WORLD size 2、ZeRO Stage 0。它验证真实 checkpoint、因果语言模型 loss、`save_pretrained`/`from_pretrained` 和保存文件逐张量对拍；不验证 Qwen 训练、BF16 或 ZeRO Stage 1–3。

沿用 Stage 1 的两 rank launcher 契约，并设置：

- `DS_MODEL_PATH`：固定 Qwen3-0.6B checkpoint 目录；
- `DS_L3_OUT`：oracle/shim 共用的仓外结果根目录；
- `DS_RUNTIME=oracle|shim`；
- 两侧各自的 `MASTER_PORT`，shim 独立 HCCL root-info 文件；
- `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

```bash
# oracle 与 shim 各同时启动两个 rank
DS_MODEL_PATH="$CHECKPOINT/Qwen3-0.6B" \
  DS_L3_OUT="$RUN_ROOT/qwen3-l3" DS_RUNTIME=oracle \
  "$ORACLE" agent/skills/deepspeed-torch-compat/scripts/qwen3_l3_probe.py
DS_MODEL_PATH="$CHECKPOINT/Qwen3-0.6B" \
  DS_L3_OUT="$RUN_ROOT/qwen3-l3" DS_RUNTIME=shim \
  "$PY" agent/skills/deepspeed-torch-compat/scripts/qwen3_l3_probe.py

# 用真 PyTorch 解释器读取并逐张量比较两侧 safetensors
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/qwen3_l3_compare.py \
  --root "$RUN_ROOT/qwen3-l3" \
  --out "$RUN_ROOT/qwen3-l3/comparison.json"
```

2026-09-24 正式结果：596,049,920 参数；loss max abs `9.53674316e-7`；8 个抽样 logits max abs `1.90734863e-5`、全场相对误差 `1.36954708e-6`；round-trip 后 loss/logits 误差 0；oracle 与 shim 保存的 310 个张量逐张量完全相等；shim fallback 0。源 config SHA256 为 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`，源权重 SHA256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。

历史真实模型 FP32 Stage 1 初始化 OOM 已通过惰性优化器状态与 ACL 一维连续切片修复。该历史 Stage 0 结果只作为前置证据；当前 Stage 1 限定 L3 及 L4 子项按下一节复现。

## 历史75aa Qwen3 Stage 2累计L0–L4复验

同一真实 Qwen3-0.6B checkpoint、单机两 NPU、FP32、ZeRO Stage 2；AdamW lr=1e-5、betas=(0.9,0.99)、eps=1e-6、weight_decay=0，每 rank batch=1、sequence=12、GAS=1，无 offload/裁剪。依赖和权重哈希同下节。先在有效调度分配内运行 oracle，再运行 shim，runtime/rank 独立缓存，默认会合预算120秒。

以下 R 是仓外本轮产物目录；站点 launcher 只负责 CANN、Slurm 分配、端口及状态目录。运行前更换有效作业号和全新输出目录，不照搬旧站点卡号。

| 层级 | 仓库可复用入口 | 本轮证据（R 下） |
| --- | --- | --- |
| L0 | qwen3_construct_manifest.py，DS_ZERO_STAGE=2 | qwen-stage2-l0-75aa-v1/comparison.json，四清单310参数+1 buffer一致 |
| L1/L2 | qwen3_training_probe_eps1e6.py，DS_ZERO_STAGE=2、DS_FULL_STEPS=3；qwen3_training_compare.py --zero-stage 2 --steps 3 | qwen-stage2-eps1e6-75aa-formal-v1/comparison.json，3,744跨运行时字段及3,720 rank同步检查通过 |
| L3 | qwen3_l3_roundtrip_probe.py，DS_ZERO_STAGE=2，读取上行三步训练报告 | qwen-stage2-l3-75aa-v1/comparison.json，四份报告训练来源哈希匹配、1,240保存张量与各自训练末步精确一致，tokenizer及任务loss/logits重载通过 |
| L4恢复 | qwen3_optimizer_resume_probe.py、qwen3_resume_compare.py --zero-stage 2 | qwen-stage2-resume-default120-v1/comparison.json，1,256字段通过；模型权重哈希精确恢复，公开加载优化器后第4步数值重放在固定容差内 |
| L4生成 | qwen3_generation_probe_mask.py、qwen3_generation_compare.py，DS_GEN_MODEL指向上行同配置L3权重 | qwen-stage2-gen-75aa-v1/comparison.json，cache与重算误差0，greedy有/无cache及beam逐token一致 |
| L4采样 | qwen3_sampling_default_probe.py、qwen3_sampling_compare.py，DS_GEN_MODEL仍指向同配置L3权重 | qwen-stage2-sample-75aa-v1/comparison.json，四个status均passed；top-k20/top-p0.95/temperature0.6，四新token两rank逐个一致 |

恢复探针没有全部参数梯度或优化器状态逐项哈希；全部参数梯度来自三步 training probe。生成是重载普通模型。full sort 151936维索引跨 runtime 每 rank 有11215处不同，完整排序值在固定容差内、每侧索引回取和降序语义正确，top20索引及top-p16候选一致；不要声称全排序索引精确一致。

可复用 probe 与比较器均在本 skill 的 scripts/。比较器进程退出0不必然代表JSON通过：查看结果前先断言独立oracle身份，再断言 comparison.json 的 status=passed、errors=[]；采样还须断言 operator_semantics_status、fixed_seed_token_status、sampling_l4_subtask_status 全部passed。缺证据、skip或CPU fallback不计支持。

站点启动脚本（仓外未版本化）：run-qwen-stage2-l0-75aa-v1.sh、run-qwen-stage2-formal-75aa-v1.sh、run-qwen-stage2-l3-75aa-v1.sh、run-qwen-stage2-l4-75aa-v1.sh，以及 run-qwen-stage2-resume-default120.sh。R/sync/tested-source-files-stage2-l4-gates.sha256.json 记录源码，包含保留的其他工作，不能当提交清单。完整门禁状态见报告，不以该限定L4替代全库门禁。

## 历史 7d20 Qwen3 Stage 1/2/3 单机双卡限定 L4

以下历史75aa真实模型验收固定Qwen3-0.6B源权重SHA f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b、FP32、Stage 1/2/3、eps=1e-6；依赖与设备限定沿用当时report。当前97b9的Stage 1/2/3 eps=1e-8结果分别见前文；这些历史eps=1e-6记录不用于补足新配置的层级。

- L0 构造：qwen3_construct_manifest.py；Stage 1/2/3 各有四份独立清单，310 个参数及 1 个 buffer 的名称、shape、dtype、device 与独立 oracle 一致。Stage 2 产物 qwen-stage2-l0-eps1e6-v1/。
- L1/L2：qwen3_training_probe_eps1e6.py 训练三步，再用 qwen3_training_compare.py 对拍完整 logits/loss、全部参数梯度及更新、输入嵌入梯度和两个 rank 的参数同步；qwen-full-eps1e6-7d20-v1/comparison.json 中 3,744 个跨运行时字段在固定容差内，shim fallback 0。
- L3：qwen3_l3_roundtrip_probe.py 对三步训练后的全部 310 个权重作 safetensors 保存加载、任务 loss 和 tokenizer 对拍；Stage 1 产物 qwen-l3-eps1e6-7d20-v2/，Stage 2 产物 qwen-stage2-l3-eps1e6-v1/。仅 Stage 0 历史权重验证不能替代三步训练配置。
- L4 确定性生成：qwen3_generation_probe_mask.py 对 prefill/decode 全词表、KV cache 与重算、greedy 有/无 cache、beam 的 token 逐一核对；产物 qwen-gen-eps1e6-7d20-v5/，两 rank 通过。
- L4 恢复：qwen3_optimizer_resume_probe.py 保存模型及 AdamW 状态并恢复第 4 步，再运行 qwen3_resume_compare.py；qwen-resume-eps1e6-7d20-v2/comparison-tolerance.json 通过。严格逐位重放 comparison.json 失败，shim 五个接近零参数最坏差 3.6379788e-12，必须保留该细节。
- L4 随机采样：qwen3_sampling_default_probe.py 使用 checkpoint 默认 top-k=20、top-p=0.95、temperature=0.6；qwen3_sampling_compare.py 读取真实训练权重、完整 logits 与采样输出。qwen-sample-eps1e6-7d20-v10/comparison.json 的 status、operator_semantics_status、fixed_seed_token_status、sampling_l4_subtask_status 均为 passed，四个新 token 在两 rank 逐个一致。v9 的跨运行时 token 失败保留作修前证据；该比较器仅验采样子项，不能据此声明全库 L4。
- Stage 2 全流程：qwen3_training_probe_eps1e6.py 设 DS_ZERO_STAGE=2、DS_FULL_STEPS=3，两rank各跑三步；qwen3_training_compare.py --zero-stage 2 --steps 3 输出 qwen-stage2-eps1e6-formal-v1/comparison.json，3,744 项跨运行时和3,720项rank同步全部通过，fallback0。qwen3_l3_roundtrip_probe.py 以 DS_ZERO_STAGE=2 读三步结果。qwen3_optimizer_resume_probe.py 保存和恢复模型/AdamW再续训第4步，qwen3_resume_compare.py --zero-stage 2 对拍 qwen-stage2-resume-eps1e6-v2/comparison.json：1,256字段通过，shim重放4字段非逐位相等、最大1.45519152e-11，仍在预设容差内。qwen3_generation_compare.py 对 qwen-stage2-gen-eps1e6-v1/comparison.json 的14项全词表logits及greedy/beam通过；qwen3_sampling_compare.py 对 qwen-stage2-sample-eps1e6-v1/comparison.json 的默认top-k/top-p固定种子四token通过。初次恢复120秒HCCL等待超时记录在 logs/qwen-stage2-resume-eps1e6-v1/，诊断重试临时设置JT_RENDEZVOUS_TIMEOUT_S=300，不能删去失败记录。
- Stage 3 全流程：qwen3_construct_manifest.py 构造四份清单，qwen-stage3-init-v1/ 验证双rank参数分片；qwen3_training_probe_stage3.py 使用 DeepSpeed 公开 safe_get_full_grad/safe_get_full_fp32_param 读取全量梯度和更新后权重。qwen3_training_compare.py --zero-stage 3 --steps 3 对拍 qwen-stage3-eps1e6-formal-v2/comparison.json：3,744项跨运行时和3,720项跨rank全部通过，fallback0；输入梯度最坏绝对误差0.000702500343，固定阈值0.001545917511。qwen-stage3-l3-eps1e6-v1/ 记录训练后310权重保存重载差0及tokenizer对拍。qwen-stage3-gen-eps1e6-v1/comparison.json 的14项全词表logits/greedy/beam与cache重算通过；qwen-stage3-sample-eps1e6-v1/comparison.json 的固定种子默认top-k/top-p采样通过。qwen-stage3-resume-eps1e6-v1/comparison.json 验证模型及AdamW状态恢复后第4步续训，1,256项跨运行时通过，恢复全权重哈希精确一致；同运行时重放并非逐位相同，oracle最坏9.536743164e-7、shim最坏1.192092896e-7，均在固定容差内。该结论仅适用本节配置。
- Stage 3 hook 修复：真实模型 ZeRO-3 首次前向触发 DeepSpeed 默认 forward pre-hook 的3参数 TypeError。独立真PyTorch确认默认pre-hook即使forward有kwargs也只收到(module,args)；python/jittor/_core/module.py 改为仅在 with_kwargs=True 或原生Jittor hook时传第三参数，tests/nn/test_module_hooks.py 增加两条回归。修前1 failed/1 passed，修后27 passed/0 skipped。此修复属于共享hook协议，不能仅凭Stage 3案例推断其他生态库均通过。
- 多机边界：老师暂缓多机，当前 npu 分区也仅有 cscg-hw01 一个节点；adapter 只允许单节点 WORLD 1/2。多机未验证，但不作为本轮单机双卡验收条件。

用独立真 PyTorch 解释器执行比较器：python agent/skills/deepspeed-torch-compat/scripts/qwen3_sampling_compare.py --root <采样根目录> --generation-root <确定性生成根目录> --model-root <训练后权重根目录> --out <采样根目录>/comparison.json。它检查完整排序数值/索引、Top-20 与 Top-p 候选、同侧种子重放、跨 rank 一致性和跨运行时 token 差异；原始数组与日志不进主仓库。

## 在 shim 上跑

原版导入复现：

```bash
JITTOR_TORCH_SHIM=1 "$PY" -c 'import torch; assert hasattr(torch, "_torch_compat_install_context"); import deepspeed'
```

组件对拍（先 cpu，再 npu，各自独立进程/缓存）：

```bash
REAL_TORCH_PYTHON="$ORACLE" JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_TORCH_SHIM=1 JITTOR_TEST_DEVICES=cpu "$PY" -m pytest -q compat/tests/torch/test_npu_device_protocol.py   compat/tests/torch/test_torch_adagrad_compat.py   compat/tests/torch/test_torch_grad_bucket_protocol.py   compat/tests/torch/test_tensor_sparse_flag.py
```

Stage 0 单卡三步训练探针保留为独立 L2 入口：

```bash
JITTOR_TORCH_SHIM=1 "$PY" agent/skills/deepspeed-torch-compat/scripts/stage0_probe.py   run --runtime shim --device npu --fixture "$RUN_ROOT/fixture.npz" --out "$RUN_ROOT/shim"
```

## 在原生 torch 上跑

```bash
JITTOR_TORCH_SHIM=0 "$ORACLE" -c 'import torch; assert not hasattr(torch, "_torch_compat_install_context"); print(torch.__version__, torch.__file__)'
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage0_probe.py fixture --out "$RUN_ROOT/fixture.npz"
JITTOR_TORCH_SHIM=0 DS_ACCELERATOR=npu "$ORACLE"   agent/skills/deepspeed-torch-compat/scripts/stage0_probe.py   run --runtime oracle --device npu --fixture "$RUN_ROOT/fixture.npz" --out "$RUN_ROOT/oracle"
```

## 对拍

```bash
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage0_probe.py compare   --oracle "$RUN_ROOT/oracle" --candidate "$RUN_ROOT/shim" --out "$RUN_ROOT/comparison.json"
```

两侧共享固定权重、三批输入和目标，检查前向、loss、全部参数/输入梯度及每步更新。
检查 mode、真实设备、版本、fixture SHA256、engine.py SHA256、shape/dtype、有限值及 fallback。
误差按整个字段最大量级为分母；阈值为 `2e-5 + 2e-4 * field_scale`。
只完成 oracle 不能执行有效对拍；玩具 shape 不可引用为性能结果。

## 断点与分流

| 断点 | 层与范围 |
| --- | --- |
| NPU 设备发现/选择/同步 | compat 薄包装 native ACL；真卡对照 torch_npu，其他 npu 协议未实现 |
| GradBucket 导入 | compat 公开同一不透明类型；构造/执行明确拒绝，不提供假的 DDP bucket |
| Adagrad 缺失 | native 实现 dense FP32；compat 仅负责状态、closure 和类型发布 |
| FX Interpreter 缺失 | 真 FX 图执行是 C4；原版 DeepSpeed 可选 DeepCompile 无条件导入阻塞 eager |
| ARM CPU SHM 构建缺 immintrin.h | 上游 x86 扩展的环境限制，真 PyTorch 也失败 |
| DeepCompile C++ ABI/CUDA | 当前 NPU 范围不支持；如需改变边界，按 C5 统一讨论 |

仓外实验可对固定版 DeepSpeed 做可选模块延迟导入的上游补丁；两侧使用相同补丁。
通过 `DS_EXPERIMENTAL_SOURCE=<独立源码根目录>` 指定该副本，记录 patch SHA256。
这不修改已安装包，不伪造 FX，不等于原版通过；`compare` 要求 engine SHA256 一致。
补丁本身和原始失败日志保存在 lab，不作为公共 compat 行为修改。

## early provider 实验（未作为正式 adapter 接入）

仓外独立 DeepSpeed 0.17.6 副本另外包含 early provider hook，以及把私有 `_modules`
写入改为公开 `add_module` 的上游补丁。两侧使用相同副本和受限 provider。
完整补丁、provider 包及 SHA256 清单见开发记录 §4.0e 和未版本化证据目录；
不能拿修改过的源库结果宣称原版直接支持。

- 在任何 `import torch` / `import deepspeed` **之前**固定源码路径与 provider 环境。
- provider 在首次 get_accelerator 时选定，不能 CPU 导入后再切换；探针检查 bound method owner。
- 密集 `Tensor.is_sparse` 仅补公开协议，不能据此声明稀疏梯度或稀疏 optimizer。
- `run` 先写 `.construction.json`，即使后续训练失败也保留参数、buffer、device/backend 证据。
- `compare` 只比较数值轨迹；构造清单和通信 backend 必须另验，不能省略。
- CPU world-size-1 可能请求 Gloo 却报 MPI，是未解决缺陷；数值恒等 collective 不证明 Gloo。
- NPU 要用 native HCCL 启动，并查询已注册 communicator 的 size/rank，实际执行一次 native collective。
  仅 `hccl_ops is not None` 或环境变量写 hccl 都不能证明初始化成功。
- HCCL binding CPU 回归：`$PY -m pytest tests/backends/comm/hccl/test_hccl_op_bindings.py -q`。
- 该历史训练模型 4 个参数、0 个 buffer；正式 L0 已另测 3 个非空 buffer，不能把两次证据混作同一运行。

## Qwen3 全量训练记录与比较

使用 `scripts/qwen3_training_probe.py`，在已分配的单机两张 NPU 上分别启动真 PyTorch 与 shim；两个 runtime 顺序运行，每侧 rank 同步启动。站点 launcher 负责 provider、CANN、HCCL、设备可见性及各 rank 独立缓存，禁止直接占用未分配设备。

- 固定 DeepSpeed 0.17.6、Transformers 4.56.2、Qwen3-0.6B 本地真实权重，FP32、eager、ZeRO-1、AdamW、batch 每卡 1、sequence 12、GAS 1；不外推其他配置。
- 设置 `DS_RUNTIME=oracle|shim`、`DS_MODEL_PATH`、`DS_QWEN_FULL_OUT`、`DS_FULL_STEPS=3` 和两 rank 启动环境。输出根目录必须是新的仓外路径，不能覆盖已有 report。
- 两侧使用同一探针；权重哈希、依赖、参数清单、设备与 oracle 身份写入报告。shim 不保证 `torch.__file__` 存在，记录其 Jittor 实现源路径；oracle 必须记录真实 torch 路径。
- 浮点输入使用从可训练 embedding 计算的 `inputs_embeds`，不 detach；保存输入梯度。整数 token 不要求梯度。该结果只声明这一输入路径。
- `engine.backward` 后、`engine.step` 前，全 rank 按相同参数顺序调用 `safe_get_full_grad`。ZeRO-1 的 `parameter.grad` 可已清空，必须检查重建后的梯度，不能把 None 当作训练成功；也不能对已经平均的全梯度再除 world size。
- 每一步保存全部参数梯度、更新权重、完整 logits、loss；每 rank 最后写 `report.json`。执行完成不等于数值通过。
- 使用 `scripts/qwen3_training_compare.py --root <输出根> --out <输出根>/comparison.json --steps 3` 比较。流式读取数组，检查覆盖、dtype/shape、finite、两侧配置、全部元素和各 rank 同步；全场参考量级用于相对误差，容差必须在实验前固定。
- 三步训练不是累计 L4 的充分条件；真实权重 round-trip、恢复及目标工作流仍需同范围证据。原始数组不进仓库。

## 同权重诊断重放

多步训练产生数值差异时，用 `scripts/qwen3_training_replay.py` 区分当前反向计算与先前权重更新累积：设置 `DS_REPLAY_SOURCE` 为完整三步训练记录根目录，`DS_REPLAY_OUT` 为新仓外目录，其余 provider/两rank 环境沿用全量训练入口。固定重放第二步更新后的权重与第三步输入，分别使用共同oracle权重、各自权重；直接调用模型，不调用DeepSpeed Engine。保留28层输出和梯度，确认fallback为0。

使用 `scripts/qwen3_replay_compare.py --replay-root <重放目录> --training-root <原始训练目录>` 查看差异及旧记录重现情况。这是固定Qwen3配置的诊断入口，不是成熟度门禁。同权重通过不能抵消多步训练失败；近零梯度反号可能经AdamW放大，先核对真实梯度与更新公式，再判断实现是否错误，不裁掉梯度或调宽容差掩盖问题。

## 坑与假绿

- GradBucket 名字存在不证明 DDP 通信 hook 可用。
- Adagrad `step` 为 CPU FP32 scalar，`sum` 必须跟随参数设备；加载后仍要验证。
- DeepSpeed `engine.eval()` 操作底层 module；验证 `engine.module.training`，不能假定 engine 自身标志同步。
- ARM CPU 原生 DeepSpeed 也可能在 SHM JIT 阶段失败；`DS_BUILD_SHM_COMM=0` 不等于运行时禁用。
- 不加载真实 torch_npu 二进制到 shim 中冒充 NPU API。
- CPU 导入后公开 set_accelerator(NPU) 会留下 CPU-bound 内存方法；不能把这个切换当作干净的 NPU bootstrap。
- 内置 NPU builder 构造依赖 torch_npu.__file__；设备查询通过不代表 builder/Stream/RNG/内存接口通过。
- skip、单边基线、patched import 和组件单测都不能升级库成熟度。

## 证据

见 [2026-09-22 开发记录](../../../docs/results/2026-09-22-deepspeed-torch-compat.md)。
原始日志、NPZ、JSON、第三方补丁在仓外 lab，未版本化。
结果过期触发条件：Jittor 基线、DeepSpeed/PyTorch/torch_npu/CANN 版本、补丁/后端，以及 adapter 依赖的 get_hccl_world_info、owned_runtime_hook、transaction 或 module_patcher 公开契约改变。

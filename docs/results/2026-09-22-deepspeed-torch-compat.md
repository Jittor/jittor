# DeepSpeed torch compat 开发记录

## 1. 一句话状态

当前工作树基线为 **97b9aab4455b145ca1fb431af3504d705ac1edf3 + 保留修改及FP32 ACL RMS公共guard修复**。97b9aab4455b145ca1fb431af3504d705ac1edf3加保留修改与公共FP32 ACL RMS guard；Qwen3-0.6B、单机两NPU、FP32 AdamW eps=1e-8、每rank batch1/sequence12/GAS1、无offload/裁剪：Stage 1/2/3各自限定累计L0–L4。各Stage三步全部310参数梯度/更新、logits/loss与输入梯度3744跨runtime/3720rank检查，rank差0、fallback0；原5e-5+5e-5×同一步同类全场oracle量级容差未放宽。严格汇总`R/rms-guard-stage123-current-l4-summary.json`。旧7d20/75aa eps=1e-6各Stage L4保留，不迁入新配置。原eps=1e-8具体复现已修复、KI002移出active；KI001裸FX与KI003 PG1开放，L5未验证。

历史基线 **7d20b83a94c80680a716a7af71e1c6ceec2dad11 + 未提交修改** 上，DeepSpeed 0.17.6 显式可选 adapter 已在真实 Qwen3-0.6B、单机两张 Ascend 910B3、FP32、ZeRO Stage 1/2/3、AdamW eps=1e-6 的各自同范围配置下完成累计 **L0–L4**：构造、三步全量训练对拍、训练后真实权重保存加载、KV cache/greedy/beam/默认随机采样，以及模型和优化器 checkpoint 恢复后第4步续训。独立对照为 PyTorch 2.7.1+cpu + torch_npu 2.7.1.post4；shim fallback 0。这是限定任务范围的历史证据，不能推广至整库、其他配置或性能。

历史75aa99778df9d2834b333800f1721fc73693f043加保留修改曾为Qwen3-0.6B、单机两张Ascend 910B3、FP32、ZeRO Stage 1/2/3、AdamW eps=1e-6分别完成限定累计L0–L4，证据为`R/stage123-current-l4-summary.json`的18份比较JSON。其core、结构与142份源码一致性证明均属于75aa当时验证树；保留原结果，不作为新97b9运行源码或eps=1e-8支持等级证明。

| 负责人 | 赵佳祥 | 最后更新 | 2026-09-26 |
| --- | --- | --- | --- |
| 支持等级 | Stage 1/2/3各自限定累计L0–L4 | 当前工作 | 按各Stage六份JSON与143源码冻结严格验收 |
| 当前固定配置 | FP32，ZeRO Stage 1/2/3独立验收，AdamW lr=1e-5、betas=(0.9,0.99)、eps=1e-8、weight_decay=0；每卡batch 1、sequence 12、GAS 1；无offload/裁剪 | 后端 | 单节点、两NPU、HCCL |
| 库与依赖 | DeepSpeed 0.17.6、Transformers 4.56.2、Python 3.11.16、NumPy 1.26.4、safetensors 0.6.2 | 设备环境 | Ascend 910B3、CANN 9.0.0、驱动25.5.1 |
| 历史与当前工作树 | 历史7d20b83a/75aa9977；当前97b9aab4，均加保留修改 | 边界 | 不继承旧L4；其他配置、多机及L5未验证 |

原始JSON、数组、日志均为仓外未版本化产物。E、N、R分别表示`$JITTOR_LAB_ROOT/state/deepspeed-20260922`、`$JITTOR_LAB_ROOT/state/deepspeed-20260925`、`$JITTOR_LAB_ROOT/state/deepspeed-20260926`，C为`R/candidate-97b9`，S为`R/stage23-rms-guard`。R内75aa历史与97b9当前按提交、配置、目录及源码快照分列，不混计。

## 2. 支持了什么

本节集中维护当前结论，第4节保留逐字历史。真实Qwen3-0.6B源权重SHA256为`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`，310参数张量、596,049,920元素。每Stage独立连接自己的训练、权重任务及恢复产物；训练adapter允许单节点WORLD 1/2，本文真实Qwen证据限定WORLD 2。历史7d20/75aa累计证据使用FP32/AdamW eps=1e-6，当前97b9按Stage独立复验eps=1e-8，其他lr/betas/batch/sequence/GAS及无offload/裁剪配置相同。生成采样来自重载训练后权重的普通模型，不外推为DeepSpeed推理引擎或分片Engine内generate。复现入口为[runbook](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/deepspeed-torch-compat/SKILL.md)。

### 2.1 历史7d20基线真实Qwen累计证据

表中目录均位于 N；每个目录的 `oracle/shim` 各含两个 rank。比较器使用独立真 PyTorch，缺包和 skip 不计通过。

| 层 | 验收范围 | Stage 1 证据 | Stage 2 证据 | Stage 3 证据 |
| --- | --- | --- | --- | --- |
| L0 | 四份构造清单；310参数、1 buffer 的名称/shape/dtype/device一致 | `qwen-l0-7d20-v1/` | `qwen-stage2-l0-eps1e6-v1/` | `qwen-stage3-l0-eps1e6-v1/` |
| L1/L2 | 三步完整logits/loss、全部参数梯度/更新、输入梯度；各3,744项跨runtime与3,720项跨rank通过，fallback0 | `qwen-full-eps1e6-7d20-v1/comparison.json` | `qwen-stage2-eps1e6-formal-v1/comparison.json` | `qwen-stage3-eps1e6-formal-v2/comparison.json` |
| L3 | 对各自三步训练后的310权重保存加载，权重差0；任务loss与tokenizer对拍通过 | `qwen-l3-eps1e6-7d20-v2/` | `qwen-stage2-l3-eps1e6-v1/` | `qwen-stage3-l3-eps1e6-v1/` |
| L4生成 | 14项完整词表/greedy/beam字段、KV cache与重算通过；两rank同步 | `qwen-gen-eps1e6-7d20-v5/` 的四份report与数组；09-26只读复算14项通过 | `qwen-stage2-gen-eps1e6-v1/comparison.json` | `qwen-stage3-gen-eps1e6-v1/comparison.json` |
| L4采样 | checkpoint默认top-k=20、top-p=0.95、temperature=0.6；固定种子四个新token逐个一致，算子语义与采样子项均passed | `qwen-sample-eps1e6-7d20-v10/comparison.json` | `qwen-stage2-sample-eps1e6-v1/comparison.json` | `qwen-stage3-sample-eps1e6-v1/comparison.json` |
| L4恢复 | 保存模型/AdamW状态、精确恢复全权重、重复第4步；各1,256项跨runtime在固定容差内通过 | `qwen-resume-eps1e6-7d20-v2/comparison-tolerance.json` | `qwen-stage2-resume-eps1e6-v2/comparison.json` | `qwen-stage3-resume-eps1e6-v1/comparison.json` |

恢复探针每rank每phase的314字段为310个updated权重加input_ids/loss/logits/input_grad；**未采集全部参数梯度，未计算优化器状态哈希**。模型权重哈希精确恢复；优化器证据是公开 `load_optimizer_states=True` 加载后第4步数值重放通过，不宣称AdamW状态逐位或哈希精确恢复。历史7d20与新75aa恢复探针同口径，累计L2的全部参数梯度来自各自三步正式训练证据。恢复通过采用预先固定数值容差，**不宣称逐位续训完全一致**。同runtime重放最大差：Stage 1 oracle 0、shim 3.637978807e-12（10个字段非逐位相同；严格比较 `comparison.json` 失败记录保留）；Stage 2 oracle 0、shim 1.455191523e-11（4个字段）；Stage 3 oracle 9.536743164e-7、shim 1.192092896e-7（分别1、8个字段）。三组均恢复模型权重哈希精确一致，跨rank参数差0，shim fallback0。

小模型历史L2、Stage 0权重前置及97be8b98定向回归仍保留原基线，见E、N及第4节，不替代真实模型闭环。75aa三Stage历史复验证据见§2.4；当前97b9结果见§2.3。旧门禁或旧模型通过不能声明新提交完整支持。

### 2.2 数值精度与失败边界

三步训练采用 `5e-5 + 5e-5 × 同一步同类字段的全场oracle最大量级`，未调宽容差，不逐元素除以近零值。下表为 N 中三份正式 `comparison.json` 的最大绝对误差。

| 历史配置 | loss | logits | 参数梯度 | 输入梯度 | 更新权重 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Stage 1、eps=1e-6 | 3.814697266e-6 | 8.010864258e-5 | 5.035400391e-4 | 1.008987427e-3 | 2.536922693e-6 | passed |
| Stage 2、eps=1e-6 | 7.629394531e-6 | 9.918212891e-5 | 6.399154663e-4 | 1.281738281e-3 | 2.536922693e-6 | passed |
| Stage 3、eps=1e-6 | 2.861022949e-6 | 4.982948303e-5 | 3.515481949e-4 | 7.025003433e-4 | 2.400949597e-6 | passed |

**eps=1e-8历史失败保留，当前同范围复验已修复**：`N/qwen-full-v2/comparison.json`第三步rank0输入梯度差0.002607345581超过固定阈值0.001540468979，7d20重放及AdamW放大诊断原件保留。新97b9公共guard在FP32 ACL grad enabled时保留第三方RMS forward/autograd图，真实NPU契约红绿与正常canonical三步Stage 1 eps=1e-8全量独立比较均通过，原容差和全部梯度/更新覆盖不变；此具体复现从活跃KI002移除。该结论针对公共自动选路及所列模型/配置，不宣称native RMSNorm内核整体错误或修复全部eps=1e-8配置，也不迁移旧eps=1e-6 L4。

### 2.3 当前97b9公共RMS修复与eps=1e-8复验

**同步与源码冻结**：`R/sync-97b9/integration.json`记录整合到97b9aab4455b145ca1fb431af3504d705ac1edf3并保留旧工作树/已有修改。`R/sync-97b9/source-after-rms-guard.sha256.json`记录143个非Markdown源文件；本轮公共增量是`compat/torch/installers/nn/module_methods.py`的RMS guard和`compat/tests/torch/test_acl_rms_module_contract.py`。清单包含其他保留工作，不是干净提交或提交清单；后续文档工作不改变这143份源码哈希。

**公共行为与真实NPU回归**：当输入及weight均FP32、backend为ACL且grad enabled时，自动RMS候选不替代第三方forward，从而保留其实际公式、继承/覆盖forward、额外可训练参数及hooks；eval状态仍受grad mode控制，不依赖输入或weight是否requires_grad。其他后端/dtype不从此回归外推。`C/logs/rms-regression-red/pytest.log`为1 failed / 1 passed，guard后`C/logs/rms-regression-green/pytest.log`为2 passed / 0 skipped（17 warnings）。第一用例含16个shape/继承或覆盖/train或eval/frozen组合，独立torch_npu对拍输出、输入/weight/额外scale梯度及hooks，并检查真实device/fallback；第二用例以spy确认no_grad标准RMS仍走ACL融合。16组合不是16个pytest通过用例。

**BF16既有算子回归**：三条旧BF16测试初次在CPU默认device下失败；仓外`C/bf16_rms_device_entry.py`仅设置`torch.set_default_device('npu:0')`并保留原断言后，`R/logs/bf16-rms-device-v2/pytest.log`为3 passed / 0 skipped、12 warnings。此结果仅证明所列BF16算子回归，不声明DeepSpeed BF16/混合精度或性能。

**当前CPU与结构门禁**：`C/core-rms-guard.log`同次native 98 passed / 41 skipped / 1 xfailed、Torch 223 passed / 1 accelerator skipped，两子入口exit 0，合计321 passed / 42 skipped / 1 xfailed。`C/norm-cpu-rms-guard.log`为22 passed / 8 accelerator skipped；新增ACL契约文件在CPU两skip、零执行，不能当作其目标设备证据。`C/structure-rms-guard.log`收集/执行1398、1394 passed / 4 skipped、exit 0（2 accelerator、2 declared，other 0）。缺包、设备skip和未执行不计通过；这些门禁不证明整库全部后端/配置通过。

**当前逐Stage结论**：97b9aab4455b145ca1fb431af3504d705ac1edf3加保留修改与公共FP32 ACL RMS guard；Qwen3-0.6B、单机两NPU、FP32 AdamW eps=1e-8、每rank batch1/sequence12/GAS1、无offload/裁剪：Stage 1/2/3各自限定累计L0–L4。各Stage三步全部310参数梯度/更新、logits/loss与输入梯度3744跨runtime/3720rank检查，rank差0、fallback0；原5e-5+5e-5×同一步同类全场oracle量级容差未放宽。

| 当前97b9 eps=1e-8 Stage | loss | logits | 参数梯度 | 输入梯度 | 更新权重 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.861022949e-06 | 5.388259888e-05 | 4.154443741e-04 | 8.293390274e-04 | 1.888908446e-05 |
| 2 | 2.861022949e-06 | 5.388259888e-05 | 4.154443741e-04 | 8.293390274e-04 | 1.888908446e-05 |
| 3 | 5.722045898e-06 | 5.912780762e-05 | 5.055665970e-04 | 1.008033752e-03 | 1.825019717e-05 |

| 层 | Stage 1：R内路径 | Stage 2：R内路径 | Stage 3：R内路径 |
| --- | --- | --- | --- |
| L0 | `qwen-stage1-l0-97b9-rms-guard-v2/comparison.json` | `qwen-stage2-l0-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-l0-97b9-rms-guard-v1/comparison.json` |
| L1/L2 | `qwen-stage1-eps1e8-97b9-rms-guard-v1/comparison.json` | `qwen-stage2-training-97b9-rms-guard-v2/comparison.json` | `qwen-stage3-training-97b9-rms-guard-v1/comparison.json` |
| L3 | `qwen-stage1-l3-97b9-rms-guard-v2/comparison.json` | `qwen-stage2-l3-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-l3-97b9-rms-guard-v1/comparison.json` |
| L4生成 | `qwen-stage1-gen-97b9-rms-guard-v1/comparison.json` | `qwen-stage2-gen-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-gen-97b9-rms-guard-v1/comparison.json` |
| L4采样 | `qwen-stage1-sample-97b9-rms-guard-v1/comparison.json` | `qwen-stage2-sample-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-sample-97b9-rms-guard-v1/comparison.json` |
| L4恢复 | `qwen-stage1-resume-97b9-rms-guard-eps1e8-v1/comparison.json` | `qwen-stage2-resume-97b9-rms-guard-v1/comparison.json` | `qwen-stage3-resume-97b9-rms-guard-v1/comparison.json` |

**权重任务与生成采样**：已完成L3的各Stage四份source_training_report_sha256绑定自己的正式训练，1240个实际safetensors张量与末步精确一致，权重roundtrip0、logits自身roundtrip0、五token一致；任务loss差如表，未把数值容差通过称逐位一致。生成14字段、cache本侧差0、cached/uncached greedy与beam逐token一致；默认top-k20/top-p0.95/temperature0.6采样四status/errors空，四新token/top20索引/16候选一致。全排序索引差异如表，不宣称全部索引相同。

| Stage | L3自身loss roundtrip | gen prefill | gen decode | sort values | top20 values | top-p probability | 全排序索引不同/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.536743164e-07 | 2.002716064e-05 | 2.217292786e-05 | 1.788139343e-05 | 1.239776611e-05 | 9.401785285e-07 | 12610 |
| 2 | 9.536743164e-07 | 2.002716064e-05 | 2.217292786e-05 | 1.788139343e-05 | 1.239776611e-05 | 9.401785285e-07 | 12610 |
| 3 | 9.536743164e-07 | 1.966953278e-05 | 1.811981201e-05 | 1.966953278e-05 | 9.536743164e-06 | 2.667238848e-06 | 14178 |

**恢复口径**：每Stage独立恢复1256字段passed，四model_restored_exact true、rank同步0、shim fallback0；第4步同runtime重放差与非逐位字段如表，原阈值通过。314字段/phase/rank为310 updated权重加input_ids/loss/logits/input_grad，不含全部参数梯度/optimizer-state hash。checkpoint_sha256实际config摘要；内部逐参数模型hash断言以model_restored_exact记录，公开load_optimizer_states=True及第4步数值重放不等于优化器hash一致。

| Stage | 恢复loss | logits | 输入梯度 | 更新权重 | oracle重放/非逐位字段数 | shim重放/非逐位字段数 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1.907348633e-06 | 2.241134644e-05 | 8.964538574e-05 | 1.888908446e-05 | 9.536743164e-07 / 1 | 0.000000000e+00 / 0 |
| 2 | 9.536743164e-07 | 2.241134644e-05 | 8.964538574e-05 | 1.888908446e-05 | 0.000000000e+00 / 0 | 0.000000000e+00 / 0 |
| 3 | 2.861022949e-06 | 3.719329834e-05 | 7.677078247e-05 | 1.825205982e-05 | 9.536743164e-07 / 1 | 0.000000000e+00 / 0 |

**探针与验收归属**：Stage 2训练v1实际zero_stage=1，S/stage2-v1-invalid-evidence.json标为无效Stage2证据、保留原件。v2真实Stage2完整比较passed；外层shell在等待比较期间被补强编辑，随后语法退出1，S/stage2-training-result-check.json独立核对comparison SHA/status passed。此为外层执行记录异常，不是数值失败，原日志保留，未重跑数学或调宽阈值。 Stage 2/3仓外S/qwen3_training_stage{2,3}_eps1e8.py分别来自版本化eps1e6训练探针和Stage 3专用探针，只改optimizer_config eps=1e-6为1e-8；来源SHA及唯一替换见S/stage{2,3}-eps1e8-provenance.json。Stage 3保留初始化前完整shape，通过公开full-grad/full-FP32-param访问器取全量张量，不能把分片占位shape当完整参数。L0为普通模型310参数/1 buffer及requested stage，不单凭manifest证明分片Engine。 严格汇总为`R/rms-guard-stage123-current-l4-summary.json`；Stage1同97b9旧已验六JSON保留，Stage2/3按自己的六JSON单独验收后合并，143非Markdown源码冻结未变，源码清单含保留工作不是提交清单。证据各自绑定本Stage同次训练与保存权重，不借用旧提交/其他Stage补层。生成与采样来自普通重载训练后模型，不是DeepSpeed inference Engine或分片Engine内generate。其他eps/checkpoint/world配置、BF16全训练、offload、多机及L5未验证；KI001裸FX和KI003偶发PG1仍开放。

### 2.4 历史75aa三Stage各自限定累计证据

下表目录均位于R，基线为历史75aa加保留修改，指定FP32/AdamW eps=1e-6；与新97b9/eps=1e-8分列，不迁移支持等级。每个Stage单独连接自己的构造、正式训练、训练后权重、生成采样和恢复产物，不借用其他Stage补层。`R/stage123-current-l4-summary.json`由仓外`collect-stage123-current-l4.py`严格核查18份比较JSON及其SHA256后生成，status passed；同时核查独立真PyTorch C扩展/路径/身份、配置与原权重SHA、每Stage的L3来源绑定、NPU设备、模型恢复和shim fallback0。

| 层 | Stage 1证据 | Stage 2证据 | Stage 3证据 |
| --- | --- | --- | --- |
| L0 | `qwen-stage1-l0-75aa-v1/comparison.json` | `qwen-stage2-l0-75aa-v1/comparison.json` | `qwen-stage3-l0-75aa-v1/comparison.json` |
| L1/L2 | `qwen-stage1-eps1e6-75aa-formal-v1/comparison.json` | `qwen-stage2-eps1e6-75aa-formal-v1/comparison.json` | `qwen-stage3-eps1e6-75aa-formal-v1/comparison.json` |
| L3 | `qwen-stage1-l3-75aa-v1/comparison.json` | `qwen-stage2-l3-75aa-v1/comparison.json` | `qwen-stage3-l3-75aa-v1/comparison.json` |
| L4生成 | `qwen-stage1-gen-75aa-v1/comparison.json` | `qwen-stage2-gen-75aa-v1/comparison.json` | `qwen-stage3-gen-75aa-v1/comparison.json` |
| L4采样 | `qwen-stage1-sample-75aa-v1/comparison.json` | `qwen-stage2-sample-75aa-v1/comparison.json` | `qwen-stage3-sample-75aa-v1/comparison.json` |
| L4恢复 | `qwen-stage1-resume-default120-v1/comparison.json` | `qwen-stage2-resume-default120-v1/comparison.json` | `qwen-stage3-resume-default120-v1/comparison.json` |

三个Stage各自L0四清单的310参数/1 buffer及名称、shape、dtype、device/config等10项一致。manifest采集的是普通模型构造后的完整shape与requested stage；真实DeepSpeed各Stage Engine构造及执行另由各自正式三步训练记录支撑，不能将manifest当作分片后参数shape证明。Stage 3专用训练probe在初始化前保留完整shape，并通过公开full-grad/full-param访问器提取梯度及更新，不比较空分片占位shape。

三个Stage各自三步完整logits/loss、全部310参数梯度/更新和输入梯度均通过，3744项跨runtime、3720项跨rank，rank差0、errors/evidence_gaps空。原`5e-5 + 5e-5 × 同一步同类字段的全场oracle最大量级`容差未放宽；下表为历史75aa三份正式比较的最坏绝对差，全部字段在原阈值内。

| 历史75aa配置 | loss | logits | 参数梯度 | 输入梯度 | 更新权重 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1、eps=1e-6 | 4.768371582e-06 | 8.296966553e-05 | 4.825592041e-04 | 9.670257568e-04 | 2.536922693e-06 |
| Stage 2、eps=1e-6 | 5.722045898e-06 | 9.441375732e-05 | 5.988087505e-04 | 1.192092896e-03 | 2.540647984e-06 |
| Stage 3、eps=1e-6 | 3.814697266e-06 | 5.197525024e-05 | 1.469254494e-04 | 2.938508987e-04 | 2.400949597e-06 |


每个Stage的L3四份各310权重roundtrip差0，实际safetensors逐块对比同次训练step2全部参数，共1240项精确一致；四份source_training_report_sha256绑定各自正式训练报告，五token tokenizer一致，两shim fallback0。权重精确重载不等于全部任务输出逐位相同：Stage 1/3 oracle自身roundtrip loss最坏9.536743164e-7，Stage 2为0；三组logits自身roundtrip差0，均在既有任务容差内。Stage 2首个L3比较因未带训练使用的固定oracle-site而缺safetensors启动失败，补回同一`E/l3/oracle-site`后exit 0，没有更换依赖。

每个Stage的生成14项完整词表/greedy/beam字段通过，KV cache与重算本侧差0，cached/uncached greedy及beam逐token一致。默认top-k=20、top-p=0.95、temperature=0.6采样的总状态、算子语义、固定种子token及采样子任务四status均passed/errors空；四新token逐个一致，top20索引和top-p保留的16候选一致。全排序索引仍有下表差异，来自跨runtime原logits微差，不宣称全部排序索引相同。

| Stage | 生成prefill | 生成decode/重算 | 采样全排序值 | top20值 | top-p概率 | 全排序索引差异/每rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.002716064e-05 | 1.788139343e-05 | 2.002716064e-05 | 1.621246338e-05 | 3.808289247e-06 | 10279 |
| 2 | 1.478195190e-05 | 2.002716064e-05 | 1.478195190e-05 | 6.675720215e-06 | 1.149894828e-06 | 11215 |
| 3 | 1.287460327e-05 | 1.978874207e-05 | 1.287460327e-05 | 1.049041748e-05 | 2.219502114e-06 | 11680 |


三个Stage各自默认120秒恢复比较均为passed，1256项跨runtime通过、四报告executed/model_restored_exact为true、两shim fallback0，跨rank更新权重差0。以下同runtime重放差均在预设5e-5绝对阈值内，不宣称逐位续训一致；每组都是一轮默认预算成功，不能关闭历史Stage 2 PG1偶发超时。

| Stage | 恢复跨runtime loss | logits | 输入梯度 | 更新权重 | oracle重放 | shim重放 | shim非逐位字段/两rank合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.768371582e-06 | 7.724761963e-05 | 6.008148193e-05 | 2.536922693e-06 | 0.000000000e+00 | 1.164153218e-10 | 4 |
| 2 | 8.583068848e-06 | 8.773803711e-05 | 7.724761963e-05 | 2.536922693e-06 | 0.000000000e+00 | 7.275957614e-12 | 6 |
| 3 | 5.722045898e-06 | 2.574920654e-05 | 4.863739014e-05 | 3.095716238e-06 | 0.000000000e+00 | 1.455191523e-11 | 10 |


**恢复字段含义**：恢复probe报告中的`checkpoint_sha256`实际计算原始模型目录`config.json`摘要，本轮均为`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`；它不是safetensors原权重SHA或恢复模型哈希。正式训练报告的同名字段另指原safetensors权重SHA，不能混用。恢复probe内部逐参数计算saved/restored模型权重哈希并断言一致，报告以`model_restored_exact=True`记录结果，没有输出该哈希表。314字段/phase/rank为310 updated权重加input_ids/loss/logits/input_grad，1256字段不含全部参数梯度，也没有optimizer-state hash；优化器证据是公开`load_optimizer_states=True`后的第4步数值重放。累计L2全部参数梯度来自各自正式三步训练。

上述历史L4仅限75aa三组各自同提交/设备/模型/配置，不证明新97b9支持。生成与采样使用重载各自同次训练后权重的普通模型，不是DeepSpeed推理引擎或ZeRO分片Engine内generate。其他模型/配置、混合精度、offload、其他卡数、多机与L5仍未验收；历史eps=1e-8失败原件保留、同范围复现已修复；PG1偶发超时仍开放。

### 2.5 速度与显存

L5未验证：没有符合独立进程、两臂紧邻交替、多轮中位比值及区间要求的速度数据；首步backward及稳态峰值显存对照未完成。历史初始化live/reserved记录不等于峰值，不据此报告性能比值。当前reduce-scatter的all-reduce加设备切片实现也不构成性能通过证据。

## 3. 怎么算“支持”：L0–L5分层

严格累计，并按同一提交、设备、模型和配置验收。历史7d20/75aa的eps=1e-6三Stage各自L4保留；当前97b9 eps=1e-8按各Stage新证据逐层验收，不能拼接旧提交或其他配置。

| 等级 | 标准 | 历史7d20/75aa指定eps=1e-6 | 当前97b9 eps=1e-8 |
| --- | --- | --- | --- |
| L0 | 显式adapter、config、参数/buffer/device构造 | 三Stage各自四清单通过 | 各Stage四L0清单passed；分片Engine能力不单凭manifest声明 |
| L1 | 同权重同输入、完整前向输出独立对拍 | 三步logits/loss通过 | 各Stage三步完整logits/loss通过 |
| L2 | 全部参数/输入梯度、更新及rank同步 | 三Stage各自三步全量通过 | 各Stage3744跨runtime/3720rank检查、310参数三步通过 |
| L3 | 前层通过后，真实权重任务及保存加载 | 绑定各自训练，1240项保存核对/任务/tokenizer通过 | 各Stage来源绑定1240项保存核对/任务/tokenizer通过 |
| L4 | 前层同范围通过后，训练、恢复及目标生成采样 | 三Stage各自限定累计通过 | 各Stage14项生成/采样四status/1256恢复通过；限定L4 |
| L5 | 对齐速度与首步/稳态峰值显存 | 未验证 | 未验证 |

独立oracle、真实目标设备计算与fallback检查是共同条件。抽样、skip、单边执行、不同配置或旧提交结果均不能补足新基线累计等级。

## 4. 开发记录

### 2026-09-26 97b9 Stage 2/3 eps=1e-8独立复验

**当前范围**：97b9aab4455b145ca1fb431af3504d705ac1edf3加保留修改与公共FP32 ACL RMS guard；Qwen3-0.6B、单机两NPU、FP32 AdamW eps=1e-8、每rank batch1/sequence12/GAS1、无offload/裁剪：Stage 1/2/3各自限定累计L0–L4。各Stage三步全部310参数梯度/更新、logits/loss与输入梯度3744跨runtime/3720rank检查，rank差0、fallback0；原5e-5+5e-5×同一步同类全场oracle量级容差未放宽。

**失效轮与外层退出记录**：Stage 2训练v1实际zero_stage=1，S/stage2-v1-invalid-evidence.json标为无效Stage2证据、保留原件。v2真实Stage2完整比较passed；外层shell在等待比较期间被补强编辑，随后语法退出1，S/stage2-training-result-check.json独立核对comparison SHA/status passed。此为外层执行记录异常，不是数值失败，原日志保留，未重跑数学或调宽阈值。

**公开来源与完整张量**：Stage 2/3仓外S/qwen3_training_stage{2,3}_eps1e8.py分别来自版本化eps1e6训练探针和Stage 3专用探针，只改optimizer_config eps=1e-6为1e-8；来源SHA及唯一替换见S/stage{2,3}-eps1e8-provenance.json。Stage 3保留初始化前完整shape，通过公开full-grad/full-FP32-param访问器取全量张量，不能把分片占位shape当完整参数。L0为普通模型310参数/1 buffer及requested stage，不单凭manifest证明分片Engine。

| 当前97b9 eps=1e-8 Stage | loss | logits | 参数梯度 | 输入梯度 | 更新权重 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.861022949e-06 | 5.388259888e-05 | 4.154443741e-04 | 8.293390274e-04 | 1.888908446e-05 |
| 2 | 2.861022949e-06 | 5.388259888e-05 | 4.154443741e-04 | 8.293390274e-04 | 1.888908446e-05 |
| 3 | 5.722045898e-06 | 5.912780762e-05 | 5.055665970e-04 | 1.008033752e-03 | 1.825019717e-05 |

**同次训练后权重任务**：已完成L3的各Stage四份source_training_report_sha256绑定自己的正式训练，1240个实际safetensors张量与末步精确一致，权重roundtrip0、logits自身roundtrip0、五token一致；任务loss差如表，未把数值容差通过称逐位一致。生成14字段、cache本侧差0、cached/uncached greedy与beam逐token一致；默认top-k20/top-p0.95/temperature0.6采样四status/errors空，四新token/top20索引/16候选一致。全排序索引差异如表，不宣称全部索引相同。

| Stage | L3自身loss roundtrip | gen prefill | gen decode | sort values | top20 values | top-p probability | 全排序索引不同/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.536743164e-07 | 2.002716064e-05 | 2.217292786e-05 | 1.788139343e-05 | 1.239776611e-05 | 9.401785285e-07 | 12610 |
| 2 | 9.536743164e-07 | 2.002716064e-05 | 2.217292786e-05 | 1.788139343e-05 | 1.239776611e-05 | 9.401785285e-07 | 12610 |
| 3 | 9.536743164e-07 | 1.966953278e-05 | 1.811981201e-05 | 1.966953278e-05 | 9.536743164e-06 | 2.667238848e-06 | 14178 |

**恢复重放实值**：每Stage独立恢复1256字段passed，四model_restored_exact true、rank同步0、shim fallback0；第4步同runtime重放差与非逐位字段如表，原阈值通过。314字段/phase/rank为310 updated权重加input_ids/loss/logits/input_grad，不含全部参数梯度/optimizer-state hash。checkpoint_sha256实际config摘要；内部逐参数模型hash断言以model_restored_exact记录，公开load_optimizer_states=True及第4步数值重放不等于优化器hash一致。

| Stage | 恢复loss | logits | 输入梯度 | 更新权重 | oracle重放/非逐位字段数 | shim重放/非逐位字段数 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1.907348633e-06 | 2.241134644e-05 | 8.964538574e-05 | 1.888908446e-05 | 9.536743164e-07 / 1 | 0.000000000e+00 / 0 |
| 2 | 9.536743164e-07 | 2.241134644e-05 | 8.964538574e-05 | 1.888908446e-05 | 0.000000000e+00 / 0 | 0.000000000e+00 / 0 |
| 3 | 2.861022949e-06 | 3.719329834e-05 | 7.677078247e-05 | 1.825205982e-05 | 9.536743164e-07 / 1 | 0.000000000e+00 / 0 |

**收口**：严格汇总`R/rms-guard-stage123-current-l4-summary.json`只计最终JSON与真实来源。143源码冻结未变，launcher哈希/eps-only来源单列，源码快照不是提交清单。证据各自绑定本Stage同次训练与保存权重，不借用旧提交/其他Stage补层。生成与采样来自普通重载训练后模型，不是DeepSpeed inference Engine或分片Engine内generate。其他eps/checkpoint/world配置、BF16全训练、offload、多机及L5未验证；KI001裸FX和KI003偶发PG1仍开放。此前整个§4原字节保留，未commit/push。

### 2026-09-26 97b9 FP32 ACL RMS公共修复后Stage 1 eps=1e-8限定累计L4

**公共修复与门禁**：整合97b9aab4455b145ca1fb431af3504d705ac1edf3、保留原工作及143份非Markdown源码冻结。仅公共FP32 ACL grad-enabled RMS候选保留第三方forward/autograd图，没有模型forward覆盖或DeepSpeed专属数值补丁。真实NPU契约红1 failed/1 passed转绿2 passed/0 skipped（16组合+no_grad融合spy）；BF16旧三测试仅在仓外设置正确NPU默认device并保留断言后3 passed/0 skipped。CPU core同次321 passed/42 skipped/1 xfailed，norm 22 passed/8 accelerator skipped（新ACL文件两skip零执行），完整结构1394 passed/4 skipped、1398收集执行、exit 0。

**正式独立训练通过**：L0 v2四清单passed；正常canonical三步Stage 1 eps=1e-8比较passed/errors与evidence_gaps空，3744跨runtime/3720rank字段、全部310参数梯度/更新及输入梯度，rank差0、fallback0、四probe SHA与仓库原训练探针一致。未改原5e-5/5e-5全场尺度容差；loss最坏2.861022949e-6、logits5.388259888e-5、参数梯度4.154443741e-4、输入梯度8.293390274e-4、更新权重1.888908446e-5。原eps=1e-8失败记录保留，但该同范围复现现已修复、KI002移出活跃表；不归咎native RMSNorm内核整体或所有配置。

**L3独立来源闭环**：v1旧固定eps前置失败保留；仓外副本仅将来源eps检查1e-6改1e-8，来源SHA见C/l3-eps1e8-provenance.json。v2首个CPU比较缺固定oracle-site/safetensors而启动失败，补回同版依赖后comparison passed；四训练报告SHA绑定本次formal-v1，1240个实际保存张量与各自step2精确一致，权重roundtrip差0、五token一致。任务自身roundtrip loss最坏9.536743164e-7、logits0，在原阈值内；新基线该配置的L0–L3独立证据成立。

**L4完成及严格汇总**：本次L3 v2保存权重重载后的普通模型生成14字段passed、cache差0、greedy/uncached/beam逐token一致；prefill最坏2.002716064e-5、decode2.217292786e-5。默认采样四status passed、四新token/top20索引/16个top-p候选一致；排序值1.788139343e-5、top20值1.239776611e-5、top-p概率9.401785285e-7，完整排序索引每rank12610项不同，不称全索引一致。恢复1256字段passed、四model_restored_exact为true、rank差0、fallback0；oracle第4步重放9.536743164e-7且1字段非逐位，shim0且无非逐位字段，原预设容差通过。恢复lab副本只改eps，保留公开load_optimizer_states=True及数值断言，没有全部参数梯度或optimizer hash；checkpoint_sha256为config摘要。C/collect-rms-guard-current-l4.py exit 0生成R/rms-guard-current-l4-summary.json，验证六最终JSON SHA及143源码冻结，当前97b9 Stage 1 eps=1e-8限定累计L0–L4成立。历史75aa eps=1e-6三Stage L4不迁入新提交；Stage 2/3、其他配置、BF16/offload/DeepSpeed推理Engine及分片Engine内generate、多机和L5未验证，PG1及裸FX保持开放。全部此前§4历史字节保留，未commit/push。

### 2026-09-26 75aa Stage 1/3独立补齐并汇总三Stage限定累计L0–L4

**同步与源码**：本机独立git ls-remote exit 0确认远端仍为75aa9977，与HEAD相同；服务器HTTP2及有界HTTP1.1失败保留在`R/sync/stage13-resume-upstream.json`。恢复前及本轮结束142个非Markdown源文件哈希均与Stage 2门禁快照一致，见`R/sync/stage13-source-unchanged.json`；新增站点包装与原始产物在仓外，无运行源码/probe修改。清单包含其他保留dirty工作，不是提交清单。

**本轮新独立证据**：Stage 1/3各自四L0清单通过；正式三步全部310参数训练3744项跨runtime/3720项跨rank通过，rank差0、errors/evidence_gaps空、原容差未放宽、shim fallback0。各自L3绑定本Stage同次formal-v1来源SHA，四份1240个实际safetensors张量与训练step2逐块精确一致；五token一致。Stage 1/3 oracle自身任务roundtrip loss最坏9.536743164e-7、logits差0，权重差0；不将数值容差通过写成任务逐位一致。生成各14字段/cache/greedy/beam通过；默认采样四status、四新token、top20索引和16候选通过，全排序索引每rank仍分别有10279/11680项差异。

**恢复与累计口径**：Stage 1/3默认120秒恢复各1256字段通过、模型恢复断言精确、跨rank更新权重差0、shim fallback0；oracle重放均0，shim分别1.164153218e-10/1.455191523e-11、4/10字段非逐位，仍在原阈值内。恢复report的checkpoint_sha256是config.json摘要；模型哈希在probe内部比较并以model_restored_exact记录，没有optimizer hash或全部参数梯度采集。公开优化器加载后的第4步数值重放与各自正式三步全部梯度证据连接，不能互相冒充。

**当前汇总**：`R/collect-stage123-current-l4.py`exit 0生成`R/stage123-current-l4-summary.json`，核查18份比较JSON SHA、独立oracle身份/原权重/配置、各Stage的1240项权重来源绑定、生成采样及1256恢复字段。当前75aa Stage 1/2/3各自指定配置累计L0–L4；不借用其他Stage、旧提交、skip或单边结果补层。生成是重载训练后权重的普通模型，不是DeepSpeed inference/分片Engine generate。core/结构门禁沿用同142源码的已通过结果，未重复全量测试。eps=1e-8、PG1偶发超时、裸FX边界仍开放，多机及L5未验证；本轮未commit/push。此前全部§4历史字节保留。

### 2026-09-26 恢复后75aa Stage 2限定累计L4与严格门禁复验

**同范围模型闭环**：75aa99778df9d2834b333800f1721fc73693f043加未提交修改，单机两NPU、Qwen3-0.6B、FP32、ZeRO-2、AdamW eps=1e-6。R中新四份L0清单通过；正式三步全部310参数训练比较3744项跨runtime/3720项跨rank通过，rank差0、errors/evidence_gaps空、shim fallback0，原5e-5/5e-5尺度容差未放宽。绑定formal-v1报告hash的L3四份各310权重保存加载精确，1240个实际safetensors张量与同次step2训练精确一致，任务与五token tokenizer通过。生成14项完整输出/cache/greedy/beam字段通过；默认top-k20/top-p0.95/temperature0.6采样四status通过、四新token逐个一致，top20索引及16候选一致，完整排序索引每rank仍有11215差异，不宣称全部索引相同。与先前同75aa/同配置默认120秒模型及公开optimizer加载后第4步数值恢复合计，当前Stage 2限定累计L0–L4成立；恢复无全部参数梯度或optimizer-state hash，PG1偶发风险不关闭，新Stage 1/3与L5不借用此结论。

**结构证据纠正**：旧restart日志实际仍在启动阶段SystemExit 3，后续canonical轮217 passed / 2 skipped / 1 failed，root MANIFEST漏四个Sort/Multinomial源码路径；生成器修复清单并check通过，固定pytest-xdist 3.6.1补缺依赖。manifest-fixed轮在用户暂停时停止，不能记通过。恢复轮358 passed / 0 skipped / 1 failed归于ACL source-count测试陈旧；仅改47→49、50→52且保留完整集合断言，精确文件14 passed / 0 skipped、exit 0。完整counts-fixed轮1390 passed / 4 skipped / 3 failed归于注册表陈旧计数与继承MKL环境污染；只改测试计数/新增launcher断言及fixture隔离，定向11 passed / 0 skipped、exit 0。registry-mkl-fixed完整轮1393 passed / 4 skipped但strict accounting把Torch主动ignore的native-only文件记作collected 0而exit 1，不计门禁通过。选中文件清单模式回归先红1 failed，再绿全文件5 passed / 0 skipped、exit 0；未执行普通文件仍保持negative control。单列native dtype结构5 passed / 0 skipped、1 warning是CPU测试契约证据。

**当前门禁**：`R/core-mode-snapshot-fixed.log`统一core同次native 98 passed / 41 skipped / 1 xfailed、Torch 223 passed / 1 accelerator skipped，两个子入口exit 0，合计321 passed / 42 skipped / 1 xfailed；与以前分次结果区别记录。最终`R/structure-mode-snapshot-fixed-full.log`为collected/executed 1398、1394 passed / 4 skipped、exit 0，无collected 0；4 skip为2 accelerator、2 declared、other 0。布局`R/layout-stage2-l4.log`exit 0，定向diffcheck通过。最终142个非Markdown文件快照与sort-grad仅6项清单/测试政策变化，NPU运行源码/probe均未变；不声明整库全部设备/配置通过。新L0/L3站点比较入口与L4比较器均保留在R，未版本化；具体产物及复现限制见§2.4、§5.5。旧§4开发记录逐字保留。

### 2026-09-26 陈旧dtype skip清理后的Torch core与真实NPU sort投影反向验证

**CPU验证**：在75aa9977加未提交修改上，仅清理两个已确认陈旧的dtype skip/测试说明后，`R/core-torch-dtype-fixed.log` 为223 passed / 1 accelerator skipped、exit 0。native core仍是前一轮98 passed / 41 skipped / 1 xfailed、exit 0；本次只重跑Torch入口，不合并成同次运行，不把skip计通过。旧严格runner exit 1及独立oracle32组精确dtype/values、原test body核对保留。

**真实设备定向证据**：`R/sort-grad-npu.log` 的 `TestACLTorchCompat::test_sort_axes_options_and_projection_backward_matches_torch_npu` 为1 passed / 0 skipped。在真实NPU上，12组3×4 FP32 sort值/索引/投影反向梯度与独立torch_npu精确一致，fallback不增；覆盖dim 0/-1、升降序、stable选项和重复键，其他shape/dtype及完整排序梯度未由本用例验收。

**未完成范围**：首轮 `R/structure-sort-fixed.log` 在执行测试前触发JIT工具更新SystemExit 3，未执行测试；规范CPU启动器物理路径后已显式重启一次，`R/structure-sort-fixed-restart.log` 仍运行中。启动日志保留，结构复验尚无通过结果。此前默认120秒Stage 2恢复只通过一轮，不关闭旧PG1偶发风险。恢复探针仍无全部参数梯度采集或optimizer状态哈希，仅模型权重hash精确恢复及公开优化器加载后的第4步数值验证。新75aa完整累计L4和整库通过均不声明。


### 2026-09-26 默认120秒Stage 2恢复通过一轮与恢复证据范围更正

**验证与基线**：75aa99778df9d2834b333800f1721fc73693f043加未提交修改，单机两NPU、FP32、AdamW eps=1e-6、ZeRO-2；本轮launcher显式120秒并隔离75aa rank缓存，oracle/shim均exit 0。原始比较 `R/qwen-stage2-resume-default120-v1/comparison.json` 为passed/errors空；四份report executed、模型权重hash精确恢复，两个shim fallback0。

**数值与时序**：1,256字段跨runtime通过，rank同步差0；oracle重放差0，shim最大7.275957614e-12、两rank共6字段非逐位相同。跨runtime最坏loss 8.583068848e-6、logits 8.773803711e-5、input_grad 7.724761963e-5、updated权重2.536922693e-6。WORLD→PG1文件mtime差29秒（11:14:59→11:15:28），两rank默认120秒初始化成功，无90秒采样；load-start/end事件已记录。本轮只证明一次默认预算恢复通过，未锁定旧偶发超时根因，不关闭PG1风险。

**恢复证据范围更正**：314字段/phase/rank是310个更新后权重加input_ids/loss/logits/input_grad，恢复脚本没有全部参数梯度采集或优化器状态hash。精确恢复仅指模型权重hash；优化器通过公开load_optimizer_states=True后第4步数值重放验收，不能据此声称优化器状态逐位精确。旧7d20脚本同口径；历史累计L4的全部参数梯度证据来自同范围三步正式训练，不由恢复探针补足。

**后续门禁**：CPU dtype两个旧skip已由独立oracle32组精确dtype/values及原test body确认陈旧；已仅更改测试skip/说明，原core Torch与结构复验待完成。NPU计算结束后仅该测试metadata变化，运行源码未改。此条不声明新75aa完整累计L4、整库通过或多轮稳定性。


### 2026-09-26 独立Tensor前端Adam惰性状态的历史类型缺陷修复

**现象与最小复现**：初次core门禁native 98 passed / 41 skipped / 1 xfailed，torch 220 passed / 3 skipped / 1 failed。失败用例 `compat/tests/torch/test_independent_frontend.py::test_independent_tensor_installation_preserves_native_type` 的子进程第185行要求state_dict中所有jt.Var均属于Torch Tensor；独立前端Adam/AdamW的exp_avg、exp_avg_sq却是native Var。仓外 `R/cpu-independent-state/red_probe.py` 在相同CPU前端安装下打印该类型差异；它exit 0，只作诊断，实际红断言来自仓库门禁。

**根因与基线归属**：`_ensure_adam_group_state` 惰性新建状态时调用 `_state_buffer` 未激活Tensor前端scope；已有两个旧工作树包含相同原实现，因此不是75aa上游整合引入的新缺陷。优化器值计算未表现出错误，失败是兼容状态类型契约。

**分层与处理**：仅在 `compat/torch/optimizer_api.py` 惰性新建state处进入 `tensor_frontend(get_install_context(jt).target_namespace.Var, like=param)`，不改native optimizer类型，不扩大ABI/C5边界。源码修复由主任务完成；无新增提交，基线仍为75aa9977加未提交修改。

**已完成验证**：`R/cpu-independent-state/green_probe.py` 以真实断言验证native SGD/Adam/AdamW状态仍为Var、独立Torch前端数值与状态类型/copy、GC、load及组内参数形状32→2动态替换后的惰性状态，exit 0；摘要在green.stdout-excerpt.txt。原失败仓库用例 `R/frontend-fixed.log` 为1 passed / 0 skipped；native hook回归 `R/native-hooks-restart.log` 为27 passed / 0 skipped。初始core日志与红探针都保留。

**尚未完成**：更广Torch定向回归与默认120秒两NPU恢复复验仍运行中；上述CPU修复证据不等于整个core/整库通过，不升级新基线L4。

### 2026-09-26 上游75aa同步与默认120秒rendezvous复验进行中

**基线**：在独立工作树整合上游75aa99778df9d2834b333800f1721fc73693f043并保留已有未提交修改；旧7d20工作树及原始产物保留。历史7d20限定L4结论集中更新§1–3，不将其自动沿用至新基线。

**诊断**：Stage 2恢复v1两个rank的WORLD均初始化成功，rank1等待PG1文件120秒失败；尚未到subgroup HCCL初始化，rank0停滞点缺栈证据。v2文件时间显示WORLD→PG1为89秒，因此300秒重试通过不能作为超时根因修复。现有栈采样晚于模型加载，后续诊断需记录加载及new_group边界时间和有效rank/path，并检查锁持有者、编译子进程与文件发布状态；不扩大C5边界。

**验证状态**：新CPU定向门禁和默认120秒恢复实验尚未结束；产物在R，启动器显式设置120秒并隔离新的75aa rank缓存。本条只记录同步与复验进行中，不声明新基线L4、整库通过或PG1稳定性已修复。历史失败日志继续保留在N的stage2-resume-v1目录。


### 4.0d 2026-09-25 同权重重放区分反向计算与训练轨迹偏差

**现象**：97be8b98基线第三步rank 0一个输入梯度超差；上游更新至7d20b83a。

**最小复现**：`qwen3_training_replay.py`读取已保存第二步权重和第三步token；依次执行同oracle权重与各自权重两种条件，直接模型forward/backward，保存输入、输入梯度、logits、loss和28层输出/梯度。仍是模型级诊断，尚非单算子最小复现。

**根因**：同权重误差0.0002384，在原阈值内；各自权重误差0.0026073，准确重现旧结果。已找到首步近零梯度反号、AdamW正确执行但更新方向相反的具体元素。最初数值分歧来源及各元素对最终超差的贡献仍未定位。

**分层**：当前新增诊断工具和证据；尚无理由修改core/compat/adapter公共行为。

**处理**：通过本机Git bundle获取真实上游提交，在新工作树保留并整合全部已有修改，未改旧工作树。增加同权重/各自权重重放及分层记录工具，保持原容差；未改变优化器算法。没有新增commit SHA。

**验证**：真实两NPU、两runtime重放完成，shim fallback 0；各自权重重放与旧记录的12个输出/梯度字段最大差均0。`N/qwen-replay-7d20-v1/comparison.json`与`N/qwen-full-v2/adam-update-diagnostic.json`可复核。新基线仅完成该诊断范围，不将旧基线全量训练结论自动沿用。



### 4.0c 2026-09-25 Qwen三步全量记录完成，发现第三步梯度超差

**现象**：首轮三步训练结束后，记录脚本读取 shim 的 `torch.__file__` 报 `AttributeError: __file__`。修正记录方式后的第二轮，两侧各两rank均完成三步，fallback 0；全量比较仍为失败，第三步一个输入梯度元素超过预先固定的容差。

**最小复现**：使用第5节全量入口，固定 Qwen3-0.6B 权重、FP32、ZeRO-1、AdamW、每卡batch 1、sequence 12、三个固定输入序列；当前是模型级复现，尚未缩成独立算子用例。

**根因**：环境记录异常来自测试误用可选模块元信息，不是训练失败；数值超差根因待定位。**分层**：前者为测试；后者暂不指定 core/compat/adapter。

**处理**：shim记录实际Jittor实现路径，oracle保留真实torch路径。新增版本化全量探针与分块比较器，逐项记录310个参数的梯度/更新、输入梯度及完整输出，保留失败结果且不放宽容差。未提交，无新增SHA。`safe_get_full_grad` 在backward后、step前按相同rank顺序调用，不对已平均梯度重复除以world size。

**验证**：完整结果 `N/qwen-full-v2/comparison.json`；输入诊断 `forward-input-diagnostic.json`；源文件哈希 `N/source-qwen-full-v2-sha256.json`；失败首轮日志 `N/logs/qwen-full-v1/`，复跑日志 `N/logs/qwen-full-v2/`。四份report各含1,872项，依赖一致：DeepSpeed0.17.6、Transformers4.56.2、NumPy1.26.4、safetensors0.6.2。语法检查及git diff --check通过，未替代完整仓库门禁。



新的在上，后续只追加条目；结论变化修改第2节。

### 4.0b 2026-09-25 Stage 3空整数张量与空广播修复后通过双卡对拍

**现象**：新同步路径先暴露 `float64[0] -> int32[0]` Cast被NPU拒绝；修正dtype后，空广播root的 `aclrtMemcpy` 返回107000。

**最小复现**：两rank各创建 `torch.tensor([], dtype=int, device="npu:0")` 并调用 `torch.distributed.broadcast(tensor, src=0)`；真实Stage 3的模块顺序检查会执行该路径。

**根因**：`compat/torch/types.py`将Python int映射到原生int32，未在构造前规范为PyTorch的int64；`backends/comm/hccl/ops/hccl_broadcast_op.cc`对零元素张量仍调用零字节显存拷贝。

**分层**：dtype属于C2协议修复；空广播属于C4后端边界实现。**处理**：仅规范builtin int到int64；root仅在num>0时拷贝，所有rank仍调用集合通信；未修改ABI或扩大浮点dtype支持。无新增commit。

**验证**：CPU通信与工厂回归26 passed/0 skipped；独立真torch_npu空张量/2**40保真回归1 passed/0 skipped；修复前Stage 3日志保留。修复后两个rank三步运行及原生PyTorch对拍通过，精度见第2节。原生参考为同日同模型输入及版本的运行，修复后重跑shim；未复用旧55d59e1c的oracle数据。

### 4.0a 2026-09-25 两NPU Stage 1重新通过，Qwen一步执行成功，Stage 3暴露dtype缺口

**现象**：新基线Stage 1固定MLP三步正式对拍通过；真实Qwen oracle/shim两个rank均执行到最终sync并输出passed，历史HCCL超时未复现。新基线Stage 3则在空列表dtype=int处触发NPU不支持的float64[0]到int32[0]Cast。

**最小复现**：正式Stage 1 probe及Qwen两rank一步脚本；dtype独立最小输入为 `torch.tensor([], dtype=int)` 与包含 `2**40` 的整型输入。

**根因**：通信恢复是在合入上游同步collective修复及本轮reduce-scatter衔接修复后观测到，尚无单补丁因果隔离证据。dtype问题独立真PyTorch 2.7.1 CPU确认builtin int=int64，当前shim映射不同。**分层**：通信整合验证；dtype属于torch compat工厂协议。**处理**：仅将 `compat/torch/types.py` 中builtin int映射为int64；补 `test_torch_factory_fidelity` 的空数组/大整数独立oracle测试。无提交SHA。

**验证**：Stage 1 comparison.json正式对拍通过，loss0、output/partition_grad/updated各1.490116119e-8、input_grad6.053596735e-9、rank同步0、fallback0。Qwen sample-comparison.json仅抽样通过：loss4.768371582e-6、16logits1.335144043e-5、4个参数首尾各8元素差0，不代表全部310参数及梯度验收。dtype CPU 26 passed / 0 skipped，NPU及Stage 3重跑进行中。证据位于 `state/deepspeed-20260925/results/stage1-formal/` 与 `state/deepspeed-20260925/qwen/`，均未版本化。
### 4.0 2026-09-25 新基线整合后修复reduce-scatter漏同步

**现象**：在独立工作树整合上游97be8b98后，通信定向CPU测试首次为1 failed / 13 passed；上游新增同步collective的 `issued` 契约，但reduce-scatter仍使用默认False，已表达的多rank通信不会在同步API返回前flush。

**最小复现**：`test_collective_is_synchronous` 与 `test_distributed_api_owners` 的同步行为用例；多rank `reduce_scatter_tensor(..., async_op=False)` 应在返回前同步output。

**根因**：`compat/torch/installers/distributed.py` 的 `_reduce_scatter_tensor` 最后调用 `_collective_result(output, async_op)`，未传 `issued=size > 1`；两个非成员分支也未明确标识不发通信。**分层**：torch compat通信公开协议，属于整合后的契约衔接。**处理**：三个调用点补齐issued，新增6个模拟sync行为case，覆盖同步/异步、singleton/非成员及两个入口；原工作树不变，未提交，无提交SHA。

**验证**：修复后两组CPU定向回归20 passed / 0 skipped；layout与git diff --check通过。模拟测试证明调用协议，不等于真实通信验收；NPU首次编译中，没有新基线HCCL/Qwen结果，不升级等级。原始测试日志路径待本轮归档补入。



倒序追加；历史失败保留，当前结论只更新第 2 节。

### 4.0m 2026-09-24 Qwen3 真实 checkpoint 暴露 meta dtype 与 tied-weight 协议缺口

**现象**：Qwen3-0.6B 首次加载依次遇到三类问题：Transformers 把 Jittor 的当前 NPU 推断为显式默认设备并要求 Accelerate；显式 `torch.empty(..., device="meta")` 返回原生 `jittor.Var`，普通 FP32 dtype 与 `torch.float8_e4m3fn` 占位对象比较时误入计算类型转换；修复加载后，round-trip 又把绑定的 `lm_head.weight` 报为缺失。真实模型 ZeRO Stage 1 还在 `deepspeed.initialize` 后同步时达到约 59.61 GiB 并 OOM。

**最小复现**：
```python
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, dtype=torch.float32, attn_implementation="eager")
engine, *_ = deepspeed.initialize(model=model, optimizer=optimizer, config=stage0)
with torch.no_grad():
    loss = engine(input_ids=ids, labels=ids).loss
engine.module.save_pretrained(out, safe_serialization=True)
reloaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
```

**根因**：meta 工厂分支绕过 `tensor_frontend`，所以返回值没有 Torch dtype 身份；`named_parameters(remove_duplicate=False)` 虽接受参数，却调用了已去重的原生公开遍历，Transformers 无法发现绑定别名；模块 `_parameters`/`_buffers` 注册表也错误去重同一对象的多个注册名。Transformers 默认设备推断则把当前 Jittor NPU 当成用户显式设置的 device map。**分层**：meta 与模块注册语义属于 core/torch compat；默认设备推断属于 Transformers adapter；Stage 1 大模型显存是未解决的范围边界，不以降低断言掩盖。

**处理**：meta 工厂仍使用 Jittor 占位存储，但进入同一 Tensor frontend 并保留 meta 设备标记；float8 仍严格禁止计算和分配。参数/buffer 枚举从核心 `_named_vars(..., remove_duplicate=False)` 取得全部注册名，再按公共参数决定是否去重；注册表保留别名。Transformers adapter 只抑制由 Jittor 当前设备被误判出的隐式 device map，显式 `meta` 保持不变。L3 正式范围固定为 Qwen3-0.6B、FP32、Stage 0；Stage 1 OOM 记录为未验证。无提交 SHA。

**验证**：独立 PyTorch 2.7.1 + torch_npu 2.7.1.post4 与 shim 均在两张 Ascend NPU 上加载 596,049,920 参数并运行语言模型 loss。loss max abs `9.53674316e-7`，logits max abs `1.90734863e-5`；`save_pretrained`/`from_pretrained` 后两侧各自输出误差 0，oracle 与 shim 保存的 310 个张量逐张量完全相等，fallback 0。正式比较结果为仓外 `l3/qwen3-formal/comparison.json`。meta/dtype/Module 定向回归 118 passed；adapter 回归 5 passed。

### 4.0l 2026-09-24 ZeRO Stage 3 参数重绑定、custom Function 和梯度分片链路贯通

**现象**：修复参数 all-gather 后，Stage 3 仍没有输入梯度和分片梯度；保存重绑定前的历史 leaf 后，记录能在 backward 后清理，但计算图仍被 DeepSpeed 的自定义 `torch.autograd.Function` 截断。补齐该语义后，反向进入梯度通信并明确报 `torch.distributed.reduce_scatter` 缺失。继续补接口后，两 rank 三步可运行，但首次严格对拍发现参数保持初值：DeepSpeed 在 step 前才把 FP32 flat partition 临时放回 AdamW 参数组，而 compat 优化器只读构造/反向期的内部 `param_group["grads"]`，未消费刚赋给 `param.grad` 的分片梯度。

**最小复现**：
```python
class Detached(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.detach()
    @staticmethod
    def backward(ctx, grad):
        return grad

flat.grad = partition_grad
optimizer.param_groups[0]["params"] = [flat]
optimizer.step()
```

**根因**：共有四个独立协议缺口：一是 `Parameter.data` 重绑丢失前向图历史 leaf；二是自定义 Function 的 forward 返回 detach 张量时，compat 未把输出挂回自定义 backward；三是公开 `reduce_scatter_tensor/reduce_scatter` 未安装；四是 torch 优化器应在 step 时读取当前 `param.grad`，而 native Jittor 优化器使用平行 grads 列表。**分层**：前三项属于 autograd/torch.distributed compat，第四项属于 optimizer compat；HCCL 当前没有公开 native reduce-scatter，因此用已有 HCCL all-reduce 加本 rank 的 device-local 切片保持数值语义，性能另行验收。

**处理**：只为带 post-accumulate hook 的可训练 backward leaf 保存 `.data` 重绑历史，并在 backward 中把历史梯度汇总到当前 Parameter、调用一次 hook 后清理；自定义 Function 在输入需要梯度时恢复浮点输出的梯度起点；安装 list/tensor 两种 reduce-scatter 及 `_reduce_scatter_base`，校验 world size、shape 和 list 长度；torch 风格 optimizer step 前按当前参数的公开 grad 槽重建内部 grads 列表。新增历史 leaf 生命周期、detach-output custom Function、reduce-scatter 路由/错误契约和 AdamW 动态参数组回归。adapter 仅开放单机两 rank、FP32 AdamW、Stage 3 默认模式，显式 contiguous、offload 等继续失败关闭。无提交 SHA。

**验证**：CPU 定向回归：distributed owner **8 passed**，动态参数组 AdamW **1 passed**，历史 leaf **1 passed**，detach-output custom Function **1 passed**；adapter 契约和可重定位 **23 tests / OK**。随后正式 adapter、无诊断旁路在两张 Ascend NPU 上完成 Stage 3 两 rank 三步，真实 HCCL，fallback 0。与独立 PyTorch 2.7.1 + torch_npu 2.7.1.post4 严格对拍：loss 最坏绝对误差 `0`、输出 `1.49011612e-8`、输入梯度 `5.58793545e-9`、FP32 分片梯度 `1.49011612e-8`、更新参数 `7.45058060e-9`；oracle 与 shim 的更新参数跨 rank 最大差均为 `0`。版本化入口为 `stage3_probe.py` 与共享 `stage1_compare.py`；原始 JSON、NPZ 和日志保留仓外。

### 4.0k 2026-09-23 ZeRO Stage 3 参数聚合恢复后仍受 autograd leaf 重绑定阻塞

**现象**：独立真 PyTorch/torch_npu 的两卡 ZeRO Stage 3 固定用例可完成 3 步；正式 shim 范围门禁正确拒绝 Stage 3。仅在仓外诊断探针临时绕过配置门禁后，先后遇到模块 repr、`_parameters` 映射替换、NPU FP64 norm 和参数 all-gather 问题。最关键的数值现象是：本地 `ds_tensor` 分片在两个 rank 上都正确，但第一次前向所见完整参数和输出全为 0。

```python
flat = torch.zeros(8, device="npu:0")
part = flat.narrow(0, rank * 4, 4)
returned = torch.cat([left, right], out=part)
# 修复前 returned 有正确值，但 part 和 flat 对应区间仍为 0。
```

**根因**：`compat` 的 `torch.cat` 接收却忽略 `out=`；DeepSpeed 用 `torch.cat(..., out=本 rank 的 flat buffer 切片)` 准备 all-gather 输入，两个 rank 因而通信了全 0。修复这一 C2 协议后，完整参数和首次前向恢复。随后出现独立的 C4 autograd 边界：Stage 3 在前向后用 `param.data = torch.empty(0, ...)` 释放完整参数，反向前再把聚合参数绑定回同一 Parameter；compat 的 `.data` 替换通过 `_update` 创建当前 leaf，前向图中的旧 leaf 不再是优化器持有的当前参数。最小单层探针中，前向后即使只把 Parameter 重新绑定到相同完整值，输入梯度仍正确，但参数梯度从正确值变为 `None`；完整 Stage 3 诊断则表现为分区梯度全 0、输入梯度未发布和权重不更新。

**分层**：`torch.cat(out=...)` 是 C2 compat 协议；跨前向/反向保存 PyTorch 式 Parameter storage/leaf 身份属于 C4 autograd/core 能力。模块 repr 的 wrapped initializer 签名、可替换 `_parameters` 映射及固定 Stage 3 NPU norm 源码变换是到达数值诊断点的前置修复。Stage 3 不是“少一个接口”即可可信支持。

**处理**：`torch.cat` 在空、单输入和普通拼接路径统一通过 `out.copy_(result)` 写回并返回同一 `out` 对象；新增覆盖 `out` 为 `narrow` view、且底层 flat buffer 同步更新的回归。为诊断补齐 wrapped initializer 的 `__code__`/`__defaults__`、Module `_parameters` MutableMapping setter，以及固定 SHA/唯一锚点的 Stage 3 NPU FP64 norm 变换。没有通过禁用参数释放来伪装 Stage 3，因为那会失去 ZeRO-3 的参数分片语义。生产 `scope.py` 仍只允许 Stage 0/1/2，Stage 3 fail-closed。无提交 SHA。

**验证**：`torch.cat(out=view)` CPU 定向回归 **1 passed / 0 skipped**；两 rank NPU 最小复现中，两个 rank 的目标切片、底层 flat buffer 和 all-gather 结果均逐元素正确，fallback 0。adapter 契约与可重定位测试在前置修复后为 **23 tests / OK**。修复后重跑已支持的 Stage 2 默认路径并对独立 oracle 严格比较：loss 误差 0，输出/分片梯度 `1.49011612e-8`，输入梯度 `5.58793545e-9`，更新参数 `7.45058060e-9`，跨 rank 更新参数差 0，fallback 0。仓外 Stage 3 诊断可完成两 rank、3 步、真实 HCCL、fallback 0，但训练语义明确失败：每步输入梯度缺失，分区梯度全 0，更新权重保持初值；该运行只用于定位，不计作 L0–L2 通过证据。正式 Stage 3 门禁未打开。

### 4.0j 2026-09-23 ZeRO Stage 2 被范围门禁拒绝但共享实现已满足固定范围

**现象**：真 PyTorch/torch_npu 的两卡 Stage 2 固定用例完成 3 步；shim 在 `deepspeed.initialize` 进入计算前由 adapter 抛出 `NotImplementedError: Only ZeRO Stage 0 and Stage 1 ... are verified`。

```python
config["zero_optimization"] = {"stage": 2}
engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
```

**根因**：Stage 1/2 共用固定版 DeepSpeed 的 `runtime/zero/stage_1_and_2.py`，已有 dtype 分桶、归约切片写回和 NPU FP32 norm 变换同时覆盖 Stage 2；阻塞点是 `scope.py` 仍按旧验证范围只允许 Stage 0/1。**分层**：第三方 adapter 范围契约和验证入口；无需新增 native/compat 算子。

**处理**：把无 offload、无额外 ZeRO 选项的 Stage 2 纳入允许集合，并要求与 Stage 1 相同的单机两 rank 真实 HCCL WORLD；Stage 3、offload 和其他未验证选项继续 fail-closed。范围契约同时覆盖 Stage 2 默认/非连续梯度和 Stage 3 拒绝。将严格比较器改为从结果 JSON 读取 `zero_stage`，新增 `stage2_probe.py`，不复制数值判定逻辑。无提交 SHA。

**验证**：adapter 契约与可重定位测试 **22 tests / OK**。Stage 2 默认连续梯度和 `contiguous_gradients=False` 均由独立真 PyTorch/torch_npu oracle 与 shim 各两个 rank 完成 3 步；loss 最坏绝对误差 0，输出 `1.49011612e-8`，输入梯度 `5.58793545e-9`，优化器分片梯度 `1.49011612e-8`，更新参数 `7.45058060e-9`；两侧更新参数跨 rank 最大差均为 0，shim fallback 为 0。证据目录为仓外 `stage2-formal/`、`stage2-formal-noncontig/`。

### 4.0i 2026-09-23 Stage 1 checkpoint 在类型构造、便携序列化和设备恢复处中断

**现象**：固定模型的两卡 Stage 1 训练 3 步后，DeepSpeed `save_checkpoint`/`load_checkpoint` 依次暴露三处兼容差异：checkpoint tag 校验中的 `torch.ByteTensor([sha1.digest()])` 无法构造；参数 shape 的 `jittor_core.NanoVector` 无法 pickle；从 CPU checkpoint 调用 `Module.load_state_dict(assign=False)` 后，NPU 参数被移到 CPU。加入 shape round-trip 回归后还发现 `torch.load(weights_only=True, map_location=...)` 会拒绝或降级兼容层的 `torch.Size`。

```python
engine.save_checkpoint(root, tag="step3", client_state={"completed_steps": 3})
with torch.no_grad():
    engine.load_checkpoint(root, tag="step3", load_optimizer_states=True)
```

**根因**：`ByteTensor` 构造器未区分裸 `bytes` 与列表内 bytes；便携序列化未把 Jittor 的 shape 值转换为 PyTorch 可保存的 shape 类型，且安全加载/`map_location` 未保留兼容层的不可变 `torch.Size` 元组子类；`load_state_dict` 只保留目标 dtype，没有保留目标 placement。**分层**：公开 torch 构造、serialization 和 `nn.Module.load_state_dict` 协议，属于 compat 修复；DeepSpeed checkpoint 工作流仍走原库公共 API。

**处理**：严格对拍真 PyTorch 后，仅对嵌套 `bytes`/`bytearray` 实现 `ByteTensor` 的 uint8 展开，裸 `bytes` 继续抛 `TypeError`；将 `NanoVector` 转成 `torch.Size`，在 weights-only 白名单中只加入兼容层自身的 `_TorchSize`，并让 `map_location` 保留 tuple 子类；`load_state_dict(assign=False)` 同时按目标参数恢复 CPU/NPU placement。回归分别加入 `test_torch_compat_dtype.py`、`test_serialization_api_owners.py` 和 `test_torch_compat_serialize.py`。无提交 SHA。

**验证**：定向回归为构造器 **2 passed**、便携序列化 **2 passed**、真实 NPU placement **1 passed**。相关三个 CPU 测试文件完整回归为 **63 passed / 4 skipped**；4 项分别因 accelerator 或额外 torch 条件不满足而未验证，不计为通过。DeepSpeed adapter 契约与可重定位测试 **22 tests / OK**。随后在新的仓外 checkpoint 目录中重跑两卡 HCCL：两个 shim rank 均完成保存、精确恢复和续训，fallback 0。与独立真 PyTorch/torch_npu oracle 对拍：保存前输出 `1.49011612e-8`、loss `0`、输入梯度 `5.58793545e-9`、分片梯度 `1.49011612e-8`、更新参数 `7.45058060e-9`；恢复权重 `7.45058060e-9`；恢复后第 4 步输出 `1.49011612e-8`、loss `0`、输入梯度 `3.72529030e-9`、更新参数 `7.45058060e-9`；两侧跨 rank 参数差均为 0。oracle 的 torch_npu 在 checkpoint 赋值时同样要求外层 `torch.no_grad()`，两侧探针使用同一公开上下文。该结果是固定模型的 L3 前置证据，仍不满足真实 checkpoint 与任务要求。

### 4.0h 2026-09-23 ZeRO Stage 1 两卡归约结果未写回参数梯度

**现象**：Stage 1 能完成两卡三步训练，但 shim 的输出最坏差约 `9.38e-4`、更新参数最坏差约 `2.39e-3`。逐 rank 诊断显示，shim 的优化器分片梯度分别等于各自本地梯度；真 PyTorch 使用两 rank 的平均梯度。
```python
engine.backward(loss)
partition = engine.optimizer.single_partition_of_fp32_groups[0].grad
# 正确结果应对应两 rank 梯度归约后的分片，而不是当前 rank 的本地梯度分片。
```
**根因**：一是 DeepSpeed 的 `split_half_float_double()` 依赖 `Tensor.type()` 返回 `torch.npu.FloatTensor` 一类设备字符串，compat 的通用类型名使梯度被静默过滤；二是默认连续梯度路径依赖 `grad.data = buffer_slice.data` 建立 PyTorch 共享存储别名，Jittor 这里只复制当前值，归约后的 flat buffer 不会自动反映回参数梯度。独立 HCCL all-reduce 及窄切片写回探针均通过，因此根因不在 HCCL。**分层**：固定 DeepSpeed 0.17.6 的第三方 adapter 源码变换；通信仍使用 native HCCL。
**处理**：按 `tensor.dtype` 分桶；连续梯度归约后显式把 flat buffer 切片复制回各参数梯度，再构造优化器分片。两处变换都要求固定源码 SHA、唯一锚点并在不匹配时失败；支持范围只扩至单机两卡 Stage 1、FP32 AdamW、无 offload/裁剪，默认连续梯度和显式 `contiguous_gradients=False`。Stage 2/3 继续拒绝。无提交 SHA。
**验证**：正式 adapter、无运行时 monkeypatch。默认连续梯度与非连续梯度两条路径均完成两卡 HCCL 三步；loss 最坏绝对误差 0，输出 `1.49011612e-8`，输入梯度 `5.58793545e-9`，优化器分片梯度 `1.49011612e-8`，更新参数 `7.45058060e-9`；oracle 与 shim 的更新参数跨 rank 最大差均为 0，shim fallback 为 0。原始证据为仓外 `stage1-formal/` 与 `stage1-formal-noncontig/`；版本化复现入口为 `stage1_probe.py`、`stage1_compare.py`。adapter 契约测试 20 tests / OK，覆盖源码锚点 fail-closed、两条连续梯度配置及 Stage 2/offload 拒绝。

### 4.0f 2026-09-22 显式 adapter 完成限定范围 L0

**现象**：仓外 provider 实验不能作为正式交付；原版 eager import 仍加载可选 FX，导入后再切 accelerator 会留下 CPU bound method，Engine 私自写 `_modules`。
**最小复现**：`activate(device="npu")` 后 `import deepspeed`；`l0_probe.py run` 构造含 4 参数、3 非空 buffer 的固定模型，分别使用最小/显式 batch 配置。
**根因**：固定 DeepSpeed 0.17.6 的导入时序与库私有模块布局依赖。**分层**：第三方 adapter；数学和通信继续归 native/compat，不伪造 FX/torch_npu。
**处理**：独立 `adapters/jittor_adapters/deepspeed/`，显式激活，在首次 accelerator 选择之前用上游公开 setter 注册受限 provider；对校验后的原文件做内存中延迟 DeepCompile 导入和公开 `add_module` 变换。只允许固定 3 份源码及 2 份 CPU/NPU 构建 manifest；共享 `require_version`、所有权事务、可重定位契约。明确拒绝 CPU Engine、未验证配置、optimizer 模式、非零 local rank 和 reload。新增 required 生态/定期 L0 门禁。无提交 SHA。
**验证**：原版未修改 oracle 与正式 adapter 的 NPU 两套构造完全一致，CPU 两类纯 Python wheel 构造子集通过；10 依赖版本逐项一致、4 参数/3 buffer 数值差 0、fallback 0。非法 batch config 两侧异常类型相同；ZeRO 1、FP16/BF16、profiler、未知配置被拒绝。独立 wheel 搬出仓内 adapters 路径后仍可构造。CPU manifest 生成依赖 `DS_ACCELERATOR=cpu DS_BUILD_OPS=0 TORCH_DEVICE_BACKEND_AUTOLOAD=0`；不需要 torch_npu 构建 CPU wheel。

### 4.0g 2026-09-22 公共 HCCL 状态避免把单进程假初始化当作真实后端

**现象**：仅查询 `torch.distributed` 的 singleton 状态不能证明实际 HCCL 已初始化。
**最小复现**：`from jittor.distributed import get_hccl_world_info; print(get_hccl_world_info())`。
**根因**：需读取实际原生 communicator，不能靠 env 字符串或已编译算子推断。**分层**：C4 native/backend 只读查询。
**处理**：native `hccl_is_initialized()` 检查实际 WORLD communicator；公开 `get_hccl_world_info()` 不加载、不初始化，缺失时返回未初始化，真实查询异常直接传播；adapter 经该公开入口限定 rank0/size1。原有分布式初始化缺陷未被改名掩盖。无提交 SHA。
**验证**：CPU 10 passed；真实 NPU 10 passed/0 skipped，执行 native all_reduce 并强制零 fallback。首轮 NPU 测试虽已加载 ACL，但原生计算开关仍为 0，严格断言失败；补测试显式 `flag_scope(use_cuda=1)` 后通过，失败日志保留。L0 探针首次还使用了旧版 `fp16_enabled` 配置字段，oracle 明确报错后改为 0.17.6 的 `float16_config.enabled`，未调整库语义来迎合测试。

### 4.0c 2026-09-22 HCCL 算子声明换行阻塞真实 communicator 初始化

**现象**：以 `JT_HCCL_WORLD_SIZE=1` 启动，`setup_hccl` 报 `Wrong op args in .../hccl_reduce_op.h`；严格检查 `hccl_mod` 为 None，未把环境变量中的 HCCL 名称算作成功。
**最小复现**：生产 `gen_jit_op_maker` 解析 HCCL 头文件；`tests/backends/comm/hccl/test_hccl_op_bindings.py`。
**根因**：`python/jittor/build/codegen.py:461` 按单行解析构造函数；`backends/comm/hccl/ops/hccl_reduce_op.h:15` 唯独将参数列表跨行。
**分层**：C4 native/backend 构建缺陷。**处理**：只合并构造声明为一行，参数类型、顺序、默认值和计算不变；补 CPU 生产生成器回归，覆盖全部 4 个 HCCL header。无提交。
**验证**：修复后真实 NPU 编译并建立单 rank HCCL；native `hccl_process_group_size(0)==1`、rank 为 0，实际 materialize 一次 `hccl_all_reduce` 值一致且 fallback 增量 0。最终 DeepSpeed `npu-hccl-v3` 先执行该严格检查，再三步训练；两侧 backend 均为 hccl、构造清单一致。`v2` 实验包装器曾在设置源码路径前导入 torch，意外载入原版 DeepSpeed；修正入口顺序，失败日志保留。不据单 rank 声称多卡通过。

### 4.0d 2026-09-22 DeepSpeed 反向传播访问密集梯度 is_sparse 失败

**现象**：实验模型构造和前向后，在 DeepSpeed `engine.py:_get_gradients_for_reduction` 访问 `grad_data.is_sparse` 报 AttributeError。
```python
import torch
x = torch.tensor([2.], requires_grad=True)
(x * x).sum().backward()
assert x.grad.is_sparse is False
```
**根因**：native Var 为密集存储，但 Tensor frontend 未发布这个只读属性。
**分层**：C2 compat 协议。**处理**：`tensor/method_api.py` 实现密集标志，`tensor/methods.py` 仅在 frontend 发布只读属性；EXACT 声明限定密集 tensor/view/grad，不表示 COO 或稀疏运算支持。无提交。
**验证**：独立二进制 PyTorch，CPU/NPU 各 2 passed/0 skipped；检查实际反向梯度、值/shape/dtype/device、赋值异常、owner/fidelity、native Var 未被修改、fallback 0。历史 `stage0-shim-cpu-v2.log` 失败后，CPU v3 与 NPU 最终三步轨迹通过。

### 4.0e 2026-09-22 early provider 与公开模型注册越过 DeepSpeed 初始化断点

**现象**：CPU bootstrap 后切 NPU 遗留旧 bound methods；内置 builder 依赖 PyTorch ABI。早期 provider 实验越过导入后，`engine._set_client_model` 又因 `self.__dict__.get('_modules')` 为 None 而失败。
**最小复现**：仓外固定 0.17.6 源码 + `stage0_probe.py run`。原始 CPU `v1`/`v2` 日志保留。
**根因**：前者是上游导入期选择/绑定；后者是 DeepSpeed 直接依赖 PyTorch 私有存储布局。Jittor 公共 `add_module` 已有真实注册能力。
**分层**：第三方上游/provider；不向 Jittor 填充伪私有 `_modules`。
**处理**：独立源码副本增加首次 `get_accelerator()` 的 `module:factory` hook，保留类型校验；provider 只提供实有 CPU/ACL 能力，extension builder 明确不兼容，load 明确失败，stream/RNG/memory 不伪造值；模型使用公开 `add_module('module', model)` 注册。两侧同补丁，不改安装包。完整上游 patch SHA256 `0b634b535371eb85d0b1ce1247bcd758831244a563b1dca7b94f3e2f4c91a0a3`；最终 engine SHA256 `3ef4e9cb2e6986651c402c77e6b11fcce8f6c4b9cb7e2226359bc97253609c74`。无提交。
**验证**：断言导入时保存的 memory bound method 所属对象就是当前 provider，避免 CPU 残留。CPU 和 NPU 双侧 37 个张量对拍含全部 4 个参数与每步输入梯度。最终 NPU 使用真实 HCCL，早期 NPU v1 和 CPU 的后端名差异单独保留；CPU 仅作数值证据。原版与当前补丁版在真实 PyTorch NPU 上的 37 个记录张量误差为 0（`upstream-patch-oracle-parity.json`）。provider 是仓外实验包，尚未满足正式 adapter 准入与 nightly 接入，不升级原版库等级。

### 4.0a 2026-09-22 修复 torch.npu 设备接口的占位行为

**现象**：底层 `jt.get_device_count()` 为 2，`torch.npu.device_count()` 却为 0，`is_available()` 为 False；`set_device(1)` 后原生设备仍为 0。
```python
import torch, jittor as jt
print(jt.get_device_count(), torch.npu.device_count())
torch.npu.set_device(1)
print(jt.current_device(), torch.npu.current_device())
```
**根因**：`compat/torch/installers/cuda/bindings.py:266` 将 npu 与不支持的其他设备一起绑定到 `_api_mod_*` 占位函数。**分层**：C2；真实能力已在 native ACL 的设备运行时中。
**处理**：新增 `cuda/npu.py` 作为独立设备协议实现，通过 native 查询/切卡/同步；安装时仅替换 5 个 npu 接口，其他设备命名空间不变。无卡明确失败；保留负序号选择的 no-op 语义。同步会等待全部已触及设备，比 torch_npu 的单设备同步更强，fidelity 明确标为 APPROXIMATE。不补假的流、随机状态或内存统计。无提交。
**验证**：CPU 无 ACL 的拒绝契约和独立 torch_npu 同设备对拍；输入含 int、字符串、torch.device、None、负序号和越界。两张卡实际计算结果完全一致、fallback 增量 0。`sync-55d59e1c/npu-api-npu-v1.log` 为 21 passed/0 skipped；`regression-cpu-final.log` 为 76 passed/0 skipped。边界探针 v4 还验证原生和兼容 current_device 都变成 1。

### 4.0b 2026-09-22 复现公开 accelerator 切换后的 CPU 残留及内置 NPU 依赖

**现象**：CPU 导入 DeepSpeed 后，公开 `set_accelerator(NPU_Accelerator())` 已使 get_accelerator 返回 npu，但 `deepspeed.runtime.utils.torch_memory_reserved.__self__.device_name()` 仍为 cpu。真实 PyTorch 和 shim 都可复现。
**最小复现**：仓外 `sync-55d59e1c/npu-boundary-probe-v4.py`，使用两侧相同的 DeepSpeed 延迟导入源码副本；只调用公开 setter，读取绑定对象用于诊断，未修改其状态。
**根因**：上游 `runtime/utils.py:35–36` 在导入时保存 bound method；setter 只替换当前 accelerator。内置 `ops/op_builder/npu/builder.py` 构造器直接使用 torch_npu.__file__；shim 调用 `create_op_builder('FusedAdamBuilder')` 报 NameError。Stream/get_rng_state 报 AttributeError；memory_reserved 返回 None，真实对照返回整数。
**分层**：上游初始化/第三方 accelerator 协议；原生流/RNG 等缺失按 C4 核实，PyTorch ABI 仍是 C5 边界。
**处理**：将 CPU bootstrap 后直接切换的方案排除为正式验收入口；不补空 torch_npu、不改 private 状态、不据此宣称 NPU 支持。接入方案需保证首次导入就选择正确 provider，并拒绝未实现的 builder。无提交。
**验证**：v3/v4 两侧 JSON 保存在 `sync-55d59e1c/`；v3 为设备修复前，v4 为修复后。诊断脚本退出 0 只表示成功收集上述差异，**不是 DeepSpeed 验收通过**。早期探针 v1/v2 因误写诊断字段报错，日志保留、不计成功。

### 4.0 2026-09-22 修复未同步状态设备并满足模块行数门禁

**现象**：NPU reload 后计数器显示 cuda；core 结构门禁还报告 optimizer_api.py 为 834 行，超过 800 行上限。
**最小复现**：新增 `test_adagrad_step_assignment_and_reload_keep_cpu_placement_before_sync`，以及 `test_canonical_module_line_budgets`。
**根因**：native host-copy 提示不随 update() 的 Python holder 转移，原生 zeros_like 清零还会丢失 placement；单设 frontend placement 请求不足以修复。
**分层**：compat 状态协议和模块职责拆分，未改数学更新。
**处理**：状态恢复和赋值使用公开 CPU tensor factory，并移除无意义的先清零步骤。将状态映射提取至 `compat/torch/optimizer_state.py`，保留 API 绑定入口；optimizer_api.py 缩为 654 行，不放宽门禁；结构测试同步声明新公开 Adagrad 类型与新的状态模块 owner，同时保留旧 API alias 的同一对象断言。无提交。
**验证**：最终 CPU 69 passed/0 skipped；NPU 14 passed/0 skipped，误差见 §2.2，fallback 增量 0。历史失败日志保留；最终 core/structure 另见 §2.1。

### 4.1 2026-09-22 可选 DeepCompile 在普通 import 时加载 FX

**现象**：`import deepspeed` 经 `runtime.engine`、`compile.backend`、`profilers.graph_profile` 报 `ImportError: cannot import name 'Interpreter' from 'torch.fx'`。
**最小复现**：shim 解释器执行 `import deepspeed`。
**根因**：上游 engine.py:121–124 无条件加载可选编译模块；`deepcompile=False` 是默认值，仍无法避免导入。仓库 FX Graph/GraphModule 仅占位，没有真实解释器。
**分层**：上游可选依赖组织；完整 FX 为 C4，DeepCompile ABI/CUDA 边界为 C5。
**处理**：只在仓外复制 DeepSpeed，延迟导入真正可选的模块，未修改公共 compat 编译行为。patch SHA256 `f677dc3b340afe0eb1c89fea47c56a8097a4dfbc35194f2aaa0eb8e1a86eacde`。无提交。
**验证**：补丁版 shim CPU import 成功；原版仍失败。补丁版真 PyTorch NPU 三步训练通过；同补丁的 shim NPU 在加速器选择中因 `torch.mps.current_allocated_memory` 缺失失败。上游优先 `import torch_npu` 选择 NPU，此 shim 环境未成功选择 NPU。不得通过伪造 torch_npu 或把 ACL 通信标为 NCCL 来通过探测。仍没有双边训练结果。

### 4.2 2026-09-22 补充 Adagrad 后发现 NPU 恢复状态设备差异

**现象**：先是 `torch.optim.Adagrad` 缺失；实现后 CPU 13 passed，但 NPU `load_state_dict()` 后 `state[p]['step'].device.type` 为 cuda，期望 cpu。
**最小复现**：`test_adagrad_matches_real_torch_multistep_state_and_resume`。
**根因**：原生缺少 Adagrad；新增实现的计数器恢复在异步复制/placement 处理上不充分。
**分层**：C4 native 优化器；compat 状态协议。
**处理**：数学更新进入 native，compat 负责参数/状态/closure；状态设备修复待复验，无提交。
**验证**：CPU 初轮 13 passed/0 skipped；NPU v2/v3 均 1 failed/12 passed/0 skipped。首次 NPU 测试还修正了测试错误引用 `jt.has_acl`（应读 compiler/runtime 能力）。失败日志保留，不能写为 NPU 通过。

### 4.3 2026-09-22 固定输入建立真实 DeepSpeed 训练基线

**现象**：ARM CPU 上真 PyTorch 也在 SHM 扩展编译失败，缺 x86 头文件 `immintrin.h`。
**根因**：上游 `csrc/cpu/comm/x86_64/shm.h`；不是 shim 的数值问题。
**分层**：上游 CPU 构建环境约束。
**处理**：保留 CPU 组件对拍；原版 DeepSpeed 使用 NPU 建立真正训练基线。探针的模式断言改为底层 `engine.module.training`，符合 DeepSpeed 实际 train/eval 行为。
**验证**：原版 NPU Stage 0 FP32 三步完成，含全部 4 个参数梯度与每步输入梯度。fixture SHA256 `53fa0f95073a5d7fef14ab407058f927aa35ef34a20de201ac1b1a39e503f19b`。只有 oracle，没有声称双边一致。

### 4.4 2026-09-22 GradBucket 公开类型导入及 ARM 构建修复

**现象**：上游更新到 0f0673ab 后 DeepSpeed 仍无法导入 `torch.distributed.GradBucket`。ARM 编译另外暴露 atomic 计数器直接格式化/比较的问题。
**分层**：GradBucket 导入协议 C2；native 构建缺陷。
**处理**：发布同一个不透明 GradBucket 类型，构造和操作显式拒绝，fidelity 为 UNIMPLEMENTED；两个 C++ 文件改为原子 `.load()`，不改变计数语义。无提交。
**验证**：独立 Torch 协议与分布式 owner 回归 17 passed/0 skipped；后续包含在 CPU 55 项回归中。新 CPU/NPU core 均编译成功，不能据此称 DDP bucket 功能已实现。


## 5. 怎么复现

### 5.1 两个解释器

```bash
export LAB=/home/zjx/.local/share/jittor-distributed-npu-20260922
export ORACLE=$LAB/envs/deepspeed-oracle/bin/python
export PY=$LAB/envs/jittor-shim/bin/python
$ORACLE -c "import torch; assert not hasattr(torch, '_torch_compat_install_context'); print(torch.__version__, torch.__file__)"
```

### 5.2 隔离状态

每次实验/并行rank必须独立状态目录。以下只示意一个rank；launcher另设可见NPU、rank、world size、HCCL根信息，禁止在登录节点直接运行占卡测试。

```bash
export RUN_ROOT=$LAB/state/deepspeed-20260925/<experiment>/<runtime>-rank<rank>
mkdir -p "$RUN_ROOT"/{home,jittor-home,tmp,xdg-cache}
export HOME=$RUN_ROOT/home JITTOR_HOME=$RUN_ROOT/jittor-home
export TMPDIR=$RUN_ROOT/tmp XDG_CACHE_HOME=$RUN_ROOT/xdg-cache
export cache_name=deepspeed
export REAL_TORCH_PYTHON=$ORACLE JITTOR_TORCH_SHIM=1
export JITTOR_REQUIRE_REAL_TORCH=1 JITTOR_REQUIRE_DEEPSPEED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JT_BACKEND_FALLBACK=error
```

oracle必须在独立进程移除shim安装/通信变量；不能继承shim入口后拿自身作对照。先在Slurm已分配卡内加载相同CANN，再启动两侧。

### 5.3 命令

历史复现入口如下；新基线尚待复验，不写成可保证全绿。

```bash
# 仓库根目录，CPU只验构造与明确拒绝CPU Engine
JT_BACKEND=cpu JITTOR_TEST_DEVICES=cpu "$PY" -m pytest -q compat/tests/torch/test_deepspeed_l0.py
# 分配的NPU内，独立状态目录
JT_BACKEND=acl JITTOR_TEST_DEVICES=npu "$PY" -m pytest -q compat/tests/torch/test_deepspeed_l0.py
# 已由launcher生成oracle/shim各两rank数据后比较Stage3
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/stage1_compare.py \
  --root "$RUN_ROOT/results/stage3-formal" \
  --out "$RUN_ROOT/results/stage3-formal/comparison.json"
```

完整两rank launcher、provider、可见设备契约见仓库 `agent/skills/deepspeed-torch-compat/SKILL.md`；历史实际脚本保存在 E 下 `multicard/run-stage3-versioned.sh`、`l3/run-qwen3-stage1-memory.sh`、`l3/run-qwen3-stage1-train-liveness.sh`，运行前需把旧实验路径/作业号改为当前已分配资源，保留脚本副本。

模板通用 `verify_repo.py --repo deepspeed` 四轴和 `nox -s ecosystem` 本轮未完整执行，不宣称已经通过；先核实入口实际覆盖范围，再补正式门禁。仓库定期 `nox -s deepspeed_l0` 注册不等于CI执行通过。

### 5.3a 本轮站点启动入口（已分配Slurm作业内）

以下是本轮实际保存的站点脚本；脚本从SLURM_STEP_GPUS读取分配卡号，不手写卡号。用当前有效作业号替换JOB_ID。日志、缓存各rank隔离；这些站点脚本未版本化，不进入主仓库。

```bash
N=/home/zjx/.local/share/jittor-distributed-npu-20260922/state/deepspeed-20260925
bash "$N/run-pair.sh" "$JOB_ID" stage1
bash "$N/run-pair.sh" "$JOB_ID" stage3
bash "$N/run-pair.sh" "$JOB_ID" qwen
```

`qwen`入口只保存一步执行与样本，不自动提供完整训练验收。其抽样比较器归档为N下 `compare_qwen.py`。本轮各阶段代码清单为 `source-before-dtype-sha256.json`、`source-before-emptyfix-sha256.json`、`source-final-sha256.json`；报告中的成功证据必须与对应阶段清单一起使用。

### 5.3b Qwen三步全量复现

站点入口 `N/run-qwen-full-pair-v2.sh` 顺序运行两侧，使用 `run-npu-full-v2.sh` 初始化环境；本轮作业773，重跑须换为有效分配作业和全新输出目录。两个入口为仓外未版本化脚本，原件与报告一起保留。

```bash
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/qwen3_training_compare.py \
  --root "$N/qwen-full-v2" --out "$N/qwen-full-v2/comparison.json" --steps 3
```

比较器只读已有张量，不重新训练。原始数组仍保留服务器仓外，不进入主仓库。报告中shim的`torch_file`为实际Jittor实现源路径，oracle为真实torch路径，两侧身份另有明确断言。

### 5.3c 同权重诊断重放

当前源码工作树为 `deepspeed-sync-7d20b83`。本轮仓外启动脚本为 `N/run-replay-pair-7d20.sh`、`N/run-npu-replay-7d20.sh`，使用已分配作业773；后续运行须重新核实作业、替换输出目录。模型来源和精度配置不变。

版本化入口 `agent/skills/deepspeed-torch-compat/scripts/qwen3_training_replay.py`；设置 `DS_REPLAY_SOURCE` 指向旧全量记录、`DS_REPLAY_OUT` 指向全新目录，再由两rank launcher启动oracle和shim。比较入口：

```bash
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/qwen3_replay_compare.py \
  --replay-root "$N/qwen-replay-7d20-v1" --training-root "$N/qwen-full-v2"
```

此入口用于诊断，不能作为库成熟度门禁。文件和层级覆盖固定为本轮Qwen3配置。

### 5.4 版本与产物

- 两侧固定依赖：einops0.8.2、hjson3.1.0、msgpack1.2.2、ninja1.13.2、numpy1.26.4、packaging26.3、psutil7.2.2、py-cpuinfo9.0.0、pydantic2.13.5、tqdm4.70.1；见 `requirements/deepspeed-l0.txt`。Qwen相关完整包版本以原始运行manifest为准，新基线复验前重新导出两侧清单。
- DeepSpeed 0.17.6原包SHA256：`b3318064ee5798e8a27d201ea8b888f0439973c4eac9af9ab381dd1862ebdf45`。
- Qwen config SHA256：`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`。
- Qwen权重SHA256：`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
- 原始JSON/日志/NPZ/checkpoint位于E及新实验目录，**未版本化，不进主仓库**。本页文档不代替原始证据归档。
- 历史结果按各自55d59e1c、97be8b98、7d20b83a及补丁记录保留；当前复验基线75aa99778df9d2834b333800f1721fc73693f043加保留的未提交修改。无新增提交SHA，commit/push前交负责人审核。

### 5.5 分基线复现入口

#### 当前97b9 Stage 1 eps=1e-8（保留已验入口）

C为R/candidate-97b9。当前正常canonical入口为`C/run-rms-guard-formal-pair.sh`与`run-npu-rms-guard-formal.sh`，执行仓库`qwen3_training_probe.py`；正式比较目录`R/qwen-stage1-eps1e8-97b9-rms-guard-v1`。L0入口`C/run-l0-rms-guard-pair.sh`对应v2目录。L3入口`C/run-l3-rms-guard-with-compare-v2.sh`及比较器`compare-l3-rms-guard-v2.py`，副本来源与仅改eps前置条件见`l3-eps1e8-provenance.json`。恢复入口`C/run-resume-rms-guard.sh`，副本`qwen3_optimizer_resume_eps1e8.py`仅改optimizer eps，见`resume-eps1e8-provenance.json`。L3及oracle比较需与训练一致的固定`E/l3/oracle-site`，缺包启动失败不计通过；新L0–L4已按六份最终JSON验收。生成/采样入口为`C/run-gen-sample-rms-guard.sh`，对应`run-qwen-stage1-gen-97b9-rms-guard-v1.sh`与`run-qwen-stage1-sample-97b9-rms-guard-v1.sh`；严格只读汇总入口`C/collect-rms-guard-current-l4.py`生成`R/rms-guard-current-l4-summary.json`，记录六JSON SHA及143源码一致性，不用准备脚本充通过。

所有站点文件在仓外、未版本化；运行前替换实际分配作业、正确工作树与全新输出目录，不复用旧job或并发缓存。

#### 当前97b9 Stage 2/3 eps=1e-8复现入口

S为`$JITTOR_LAB_ROOT/state/deepspeed-20260926/stage23-rms-guard`；R为对应日期state，C为R/candidate-97b9。全部脚本/report/数组/checkpoint为仓外未版本化产物。

入口S/run-stage{2,3}-l0-pair.sh、run-stage{2,3}-training-pair.sh、run-stage{2,3}-resume-pair.sh、run-stage{2,3}-dependent.sh；dependent串行执行本Stage l3/gen/sample pair。训练Stage2必须使用v2，v1实际Stage1标invalid。Stage3使用专用full-grad/full-param探针，不能只改普通canonical环境变量。只改eps的来源见S/stage{2,3}-eps1e8-provenance.json，launchers冻结见S/launcher-freeze.json。L3/resume副本沿用C的eps1e8 provenance，oracle-site固定依赖与训练一致。

比较入口S/compare-stage{2,3}-l0.py、compare-stage{2,3}-l3.py、compare-stage{2,3}-generation.py；正式训练与恢复复用版本化完整比较器。S/collect-stage{2,3}.py只读严格汇总为R/rms-guard-stage{2,3}-current-l4-summary.json，S/collect-stage123.py生成R/rms-guard-stage123-current-l4-summary.json。先核对真实stage/配置/独立oracle身份/设备/源码SHA，再检查六JSON status/errors及训练gaps，不用脚本存在或单边passed充层级。

复跑前替换实际有效分配作业、正确工作树、全新输出目录与独立rank/runtime缓存；WORLD_SIZE=2，两runtime串行、每runtime两rank同时，首次JIT串行。不要复用旧job/并发缓存；公共文档不写个人home/host/固定卡号。生成为普通重载模型，1256恢复字段和hash范围见§2.3。

#### 历史75aa三Stage eps=1e-6

以下为旧75aa验证树/配置复现，不能作为97b9运行源码或eps=1e-8通过证明。

R内各Stage的`run-qwen-stage{1,2,3}-l0-75aa-v1.sh`、`run-qwen-stage{1,2,3}-formal-75aa-v1.sh`、`run-qwen-stage{1,2,3}-l3-75aa-v1.sh`和`run-qwen-stage{1,2,3}-l4-75aa-v1.sh`记录四清单、正式三步训练、绑定各自训练后权重的L3及gen/sample。Stage 1/3恢复入口为`run-qwen-stage1-resume-default120.sh`、`run-qwen-stage3-resume-default120.sh`；Stage 2先前默认恢复产物保留本节表中路径。这些是仓外未版本化站点脚本，本轮使用已有分配作业；重跑前替换实际有效作业、工作树及全新输出目录，不能直接复用旧作业号或并发复用缓存。

Stage 1/3站点比较器为`compare-l0-stage{1,3}-75aa.py`、`compare-l3-stage{1,3}-75aa.py`、`compare-l4-stage{1,3}-75aa.py gen|sample`；Stage 2沿用`compare-l0-75aa.py`、`compare-l3-75aa.py`、`compare-l4-75aa.py gen|sample`。L3 oracle使用与训练一致的`E/l3/oracle-site`固定依赖目录。训练Stage 1/2使用版本化`qwen3_training_probe_eps1e6.py`，Stage 3使用专用`qwen3_training_probe_stage3.py`，不能仅把前者环境变量改为3。所有日志/report/NPZ/safetensors/checkpoint均未版本化。

设置STAGE为实际1、2或3后，正式训练独立比较入口仅读该Stage已有数组：

```bash
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/qwen3_training_compare.py \
  --root "$R/qwen-stage$STAGE-eps1e6-75aa-formal-v1" \
  --out "$R/qwen-stage$STAGE-eps1e6-75aa-formal-v1/comparison.json" --steps 3
"$ORACLE" agent/skills/deepspeed-torch-compat/scripts/qwen3_resume_compare.py \
  --root "$R/qwen-stage$STAGE-resume-default120-v1" \
  --out "$R/qwen-stage$STAGE-resume-default120-v1/comparison.json" --zero-stage "$STAGE"
```

全证据汇总入口为R内`collect-stage123-current-l4.py`，产物`stage123-current-l4-summary.json`包含18份比较JSON摘要。训练比较不代替L0、L3、生成采样、恢复或性能的各自证据；恢复字段及两类checkpoint_sha256含义见§2.4。

## 6. 遗留问题

未验证范围见2.5；本节记录已复现的遗留缺陷。

| 问题 | 严重度 | 当前绕行/状态 | 退出条件 |
| --- | --- | --- | --- |

| Stage 2 PG1偶发120秒超时 | 中 | 历史75aa各Stage一轮默认预算成功不定位旧停滞；新97b9尚不关闭该风险 | 定位发布/可见性原因，以独立路径/缓存重复验证默认预算并保留失败证据 |
| 原版裸导入依赖不完整FX路径 | 中 | 已文档化显式adapter接入；CPU Engine/ABI边界见2.5 | 完成真实公开导入契约或上游可选依赖修复；公共ABI边界变化按C5统一确认 |

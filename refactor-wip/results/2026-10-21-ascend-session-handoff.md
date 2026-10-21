# 昇腾 910B3 多库适配：新会话交接

状态：2026-10-21 的交接快照。维护者：本轮适配协调者。计算代码基线为 `d93b6959bc44e60320d64d822909fc622681afdb`；包含本文的后续提交仅调整文档，不是新的模型验证。代码、依赖、编译配置或 fixture 改变后，按影响范围复查。

本文供新会话从服务器已有工作区继续，不需要重建环境或重新下载权重。活动状态的唯一来源仍是 [整改看板 ASC-01—ASC-06](../architecture/refactor-board.md)；本文不建立第二份活动看板。原始日志、环境、缓存和 checkpoint 均未版本化，位于仓库同级 lab。

## 1. 目标、基线与接手顺序

用户负责 Jittor 的 Torch 生态兼容层，已有 CUDA 适配，当前目标是迁移并验证 Transformers、Tokenizers、PEFT、Accelerate、safetensors 和 FSDP2 在昇腾上的日常组合场景。优先推理、训练、微调、保存加载、fresh-process 恢复；正确性通过后比较速度和显存。复用已有模型，暂不优先极端场景。

上一轮“执行至北京时间 2026-10-21 12:00 左右”的时间窗口已经结束，不能把旧脚本的截止 epoch 套到新会话。新会话按用户的新指令继续；没有新的时长或目标预算时，不自行创建、恢复或重新设定 goal。

已知仓库状态：

- 分支：`codex/ascend-integrated-compat-20261021`。
- 已提交的计算修复：`d93b6959bc44e60320d64d822909fc622681afdb`，中文提交，126 个明确路径；本轮计算结束时主树干净，未 push。
- 该提交父节点及最初同步目标：`305c2f0866f163b0549d47b2e4264c5d41a86385`。
- remote：`origin`，仓库为 Jittor/jittor；本任务同步目标仅 `origin/wip/cgq-day23-fix-20261021`。不得自行合并 main、master、2.0-refactor 或其他开发分支。
- 本次交接前两次 fetch 分别出现 HTTP/2 错误和 45 秒超时。缓存中的目标远端 ref 仍为 `305c2f...`，本地与缓存 ref 的计数为 1 ahead / 0 behind；**不能据此声称远端当前没有更新**。证据为 `$STATE/handoff/remote-sync.json`。本次保留既有计算基线，只增加交接资料；新会话开始实现前须重新尝试同步，失败时如实保留状态。

新会话先检查 remote、分支、HEAD、dirty state、正在运行的本任务进程，再按仓库规则同步指定目标，保留本地提交，不 reset、不自动 stash。同步期间冻结同树写入与测试，记录整合后的 SHA；旧结果保留原基线身份。

按顺序阅读：

1. [AGENTS.md](../../AGENTS.md)、[协作手册](../../agent/manuals/collaboration.md)、[jittor-dev-context](../../agent/skills/jittor-dev-context/SKILL.md)、[项目上下文索引](../../agent/manuals/project-context.md)。
2. 本文、[整改看板](../architecture/refactor-board.md)、[综合计划](../testing/2026-10-21-ascend-integrated-compat-plan.md)。
3. [完整报告](2026-10-21-ascend-integrated-compat.md)的开头结论与末尾“最终真实 BF16 LoRA 保存与 fresh 恢复闭环”；中间章节为按时间保留的历史，不能把早期 pending/FAIL 当最终状态。
4. [环境规则](../../agent/manuals/environment.md)、[源码架构](../../docs/development/source-architecture.md)、[兼容原则](../architecture/torch-compatibility-principles.md)。
5. 修改或验证下游生态前读 [downstream-library-adaptation](../../agent/skills/downstream-library-adaptation/SKILL.md)；做分布式或性能时再读对应 `agent/skills/jittor-distributed-verification/SKILL.md`、`agent/skills/jittor-transformers-perf/SKILL.md`。

## 2. 工作区约定与私有环境

下文使用这些变量，机器绝对路径及本机入口另存于 `$STATE/handoff/machine-context.json`：

```bash
REPO="$(git rev-parse --show-toplevel)"
export JITTOR_LAB_ROOT="$(dirname "$REPO")/jittor-lab"
STATE="$JITTOR_LAB_ROOT/_state/ascend-compat"
```

环境已经安装，不要在接手时无约束升级、重装或改变锁定版本。计算开始前核验现状；有活跃进程时不得修改其环境、源码或缓存。

| 环境/组件 | 已有配置与用途 |
| --- | --- |
| 主机 | openEuler 22.03 SP4 / aarch64 / GCC 10.3.1；8 张 910B3，每张 64 GiB，驱动 25.5.1。资源分配必须重新查询 |
| 用户基础环境 | 私有 Miniforge/Conda 的 `ascend` 环境，Python 3.11.17，私有 CANN 8.5.0；不改系统驱动/公共 Toolkit |
| `$STATE/envs/dev` | Jittor 开发环境；torch 2.11 元数据是兼容 facade，不是真实原生 Torch；没有 torch_npu |
| `$STATE/envs/oracle-npu-cann` | 独立原生 Torch 2.7.1+cpu / torch_npu 2.7.1.post2；包含 CANN SDK 私有依赖，无 Jittor/shim；当前 NPU 对拍选它 |
| `$STATE/envs/oracle-cpu` | 独立原生 CPU Torch，无 torch_npu；CPU optimizer/checkpoint 对拍必须选它，避免扩展懒初始化 NPU |
| `$STATE/envs/oracle` | 最初的原生参考环境，保留只读；不含后来补齐的 SDK Python 依赖，不能与 oracle-npu-cann 混用 |
| 固定下游依赖 | Transformers 4.56.2、Tokenizers 0.22.2、PEFT 0.17.1、Accelerate 1.10.1、safetensors 0.8.0、NumPy 1.26.4 |

完整环境使用说明见 `$STATE/ecosystem/final-delivery-host-r1/environment-usage.md`；安装集合、METADATA/RECORD 和源码 overlay 的终态核验见同目录 `final-receipt.json`。原生 Torch 的 `+cpu` wheel 后缀不代表无 NPU，NPU 由实际 torch_npu 扩展提供，已实卡验证。

常规 adapter overlay 为 `$STATE/ecosystem/trainer-native-fa-r1/adapter-overlay-r2-manifest.json`，SHA256 `20c9a839d4fa094646d9942439f84cc903d78b56345dd6350b208d76abc04606`。其中 21 个文件保持冻结，13 个 Python 文件与当时最终主树逐字节一致。已有 Transformers/Accelerate hook 的正规 metadata 需可见；不要直接补丁共享 site-packages。

## 3. 已有成果及证据范围

主证据索引为 `$STATE/delivery/evidence-index.json`，包含文件路径和哈希。`$STATE/delivery/commit.json`、`changes.patch` 记录代码提交和补丁。以下 source 数字表示冻结变更文件集合，**不是整个仓库只有这些文件**。

| 场景 | 已验证结论 | 原始证据入口（相对 `$STATE`） |
| --- | --- | --- |
| 最终 MAIN-r28/source123 | 完整 structure 1430 passed / 2 CUDA skips、layout 通过；1432 不同节点与上一版相同；97.py + 1.pyi AST3.7 通过，配置 Ruff 相对 HEAD 新增0/减少12 | `device-api/main-r28-final-cpu-r1/compact-terminal.json`；`ecosystem/final-delivery-host-r1/source123-static/receipt.json` |
| 小模型综合矩阵/source65 等 | BERT/GPT2/Llama/T5/ViT/Whisper 及 LoRA Llama/Qwen2 八场景，完整输出、271 参数梯度和2输入梯度对拍通过；真实 NPU / 零回落 | `ecosystem/final-delivery-host-r1/eco-results-index-final.json` |
| 综合 Trainer/source65 等 | FP32 训练、累积、clip、AdamW、scheduler、评估、safetensors 保存重载及 fresh 恢复通过；连续/恢复完整216状态及轨迹 exact；dropout0、顺序 fixture | 同上，及完整报告中的 r10 章节 |
| Qwen3-0.6B 推理/既有冻结源 | 普通 BF16 三输入完整 logits、cached/uncached greedy 通过；默认 FP32 严格数值仍失败 | 完整报告的 `inference-functional-grid-r2/06b-bfloat16-b1-s128` 章节 |
| Qwen3-1.7B/4B 推理/source121 | 每模型三输入、15完整数组、58,343,424 logits 全部 bitexact；cached/uncached greedy8 exact；不是 source123 重跑 | `ecosystem/larger-models-final-r27-prepared/functional-r27-summary.json` |
| 最终 source123 AMP | actual MAIN 无候选 loader，正式10P/0S/零回落；BF16/FP16 activation、FP32 A/B master 两步 Linear/LoRA 目标84完整数组/2880元素 bitexact | `device-api/acl-matmul-amp-mixed-r1/handoff3-main123-terminal.json` |
| 最终 source123 真实 BF16 LoRA/R6 | Qwen3-0.6B 完整单次工作流 rc0：4updates/8micro、112个 FP32 **参数张量**、448完整梯度及 dtype 标签、评估/预测/生成/adapter 保存重载及合并检查 | `real-model/qwen3-06b-training-npu-r6/bf16-native-deterministic-reference/jittor/full/output/report.json` |
| 最终 source123 fresh 恢复/R7 | 从 R6 真实 checkpoint-2 新进程恢复2步；933完整最终状态和轨迹 strict0；恢复阶段224完整梯度及224 dtype标签 bitexact；恢复载入状态 exact | `real-model/qwen3-06b-training-npu-r7/closeout.json`；其 `bf16-native-deterministic-reference/jittor/compare/result.json` |
| 独立 source88 FSDP2 | 固定双rank FP32 fixture；Native/JT 各 full8、prefix3、fresh-resume5，共6 fresh worlds/12 rank-phase；完整数组按原预算 cross、各 runtime 恢复 exact | `distributed/hccl-training-fsdp2-fp32-r7/acceptance-summary.json` |
| 最终 source123 多卡 | 仅 host 复核；新 NPU world=0/report=0，NOT_RUN_RESOURCE | `distributed/fsdp-checkpoint-placement-owner-r1/closeout-main-r28-source123/final-evidence-index.json` |
| NPU 性能 | 无可接受结论。1.7B 实际18交替calls后中断，正式 generation/decode 未完成；4B未跑；训练控制器仅CPU资格 | `ecosystem/larger-models-final-r27-prepared/handoff3-terminal-r27.json` |

最终计算 manifest：`$STATE/main-source-r28.json`，SHA256 `d3904da402f4a0c0b17d61102ec8bf704c97391f338415d7f56ac567e311510e`。代码提交 d93b6959 的每个声明文件均与该 manifest 核对过。

真实 LoRA 最终 closeout SHA256 `69c34f256d848389f2e751bd3275c2cad072b399355fa2e06bcbc2cbabd9b7b3`；strict0 compare SHA256 `f3b42f3423ac1c5a5caaa8264d172d32f9d8463d3f420b9a1922f04446b22234`。最后计算收尾时本任务无存活进程；这不构成新会话的用卡许可。

## 4. 未完成项与优先顺序

### 优先：真实 LoRA 的原生完整训练对拍

Jittor 自身训练恢复已通过；原生 PyTorch 同模型 BF16 完整训练 cross 仍未完成。新会话可优先在相同 fixture、权重、精度、AdamW/scheduler 和更新协议下补 Native full/prefix/resume，再逐项对拍输出、全部梯度、参数和优化器状态。只有 manifest/工具/输入/依赖/精度完全相同且复用关系明确时，才重用 R6/R7 的不可变 Jittor 结果，否则新 cohort 重跑；不得把小算子84数组通过等同于整模训练 cross 通过。

真实模型协议：`$STATE/real-model/fixtures-r1/protocol.json`，SHA256 `ab11ec4b11b95e23ca72075ea7e9012ca91801adf7197fb7cc39c298bc8f9592`；16 train/4 eval、batch2、accum2、4updates、max_length128、LoRA r4/alpha8/q_proj+v_proj、dropout0、seed821、固定顺序。共享权重通过既有初始化目录加载，不复制大权重。

既有固定预算：FP32 forward atol1e-5/rtol1e-4、gradient atol1e-5/rtol1e-3；BF16 forward .005/.03、gradient .01/.05；同 runtime 恢复 atol=rtol=0。失败后不放宽；完整数组和所有适用梯度必须齐全。

R6 是完整 continuous 工作流；R7 使用 **R6 已实际写出的 checkpoint-2**，不是另跑的 prefix2。既有 compare 的 `--first` 明确支持 continuous-prefix，R7 按该公开 harness 合同恢复并比较。不能把它改述为 JT full/prefix/resume 三个独立训练进程或双方共六进程已完成。

### 数值与功能缺口

- 默认 FP32 Qwen3-0.6B 严格 logits 对拍仍失败；逐层诊断见完整报告中的 RoPE/RMSNorm 章节。部分 pow、cos/sin 和归一化末位差已有证据，不能据此直接认定整个失败均已解释或可忽略。
- 最终 AMP 完整168数组的原 FP32 strict probe 仍 FAIL：164 bitexact；4个 FP32input LoRA input_grad 非bitexact，其中 BF16 两数组45元素超原阈值，maxabs约2.75e-4；FP16对应差异在原预算内。完整记录 `device-api/acl-matmul-amp-mixed-r1/production-main123-full-scope.json`。低精度input84 exact只授予限定资格。
- `torch.utils.checkpoint` 当前直接调用函数，标记 APPROXIMATE，没有激活重算/节省显存语义。**磁盘断点恢复通过不等于激活重计算支持。** 真实 LoRA 基线关闭 gradient checkpointing。
- Jittor 的确定性 setter 未实现、getter False；不能伪装 True。Native CE 的 fresh True-before-first-kernel 诊断通过，旧的 warmup 后切换开关实验不能替代它。双方尚无相同确定性策略下的性能资格。
- 随机 dropout 恢复、长训练任务质量、全参数大模型训练和广泛长上下文未覆盖。短受控 fixture 的闭环不外推这些场景。
- 补充 raw CPU core 四文件/source121 实际46P/9S/1F；唯一失败为 `test_concat_random` 原样100 seeds×1000图操作的压力测试触发300秒超时。失败与原日志保留，不能称 core 全通过，也不要为了全绿删除/缩减测试。
- FP16 GradScaler 已有小规模协议验证，不等于完整 FP16 大模型训练通过；原生 NPU argsort 的整数 key 等未覆盖/不支持范围保持显式边界。

### 最终主树多卡与性能

两张卡均稳定空闲时，按最终同步后的源码重新绑定 manifest，串行预热各 rank 独立 cache，再重跑固定 FP32 FSDP2 全训练/恢复。旧 `main-r26-final-resume-entrypoint.json` 与 source120 脚本已登记为 ARCHIVED_SOURCE120_BOUND_NOT_CURRENT123_QUALIFIED；新增 ACL matmul owner 不在旧 proof closure，不能原样启动旧脚本或只凭旧11+5哈希相等宣称最终资格。

性能只在对应模型/精度/配置的当前源码完整功能准入后运行：相同模型、dtype、attention、batch/seq、同步计时，首轮/JIT与稳态分开，保留预热和至少10对交替测量，分别检查 prefill、generation/cached decode、训练 step。记录 allocator 与 vendor workspace 的口径，不用初始化墙钟或不一致 HBM 口径声称性能优势。此前只完成部分的1.7B性能不是可引用的结论。

## 5. 实现入口与可复用工具

| 主题 | 主要入口 |
| --- | --- |
| NPU设备/AMP前端 | `compat/torch/installers/cuda/api.py`、`bindings.py`、`compat/torch/amp.py`、`grad_scaler.py` |
| 最后修复的混合类型 matmul | `backends/acl/kernels/ops/matmul_op.py`；测试 `compat/tests/torch/test_torch_npu_linear_autocast_master.py`；复用 `python/jittor/nn/functional/matrix.py` 的既有dtype策略 |
| 排序/归约/索引 | `backends/acl/kernels/{native,ops}/` 下 argsort 与 gather_scatter/getitem/setitem；`src/ops/composite/{argsort,arg_reduce,mapped_matmul}_op.cc`；相关 `tests/backends/acl/` |
| 优化器/scheduler | `compat/torch/optim_frontend.py`、`optimizer_api.py`、`lr_scheduler.py` 及定向测试 |
| FSDP2/状态与通信 | `compat/fsdp2/`、`compat/collectives.py`、`backends/comm/hccl/`；分布式 skill 的 `verify_hccl_stack.py`、`verify_hccl_training.py` |
| 生态hook | `adapters/jittor_adapters/transformers.py`，正规 metadata overlay；不要在下游私有函数加整模专用绕路 |
| 模型/训练harness | `compat/tests/torch/_ecosystem_{cases,runner}.py`、`_hf_trainer_*_runner.py`、`_release_hf_workflow_runner.py`、`_release_hf_state.py`、`_release_hf_cross_runtime.py` |
| 真实推理/配对性能 | `agent/skills/jittor-transformers-perf/scripts/benchmark_qwen3_ascend.py`、`compare_qwen3_ascend.py`、`paired_benchmark_control.py`；训练计时 `benchmark_hf_trainer_steps.py`、`_trainer_step_protocol.py` |

机器已有工作树位于 `$JITTOR_LAB_ROOT/worktrees/` 的 ascend-device、ascend-ecosystem、ascend-distributed。先检查其 branch/dirty/进程再复用；这些树包含不同的历史候选，不自动把它们当作最终主树。

## 6. 启动、缓存和资源的必要注意事项

1. 每个计算阶段开始前 fresh `npu-smi info`，核对实际进程、健康和内存，只用当时空闲的卡；旧文档中的任何卡号/交接均已失效。双卡需要两张均满足条件。只清理本任务且 PID/start_ticks 匹配的进程，不读取或修改外部任务环境、不对外部 PID 发信号。
2. 首次 JIT 串行；并行任务隔离 JITTOR_HOME/cache_name，同一缓存不能有并发 writer，benchmark 与单测不并发共用缓存。等待有效编译时先查锁/子进程，不按“长时间没输出”直接杀。
3. 编译配置精确文件为 `$STATE/device-api/acl-argsort-r1/canonical-foundation-build-env.json`，SHA256 `13916f23667df459417edfd677fa26a8aa1c79b14f1a631d7b1566f803838496`。字段“未设置”与空字符串可能影响缓存身份，按 manifest 应用，不手猜 flag；不要改 stamp 冒充新鲜编译。
4. R6/R7 使用 `$STATE/runs/foundation-frontend-npu-r1/jittor-home` 与 `foundation-frontend-npu-r1` cache，已无旧 worker；新的串行运行复用前仍核对配置与 freshness。CPU暖环境捕获于 `$STATE/device-api/cpu-main-r18/environment.json`，CPU检查不能误用NPU构建环境触发大重编译。
5. ECO 曾中断的1.7B性能缓存带“再用前需审计”标记，见 `ecosystem/larger-models-final-r27-prepared/handoff3-terminal-r27.json`；不要当作普通暖 cache 自动复用。空可见设备下曾出现 GetDeviceCount/raw -6，完整 ELF 编译产物不等于 NPU runtime 已验收。
6. source 私有 CANN `set_env.sh` 时不提前启用 `set -u`。Native 保留官方 SDK 的两个 Python 路径，不能一律清空后丢掉 SDK：`python/site-packages` 与 `opp/built-in/op_impl/ai_core/tbe`。Jittor 再叠加当前 TREE/python、TREE/adapters、正规 overlay；Native不能导入shim/Jittor。
7. `set_env.sh` 或 SDK 初次加载会重设 OpenBLAS 线程，已有 R6/R7 entry 先 tiny FP32 SDK warmup，再用 `blas-controls.py` 恢复并核验实际2线程。只设置一次环境变量不足以证明真实线程池已是2。
8. 原生参考使用 `entry.py` 在**首个 NPU 初始化和算子前**设置 deterministic True、HF32关闭、matmul highest；Jittor实际记录 deterministic False/UNIMPLEMENTED，不改报告伪造相同策略。精确入口在 R6 的 `entry.py`，复用前先读。
9. CPU/oracle后处理显式空 CUDA/Ascend masks，使用 oracle-cpu；Jittor CPU完整structure需 Torch模式。HOME/TMP/XDG/HF/编译/厂商缓存均放lab；不要把 TMPDIR 设得过长而超过底层 IPC socket 路径限制。
10. `$STATE/real-model/qwen3-06b-training-npu-r6/driver.py` 与 r7 是已完成历史启动器：输出目录已存在，且包含已过去的截止 epoch。新建 cohort 目录、重新绑定源码和资源，明确复用或重跑的关系，不直接重启原脚本、不覆盖其日志/输出。所有新运行先记录执行命令、源/依赖/fixture哈希和PID/ticks。

## 7. 本次交接与提交约定

本次仅补充交接文档和上下文入口，不新增模型实验，不改变原代码数值、fixture或阈值。原模型证据继续对应 d93b6959 与各自 source manifest。文档引用、布局、source123哈希和必要结构门禁在提交前核验；结果另存 `$STATE/handoff/documentation-checks-r2/`，与原计算结果分开。

完成修改时只暂存当前任务明确文件，禁止 `git add -A`；按仓库要求使用简明中文本地提交，除非用户明确要求，不 push。原始失败和中断记录保留。当前资源和安装状态以后续 fresh 核验为准，不把本交接快照当作长期预留或全栈支持声明。

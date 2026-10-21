# Ascend 910B3 多库综合适配计划

## 目标与边界

本轮在新 Ascend 910B3 主机重建开发环境，验证 Transformers、Tokenizers、PEFT、Accelerate、FSDP2 与 safetensors 组合的日常推理、训练、微调和保存恢复。功能与数值正确性优先，对通过正确性的场景比较独立 PyTorch 的速度与显存。范围为单机 CPU、单卡与双卡 NPU；跨机与极端参数不属于本轮退出条件。

按用户授权执行至计划完成，或北京时间 2026-10-21 12:00 左右。期限到达保留失败、未完成项和可恢复入口，不把时间到期写成适配完成。唯一活动状态源为[整改看板](../architecture/refactor-board.md)的 ASC 系列；本文规定协议，不另维护活动状态。

## 基线、环境与资源

只同步 `origin/wip/cgq-day23-fix-20261021`，初始 HEAD 为 `305c2f0866f163b0549d47b2e4264c5d41a86385`。工作分支为 `codex/ascend-integrated-compat-20261021`，不自动合并其他分支。原机器的 Day2–3 暂停与历史 CPU/CUDA/NPU PASS 保留原边界，不外推到新主机。

主仓库只存源码、测试、文档和可复用工具。`JITTOR_LAB_ROOT` 为同级 jittor-lab；环境、日志、模型配置、checkpoint、临时文件、编译缓存与产物均在 lab。开发和 oracle 环境分离，各并行任务使用独立 `JITTOR_HOME/cache_name`。首次 core/ACL JIT 串行，关闭多进程和并行算子编译；不借用旧机器二进制缓存。

新主机为 ARM64 Kunpeng、8×910B3（每卡64 GiB），驱动25.5.1；用户目录已有 CANN Toolkit/910b ops8.5.0、Python3.11.17、PyTorch2.7.1、torch_npu2.7.1.post2，实卡 smoke 已通过。项目新环境置于 lab，不修改原用户栈。CANN8.5 的头文件/符号核对仅证明前置，真实编译仍需验证签名/ABI，不能用旧 CANN9/Python3.9 报告替代。

开发环境按 `pyproject.toml`、`uv.lock` 和环境手册，Python3.11、NumPy1.26.4；生态双方固定 Transformers4.56.2、Tokenizers0.22.2、safetensors0.8.0、PEFT0.17.1、Accelerate1.10.1。后两项是 lock 外追加层，单独保存解析清单。真实 oracle 固定上述 Torch/NPU 栈，不装 shim，安装下游不得替换 Torch；双方记录 import 来源、版本与 `pip check`。

每次计算前查询 NPU 健康、显存、实际进程和磁盘，只使用当时空闲卡；一次查询不视作长期预留。每卡一个本任务计算进程，双卡任务暂停单卡调度并串行预热 rank cache。限制 CPU/BLAS/编译线程并降低调度优先级，实际资源映射保存在仓外配置。不终止其他用户任务，不改系统驱动、全局包或系统配置。

共享模型只读引用、不复制权重。每阶段记录 lab 用量和文件系统余量；余量不足以容纳下阶段预计产物加两倍空间时停止新产物调度，优先清理本任务已结束、可重建的临时下载和中间件；保留失败证据、checkpoint、活跃缓存。无法下载的模型说明原因暂缓，其余场景继续。

## 对拍与验收合同

两个独立解释器运行同一 fixture，oracle 断言没有 `_torch_compat_install_context`。固定同权重、输入、mask、标签、seed、dtype、attention 实现、batch/sequence、累积、optimizer；常规正确性关闭 dropout，随机恢复显式管理 RNG。参数、输入、输出和梯度均验证实际 NPU placement，scope 内同步，再检查 fallback 尝试计数为零。显式 staging/fetch 不代表模型允许 CPU 回退。

保存完整输出、全部适用参数/输入梯度，检查所有 trainable 梯度和 frozen 参数不变；不以单一 loss、切片或生成文本验收。容差复用已有 case/harness 的固定值，新增协议先保存阈值与依据，失败后不放宽。报告最大绝对/相对差、超差计数与完整性。tiny 配置、pretrained 权重、性能分开声明。

每个运行记录源码 SHA/dirty 指纹、依赖、CANN/驱动、实际执行项数、设备、命令、退出码、日志/JSON 与输入权重指纹。失败原数据不覆盖，修复后新 cohort。CPU 独立基线先建立一次，再按受影响范围复验；Jittor CPU 自比较仅作定位证据。

## 分阶段矩阵

| 阶段 | 多库与模型 | 日常条件及准入 |
| --- | --- | --- |
| ASC-01 环境与构建 | CPU selftest；ACL matmul正反向；独立Torch frontend | 串行新cache；sync_run=1及常规异步；FP32、BF16 AdamW、RMSNorm/SDPA前置；真实placement、零fallback |
| ASC-02 双解释器 | BERT/GPT2/Llama/T5/ViT/Whisper+Transformers；Llama/Qwen2+PEFT | CPU FP32后NPU；同权重完整forward及全部适用梯度；补齐NPU迁移、注册和逐张量设备守卫 |
| ASC-03 综合训练恢复 | fast tokenizer→变长collator→BERT分类/causal LM→Trainer/Accelerate→PEFT→safetensors | 4–10 updates、累积2、clip、AdamW/SGD、scheduler、eval/predict/best reload；FP32先行再BF16/FP16；连续与fresh-process恢复比较model/optimizer/scheduler/RNG/data cursor；adapter保存/加载/合并 |
| ASC-04 真实权重闭环 | Qwen3-0.6B→1.7B→4B；可用时Llama-2-7B | batch1/2、变长padding、短/常见中等上下文；prefill/decode、greedy cache on/off、sampling合同；BF16 LoRA、累积/clip/checkpointing/save/reload；至少一个真实模型完整闭环 |
| ASC-05 双卡 | HCCL primitives→DDP→FSDP2+DeviceMesh/Accelerate | MPI-free启动；FP32后BF16；no_sync累积、完整梯度/参数及单卡oracle、optimizer/scheduler/save/resume；显式核实两rank与实际shard |
| ASC-06 性能稳定性 | 已通过正确性的真实推理/训练模型 | 相同dtype/attention/batch/seq/optimizer；JIT/首步/稳态分开；同步计时、交替至少10次；prefill/decode/train-step分别统计；真实live/reserved峰值，记录共享负载 |

ASC-01 是实际计算前置，ASC-02 通过后推进 ASC-03/04；ASC-05 的只读审计可提前，双卡实测待单卡通过和资源可用。性能必须有对应功能格准入，不以繁忙共享主机墙钟作硬回归门槛。每格都给 PASS/FAIL/SKIP/未运行与证据，模型无法取得不跳过整个后端。

优先复用 `_ecosystem_cases.py`、`_ecosystem_runner.py`、`_ecosystem_harness.py`；综合训练使用 `_hf_trainer_e2e_runner.py`、`_hf_trainer_resume_runner.py`、`_release_hf_workflow_runner.py`、`_release_hf_lifecycle.py` 和 `_training_trajectory_runner.py`。单库测试只作前置与定位。先保存最小复现再修，数学/placement/通信归core/backend，Torch参数/API归compat，第三方私有约定才归adapter。主agent统一看板/整合，写任务使用隔离worktree；首次JIT不并发。

## 初始审计候选缺口

生态runner的NPU to_device目前identity，逐张量placement守卫偏CUDA；Trainer/resume/release工作流CLI与scope仅CPU/CUDA。先最小复现，再扩真正NPU路径，不能只改device标签。性能工具oracle显存pass硬写torch.cuda，修复并核对Jittor ACL统计口径前不能提供NPU峰值对比。

`torch.npu` discovery/device/sync/AMP接口、WORLD all_gather路由、HCCL BF16表与原生AllGather梯度是分布式候选缺口，必须最小执行确定FSDP2调用链。优先用已有native ProcessGroup owner，MPI-free launcher，不为绕过编译扫描引入MPI运行时。默认mesh选择也需核实。

可选AscendC fusion默认关闭，GCC10.3.1形式的标准库头发现存在可移植性缺口；先普通ACL，性能阶段独立验证融合。窄整数reduction、atan2、irfft等已知crash/hang保留相应边界，不整体跳过正常训练。

共享Qwen3及Llama-2资产可覆盖当前固定依赖；Qwen3.5/Qwen3-VL等要求更高Transformers的模型首轮暂缓。BERT/T5/ViT/Whisper/GPT2没有可取得权重时使用tiny/config，明确其证据强度。

## 收口

修复后完成受影响CPU和真实NPU回归；提交前执行布局检查和完整structure，广泛核心变更再跑core，不重复无关全套。dated results列出基线、各格结果、功能边界、性能/显存及证据索引，更新唯一看板。仅按明确清单提交自己的源码/测试/文档，不push。期限到达时停止新增长任务并保存PID/启动时间、缓存、已完成阶段和继续入口；不把未测项标成通过。

## 本轮后段执行约束

在MAIN最终源码冻结后，仅允许独立cache和完整字节镜像的ECO树并行：MAIN使用一张已正式交接的空闲卡验证0.6B BF16 LoRA及中等上下文，ECO使用另一张卡验证1.7B/4B推理。两者在各自实际路径做功能对拍，不将mirror证明替代模型实测，也不共享可写JIT cache。双卡FSDP短回归先独占两卡完成并正式交回，再启动单卡模型队列。

新的真实LoRA原生参考显式在首个NPU初始化或kernel之前设置deterministic algorithms True，以验证原生固定CE诊断发现的可复现条件；Jittor公开接口仍False/UNIMPLEMENTED并如实记录。原生旧False参考的strict0 loss失败继续保留，新协议不改变任何数值预算与same-runtime full/prefix/fresh-resume strict0要求，也不据此给出共同确定性policy的性能比较。公共checkpoint当前没有激活重算，因此真实BF16基线关闭checkpoint，不宣称该接口有显存节省。

约11:20停止启动新的长计算阶段，约11:40为已知本任务worker保留受控结束边界，给最终结构/布局检查、证据整理、中文本地提交及交付留出时间；不终止其他用户进程，不在deadline后以新增任务填满时间。实际启动deadline和hard-stop epoch随队列保存，未启动项按NOT_RUN记录。12点左右按已完成的实际范围收口，未验收项目继续列为未完成。


## 最后正确性窗口（2026-10-21 11:30）

4B/1.7B 最终121源码功能已完成，性能流程结束并归还物理3。最后窗口优先验证真实0.6B BF16 LoRA暴露的ACL AMP混合activation/master路径：独立Native参考与小规模完整梯度/两步更新对拍通过后，才整合最小修复并以actual MAIN验证，再启动真实LoRA完整训练/保存/新进程恢复。该最终功能窗口在11:45停止启动新计算阶段、11:50停止本任务仍在运行的已记录进程，保留约10分钟完成资源核对、证据封存、中文本地提交和交付；北京时间12:00左右的用户期限不变。原样压力测试300秒超时、未完成性能和最终双卡资源不足均保留，不降低数值、strict0恢复或10对性能合同。


11:48补充：R6 continuous full已完整返回0（4updates/8micro、全部梯度、评估生成与adapter生命周期），原R6按11:45新阶段截止保留stop记录。利用同一连续运行已实际保存的step2 checkpoint补一个fresh2 resume，是已准入工作流的恢复收尾，复用现有compare --first对连续prefix的原生支持与strict0，不新增模型/fixture/放宽阈值。仅该恢复阶段硬截止11:55，保留约5分钟结束提交与本地交付；原生Torch大模型全交叉仍未运行。

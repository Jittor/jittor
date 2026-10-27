# 已登记 Swift FULL SFT 的完整安装来源证据

- 状态：下列限定的 CPU/CUDA 场景已验证；达到 L3，观测到零后端 fallback。
- 日期：2026-10-15
- 基线提交（实际测量）：`606b602e7e79b5bfa5573b9693144f8d224c6a07`。
- collector 加固：`f8731d454f95070c018af7ece9068a182f49f359`；它在测量之后独立核对所存 wheel 与完整 payload，不改变张量或其来源身份。
- 共享审计器：`ac4c39d9da541c71e4ff9d083e55c0b69f52a660`（含共享 release-proof 改动 `1ecea7d5879761ec487d2497a25ef07bad47ae17`）。
- 维护者：Jittor 维护者。
- 复查条件：Swift/Transformers/PEFT/Accelerate 版本，模型加载/模板/Trainer 选择，Torch hooks/autograd/optimizer/device 行为，或来源证明与门禁契约变化。

## 结果与覆盖广度

官方 Swift 模型加载器、Llama 模板以及 `SwiftSft.main()` / `Seq2SeqTrainer.train()` 在独立的真实 Torch 与 Jittor 进程中完成三次真实 FULL SFT 更新。观察器只挂接公开 forward hook 和一个 Trainer callback；loss、backward、optimizer 创建与步进均由 Swift/Transformers/Accelerate 负责。没有 adapter，也没有替换训练循环。

干净的 Swift 4.5.2 checkout 为来自 `modelscope/ms-swift` 的 `ff5128777dd3dc5538f1a95fbcc29442c327d7d2`。分母由该确切 checkout 重新生成。下表每一行表示各设备上的一个场景，而不是对某个模型族内每个 API 或生产 checkpoint 的声明。

| 声明的 registry 轴 | CPU | CUDA | 实际场景 |
| --- | --- | --- | --- |
| model_family | 1/222, L3 | 1/222, L3 | `llama`，真实 `LlamaLoader`，完整 tiny Llama |
| trainer | 1/13, L3 | 1/13, L3 | `causal_lm`，已登记 `swift.trainers.Seq2SeqTrainer` |
| task | 1/5, L3 | 1/5, L3 | `causal_lm` |

`FULL` 即官方 `tuner_type='full'` 参数。本 scorecard 对另外 11 项的 `SWIFT_MAPPING` tuner 轴覆盖为 0/11。CLI、RLHF、Megatron、分布式训练、生成与更大模型在此未验证。历史 Swift LORA 场景仍是独立 case 与基线，见[此前的 oracle 报告](2026-10-14-downstream-oracle-scorecards.md)。

## 固定协议与来源

本地 checkpoint 由真实 Transformers 在 CPU 上以 seed 7 创建：一层、width 32、intermediate width 64、四个 query head/两个 KV head、词表 64、128 个位置、不共享 embedding、attention dropout 为零。真实 WordLevel tokenizer 与上游 Llama 模板每批产生 120 个 token。共有 12 个完全可训练参数张量、13,408 个标量参数和一个 rotary buffer；没有冻结参数。

所有运行都从相同的六个 checkpoint/tokenizer 产物开始。训练使用 FP32、batch 为 1、三次更新、学习率 `.001` 的 SGD、常数 scheduler、零 weight decay、无梯度裁剪、无 AMP/checkpointing、无样本乱序。SFT 流水线编码三条实际 message 记录。训练/数据加载 seed 保持 0；输入记录 seed 为 0、7、19。扰动保留全部初始参数/buffer，并改变实际 token 与输出。

`input_ids`、`attention_mask` 与 `labels` 是整数模型输入。它们请求的输入梯度为零。其 dtype、shape、数值与 SHA256 在真实模型调用处观测并精确比较。每步还比较一个额外保留的 embedding 输出梯度；它被标记为中间梯度。整数 token 的量级不进入浮点误差尺度。

所选类、加载器、模板与 trainer 身份取自实际 model/Trainer 对象。Python 调用观测另确认了真实登记的加载器与模板入口，以及 `Seq2SeqTrainer.compute_loss` 与 `Template.compute_sft_loss` 各三次调用。工厂观测包含一次官方选择与一次身份核对调用。

Swift 从由干净固定 checkout 构建的实际 wheel 安装。其 SHA256 为 `ae40b628d3b45e4916d07debdee091b6862ad6fedabf50eb3be526b6d0b7a9ae`。这是源码构建的 wheel，不是对已发布 PyPI wheel 的声明。全部 517 个 Python 源文件与全部 532 个 package payload 文件（含 15 个数据文件）与 checkout、wheel 及实际安装一致。两个 runtime 都在训练前后对已安装包取哈希。collector 独立读取 wheel METADATA 与每个 payload 成员，并拒绝缺失、多余、陈旧或未跟踪的 Python 内容。

Python 为 3.11.16；Transformers 为 4.56.2，PEFT 0.17.1，Accelerate 1.10.1，NumPy 1.26.4。两侧还记录 Swift、Transformers、PEFT 与 Accelerate 的完整已安装文件哈希。本报告中的上游分母与 release-wheel 证明针对 Swift。

| 平台事实 | CPU | CUDA |
| --- | --- | --- |
| 主机 | AMD EPYC 9554 | 同一主机 |
| 加速器 | 显式禁用 | 一张已分配的 RTX 4090 |
| 真实 oracle | Torch 2.11.0+cu128，CPU 张量 | Torch 2.11.0+cu128，CUDA 张量 |
| 候选工具链 | CPU 构建 | CUDA 12.2.140 / cu122 provider，SM89 |
| 驱动 | 不适用 | 580.178.04 |
| 编译器 | GCC 11.5.0 | GCC 11.5.0 / NVCC 12.2.140 |
| 精度 | TF32 关闭；`highest` | TF32/cuDNN TF32/autotuning 关闭；`highest`；严格 native CUDA kernel math |
| native 图标志 | 默认 replay，retain budget 268435456 bytes | 相同 |

oracle 没有 shim 安装上下文，加载真实 Torch `_C` 扩展，其 SHA256 为 `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`。平台产物包含两个解释器的 NVIDIA 发行物清单与实际设备分配。每个被观测的候选参数、输出、loss、梯度与更新后参数都在 host 序列化前同步，并检查物理驻留与 native 分发；真实训练作用域内禁止 fallback。

## 数值与维护门禁证据

每份 runtime trace 含 94 个浮点数组：13 个初始状态数组，加上每步的输出、loss、额外中间梯度、12 个参数梯度与 12 个更新后参数。共有六组配对比较 / 12 个原始 NPZ 文件、564 项字段比较与 1128 个实际浮点数组。所有数组均有限，shape 与 dtype 精确一致。原始验收预算保持为每字段 `max_abs <= .002 OR scaled_relative <= .01`；没有放宽任何容差。

| 设备 | 三个输入 seed 上比较的字段数 | 最大绝对误差 | 最大 scaled relative 误差 | 审计的实际参数更新 |
| --- | --- | --- | --- | --- |
| CPU | 282 | 4.768371582e-7 | 1.146249123e-7 | 108 对；全部非零 |
| CUDA | 282 | 4.768371582e-7 | 1.147065891e-7 | 108 对；全部非零 |

这些测量中每个被请求的参数/中间梯度都存在且非零。两侧最小参考参数更新 L2 范数均为 `4.945786551e-7`。逐参数/逐步范数与更新误差分开存储。CUDA 训练作用域观测到零 fallback。

实际 `nox -s ecosystem_swift` session 运行具体节点 `compat/tests/torch/test_ecosystem_swift.py::EcosystemSwift::test_ms_swift_trainer_full_llama`。CPU：**1 passed, 0 skipped**，139.36 秒；CUDA：**1 passed, 0 skipped**，327.53 秒。每个节点独立重建真实 CPU checkpoint，并对全部三个输入 seed 执行两个 runtime。这些时间包含独立 unit cache 中的首次模型 kernel 编译，不是速度比值或 L4 证据。当时存在外部 GPU 进程；本报告只授予正确性/轨迹证据。

测量之后的来源证明改动新增独立 wheel/payload 审计与负向测试，不改变生产者、模型、数组或数值预算。collector 加固后的工具回归：**60 passed**；共享审计器基线上最终的来源证明/工具合并选择通过 **67 tests**；collector 改动后的完整结构：**1510 passed, 4 hardware/declared skips**。Ruff 与仓库布局检查通过。较早一次结构运行暴露出该纯源码 venv 收到了 compat editable 包却没有安装 native core；在同一私有 venv 中补齐 native core 后修复了两个打包解析检查。其失败日志与成功重跑均保留。没有改动共享环境。标准 CPU seed 路径使用公开的[无设备 CUDA seed 契约](2026-10-15-cuda-seed-without-devices.md)；Swift 的 seed 例程未变。

加入本报告后最终工作树的结构运行记录为 **1506 passed, 4 skipped, 4 failed**。两项失败等待集成方负责的结果索引与生成的 MANIFEST 条目。另两项失败是因为本次调用禁用了 pytest plugin autoload，使其 xdist fixture 子进程缺少 `-n`/`--dist`；显式加载 `xdist.plugin` 后受影响文件完整通过（**10 passed, 0 skipped**）。没有施加源码绕过。两份封存后日志及其哈希保留在 `swift-post-seal-maintenance-v1.json` 中；170 文件数值证据封存未变。集成方必须在中央登记后重跑结构检查。

严格共享 scorecard 接受 **两条 L3 记录、零 rejected、零 blocked**，每设备一条。另一次只读审计从 1128 个原始数组重算全部 564 个字段对，并核对每个被引用产物的哈希；一份 TF32 元数据造假的复制记录在每台设备上都被拒绝。完整覆盖行保留 case、registry 引用、tiny shape 与全部可训练参数名。[共享 release-source 审计](2026-10-15-release-installed-source-proof.md)独立验证同一完整绑定。合成的 incomplete-payload fixture 单独保留：它保留原始数组，但删除已安装文件声明。旧 collector 接受它；加固后的 collector 与共享 release-source 检查器拒绝它。

## 复现与封存产物

使用两个独立解释器与固定依赖；硬件解释器除 compat 源码外还需安装 native core。将产物、模型文件、HOME/TMPDIR、JIT 状态与依赖 target 放在 `$JITTOR_LAB_ROOT` 下。可选加入的门禁要求干净的固定 Swift checkout 与真实安装的 wheel：

```bash
export JITTOR_CI_PYTHON=<jittor-python>
export REAL_TORCH_PYTHON=<real-torch-python>
export JITTOR_SWIFT_UPSTREAM=<clean-ms-swift-checkout>
export JITTOR_SWIFT_WHEEL=<actual-ms_swift-4.5.2-wheel>
JITTOR_TEST_DEVICES=cpu nox -s ecosystem_swift
CUDA_VISIBLE_DEVICES=<allocated-gpu> JT_BUILD_NVCC_PATH=<nvcc> \
  JITTOR_TEST_DEVICES=cuda nox -s ecosystem_swift
```

可复用的 `swift_training_probe.py prepare` 在真实 Torch CPU 中创建本地 checkpoint；`run` 执行真实 Trainer。`collect_swift_scorecard.py` 重新核对全部 seed 对、历史生产者字节、完整 wheel/源码/payload 来源与共享 scorecard schema。其 `--run cpu=<directory> --run cuda=<directory>` 输入中含 `seed{0,7,19}-{torch,jittor}.{npz,json}`。完整 argv/环境记录与确切命令与原始产物一并保存，而不要求本报告写入机器路径。

未版本化产物位于 `$JITTOR_LAB_ROOT/_state/downstream-training-evidence/20261015/`：`formal-606b602e7e79/`、`swift-scorecard-final-v1.json`、wheel/源码证明、实际门禁日志与合成负向 fixture。封存 manifest 为 `swift-sealed-evidence-manifest-v1.json`，170 个文件 / 38361434 字节，SHA256 `e22d32542a3a05ce29ef4834faa5d277e2a6c91f20c8e0c7476b2ec518309a58`。它不含缓存与环境树；数值产物保持其原始 `606b602e` 身份。

最终 scorecard SHA256 为 `70837b67cb082ffdd41284f1de44d1d002075c5b0c5e6456486523a766b615ae`。合成的 incomplete-payload 与平台精度失败作为负向证据保留，不贡献覆盖。

## 主分支整合与独立复核

整合先同步 `origin/2.0-refactor` 至 `8929633bef1e11294c3f0dfbef01bab9c0e015fe`。随后两个维护设备 session 在干净、冻结的整合源码 `fee77cad724f33a19ea858c69639a09defffe5e4` 上运行：**CPU 1 passed, 0 skipped, 149.55 秒；CUDA 1 passed, 0 skipped, 332.18 秒**。每个 session 重建 checkpoint，并在 seed 0/7/19 上对全部三次 Trainer 更新核对两个 runtime。十五个整合文件哈希在两次终端运行间未变。独立 CPU/CUDA unit cache 与新的私有候选环境保留了原生产者的环境与证据。session 时长包含冷编译；这些仍是正确性结果，不含速度声明。

新的主源码门禁日志与持久的 `606b602e` 数值 trace 不同。维护节点成功后会删除其临时数组，因此其通过不被呈现为新的持久数组 scorecard。原始测量保留其历史生产者 SHA。在共享审计器 `8a7bb3873aa3066509f2518dda0669b1ab84dc5a` 下的独立复核重新生成 Swift 分母，验证完整实际 wheel/payload 与全部三个 seed 对，并接受 **两条 L3 记录、零 rejected、零 blocked**。另一次仅用 NumPy 的绝对误差计算检查全部 **564 对**、精确 shape/dtype 与有限性：每台设备最大绝对误差仍为 `4.76837158203125e-7`。全部 **170 个封存文件**在收集前后重新哈希，无变化。可写比较的重新生成使用封存树之外的字节副本。

一份负向副本暴露了共享审计器此前忽略额外扰动元数据的问题：只改变第二个扰动的参考精度，该审计器仍会通过。专用 Swift collector 已检查该阶段。共享修复现在校验每一对额外 `perturb<digits>-comparison`，包括现有的 `perturb2-comparison` 与 `perturb19-comparison` 拼写。声明 `perturb_seed2` 却没有额外比较会被拒绝；历史单扰动协议仍然有效。两个 runtime 的源码/精度/设备绑定、已通过执行与候选零 fallback 检查都适用于额外配对。同一份合成 invalid-precision 副本现在会被拒绝；它不贡献数值覆盖。合并工具回归通过 **135 tests**，无 skip，容差未变。

整合保留了远端 registry 清单中扩充的 package 文件声明与现有 PEFT session。纯源码覆盖审计现在能识别 Swift 的真实类方法节点与其专用 nox session，同时在提供实际证据前保持 `verified_level` 未设置。中央结果索引与 MANIFEST 登记解决了上述两项集成方负责的文档失败。在额外审计器加固之前的整合源码上，core 通过 **322 tests / 42 skips / one existing xfail**，合并结构通过 **1622 tests / six classified skips**。这些 skip 是该验证环境中的两项硬件、两项 declared 与两项 native 发行物未安装检查；它们与必需的 Swift 设备节点无关，后者执行时无 skip。

新的未版本化整合证据位于 `$JITTOR_LAB_ROOT/_state/swift-main-integration-892/`：`clean-source-freeze.json`、`cpu-nox.log`、`cuda-nox.log`、`main-validation.json`、`reaudited-scorecard-fixed.json`、`reaudit-fixed-summary.json`、`absolute-audit.json`，以及单独标注的 `perturb2-gap-reproduction.json` / `perturb2-gap-fixed.json`。初次私有环境准备在任何数值工作之前即失败；该次尝试与成功准备保留在 `provision-history.json`。

定时 ecosystem workflow 目前选择较旧的通用 `ecosystem` session，而不是该专用 session。上述真实维护手动门禁与限定数值证据可用；专用 nightly 准备/选择仍未完成。本报告不升级历史 LORA case、分布式 Swift、未测 trainer/tuner 或整库整合。

额外扰动修复与报告/索引登记后的最终合并结构：**1622 passed, six classified skips**，226.80 秒。仓库布局、生成 MANIFEST 一致性、Ruff 与空白检查通过。这些检查验证整合后的代码与文档；它们不拓宽限定的数值或硬件结论。

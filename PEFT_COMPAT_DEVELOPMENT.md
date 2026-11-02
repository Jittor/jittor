# PEFT torch compat 开发记录

> 本文档用于开发者之间交流 PEFT 在 Jittor torch compat 上的当前适配状态。它是本地协作文档，不随 Jittor 仓库提交；详细实验事实与后续交接以 `/home/cgq/jittor-lab/PEFT/LOCAL_WORK_LOG.md`、`VALIDATION_HISTORY.md` 和 `COVERAGE_PLAN.md` 为准。
> 更新、修改本文档请查看 `/home/cgq/jittor/agent/skills/torch-compat-library-docs/SKILL.md`。

## 一、基本信息

- **第三方库**：Hugging Face PEFT
- **负责人**：cgq
- **目标库版本**：PEFT `0.17.1`
- **配套依赖版本**：Transformers `4.56.2`、Accelerate `1.10.1`
- **PyTorch 对照版本**：`2.6.0+cu124`
- **Jittor 集成分支**：`feature/cgq_transformers`
- **Jittor 适配提交**：`cbab14feeafe3a408ba2bc5a005e0613a616e36c`（“修复 Qwen PEFT 兼容并补齐训练与状态回归”）
- **远程分支**：`origin/cgq_transformers`；上述提交尚未 push
- **当前支持等级**：L3；固定真实 Qwen2.5-0.5B 的 L4 工作流部分完成，尚缺真实模型 sampling；L5 性能未验证
- **已验证范围**：Python 3.11、CPU、NVIDIA CUDA、tiny Qwen2 的 FP32/FP16/BF16、真实 `Qwen/Qwen2.5-0.5B` 的 FP32 LoRA；独立真实 PyTorch oracle
- **最后更新时间**：2026-09-14
- **主要结果报告**：`/home/cgq/jittor/refactor-wip/results/2026-09-14-peft-qwen-compatibility.md`
- **原始实验根目录**：`/home/cgq/jittor-lab/_state/PEFT/`（未版本化）

## 当前收口摘要（2026-09-14）

PEFT 0.17.1 注册 26 个 `PeftType`。上游 PEFT 自身拒绝在 `model_type=qwen2` 上构造 `ADAPTION_PROMPT`，其余 25 个类型均已在固定 tiny Qwen2 上完成 Jittor CPU 与 CUDA 验证；rsLoRA 和 DoRA 是 LoRA 配置变体，另行验证，不计入 26 个类型的分母。CPU 标准协议覆盖构造、确定性前后向、全部预期训练参数的有限梯度、一次公共 SGD 更新、冻结基座不变、adapter 保存、基于同一 base 的 fresh reload 与重载后前向。CUDA 方法矩阵采用固定顺序：在 CPU 构造 base、恢复 fixture、注入 PEFT，再显式迁移模型和输入到 CUDA；这项结果不等于支持显式 CUDA `torch.Generator`。

真实模型范围固定为 `Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987`。FP32 LoRA 在 24 层 q/v projection 上完成三步 AdamW、96/96 可训练张量梯度与更新、冻结参数不变、adapter 重载、继续训练、greedy/beam generation、cache prefill/decode 和 `inputs_embeds`。真实 0.5B 没有运行 FP16/BF16；低精度结论只来自 tiny Qwen2 LoRA，其中 FP16/BF16 的跨 runtime logits 最大误差分别为 `2.62e-4` 和 `2.05e-3`。

adapter safetensors 已完成 PyTorch→Jittor→PyTorch 以及 Jittor→PyTorch 双向互通，输出最大误差 `6.33e-8`。同 checkpoint 的 optimizer continuation 已验证，但 Jittor optimizer `.pt` pickle 会被原生 PyTorch 以 `Invalid magic number` 拒绝，不能把 adapter 权重互通外推为 optimizer pickle 双向互通。真实三步 AdamW 的最终参数最大绝对差约 `3.08e-4`；同梯度 replay 的参数误差为 `3.73e-9`，它证明优化器公式与状态语义，但不能代替真实端到端轨迹。

当前累计支持等级为 L3。固定真实 0.5B 的训练、重载、greedy/beam、cache 和 `inputs_embeds` 已构成 L4 部分证据，但真实模型没有 sampling 证据；top-k/top-p sampling 只在 tiny Qwen LoRA 上验证，不能补齐真实 checkpoint 的累计 L4。当前也不声明量化或 QLoRA、分布式训练、`torch.compile`、NPU/ROCm、PEFT/Transformers 新版本、其他 base architecture、低精度真实 0.5B、L5 性能或广泛 Trainer/Accelerate 生态覆盖。仓库根 `tests/structure` 为 `1331 passed, 2 accelerator skips, 0 failed`；另一个额外的 `compat/tests/structure` 测试树为 `105 passed, 9 failed`，且九项在未修改的 `b67236dc` 基线上同样失败。两者是不同入口，不能混为一次全绿或一次门禁失败。

## 二、适配目标与边界

### 本次覆盖

- PEFT 0.17.1 在独立 Jittor Torch frontend 下的公共导入、配置、Qwen2 adapter 注入与模块生命周期。
- 固定 tiny Qwen2 上 25 个上游适用 `PeftType`，以及 rsLoRA、DoRA 两个 LoRA 变体的 CPU/CUDA 方法闭环。
- LoRA 在 causal LM、sequence classification、token classification、question answering 和 feature extraction wrapper 上的训练与 fresh reload。
- adapter 的 add/set/disable/delete/load、`modules_to_save`、配置及 state-dict API、AutoPeftModel、merge/unmerge/unload、safe merge 与 weighted merge。
- adapter safetensors 跨 runtime 互通、同 checkpoint optimizer continuation，以及固定真实 Qwen2.5-0.5B FP32 LoRA 的训练、greedy/beam、cache 和 embedding 工作流。
- PEFT 所需的共享 Torch 协议，包括 module 表示和 MRO、ModuleDict live view、device/load 语义、稀疏加法、随机数与 Generator、线性代数、factory overload、seed barrier 和 AdamW 默认值。

### 明确不覆盖

- `ADAPTION_PROMPT` 在 Qwen2 上不适用：这是 PEFT 0.17.1 上游构造期的明确拒绝，不计作 Jittor 失败。
- Qwen2 没有 Transformers 4.56.2 的 `AutoModelForSeq2SeqLM` mapping，因此固定 Qwen2 的 Seq2Seq wrapper 不在本轮范围。
- 量化、bitsandbytes 4/8-bit、QLoRA、GPTQ/AWQ、LoftQ、分布式、FSDP/DeepSpeed、张量并行、`torch.compile`、NPU、ROCm 和其他模型家族均待独立验证。
- 真实 Qwen2.5-0.5B 只验证 FP32 eager attention；不声明其 FP16/BF16、autocast、GradScaler、长上下文、性能或显存峰值。
- 显式绑定 CUDA device 的 `torch.Generator` 当前会明确拒绝，不做静默 CPU 仿真；已通过的方法矩阵依赖 CPU 构造后迁移协议。
- PEFT 0.17.1 的 `PeftMixedModel` 不提供公共保存支持，因此 LoRA+LoHa 只声明组合、训练和梯度，不声明 checkpoint。

## 三、已完成内容

### 方法矩阵与任务包装

25 个 Qwen2 上游适用类型为 `LORA`、`ADALORA`、`IA3`、`PROMPT_TUNING`、`MULTITASK_PROMPT_TUNING`、`P_TUNING`、`PREFIX_TUNING`、`CPT`、`LOHA`、`LOKR`、`OFT`、`BOFT`、`POLY`、`LN_TUNING`、`VERA`、`FOURIERFT`、`XLORA`、`HRA`、`VBLORA`、`BONE`、`MISS`、`RANDLORA`、`TRAINABLE_TOKENS`、`SHIRA` 和 `C3A`。它们均有 CPU 与 CUDA 实证；rsLoRA、DoRA 也完成相同标准协议。CPT 的 `prompt_encoder.default.embedding.weight: grad=None` 是两端一致的精确上游例外，测试中显式登记，不允许其他训练梯度缺席。

方法特有协议另有定向检查：AdaLoRA rank allocation；Multitask Prompt Tuning 与 Poly task id；CPT token mask；VBLoRA vector bank 扰动；FourierFT/RandLoRA seeded reconstruction；SHIRA sparse update；XLORA 双专家非零路由、classifier 梯度和 fresh reload。LoRA+LoHa `PeftMixedModel` 将所有乘性 LoHa 因子初始化为非零值，观察到 LoRA-only、LoHa-only 和 joint delta，并得到 12 个有限梯度。

LoRA 任务包装覆盖 `Qwen2ForCausalLM`、`Qwen2ForSequenceClassification`、`Qwen2ForTokenClassification`、`Qwen2ForQuestionAnswering` 和 `Qwen2Model` feature extraction。每项均使用相同 tiny fixture 检查训练参数、梯度、SGD 后冻结不变和 adapter fresh reload，避免从 causal LM 结果外推其他 wrapper。

### Adapter 生命周期与合并

公共生命周期已覆盖 add/set/disable/delete/load adapter、`modules_to_save`、配置与 state-dict API、AutoPeftModel loading、merge/unmerge/unload、safe merge NaN 原子拒绝，以及 linear/cat/SVD weighted merge。CPU/CUDA placement 和 load 后首次 forward 都纳入检查，防止 adapter 参数在 base 与 input 位于 CPU 时残留在 CUDA。

Adapter safetensors 跨 runtime 的 keys、shape、dtype 和 tensor hash 一致，双向重载后的输出误差至多 `6.33e-8`。跨 runtime resumed-step 的梯度误差至多 `2.68e-7`，更新误差至多 `3.73e-9`。这些结果只证明 adapter safetensors 和受控 continuation，不证明任意 Python pickle 或 optimizer `.pt` 互通。

### 真实 Qwen2.5-0.5B FP32 LoRA

公开 checkpoint 固定 revision 和文件哈希，LoRA 使用全部 24 层 q/v projection、rank 2、alpha 4、dropout 0。PyTorch 与 Jittor 均完成三步 AdamW，96/96 可训练张量每步都有有限梯度并发生更新，所有冻结张量保持不变；fresh reload logits 为 exact，随后继续训练仍有完整梯度。推理覆盖 greedy、beam、prefill/decode cache 和 `inputs_embeds`，Jittor cache 长度从 5 增至 6，cached/full next-token logits 最大差 `2.34e-5`，input-id 与 embedding logits exact。

两端最终轨迹各包含 1728 个同名数组：96 个 step-0 参数完全一致，288 个 optimizer step 值按步分别为 1、2、3。三步梯度最大绝对差为 `6.08e-5`、`4.03e-5`、`1.14e-4`；更新后参数最大绝对差为 `3.01e-4`、`3.06e-4`、`3.08e-4`。AdamW 的 `eps=1e-8` 会放大真实训练中近零梯度的差异，因此结论是功能闭环和已量化数值范围，而不是 exact 参数轨迹。

### Torch compat 与原生能力

本轮没有在 PEFT 中维护私有补丁，而是修复 PEFT 实际依赖的共享协议。C2 薄适配包括 module 表示、Linear frontend MRO/执行、ModuleDict live views、CPU/CUDA load placement、factory 与 overload、deepcopy data ownership、随机消费、公共 seed barrier 和 AdamW 默认 `weight_decay=0.01`。C3 组合包括 `torch.linalg.solve(left=False)` 的右解协议与 sparse-to-dense addition，它们复用现有可微 device 运算。C4 原生能力包括精确 CPU Generator/randperm 状态和 Generator 驱动的 uniform value generation。

对应实现提交为 `cbab14fe`，包含共享源码、定向测试、PEFT 方法维护门禁与稳定结果报告。主要回归入口为 `compat/tests/torch/test_peft_qwen_methods.py`、`test_sparse_dense_add.py`、`test_torch_compat_linalg.py`、`test_torch_compat_optim.py`、factory/module/frontend 测试及原生 Generator 和 view-storage 测试。

## 四、问题与处理

### Adapter load 后参数残留在错误设备

- **问题级别**：C2
- **复现条件**：base 与 input 在 CPU，从磁盘 fresh load adapter 后首次 forward；早期路径创建的 LoRA A/B 仍在 CUDA。
- **影响范围**：多种 tuner 的 load/reload，看起来像多个方法分别失败，实际是共享 module/load placement 协议。
- **原因定位**：兼容层 module conversion 和 `_make_cpu_resident(inplace)` 路径没有保持公开 `.to(base.weight.device)` 的设备语义。
- **解决方案**：修正共享 module/load 行为并增加 CPU/CUDA placement 与 fresh reload 回归，不在各 tuner 中追加 `.to()` workaround。
- **当前状态**：已解决。
- **验证证据**：25 个适用类型 CPU/CUDA fresh reload 通过；最终 CUDA 通过项均保持参数对象身份、`placement_backend=1`，同步后实际位于目标 GPU，reload error 为 0。

### Linear、ModuleDict 与 deepcopy 生命周期协议不完整

- **问题级别**：C2
- **复现条件**：PEFT `extra_repr` 检查 initializer、遍历/修改 ModuleDict、Linear MRO 执行，以及 SHIRA deepcopy 后 sparse update。
- **影响范围**：adapter 状态展示、动态模块视图、多个 tuner 的基础 Linear 前向和 SHIRA 数据所有权。
- **原因定位**：initializer 不一定暴露 `__code__`；ModuleDict view 不是 live view；frontend Linear MRO 曾遮蔽基础执行；deepcopy 的 holder/data ownership 不符合所需协议。
- **解决方案**：统一在共享 frontend/module 层修复，并用通用定向测试覆盖，不做 PEFT 类型特判。
- **当前状态**：已解决。
- **验证证据**：生命周期、ModuleDict、frontend identity/MRO 和 SHIRA strict method probe 均通过。

### 稀疏更新切断图

- **问题级别**：C3
- **复现条件**：SHIRA 对冻结 dense holder 执行 sparse update，随后通过 sum/matmul/linear 检查可训练 RHS 梯度。
- **影响范围**：SHIRA 及共享 sparse update 图传播。
- **原因定位**：早期 `dense += sparse` 路径受冻结 holder 状态影响，导致可微 RHS 被切断；单纯把 `__iadd__` 标为支持会产生假通过。
- **解决方案**：撤回未经证明的原地假支持，改用现有可微 device primitive 组合出通过梯度测试的 sparse-to-dense addition；相关 holder/view 生命周期由独立原生回归约束。
- **当前状态**：已解决。
- **验证证据**：`test_sparse_dense_add.py`、SHIRA 完整梯度/step/freeze/save/reload，以及方法联合定向选择。

### 随机初始化与 seed 后延迟物化不一致

- **问题级别**：C2
- **复现条件**：RandLoRA/FourierFT seeded reconstruction、`Tensor.uniform_(generator)`，或 tiny Qwen sampling 中随机权重图在后续 sampling seed 下才物化。
- **影响范围**：adapter 初始化重建、全局 RNG 隔离、top-k/top-p 重复性。
- **原因定位**：公开 Generator overload、随机消费与 seed barrier 协议不完整；pending 随机图跨越公共 seed 边界。
- **解决方案**：补齐 Generator overload、随机消费和公共 seed barrier。显式 CUDA Generator 保持明确拒绝，CUDA 方法采用 CPU 构造后迁移；所依赖的原生随机能力单列为下一项 C4。
- **当前状态**：声明范围已解决；显式 CUDA Generator 暂不支持。
- **验证证据**：FourierFT/RandLoRA standard seeded checkpoint reload、CPU sampling 原始未预物化协议 exact 重复、最终 CUDA rand/multinomial 与 alias 门禁。

### CPU Generator 与 randperm 缺少精确原生能力

- **问题级别**：C4
- **复现条件**：使用固定 CPU Generator 连续执行 seeded randperm 或 Generator 驱动的 uniform 初始化，并比较状态推进、数值和全局 RNG 隔离。
- **影响范围**：RandLoRA、FourierFT 等依赖随机 projection/indices 重建的方法，以及共享 Torch Generator 协议。
- **原因定位**：仅靠兼容层组合无法同时满足精确 Generator state consumption、值生成和 JIT 物化边界。
- **解决方案**：新增原生 `generator_randperm` 与 `generator_uniform` 能力并接入公共 frontend，保留显式 CUDA Generator 的拒绝边界。
- **当前状态**：CPU 声明范围已解决。
- **验证证据**：原生 Generator/randperm 定向测试、FourierFT/RandLoRA seeded reconstruction、全局 RNG 隔离和标准 checkpoint reload。

### OFT advanced indexing 写入的 placement 不一致

- **问题级别**：C2
- **复现条件**：OFT 在 CUDA 构造 skew-symmetric 矩阵时，以 advanced indices 写入 device tensor。
- **影响范围**：OFT CUDA 方法闭环。
- **原因定位**：indices/value 与目标 tensor 的 placement 没有按 Torch 协议对齐。
- **解决方案**：在共享 advanced setitem 路径对齐 placement，并增加定向 CUDA 回归。
- **当前状态**：已解决。
- **验证证据**：`method_cuda_jt_oft_fixed.json`；OFT strict train/save/reload 通过且 reload error 为 0。

### BOFT CUDA solve provider 缺失

- **问题级别**：C3；环境依赖缺失不记框架算法缺陷。
- **复现条件**：BOFT CUDA fallback 使用 `torch.linalg.solve(left=False)`，官方 `fbd_cuda` 可选扩展在两端均编译失败。
- **影响范围**：BOFT block diagonal rotation 的 CUDA 前后向和保存重载。
- **原因定位**：Jittor `numpy_code` GPU provider 环境缺少 CuPy；旧 fixture 还曾漏登记新增 `boft_P`，该次 KeyError 是证据工具过期。
- **解决方案**：实现右解组合语义，并在两环境固定安装官方 `cupy-cuda12x==13.6.0`、`fastrlock==0.8.3` wheel；不要求不可用的 `fbd_cuda`。
- **当前状态**：已解决，依赖固定环境。
- **验证证据**：batched 左右 solve 最大误差 `1.19e-7`、解析梯度有限、原始 `block_diag` GPU 值/梯度及 BOFT 完整闭环通过。wheel 版本和 SHA 见 `_state/PEFT/environment_cupy_20260914/manifest.json`。

### VeRA projection 保存键的上游边界

- **问题级别**：不适用（PEFT 0.17.1 两端共同的上游边界，不是 Jittor 兼容缺陷）。
- **复现条件**：PEFT 0.17.1 默认 `save_projection=True`，先主动扰动 VeRA projection，再保存和 fresh reload。
- **影响范围**：VeRA projection 的完整持久化声明。
- **原因定位**：保存键 `base_model.vera_A/B` 缺少目标侧 `.default` 组件，随后被两端的 upstream `strict=False` load 忽略。
- **解决方案或当前限制**：主验收显式使用 `save_projection=False`，通过共同 `projection_prng_key` 重建；默认 True 仅在 projection 未扰动且同 seed 时观察到 reload，不声明扰动 projection 完整持久化。
- **当前状态**：主矩阵已完成；默认 True 的扰动保存属于已记录上游限制。
- **验证证据**：两侧 `vera_save_projection_false_final.json` 与扰动 projection 诊断。

### AdamW 默认值与真实轨迹数值差

- **问题级别**：默认参数差异为 C2；端到端浮点误差为已量化数值边界。
- **复现条件**：真实 Qwen2.5-0.5B FP32 LoRA，三步 AdamW，默认公开参数。
- **影响范围**：训练参数是否实际更新，以及 Jittor/PyTorch 参数轨迹比较。
- **原因定位**：Jittor 公开 AdamW 默认 weight decay 原为 0，而 PyTorch 为 `0.01`；修正后，接近 `eps=1e-8` 的梯度差会被 AdamW 归一化放大。
- **解决方案或当前限制**：对齐公开默认值；同时报告真实端到端轨迹和同梯度 optimizer replay，不用 replay 覆盖真实差异。
- **当前状态**：默认值已解决；真实端到端结果按测量误差通过功能验收，不声明 exact parity。
- **验证证据**：96/96 参数每步更新；参数最大绝对差三步为 `3.01e-4 / 3.06e-4 / 3.08e-4`，同梯度 replay 为 `3.73e-9`。

### Optimizer pickle 不能反向互通

- **问题级别**：C2 序列化协议边界。
- **复现条件**：原生 PyTorch 直接读取 Jittor 保存的 optimizer `.pt` pickle。
- **影响范围**：跨 runtime 完整 optimizer pickle 恢复。
- **原因定位**：Jittor 保存格式不是原生 PyTorch serialization magic 格式。
- **解决方案或当前限制**：adapter 权重使用已验证的 safetensors；optimizer 仅声明同 checkpoint continuation，不声明 Jittor `.pt`→PyTorch。
- **当前状态**：暂不支持该反向 pickle 路径，失败为 `Invalid magic number`。
- **验证证据**：adapter lifecycle 双向 safetensors 与 optimizer continuation 证据目录。

## 五、兼容性验证

### 验证环境与复现原则

- **运行环境**：两个 Python `3.11.3` 隔离环境分别承载真实 PyTorch oracle 和 Jittor 独立 Torch frontend；两侧 PEFT `0.17.1`、Transformers `4.56.2`、Accelerate `1.10.1`、Safetensors `0.6.2`、Hugging Face Hub `0.36.2`、Tokenizers `0.22.1`、NumPy `1.26.4`、Jinja2 `3.1.6`、MarkupSafe `3.0.3`。oracle 为 PyTorch `2.6.0+cu124`；Jittor/jittor-torch distribution 为 `1.3.11.0`，环境中的 `torch==2.11.0` 仅是 shim metadata，不是 oracle 版本。
- **设备与 dtype**：CPU 与 NVIDIA CUDA；tiny Qwen2 FP32/FP16/BF16，真实 Qwen2.5-0.5B 仅 FP32。
- **CUDA 数学条件**：真实模型 PyTorch 禁用 matmul TF32，Jittor 使用 `JT_CUDA_KERNEL_MATH=strict`；两端使用 eager Qwen2 attention。
- **状态固定**：不能用“相同 seed”代替相同初始化。正式对拍恢复相同 `named_parameters + named_buffers` fixture、固定输入和 plain tensor 状态，再检查 key、shape、dtype、device、finite、grad-None pattern 和训练参数数量。
- **CUDA 方法顺序**：CPU base 构造→fixture restore→PEFT 注入→显式 model/input CUDA 迁移。变更该顺序时必须重新验证，不能从现有结果推断显式 CUDA Generator 可用。
- **BOFT 环境**：额外需要 `_state/PEFT/environment_cupy_20260914/wheels/` 中 manifest 锁定的 CuPy/fastrlock wheel。

### L0：导入与构造

- **状态**：已通过。
- **结果与证据**：官方 PEFT wheel 两侧文件逐字节核验；独立 frontend/oracle 来源、公共 import、tiny LoRA 构造、参数 key/shape/dtype/trainable inventory 和 25 个上游适用类型构造均有证据。
- **已知差异**：`ADAPTION_PROMPT` 在 Qwen2 上由上游拒绝；显式 CUDA Generator 不支持。

### L1：前向语义

- **状态**：已通过声明范围。
- **结果与证据**：25 个类型及 rsLoRA/DoRA 的固定 tiny Qwen2 确定性前向、任务 wrapper、XLORA 双专家路由和真实 Qwen2.5-0.5B logits/cache/embedding 路径通过。
- **已知差异**：低精度只在 tiny LoRA 验证；FP16/BF16 跨 runtime logits 误差分别为 `2.62e-4`、`2.05e-3`。

### L2：训练语义

- **状态**：已通过声明范围。
- **结果与证据**：方法矩阵检查完整有限梯度、公共 SGD step、冻结不变；真实模型三步 AdamW 覆盖 96/96 LoRA 张量并继续训练。CPT 唯一 `grad=None` key 与上游 exact 一致。
- **已知差异**：真实三步 AdamW 参数差最高约 `3.08e-4`，结论不是 exact trajectory parity；未覆盖真实低精度和分布式训练。

### L3：权重与任务

- **状态**：已通过声明范围。
- **结果与证据**：adapter save/fresh reload、AutoPeftModel、state-dict、safetensors 双向互通、任务 wrapper、merge/unmerge/unload 和 weighted merge 均有验证。
- **已知差异**：VeRA 主矩阵使用 `save_projection=False`；默认 True 的扰动 projection 存在两端共同上游键问题。Jittor optimizer `.pt` 不能由原生 PyTorch 读取。

### L4：端到端能力

- **状态**：部分完成，当前累计等级不据此提升到 L4。
- **结果与证据**：固定 Qwen2.5-0.5B revision 完成三步训练、adapter 保存重载、继续训练、greedy/beam、cache prefill/decode 和 `inputs_embeds`；reload logits exact。tiny Qwen LoRA 另有 top-k/top-p sampling 和三步 `[2,41]` score tensor 证据。
- **已知差异**：真实 Qwen2.5-0.5B 未执行 sampling，tiny sampling 不能替代真实 checkpoint 验收；也不覆盖真实低精度、其他 architecture、长训练或广泛 Trainer/Accelerate 工作流。

### L5：性能

- **状态**：待验证。
- **结果与证据**：Jittor 真实模型进程观察分配约 4.83 GiB，但 CUDA memory API 没有返回 peak counter；没有完成同设备、同 dtype、同 shape、固定 warmup/measurement 的吞吐和显存对比。
- **已知差异**：上述进程分配是运行健康信息，不构成峰值显存或性能结论。

### 维护门禁

PEFT 方法源码门禁 `python -m pytest -q compat/tests/torch/test_peft_qwen_methods.py` 为 `2 passed, 0 skipped`；两个 pytest node 实际执行 24 个非 XLORA 类型、rsLoRA/DoRA 和独立 XLORA，共 27 条配置。最终源码指纹下的完整复跑耗时 646.64 秒，主要来自此前未编译的方法 shape 算子。

方法、稀疏与 solve 定向选择为 `10 passed, 0 skipped`；最终 CUDA 兼容选择集为 `23/23`，受影响 structure 子集为 `33/33`，layout 通过。AGENTS 要求的仓库根入口 `python -m pytest -q tests/structure` 为 `1331 passed, 2 accelerator skips, 0 failed`，耗时 220.89 秒。额外 `compat/tests/structure` 为 `105 passed, 9 failed`，九项均在 `b67236dc` 原基线复现，保留为既有问题而非本轮回归。

## 六、开发与复跑方式

### CPU 方法矩阵

在 `/home/cgq/jittor-lab/PEFT` 中执行 `source /home/cgq/jittor-lab/_state/PEFT/core_training_rng_20260913/env_cpu_warm.sh`，覆盖本次运行的独立 runtime root 后，运行 `python /home/cgq/jittor-lab/_state/PEFT/qwen_methods_20260913/scripts/method_probe.py --runtime jittor --device cpu`。维护者日常回归优先运行仓库内 `compat/tests/torch/test_peft_qwen_methods.py`；只有修改方法 fixture 或扩大 PEFT 类型覆盖时，才需要复跑完整实验脚本。

### CUDA 方法矩阵

在 `/home/cgq/jittor-lab/PEFT` 中执行 `source /home/cgq/jittor-lab/_state/PEFT/core_training_rng_cuda_20260913/env_final_cuda_warm.sh`，设置 `PEFT_CUDA_DEVICE`，再运行 `python /home/cgq/jittor-lab/_state/PEFT/qwen_methods_20260913/scripts/method_probe.py --runtime jittor --device cuda`。必须保留“CPU 构造和 fixture restore、PEFT 注入、显式迁移”的顺序，并在结果中检查参数身份、`placement_backend=1`、同步后的实际 device 和 reload error。BOFT 运行前核对 CuPy manifest，不应把缺 provider 的环境错误归为 `block_diag` 或 Qwen reshape 缺陷。

### 真实模型

使用 `/home/cgq/jittor-lab/_state/PEFT/qwen_methods_20260913/scripts/real_qwen_lora.py`、固定 snapshot `Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987` 和 `--oracle-trajectory`。最终 PyTorch/Jittor 结果分别位于 `/home/cgq/jittor-lab/_state/PEFT/qwen_methods_20260913/pytorch/real_qwen_cuda_v5_fixed_steps_and_initial/` 与 `/home/cgq/jittor-lab/_state/PEFT/qwen_methods_20260913/jittor/real_qwen_cuda_v2/`。比较时必须同时检查 1728 个轨迹 key、step-0 初态、optimizer step、逐步梯度、更新后参数和冻结集合；不要只看 loss 或最终 adapter reload。

### 排错顺序

1. 先确认 PEFT/Transformers/Accelerate 版本、runtime 来源、目标 commit、dirty state 和设备，避免把真实 PyTorch 与独立 Jittor frontend 混用。
2. 先用单方法、单 device 的固定 fixture 复现，再扩大到完整矩阵。若多种 tuner 在 reload 首次 forward 同时报 device mismatch，应优先检查共享 module/load placement。
3. 随机方法必须保存和恢复完整 parameters、buffers、indices 或 PRNG key；不能用同 seed 推断初始化相同，也不能让 pending 随机图跨过后续 seed。
4. CUDA 异步错误要在关键步骤后同步定位。BOFT 首先核对 CuPy provider，OFT 首先核对 advanced indices/value placement。
5. 梯度验收逐张量检查 trainable key、finite 和 `grad=None` pattern，冻结参数逐值比较。除 CPT 登记例外外，任何训练梯度缺席均失败。
6. optimizer 数值差要分成真实 end-to-end 轨迹与 same-gradient replay；后者只能缩小根因范围，不能替代前者。
7. structure 结果必须写明测试树。`tests/structure` 与 `compat/tests/structure` 是两个入口，禁止合并计数或用后者基线失败改写前者门禁结论。

## 七、剩余工作

1. 若下一阶段需要 L5，建立同设备、同 dtype、同 shape 的真实 LoRA 训练及 merged/unmerged 推理基准，记录 warmup、测量次数、同步、median/p95、tokens/s、峰值显存和冷 JIT 成本。
2. 按实际业务优先级分别验证真实模型 FP16/BF16/autocast、Trainer+Accelerate 单设备路径；不得从 tiny 低精度或局部 import 外推。
3. 将量化/QLoRA、分布式/FSDP/DeepSpeed、`torch.compile`、NPU/ROCm 和其他模型家族作为独立矩阵，并在确认无法支持时建立明确失败边界。
4. 上游 PEFT 版本变化时重查 `PeftType`、Qwen mapping、VeRA projection 保存键、MixedModel 保存能力和 BOFT optional dependency，不能沿用 0.17.1 分母。
5. 继续保留额外 `compat/tests/structure` 九项的基线证据；只有其根因或基线状态变化时另行处理，不把它们混入 PEFT 功能结论。

## 八、当前结论

在 PEFT 0.17.1、Transformers 4.56.2、Accelerate 1.10.1 和固定 Qwen2/Qwen2.5 范围内，Jittor 独立 Torch frontend 已完成 25 个上游适用 `PeftType` 的 CPU/CUDA 方法闭环、rsLoRA/DoRA 变体验证、adapter 生命周期与 safetensors 双向互通，以及真实 Qwen2.5-0.5B FP32 LoRA 的训练、重载、greedy/beam、cache 和 embedding 工作流。按累计验收规则，当前支持等级为 L3；真实 checkpoint 的 L4 只部分完成，因为 sampling 尚未验证，tiny top-k/top-p 结果不能替代它。其他主要边界是显式 CUDA Generator、VeRA 默认 True 的上游扰动 projection 保存问题、optimizer `.pt` 反向 pickle、真实低精度、量化/分布式/其他后端与 L5 性能；这些边界均不能由现有 tiny 或功能通过结果外推为支持。

# Hugging Face Datasets torch compat 开发记录

> 本文档用于开发者之间交流 Hugging Face Datasets 在 Jittor torch compat 环境中的当前适配状态。它是本地协作文档，不随 Jittor 仓库提交；长期实验记录以 `$JITTOR_LAB_ROOT/datasets/LOCAL_WORK_LOG.md`、`VALIDATION_HISTORY.md` 和 `COVERAGE_PLAN.md` 为准。
>
> 更新本文档前请阅读 `agent/skills/torch-compat-library-docs/SKILL.md`。

## 一、基本信息

- **第三方库**：Hugging Face Datasets。
- **负责人**：cgq；本轮由根代理统筹。
- **目标库版本**：Datasets `3.6.0`。
- **PyTorch 对照版本**：`2.6.0+cu124`。
- **Jittor 版本**：`1.3.11.0` torch shim。
- **Jittor 分支与调查基线**：`feature/cgq_transformers`；文档收口时工作树 HEAD 为 `cbab14feeafe3a408ba2bc5a005e0613a616e36c`，原始内联探针没有单独持久化实验提交字段。
- **主要依赖**：Python `3.11.3`、pyarrow `25.0.1`、fsspec `2025.9.0`、huggingface_hub `0.36.2`、dill `0.3.8`、NumPy `1.26.4`。
- **当前支持等级**：L3（L4 轻量 collator 交接已通过，完整下游交接仍待验证）。
- **目标支持等级**：本轮完成离线核心与固定 revision 数据流程，保留完整 Trainer/模型交接与 L5 为后续阶段。
- **已验证范围**：Linux、Python `3.11.3`、离线主机侧 Arrow/Python 小样本及 PyTorch/Jittor torch shim 的整数、浮点张量格式化；`set_format('torch')` 张量的实际驻留设备未记录，本条只声明数据语义证据。
- **最后更新时间**：2026-09-14。
- **代码变更**：未修改 Jittor 核心、`jittor.compat.torch` 或新增 Datasets adapter；新增可重跑审计脚本 `datasets/audit_core.py`。

## 二、适配目标与边界

### 本次覆盖

- 在真实 PyTorch 与 Jittor torch shim 的独立环境中导入相同版本的 Datasets 及主要依赖，并确认运行时身份。
- 以固定的离线小样本验证 `Dataset.from_dict`、`Features`、`Value`、`Sequence`、行列访问与切片，以及 `map`、`filter`、`shuffle`、`select` 和 `to_dict`。
- 验证 `set_format('torch')` 的张量类型、值和 dtype，并验证 `save_to_disk`/`load_from_disk` 的本地 Arrow 行值与 schema 往返。
- 区分 Datasets 自身的 Arrow/NumPy/Python 路径和实际使用 Torch frontend 的格式化边界，只有观察到真实兼容缺口时才修改公共实现或增加 adapter。

### 明确不覆盖

- 当前结论不外推到 Datasets 其他版本、其他 pyarrow/NumPy/fsspec 组合、所有 Features 类型或所有数据源。
- 尚未验证 Hub 下载与鉴权、远程或流式数据集、`IterableDataset`、多进程 `map`、大表、缓存并发、网络失败和私有数据集权限。
- 尚未验证 Transformers data collator/Trainer 交接、真实训练、CUDA/NPU 上的下游计算或设备迁移。当前 host 数据处理结果不能作为任何加速设备支持证据。
- 当前没有等价性能实验，不声明吞吐、延迟、内存优势或 L5。

## 三、已完成内容

### 环境与运行时身份

- **实现内容**：在 PyTorch oracle 与 Jittor torch shim 环境中安装并锁定 Datasets `3.6.0` 及主要依赖，采用离线模式运行。
- **实现位置**：环境入口为 `$JITTOR_LAB_ROOT/transformers_compat/env_pytorch.sh` 与 `env_jittor.sh`；运行状态隔离在 `$JITTOR_LAB_ROOT/_state/datasets/`。
- **当前状态**：已完成。
- **验证摘要**：两端均可离线导入 `datasets`、`pyarrow`、`fsspec` 和 `dill`；PyTorch oracle 无 shim 标记，Jittor 端 `torch.__version__` 为 `1.3.11.0` 且存在 `_torch_compat_install_context`。

### 离线核心 Dataset 操作

- **实现内容**：使用 4 行固定样本与显式 `Features(Value/Sequence)` 执行 `map`、`filter`、`shuffle(seed=7)`、`select`、切片和 `to_dict`。
- **实现位置**：无需新增实现；该路径由 Datasets 与 Arrow/NumPy/Python 的现有能力完成。
- **当前状态**：已完成。
- **验证摘要**：两端最终结果完全一致，均为 `{'id':[1,3], 'text':['bb','dddd'], 'labels':[[2,3],[]], 'n':[2,4]}`。

### Torch 格式化与本地持久化

- **实现内容**：验证 `set_format('torch')` 后切片的整数、浮点张量表示，并执行本地 `save_to_disk`/`load_from_disk` 往返。
- **实现位置**：无需新增实现；未版本化 Arrow 产物位于 `$JITTOR_LAB_ROOT/_state/datasets/persist_pt/` 与 `persist_jt/`。
- **当前状态**：进行中，已有局部通过证据。
- **验证摘要**：PyTorch 端返回 `torch.Tensor`，Jittor 端返回 torch shim 的 `jt.Var`，值与 dtype 一致；磁盘重载后的行值和 Arrow schema 一致。重载对象会按 Datasets 语义重新计算 fingerprint，因此不把 fingerprint 字符串与保存前对象完全相同设为验收条件。

### 核心 API 审计与固定数据集流程

- **实现内容**：`audit_core.py` 覆盖 DatasetDict、Features/ClassLabel/Sequence、单样本与批量 map、filter、shuffle、select、shard、切片、NumPy/Torch 格式、错误契约及 save/load；另以固定 revision 的 `rotten_tomatoes` 完成 8 行加载、map 和保存重载。
- **当前状态**：已完成 L2 核心与 L3 窄范围流程。
- **验证摘要**：双环境行值、schema、错误类型和持久化结果一致；固定 revision `aa13bc287fa6fcab6daf52f0dfb9994269ffea28` 的 8 行 train split 两端一致。

### Transformers collator 轻量交接

- **实现内容**：本地 Pegasus tokenizer 与 `DataCollatorWithPadding(return_tensors='pt')` 两文本批处理。
- **当前状态**：L4 局部通过。
- **验证摘要**：两端 `input_ids`/`attention_mask` 的 shape `[2,5]`、dtype `torch.int64`、padding 和 token ids 逐值一致；未执行模型前向、CUDA 迁移或 Trainer。

## 四、问题与处理

本轮未发现兼容问题，C1-C5 问题条目均未建立；也没有产生 Jittor 核心、`jittor.compat.torch`、测试或 Datasets adapter 变更。已验证的核心变换和持久化主要由 Datasets、Arrow、NumPy 与 Python 执行；请求 Torch 格式输出时进入现有张量转换路径，并未暴露契约缺口。初始环境缺少 Datasets 和 pyarrow 属于验证依赖前置，补装到两个隔离 venv 后关闭，不归类为兼容实现问题。若后续路径出现失败，须先建立 PyTorch/Jittor 最小复现，再按实际修复边界独立分为 C1-C5。

## 五、兼容性验证

Datasets 并非模型库，本记录沿用 L0-L5 的累计成熟度原则，并将各级验收映射到环境导入、核心数据语义、格式与持久化、真实离线流程、下游交接和性能扩大范围。孤立的高等级证据不会越过尚未完成的前级验收。

### 验证环境

- **操作系统**：Linux。
- **设备**：离线主机侧 Arrow/Python 小样本；`set_format('torch')` 张量的实际驻留设备未记录。环境脚本默认配置不能替代设备断言，本轮没有记录 `jt.flags.use_cuda=0`、tensor placement 或 `is_cuda` 检查，因此不形成设备兼容证据。
- **dtype**：已验证 Torch 格式输出中的整数 `int64` 与浮点张量；精确浮点 dtype 待技术记录补充。
- **后端**：PyTorch `2.6.0+cu124` oracle；Jittor `1.3.11.0` torch shim。
- **依赖版本**：Python `3.11.3`、Datasets `3.6.0`、pyarrow `25.0.1`、fsspec `2025.9.0`、huggingface_hub `0.36.2`、dill `0.3.8`、NumPy `1.26.4`。
- **运行隔离**：在 `$JITTOR_LAB_ROOT/transformers_compat/` 下分别加载 `env_pytorch.sh` 与 `env_jittor.sh`；设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，实验根为 `$JITTOR_LAB_ROOT/_state/datasets/`。
- **测试入口**：以 `$JITTOR_LAB_ROOT/transformers_compat/` 为 cwd，两端加载对应环境脚本后运行内联 Python 探针；原始命令未持久化为脚本或日志。已记录的操作序列为 `Dataset.from_dict`、`map`、`filter`、`shuffle(seed=7)`、`select`、切片、`to_dict`、`set_format('torch')` 和 `save_to_disk`/`load_from_disk`，当前只能复核操作描述与未版本化产物，不能声称存在一条可直接重跑的完整命令。

### L0：环境与导入

- **状态**：已通过。
- **结果与证据**：两端相同 Datasets 和主要依赖版本均可离线导入，且运行时身份断言区分真实 PyTorch 与 Jittor shim。
- **已知差异**：两端 `torch.__version__` 与 Tensor 实现按运行时预期不同。

### L1：离线核心数据结构

- **状态**：已通过。
- **结果与证据**：固定 4 行数据和 seed `7` 的 `Dataset.from_dict`、Features、索引/切片、`map`、`filter`、`shuffle`、`select` 和 `to_dict` 结果完全一致。
- **已知差异**：`DatasetDict`、`ClassLabel`、`Sequence`、`shard` 和错误契约已由核心审计覆盖；更广嵌套/null Features、`cast` 组合及更多本地格式仍保持待验证。

### L2：格式、批处理与持久化

- **状态**：已通过（声明范围内）。
- **结果与证据**：`audit_core.py` 与 `format_persist_probe.py` 覆盖 NumPy/Torch 格式、批量 map、with/set transform、JSON/CSV、缓存/指纹、invalid format、save/load；双端值、dtype、shape、schema 和错误契约一致。
- **已知差异**：Jittor shim 的对象类型与真实 PyTorch 不同但值语义一致；本轮 Jittor probe 明确为 CPU，尚未验证 CUDA。

### L3：真实离线数据集流程

- **状态**：已通过（窄范围）。
- **结果与证据**：固定 revision `rotten_tomatoes@aa13bc287fa6fcab6daf52f0dfb9994269ffea28`，`train[:8]` 加载、字符数 map、save/load 两端均通过，8 行及 Features schema 一致。
- **已知差异**：仅覆盖该公开数据集和 8 行切片，不外推其他 Hub 数据集或流式路径。

### L4：下游交接

- **状态**：进行中（轻量交接通过）。
- **结果与证据**：本地 Pegasus tokenizer + DataCollatorWithPadding 的双环境 batch 字段、shape、dtype、padding 和 token ids 一致；证据位于 `$JITTOR_LAB_ROOT/_state/datasets/l4_collator_{pt,jt}/run.log`。
- **已知差异**：未执行模型前向、Trainer、CUDA 迁移或完整训练数据管线。

### L5：性能与扩大范围

- **状态**：待验证。
- **结果与证据**：尚未执行等价的吞吐、延迟、峰值内存、大表、流式或多进程实验。
- **已知差异**：无结论。

## 六、性能结果

当前没有可报告的等价性能结果。任何后续比较都必须固定 device、dtype、数据规模与 schema、batch size、进程数、缓存状态、warmup、测量次数和统计量，并同时报告吞吐或延迟与峰值内存；在这些条件缺失时不得给出相对 PyTorch 的整体性能结论。

## 七、待办事项

1. **P1，完成 L4**：在 collator 交接基础上补充模型前向或 Trainer 小流程，逐字段比较 batch、loss、padding、dtype 与设备迁移。
2. **P1，扩展离线语义**：继续覆盖更多固定本地/公开数据格式和数据集分片，但保持 revision、缓存和权限条件可复现。
4. **P2，扩大边界**：只有在前级稳定且资源、网络与权限明确时，才推进 Hub/流式、多进程、大表、真实设备和性能验证；各项必须独立记录，不从当前主机侧小样本外推。

## 八、当前结论

Datasets `3.6.0` 在当前 PyTorch `2.6.0+cu124` oracle 与 Jittor `1.3.11.0` torch shim 环境中，已完成 L0 导入、L1 核心数据语义、L2 格式/批处理/持久化和固定 `rotten_tomatoes` revision 的 L3 窄范围流程，当前累计成熟度为 L3。轻量 Transformers data collator 交接已通过，但完整 Trainer/模型流程仍使 L4 保持进行中。未发现 Torch shim 或 Jittor 核心缺口；无 adapter 修复。未覆盖 CUDA、Hub 广泛路径、流式、多进程、完整 Trainer 和性能 L5，当前结论不得外推到这些范围。

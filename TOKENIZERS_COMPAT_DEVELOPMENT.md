# Tokenizers torch compat 开发记录

> 本文档是 Tokenizers 兼容工作的维护者交接记录，不随 Jittor 提交。实验原始证据和可复跑命令位于 `/home/cgq/jittor-lab/tokenizers/` 与其 `_state` 目录。

## 基本信息与当前状态

第三方库为 Hugging Face `tokenizers==0.22.1`，下游 Transformers 固定为 `4.56.2`。对照环境使用 PyTorch `2.6.0+cu124`，集成环境使用 Jittor `1.3.11.0` Torch shim，Python 3.11，基线 `54cd55cc`。L0-L3 已通过，L4 已通过固定 Pegasus/A800 下游交接；L5 目前为 PARTIAL。

## 阶段记录

2026-09-14 的核心 fixture 使用固定 WordLevel 词表，覆盖单条与批量编码、解码、special/unknown/pad tokens、padding、truncation、offsets、word_ids、错误路径和 JSON save/load。两解释器逐字段一致，证据在 `/home/cgq/jittor-lab/_state/tokenizers/20260914_core/`。

L3/L4 复用了固定公开 `google/pegasus-xsum@8d8ffc158a3bee9fbb03afacdfc347c823c5ec8b` 的既有真实 CUDA 报告，覆盖 `PreTrainedTokenizerFast`、BatchEncoding、collator、模型输入、显式 `.to('cuda')` 以及 A800 `cuda:0` 执行。Tokenizer 的 Rust 文本处理保持 CPU，这是预期架构；只有产生张量后的模型路径进入 CUDA。

L5 已完成可控单进程和多进程压力：`run_l3_l5.py` 固定 64x256 批次，CUDA 张量交接吞吐约 PyTorch 67.1M、Jittor 95.5M token/s；`run_l5_mp.py` 固定 2 workers、2 warmup、5 repeats、每 worker 256 条 512-token 文本，CPU tokenizer 吞吐约 PyTorch 2.41--2.52M、Jittor 2.36--2.57M words/s，RSS 约 478MB/491MB。原始结果在 `/home/cgq/jittor-lab/_state/tokenizers/20260914_l3_l5/` 和 `20260914_l5_mp/`。

## 问题归属与边界

审计未发现 C2 核心能力缺失、C3 Torch API 签名缺口或 C4 设备路径缺口。Rust tokenizer 本体不执行 CUDA，不能据此宣称 CUDA tokenizer；CUDA 证据仅适用于 tokenizer 输出张量的后续交接。当前 L5 仍缺大词表文件、长时间 soak、更多 worker 矩阵、峰值显存和跨版本性能，因此不提升为完整 L5。未覆盖 NPU、ROCm、流式多进程长期运行以及 Trainer 训练收敛。

## 后续计划

只有在资源允许且不影响外部 GPU 任务时，才在 `/ssd2/cgq` 生成更大词表和文本集，固定总 token 数、worker 数、warmup/repeats、运行时长，补充 CPU tokenizer 吞吐/RSS/输出 hash 与 CUDA 张量交接显存峰值。Tokenizers 或 Transformers 版本升级、fast tokenizer tensor conversion 改动或新增设备声明时应重新执行 L0-L4，并复查 L5 边界。

## 维护者交接日志

### 2026-09-14：初始化与环境锁定

本主题从 Transformers 兼容工作中单列，原因是 Tokenizers 的 Rust 文本路径、序列化文件和 Torch 张量交接具有独立生命周期。PyTorch 与 Jittor 使用各自隔离环境，版本严格固定，离线变量保持开启，避免 Hub 网络或解释器 site-packages 差异污染对拍。

实验脚本、日志和缓存放在 `/home/cgq/jittor-lab/tokenizers/` 及 `/home/cgq/jittor-lab/_state/tokenizers/`。大文件若后续确有必要，只允许写入 `/ssd2/cgq`。当前没有新建 adapter，也没有修改 Tokenizers 第三方包。

### 2026-09-14：L0-L2 核心契约

`run_core_matrix.py` 使用内置 WordLevel 词表，输入集合固定为 `hello world !`、`hello world` 和 `hello unknown world !`。脚本同时执行单条与 batch encode/decode、padding、truncation、offsets、word_ids、unknown/pad token 和 JSON round-trip。

两侧 JSON 结果逐字段相同。PyTorch 结果 SHA256 为 `178b6eb82dc94c9a492d4412d2f12b0ee2f238c8411bd1d9490c2cfa5bceb7a0`，Jittor 结果 SHA256 为 `beef42ec908b9db997f77a0619cdd178826994b3cfadeb04bffd2cc1c336ff23`。该阶段证明文本和序列化语义，不证明任何设备执行。

### 2026-09-14：L3-L4 Pegasus 交接

固定公开 revision 为 `google/pegasus-xsum@8d8ffc158a3bee9fbb03afacdfc347c823c5ec8b`。既有 seq2seq L4 报告已重新核对 fast tokenizer、原始 CPU BatchEncoding、padding、collator 和模型输入；随后显式迁移张量到 A800 `cuda:0`，完成真实模型 forward、loss、cache 和 generation。

这条证据的关键边界是 tokenizer Rust 处理仍在 CPU，这是上游库预期架构。CUDA 结论只覆盖 tokenizer 输出张量进入模型后的路径，不能改写为“CUDA tokenizer”。该事实已在 L4 交接和后续复查条件中固定。

### 2026-09-14：L5 单进程与多进程

`run_l3_l5.py` 固定 64 条文本、最大长度 256，warmup 3 次、测量 10 次，验证张量交接的 shape、device 和吞吐。两侧均得到 `cuda:0` 张量，PyTorch 约 67.1M token/s，Jittor 约 95.5M token/s；该数字只代表后续张量操作，不代表 Rust tokenizer 在 GPU 上运行。

`run_l5_mp.py` 固定 2 workers、2 次 warmup、5 次 repeats，每 worker 处理 256 条约 512-token 文本。CPU tokenizer 吞吐为 PyTorch 2.41--2.52M words/s、Jittor 2.36--2.57M words/s，RSS 约 478MB 与 491MB，输出长度集合一致。Jittor 启动诊断写入日志前缀，最终 JSON 行才是机器可读结果。

L5 仍是 PARTIAL。当前没有大词表文件大小统计、长时间 soak、4 worker 或更高并发矩阵、峰值显存采样、跨版本性能回归，因此不能把两次小规模 benchmark 外推为生产吞吐保证。

## C1-C5 问题归属

本轮未发现 C1（第三方库自身行为）之外需要规避的兼容错误。Tokenizers 的 CPU Rust 实现和 special-token 行为均按上游版本工作。

未发现 C2：Jittor 核心不存在因 Tokenizers 核心能力缺失而无法运行的证据。未发现 C3：Torch shim 的 encode/decode、BatchEncoding 和 `return_tensors='pt'` 签名在覆盖范围内一致。未发现 C4：真实 CUDA 交接张量已在 A800 上执行，且 CPU tokenizer 与设备张量边界清楚。

C5 仅作为未覆盖证据记录，不是已证实缺陷：大词表、超长文本、多进程长时间 soak 和跨版本矩阵仍没有足够数据。若后续发现真实设备后端或 Rust ABI 的不可行性，应单独建立 C5 记录并给出复现命令、受影响版本和退出条件；当前没有这样的已证实事件。

## 复跑说明

核心复跑命令为 `source transformers_compat/env_pytorch.sh && python tokenizers/run_core_matrix.py` 以及对应的 Jittor 环境命令。L5 单进程命令使用 `TOKENIZER_FILE` 指向 Pegasus runtime tokenizer；多进程命令同样要求该环境变量并固定 worker 参数。

运行前必须确认 `tokenizers==0.22.1`、`transformers==4.56.2`、PyTorch oracle 或 Jittor shim 身份，以及独立的 `HOME`、`HF_HOME`、`JITTOR_HOME` 和 `TMPDIR`。任何未满足版本、缓存或离线条件的结果都只能标记为环境探针，不能覆盖本记录的 PASS。

## 已验证与未验证边界

已验证内容包括纯文本核心 API、JSON 序列化、fast tokenizer CPU batch encoding、padding/truncation、collator 交接、Pegasus 固定 checkpoint 下游模型输入和 A800 CUDA 张量执行，以及固定参数的 2-worker CPU 压力。

未验证内容包括 NPU、ROCm、流式 tokenizer、多进程长时间稳定性、超大词表内存上限、峰值 CUDA 显存、Trainer 多步训练、量化 tokenizer、Transformers 5.x 和 tokenizers 未来版本。没有这些证据时，当前累计等级仍是 L4 PASS、L5 PARTIAL。

## 复查触发条件

当 Tokenizers 或 Transformers 版本变化、Rust tokenizer JSON schema 变化、Transformers `BatchEncoding` tensor conversion 改动、Jittor shim 的 device placement 逻辑变化，或新增 NPU/ROCm/CUDA tokenizer 声明时，必须重跑 L0-L4。若要提升 L5，需要在 `/ssd2/cgq` 生成可审计的大词表与文本 manifest，固定总 token 数、worker 数、warmup、repeats、soak 时长、RSS/显存采样和输出 hash，并保存两侧原始日志。

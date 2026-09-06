# Transformers 4.56.2 文本核心 17 模型严格累计 L4 验证

- 状态：PASS（本文列出的 17 个文本 modeling implementation 达到严格累计 L4）
- 验收日期：2026-09-06（Asia/Shanghai）
- 代码基线：`508b0118416cdcbebbba9507350bdfeef738e205` 加本文冻结清单中的兼容修复
- 模型运行冻结清单：`$JITTOR_LAB_ROOT/transformers_compat/final_source_snapshot_20260906.sha256`
- 模型运行清单 SHA256：`9e5098dd3fbc7c4a793fd7a2211215a807c3f6aa47b7244e6428a7b9d8d30a0b`
- 交付清单：`$JITTOR_LAB_ROOT/transformers_compat/delivery_source_snapshot_20260906.sha256`
- 交付清单 SHA256：`590375841c49974e65a9e74ebb1e7f4b258d8eb6755e3ab96eb224b645999f9c`
- 环境：Transformers 4.56.2、真实 PyTorch 2.6.0+cu124、Jittor 1.3.11.0 torch API、NVIDIA A800 CUDA、FP32

## 结论与范围

Encoder-only 4/4、Decoder-only 10/10、Encoder-decoder 3/3 共 17/17 个独立文本
modeling implementation 完成累计 L0-L4。L1/L2 使用 tiny 模型在 CPU 和真实 CUDA
分别对拍完整前向、task loss、全部实际参与计算的梯度、train/eval、dropout、两种
`zero_grad`、重复 backward、AdamW 参数更新和 optimizer state；L3 覆盖完整 state、
safetensors、`save_pretrained`/`from_pretrained` 与 Jittor 导出后真实 PyTorch CUDA
round-trip；L4 使用固定 revision 的公开 checkpoint 和真实 tokenizer，并对 13 个生成
implementation 继续覆盖 prefill/decode、cache 增长和 reorder、cached-vs-full、greedy、
beam、EOS early-stop 以及 temperature/top-k/top-p/combined sampling。

这里的“严格累计 L4”只属于下表 17 个文本 implementation 和上述 FP32 正确性协议，
不是整个 Transformers 项目的兼容等级。25 个核心实现中的 ViT、Swin、ConvNeXt、
Whisper、Wav2Vec2、CLIP、BLIP-2、Qwen2.5-VL 尚未完成本轮门禁；L5 性能、FP16/BF16、
长上下文、大 batch、Trainer/Accelerate/PEFT/LoRA、分布式、ROCm/NPU 和 Transformers
5.x 也不在结论内。公开大 checkpoint 负责 L4 端到端推理，L2 完整训练生命周期由同一
implementation 的 tiny 配置提供，本文不声称对全部公开大模型执行完整 backward。

## 固定公开 Checkpoint 与结果

| 类别 | implementation | checkpoint@revision | 公开 CUDA 最坏 forward | cache | generation |
| --- | --- | --- | ---: | ---: | ---: |
| Encoder | BERT | `google/bert_uncased_L-2_H-128_A-2@30b0a37c...` | `1.156696e-6` | - | - |
| Encoder | RoBERTa | `distilbert/distilroberta-base@fb53ab88...` | `1.612555e-6` | - | - |
| Encoder | DeBERTa-v2 | `microsoft/deberta-v3-xsmall@4b419818...` | `6.599122e-7` | - | - |
| Encoder | MPNet | `microsoft/mpnet-base@6996ce1e...` | `1.163153e-6` | - | - |
| Decoder | GPT-2 | `distilbert/distilgpt2@2290a626...` | `2.290552e-6` | `1.842316e-6` | `2.802764e-6` |
| Decoder | GPT-NeoX | `EleutherAI/pythia-14m@cf967c0a...` | `9.275405e-5` | `5.083595e-5` | `8.058838e-4` |
| Decoder | Llama | `HuggingFaceTB/SmolLM2-135M@93efa2f0...` | `2.496043e-6` | `2.501934e-6` | `1.172211e-6` |
| Decoder | Qwen2 | `Qwen/Qwen2.5-0.5B@060db649...` | `2.341671e-6` | `6.597519e-6` | `4.580396e-6` |
| Decoder | Qwen3 | `Qwen/Qwen3-0.6B@c1899de2...` | `2.818694e-6` | `2.564074e-6` | `3.833151e-6` |
| Decoder | Mistral | `M4-ai/TinyMistral-248M-v3@5afbc96d...` | `3.245860e-7` | `4.721113e-7` | `3.577033e-7` |
| Decoder | Gemma2 | `shibatch/tinygemma1m@f96527d9...` (`hf`) | `1.103387e-6` | `6.278194e-7` | `4.104412e-6` |
| Decoder | Phi-3 | `Vasanth/phi3-tinystories-pretrain@935abc8d...` | `3.481323e-7` | `3.214434e-7` | `3.708461e-7` |
| Decoder | Falcon | `tiiuae/falcon-rw-1b@e4b9872b...` | `6.531007e-6` | `1.173055e-6` | `3.343675e-6` |
| Decoder | Mixtral | `shibatch/tinymoe2m@5ffb36a7...` (`hf`) | `8.210108e-7` | `9.153222e-7` | `1.236796e-6` |
| Seq2Seq | T5 | `google-t5/t5-small@df1b051c...` | `2.950256e-7` | `5.286621e-7` | `3.807150e-7` |
| Seq2Seq | BART | `facebook/bart-base@374b7507...` | `5.685612e-7` | `7.487241e-7` | `5.564651e-7` |
| Seq2Seq | Pegasus | `google/pegasus-xsum@8d8ffc15...` | `2.231854e-6` | `1.339275e-5` | `1.805278e-6` |

所有公开模型都重新核对 tokenizer、原始 CPU `BatchEncoding` 和显式 CUDA 移动；所有
Jittor 导出都由 fresh 真实 PyTorch 2.6 CUDA 进程读取。加载后的 missing、unexpected、
mismatched key 必须为空或与同 model class 的真实 PyTorch 固定白名单完全一致，完整
state digest 和 round-trip 输出通过。Falcon 的公开导出因体积使用两片 sharded
safetensors 和 index；其余导出形式按各模型证据记录。

## 生命周期、梯度与 MoE

四个 Encoder 在 CPU/CUDA 的实际梯度分别为 BERT 41/41、RoBERTa 41/41、DeBERTa-v2
50/50、MPNet 41/41。十个 Decoder 的 20 个 tiny CPU/CUDA comparison 全部通过，最坏
CUDA forward、全梯度、AdamW 更新和 optimizer-state 相对误差分别为 `4.5798e-7`、
`6.3862e-7`、`8.5971e-6` 和 `9.8720e-8`。T5 与 BART 分别为 26/26 和 49/49 梯度；
Pegasus 为全部实际参与计算参数 47/47，另有两个 `embed_positions.weight` 在真实
PyTorch 4.56.2 中同样 `requires_grad=True` 但与 forward 图断开，比较器要求两端缺失
集合精确一致，不能把它记作 49/49。

Mixtral 的独立 router/aux CPU 和 CUDA 门禁均通过。两端
`causal_lm_loss=4.6081738472`、`aux_loss=2.0`、`total_loss=4.6281738281`，满足

$$
L_{\mathrm{total}}=L_{\mathrm{causal}}+0.01L_{\mathrm{aux}}.
$$

其 29/29 参数梯度存在，2/2 gate 各 64/64 元素非零，12/12 expert 矩阵各
2048/2048 元素非零。cache/generation 关闭 `output_router_logits`，因为该选项与 cached
decode 同开时的 router mask 错误已在真实 PyTorch 4.56.2 复现，属于上游组合限制。

## Generation 与明确白名单

13 个生成 implementation 都验证 cache 结构和增长、cache reorder、cached-vs-full、
greedy、beam、left/right padding、EOS early-stop，以及四种 sampling 变换。greedy、beam
和 EOS 序列逐 token exact；sampling 不要求不同随机数实现抽出相同 token，只要求各后端
固定 seed 重放、token 合法、filter keep mask 与概率变换对齐，并额外使用无 tie 的固定
logits 检查 Top-P keep mask exact。

Decoder 的 left-padding 比较只排除样本开头 `visible_key_count=0` 的 inactive query 行；
right-padding 全张量、任一 active token、loss 和梯度均不豁免。十个架构都有真实
PyTorch eager-vs-SDPA 区域证据；Falcon 还在真实 PyTorch CUDA 上复现 inactive 行差异
而 active/right 区域为零。GPT-NeoX 是唯一公开 top-p cutoff 边界例外：每个 batch 的
保留数一致，仅交换 tie 边界 token 510 与 19728，概率相对误差 `8.0588e-4`、最大 TV
`4.9303e-4`，均低于独立 `2e-3` 边界阈值；temperature、top-k、combined 以及固定无 tie
Top-P 门禁仍 exact。该白名单不泛化到其他模型。

## 本轮兼容修复

本轮问题均先以真实 PyTorch 与目标 device 的最小复现确认，再修改共享 torch compat：

1. 修正全局 CUDA 开启时 `torch.tensor`、`as_tensor`、`from_numpy` 从 Python/NumPy
   默认错误落 CUDA 的行为，使其保持 PyTorch 默认 CPU 语义；从既有 tensor 构造时保留
   source device，显式 `device` 仍负责迁移。
2. 修正 CPU Var 在全局 CUDA 下执行 clone/cast/detach/host export 时因 lazy materialize
   迁移源 Var 的问题；在 source device scope 中完成 copy/cast 并保留 residency。
3. 修正 `empty_like` 和其他 like/input factory 在 meta context 中被错误覆盖 device/dtype
   的问题；无显式参数时继承输入的 dtype 和真实/semantic device。
4. 为 Jittor 没有真实 meta storage 的边界增加持续的 semantic meta placeholder，使
   `torch.device("meta")`、`.to("meta")`、Transformers/Accelerate
   `init_empty_weights` 的 Linear/Embedding 和 tied 参数在 context 退出后仍走正确加载路径。
5. 实现 `load_state_dict(assign=True)` 的 holder 替换语义，保留目标
   `requires_grad`、Parameter/Buffer、persistent 标志和 leaf registry，同时不修改调用者
   state tensor；tied duplicate key 与 ParameterList 路径有定向覆盖。
6. 修正 fallback `empty_like` 未继承输入 device 的问题，并把 sort/create CUDA 测试改为
   显式设备请求，避免以全局标志替代真实 residency 证明。

Jittor 仍没有真实 meta storage，meta context stack 也不是 thread-local；tensor-overload
`torch.normal` 与 typed Parameter subclass 的全部元数据行为不在本轮完成范围。这些边界
不能用本文结果外推为完整 PyTorch API 兼容。

## 回归与证据

冻结后的仓库定向回归结果为：HF CUDA device `8/8`、serialization `31/31`、dtype
`25/27`（2 项既有条件 skip）、sort/create `11/11`、HF alias `3/3`。提交前又执行
`git diff --check`、九文件 `py_compile`、交付源码清单 9/9 校验、仓库布局门禁和隔离
CPU structure 门禁。排版等价后的最终源码再次合并运行五个定向模块，在真实 A800 CUDA
上为 `Ran 80 tests, OK (skipped=2)`；布局检查通过，structure 为
`232 passed, 2 skipped in 269.87s`。

structure 首轮指出 `installers/nn.py` 为 2622 行，超过既有 2600 行预算。最终没有放宽
门槛，只压缩本轮新增代码的空行、注释和多行表达式；修改前后完整 Python AST 摘要均为
`25018eb18203cf856c6269c73f02140bac70dbd6d55b847a5eb7d38290c9ead7`，其余八个冻结文件
哈希不变。模型运行的固定 revision 清单和数值证据继续保留，交付清单固定排版等价后的最终源码；等价
证明位于 `$JITTOR_LAB_ROOT/transformers_compat/source_layout_equivalence_20260906.md`。

可复现的维护者报告与机器结果位于：

- `$JITTOR_LAB_ROOT/transformers_compat/encoder_l4_20260906/ENCODER_L4_FINAL_REPORT.md`
- `$JITTOR_LAB_ROOT/transformers_compat/decoder_l4_20260906/DECODER_ONLY_L4_FINAL_REPORT.md`
- `$JITTOR_LAB_ROOT/transformers_compat/seq2seq_l4_20260906/frozen_source/SEQ2SEQ_L4_FROZEN_SOURCE_RESULT.md`
- `$JITTOR_LAB_ROOT/transformers_compat/decoder_l4_20260906/final_evidence_audit.json`

验证完成时曾保留固定 checkpoint、tokenizer 和导出模型。维护者随后授权清理可重建的大
权重：10 个 decoder Hub cache root 和 33 个公开 checkpoint/export/runtime 权重文件已删除，
实际回收 `34,009,661,440` bytes（`31.674 GiB`）。固定 model ID/revision 清单、tokenizer/
config 元数据、NPZ/JSON、comparison、日志、报告和 tiny round-trip 权重继续保留；完整 L4
复跑前需按清单重新下载公开 checkpoint 并重新生成导出权重。清理明细位于
`$JITTOR_LAB_ROOT/transformers_compat/weight_cleanup_20260906.md`。状态更新后的布局门禁通过，
隔离 CPU structure 为 `232 passed, 2 skipped in 68.61s`。

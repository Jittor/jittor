# Transformers 4.56.2 refactor 分支 17 模型 L4 复验

- 状态：PASS（17/17 public checkpoint comparison）
- 验收日期：2026-09-11（Asia/Shanghai）
- 代码基线：`integrate/cgq-transformers-2.0-refactor`，HEAD `e12d2d9094a52fcb69b999983d79ce90e07e14a4`；验证时包含当前工作树未提交的兼容修复
- 对照分支：`origin/2.0-refactor` 的 refactor 架构与 `feature/cgq_transformers` 已验证功能合并结果
- 环境：Transformers `4.56.2`、真实 PyTorch `2.6.0+cu124`、Jittor `1.3.11.0`、NVIDIA A800、CUDA 12.4、FP32
- 实验状态：`$JITTOR_LAB_ROOT/_state/transformers_l4_refactor_20260911`

## 结论

当前工作树在真实 CUDA 上重新完成 4 个 encoder、10 个 decoder 和 3 个 encoder-decoder
共 17 个固定 public checkpoint 的 L4 对拍，最终比较文件全部 `pass: true`。对拍使用真实
PyTorch 进程生成参考，Jittor 进程使用独立 `torch.Tensor` frontend；参数、输入、输出和
梯度都检查了真实 CUDA placement，没有 CPU fallback。L4 public 阶段覆盖固定 revision
权重加载、tokenizer、显式 device、padding、forward、cache、cached-vs-full、greedy、beam、
EOS early-stop、temperature/top-k/top-p/combined sampling；encoder 另外覆盖 Jittor 导出
到 fresh PyTorch 的 round-trip。

该结论只适用于下表 17 个文本 modeling implementation、Transformers 4.56.2、A800 CUDA
和当前 FP32 协议，不外推到其它 Transformers 架构、L5 性能、低精度、长上下文、Trainer/
Accelerate/PEFT、分布式或其它后端。

## Public 结果

| 类别 | implementation | public forward 最坏相对误差 | cache 最坏相对误差 | generation 最坏相对误差 |
| --- | --- | ---: | ---: | ---: |
| Encoder | BERT | `1.156696e-6` | - | - |
| Encoder | RoBERTa | `1.612555e-6` | - | - |
| Encoder | DeBERTa-v2 | `6.599122e-7` | - | - |
| Encoder | MPNet | `1.163153e-6` | - | - |
| Decoder | GPT-2 | `2.290552e-6` | `1.842316e-6` | `2.802764e-6` |
| Decoder | GPT-NeoX | `9.275405e-5` | `5.083595e-5` | `8.058838e-4` |
| Decoder | Llama | `2.496043e-6` | `2.501934e-6` | `1.172211e-6` |
| Decoder | Qwen2 | `2.341671e-6` | `6.597519e-6` | `4.580396e-6` |
| Decoder | Qwen3 | `2.818694e-6` | `2.564074e-6` | `3.833151e-6` |
| Decoder | Mistral | `3.245860e-7` | `4.721113e-7` | `3.577033e-7` |
| Decoder | Gemma2 | `1.103387e-6` | `6.278194e-7` | `4.104412e-6` |
| Decoder | Phi-3 | `3.481323e-7` | `3.214434e-7` | `3.708461e-7` |
| Decoder | Falcon | `6.531007e-6` | `1.173055e-6` | `3.343675e-6` |
| Decoder | Mixtral | `8.210108e-7` | `9.153222e-7` | `1.236796e-6` |
| Seq2Seq | T5 | `2.165229e-7` | `7.300083e-7` | `3.265333e-7` |
| Seq2Seq | BART | `5.685612e-7` | `7.487241e-7` | `5.564651e-7` |
| Seq2Seq | Pegasus | `2.231854e-6` | `1.339275e-5` | `1.805278e-6` |

Encoder comparison files are under `results/encoder_current/{bert,roberta,deberta_v2,mpnet}/`;
decoder files are under `results/public_decoder/`; seq2seq files are under
`results/public_seq2seq/{t5,bart,pegasus}/`. The comparison contract also requires exact
tokenizer IDs, state/metadata compatibility, and zero non-whitelisted missing or unexpected
keys. All 17 checks satisfy those conditions. GPT-NeoX has one documented top-p cutoff tie where
two equal-boundary tokens exchange order; the retained count is identical and the dedicated
boundary threshold accepts it. This is not a general sampling failure.

## Compatibility changes exercised

The public matrix exercises the following refactor-compatible fixes: reflected Python scalars are
created on the tensor's actual CPU/CUDA placement; `Tensor.type_as` copies both dtype and device;
input random factories inherit their first tensor's device; ordinary constructors default to
explicit CPU when no device is requested; integral `torch.arange` defaults to `int64` while float
bounds retain the configured floating dtype; `torch.load(map_location="meta")` keeps real CPU
storage with a semantic meta marker; tied aliases remain visible to
`named_parameters(remove_duplicate=False)`; `zero_grad(set_to_none=False)` handles mixed
compatibility gradients; and safetensors reads only the requested payload range after parsing its
header.

## GPT-NeoX CPU numerical boundary

The tiny lifecycle comparison has one deterministic CPU AdamW boundary. For
`gpt_neox.layers.0.attention.query_key_value.bias[167]`, Jittor's gradient is approximately
`-7.2759576e-11` while PyTorch's is `1.5279511e-10`; the resulting first-step parameter delta
difference is `2.2250933562e-05` against a `2e-05` tiny threshold. Three repeated CPU runs produce
the same values. The corresponding CUDA difference is `1.1859491567e-05`, and the public
GPT-NeoX CUDA comparison passes. The discrepancy is a float32 cancellation residual in a
near-zero reduction gradient, amplified by AdamW's epsilon; no semantic clamp or relaxed gate was
introduced. A future fix must start with a minimal CPU reduction/broadcast-backward reproduction
and prove that ordinary non-zero gradients and performance are unchanged.

## Regression evidence

- `compat/tests/torch/test_torch_compat_ops.py`: 31/31 passed, including CPU/CUDA `type_as` device/dtype parity.
- `compat/tests/torch/test_torch_compat_serialize.py`: 34/34 passed.
- `compat/tests/torch/test_torch_hf_cuda_device.py`: 11/11 passed on the real A800 CUDA path,
  including explicit-device round trips, meta/tied-weight loading, T5 and Mixtral cache/generation,
  and CPU-default data constructors with integral `arange` dtype parity.
- A combined 88-case ops/serialization/optimizer invocation produced 73 passed and 15 failures;
  all failures are in the old optimizer test's direct use of native `jittor.Var.backward`,
  native `.grad`, or NumPy-backed `.data.normal_()` paths. The current independent frontend
  contract intentionally keeps `torch.Tensor` separate from `jt.Var`; a direct
  `torch.tensor(..., requires_grad=True)` backward and optimizer step pass. Those stale test
  assumptions are tracked in `LOCAL_WORK_LOG.md` and are not part of the 17-model public L4
  result.

## Reproduction and review

Use the isolated environment and model harness recorded in
`$JITTOR_LAB_ROOT/transformers_compat/DEVELOPMENT_WORKFLOW.md` and
`LOCAL_WORK_LOG.md`. Re-run public comparisons with a new `JITTOR_HOME`/`cache_name` when source
or compiler behavior changes. Review this report when Transformers changes its cache/generation
contract, the independent Torch frontend boundary changes, or CPU reduction accumulation changes.

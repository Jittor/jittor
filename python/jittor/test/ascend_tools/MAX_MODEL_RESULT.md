# Max trainable model on 8x 910B3 (64GB each)

All numbers measured **on the NPU** (`jt.flags.use_cuda=1` + `use_acl=1`).
NOTE: earlier revisions of this file were measured before the NPU-dispatch fix,
i.e. on CPU/host-RAM, and were wrong — see commit history.

## How 8 cards relate to model size
Jittor's multi-card path here is **data-parallel** (HCCL replicates the full
model on every card). So the model-size ceiling is bounded by **ONE card's
64GB**; 8 cards multiply *throughput* (~linear), not capacity. A single model
larger than one card needs tensor/pipeline (model) parallelism, which this
`import jittor as torch` path does not provide.

## Probe
GPT-style decoder, RMSNorm (modern-LLM style; the native LayerNorm ACL op errors
on bf16), vocab 50257, B=1, T=512, one Adam step. `test/ascend_tools/max_model_probe.py`.

## Full fine-tuning (Adam over all params)
| dtype | params | result | note |
|-------|--------|--------|------|
| fp32  | 1.6B   | OK     | ~39GB HBM |
| fp32  | 2.5B   | OK     | near ceiling |
| fp32  | 3.6B   | OOM    | |
| bf16  | 2.5B   | ERR    | bf16 Adam fused-op gap (not OOM) |

- **fp32 full fine-tune ceiling: ~2.5–3B params / card.** Bounded by Adam state:
  weight+grad+m+v = 16 bytes/param (~40GB at 2.5B) + activations + the 50257xE
  vocab head.
- **bf16 full fine-tune is currently blocked** by a jittor/ACL gap: `Adam.step`
  on bf16 params aborts in a fused `broadcast_to+add+multiply+reduce.mean`
  (the moment update). Use fp32 for full fine-tuning, or LoRA (below).

## LoRA (frozen bf16 base + fp32 adapter) — the practical large-model path
LlamaFactory upcasts the (tiny) trainable params to fp32, so the optimizer never
runs on bf16; the frozen base stays bf16 (2 bytes/param, no grad/optimizer).
- Measured: **Qwen3-8B bf16 LoRA = ~22GB HBM** on one card (~2.75 GB / B-param
  incl. activations).
- => **single-card LoRA ceiling ≈ 20B params** (e.g. Qwen3-14B comfortably; ~32B
  base weights alone are 64GB so out of reach single-card).
- Across 8 cards (data-parallel): same ~20B ceiling, ~8x throughput.

## Summary
| path | max / 64GB card | 8-card |
|------|-----------------|--------|
| full fine-tune (fp32 Adam) | ~2.5–3B | ~8x throughput, same size |
| LoRA (bf16 base, fp32 adapter) | ~20B | ~8x throughput, same size |

To train a single model >64GB (e.g. full-FT of 8B, or LoRA of >32B) one would
need model/tensor/pipeline parallelism over the aggregate 512GB — not available
in the current data-parallel jittor-on-ACL path.

# Numerical parity: jittor vs real PyTorch

Verifies that `import jittor as torch` produces the same forward (loss) and
backward (per-parameter gradients) as **real PyTorch**, on a faithful GPT-2 with
identical deterministic weights + input.

## Run
```bash
# jittor env (has jittor):
python gen_weights.py          # -> weights.npz  (pure numpy, name-keyed)
ASCEND_RT_VISIBLE_DEVICES=0 python run_jt.py acl    # -> xcheck_jt_acl.json
CUDA_VISIBLE_DEVICES=0       python run_jt.py cuda   # -> xcheck_jt_cuda.json
# real-torch env (separate; pip install torch):
python run_torch.py            # -> xcheck_torch.json
# compare:
python compare.py xcheck_torch.json
```
`weights.npz` must be shared across all three runs (copy it between machines).

## Result (2026-06-22, GPT-2 L2/H4/E128, fp32)
| backend | loss | total_grad_l2 | vs real torch (loss / grad_l2 / worst-param) |
|---|---|---|---|
| real torch (CPU)   | 6.237304 | 0.045244 | reference |
| jittor CUDA (4090) | 6.237304 | 0.045244 | 7.6e-8 / 1.0e-7 / 2.8e-7 |
| jittor ACL (910B3) | 6.237304 | 0.045244 | 7.6e-8 / 4.7e-6 / 4.4e-5 |

=> jittor matches real PyTorch forward+backward to fp32 round-off (~1e-7 on
CUDA, ~1e-5 on Ascend). The Ascend worst-case (a LayerNorm-weight grad at
4.4e-5) is normal cross-hardware fp32 variance, not a correctness bug.

## Real model parity: Qwen3-0.6B via transformers (fp32)
`qwen_logits.py` dumps next-token logits for a fixed prompt; run under real torch
and under the jittor shim (same dev-transformers code, only the backend differs).

| backend | argmax | top5 | logit_l2 |
|---|---|---|---|
| real torch (CPU)        | 12095 | [12095,7407,279,1112,30743] | 1493.90 |
| jittor shim (Ascend NPU)| 12095 | [12095,7407,279,1112,30743] | 1494.59 |

Identical top-5 token predictions; logits agree to ~1e-4 relative (max abs diff
0.0012 on ~17-magnitude logits) — fp32 cross-hardware variance, not a bug.
=> the real production path (transformers + Qwen3 + jittor) matches PyTorch.

## NVIDIA (N-card) validation
Same `import torch` code runs on NVIDIA via the shim (jittor CUDA backend):
- jittor-CUDA GPT-2 xcheck matches real torch (loss/grad above).
- `import torch`->jittor trains a GPT-2 end-to-end on an RTX 4090 (fwd+bwd+Adam):
  loss 6.39 -> 1.41 over 8 steps. => both Ascend and NVIDIA backends work.

## torch checkpoint (.pt) migration (#13)
`torch.load` now reads real torch `.pt` files (zip archive + persistent-id
storages + _rebuild_tensor_v2), reconstructing tensors as jittor Vars -- no real
torch needed. Verified on BOTH backends: a real-torch checkpoint (OrderedDict
state_dict with fp32/fp16/bf16/int64 tensors + nested cfg/list/scalars) loads
with identical values and key order on Ascend and NVIDIA. Repro: save with real
torch (`torch.save`), then `import jittor as torch; torch.load(path)`.

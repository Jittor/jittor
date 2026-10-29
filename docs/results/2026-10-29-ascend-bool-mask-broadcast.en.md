# Expanding a boolean attention mask to score size on Ascend

- Status: verified on Ascend 910B3 + CANN, including before/after.
- Date: 2026-10-29
- Baseline commit: `008cae211`; the fix is `21e7dfca2`.
- Verification scope: Ascend 910B3 (CANN 9.0.0), one card; Qwen3-0.6B bf16
  prefill at 2048 tokens; the same workload against torch_npu 2.10.0.
- Maintainer: Jittor core maintainers
- Review conditions: the bool branch of
  `backends/acl/kernels/ops/flashattention_op.py`, aclnn FlashAttentionScore's
  attenMask shape contract, or a CANN major version changing.

## Symptom

`benchmarks/torch_compat/run.py --device npu --size full` measured
`qwen3_prefill` at 54.99 ms a step with 55.07 ms of device time (device-bound),
while PyTorch's same workload took 77.41 ms and is almost entirely host time.
The device profile (msprof) showed 28 tensor-sized operators a step -- one per
layer:

| Operator | Calls/step | Time/step | Per call |
| --- | ---: | ---: | ---: |
| Cast | 28 | 11.36 ms | 406 us |
| BroadcastTo | 28 | 5.79 ms | 207 us |
| LogicalNot | 28 | 2.31 ms | 83 us |

The operators doing the work, `FlashAttentionScore` (9.82 ms) and `MatMulV2`
(8.20 ms), account for far less, so 19.5 ms of the step was not attention.

## Root cause

`scaled_dot_product_attention_acl` handled a boolean mask as
`reshape -> broadcast([b, n, s, s]) -> logical_not`:

- Torch's boolean mask is `[1, 1, s, s]` with True meaning "keep", while CANN's
  attenMask marks positions to drop, so the inversion is required;
- but expanding to `[b, n, s, s]` is 64 MiB at 2048 tokens and 16 heads, once
  per layer;
- on the ACL backend that expansion also brought two score-sized casts with it
  (a controlled call produced exactly 2 Casts + 1 BroadcastTo + 1 LogicalNot,
  all scaling with the score matrix).

A controlled comparison (same provider, bf16, causal boolean mask, mean of
three calls):

| Input | Extra operators |
| --- | --- |
| `[1,1,S,S]`, expanded inside the provider (before) | Cast x2 + BroadcastTo + LogicalNot |
| `[1,16,S,S]`, expanded by the caller | LogicalNot only |
| `[1,1,S,S]` handed straight to `FlashAttentionACL` | **none** |

So aclnn already broadcasts the mask over batch and heads; the expansion was
redundant.

## Fix

The bool branch inverts the small mask and expands it only along a query or key
axis that still needs it. The float "pse" slot keeps its broadcast, because
aclnn makes no such promise there.

## Results

Mask semantics against the float64 reference on a 910B3 (bf16 quantization
level):

| Case | max abs diff |
| --- | ---: |
| `[1,1,S,S]`, batch 1 | 0.0074 |
| `[1,1,S,S]`, batch 2 | 0.0075 |
| `[B,1,S,S]`, batch 2 | 0.0074 |
| `[1,1,1,S]` (key axis expanded, old path) | 0.0040 |

`qwen3_prefill` (same card, same cache):

| Metric | Before | After |
| --- | ---: | ---: |
| Wall clock per step | 54.99 ms | **35.69 ms** |
| Device per step | 55.07 ms | **34.64 ms** |
| Against PyTorch (77.41 ms) | 0.710 | **0.461** |
| Cast per step | 11.36 ms | 0.013 ms |
| BroadcastTo (mask) per step | 5.79 ms | 0 (the rest are 10-us operators, 1.43 ms together) |
| LogicalNot per step | 2.31 ms | 0.310 ms |

The device side is now below PyTorch's 38.94 ms; PyTorch's wall clock is still
host-bound.

## Reproduction

```bash
# On the 910B3, Jittor side
python benchmarks/torch_compat/run.py --device npu --gpu 6 --size full \
  --runtimes jittor --workloads qwen3_prefill
# Reference
python benchmarks/torch_compat/run.py --device npu --gpu 6 --size full \
  --runtimes torch --workloads qwen3_prefill
```

The device operator table comes from msprof, or from `jt.profile(device=True)`
where CUPTI is available.

## Regression coverage

- Full ACL suite: 1 failed / 440 passed / 11 skipped; the only failure is
  `test_acl_indexing.py::test_acl_rfft_keeps_lazy_dft_constants_alive`
  (`unary/imag` unregistered, KI-COMPLEX-001), unrelated to this change.
- `tests/structure/backends/acl/`: 295 passed.

## Open

- `qwen3_decode` is about 100% host on both runtimes (roughly 150k operators a
  step) and its 1.065 ratio is inside the drift band; closing it means fewer
  operators per token or cheaper host work per operator, not the device.
- The `is_causal=True` fast path (45.9 us against 54.5 us) needs an "is this
  mask exactly causal" test with real semantic risk; not done.
- `qwen3_train` (422.6 ms, host-bound, ratio 1.190) no longer needs its own
  itemisation: with the optimizer resolving readers once per parameter group,
  a paired run measures 412 -> **265 ms** (about 0.74 of torch's 357) and
  steady memory 26.9 -> 28.92 GB. A flat-buffer variant measured 264-270 ms
  for 6.1 GB more memory, so it is not being merged.

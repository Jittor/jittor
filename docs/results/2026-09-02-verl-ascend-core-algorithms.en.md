# Numerics, gradients and performance of verl core algorithms on Ascend 910B3

- Status: six families of policy loss, their gradients and the GRPO advantage are
  bit-identical to native `torch_npu` on a real 910B3 and are accepted; **performance (five
  paths `1.25x-1.75x` slower) and end-to-end NPU PPO remain open**
- Date: 2026-09-02
- Baseline commit: Jittor `97dc6ce9` (behaviour from `3758c4ab`); verl source
  `3d66a3d7ca1cf783df949816ec6862d5a7af9406`
- Validation scope: one Ascend 910B3, CANN 9.0.0; the native reference process loads
  `torch_npu`, the Jittor process does not, and the two run serially on the same card;
  `verl/trainer/ppo/core_algos.py`, `verl/utils/torch_functional.py` and the groupwise
  implementation are loaded directly from that verl commit, with a minimal shell provided
  only for the worker config and optional dependencies that take no part in the computation.
  Not covered: a full verl installation, Ray workers, FSDP2, weight transfer, rollout,
  1-step PPO (these previously had only CPU/CUDA evidence)
- Maintainers: Torch compatibility layer and verl adapter maintainers
- Review triggers: changes in verl's policy-loss formulas, Jittor clamp/autograd semantics,
  or CANN or the NPU adapter

## Conclusion

Six families of policy loss are covered — vanilla PPO, GSPO, SAPO, GPG, geometric-mean and
CISPO — along with the full gradient of each with respect to `log_prob`, the GRPO outcome
advantage and returns, and the backward of clamp at scalar/tensor boundaries, at exact
boundaries, and at the post-clip minimum. The fixed `4 x 6` input contains both positive and
negative advantages, a ragged mask, upper and lower clipping, and exact clip boundaries:

| Path | Max absolute loss error | Max absolute gradient/output error |
| --- | ---: | ---: |
| vanilla PPO, GSPO, SAPO, GPG, geometric-mean, CISPO | `0` | `0` |
| GRPO advantage/returns | — | `0` |

All Jittor results stay resident on the device, and within the capture window
`fallback_count=0` and `cpu_compile_count=0`.

## Defect reproduced first, then fixed: gradients at exact clamp boundaries

Before the fix, `clamp(x, -0.15, 0.25)` produced a zero input gradient at both exact
boundaries:

```text
input:       [-0.16, -0.15, -0.14, 0.24, 0.25, 0.26]
torch_npu:   [ 0.00,  1.00,  1.00, 1.00, 1.00, 0.00]
Jittor NPU:  [ 0.00,  0.00,  1.00, 1.00, 0.00, 0.00]
```

The geometric-mean loss has a `negative_approx_kl` that lands exactly on the clip upper
bound, so that gradient was 0 in Jittor and `0.0051804874` natively. `fdae6a0f` expresses
clamp as an inclusive three-way selection that preserves NaN: the gradient is 1 when
`x == min/max`; for a scalar boundary in backward it returns all `max` as PyTorch does;
input/min/max gradients for tensor boundaries are routed separately; Python scalar
boundaries on FP16/BF16 keep the same dtype; integer tensors with a float boundary are
promoted to float32. `3758c4ab` makes the forward of ACL float32 with ordered two-scalar
boundaries call `aclnnClampTensor`, while the custom backward still keeps "gradient 1 at
the boundary, gradient 0 for NaN", rather than treating the CANN forward as the complete
semantics.

## Performance (still open)

Timing includes the upstream function's internal metrics `.item()`, the loss forward, the
backward with respect to `log_prob`, and the device synchronisation; the inputs were already
resident on the NPU before timing began, and the numeric gate passed separately before
timing.

| Shape | Path | torch_npu | Jittor | Jittor/native |
| --- | --- | ---: | ---: | ---: |
| `512 x 512` | vanilla | 2.679 ms | 3.692 ms | 1.378x |
| `512 x 512` | GSPO | 2.707 ms | 4.728 ms | 1.746x |
| `512 x 512` | SAPO | 2.511 ms | 3.403 ms | 1.356x |
| `512 x 512` | GPG | 0.846 ms | 0.655 ms | 0.774x |
| `512 x 512` | geometric-mean | 2.397 ms | 3.823 ms | 1.595x |
| `512 x 512` | CISPO | 1.698 ms | 2.561 ms | 1.508x |

The ratios for `64 x 120` are almost the same (`0.776x-1.653x`). Native CANN Clamp has
already brought the five slow paths down from about `2.3x-2.6x` to the table above.

**Attribution: the gap is in the lazy graph segmentation triggered by metrics, not in the
computation itself.** The vanilla device profile has three large fused graphs, triggered in
stages by metrics `.item()`, the loss and the backward respectively; GPG has no inlined
metrics, only 8 launches, and is already no slower than native. When only the three metrics
returned by vanilla are replaced with an empty dict (loss, gradients, inputs and
synchronisation protocol unchanged), the median over 21 samples is `2.619 ms` native and
`2.589 ms` Jittor (`0.989x`): the three metrics add only `0.060 ms` on the native side but
`1.103 ms` on the Jittor side.

Tried and withdrawn (all numerically consistent, but none met the target): `stop_fuse` on
the four shared intermediates (`1.328x`); changing ACL `.item()` to a whole-graph
`sync_all` (degraded, with bimodal samples); fetching metrics and loss in one go (degraded);
stacking per-token metrics first and then reducing them together (best, still `1.220x`); a
grouped ACL CodeOp masked mean (slower). The only next steps are to make the policy-loss
forward, the metrics and the backward share a single dedicated fused computation, or to
uniformly defer metric extraction in the verl NPU adapter; global `.item()` must not be
changed to `sync_all`.

## Reproduction

The comparator probe (`core_algos_parity.py`) is not in the repository. Protocol: first
`source "$CANN_SET_ENV"` and expose only `ASCEND_RT_VISIBLE_DEVICES=<allocated-device>`;
start one process in the native `torch_npu` environment and one in the Jittor environment
with `JITTOR_TORCH_SHIM=1`, load the source from the verl commit above, feed the same fixed
inputs, and compare the maximum absolute error of loss, gradients and GRPO outputs; for
timing, warm up first, then take the in-process sample median, and record fallback counts
and CPU compile counts.

Main-repository maintenance regressions: the clamp CPU boundary/NaN cases and the Torch
compatibility cases (`compat/tests/torch/test_torch_compat_ops.py`), plus
`tests/backends/acl/test_acl.py` on NPU (including CANN Clamp forward and backward).

## Still open

- Making all six protocols no slower than native (see the attribution and next steps above);
- Restoring or rebuilding a maintainable verl NPU adapter that runs a real worker import,
  TensorDict batches, actor/critic forward and backward, the optimizer, weight transfer and
  1-step PPO;
- FSDP2/HCCL on NPU, Ray multiprocessing, vLLM rollout and Qwen3 scale each need separate
  validation; until then the full-PPO conclusions from CPU/CUDA must not be extrapolated to
  NPU.

Checked on `e3c369acb`: the clamp fix and the CANN Clamp runner
(`backends/acl/kernels/native/clamp_op_acl.cc`) are still present, with no new NPU re-test
since; the known-issues ledger has no separate entry, and this report is the sole record of
this performance gap and of the NPU PPO gate.

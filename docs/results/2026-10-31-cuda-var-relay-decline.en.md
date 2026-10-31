# Decline a relay whose member is outside the fused op instead of aborting

- Status: verified on a real CUDA device. The change is an admission test in
  codegen: a relay that cannot be represented is not taken, and the model runs
  the fused op's own implementation.
- Date: 2026-10-31
- Baseline commit: `3dc5c7ef3` (reproduction)
- Verification scope: RTX 4090 24 GB, CUDA 12.2.140 with the `jittor[cuda12]`
  component wheels, Torch frontend (`JITTOR_TORCH_SHIM=1`, `enable_tuner=1`),
  with the same models compared against `enable_tuner=0`.
- Maintainer: Jittor core maintainers (with the codegen maintainers on where a
  relay may end)
- Review conditions: `add_relay_group`'s member requirement, how `count_fuse`
  treats a broadcast view, or the relay gaining support for a member outside the
  fused op

## Symptom

Under the Torch frontend, a ConvBERT forward raised
`JittorError: var_relay.cc:107: [check failed: fnodes.count(v)]` at the span
conv in `modeling_convbert.py`; the failing operator is the two-convolution
product (depthwise 3x3 and 1x1) as `fused_op(reindex, reindex,
binary.multiply, reindex_reduce.add)`. ConvNeXt's depthwise conv hits the same
place (invisible earlier because the Var left by ConvBERT's failure poisoned
every later `sync`).

Smallest reproduction (on the device, under the shim; the shim turns
`enable_tuner` on):

```python
import torch, torch.nn as nn
d = nn.Conv1d(64, 64, 3, groups=64, padding=1, bias=False)
p = nn.Conv1d(64, 32, 1, bias=False)
v = torch.randn(1, 8, 64).transpose(1, 2)   # a transposed view made outside no_grad
with torch.no_grad():
    p(d(v)).numpy()                          # var_relay.cc:107
```

On the same card: `p(d(v.contiguous()))` passes, one convolution instead of two
passes, a view made inside `no_grad` passes, and the same graph on the CPU
(`jt.flags.use_cuda = 0`) passes.

## Root cause

When it builds a relay group, `VarRelayManager::add_relay_group` requires every
member of a relayed op other than the replaced output to be a node of the fused
op (execution reads them through the fused op's node ids). A convolution over a
broadcast view has one such member outside the fused op, so the assertion
aborted. It ran after the group's inputs had already been detached from their
ops, so "skip and carry on" was not an option either.

## Fix

Do the same check before touching any state and decline the relay group when it
fails:

```cpp
for (const auto& p : group) {
    Op* op = p.first->input();
    const auto& op_info = op->codegen();
    for (uint i=0; i<op_info.var_members.size(); i++) {
        Var* v = GET_VAR_MEMBER(op, op_info.var_members[i].second);
        if (v && v != p.first && !fnodes.count(v)) {
            relay_groups.pop_back();
            return -1;
        }
    }
}
```

With the group declined the operator stays inside the fused op and jittor's own
fused implementation computes it -- slower than the library kernel, and correct.
This is not the "relax the membership test" that KI-TUNER-001 warns about; both
models reaching the library kernel still requires the member to join the fused
op, which is a separate change.

## Results

- ConvBERT and ConvNeXt both run with `enable_tuner=1`, and their outputs are
  **bit-identical** to `enable_tuner=0` (`max abs diff 0.0`, batch 1, same seed).
- `python/jittor/compat/tests/torch/test_torch_hf_models.py::TestTorchHFModels::test_forward_and_eval_determinism`:
  1 passed (33 architectures, two forwards each, 97 s).
- `tests/codegen`: 169 passed / 2 failed / 38 skipped / 5 xfailed; the two
  failures are the KI-TEST-017 cases and are unrelated.
- The relay-related `tests/backends/cuda` cases (`test_cublas_matmul_grad`,
  `test_cublas_test_op`, `test_cuda_op_capabilities`,
  `test_cudnn_conv_amp_dtype`): 20 passed / 2 xfailed / 2 failed, both failures
  missing CUB headers on this machine
  (`cub/device/device_segmented_radix_sort.cuh`) and unrelated.
- `tools/run_test_suite.py --tier core`: 327 passed / 1 xfailed.

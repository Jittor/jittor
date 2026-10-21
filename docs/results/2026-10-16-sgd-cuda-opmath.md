# SGD CUDA low-precision update boundaries

- Status: bounded CPU/CUDA unit contracts verified; CPU low-precision parity remains open.
- Reviewed: 2026-10-16.
- Owner: native optimizers and Torch compatibility maintainers.
- Implementation delivery: `c5ed1212851dda005a7c64f1fc512974e1ed1e77`.
- Original CPU unit source: `5740cb4c761950fd0d1458ad79810a5b2bcf3228`.
- Original CUDA unit source: `2bc8ee53594fee93aa4aea935266a8a162ee4ea9`.
- Review when: SGD arithmetic, dispatch, momentum state, FSDP updates or precision policy changes.

Torch CUDA FP16/BF16 SGD performs scaled addition with FP32 intermediates and
one destination cast. The previous native expression rounded the scaled product
before addition, changing small parameter updates. The Torch frontend now requests
an explicit native `torch_math` policy. Native SGD defaults to `False`; CPU,
FP32 and existing fused-kernel selection keep their original arithmetic.

The existing shared SGD owner implements weight decay, parameter updates and
Nesterov scaled additions. Momentum multiplication retains its separate cast,
matching the actual `mul_` followed by `add_` boundary. FSDP2 forwards the same
policy to that owner. No adapter implements optimizer mathematics, and no host
update or backend fallback is added.

The three production files match the original measured sources byte for byte.
Together with the six existing `TestSGD` cases, CPU units pass **11 / 7 CUDA-only
skips**; the CUDA configuration passes **17 / 1 CPU-only skip**. The new precision
file alone contributes CPU **5 / 7 skips** and CUDA **11 / 1 skip**; the latter's
tensor locations are explicitly checked. Goldens cover FP16/BF16/FP32, plain SGD, momentum with weight decay and
dampening, and Nesterov over three steps, including each momentum buffer.
CPU low-precision units assert preservation of the existing native expression;
they do not claim low-precision agreement with Torch CPU.

Independent read-only verification rehashed all 58 original evidence files and
compared 90 published parameter/momentum arrays against the original genuine
Torch `2.13.0+cu130` NPZ data. New genuine CPU then CUDA replay compared another
144 complete arrays, including gradients and momentum, bit for bit with those
original records. The actual oracle `_C` SHA256 is
`819a187ea3b1f4d781200fe73fba3264689835d2285f7e6d10b986257a802dce`.
These are correctness checks, without performance claims.

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
python -m pytest -q compat/tests/torch/test_torch_compat_sgd_precision.py \
  compat/tests/torch/test_torch_compat_optim.py::TestSGD
```

CPU vector blocks and scalar tails have distinct low-precision rounding in this
Torch version. That difference remains unresolved. The new FSDP2 unit proves
single-update policy forwarding only; it is not distributed momentum or fresh
checkpoint/resume evidence. Generic saved-value/version checks and differentiable
`copy_` storage identity also remain separate open contracts. This result does
not grant complete optimizer ABI, other accelerators or full-model support.

Original evidence is unversioned under
`$JITTOR_LAB_ROOT/_state/omni-training-profile/20261016/`; independent review is
under `$JITTOR_LAB_ROOT/_state/sgd-policy-independent-review/20261016/`.
The latter's 21-file seal SHA256 is
`275f438c901ed80573bc9363f794779421ccf33c5c2994660b32aa003dd9f934`.
Main integration commands, exact source mapping and subsequent validation live
under `$JITTOR_LAB_ROOT/_state/sgd-main-integration/20261016/`.

Integration on `46391b45a61a298a246630b6e9f1e896cec9208a` plus the exact five-file
delivery passes CPU adjacent contracts **50 / 40 CUDA skips**, a combined CUDA
configuration **95 / 1 CPU skip**, native CUDA optimizer contracts **46 / 0 skips**,
core **322 / 42 skips / 1 existing xfail**, and structure **1628 / 6 skips**.
Both CUDA command guards record fallback before/after zero. These counts describe
their respective commands, including CPU controls in CUDA-visible environments.
A supplementary legacy-only command passed six assertions but exited one because
it declared no accelerator cases; that failed gate remains in the raw evidence.
The final combined command includes the explicit device contracts and passes
the unchanged accelerator gate.

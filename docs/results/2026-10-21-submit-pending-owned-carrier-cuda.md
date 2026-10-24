# CUDA owned-carrier partial submission

- Status: Focused CUDA regression passed; not complete PPO or L1–L5 acceptance.
- Date: 2026-10-21
- Baseline commit: `897d9cc61d9f0d9ddc6e51d02da5289cf56e085e`
- Owner: Torch compatibility maintainers
- Review when: carrier recognition, partial submission, or frozen FSDP output handling changes.

## Minimal repair

`submit_pending` now validates operands with the existing native
`core._is_tensor_carrier` predicate instead of accepting only `Var` instances.
This is one production-code line. Submission, synchronization, return identity,
autograd, CUDA kernels and PPO formulas are unchanged. Foreign objects remain
rejected before their methods can run. Existing unrelated work was preserved.

The failure was reproduced in the original VERL Reference builder's frozen FSDP2
forward: native storage passed a genuine independent Torch Tensor to the public
submission function, which rejected it before reaching the native executor.

## Executed validation

The six new `TestSubmitPendingCUDA` methods in
`compat/tests/torch/test_as_tensor_carrier_cuda.py` ran from an identical,
unversioned private copy with a scalar-only unittest result collector:
**6 executed, 0 failures, 0 errors, 0 skips; fallback count 0**.
They cover native Var, independent Tensor and complete input VJP, multiple-root
identity and per-root synchronization, default non-synchronizing submission,
foreign objects with forged methods, and the existing empty-root error.
Nonempty allocations were checked using actual CUDA driver pointer attributes.

A separate original-Reference prerequisite ran on two RTX 4090 devices using
VERL commit `1876b06d0a3e4e71e06230be10af14492ca8a75b`:

- Original builder called once per rank; two frozen forwards per rank passed.
- Eight global parameter elements, four physical shard elements per rank.
- All compared output elements had maximum absolute error 0, tolerance `1e-5`.
- Default-group broadcast, SUM/AVG all-reduce and barrier passed.
- Tensor return identity, foreign-object rejection and CUDA pointer checks passed.
- Both ranks exited successfully, were independently confirmed gone, and recorded fallback count 0.

Raw streams were suppressed; tensor arrays and model/config values were not saved.
No CPU or NPU framework computation was added in this validation.

## Provenance and limits

Unversioned receipts remain in the external lab state:

- Native source SHA256: `10620d01e9481f76bc4a0ce06d2fff8dbe47e88d73c8c2564b12f34ea10d963e`.
- Remote 3251-member source manifest SHA256: `2a09df4335193c7ddd9c4a44b0993ef390214f64bb72e5992a2736b54e4314ff`.
- Six-case CUDA receipt SHA256: `c50dba30d744387c7bfa7e5403c0932642d4d480810b62a01501e81e2bf48f33`.
- Dual-rank Reference completion SHA256: `84cc66ff405cf91361e664d02fadab25411f30b484738418b09eb34241af6173`.

The remote frozen source excludes the later local test/report additions; those
were not silently claimed as part of that manifest. Original failures and source
bytes were retained. This tiny Reference result is not Qwen model training,
independent PyTorch parity, a speed measurement, or complete PPO/GRPO acceptance.
The core/smoke/full suites and NPU/ROCm validation were not run in this scope.

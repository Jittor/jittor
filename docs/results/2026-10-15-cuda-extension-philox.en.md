# CUDA extension Philox and checkpointed RNG ownership

- Status: actual CUDA extension replay and fused FlashAttention training regressions verified; independent binary Torch FlashAttention comparison remains pending.
- Date: 2026-10-15.
- Synchronized baseline: `f2be6354924f595296c7d320a3f3f9304756e9d6`.
- Source fix: `fa25b2d43`.
- Owner: native CUDA RNG and Torch compatibility maintainers.
- Review when: CUDA generator ownership, checkpoint offsets, executor entry, extension generator interfaces or FlashAttention source changes.

## Defect and shared owner

The official FlashAttention extension failed to load because the compatibility
`CUDAGeneratorImpl` header referenced the removed `jittor::current_offset` symbol.
A direct module-load diagnostic exposed this error. The high-level loader had
replaced it with a later optional-module miss; `bd38eba39` separately preserves
all errors from a load attempt.

The extension now reserves Philox positions through the existing native
per-device checkpoint cursor. Reservations reject invalid increments and overflow,
round increments up to four positions, and enter the native executor when called
externally. The physical cuRAND generator advances with its recorded cursor.
Updating only the recorded cursor would make mixed native/extension continuation
incorrect after a checkpoint. No second RNG owner or checkpoint format is added.
The initial CUDA generator seed also comes from the existing native seed owner.

An actual compiled CUDA extension draws 128 Philox values. Its tests cover seed
reset, state save/restore, interleaved native normal/extension/native uniform
continuation, reservation rounding and invalid-increment atomicity. The two draw
tests use `forbid_backend_fallbacks()` and assert CUDA residency.

## Platform and verification

Linux x86-64, Python 3.11.16, one NVIDIA GeForce RTX 4090 (SM89), driver
580.178.04, GCC 11.5.0 and real NVCC 12.2.140. The selected physical card is mapped
to process-local CUDA device 0. CPU processes explicitly hide CUDA and clear
nvcc; CUDA processes select CUDA and enable native-required FlashAttention.
MKL, NCCL, MPI, CUTT and parallel op compilation are disabled in these checks.
CUDA kernel math is strict. These are correctness checks; concurrent independent
jobs preclude any speed or memory comparison.

The upstream source is official FlashAttention 2.7.4.post1, commit
`5231d95fe13733fb534c01895f7ea88c6a6c7793` from
`https://github.com/Dao-AILab/flash-attention.git`, with CUTLASS commit
`c506e16788cb08416a4a57e11a9067beeee29420`. Candidate compilation selects head
dimensions 32/64 and fp16/bf16. Compilation is serial within each cache; CPU core,
structure and CUDA checks use separate caches.

| Synchronized source check | Result | External log |
| --- | --- | --- |
| New compiled extension contracts plus CUDA RNG/fork_rng/generator-stream regressions | 25 passed, zero skipped | `philox-synced-cuda.log` |
| Actual native-required fused FlashAttention forward | 1 passed, zero skipped | `candidate-native-synced.log` |
| Actual fused backward and dropout replay, fp16/bf16, head dimensions 32/64 | 8 passed, zero skipped | `candidate-native-training-synced.log` |
| CPU core, native and Torch process modes | 322 passed, 42 skipped, one existing xfail | `philox-synced-core.log` |
| Complete Torch-mode structure suite | 1460 passed, six skipped | `philox-synced-structure.log` |

The fused backward tests compare all Q/K/V gradients with independent NumPy SDPA
math. Dropout tests check exact seeded output/gradient replay, finite nonzero
gradients and changed output as the RNG advances. They do not establish equality
to binary PyTorch's FlashAttention implementation or its RNG algorithm. Multi-device
extension RNG histories and CUDA graph capture are outside this acceptance.
The existing xfail concerns complex dtype placeholders; the six structure skips
are two accelerator checks, two declared checks and two isolated interpreters
without an installed Jittor distribution.

## Reproduction and evidence

Prepare the pinned upstream checkout and real CUDA toolchain, then set
`JITTOR_TORCH_SHIM=1`, source-pinned `PYTHONPATH`, `JITTOR_TEST_DEVICES=cuda`,
`CUDA_VISIBLE_DEVICES=<allocated-card>` and private `JITTOR_HOME`/`TMPDIR`.
Set `JITTOR_FLASH_ATTN_JITTOR_SRC=<upstream-checkout>`,
`JITTOR_FLASH_ATTN_HEAD_DIMS=32,64`, `JITTOR_FLASH_ATTN_DTYPES=fp16,bf16`,
`JITTOR_FLASH_ATTN_JITTOR_REQUIRED=1`, and both
`JITTOR_FLASH_ATTN_TRAINING_MIN_SCORES=0` and `JITTOR_FLASH_ATTN_MIN_SCORES=0`.

```bash
python -m pytest -q compat/tests/torch/test_cpp_extension_philox.py \
  compat/tests/torch/test_torch_cuda_rng_state.py \
  compat/tests/torch/test_torch_fork_rng.py compat/tests/torch/test_generator_streams.py
python -m pytest -q compat/tests/torch/test_torch_compat_attention.py \
  -k 'test_sdpa_native_flash_attn_fp16_cuda or test_sdpa_native_flash_attn_backward_fp16_cuda or test_sdpa_native_flash_attn_backward_bf16_cuda or test_sdpa_native_flash_attn_backward_hdim64_fp16_cuda or test_sdpa_native_flash_attn_backward_hdim64_bf16_cuda or test_native_flash_attn_dropout_replays_seed_and_backward or test_native_flash_attn_dropout_hdim64_fp16 or test_native_flash_attn_dropout_hdim64_bf16'
```

Raw logs are unversioned under
`$JITTOR_LAB_ROOT/_state/flash-attention-audit/logs/`. The original symbol failure
is in `candidate-official-diagnostic.log`; `philox-native-first.log` retains the
first binding-generation compile failure. The successful pre-sync fused run is
`candidate-native-fixed.log`; it remains tied to `aec3b683b` plus the four
externally hashed RNG files. After all worktree tests stopped, a fast-forward to
the baseline above preserved those changes and the synchronized checks were
rerun. No elapsed compilation timeout was treated as a terminal result.

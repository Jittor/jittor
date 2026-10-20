# FlashAttention public CUDA training against independent PyTorch

- Status: 48 historical CUDA comparisons and 12 maintained CUDA training cases
  passed. Twelve L3 records pass the shared schema; breadth remains 0/7 because
  matching independent upstream CPU execution is unavailable.
- Measured Jittor source: clean `71c5be74ef6991b75fa79d73fe52eb04c0319d59`.
- Owner: native CUDA and Torch compatibility maintainers. Reviewed: 2026-10-15.
- Review when: the upstream pin, extension ABI, public facade, native backward,
  optimizer arithmetic, precision protocol or comparison tool changes.

## Maintained training and controlled perturbation

Clean `06ed62ad728fae399ccbdbadec7a2441a13eaed9` ran all 12 nodes in
`compat/tests/torch/test_ecosystem_flash_attention.py`, without skips. The six
public training APIs each executed FP16 and BF16 at head dimension 32 and fixed
noncausal attention. Each node compared independent Torch and Jittor for both
the base input and a `0.125` noise perturbation. Q/K/V, target, causality and
shape stayed fixed; input hashes and actual changed outputs proved the
perturbation independently on both runtimes.

The 24 paired three-step trajectories contain 720 array comparisons and 216
separate update comparisons. All passed the original predeclared per-dtype
budgets. Maximum normalized error, including updates, was
`0.002433830240340736`. Actual CUDA placement, zero fallback, all gradients,
FP32 masters, precision readback and 58 tracked upstream Python file hashes
were recorded. The upstream and independent binary Torch pins and platform
remain as documented below. This is correctness evidence; concurrent jobs
preclude speed acceptance.

The reusable collector independently rechecks original arrays, immutable
comparison metadata, source identities and input perturbation before invoking
the shared scorecard auditor. Its measured schema admits 12 CUDA L3 records
and rejects none. **Shared accelerator breadth is still 0/7**, since that
metric requires a matching independent upstream CPU comparison. Upstream
FlashAttention provides no CPU execution; no CPU evidence or exemption was
invented. A separate independent binary-Torch probe explicitly hid CUDA,
allocated CPU FP16 Q/K/V, and called the real pinned public function. Torch
rejected `flash_attn::_flash_attn_forward` for the CPU backend; the full result
is preserved as unversioned `official-cpu-boundary.json` in the same state root.
Actual executed CUDA API scope is six of the seven pinned public
APIs, with KV cache unmeasured.

Raw `gate-schema-complete/` artifacts and generated scorecards stay unversioned
under `$JITTOR_LAB_ROOT/_state/flash-attention-audit/`. The earlier clean
`799af7118b352d5ac4621c5d657934dd2f6acc71` gate also passed 12 CUDA nodes;
its raw files remain unchanged. Canonical training metadata and source hashes
were then added and all 12 nodes rerun at `06ed62ad7`; old metadata was not
patched. The historical 48-case protocol below retains its original source pin.

## Historical executed scope

Both independent interpreters imported the public FlashAttention API. The
reference used binary Torch `2.11.0+cu128`, asserted its real `_C` extension,
and did not import Jittor. Its FlashAttention Python package came from the
clean official `Dao-AILab/flash-attention` checkout, version `2.7.4.post1`,
commit `5231d95fe13733fb534c01895f7ea88c6a6c7793`. The reference extension
was built from all 85 source declarations in that upstream `setup.py`, with
the upstream CUDA flags and no source edits. Actual CUDA forward/backward
smoke passed before numerical comparisons.

The candidate used Jittor's public Torch facade and the native-required
upstream CUDA extension bridge. Every deployed facade Python file matched
the measured repository resource. The probe checked actual CUDA placement,
real half/BF16 input and output dtypes, FP32 masters and gradients, finite
values, parameter updates and a zero-fallback guard. Importing an extension
was not used as evidence of execution.

The automatically regenerated upstream `attention_api` denominator has seven
members. Six were executed here, **6 / 7 at the pinned upstream SHA**, each
with eight scenarios:

| Public API | fp16 scenarios | BF16 scenarios |
| --- | ---: | ---: |
| `flash_attn_func` | 4 | 4 |
| `flash_attn_kvpacked_func` | 4 | 4 |
| `flash_attn_qkvpacked_func` | 4 | 4 |
| `flash_attn_varlen_func` | 4 | 4 |
| `flash_attn_varlen_kvpacked_func` | 4 | 4 |
| `flash_attn_varlen_qkvpacked_func` | 4 | 4 |

The scenario axes are head dimensions 32/64 and noncausal/causal execution.
Causal scenarios also perturb one differentiable input element by `0.125`.
That pairing is not an independently controlled perturbation comparison at
fixed causality. `flash_attn_with_kvcache` remains outside this training probe.
These counts do not establish coverage of internal APIs, all attention
features, production model sizes or the whole library.

Each run creates three FP32 trainable Q/K/V tensors and a separate FP32
differentiable input, casts attention operands to the requested fp16/BF16
dtype, and performs three actual SGD updates at learning rate `0.01`.
Dense tensors have shape `(2, 11, 2, head_dim)`; packed variable-length inputs
have lengths `[4, 7]`. Fixed NumPy seed `20261015` supplies the same nonzero
weights, input and target on both sides. Dropout is zero. Each runtime
retains 30 arrays per scenario: initial parameters and every step's output,
loss, input gradient, all parameter gradients and updated parameters.

## Platform and acceptance

| Field | Candidate | Independent reference |
| --- | --- | --- |
| Source/runtime | Jittor 2.0.0 at the measured SHA | Binary Torch 2.11.0+cu128 |
| Actual accelerator | One visible NVIDIA GeForce RTX 4090 | Same allocated GPU |
| CPU / OS | AMD EPYC 9554; Linux 6.1.188, glibc 2.34 | Same host |
| Driver | 580.178.04 | 580.178.04 |
| NVCC | 12.2.140 | 12.8.93 |
| C/C++ compiler | GCC 11.5.0 | GCC 11.5.0 |
| Precision | TF32 disabled, highest float32 matmul precision | Same requested policy |
| Threads / inputs | Two runtime threads; offline inputs | Same |
| Graph mode | Default native replay enabled | Ordinary Torch execution |

Per-runtime raw JSON retains compiler output, NVIDIA wheel versions, full
driver/device inventory, actual binary paths/hashes and native flags. Native
`use_acl` is a legacy alias of `use_cuda`; its recorded value does not claim
NPU execution. Other work used separate devices/caches on the host, so these
results establish correctness and make no speed or whole-process memory claim.

The comparator's limits were committed before the first matched comparison:
fp16 forward/state maximum normalized error `0.003`, gradients/updates
`0.006`; BF16 `0.03` for both. Normalization uses each reference tensor's
maximum absolute value with floor `1e-8`. Initial parameters must match
exactly. Each of the nine parameter updates is also compared separately, so
an omitted small update cannot hide behind a large initial parameter.

All **48 comparisons passed**, covering **1440 arrays per runtime** and
**1872 tensor/update checks**. The maximum normalized errors were:

| Field | fp16 | BF16 |
| --- | ---: | ---: |
| Attention outputs | 0 | 0 |
| All parameter and input gradients | 0 | 0 |
| Loss | 0.0000292143 | 0.00147610 |
| Updated parameter values | 0.0000000828791 | 0.0000000414392 |
| Separately measured parameter updates | 0.00305811 | 0.00304879 |

Outputs and every gradient array matched bitwise; each gradient field was
nonzero. Loss and optimizer update agreement is bounded by the original
limits rather than claimed bitwise. The comparator also validated matching
protocols, clean source commits, script hashes and NPZ hashes. Its 21 host
contracts cover missing gradients/updates, dirty sources, fallback reports,
protocol mismatches, corrupt arrays and Python `-O` disabling assertions.

## Reproduction and evidence

Use the isolated interpreters and commands in the
[FlashAttention skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/flash-attention-torch-compat/SKILL.md).
Run the committed `flash_training_probe.py` once per runtime with identical
API/dtype/head-dimension/causal/perturbation arguments, then invoke
`compare_training.py` on the two NPZ files. Preserve the measured checkout;
using a newer tool or runtime requires new evidence.

Unversioned evidence is under
`$JITTOR_LAB_ROOT/_state/flash-attention-audit/formal-71c5/`:
`candidate/`, `reference/`, `comparison/` and `accepted-evidence.json`.
The acceptance manifest binds 385 raw files and has SHA256
`def6b6f28daa4ad89184a6f4ba63c2b678578e9dfaf066413d910d441a163d91`.
The script SHA256 is
`c5e1ea376038e4e04da968599ac69bd3a2618815864e43a82b0aa7a7259469aa`.
The sibling `oracle-build.json` and `logs/oracle-build-v3.log` preserve the
complete build protocol and actual binary smoke result. Earlier failed
toolchain setup attempts and dirty-source diagnostics remain separate.

Upstream FlashAttention has no CPU CUDA-extension execution to compare here.
CPU facade contracts and native CPU regressions are separate evidence.
Dropout RNG identity across frameworks, KV-cache inference, full models,
performance, NPU and multi-GPU attention are not certified by this result.

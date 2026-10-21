# Accelerate maintained CPU fresh-process nightly

- Status: full local CPU nightly and strict L3 protocol accepted; hosted schedule unverified.
- Reviewed: 2026-10-16.
- Measured commit: `0dc8fdbd863c784dbe6564ebfeb9efb764b7b272`.
- Owner: downstream verification maintainers.
- Review when: upstream/wheel pins, installed-source proofs, Module/optimizer/RNG
  recovery, phase launchers, platform capture, provisioning or source closure changes.

The new `accelerate` profile provisions three independent environments and runs
`ecosystem_accelerate` with the complete original world-one FP32 fresh-process
protocol. Both input seeds 0/7 retain identical initial weights; each producer
performs three AdamW/StepLR updates, saves, computes two expected continuation
updates and terminates. New consumer processes construct fresh objects, restore
and perform the same two updates. No phase, field, gradient or budget was reduced.

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 \
python tools/provision_ecosystem_nightly.py --library accelerate \
  --state-root "${JITTOR_LAB_ROOT:-../jittor-lab}/_state/accelerate-nightly/fresh-run" --run
```

| Platform | Card / count | Precision / specification | Oracle | Actual flags |
| --- | --- | --- | --- | --- |
| Linux 6.1.188, AMD EPYC 9554, GCC 11.5 | CPU / zero visible cards | FP32 train, Linear(3,2), batch 4, two runtime threads | Official Torch 2.11.0+cpu, real binary checked | CUDA/ACL 0, eager, TF32/HF32 false |

CUDA/CANN/driver/nvcc are recorded as not applicable. Workers explicitly capture
empty `CUDA_VISIBLE_DEVICES`; candidate CPU observations and native flags exclude
accelerator execution. Both sides use offline inputs and HF offline mode. This is
correctness evidence, without speed or peak-memory claims.

The official Accelerate 1.10.1 source remains
`623cc0ba588d192c77758939a6e340adf07edd4f`; its official wheel SHA256 is
`3621cff60b9a27ce798857ece05e2b9f56fcc71631cfb31ccf71f0359c311f11`.
Normal pip resolves the complete runtime Requires-Dist closure separately into
candidate/oracle prefixes; the nox controller has its own tool closure. All three
pass `pip check`. Source editable installs reuse this resolved closure. No installed
prefix or JIT cache was copied. Actual pip RECORD is retained, not replaced by
archive RECORD; receipt capture alone does not claim byte equality of every
transitive dependency's generated pyc.

Independent read-only audits compare each Torch installation against its official
archive's 11,945 payload files, and each Accelerate installation against its 88
payload files, excluding archive RECORD.
Every actual hashed RECORD row matches: 11,949 per Torch installation and 96 per
Accelerate installation. Torch's 11,781 package files are a narrower scope than
its complete wheel. Installed Torch on the candidate supplies dependency metadata;
actual numerical execution uses the asserted native shim. The oracle's real binary
is separately hashed. All 2,934 framework Git blobs, executable modes and the
symlink target text match the measured archive; all eight worker maps match their
14 selected committed sources, including the Module loader and called controls.

| Model family / method / exact spec | CPU① level | Device③ level | Speed ratio / agreement | Status / evidence |
| --- | --- | --- | --- | --- |
| Protocol only: Linear(3,2), batch 4, AdamW/StepLR, 3+2 steps, fresh process | L3 | Not rerun; separate Module units only | Speed unmeasured; original numeric budget | 1 accepted / 0 blocked / 0 rejected; `arrays-v2.json` |

The maintained collector accepts exactly one L3 CPU record, zero blocked and zero
rejected. Independent NumPy recomputation verifies all 438 array pairs, 28 complete
parameter-gradient pairs and 14 input-gradient pairs, fixed full initial weights,
input perturbation, 56 actual per-parameter updates across both sides/seeds/phases,
and exact fresh-object model/optimizer/scheduler/RNG and next-random-draw recovery.
Maximum cross-runtime absolute error is `5.960464477539063e-8`; the original
`atol=2e-6, rtol=2e-5` elementwise budget is unchanged. Eight distinct worker PIDs
are verified. The 16,616 recursive exact checks include RNG scalars and are not
interchangeable with earlier reports' differently counted checks.

An initial complete run at `7cba3f5462f0ba08f3c0c56053611640bdf8b588`
preserved two actual candidate consumer failures: copying loaded values into
trainable leaves ran outside no-grad. The existing contiguous copying path now
performs the copy inside an internal native no-grad scope through the public
`copy_` owner, preserving caller grad mode, allocations, aliases, leaf and frozen
flags. The `assign=True` path is unchanged.
It adds no optimizer or copying mathematics. Independent genuine/candidate CPU and
CUDA each pass seven FP32/FP64 contracts, with actual CUDA pointers checked;
existing candidate storage regressions pass three per device, including CUDA BF16.
These units were measured at `23913169dde347754a8ee4d8817085196d0002da`.

A separate reproducible native autograd gap remains open: public `copy_` followed
by backward through an older square consumer fails to reject stale saved inputs.
The Module change does not close it or claim complete mutation/version ABI parity.
An old constant-gradient multiplication consumer remains valid on both runtimes.
The original before-fix diagnostics and failed full run are retained unchanged.

Complete CPU mode is explicit through `JITTOR_ACCELERATE_COMPLETE_CPU=1` and the
collector's default-off `--require-complete-cpu`. Existing ordinary CPU/CUDA calls
remain available. Persistent artifacts use `JITTOR_ACCELERATE_EVIDENCE_DIR`; unit
and benchmark caches are separate. Baseline Git instrumentation declares captured
control sources; missing/modified current sources fail. Historical c7's 48 source
maps and four records retain their fields and replay to two accepted/two unpaired
blocked/zero rejected. No registry denominator or old raw field is invented.

Mandatory validation: core 322 passed / 42 skipped / 1 xfailed;
`tests/structure` 1,516 passed / 4 skipped; host 147 passed. Canonical Ruff,
layout, manifest and diff checks passed. Evidence is unversioned
under `$JITTOR_LAB_ROOT/_state/accelerate-nightly/20261016/`: `fresh-v2`,
`arrays-v2.json`, `git-v2.json`, `installations-v2.json` and their `independent_*`
recheckers. `measurement-seal-v2.json` binds 590 regular files / 65,569,144 bytes plus
one symlink target text (591 entries); SHA256
`0958756b13f54b073c532d7ae0fa4b2d3bd104da392a3f1b8c45d6ca1bcc1c46`.
The earlier failed run has a separate immutable seal; validation/delivery receipts
are sealed separately without changing the measured commit.

This local nightly does not grant CPU world-two FSDP2, substitute DDP, rerun the
CUDA fresh protocol, prove multihost/sharded checkpointing or whole-library support.
The [historical CPU/CUDA source-bound results](2026-10-16-accelerate-installed-source-protocols.md)
retain their original commit and scope. Workflow wiring and local execution do not
prove GitHub's hosted schedule ran.

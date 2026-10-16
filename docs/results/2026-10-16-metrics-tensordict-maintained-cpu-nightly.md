# Metrics and TensorDict maintained CPU nightly

- Status: complete fresh local CPU nightly passed; hosted scheduled execution remains unobserved.
- Reviewed: 2026-10-16; owner: downstream verification maintainers.
- Measurement: `c47033ed0a6bb692922efa98ecbbf7896bfe867f`, clean throughout execution.
- Recheck when: source/wheel pins, installation proofs, builders, gradients, precision,
  provisioning, runtime isolation or source-closure validation changes.

The maintained `metric_tensordict` profile now runs both complete original component
protocols. It does not relabel the historical `912` CPU/CUDA measurements or the
older `90ee` records. The [installation report](2026-10-16-metrics-tensordict-installed-source.md)
retains those original identities and scopes.

| Actual CPU component | Complete protocol | Result | Public API coverage |
| --- | --- | --- | --- |
| TorchMetrics 1.7.4 | MeanAbsoluteError + MeanSquaredError; FP32 prediction/target `(4,8)`; seed 0 and fixed-weight seed 7; both input gradients; zero parameters | Four workers, six array pairs, exact equality, L2 | 2/105 |
| TensorDict 0.10.0 | TensorDict + actual lazy_stack; FP32 sample `(2,3,4)`, batch `[2,3]`, Linear(4,3); both parameter gradients and sample gradient; seed 0/7 and three SGD steps at lr .001 | Six workers, 29 pairs, max abs 9.536743e-7, L3 | 2/60 |

The unchanged budget is **.002 absolute OR .01 per-phase globally scaled**.
The common auditor accepted two CPU records, with zero blocked/rejected rows.
An independent NumPy audit recomputed all 35 pairs, 38 nonzero gradient arrays and
12 real parameter updates across both runtimes. Maximum globally scaled error was
6.690514e-8. Update arithmetic satisfied the independent two-epsilon float32
rounding bound; its maximum deviation was 1.323223e-8. Every phase used the same
complete initial weights, and each runtime's output changed under seed 7.
The real clean upstream inventories preserve all 105/60 public exports.

## Actual environment and source authority

All three private environments were newly provisioned with complete normal pip
resolution and passed `pip check`: candidate, genuine binary Torch oracle and
nox controller. NumPy/SciPy were 1.26.4/1.13.1. The oracle was Torch 2.11.0+cpu;
the candidate reported Jittor Torch frontend 2.0.0 and executed native CPU kernels.
Real Torch distribution metadata in the candidate is not its numerical provider.
No dependency-closure bypass was used; local editable source installs reused the
already checked closure. No installation or JIT cache was copied.

All ten workers directly captured `CUDA_VISIBLE_DEVICES=''`, CUDA availability
false and zero allocated cards. Their 74 tensor observations confirmed CPU;
native observations additionally required CPU residency/dispatch and fallback
error/count0. The host was Linux 6.1.188/glibc 2.34, AMD EPYC 9554, g++ 11.5.0.
Both runtimes used two threads, highest precision, TF32/benchmark disabled.
Native graph replay was 1 with a 268435456-byte retention budget.
HOME/JITTOR_HOME/TMPDIR/XDG roots were separate per runtime; unit caches were
independent of the benchmark and first compilation was serialized.

| Fixed official artifact | SHA256 | Whole archive files excluding RECORD / package files / actual hashed RECORD rows, each compute side |
| --- | --- | --- |
| Torch 2.11.0+cpu | `8a56a8c95531ef0e454510ba8bbd9d11dc7a9000337265210b10f6bfeacdd485` | 11945 / 11781 / 11949 |
| TorchMetrics 1.7.4 | `9298ad0e893b0cf2956bee95b0f7eecdc65205ab84ddb4f6762eff157c240518` | 350 / 346 / 353 |
| TensorDict 0.10.0 | `bccb6a7d7f02a8c83ca41815943bfa284da7d28019ab4276135809bda5ba05a6` | 46 / 42 / 49 |

The actual upstream commits are `986eb900b8bc660ff8174c95ec8f702e0e5cb6ca` and
`c34264a58b12471dc3e2913ea586b32431872d4c`. Provisioning fetches the real
TensorDict `v0.10.0` tag and requires its peeled commit; a tagless checkout was
actually rejected before this fix. TensorDict's generated `_version.py` and
native extension are protected by its fixed whole-wheel hash.
Actual pip METADATA/RECORD and complete target package payloads were rechecked;
this profile never restores archive RECORD over installed receipts. The independent
Torch proof includes functorch/torchgen/dist-info, beyond the torch-only scope.
The oracle binary hash is captured; candidate JIT binary SHA is not captured by
this protocol, so native source/dispatch evidence is not called a binary hash proof.
All 2929 Git blobs, modes and the symlink target text matched the archived tree;
every worker's 1613 selected source hashes matched that same commit.

Original metadata/receipts for all 85 installed distributions across three prefixes
are retained. An auxiliary all-dependency receipt checker exposed NumPy's generated
`conv_template.pyc` duplicate row and stale original bytecode hash; its two failed
diagnostics are preserved. This does not assert literal equality of all transitive
RECORD rows or weaken target-package validation. No installed bytes were changed.

## Reproduce and validation boundary

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 \
python tools/provision_ecosystem_nightly.py --library metric_tensordict \
  --state-root "${JITTOR_LAB_ROOT:-../jittor-lab}/_state/metric-td-nightly/fresh-run" --run
```

Optional read-only `--source-cache`/`--wheel-cache` reuse download inputs only.
The dedicated nox session requires both components and all ten workers, calling
`run_metric_td_gate.py` with the existing producer, verifier, training probe and
collector. It permits no case/step/budget reduction. Existing six profiles retain
original pins, roots and session behavior. The workflow adds this profile to the
CPU matrix; wiring and this local execution do not prove a hosted schedule ran.
Historical `912` replay still accepted four records with the exact original 1609-field
source map; the new runner is captured only when present in the measured Git tree.
A separate root integration review replayed all three complete array/Git/target-installation
audits successfully against the original c470 artifacts.

Validation: core 322 passed / 42 skipped / one expected failure; complete `tests/structure`
1516 passed / four skipped; targeted host contracts 135 passed. Final documentation,
layout, generated-manifest, Ruff and whitespace checks passed. Raw arrays, commands, locks, original receipts,
independent recheckers, failed diagnostics and source archives are under
`$JITTOR_LAB_ROOT/_state/metric-td-nightly/20261016/`; `measurement-evidence-seal-v1.json` (761 files, 351659740 bytes; SHA256
`0ef2211694cc8dd22e48ef45de6627e96c5f8d4630706894c89545d5d6a66fd8`) binds the evidence.
This batch grants no new CUDA measurement, whole-library, other dtype/backend,
distributed, L4 or L5 conclusion.

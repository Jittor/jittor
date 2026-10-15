# Metrics and TensorDict maintained CPU nightly

- Status: maintained provisioning and gate implementation; fresh execution pending.
- Reviewed: 2026-10-16.
- Owner: downstream verification maintainers.
- Implementation base: `d63976115fb6a31d94c4474871b07f2aba69d7e7`.
- Recheck when: source/wheel pins, installation proofs, component builders, gradients,
  precision, provisioning, launcher isolation or source-closure validation changes.

This entry preserves the complete protocols measured at
`91242395f555f3c24f60df665c3ea66e13d01caf`; it does not relabel those historical
CPU/CUDA arrays. The [installation report](2026-10-16-metrics-tensordict-installed-source.md)
retains their original scope and four shared-scorecard records.

| Component | Complete CPU protocol | Highest eligible level | Public API scope |
| --- | --- | --- | --- |
| TorchMetrics 1.7.4 | Four workers; seed 0 and fixed-weight seed 7; output and both input gradients; six array pairs; zero parameters | L2 | MeanAbsoluteError and MeanSquaredError, 2/105 |
| TensorDict 0.10.0 | Six workers; both parameter gradients and sample gradient; fixed-weight seed 7; three SGD steps at lr .001; 29 array pairs | L3 | TensorDict and actual lazy_stack, 2/60 |

The runner requires both components, ten distinct workers and all 35 array pairs.
The unchanged numerical rule is `.002 absolute OR .01 per-phase globally scaled`.
The complete public API denominators are regenerated from both clean pinned
upstreams; no dependency-based filtering or smaller model is substituted.

## Reproduce

Use a clean committed checkout and an empty external state directory:

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 \
python tools/provision_ecosystem_nightly.py \
  --library metric_tensordict \
  --state-root "$JITTOR_LAB_ROOT/_state/metric-td-nightly/fresh-run" --run
```

Optional `--source-cache` and `--wheel-cache` inputs are read-only download
inputs, not installed environments or compilation caches. Provisioning resolves
the full runtime dependencies into independent candidate and genuine binary
Torch environments, plus a separate nox controller; every environment must pass
`pip check`. Local source editable installation reuses that resolved closure.
The new profile preserves actual pip RECORD bytes for all pinned wheels.
TensorDict's actual `v0.10.0` tag is fetched and its peeled commit must match the
fixed source; a tagless `source@SHA` inventory is not silently called a release.

The dedicated nox session `ecosystem_metric_tensordict_source` calls
`tools/run_metric_td_gate.py`. It requires explicit upstream/wheel paths and an
independent oracle. The runner reuses `component_measure.py`, `verify_repo.py`,
`training_probe.py` and the common collector; it introduces no training math.
Both components must pass source, wheel, actual METADATA/RECORD, complete package
payload, platform, gradients, initial-weight and fallback validation before the
shared auditor accepts its two CPU records.

Each runtime receives its own HOME/JITTOR_HOME/TMPDIR/XDG cache. Benchmark
compilation is separate from unit testing and initial compilation is serialized.
The runner and its environment helper must be committed and captured in the
measurement closure. Historical baselines lacking the new runner retain their
original source maps.

## Execution boundary

Fresh execution results and the measured commit will be recorded after the
complete maintained command terminates. Workflow wiring alone does not establish
numerical passage. No hosted scheduled run has been executed or observed by this
change. This CPU profile does not grant new CUDA, other dtype, full-library,
distributed, L4 or L5 evidence.

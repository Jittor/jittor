# Benchmark coverage and comparison-protocol completion

- Status: Implemented and verified as described below; devices and scenarios remain uncovered, so this is not a full performance acceptance.
- Date: 2026-10-14.
- Baseline: `56b5271471ca9d872f287ea54f9d135875bafa59`, plus the commit containing this report.
- Scope: ASV convolution, Accelerate CPU training, real-model comparison protocol, catalog listing and CI selectors.
- Maintainer: Jittor maintainers.
- Review when: ASV/nox entry, precision configuration, comparison protocol, backend execution or CI selectors change.

## Changes and coverage

Added `convolution.Conv2dBenchmarks`, covering plain 3×3, stride-2 1×1, and depthwise 3×3 convolutions, each measuring forward and forward-plus-backward separately. The VJP of input, weight and bias and the forward result are first checked against an independent NumPy float64 reference at the actual timed shapes, then FP32 timing follows. The CUDA path pins and restores the precision configuration, and checks actual tensor placement and zero fallback.

The model harness now aligns CPU thread configuration and records the actual device, precision, dependencies and fallback. When conditions are missing or inconsistent it keeps the raw times but generates no comparable speed ratio. Crashes and timeouts after a success marker also fail the run. `--list --json` shows all 14 workloads without importing models.

`benchmarks/catalog.py` aggregates six entry categories, distinguishing declared capability from existing verification; this run found 20 ASV methods, 14 model workloads, 3 ACL models, 7 ecosystem speed cases, 4 graph construction cases and 17 profiling tools. These are catalog entry counts, not hardware verification counts for all combinations.

The CPU PR benchmark gains convolution, the regular benchmark gains reductions; the CUDA regular benchmark gains convolution, reductions, transfer. Added an optional `nox -s benchmark_accelerate` entry comparing the cost of the same Jittor training under direct invocation and an Accelerate wrapper. Fixed ASV configuration relative paths and CPU-only Accelerate initialization; install, compile or execution errors are no longer treated as missing devices and skipped.

## Real execution and regression results

| Verification | Results and boundaries |
| --- | --- |
| Convolution: Jittor/PyTorch × CPU/CUDA × 3 convolutions × forward/fwd+bwd | 24 combinations passed; forward and input, weight, bias gradients checked against NumPy; actual CUDA execution on RTX 4090, zero fallback |
| Model harness: CPU tiny ResNet50 inference | both independent interpreters succeeded; actual CPU, FP32, 2 threads, same NumPy 1.26.4; Jittor fallback 0; scalar mean sequences agree within 2e-6. Only the harness is verified, not equivalent to elementwise/gradient parity or performance conclusions |
| Accelerate: CPU SGD/AdamW × direct/full | 4 combinations actually trained and passed, finite parameters, actual CPU, zero fallback; corresponding direct/full parameter max absolute differences are 7.45e-9 and 1.95e-7, within the FP32 `rtol=1e-5, atol=1e-6` budget |
| CPU core | 322 passed, 42 skipped, 1 xfailed (native and Torch modes combined) |
| All structure tests | 1429 passed, 4 skipped, 1 failed; the only failure is the baseline pre-existing stale DropoutACL gradient inventory item |
| New targeted tests | ASV entry 10, model protocol 11, convolution 6, catalog listing 6 passed |
| Static gates | ASV actual configuration check, catalog with no site-packages enumeration, layout, documentation links, maintenance-scope Ruff checks passed |

CPU convolution and Accelerate use generic CPU kernels; this run gives no conclusion about CPU optimized-library performance or cross-framework speed ratios. The model smoke's timing samples are insufficient to publish a performance comparison. The single structure-test failure is `test_backend_grad_contract.py::test_the_manifest_is_exactly_the_gradients_in_the_tree`, pointing at the stale `DropoutACL` entry in the manifest; that backend source is outside this change.

## Reproduction and artifacts

Run from the repository root; actual timing needs a configured interpreter, device and separate compilation cache; full parameters are in the [benchmarking guide](../performance/benchmarking.en.md) and the [coverage matrix](https://github.com/Jittor/jittor/blob/2.0-refactor/benchmarks/README.md).

```bash
python -S benchmarks/catalog.py --json
python -m asv --config benchmarks/asv/asv.conf.json check
nox -s benchmark -- --bench '^convolution\.' --quick
nox -s benchmark_cuda -- --bench '^convolution\.' --quick
nox -s benchmark_accelerate -- --quick
python tools/run_test_suite.py --tier core --backend cpu --serial-compile
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

`--quick` suits smoke runs, not publishing stable speed ratios. Raw logs, caches and JSON are all in the external lab directory, **unversioned**: convolution in `$JITTOR_LAB_ROOT/_state/benchmark-asv/asv-{cpu,cuda}/logs/`; model JSON in `$JITTOR_LAB_ROOT/_state/benchmark-completion/harness/results-cpu/results.json`; integration checks and Accelerate logs in `$JITTOR_LAB_ROOT/_state/benchmark-completion/integration/logs/`.

## Still-unverified scope

This run did not rerun the complete model CUDA/NPU harness, nor verify Accelerate CUDA. NPU's FP32 comparison protocol still relies on declarations; Jittor's cube precision configuration lacks public readback, and changing related configuration needs real-NPU re-verification. ROCm/Corex regular baselines, NPU ASV/CI, multi-card communication and distributed training, data input pipelines, cold start and systematic shape/dtype sweeps remain gaps; the coverage matrix is the authority for continuous coverage status.

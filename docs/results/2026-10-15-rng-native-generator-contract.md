# Native RNG seed and explicit-generator contracts

- Status: verified for the CPU and CUDA paths listed below; no cross-framework sample-bit identity claim.
- Reviewed: 2026-10-15.
- Owner: Jittor runtime and Torch compatibility maintainers.
- Initial reproduction baseline: `4a4a480e628dae634a9d660f320c3aa957341e99`.
- Final integration baseline: `b2aeec10874136c1b4b16c1598241252a5f92c87` plus this change. Intermediate CPU core and CUDA results at `a5932248fdb06fb52293f5ddd8a357f4920195ff` retain their original source seal. Synchronization preserved every owned file; final device and CPU checks ran again on the synchronized tree.
- Recheck when: seed bindings, native generator engines, CUDA stream reservation, tensor factory placement, or checkpoint RNG ownership changes.

## Confirmed failures and resulting behavior

Independent binary PyTorch 2.11.0+cu128 accepted the seed interval `[-2**63, 2**64-1]`, mapped negative seeds into unsigned 64-bit metadata, rejected both adjacent out-of-range values without mutation, and returned its default generator from `manual_seed`. The original Jittor frontend narrowed valid large values through a C++ `int` binding; `-1` could change its reported identity after a state restore. CUDA-only seeding additionally reseeded CPU/Python/NumPy through a convenience function. These were execution results, not conclusions inferred from the earlier PR audit.

An explicit shim generator previously drew from `numpy.random.Generator`, including when its final tensor was copied to CUDA. The old probe recorded actual NumPy calls. Its CUDA state tensor also lived on CUDA. An output pointer on a CUDA device did not establish a CUDA sampling provider.

The frontend now validates the Torch seed domain before mutation, delegates a full-width global seed to the existing native CPU owner and CUDA reset callback, and returns `torch.default_generator`. CUDA-only setters preserve CPU/Python/NumPy state. Native CUDA defaults retain one seed and cursor per device; legacy version-1 signed-seed CUDA checkpoints remain readable, and version 2 stores unsigned seed bits.

Explicit generators wrap the existing native `jt.Generator`: the native CPU engine on CPU and the existing counter-based CUDA Philox kernel on CUDA. Supported factories pass the actual native handle through the random op; no second PRNG or NumPy sampling algorithm is added. Their opaque states are one-dimensional CPU ByteTensors. Pending CPU draws are completed before capture or reseeding. Foreign/corrupt states and mismatched devices fail before changing the owned stream. The existing native state parser now also refuses trailing bytes before committing a replacement engine.

The CUDA factory regression additionally exposed `normal(Tensor mean, scalar std, generator=...)` choosing the default CPU placement. The factory now derives placement from a Tensor mean or std, including keyword arguments. Both forms execute on their requested native CUDA stream.

A two-device checkpoint reproduction seeded devices with `2**64-1` and `2**63+17`. The native checkpoint consumer copied the current device's seed into both snapshot entries and restored the wrong second stream. It now queries the seed for each device. A second minimal reproduction confirmed that old native checkpoint tuples containing signed `-1` could no longer cross the unsigned restore binding. The consumer now validates the 64-bit interval and maps valid legacy signed bits without masking overflow. Legacy continuation and invalid boundary cases execute in the two-device test. Checkpoint layout and FSDP/optimizer mathematics are unchanged.

## Verification

First JIT and extension compilation ran serially. CPU and CUDA had independent task-owned homes/caches, and the core runner separated native and Torch process caches. CUDA used the existing strict CUDA 12.2 wheel/toolchain fingerprint `pipcu122_a89fa4fed08c`, with backend fallbacks forbidden. Actual allocations were checked with the CUDA driver device-ordinal query before host reads.

| Scope | Result | Evidence |
| --- | --- | --- |
| Torch seed/provider/state CUDA matrix, including two physical devices and native checkpoint consumer | 96 passed, 0 skipped | `logs/final-cuda-session-checked.log`, `final-cuda-session-checked.json` |
| Native Generator and RandomOp CPU/CUDA, with independent Torch 2.13.0+cu130 statistical reference | 35 passed, 0 skipped | `logs/final-native-session.log`, `final-native-session.json` |
| Independent Torch 2.11 / 2.13 and repaired shim CPU seed API comparison | 18 seed/state/error rows agree; 10 explicit native-handle primitive calls | `cpu-api-comparison.json`, `reference-cpu.json`, `reference213-cpu.json`, `candidate-cpu-after.json` |
| Native checkpoint two-device counterexample and repaired restore | before: second seed incorrectly copied; after: both exact, fallbacks 0 | `checkpoint-per-device-{before,after}.json` |
| Legacy native checkpoint negative seed | before: unsigned binding overflow; after: unsigned metadata and exact replay, fallbacks 0 | `checkpoint-legacy-{before,after}.json` |
| CPU Torch RNG matrix | 53 passed, 28 accelerator skips; fallbacks 0 | `logs/final-cpu-session-retry.log`, `final-cpu-session.json` |
| Native CPU checkpoint suite | 30 passed, 0 skipped; fallbacks 0 | `logs/final-checkpoint-cpu-session.log`, `final-checkpoint-cpu-session.json` |
| Final compute/test source structure | 1486 passed, 6 skipped before adding this report | `logs/structure.log` |
| CPU core at the intermediate baseline | 322 passed, 42 skipped, 1 xfailed | `logs/core.log` |
| Layout and whitespace | passed | `logs/layout.log`; `git diff --check` |
| Ruff against the original touched-source baseline | 0 new diagnostics; existing total 132 to 131 | `ruff-comparison.json` |

The CUDA matrix includes full-width positive and negative seeds, invalid-seed atomicity, default-generator identity, Python/NumPy/global-stream isolation, lazy draw capture, fresh-generator restore, pickling, corrupt state rejection, mismatched-device rejection, and actual native handles for twelve factory forms. `randint` is exercised on a small representable interval. Seven CUDA-extension rows cover mixed native/extension continuation, reservation rounding, invalid increments, unsigned full-width seed bits, and exact replay after restoration.

For two devices, the test checks distinct seed metadata, CPU ByteTensor snapshots, real allocation ordinals, unchanged state on the untargeted device, exact per-device continuation, all-device restoration, malformed multi-state refusal before mutation, independent explicit-generator state, and native checkpoint snapshot/restore. This is single-host device isolation, not a multi-host result.

Earlier failed attempts are retained: private interpreter bootstrap/import ordering, an observer attached to the Python package rather than the actual core random primitive, an incorrect old expectation that `manual_seed` returns the Torch module, a bare-CUDA constructor's shadowed `max`, the Tensor-normal placement failure, and missing private pybind11 headers before extension compilation. No failure was converted into a permissive assertion. The old shared Torch-ref installation subsequently lacked its shared-memory helper; it was left untouched. A separately verified complete binary Torch 2.13 installation supplied the final RandomOp reference and repeated all eighteen CPU seed API rows with the same outcomes. Its binary identity is recorded independently. A later CPU cache-path switch produced the documented JIT-tool update exit code 3 before any test ran; that startup was retained and the same source was rerun after rebuilding.

After this report was added, the focused documentation/packaging run passed 21 checks and failed the two central registration checks: the new page is absent from the results toctree and generated root manifest. The integration owner maintains those two shared files; this standalone change does not claim those checks passed. The preceding 1486/6 structure result applies to the unchanged compute/test source before adding this report.

Integration onto `ab006244a34c550a5f7235b1f65ec7d4165bde9d` registered this page in the results index and generated manifest. All 120 sealed raw artifacts and all 17 committed file hashes were independently rechecked. The integrated structure/tool suite passed 1690 checks with six declared/environment skips; core passed 322 checks with 42 skips and one existing expected failure. A combined CPU RNG/pytree follow-up passed 62 checks with 36 accelerator skips. These integration logs are unversioned under `$JITTOR_LAB_ROOT/_state/rng-integration-ab006/`. The earlier standalone registration failures remain historical evidence.

## Reproduction

Use a private Python environment with the repository's pytest/plugin pins, NumPy, pybind11 for the extension rows, and a complete independent binary PyTorch installation for native reference rows. Set `PYTHONPATH` to this checkout's `python`, isolate `HOME`, `JITTOR_HOME`, and `TMPDIR`, and select CPU/CUDA with the canonical `JT_BACKEND` / `JT_USE_CUDA` settings. With pytest autoload disabled, enable both `pytest_timeout` and `xdist.plugin`.

```bash
# Torch mode; CUDA build, two visible physical devices, strict fallback scope in tests.
JITTOR_TORCH_SHIM=1 python -m pytest -q \
  compat/tests/torch/test_generator_native_contract.py \
  compat/tests/torch/test_generator_streams.py \
  compat/tests/torch/test_torch_sampler_rng.py \
  compat/tests/torch/test_torch_fork_rng.py \
  compat/tests/torch/test_core_misc_owner.py \
  compat/tests/torch/test_torch_cuda_rng_state.py \
  compat/tests/torch/test_cpp_extension_philox.py

# Native mode, CUDA build with CPU ambient placement. Tests select CUDA where declared.
# REAL_TORCH_SITE must resolve to the actual complete binary package's site directory.
JITTOR_TORCH_SHIM=0 JT_USE_CUDA=0 use_cuda=0 \
  REAL_TORCH_SITE=<independent-torch-site> JITTOR_REQUIRE_REAL_TORCH=1 \
  python -m pytest -q tests/ops/test_random_generator.py tests/ops/test_random_op.py

python tools/run_test_suite.py --tier core --serial-compile
bash tools/check_repo_layout.sh
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

Raw JSON, source hashes, independent runtime identity, exact environment scripts, failed runs and logs are unversioned under `$JITTOR_LAB_ROOT/_state/rng-generator-entry-contracts/20261015/`. `baseline-source.json`, the synchronization records, `final-compute-source.json` and `post-sync-compute-source.json` keep each baseline and owned-source snapshot separate; `SHA256SUMS.json` lists retained raw artifacts while explicitly excluding runtime caches and dependencies.

## Boundaries

Native CPU, native explicit CUDA Philox, CUDA default cuRAND, and PyTorch use different algorithms/state formats. The guarantee tested here is seed metadata, stream ownership, isolation, and exact within-runtime replay. PyTorch or former NumPy-backed shim state bytes cannot be replayed by these native owners and are refused. Cross-device explicit state migration is intentionally rejected.

This does not upgrade every factory's fidelity: native integer sampling's existing float-based approximation, large integer intervals, stable tie handling, unsupported `out=`, and broader layout/memory-format semantics remain outside this result. The default-generator delegate was a separate Python class in this measurement. The [default-generator follow-up](2026-10-15-default-generator-state-contract.md) unifies the public Python class, CPU snapshot exchange and private-copy semantics; complete C-extension ABI parity remains outside these results. No ROCm/NPU/Corex hardware claim follows from shared source edits. The legacy callback ABI and legacy native `jt.set_seed(int)` entry remain unchanged; full-width behavior of other RNG consumers such as cuDNN RNN dropout was not executed here. No benchmark, speed, or memory-capacity conclusion is drawn from these small correctness tests.

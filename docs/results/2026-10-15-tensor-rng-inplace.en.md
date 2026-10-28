# Generator and autograd for in-place random Tensor filling

- Status: the two Tensor methods scoped to CPU / a real single CUDA device pass; the structure gates are detailed in the table below.
- Date: 2026-10-15.
- Baseline: synchronized `origin/2.0-refactor` at `97d7d478a48ecc80f13f2858f0bc8fa78e377480`; after retaining the committed nn.init fix and historical results, integrated as `19655e404a89b08f61d4585de6558f1234c30149`.
- Code: `11430ebac72bc214fb2562158974c96666352a5c` (production semantics); `6afbca891d00f4fda560553086a12c3ff883d3b5` (test helper extraction).
- Owner: Torch Tensor / native RNG maintainers.
- Review when: the Generator factory, random initialization, in-place storage, or autograd graph owner changes.

## Actual breakage

On the standalone binary Torch `2.11.0+cu128` CPU, a minimal probe first showed that
`Tensor.uniform_` / `normal_` advance a private Generator, preserve the global stream,
replay after full state restoration, and reject modifying a requires-grad leaf tensor in
grad mode. The same calls on the frozen candidate baseline did not advance the private
stream, changed the global stream, could not replay, and accepted modifying leaf tensors.
Another real minimal probe proved that, when overwriting a non-leaf tensor, real Torch
gives the old input a complete zero gradient and preserves the non-leaf identity; the
candidate rebound it as a leaf tensor and the old input's grad was None.
The four actual oracle/candidate baseline logs are retained, not replaced with post-fix
records.

## Implementation and boundaries

The validated nn.init validation / dtype / device / draw / affine logic was extracted into
two private values helpers. The two public nn.init entry points keep their own no-grad
initialization and assignment; the Tensor methods reuse the helpers, keep their own
in-place semantics, and do not call the other consumer that ignores the Generator.
The private stream is still executed by the existing factory and the native CPU engine /
CUDA Philox; no PRNG was added.

Tensor rejects leaf tensors in grad mode and views of leaf tensors before any sampling or
counter reservation. no_grad allows initializing a trainable leaf, preserving its identity
and training switch. Overwriting a non-leaf tensor uses the existing false-ternary path
shared with zero_: the new values keep a zero-derivative connection to the old graph, and
NaN/Inf is not propagated into the new random values through `old * 0`.
The write-back reuses the existing `_ip(..., preserve_storage=True)`; the shared `_ip` and
native autograd are unmodified.

This batch only declares the two Tensor methods for the four real floating dtypes and the
shared nn.init regressions. It does not grant all in-place operators, other initializers,
the full Generator ABI, integer range precision, cross-Torch RNG byte interoperability, or
untested ROCm/NPU. The first batch's report and original source/oracle SHAs keep their
original measurement identity and are not relabeled by this synchronization or the helper
extraction.

## Actual verification

The new tests and standalone Torch use the same file; only the candidate native observer is
excluded. The oracle interpreter rejects the Jittor import before and after the tests and
records the actual binary extension, interpreter, version, CUDA-available devices, and the
test SHA256. States / outputs / gradients are compared as complete arrays.
Cross-framework samples with different PRNGs are not required to be bitwise identical; what
is verified is the state and in-place behavior contract and the actual distribution
parameters.

The new tests cover float16/bfloat16/float32/float64, global CPU/CUDA/Python/NumPy stream
isolation, full replay with consecutive fills mixed with factories, default CPU Generator
identity, atomic rejection of dtype bounds/std, wrong generator/device, frozen targets,
leaf/view rejection, the non-leaf zero derivative (including the original NaN/Inf), and
write-back through a retained view. All new candidate nodes execute inside a scope that
forbids fallback.
The native observer forbids NumPy sampling, checks the original native handle and the
float64 sampler dtype, and verifies the actual allocation ordinal with the driver pointer
attribute before CUDA host readback.

| Check | Actual result | Scope |
| --- | --- | --- |
| Standalone Torch CPU | 41 passed / 41 CUDA skipped / 4 native observer deselected | Complete general Tensor tests including the appended no_grad non-leaf checks |
| Standalone Torch CUDA | 82 passed / 0 skipped / 4 native observer deselected | Real single CUDA |
| Candidate CPU | 145 passed / 103 skipped | The eight maintenance files below |
| Candidate CUDA | 270 passed / 1 skipped | Same eight files, require CUDA / min execution; only the both-device item skips |
| Appended no_grad non-leaf checks | CPU 2 passed / 2 CUDA skipped; CUDA 4 passed | No production source change, run separately as a targeted check; old graph / pointer / pending reader |
| CPU core | 322 passed / 42 skipped / 1 existing xfail | Canonical native / Torch both modes exit 0 |
| Final shared test helper regression | CPU 74 passed / 74 CUDA skipped; CUDA 148 passed | nn.init + Tensor two files; the extra 35 dedup/pytest contracts pass |
| Final standalone Torch | CPU 70 passed / 70 CUDA skipped; CUDA 140 passed | Same two files, all general tests, 8 native observer deselected |
| Complete structure / layout | 1622 passed / 6 skipped; layout passes | tests/structure + compat/tests/structure at clean 6afb; before registration on this page |

The very first expanded regression command misspelled one existing file name, pytest exited
4 with zero nodes executed; that log is retained, and it was rerun afterwards against the
actual eight files, without counting a collection failure as a pass.
The first complete structure run also had two pytest collection-contract failures: importing
a fixture across test modules, where the fixture alias is not recognized by the static rule.
Defining it locally was then rejected by the dedup check. In the end the shared logic was put
in `_rng_init_support.py`, the two tests keep only explicit thin fixtures, and no allowlist
entry was added. The AST of every test_* function equals the earlier measurements one by
one; the final two files were fully re-verified on standalone Torch / candidate CPU / real
CUDA, and the two production files are unchanged.
All these original failures are retained; the other two first-batch docs/MANIFEST
registration failures disappeared after integrating into mainline 9ba.
The new tests and shared helpers pass Ruff completely; the existing 21 diagnostics of the
two modified production files match the baseline item by item, with no additions.

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <candidate-python> -m pytest -q \
  compat/tests/torch/test_tensor_rng_initializers.py \
  compat/tests/torch/test_nn_init_generator.py \
  compat/tests/torch/test_torch_init_package.py \
  compat/tests/torch/test_generator_native_contract.py \
  compat/tests/torch/test_default_generator_contract.py \
  compat/tests/torch/test_tensor_inplace_grad_views.py \
  compat/tests/torch/test_frozen_tensor_in_place.py \
  compat/tests/torch/test_tensor_inplace_storage.py
```

Before running, the environment manual still requires independent HOME/JITTOR_HOME/TMPDIR,
an explicit CPU with CUDA hidden, or a real CUDA toolchain with strict fallback. The cache
is the same independent unit cache of the same R33 consumer task; it is separate from
benchmarks and runs CPU before CUDA. Other processes share the GPU, so only correctness is
granted.

Unversioned evidence directory:
`$JITTOR_LAB_ROOT/_state/nn-init-generator/20261015/tensor-followup/`.
`source-freeze.json` corresponds to test source SHA
`7d8eec1939a181eafd20677b0af7f64bd66cade57df2eb1c12432cc7e4347be2` of the complete
eight-file regression. Afterwards only four no_grad non-leaf nodes were appended to the end
of the same test file; the two production files are byte-for-byte unchanged, the original
test file is an exact byte prefix of the appended file, and the original copy and SHA are
retained. `no-grad-recheck/` records the appended standalone oracle's complete general
tests and the candidate's four new targeted nodes; the already-passed eight files are not
rerun and the old logs are not rebound to the new test SHA.
`source-freeze-extended.json` records the two test source identities separately.

After fixing the test collection/dedup contract, `common-helper-recheck/` holds the final
two-file oracle and candidate CUDA, and `common-helper-cpu.log` holds the candidate CPU and
the 35 targeted structure gates; `source-freeze-common-helper.json` binds the final five
code/test/helper files and verifies that the production files still equal 114.
At this point only the test organization changed, the AST of every test_* function is
unchanged; the actual new SHAs after re-verification are also retained, and old records are
not backfilled. The final structure ran on the clean 6afb and integrated-mainline 9ba trees,
and the old eight-file mathematical measurements keep the 19655 baseline.

The source freeze, baseline/oracle/final logs, oracle metadata, and final manifest bind the
actually measured source / original file hashes. At measurement time three files were not
yet committed, and the Git blobs were checked file by file afterwards; historical
dirty-source results are not relabeled as clean-source.
The central results index and MANIFEST are registered by the integration maintainer, and
the final documentation-tree gate must be re-verified after registration.

An additional minimal CPU audit confirms that the existing nn.init._assign reversible
freeze for `requires_grad_(False)` may still unfreeze; the first batch of tests only covers
the construction-time gradient-stopped state and cannot be extended to claim all frozen
states pass. That separate old defect is fixed elsewhere, and this batch's CPU control for
the reversible freeze of the Tensor methods keeps False; it is not written up as the still
unfinished nn.init CPU/CUDA acceptance.

## Main-tree integration re-check

After synchronizing remote `14aeca5df0bf841e99d6241e06b076e34ef08c05`, the three original
commits integrated without conflict, and the five production/test/common-tool files are
byte-equal to the original 114 / 6afb freeze. The central index and MANIFEST are registered.
The SHA256 of the original 59 artifacts recomputed item by item is unchanged, and the
historical measurements are not relabeled.

On the clean main tree `74e39acd1fd4b211c34cc5460e623f4499760f6e`, the eight files give CPU
**147 passed / 105 CUDA skipped**; correctly configured real single CUDA gives **274 passed
/ 1 both-device skipped**; core is **322 passed / 42 skipped / one existing xfail**. The
no-grad nodes added by the new and old tests keep separate source identities. One CUDA run
mistakenly set the minimum accelerator count to 200 while 122 were actually selected;
although all 274 assertions passed, the process still exited 1. That failure is retained,
and after correcting the configuration the same set, same source, and same budget passed
again, without passing by modifying assertions or deleting cases.

The main-tree re-check also found that the size factories could mark gradient-requiring
creation results as non-leaves; this was then fixed and verified separately, see
[size-factory leaf identity](2026-10-15-size-factory-leaves.en.md). The non-leaf `ones`
target in the old short probe cannot be taken as evidence of a real leaf rejection; the
explicit leaf target created by `torch.tensor` and the maintenance tests confirm that this
batch's guard is effective. The subsequent factory work neither rewrote the native RNG nor
expands this report's backend or full-Generator-ABI conclusions.

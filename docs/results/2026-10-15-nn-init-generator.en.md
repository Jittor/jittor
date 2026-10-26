# Generator-driven in-place floating-point initialization

- Status: two nn.init consumers pass on the bounded CPU / real single CUDA scope; no other initializer is granted support.
- Date: 2026-10-15.
- Baseline: `8bb59861de5bbf98954728a331a4efe42ea8a389` on `origin/2.0-refactor`; after the existing runner refs were preserved, integrated as `b196ab01a6d9a0bdef1e9143205983c2048f8b66`.
- Code: `222c46662fba3b4fa7550ccfef66401cde3f4787` (the measurement files were checked byte-for-byte against that commit).
- Owner: Torch frontend / native RNG maintainers.
- Review when: the factory random stream, Generator, initialization dtype, in-place write-back or device publication changes.

## Actual break and fix scope

An independent binary Torch `2.11.0+cu128` on CPU first ran a minimal state comparison for two initializers.
An explicit Generator advances its own complete state, does not change the global CPU state, and replays the complete array after its state is restored.
On the frozen baseline, the candidate's actual calls instead showed, for both, that the private state did not advance, the global state changed and replay failed.
The original `candidate-cpu-baseline.log` is retained; that failure is not relabelled as a result of the later fixed source.

`torch.nn.init.uniform_` / `normal_` now call the existing native Generator owner through the installed `rand` / `randn` factories.
The factory is responsible for checking the Generator type, device and CUDA ordinal; no PRNG, global reseed, NumPy sampling or cross-device fallback was added.
The output follows the target tensor's actual dtype/device; float64 samples in float64, and the float16/bfloat16 scale and offset are first computed in float32
and then written back in the target dtype. This batch verifies the same Python object, the trainable/stop-gradient state at construction, and the existing retained-view write-back.
Reversible `requires_grad_(False)` and the old graph of non-leaf tensors are outside this pass: a later independent probe found that the shared
write-back could still undo reversible freezing and turn non-leaves into leaves; that pre-existing gap is accepted separately through the later shared write-back fix.
Invalid std, non-floating targets, and reversed/non-finite/out-of-dtype-range uniform bounds are rejected before sampling.

## Verification method

The same generic function body of the new tests is executed by the real Torch of an independent interpreter and by the candidate.
The reference interpreter records its own `torch._C` binary hash, package versions, actual interpreter, GPU availability and test-file hashes,
and rejects `jittor` imports before and after the tests. The candidate-specific native draw observer runs separately;
it rejects NumPy sampling, checks the passed native Generator handle, reads the actual dtype of the random output, and
checks the real allocation ordinal with the driver pointer attribute before CUDA host readback.
Every candidate node runs inside a `forbid_backend_fallbacks()` scope.

All sample/gradient/state comparisons use complete arrays, not just the seed or statistics.
What is verified across frameworks is the state/device/in-place behavior contract; no claim is made that the two original PRNGs produce bit-identical samples.
Bounded uniform and non-default normal sample in practice and check the complete range/mean/std,
and cover the behavior that std=0 and equal bounds still advance the random stream.

| Check | Result | Scope |
| --- | --- | --- |
| Independent Torch CPU | 29 passed / 29 CUDA skipped / 4 native observer deselected | same generic tests |
| Independent Torch CUDA | 58 passed / 0 skipped / 4 native observer deselected | same generic tests, real CUDA |
| Candidate CPU | 84 passed / 61 skipped | new init, legacy Kaiming, native Generator, default Generator |
| Candidate CUDA | 149 passed / 1 skipped | same four files; only the node with two visible CUDA devices skipped |
| CPU core | 322 passed / 42 skipped / 1 existing xfail | native 98 + Torch 224; both modes exit 0 |
| Layout / full structure | layout passes; 1622 passed / 6 skipped | tests/structure + compat/tests/structure; source frozen before the code commit |

The initially added tests read CUDA state unconditionally in the CPU-only build, and the observer compared the public dtype as a native string by mistake;
both were fixed (available-device condition and dtype object comparison respectively) and re-run. The original logs of those test-fixture failures are retained;
no CUDA API was changed to accommodate the new tests.
The first retained-view normal reference failed because it used a contiguous target as the expected value, since Torch CPU normal's contiguous/noncontiguous paths consume different sequences;
the final regression replays the same retained-view layout rather than requiring different layouts to consume the same sampling sequence.

## Reproduction and evidence

Unversioned directory: `$JITTOR_LAB_ROOT/_state/nn-init-generator/20261015/`.
Its `probe.py`, oracle metadata, CPU/CUDA environment allowlist scripts, original failures and final logs,
source freeze and final manifest bind the actual measurement source and the raw-file SHA256.
The two modified files at measurement time were not yet committed; after the commit they were checked file by file against Git blobs, and this is not relabelled as a historical clean-source measurement.
All caches use task-private HOME/JITTOR_HOME/TMPDIR, CPU and CUDA separate, with the first compilation serialized. The core runner's native/Torch sessions each use their own directory,
starting from this task's completed CPU compile cache of the same tree and configuration, then writing independently.
Other processes were present on the GPU; this round claims correctness only and compares neither speed nor whole-process peak.

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <candidate-python> -m pytest -q \
  compat/tests/torch/test_nn_init_generator.py \
  compat/tests/torch/test_torch_init_package.py \
  compat/tests/torch/test_generator_native_contract.py \
  compat/tests/torch/test_default_generator_contract.py
```

Actual execution must also follow the environment manual to set an independent state directory, explicitly hide CUDA on CPU or enable real CUDA,
and set strict fallback and the installed toolchain; the line above lists only the maintained test selection.

This batch does not grant Generator support for all initializers: the native Kaiming helper still rejects it explicitly, and
existing NumPy/approximate paths such as trunc_normal_ are unchanged; it also does not grant a complete C-extension Generator ABI,
integer-range accuracy, cross-framework RNG state interchange, or untested ROCm/NPU.
In this round's frozen-baseline CPU-only build, `torch.cuda.get_rng_state_all()` rejects, while real Torch returns an empty list at that point;
the new fixture collects only the state of actually available devices and does not grant equivalence of that CUDA API. A later main-line commit `97d7d478a48ecc80f13f2858f0bc8fa78e377480` independently fixed that gap;
this table and the original results keep the b196 measurement identity and are not relabelled as results of 97d.
The analogous independent Tensor.uniform_ / normal_ consumers will be fixed and accepted separately and cannot be covered by the passing of these two nn.init
entries; the original R33 PR rebase/author handling has still not been performed.

The full structure check was executed against the frozen state of the code and tests; this page was committed separately afterwards.

After synchronizing `1ecea7d5879761ec487d2497a25ef07bad47ae17`, the main tree independently reproduced the gap above,
with integrated code `661e1851e` and report `d8038c44b`; the central index and MANIFEST were registered afterwards.
Under that frozen state, CPU was 84 passed / 61 skipped, real CUDA (two visible cards) 145 passed / 0 skipped,
core 322 passed / 42 skipped / 1 existing xfail, and the full tests/structure plus compat/tests/structure
1622 passed / 6 skipped. The structure skips comprise two for the missing target accelerator, two declared skips, and two where that interpreter has no
native package installed, so installation-path ownership cannot be checked. The main tree already has later default Generator test changes, so node counts are not
merged with the b196 historical results. This table keeps the original measurement identity; the main-tree logs and the inventory of the independently recomputed 29 raw evidence files are in the
unversioned `$JITTOR_LAB_ROOT/_state/nn-init-main-integration-1ece/`.

# Default generator CPU state, private copies and device isolation

- Status: the default generator contracts bounded to CPU and real dual CUDA devices pass.
- Date: 2026-10-15.
- Baseline: modifications frozen after syncing `7a1e66b3858279b40eaec0c4895c6059ce8eb8db`; the actual pending-commit source hashes are in `source-freeze-release.json` below, with per-file Git blobs mapped after the commit, and historical results are not relabelled as clean-baseline measurements.
- Owner: Native RNG and Torch frontend maintainers.
- Review when: default/explicit generators, CPU state encoding, the CUDA seed owner, pickle, or installation dependency order changes.

## Problem and fix

Independent binary Torch `2.11.0+cu128`'s default generator and an explicit object have the same public type;
the default object's `manual_seed` and `seed` modify only the global CPU stream. An object obtained by
pickle/copy holds a private stream and can no longer operate on the process default stream. Default and
private generators of the same backend can swap complete state.

The minimal actual reproduction on the synchronized baseline did not match: `default_generator` was another proxy class,
both seeding methods reset the CUDA state; seeding a pickle copy modified the global CPU state;
and default/private CPU state restoration in either direction was rejected. The raw logs for CPU and actual CUDA are both retained.

A public Generator owner is now defined by `compat/torch/generator.py`. The default object holds a
view of the global CPU owner, while explicit and deserialized objects hold the original private native Generator.
Seeding the default object uses the existing `rng_manual_seed("cpu", ...)`, preserving
`torch.manual_seed`'s global seeding semantics. Seed validation completes before any state is modified, without changing the Python/NumPy streams.
The original `installers.tensor.Generator` continues to import the same class, preserving the old Python class lookup path.
Historical pickles of the old `_DefaultGenerator` proxy without complete state are outside the recoverable range;
such empty-state objects are not silently mapped to a resumable private stream.

The CPU Torch state shares the runtime's existing `JITTOR_RNG_1` full-engine encoding and strict parser;
a new native private getter provides that encoding. The native `jt.Generator.get_state()`'s existing frame,
CUDA offset and Philox paths are retained. Reading old explicit CPU frames (with or without a stream tag) is still checked,
and rejection of CPU/CUDA wrong devices, corruption and trailing bytes remains with the original owner.
No new PRNG, random compute kernel, Python state parser or cross-framework byte conversion was added.

## Actual verification

The candidate CPU explicitly hides CUDA; CUDA uses an actual RTX 4090, NVCC 12.2.140 and strict kernel math,
with the final regression exposing both devices. Unit caches and benchmarks are separate; first compilations run serially.
The new contract checks all execute inside a scope that forbids fallback. Correctness runs grant no performance conclusions.

| Check | Actual result | Scope |
| --- | --- | --- |
| Independent Torch CPU | 20 passed / 3 CUDA skipped | The original test body for the same 23 contracts |
| Independent Torch CUDA | 23 passed / 0 skipped | Binary extension, Jittor import refused |
| Candidate CPU expanded regression | 83 passed / 39 skipped | Default, explicit, fork, sampler, installer owners and ten structural fixtures |
| Candidate actual CUDA expanded regression | 112 passed / 0 skipped | The same regression group plus existing dual-device entries; every visible CUDA stream has a real continuation check |
| native CPU random/generator | 14 passed / 21 skipped | CPU; the remainder are CUDA/optional-Torch nodes |
| native CUDA random/generator | 33 passed / 2 skipped | Actual CUDA, both old CPU frames, native counter retained; optional-Torch nodes not executed |
| CPU core | 322 passed / 42 skipped / 1 existing xfail | native 98 + Torch 224, both modes exit 0 |

The new contracts cover copy/deepcopy/pickle copy independence, full-width seeds, default/private CPU state
interchange, random operations not read back before saving, the real random sequence after restoration, and protection of the actual
CUDA subsequent sequence by all three default-object methods. Each comparison covers the complete array and state; it is not only a seed comparison.

The independent reference reuses all function bodies and parameters of the new tests. It only removes the candidate's Jittor import and fallback/sync
fixtures, replacing them with a real Torch state save/restore fixture; every run refuses the Jittor import and checks the real
`torch._C` extension. The converted script and the original test SHA256 are retained separately, with no replacement of the computation,
expected values or devices in the test. The cross-framework comparison covers these behavioural contracts and does not require different PRNGs to produce identical sample values.

The first Python owner merge triggered a real import cycle and was re-verified after being moved into a dependency-neutral Generator module.
When the native CPU get_state encoding was initially changed directly, the native CUDA regression found that the CPU handle's existing
CUDA counter path lost the continuation position (1 failed / 32 passed / 2 skipped). After finally retaining the native
public frame and providing a separate Torch CPU getter, the same regression gave 33 passed / 2 skipped.
The original failure was not deleted and does not count as a final pass.

When expanding to the compat structure suite, two old fixtures also failed: they still required the deleted global reseed entry,
and a minimal NativeModule stand-in lacked the existing child-view protocol. Both failures were reproduced independently with stdlib unittest
on archived source at the exact 7a baseline. The fixtures now check private draw delegation
without global reseeding, and load the real native child-view descriptor to check copy binding;
no NN production code was modified. The NumPy re-export dependency of the shape helper was also explicitly preserved when moving out the Generator,
and module import plus the installer owner are re-executed in the final regression.

## Evidence and boundaries

Unversioned directory: `$JITTOR_LAB_ROOT/_state/default-generator-integration-7a1/`.
`sync.json`, the baseline/oracle probes, both source freezes, the final CPU/CUDA/core logs,
the original failures and `oracle-test-binding-release.json` preserve the actual identity and verification scope.
`integration-manifest.json` records the final tool/structure checks, original file SHA256 values and the final commit mapping;
the directory references only allowlisted fields and does not collect the whole process environment.

Existing Ruff diagnostics on the source files were compared item by item with the baseline, adding no new diagnostics; the new Generator owner and new tests
pass Ruff completely. The final layout, manifest and complete structure gates are recorded separately in that manifest.

This batch only fixes default object type ownership, CPU state interchange, private copies and device seeding isolation.
It does not grant the complete C-extension ABI, all Generator methods, integer sampling precision, cuDNN RNN
full-width seeds, dual-CUDA/Ascend/ROCm scenarios beyond this table, or interoperable RNG bytes across Torch versions.
R33's author rebase and the maintainer's final disposition are still not done and are not closed by this report.

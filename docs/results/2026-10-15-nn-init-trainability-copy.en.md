# Initialization trainability flags and no-grad retained-view write-back

- Status: the bounded CPU / real single CUDA contract passes; the native CUDA expand memory gate is listed separately.
- Date: 2026-10-15.
- Owner: Torch initializer / native storage maintainers.
- Baseline: after `bb910319abe420c93b641bc1ac64c48bfac09efa`, the delivered Generator/Tensor historical evidence was preserved; synchronized first to `8929633bef1e11294c3f0dfbef01bab9c0e015fe`, with the measurement frozen tree at `b0bd14bccf6d778305a5dd5e0d2716de2659ef09` plus per-file-recorded uncommitted modifications. With no tests in flight, `14aeca5df0bf841e99d6241e06b076e34ef08c05` was then integrated as `369c5c255921fc117d216f0f99266246b55f1ed6`, with the seven measurement owner files byte-for-byte unchanged.
- Code: `cc5fc5eeabcb5dc66e3fcba2e76b7fc014d3d31f` (initializer); `61c67b67a4d864c39810d47bfddf96d605a2d295` (native / Tensor writeback and the final tests).
- Review when: VarHolder in-place write-back, view provenance, storage, stream or the initializer owner changes.

## Actual break

First compared against an independent binary Torch `2.11.0+cu128` on CPU, then reproduced on the candidate baseline:
nine initializers each applied to the reversible `requires_grad_(False)` of a Tensor / Parameter, and the candidate wrongly unfroze 18/18; real Torch kept False.
For initialization of the real non-leaf tensor `root * 3`, the candidate rebound it as a leaf and lost the root gradient; real Torch kept the non-leaf and the complete gradient of value 3.
After the first round's contiguous non-leaf and frozen-state fixes passed, three strict leaf-view tests still failed.
A plain `with no_grad(): view.copy_(...)` independently reproduced the same break, which was traced to the shared storage write-back and was not closed with an xfail.

After the physical stride fix, the first CPU round still had 17 failures: `.T` with the default `transpose_storage_view=0` is a materialized view that keeps its provenance.
The old fallback replaced the Var and graph identity of the base / view / sibling, and the root gradient was None.
Diagnostic results with the physical transpose flag temporarily enabled were used only for attribution; the formal acceptance asserts the flag is still 0.
These original failures, the staged patches and the source copies are retained and not rewritten as a final pass.

## Implementation

`nn.init._assign` reuses the Tensor storage write-back under no-grad; the native initializer runs first on a temporary value of the target's device / dtype and then writes back to the original target.
A native stopped marker is not substituted for the actual public requires_grad, and no existing non-leaf graph is start_grad'ed again.
The new tests actually cover construction-time freezing, reversible freezing, re-enabling, real leaf / nonleaf and Parameter.
The earlier Generator report covered only the construction state; this page is later new acceptance and does not backfill the old scope.

For targets with positive stride that are provably non-overlapping, native `_copy_into` calls the existing backend_copy in contiguous runs, keeping the dtype/device/stream owner.
Directly expanded / overlapping targets are rejected before writing; self-copy of the same layout leaves values untouched.
For aliases whose address spans intersect but whose elements are provably disjoint under the stride modulo residue, a snapshot is taken first by the existing device CopyOp.
Sparse aliases whose safety cannot be proven are rejected explicitly; no claim is made to cover Torch copy_ for all arbitrary strides.

The new private `_copy_view_into` handles only no-grad write-backs that have materialized retained views.
It reuses the original view steps / SetItem mathematics, prepares the root and all materialized sibling values in an independent temporary buffer, and writes the original buffer only after complete verification.
The temporary SetItem operator propagates values only; the original root / view / sibling Python holders, native graph-node identity, data pointers and old graphs are all preserved.
Where a cross-synchronization may release the GIL, the original holders are explicitly held.
Physical siblings are updated through the shared buffer; a read-only expanded sibling has its repeated axes compressed and then refreshes its unique backing storage, while a directly expanded write target is still rejected.
The caller first completes pending readers / old clones, and the CUDA stream and CPU→CUDA transfers continue to be awaited and submitted by the existing backend_copy; there is no host numerical fallback and no separate AD implementation.

## Verification approach

The two new maintained test files run the same generic nodes on independent real Torch and on the candidate, comparing complete arrays and gradients; the four native-only observer functions run on the candidate only.
The nine initializers are uniform_ / normal_ / zeros_ / ones_ / constant_ / kaiming_uniform_ / kaiming_normal_ / xavier_uniform_ / xavier_normal_.
The copy tests cover slice / step / transpose / nested, leaf / nonleaf base, float16 / bfloat16 / float32 / float64, CPU source→CUDA target, source alias, rejection atomicity, materialized siblings, expanded read-only sibling and the original native graph-node identity.
Candidate full tests run in a forbid-fallback scope and record the actual native allocation / ordinal; the seven owner file hashes are re-checked before and after the formal measurement.
Bit-identical random samples across different PRNGs are not required; the reference verifies the initialization and autograd behavior contract.

| Check | Actual result |
| --- | --- |
| Independent Torch CPU, same two final test files | 114 passed / 114 CUDA skipped / 8 native-only deselected |
| Independent Torch CUDA, same two final test files | 228 passed / 0 skipped / 8 native-only deselected |
| Candidate CPU, final expanded initialization / copy regression | 197 passed / 192 CUDA skipped; fallback 0→0 |
| Candidate CUDA, same expanded regression | 394 passed / 0 skipped; fallback 0→0; default transpose flag 0 |
| Native CPU existing storage / view regression | 16 passed / 6 CUDA skipped |
| Native CUDA storage / view / graph capture regression | 29 passed / 1 expand memory-budget failure; fallback 0→0 |
| Native GraphReplay strided-result private copy consumer | CPU 1 passed / CUDA 1 passed; fallback 0→0; CUDA original kernel-count assertion passes |
| Canonical CPU core | 322 passed / 42 skipped / 1 existing xfail, native / Torch both modes exit 0 |

Formal test selection:

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <candidate-python> -m pytest -q \
  compat/tests/torch/test_nn_init_trainability.py \
  compat/tests/torch/test_tensor_copy_strided.py \
  compat/tests/torch/test_nn_init_generator.py \
  compat/tests/torch/test_tensor_rng_initializers.py \
  compat/tests/torch/test_torch_init_package.py
```

Isolate HOME / JITTOR_HOME / TMPDIR per the environment manual; hide CUDA explicitly on CPU, then serially do the first real CUDA compilation.
The candidate unit cache is separate from the benchmark, the device already carries external load, and only correctness is verified, with no throughput or peak-performance claim.
ROCm / NPU are not verified.

The only failure of the native CUDA expanded regression is the existing `test_expand_does_not_materialize_when_it_is_consumed`:
`(ones((4096,1)).expand(4096,4096)*2).sum()` is numerically correct, but the allocator grows by 68,157,440 bytes, exceeding the original assertion's 16,777,216 bytes.
A single-node re-run on this source fails the same way; a clean detached worktree of remote `14aeca5df` was then created with a fresh private cache and the same physical CUDA / toolchain,
actually recompiled, and ran the same maintained node, with the same 68,157,440-byte growth and fallback 0→0.
This is an unfixed existing CUDA expand memory-budget problem; the expanded suite is not recorded as fully green, and the original assertion is not changed.

The full `tests/structure compat/tests/structure` run was 1620 passed / 6 skipped / 2 failed;
the two are only that the old `tensor-rng-inplace` report is not yet registered in this independent tree's central results index / MANIFEST.
The other structure items pass; the central files belong to the integration maintainers, and the full re-verification after registration is separate main-line evidence.
The repository layout check passes; the targeted documentation / private-path check for the new report was 18 passed / 1 orphan-registration node deselected (the full run above retained that node's failure). Ruff passes on the two new test files, and the 21 pre-existing Ruff diagnostics on the two production files are item-for-item equal to `bb910`.

## Boundaries and evidence

This fixes no-grad initialization / copy write-back and grants neither all grad-enabled copy_, nor arbitrary overlapping views, nor all initializers, nor the complete R33 Generator ABI.
An independent CPU probe also confirmed that the old non-leaf rebinding difference of grad-enabled plain copy_ is the same on the old cast path and on the new path that removes the no-op same-dtype cast; it is kept as an existing separate boundary and the AD is not changed.
Materialized view write-back needs temporary values for the root and retained siblings, so no zero-temporary-memory claim is made.

Unversioned evidence: `$JITTOR_LAB_ROOT/_state/nn-init-generator/20261015/trainability/`.
`probe.py` / `nonleaf_probe.py` / `view_copy_probe.py` and the oracle/candidate baseline logs retain the original break.
`strided-copy/` retains the raw failures, patches, source copies and final CPU/CUDA metadata of the physical / materialized / expanded stages.
`source-freeze-expanded.json` binds exactly the seven owner files of the historical dirty source; the formal results cannot be relabelled as measurements of a later clean commit.
The independent oracle metadata binds the actual interpreter / Torch binary extension / test hashes; the shared Torch installation was not modified, and this single binary hash is not claimed as the provenance of the complete installation package.
`manifest.json` seals the raw files' relative paths / byte counts / SHA256, and `strided-copy/source-final-git-proof.json` verifies one by one that the seven final files equal the actual measurement hashes above and the `61c67b67a` Git blobs. The central index / MANIFEST are registered by the integration maintainers.


## Main-line integration re-verification

After synchronizing remote `bda0f82df2eb4322a3c81eee2c36d31597a9237d`, the main line ran the three original minimal reproductions separately
(reversible freezing, non-leaf initialization, view no-grad copy) and all failed for real.
The three delivered commits were replayed in order as `53ae4b44e`, `6d758252a`, `d07310011`, and after the central index and
MANIFEST registration the frozen clean state was `0e85bafad1df77d5b3c200b311db8888b256215c`.
All seven owner files are byte-for-byte equal to the original sealed measurement and the `61c67b67a` Git blobs; all 122 original
raw files were re-hashed, totaling 675,935 bytes, and the historical results were neither rewritten nor rebound to the main line.

This main-line round completed the new CPU core compilation first, then the new real CUDA core compilation, with an independent unit cache per run;
real CUDA used one RTX 4090, candidate GCC 11.5.0 / NVCC 12.2.140 / cu122.
The following results are this round's clean main-line measurements and are kept separate from the subagent dirty-source measurements above.

| Main-line check | Actual result |
| --- | --- |
| Five-file initialization / copy CPU | 197 passed / 192 CUDA skipped, 34.99 s |
| Same five files, real CUDA | 394 passed / 0 skipped, 182.75 s |
| Expanded factories / in-place / autograd, eight files, CPU | 203 passed / 153 CUDA skipped, 9.25 s |
| Same eight files, real CUDA | 379 passed / 0 skipped, 20.62 s |
| Native storage / view / graph capture plus the exact GraphReplay node, CPU | 17 passed / 14 CUDA skipped, 12.13 s |
| Same native set, real CUDA | 30 passed / 1 existing expand memory-budget failure, 63.35 s |
| Canonical CPU core | 322 passed / 42 skipped / 1 existing xfail, both modes exit 0 |
| Full structure gate | 1622 passed / 6 skipped, 210.17 s |

The 14 skips in the CPU native set comprise six storage CUDA fixtures and eight CUDA-only graph capture nodes;
they are not recorded as CPU capture support. The original maintained GraphReplay node
`TestGraphReplay::test_a_strided_result_is_replayed_once` passes on both devices, and the CUDA original kernel-count
assertion is retained.

The only native CUDA failure is still
[KI-CODEGEN-005](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/manuals/known-issues.md#ki-codegen-005-cuda-full-sum-materializes-a-pending-expanded-elementwise-consumer):
the node is numerically correct, the actual growth is 68,157,440 bytes against the original budget of 16,777,216 bytes; the same reproduction as the subagent's clean remote baseline,
so the whole native CUDA set is not recorded as fully green, and the assertion is neither modified nor xfailed.
Layout, MANIFEST, whitespace checks and Ruff on the two new test files pass.

This round's unversioned records are in `$JITTOR_LAB_ROOT/_state/native-copy-main-integration-d9/`:
the three `baseline-*.log`, `source-freeze-main.json`, `agent-evidence-readonly-verification.json`,
`cpu.log` / `cuda.log`, `shared-*.log`, `native-*.log`, `core.log`, `structure.log`, and the
exact test selections `run-unit.sh` / `run-shared.sh` / `run-native.sh`.

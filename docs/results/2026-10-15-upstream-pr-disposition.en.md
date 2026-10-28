# Upstream PR disposition and NumPy ABI measurements

Status: 2026-10-15 read-only review of official PR states and current source disposition; the historical Linux CPU NumPy 1/2
regressions remain bound to the original measurement baseline below, and this round neither reran the numerics nor changed any GitHub PR state. Windows MSVC,
the CUDA 13 packaging stack, and new Ascend acceptance are outside this round.
Maintainer: Jittor core and compatibility-layer maintainers.
Historical initial frozen baseline: `f02e7aec0b43daf1b275a97eab340de7c4e917d3`, synchronized from
`origin/2.0-refactor` followed by a new independent branch, retaining the earlier checkpoint and graph-replay branch.
Historical numerical integration baseline: `a1a3f44afdcec9d0ce3670ff77e78c09f16b80a2`; the tree was frozen after the original baseline tests were fully
wrapped up, fetched and fast-forwarded, with three local test/report files retained, without stash or discarding changes.
Current read-only review baseline: `4a4a480e628dae634a9d660f320c3aa957341e99`. The new task first confirmed the tree was clean with
no local tests, then synchronized `3ba5bb7714ad93a7a1cef6a67d2f85a234b412a6` and created another branch;
the earlier TRELLIS/TP refs are retained. The official PR snapshot time is `2026-10-15T01:56:19Z`. It was then confirmed that no local tests were running, and after
backing up and hashing the only uncommitted report, the tree was losslessly fast-forwarded to 4a4; the Windows
official API was re-checked at `2026-10-15T02:15:23Z`. This round only updates this report; the original NumPy/shape/jtune
tests and historical logs are not rewritten, and no new computation, binding, adapter, or runtime acceptance was added.
Recheck conditions: PR head/base updates, or changes to the NumPy C-API, shape code generation, the RNG owner, FSDP FULL,
or the CUDA wheel resolution protocol.

## Official state and source disposition

The existing reports and PR API snapshots were read first, then the base/head/state of the ten PRs were re-fetched from the official GitHub
REST API, along with the three pages of 209 files for #18031 and the complete files patches for the other seven small PRs.
The state, head/base refs of the ten PRs are consistent with the old snapshot: nine open, #18042 merged.
The SHAs below are from the official PR responses; a PR's historical base is not treated
as the latest tip of the target branch. The complete responses, HTTP headers, and supplementary Git diffs are stored in the external experiment area.
The 209 files of `#18031` were fully re-fetched across three pages this round; `#18042` reuses the already paged 277-file
snapshot from the original audit, and this round confirms its head is unchanged and re-checks merge ancestry. The
3184 files of `#18028` is the API's changed_files, exceeding the REST files endpoint's 3000-file limit; the complete
content comparison uses the already fetched Git commits. The current Git default rename detection counts
`git diff --name-status 9eb696d9...e5cffe1a` as 2935 entries, while `--no-renames` gives
3717 entries (323 M, 2294 A, 1100 D). These counts use different conventions and none is passed off as the REST 3184;
the complete file list, stat, and merge-base are saved, and the first page was not treated as the complete diff.

| PR (official source) | Current state; base → head | Disposition for the frozen tree |
| --- | --- | --- |
| [#18031](https://github.com/jittor/jittor/pull/18031) | open; `2.0-refactor/e184a85f` → `cgq_transformers/e5cffe1a`; 209 files | The remaining diff must be reorganized on the latest 2.0-refactor. Multimodal, FSDP, native distributed, and RNG have been integrated to differing degrees; the original diff carries both old directories and later owner conflicts, so it cannot be granted blanket approval by title. |
| [#18028](https://github.com/jittor/jittor/pull/18028) | open; `2.0/9eb696d9` → same `e5cffe1a`; 3184 files | Not applicable to the current target base; the author/maintainer should mark it as a duplicate review entry to #18031; keep the head and review only the remaining functionality on the latest refactor base. |
| [#18033](https://github.com/jittor/jittor/pull/18033) | open; `master/06f5d3d2` → `1cf2a4e1`; 3 files | Already covered: `python/jittor/contrib/loss_ssl/` and `tests/nn/test_loss_ssl.py`; integration commit `8224841d4`. Existing CPU numerical, gradient, and joint-training cases pass, and the old top-level `loss_ssl` directory is not re-created. |
| [#18029](https://github.com/jittor/jittor/pull/18029) | open; `master/06f5d3d2` → `ef7619fb`; 1 file | Already covered: `python/jittor/tools/jtune.py` uses `.format(cmd=cmd)` and explicitly reports command failure. The existing unknown-command regression retains the verification of real historical error messages. |
| [#18024](https://github.com/jittor/jittor/pull/18024) | open; `master/5e412c63` → `2eb4233c`; 1 file | The original patch does not apply to the current implementation: `backends/cuda/kernels/cufft/cufft_fft_op.cc` already uses `CufftPlanKey`, with no remaining `std::array` use site. The source was not modified for a missing unused include; this source-level judgment is not a new GCC12/CUDA measurement. |
| [#18022](https://github.com/jittor/jittor/pull/18022) | open; `master/3fe57d77` → `b7f16ef8`; 5 files | The NumPy ABI is covered and historically passed real CPU verification (original measurement baseline); the byte count is currently derived from the dtype, avoiding the patch's descriptor-private offset read. Its normalization part is also covered by the more complete stable-normalization owner; the old 1.3.x version-number change does not apply to 2.0. |
| [#18021](https://github.com/jittor/jittor/pull/18021) | open; `master/3fe57d77` → `fa80a654`; 3 files | The property and variable-length dimension annotations are covered; the current `Tuple[int, ...]` is retained and the PR's fixed single-element `Tuple[int]` is not ported. The native return type is still NanoVector, see the actual binding evidence below. |
| [#18020](https://github.com/jittor/jittor/pull/18020) | open; `master/3fe57d77` → `c934ac2e`; 3 files | The old packed `data.gz`/`data.o`/`vdp` paths no longer exist in the current build/core source, so the old compiler patch cannot be ported as-is. Windows MSVC import still awaits real Windows build, import, and numerical acceptance; Linux cannot close this hardware item. |
| [#18041](https://github.com/jittor/jittor/pull/18041) | open; `2.0-refactor/cded3102` → `7ccc796c`; 27 files | The uv lock/dev group already exists; the original f02 baseline still lacked CUDA13, and the final a1 baseline has integrated the fixed CUDA13.0 standalone scheme described below. The PR's mixed CUDA12 runtime, dual CCCL, and Cu12/NCCL ranges still differ from that scheme and need independent evaluation, and cannot be recorded as the PR being accepted as a whole. |
| [#18042](https://github.com/jittor/jittor/pull/18042) | closed/merged; `2.0-refactor/cc193436` → `73bffaed`; 277 files | Already merged: official `merge_commit_sha=659e6a47f42466ba407f2d453c0d1224ba5ca0c6`; the Git ancestry check confirms it is an ancestor of the frozen baseline. Importing its code does not automatically remove subsequent documented limitations. |

The initial official branch-ref API confirmed that the `2.0-refactor` tip at that time was `3ba5bb771…`;
the final Git fetch has synchronized the later `4a4a480e6…`. In the initial official response the
`2.0` tip was `88ae21d1fad87438dfd17815d9f0400e7a521d5f`, and the head refs of both PRs
are the complete `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`. Thus #18028 and #18031
are different-base review entries for the same source head, not two independent feature sets. The API mergeable for #18031 is
unknown/null and for #18028 false/dirty; only the response is recorded, and it is not treated as the reorganization having passed.
The merge commit of #18042 remains an ancestor of the current baseline. The "already covered / not applicable" judgments for the master-target PRs
refer only to the current 2.0-refactor; the old master versions that still need separate maintenance are not closed on that basis.

The integration owner additionally provides the completed `98da22192aa886c584f9d4aef73a45c3810a5fb7`
and `docs/results/2026-10-15-cuda13-toolchain-packaging.md`: pinning the five CUDA 13.0
toolchain wheels, C++17, the device-property API, exact `libcudart.so.13` linking, and
CUDA numerical/gradient verification after a real uv install; the original cu122 fingerprint and 23 regressions are retained.
That scheme has been integrated into the final a1 baseline, corresponding to the public [CUDA13 verification report](2026-10-15-cuda13-toolchain-packaging.en.md); it does not accept #18041's
overall relaxation of the Cu12/NCCL ranges, nor does it grant arbitrary CUDA13/runtime combinations.

The NumPy PR's `instance_norm`/`layer_norm` centered variance and low-precision FP32 intermediates
are already reflected in `nn/functional/normalization.py`'s unified `_ln_normalize`, stable custom backward,
and dtype-restoration path. The module calls that owner, avoiding restoring the old `nn.py` or reassembling
an implementation that only fixes forward. This run does not re-grant normalization's CUDA/NPU coverage.

## cgq's reorganization plan and the API superset that must be retained

Keep the original `cgq_transformers` ref and do not overwrite the latest tree with its complete files. It is recommended to create a new review branch on the latest
`origin/2.0-refactor`, first using
`git log --left-right --cherry-pick --no-merges origin/2.0-refactor...origin/cgq_transformers`
to list commits whose patch-IDs are not equivalent, then replay the still-missing functionality item by item against the target owners. Keep the author and original commit
provenance, and organize the old `refactor-wip/results/` evidence according to the current public results directory rules.
All dependencies need review, rather than simply reducing 209 files or 3184 files to the same batch of cherry-picks.

The actual Git merge-base at the 3ba freeze is `e184a85fe9d7704e8319627a2382cd3477ea35bf`;
`rev-list --left-right --count 3ba5bb771…...e5cffe1a…` gives 208/84, and after excluding merges and
deduplicating by patch-ID it is 205/72. **72 is the number of commits awaiting semantic review, not 72 missing features**.
Recomputation on the final 4a4 baseline gives ancestry 209/84 and patch-ID dedup 206/72, with the merge-base unchanged;
the two results are sealed separately. Of the 209 paths in PR #18031, 46 do not exist in the 3ba layout, including the retired refactor-wip
records, split tests, and possibly migrated RNG files; a nonexistent path does not imply a missing capability.

The concrete review units for the reorganization are as follows; keep the original head/ref, and it is recommended to form small commits group by group on the latest target,
without a one-shot overwrite import or automatic replay of all 84 old commits:

| Review unit | Current retention points and pending work | Reusable evidence |
| --- | --- | --- |
| Transformers multimodal/operators/adapter | Compare against the actual model families, all gradients, and training inputs; missing capabilities go native, Torch APIs go to compat, and only library-private differences enter an adapter. Old large-model/speed records cannot replace the current matrix | [Existing matrix](2026-10-15-transformers-model-family-matrix.en.md): 4.56.2, 8/374 model families and 2/46 tasks, nine tiny L3 scenarios each on CPU/CUDA, explicitly bound to the 8e388 measurement source |
| FSDP/Module/optimizer/checkpoint | Retain shared native FSDP math, FULL gather materialize, FQN/group conversion, refreshed optimizer shard binding, native Module strict/copy routing, and alias/lifecycle fixes | [FULL bf16 continuation](2026-10-15-bf16-fsdp2-checkpoint-continuation.en.md), [options](2026-10-15-checkpoint-state-options.en.md), [lifecycle fix](2026-10-15-fsdp-compat-lifecycle.en.md); the two compat failures in the original native FSDP report already have this follow-up fix and can no longer be described generically as unfixed |
| RNG/explicit Generator | Retain the CPU engine, CUDA seed/offset, per-ACL-device state, and synchronization/validation owners. The 64-bit manual_seed boundary and generator-aware randint/randperm still need case-by-case triage, and the existing native owner must not be overwritten by a second Generator/Philox owner | `src/runtime/rng_state.cc`, `random_generator.{cc,h}` and the curand owner; `tests/ops/test_random_generator.py`, the compat generator/RNG regressions, and the [RNG/FULL report](2026-10-14-checkpoint-rng-fsdp2-continuation.en.md) |
| Native training and engineering files | Compare against the existing process-group/launcher/DDP, ZeRO1, TP, FSDP, and checkpoint public entry points; do not backfill old AGENTS, directories, MANIFEST, results boards, or another set of collective/optimizer math | The real CPU/single-machine two-rank reports itemized below; multi-node and arbitrary combinations still require separate acceptance |

The actual head source has three classes of differences that need explicit handling:

- cgq implements a complete CPU engine snapshot for `get_cpu_rng_state`/`set_cpu_rng_state`;
  **its outdated `_MISC_DETAILS` "seed-only" description must not be used to assert that the runtime only restores the seed**.
  Its added `set_cpu_seed(uint64)`, 64-bit seed range check, and lazy work flush should each be
  evaluated as behavior to retain. On the `4a4a480e6` baseline of this original read-only audit, compat
  `_manual_seed` still called the int-argument `jt.set_seed`, and the explicit Generator private path still sampled
  in NumPy; these were real gaps at the time. Later work independently reproduced and fixed the full-width seed,
  native handle factory passing, and two-device checkpoint seed ownership, see
  [native RNG acceptance](2026-10-15-rng-native-generator-contract.en.md). The reorganization should retain that fix,
  not restore the NumPy sampling path; different PRNGs are not promised to produce bitwise-identical cross-framework samples. The latest tree already has `jt.core.rng_state/set_rng_state/rng_manual_seed`
  and a unified runtime owner; during the reorganization the behavior should land on that owner, avoiding restoring the old global `eng`
  and `current_offset`. cgq's `JITTOR_CPU_RNG_V2` and the main tree's `JITTOR_RNG_1` are different
  opaque formats, cannot be read interchangeably, and cannot be converted to seed-only to resolve the format conflict.
- The latest tree retains the full CPU stream position, CUDA cuRAND seed/offset, and per-ACL-device
  seed/counter isolation; `set_rng_states` fully validates before modifying state, and CPU restoration does not reset device streams.
  cgq's deterministic setter is a no-op/getter fixed at False and cannot replace the current implementation that directly reads and writes
  the backend policy. Continue to guard the format/device rejection, lazy/async synchronization, and actual stream continuation regressions.
- The current `checkpoint_state` owner's `get/set_optimizer_state_dict`, FQN/group
  mapping, merged model+optimizer API, CPU offload, and rank-zero broadcast must all be retained.
  cgq's combined setter only loads the model, and its getter directly returns the optimizer's raw index dict,
  which cannot replace this superset. Retain FULL CPU-offload gather materializing before each rank discards,
  refreshing the FSDP optimizer shard binding after load, and the importable-but-UNIMPLEMENTED
  `DefaultSavePlanner`/`DefaultLoadPlanner` boundaries; do not count FULL success as SHARDED I/O.

The main tree already has [CPU/CUDA RNG and two-rank FULL continuation evidence](2026-10-14-checkpoint-rng-fsdp2-continuation.en.md).
The real two-rank bf16 Accelerate FULL continuation regression at `3d93c2ca91fe2a24cdb8945f3534e116489b72f8` has also been integrated into the final a1 baseline, see the [bf16 continuation report](2026-10-15-bf16-fsdp2-checkpoint-continuation.en.md).
The final tree additionally contains the StateDictOptions option/metadata/strict-load acceptance, see
the [checkpoint options report](2026-10-15-checkpoint-state-options.en.md). Retain these more complete owner contracts as well when reorganizing cgq;
this CPU PR audit did not rerun that CUDA acceptance.
As of this round's baseline, the old blanket statement that "TP/ZeRO/native sharded checkpoint are all unfinished" is outdated:

- [Native sharded training checkpoint](2026-10-15-native-sharded-checkpoint.en.md) already has actual CPU and two-rank CUDA save/restore of native
  Module/optimizer/four kinds of RNG, BF16 raw bits,
  completeness/gap rejection, and independent reference continuation; this is checkpoint storage capability, and training sharding is not claimed on that basis.
- [ZeRO1](2026-10-15-native-zero-optimizer.en.md) has verified actual owned moments, multiple groups,
  accumulation/DDP deduplicated reduction, CPU/single-machine two-rank CUDA, and continuation into new objects; parameters/gradients are still replicated,
  ownership/world fixed, and it cannot be written as ZeRO2/3 or an arbitrary backend pass.
- [Native TP](2026-10-15-native-tensor-parallel.en.md) and
  [BF16/AdamW follow-up acceptance](2026-10-15-native-tensor-parallel-adamw-bf16.en.md) already have real
  Column/Row local parameters, communication-adjoint gradients, an independent complete Torch reference, and fresh resume;
  the subsequent shared Linear re-verification keeps the frozen BF16 budget; after the latest tanh integration, the real CUDA/NCCL
  two-rank re-verification of 13 items each also passes, still limited to the publicly declared uniform Linear shard combinations.
- [Native FSDP](2026-10-15-native-fsdp.en.md) already has real FP32 parameter/state sharding, FP32/BF16
  two-rank training and fresh-launcher exact continuation; the [saved-output tanh follow-up verification](2026-10-15-tanh-fsdp-bfloat16.en.md)
  after pinning the original fixture/budget at `cc45f12a216af639faf1e7bd3fc2ff916ce16653`, all 80/80
  BF16 gradient arrays match the independent complete Torch bitwise, and all 368 fresh-resume arrays are exactly identical.
  The independent real Torch FSDP2's 396 arrays are also bitwise equal to its dense oracle. The original report's 11.2242%
  small-gradient difference is retained as an old-source result and is no longer a current unfixed defect; the actual CUDA/F0 evidence is in the follow-up report.

None of these sub-items is evidence from two real hosts. Multi-node, arbitrary FSDP/TP/ZeRO combinations, world-size
migration, complete Torch SHARDED planner, and each undeclared device/precision range continue to be retained independently.

## Reuse boundaries of this round's review

The #18033 functionality is in `jittor.contrib.loss_ssl`, and #18029 is in the current jtune's explicit
`AssertionError("unknown cmd: {cmd}".format(cmd=cmd))`; the current 2.0 public layout is retained,
without copying the old `jittor.loss_ssl` directory. #18022's versioned NumPy CopyInto slot with dtype-derived
byte count, stable normalization, and #18021's property/`Tuple[int, ...]`
are still in the current owners. The original generated bindings' runtime NanoVector and static tuple approximation
must continue to be distinguished; this round does not re-grant any NumPy ABI binary pass.

Of the 13 runtime/test files compared byte-for-byte against the delivered history regression `0cce2377482083282b5d81aee1df43c30fe57b95`,
12 are completely identical. The only different one, `py_converter.h`, only adds GradCallback's `python_callable` recording; this difference is also fully retained, and it is not described as the whole file being identical.
This comparison proves that the specific source/tests cited by the report remain reusable, and **does not replace actual NumPy1/2 execution on the new baseline**.
The Linux numerical results in the next section keep their original measurement date and the a1 baseline.

The old packed-data path of #18020 still has no current porting point. The newly added Windows workflow and
`tools/release/check_windows_msvc.py` require real Windows, cl.exe, an out-of-repo installed wheel,
actual compilation/training, and zero fallback; a Linux negative guard cannot produce a passed certificate.
The official contents API retrieved `windows.yml@4a4a480e6…`, byte-identical to the current file;
its SHA-256 is `73e028e312b178404a7de928b90a39adf025c98df6b1df2e3578053fff2ca543`.
The runs API specifying `windows.yml` returned HTTP 404, which alone does not imply zero runs. A supplementary check of the public
workflow catalog returned HTTP 200, listing only three dynamic Copilot workflows; the whole-repository runs API
returned HTTP 200 with `total_count=7` and all seven entries returned, all Copilot, with no Windows build;
the runs API limited to `branch=2.0-refactor` returned HTTP 200 with `total_count=0`. Within this visible scope there is
no Windows job, source-SHA build log, or passed artifact that could grant R16, and the workflow configuration or Copilot success cannot be treated as MSVC passing. Only the public
API results visible at that time are recorded here, without claiming that other private systems or future runs do not exist. The maintainer must provide real Windows
results for the specified source SHA, the raw compilation log, and the tool's passed JSON before this hardware boundary can be closed; this round did not trigger the workflow.

## Historical NumPy and real generated-binding verification

Using an independent Python 3.11.16 environment and pytest 7.4.4; NumPy 2.4.6 was installed into a task-private target,
without modifying the shared environment. NumPy 1.26.4 and 2.4.6 each use independent CPU JIT caches, with the first compilation serial.
All native executions explicitly pin the current worktree's `PYTHONPATH`, `JT_USE_CUDA=0/use_cuda=0`,
`JT_BUILD_NVCC_PATH=''`, and record the actual package files and active cache paths.

The initial probe failed its assertion that int64 defaults to int32 output, attributed to the existing
`auto_convert_64_to_32=1`, unrelated to the NumPy ABI. The original failure output is retained, and byte copying was then verified with explicit dtypes,
without changing the default conversion behavior. The final coverage includes bool, 8/16/32/64-bit signed/unsigned
integers, float16/32/64, and complex64/128 in C-contiguous, negative-stride, and Fortran layouts.
The new tests read back all 42 array groups from Var and RingBuffer respectively and assert the dtype and all element values;
the negative-stride/Fortran paths actually enter the C-API `PyArray_CopyInto`. Scalars and empty arrays are also checked.
An independent probe also verifies that Var.data keeps its data after the original Var wrapper is deleted.
C-API regressions that could crash are isolated through the project's `run_python_child(..., crash_isolated=True)`.

The real active `gen/pyjt_jit_op_maker.cc` `shape` getset entry calls:

```cpp
return to_py_object<NanoVector>(((GET_RAW_PTR(VarHolder,self))->shape()));
```

The runtime descriptor is `getset_descriptor`, the return value is `jittor_core.NanoVector`,
not a callable and not a Python tuple; `tuple(var.shape)` gives the expected integer dimensions.
The new regression covers rank 0, 1, 3, 10. `src/core/var_holder.h`'s
`@pyjt(__get__shape)`, `tools/build/gen_pyi.py`'s property generation, and the current `.pyi`
variable-length shape declarations are consistent. The Tuple annotation is a static approximation of the return value, **not an exact runtime type guarantee**;
if full accuracy is required later, the typing owner should model NanoVector/integer-sequence output separately, rather than
changing the native return API for the annotation or narrowing all NanoVector inputs together.

| Actual verification | Result |
| --- | --- |
| NumPy 2.4.6 private probe: 42 array copies, RingBuffer CopyInto, 0D/empty, view lifetime, shape | pass |
| New NumPy ABI + shape regressions, NumPy 1.26.4 | 2 passed, 0 skip |
| Same regressions, NumPy 2.4.6 | 2 passed, 0 skip |
| NumPy 2.4.6: existing array, SSL, jtune, pyjt binding protocol | 71 passed, 7 accelerator skips |
| New unknown-command and existing jtune regressions, NumPy 2.4.6 | 5 passed, 0 skip |
| Official CPU `core` runner, NumPy 2.4.6, native/Torch separate sessions | 322 passed, 42 skipped, 1 existing xfail; both sessions exit 0 |
| Final a1 baseline: same-source NumPy2 ABI/shape, jtune, SSL, stable norm regressions | 31 passed, 11 CUDA skips |
| Final a1 baseline: same-source NumPy1 ABI/shape regressions | 2 passed, 0 skip |
| Final a1 baseline: official CPU `core` runner, NumPy2, both modes | 322 passed, 42 skipped, 1 existing xfail; both sessions exit 0 |

The official runner first reported an unrecognized `--timeout=900` because pytest plugin autoload was disabled while the explicit timeout plugin was omitted,
and the original exit 4 log is retained; afterwards `PYTEST_PLUGINS=pytest_timeout,xdist.plugin` was set explicitly and the complete core above was rerun. That environment has no cmake,
and warmup explicitly reports that oneDNN is not enabled, so computation uses the native generic CPU kernels; oneDNN acceptance is not granted.

The complete structure on the original f02 baseline: 1430 passed, 6 skipped, 2 failed; the two failures correspond exactly to
this new report not yet being registered in the public `docs/results/index.md` and the generated `MANIFEST.in`,
which are the integration owner's responsibility. The first structure run was restarted under the explicit exit 3 protocol because jit_utils was updated,
and the original output of the cases not yet executed is also retained. These results are not recorded as all-green or as ACL failures.

The complete structure on the final a1 baseline after synchronization: 1439 passed, 6 skipped, 2 failed; the failures still correspond exactly
to this report's index and generated manifest registration, with no new source-contract failures.
The final layout, staged/unstaged whitespace checks pass. Both complete structure runs contain two environment skips for source-installation
resolution cases (this private interpreter has not installed Jittor), and they are not recorded as executed passes.
The raw logs of the final NumPy1/2 targeted runs, core, and structure are saved with the `a1-` prefix respectively.

## Reproduction and boundaries

The historical numerical experiment root is `$JITTOR_LAB_ROOT/_state/upstream-pr-disposition/`:
`api/` holds the official responses and complete Git file statistics, `private/numpy2/` holds the isolated NumPy installation,
and `logs/` holds the raw probe and test output. Sources, wheels, caches, and API JSON are not placed in the documentation tree.
This round's read-only API/headers, branch refs, Git relationships, and owner byte comparisons are stored separately in
`$JITTOR_LAB_ROOT/_state/upstream-pr-disposition-refresh/2026-10-15/`, unversioned;
the final `*-4a4.json` for `api/`, `snapshot.json`, the complete rename/no-rename file lists, the Git relationships, and the byte comparisons, plus `windows-api-summary.json`, can be reviewed directly. All network requests were GET only; no
comments or messages were sent, no PR/base was modified, and nothing was pushed to the remote. Run the following after pinning the source under test in each NumPy environment:

```bash
JITTOR_TORCH_SHIM=0 JT_USE_CUDA=0 use_cuda=0 JT_BUILD_NVCC_PATH='' \
  PYTHONPATH=python python -m pytest -q tests/bindings/test_numpy_abi.py
JITTOR_TORCH_SHIM=0 JT_USE_CUDA=0 use_cuda=0 JT_BUILD_NVCC_PATH='' \
  PYTHONPATH=python python -m pytest -q tests/tools/test_jtune.py tests/nn/test_loss_ssl.py
```

These conclusions grant only the executed Linux CPU NumPy/SSL/CLI/binding scope. The PRs' current open/merged
states and hardware admission are recorded separately; items requiring author rebase, maintainer marking as superseded, or real Windows/
CUDA13/NPU acceptance were not substituted for by sending GitHub comments or changing PR states.

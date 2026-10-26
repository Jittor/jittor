# PR review of the last 30 days and current-branch integration

- Status: integrated; the sub-item acceptances below are complete, CPU smoke was stopped at the user's request, no whole-suite smoke pass conclusion granted.
- Date: 2026-10-25
- Initial synchronized baseline: `ca56aa802c2acf9a22bca32d1bee3276381ed203`
- Integration runtime source reviewed: `0dcd03a9b9e87b8666112e72af405a075c67221b`
- Interrupted smoke source: `90fb7b6423d59b0b64332ddaaea6249cca05d9ed` plus the documented retained WIP and documentation overlay
- Owner: Jittor core and compatibility maintainers
- Review when: the PR heads below, target branch, capture/storage/optimizer/numerical ownership, or declared device scope change.

## Scope and verdict criteria

This "last 30 days" search used `updated_at >= 2026-09-25T00:00:00Z`, including open, closed and merged PRs, to avoid missing recently updated but earlier-created fixes. The saved GitHub REST snapshot has **25 PRs**, 13 open and 12 closed; only #18042 among closed has `merged_at`. Restricting to `created_at` on the same date gives **20**, and the additionally reviewed earlier-created items are #18028, #18030, #18031, #18033, #18034. PR status here is bound to this snapshot and does not infer future remote state.

Before finishing, the REST list re-query was rate-limited; among 294 pull heads reviewed via the Git protocol, only #18052 among this run's 25 was updated, and there was no new higher-numbered PR. Its two new commits were separately reviewed and merged; other heads unchanged. PR status and updated_at range remain bound to the initially saved REST snapshot; the Git reference review is not passed off as a status list refresh.

Each item compares the actual Git head, merge-base, commit groups and current source owners. A commit not in ancestry, differing patch-ID or a nonexistent old file path cannot alone prove a feature is missing. Historical reports are used only to locate and understand boundaries; whether the new integration passes depends on the gates and device evidence actually executed this round.

"Full merge/port" means this round accepted the PR's complete changes or complete standalone fix, without requiring preserving the original commit SHA; "partial merge" grants only the contents explicitly listed in the table. The rest stays in the original PR; a partial fix cannot be written as accepting the whole library, whole backend or whole PR. `29adfeb69` only narrows the new tests to the committed optimizer path, `33f640659` only fixes collection/CI-cache structure checks, `90fb7b642` only updates the release baseline and manifest; they did not change `0dcd03a9b`'s runtime payload. `509863fed` then only integrates upstream gate fixes in three test/helper files. All integration targets are the current `2.0-refactor`; this review did not change GitHub PR status or push to the remote.

## Disposition of the 25 PRs

| PR; snapshot head | Disposition | Current source evidence and reason against wholesale import |
| --- | --- | --- |
| [#18054](https://github.com/Jittor/jittor/pull/18054); `783968e1f` | full port | `34c8650b8` introduces the DataTrove CPU dataset/local pipeline's three scenarios, strict dependency-absence policy, tools and reports; the original head and integration tree are patch-ID equivalent. It adds no production adapter. Linux forkserver, CPU and historical external-library reports do not automatically grant Windows, accelerator or current complete nightly coverage. |
| [#18053](https://github.com/Jittor/jittor/pull/18053); `12bcfc60b` | partial merge | `cd2471909` keeps the pad integer-width/negative-clip and log1p, softplus, sigmoid precision/gradient fixes, and `e2f6924f7` fixes const StackVector access. Did not import the 512-file owned-carrier root migration, coupled symbol/nested tensors, NPU/FSDP/checkpoint/communication stack. The original report itself also retains PPO metadata strict-check and reconstruction acceptance gaps; this round does not claim complete verl PPO support. |
| [#18052](https://github.com/Jittor/jittor/pull/18052); `5c2349d05` → `26b26c5b8` | full merge, plus fixes | The original head is an ancestor of the integration tree, merged by `65c92ca41` with the accumulated training, storage, rematerialization, custom Function, captured gradient publication, CUDA interface and gate fixes. `f18a98d79` additionally fixes unbatched collate/default_convert and sized IterableDataset branching. The accumulated dependency chain is kept, not just the earliest Swift interface commit. Before finishing, `509863fed` was also merged, completing two subsequent environment-skip classifications and the cache path contract; the latest head is its second parent commit, 17 targeted tests passed. Full source merge does not mean the original historical CUDA/ms-swift matrix automatically becomes a current measurement. |
| [#18051](https://github.com/Jittor/jittor/pull/18051); `25823c28f` | covered by the new integration's implementation | Four typing issues are covered by #18052's `7556ce99c`, `1b25b3d39`: callback deque, native copy/leaf declarations, nox argument list, CUDA wheel tuple type. The original patch-ID is not equivalent, but the actual owners have retained these contracts; newer type declarations are not re-covered. Final typing results listed separately. |
| [#18050](https://github.com/Jittor/jittor/pull/18050); `80903a3f9` | partial merge | `cb666a7bc`, `2f9268f79`, `8aa8319ca` keep leaf reduction writeback, setup_context, reduce_scatter, record_stream, optimizer rebinding and mixed FSDP/plain parameter protection. Rejected the adapter `_zero.py` performance shortcut that directly calls the Tensor receiver's `_storage_is_contiguous`, `_mem_ptr_now`, `_copy_into` private interfaces; native.Var and public placement/device_id themselves are not grounds for rejection; continue using the current public bridge/native owner, and do not put DeepSpeed private metadata into public Tensor semantics. |
| [#18049](https://github.com/Jittor/jittor/pull/18049); `403b90455` | full port of a standalone fix | `c1de81a97` introduces the cgroup v1/v2 memory hierarchy budget, keeping affinity, CPU quota, user limits and dynamic native compiler worker policy. It does not replace container budget with host total memory, nor treat the source AST stub as the only verification; there is a real C++ worker-count harness. |
| [#18048](https://github.com/Jittor/jittor/pull/18048); `ff5ed494f` | partial merge | `d50d0a6f7` introduces exact index_fill, avoids duplicate factory cast and keeps the existing fresh-leaf/triangular op gradients; `f54a96228` introduces the external operation barrier, Tensor.data_ptr/Triton boundary and outer CUDA capture ownership, extended to StepCapture. Rejected wholesale import of vLLM/old owner overrides: the original PR explicitly documents NPU numerical divergence and an experimental CUDA H2D crash, and some adapters also access private ProcessGroup APIs. Complete vLLM, NPU and unselected kernels do not receive new acceptance. |
| [#18047](https://github.com/Jittor/jittor/pull/18047); `e13959314` | already covered | `244ebec9c` already has the dispatch membership fast path when no kernel is registered. Keeps the current subsequent scheduler optimization, does not re-import the old function body. |
| [#18046](https://github.com/Jittor/jittor/pull/18046); `dde61ffbf` | not merged | The corresponding source-signature memoization was once introduced by `2872b4b3c`, then reverted by `23eb049a3` due to same-process source invalidation and extracted-function regressions; the optimized source traversal is also not on the measured hot path. An existing deliberate revert cannot be treated again as a missed merge. |
| [#18045](https://github.com/Jittor/jittor/pull/18045); `19af33f35` | already covered | `92cb97515` already has per-dependency compile cache key memoization; at review time `src/utils/cache_compile.cc` matches the PR head. |
| [#18044](https://github.com/Jittor/jittor/pull/18044); `ee80c29db` | already covered and later enhanced | `d5c631a2f` already extends the Python 3.9/3.10 launch-origin cache; `3577b70b4` and later handle pre-3.11 `f_lasti` and the origin hot path. Keeps the newer implementation. |
| [#18043](https://github.com/Jittor/jittor/pull/18043); `bde3aeccf` | partial merge | `1101b06fa` adds holder guards for registered buffer/child/non-persistent buffer to GraphReplay and automatic PolicyStep, keeping in-capture state adoption. 310P SoC, NZ packed Linear, PFA prefill, raw ACL free synchronization etc. remain independent gaps needing real 310P/shared-backend acceptance; the PR's complete CPU smoke was unfinished at the time, and historical Qwen performance tables are not this round's measurements. |
| [#18042](https://github.com/Jittor/jittor/pull/18042); `73bffaed5` | officially merged | The official merge SHA `659e6a47f42466ba407f2d453c0d1224ba5ca0c6` is an ancestor of the integration tree. The Git head SHA not being directly in ancestry is no reason to re-merge; the actual proof object is the official merge commit. |
| [#18041](https://github.com/Jittor/jittor/pull/18041); `7ccc796ca` | partial merge | Existing `b507d0af9` already covers exception classification/provenance; this round `eed787100` adds SIGINT left to Python, post-initialization workers ignoring SIGINT, and exception type and traceback regressions; exception transport keeps #18052's newer JSON owner. uv/dev configuration already exists. Rejected the wholesale replacement of mixed CUDA12 runtime/dual CCCL/range packaging: it would delete the current standalone CUDA13.0 pinned toolchain, C++17 and runtime13 linking implementation. |
| [#18040](https://github.com/Jittor/jittor/pull/18040); `42dbb52ea` | overlaps #18041, not re-imported | Its head is in #18041's historical commit chain, reviewed as the same feature group. uv/error reporting already exist or were selected; CUDA packaging differences keep the rejection reason in the #18041 row, and the two entries are not counted as two new features. |
| [#18039](https://github.com/Jittor/jittor/pull/18039); `7282e9b59` | already partially covered, rest not wholesale merged | `d7ef14970` already selected the detached Function output gradient edge; the incompatible control-dependency exec-plan change was explicitly reverted by `ce7e54a16`. NPU device/deterministic/HCCL WORLD basics are already selected; the current real backward callbacks, post-accumulate and native distributed owners cannot be covered by old no-op callbacks/environment-variable branching. The old dual-NPU DeepSpeed report does not replace new acceptance. |
| [#18038](https://github.com/Jittor/jittor/pull/18038); `48c5da681` | keep the existing alternative, rest not imported yet | The current source-pinned DeepSpeed eager integration replaces the old finder/activation/scope adapter. Adagrad math and some GradBucket capabilities are still missing, not recorded as supported; the original patch also introduces old optimizer_state/Adam ownership, needing independent migration and device/state verification, and cannot override the current optimizer and user local storage identity modifications. |
| [#18037](https://github.com/Jittor/jittor/pull/18037); `00f81c168` | keep the existing alternative, rest not imported yet | Overlaps #18038's old adapter/Adagrad group, and additionally contains old NPU, RNG, HCCL plans and many historical probes. Keeps the current native RNG/HCCL/gradient owners; the original no-op queue_callback and the method table missing the current detach_/post-accumulate would produce regressions, and old binary hashes cannot grant current release acceptance. |
| [#18036](https://github.com/Jittor/jittor/pull/18036); `a79e57069` | already partially covered, rest not wholesale merged | `4ae54723f`/`4896af611` already have PEFT replay, NumPy indexing, CPU allocator safety fixes; `f1e7d1530`, `4233b3f97`, `f7ab3738b`, `223bbdd3f`, `fa077c459` already have ACL indexing, device, determinism, peer copy, per-device RNG. The original broad-exception swallowing, old RNG/threading owner and historical convergence matrix do not cover the new implementation. |
| [#18035](https://github.com/Jittor/jittor/pull/18035); `088593618` | required fixes already covered | `60e7f463b` already has fused AdamW batching, meta frontend, same-device ecosystem runner; `99bb20ddb`/`ae1c65311`, `52edb2b73`, `de2e2f1be` already have RMS rounding, AdamW defaults, launcher alias and python-config headers. Keeps the subsequent attention/output-backward/storage owners, does not import back the old refactor-wip results directory. |
| [#18034](https://github.com/Jittor/jittor/pull/18034); `cfe1b415e` | already partially covered, rest not imported yet | `f152bf849` already has native differentiable STFT, `2acafe9f3` already has the archive delay module fix. bool SparseCOOTensor/to_sparse/Whisper gate still missing, cannot be called supported. The original SparseVar files would delete the current validation, coalesce, CSR; the capability should be added independently and re-verified per current downstream admission, rather than replacing the current standard with the old CPU L2 definition. |
| [#18033](https://github.com/Jittor/jittor/pull/18033); `1cf2a4e19` | already covered | `8224841d4` already provides TotalCodingRate in `python/jittor/contrib/loss_ssl/` and `tests/nn/test_loss_ssl.py`; does not rebuild the old top-level loss_ssl. |
| [#18031](https://github.com/Jittor/jittor/pull/18031); `e5cffe1ae` | partial merge, the rest kept/rejected per owner | This round `1ddc66497` selects safetensors missing metadata returning None, Conv1d spatial sequence parameter normalization, Kaiser negative length rejection. Most of its BERT/multimodal, native distributed/FSDP/checkpoint/RNG etc. already have newer owners; the old safe_globals no-op, generated scalar monkeypatch, duplicate shard math cannot override them. Unselected differences such as pointwise AMP, type_as cross-device, TensorDataset batch gather remain explicitly retained as unintegrated scope; the 72 patch-distinct commits cannot all be called covered or all called missing features. |
| [#18030](https://github.com/Jittor/jittor/pull/18030); `8bdb533c9` | already covered and uses later fixes | The current `src/ops/composite/getitem_op.cc` folds implicit dims only on the storage_view path, keeping independent strides on the copy path; `7108fa6c5` and the existing storage-strides split regressions cover multi-dim transpose prefixes. Keeps the later high-rank/strided addressing handling, not overriding with the old two-line change. |
| [#18028](https://github.com/Jittor/jittor/pull/18028); `e5cffe1ae` | duplicate entry with the same head as #18031 | Both PRs' complete head SHA is `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`; #18028's base is the old `2.0`, #18031's base is `2.0-refactor`. Reviewed once as #18031's commit group, not re-importing another old layout. |

## Reproduction and per-item verification evidence

The following is per-item evidence explicitly bound to the tested sub-branch, **not a substitute for the final integration tree passing in full**. Skips, unfinished processes, read-only source conclusions or historical PR reports are not counted as this round's device passes.

| Scope | Minimal reproduction and executed results |
| --- | --- |
| cgroup compiler budget | the original get_total_mem still returned the 96 GiB host value inside a 2 GiB cgroup fixture; after the fix 19 passed, including the real worker-count C++ harness, v1/v2/ancestor/malformed/unlimited, CPU/affinity/quota and same-process quota changes. |
| worker exceptions and SIGINT | the original actual Dataset method wrapped the ValueError traceback as RuntimeError; after the fix 7 targeted passed, and the complete dataset file 18 passed/5 skipped (3 network, MPI, accelerator). SIGINT uses a bounded child helper; the worker case waits for two workers to finish initialization, avoiding treating the window before the Python fork hook finishes as an initialized-worker contract. |
| registered buffers | a real CPU buffer update regression failed before the change; after, TestGraphReplay 29 passed/0 skipped, including load/update/assign, nested shared holder and in-capture accumulated state. |
| opaque operations / nested capture | on the pre-change real CPU/original owner, a ctypes pointer write caused explicit replay to return wrong data; after the fix native GraphReplay+StepCapture 44 passed/0 skipped, Torch foreign pointer eager/automatic/explicit/step, non-pollution outside capture and API contracts 8 passed/0 skipped. The CPU mock only covers native-capture routing and cannot alone prove CUDA graph recording. There are also real CUDA outer-capture/Triton tests for device acceptance. |
| #18031 small contracts | before the change, real Conv1d tuple spatial parameters raised TypeError, negative Kaiser was not rejected, missing metadata returned {}; after the fix native ConvParityCPU 11 passed, compat metadata/Kaiser/Hann 5 passed. |
| #18048/#18053 numerics and padding | before the change, actual const StackVector iteration failed to compile, negative pad generated `i0--5` rejected by actual g++, CPU log1p lost tiny inputs, sigmoid(-350) seriously distorted; actual original index_fill/cast calls also reproduced large int64/NaN and redundant-conversion issues. After the fix 66 Torch, 3 native padding passed, plus 10/1 accelerator deselects; checked outputs and complete input VJPs. Final actual device results listed separately. |

This round's maintenance entry and mode split as follows; first preset the toolchain per the environment manual, and assign an independent lab `JITTOR_HOME`/`TMPDIR` for each mode. The wheel's `$core_wheel`/`$compat_wheel` are files produced by the canonical tools above; do not use a wheel containing user WIP to update the release baseline.

```bash
python tools/run_test_suite.py --tier core --backend cpu --jobs 0 --serial-compile
python tools/run_test_suite.py --tier smoke --backend cpu --jobs 0 --serial-compile
JITTOR_TORCH_SHIM=0 PYTHONPATH=python python -m pytest -q tests/structure/backends/acl/test_acl_dtype_preservation.py
JITTOR_TORCH_SHIM=1 PYTHONPATH=python:adapters python -m pytest -q \
  tests/structure python/jittor/compat/tests/structure adapters/tests/vllm/test_structure.py \
  tools/release/test_check_wheel_contents.py tools/release/test_check_sdist_contents.py \
  --ignore=tests/structure/backends/acl/test_acl_dtype_preservation.py
python tools/release/check_wheel_contents.py compare "$core_wheel" --baseline tools/release/baselines/wheel-contents-final.txt
python tools/release/check_wheel_contents.py compare "$compat_wheel" --profile compat --baseline tools/release/baselines/wheel-contents-compat-final.txt
python -m mypy
bash tools/check_repo_layout.sh
```

The actual native CUDA targeted set is `tests/core/test_graph_replay.py::TestGraphReplayCuda`, `tests/core/test_alias_autograd_state.py`, `tests/backends/cuda/test_fused_attention_f32.py`, `tests/backends/cuda/test_parallel_level_zero.py`; the Torch CUDA set is `test_activation_checkpoint`, `test_activation_materialization`, `test_activation_rematerialization`, `test_compile_external_autograd`, `test_raw_cuda_stream`, `test_torch_dataloader_iterable_cuda`, `test_custom_function_forward_mode`, `test_custom_function_saved_inputs`, `test_graph_replay_foreign_write`, `test_strided_gradient_writeback` (all in `python/jittor/compat/tests/torch/`, file suffix `.py`). Device configuration established by the same maintenance runner's `_session_environment(..., backend="cuda")` and through real CUDA warmup, pytest timeout 300 seconds per case.

Per-item logs and API JSON are in the unversioned `$JITTOR_LAB_ROOT/pr-audit-2026-10-25/`, `$JITTOR_LAB_ROOT/_state/pr-audit-20261025/` and `$JITTOR_LAB_ROOT/_state/pr-audit-numerics/`. The review table source material is `prs.json`, `older-audit.md`, `ecosystem-audit.md`, `perf-typing-disposition.md`, `pr18031-groups.md`; the repository does not copy raw logs, caches, build artifacts or downstream checkouts.

## Storage boundaries found during integration

The original main tree's three uncommitted files and untracked tests are saved in independent Git snapshot `preserve/local-optimizer-before-pr-audit-20261025` (`deb54e50d07e44406ce288f041b3d8fc94ae40cc`). The new `_optimizer_update` owner already covers the original frontend/base hook intent; the fused Adam's three `_update_preserve_dtype` calls remain user-local work, not entering the release wheel's source baseline. The original test file blob `90b7573f81575f45a12da54381b91dcbfc86ae3f` remains unchanged and untracked.

User tests exposed a pre-existing saved-output defect in eager storage write: an unobserved `p.exp().sum()` backward after updating parameters incorrectly recomputes exp. Both `deb54e50d` and the unfixed integration `1fa76c8cc` have the same four CPU failures, not attributable to #18052's introduction. `0dcd03a9b` in the native write owner saves the most recent reader outputs that are not yet independently allocated and still needed for backward, traversing along storage aliases and unallocated intermediate graphs, preserving kept/replay owners and input-saving version guards; no new global flush or execution-independent graph. CPU Torch targeted 85 passed, native alias/autograd whole file 32 passed; full identity/updates/compile matrix 140 passed, 0 skipped, of which 71 actual CUDA, 69 CPU (including user WIP), plus integration-tree native CUDA 92 passed, 0 skipped. The new tracked regression covers only committed SGD plain/fused and portable AdamW; user fused AdamW work is separately verified by the original untracked test, and a WIP-containing success cannot be passed off as clean release source results.

Independent review also reproduced three old logical/physical transpose and weakref test boundaries on the unfixed integration baseline, as well as the saved-value limitation of newly created old graphs between warm replays; recorded as KI-TEST-015 and KI-EXEC-015 respectively. The clean fused AdamW holder rebinding boundary is recorded as KI-COMPAT-016. These cannot describe this round's native fix as resolving arbitrary cross-replay historical graph claims.

## Final integration acceptance

The runtime gate uses Python 3.9.25, NumPy 1.26.4, g++ 11.5 and CUDA 12.2.140, real NVIDIA device RTX 4090. Caches, Python headers and toolkit overlay in lab isolated directories; system Python or CUDA installations not modified. This environment is not the canonical nox Python 3.11 typing environment, so mypy 1.8.0's 68-file pass cannot be passed off as the latter. Wheels are built separately with Python 3.11, setuptools 83.0.0, wheel 0.45.1 in the isolated canonical tool, preserving the existing METADATA/WHEEL/license layout.

| Gate | Corresponding source and result |
| --- | --- |
| CPU core | `0dcd03a9b` + retained Adam WIP; native 98 passed/44 skipped/1 xfailed, Torch 225 passed/1 skipped, total 323 passed/45 skipped/1 xfailed, exit 0. |
| CPU smoke (aborted) | `90fb7b642` + same Adam WIP and three documentation overlays; ran `--tier smoke --backend cpu --jobs 0 --serial-compile`. The native session advanced to ops tests and was terminated at the user's request during the first JIT; the native/Torch two segments were not completed, no complete counts or acceptance exit code obtained, not recorded as passed. The external `smoke-user-stop.json` saves the stop time and controlled-process origin. |
| Native CUDA targeted | as above; GraphReplayCuda, alias/autograd, fused attention FP32, parallel-level-zero total 92 passed/0 skipped, exit 0. |
| Torch CUDA targeted | `0dcd03a9b` + retained Adam WIP; checkpoint/materialization/rematerialization, external autograd, custom Function, CUDA stream/DataLoader, foreign writes, strided gradients total 119 passed/0 skipped, exit 0. |
| Clean-source structure | the test tree then recorded as-is as `6edcf1ae1` (runtime `11888f1ae` + the subsequently committed three baseline files, no user WIP); in an independent source-path environment native-only ACL recorder 5 passed, Torch/packaging/adapters 1690 passed/4 skipped; both segments exit 0, strict execution accounting. |
| Original worktree final structure | `509863fed` + three documentation overlays, Adam three-line WIP and the untracked original test; complete Torch/packaging/adapters 1690 passed/4 skipped, exit 0, 374.58 seconds. Strict execution accounting, HEAD/status/diff/coverage-file hashes all consistent before and after; native-only ACL recorder separately 5 passed, not re-executed in that Torch session. |
| Storage full matrix | `881e9ece6` + same Adam WIP + original test; identity 128, updates 10, CUDA compile 2, total 140 passed/0 skipped, exit 0; actual CUDA 71 items, CPU 69 items. |
| canonical wheels | clean committed `0dcd03a9b`; core 1,141/compat 185 member resource review passed, both exact baseline comparisons +0/~0/-0. Core adds only cgroup Python helper/headers versus the old baseline; compat additionally reviewed +5/~38/-0 against the `ca56aa802` canonical build (37 source plus RECORD). All source payloads are byte-identical to Git blobs, no metadata, license or deletion drift, no user WIP. |
| latest #18052 gate | `509863fed`, complete gate-scope/cache set 17 passed/0 skipped; test-related files only, runtime/wheel payload identical to `0dcd03a9b`. |
| reports and layout | after the final new report, index and SGD dated follow-up, documentation/private path targeted 19 passed/0 skipped, exit 0; layout and documentation governance passed (298 active Markdown files), `git diff --check` passed. |
| typing | the same `6edcf1ae1` test tree, 68 checked sources identical to `0dcd03a9b`; mypy 1.8.0, 68 source files, exit 0; scope limited by the Python 3.9 tool environment above. |


- After the user requested wrap-up and stop, this round's CPU smoke was aborted. Completed core, real CUDA, structure, typing and wheel evidence retained per the table; no complete smoke result, so no claim the whole test tree passes. Original worktree final structure check completed.
- This round did not install or run real DataTrove, nor re-conduct ms-swift, vLLM, DeepSpeed or verl/PPO end-to-end acceptance; #18054's historical DataTrove CPU report is review material only. This round had real single-card CUDA targeted execution; ROCm, Ascend/NPU/310P and multi-machine/multi-card collective communication not accepted. Corresponding hardware scope recorded in deferred-hardware; cannot be filled by import, CPU fallback, historical numbers or mock routing.
- The original worktree returned to `2.0-refactor` via three-way checkout and ref update with old-value validation, code commit `509863fed`. Two hook overlaps use the new owner, fused Adam's three writeback lines still uncommitted; original untracked test blob remains `90b7573f81575f45a12da54381b91dcbfc86ae3f`. Complete four-file raw bytes retained in `preserve/local-optimizer-before-pr-audit-20261025`. No stash, discarding local work or staging the user's original files; the actual runtime overlay matches the accepted storage matrix.

This report's conclusion is to select and integrate the fixes the current branch needs by change content, not to close all incomplete downstream capabilities. Unselected content and real device boundaries remain with the corresponding PR/issues ledger; old charts, old adaptation levels or old environment wheel hashes are not upgraded to this final release conclusion.

## Second-round supplemental integration same day

- Status: integrated; CPU core, structure gate and targeted CUDA execution passed, no complete smoke/full run.
- Baseline: `280d44854b3aeeb0e30b02b7b2e4e14e7a94b1ba` and its above per-PR dispositions committed.
- Final commit of this round: `220c0bc02` (plus this report commit).
- Owner: Jittor core and compatibility maintainers.
- Review when: the dispositions in the table above, downstream adapter admission surface or declared device scope change.

This round only supplements the parts explicitly recorded as "still missing/not integrated" in the previous section and verifiable in the current environment; it does not re-import rejected old owners, private Tensor/ProcessGroup interfaces, historical wheel/CI baselines or NPU-specific implementations.

| Group | Source | Merged content | This round's evidence |
| --- | --- | --- | --- |
| gate collection | #18052's two new commits | the xdist worker's collected-file set converges to the controller before session policy; when CUDA is unavailable, keep the skip reason carrying backend identity | gate-scope 18 passed; real `-n 2` 24 passed |
| type contract | #18051 residual | `copy(..., preserve_format=False)` stub default (the other four already in the tree) | structure tests passed |
| DeepSpeed | #18050 | `_finite`/`_norm` public API fast path, `torch._C._current_graph_task_id`, owner-state weak cache, NCCL wheel selection, concat scratch no-grad edge, BF16 GELU training kernel | adapter 49 passed; concat/grad/NCCL 25 passed; real CUDA GELU 14 passed/2 skipped |
| Torch semantics | #18031 | `Tensor.type_as` inherits dtype and device, reflected scalar stays on the tensor backend, TensorDataset single gather (and fixes double collate) | 190 passed; TensorDataset 9 passed; CUDA targeted 2 passed |
| factory semantics | #18043 | observable meta identity for the `device="meta"` factory parameter | 6 passed |
| Adagrad | #18038 | native dense FP32 Adagrad, `torch.optim.Adagrad` wiring and state migration; sparse/foreach/fused/differentiable explicitly rejected | 45 passed/1 skipped (including optimizer regressions) |
| verl ecosystem | #18053 | tensordict, verl, transformers_loading, transformers_qwen35 adapters and public compat entry | gate tests 27 passed; adapter self-tests 283 passed/15 skipped |
| vLLM | #18048 | `Tensor.exponential_`, buffers/model_loader/runner_policy, sampling/paging/log-probability/UVA regressions | vLLM harness 120 passed/72 environment skipped; main-tree targeted 208 passed/75 skipped |
| gates and packaging | #18034/#18037 | declared skip bucket takes precedence over torch, native-only files not entering the Torch snapshot, wheel RECORD approval, ASV matmul warmup, CPU cache key isolation | 41 passed |
| repository consistency | — | committed results docs at HEAD not registered in the generated `MANIFEST.in`, structure gate fails on the baseline; regenerated | packaging 8 passed |

Reconfirmed not to merge:

- **#18046**: process-level `core_source_signature` memoization was introduced by `2872b4b3c` and reverted by `23eb049a3` citing same-process source invalidation and extracted-function regressions. Not re-introduced this round; it is not a measured hot path either.
- **#18050's `_zero.py` and `ds_shape`**: the former directly drives Tensor's private `_storage_is_contiguous`/`_copy_into`/`_mem_ptr_now` interfaces, the latter puts DeepSpeed private metadata into public gradient semantics; both removed per the previous section.
- **#18036/#18039/#18043's NPU/310P-specific parts, #18035's old CUDA kernel version, #18031's old RNG/distributed owner**: no NPU/ROCm hardware or already superseded by the current implementation, remain unverified/not imported.
- **#18054/#18049/#18033/#18042/#18030/#18044/#18045/#18047**: already covered by the current implementation.

Verification environment and boundaries: Python 3.11.16, g++ 11.5.0, CUDA 12.2.140 (conda nvcc) and real RTX 4090 (sm_89); CPU core native 98 passed/44 skipped/1 xfailed and Torch 226 passed/1 skipped, total **324 passed/45 skipped/1 xfailed**; `tests/structure` **1541 passed/6 skipped**. DeepSpeed, vLLM, verl, tensordict upstream packages not installed, ecosystem tests cover only stub/fixture contracts and gate-visible behavior, cannot be recorded as real-machine end-to-end acceptance; NPU, ROCm, 310P and multi-machine collective communication not verified. This round did not run complete smoke/full, does not claim the whole test tree passes. During integration another session's smoke process in the same worktree made no CPU progress from `22:0x` to wrap-up; this round did not count any part of it as a result.

## 2026-10-26 continuous fixes and test completion

- Status: in progress; the table items are fixed and verified on CPU and real CUDA, the issues ledger closed or narrowed accordingly.
- Start baseline: `ffb0f9987` (after the second-round integration).
- Owner: Jittor core and compatibility maintainers.
- Review when: the fixes below, gate wiring or declared device scope change.

Environment: Python 3.11.16, g++ 11.5.0, CUDA 12.2.140 (conda nvcc), RTX 4090 (sm_89). Each worktree uses independent `JITTOR_HOME`/`TMPDIR`; first compilation serial. Upstream DeepSpeed/vLLM/verl still not installed, related conclusions cover only stub/fixture contracts.

| Defect | Disposition | Evidence |
| --- | --- | --- |
| KI-COMPAT-016 | fused AdamW writeback switched to the storage-preserving `_update_preserve_dtype` | storage identity 128 passed; direct and transitive old-output 8 passed; CPU+CUDA |
| KI-COMPILER-008 | `precompile` keeps the exception category; @ parsing, # directives, braces, @for/@strcmp/@alias, arity migrate to `USER_CHECK`; kernel_ir/cache_compile/op_register public paths likewise migrated, expr/swap recorded as non-public | targeted 181 passed/5 skipped/4 xfailed; C++ JIT 93 passed |
| KI-COMPAT-014 | complex128 becomes a frontend dtype (construction, cast, type, reporting), complex32 still a placeholder | 83 passed/10 skipped |
| KI-DTYPE-003/004 | Python float keeps 53 bits against float64; factory results no longer promoted by weak scalars, placement maintained by `_is_constant` | CPU 77 passed; CUDA 55 passed; strict xfail markers removed |
| KI-COMPAT-007 / KI-AUTOGRAD-003 | baseline already fixed, completed `copy_`-no_grad and register_hook residency regressions | 7 passed/1 skipped |
| KI-TEST-015 | storage-view fixture distinguishes logical/physical transpose and fixes lifetime assertions | file 14 passed |
| KI-TEST-002 | `run_test_suite` and nox wired to `check_session_completed.py`; xdist convergence, setup-skip counting | structure/tools 72 passed |
| KI-TEST-007 | `test_fused_op_relay_matmul` passes after all three historically polluting files | each of the three orders 2/4/5 passed |
| KI-TEST-011 | generated source and caches read as UTF-8 (all similar reads in the tree fixed together) | C locale + locale coercion/UTF-8 mode off, CUDA 1 passed |
| KI-OPS-014 | all-masked `finfo.min` row fixed by separating max and `1/l` (already in baseline) | dense-additive/key-padding CUDA 2 passed |
| KI-OPS-015 | SDPA CUDA backward already fixed by `4c722fee0`; this round records its basis | two device-parity cases pass on RTX 4090 (2 passed, about 1e-6) |
| KI-TEST-010 | LayerNorm fixture explicitly placed on the device, fused path genuinely covered | norm file CPU+CUDA 28 passed |
| KI-TEST-014 | resnet notebook changed to assert a property (non-monotonic) and removed the registered expected failure | manual smoke 3 passed (use_mkl=0) |
| KI-COMPAT-013 | module forward follows the module's own placed parameters when inputs are not placed | autocast suite CPU+CUDA 12 passed |
| KI-MEM-002 | host read returns a copy, does not migrate the source storage | CUDA residency contract 16 passed; CPU 6 passed |
| KI-OPS-012 / KI-BACKEND-009 / KI-BACKEND-014 | half-precision max/min identity (already in baseline), 16/64-bit atomics and bool results for logical/bit reductions, cublas/cudnn self-test op device placement | reduction matrix+minmax 24 passed; three library self-test classes 3 passed; all on RTX 4090 |
| KI-COMPAT-017 | capture-step state commit executes under `keep_graph=2`, replay keeps a rerunnable forward | `test_torch_compile_replay.py` 11 passed |
| KI-COMPAT-018 | portable snapshot makes non-contiguous tensors contiguous first, then batch fetch | two portable nodes passed (the safetensors dtype-bytes node is an environment gap) |
| KI-COMPAT-019 | under autocast, LayerNorm main body accumulates in float32 per ATen policy and narrows the result once | bf16 training loss 0.128 vs float32 0.122 (4.5% < 25%); half suite and CUDA core 361 passed |
| KI-CODEGEN-005 / 006 | 2-D and mean/jt.sum full-sum fusion fixed and gate extended (005 closed); rank-1 fusion over reshape still missing (006 open) | CUDA full-reduce 12 passed; rank-1 still allocates 68,157,440 bytes |
| KI-EXEC-009 | backward liveness releases only outputs still holding a count; the leak invariant now provably holds | 151 passed/3 skipped/4 xfailed (including the new C++ regression); leak matrix 10 passed; non-strict xfail markers deleted |
| KI-CODEGEN-002 | `ParallelPass` fixes the grid at the call site with `stride{i}`, the tail guarded by a live `::min` condition; the tuner CUDA guard is kept with the reason noted | CPU 2 passed; CUDA positive case passed; complete codegen 171 passed/38 skipped/5 xfailed |
| KI-EXEC-007 | read-path entries hold the synchronized/copied Var with VarPtr; `migrate_group` no longer holds `share_ring_mutex` during copy, instead pins the old block and re-validates after re-locking; `graph_check` runs outside the pin | canonical NT=4 35/35; readback NT=4 25/25 (separately re-verified 5/5, NT=1 all OK); CPU core 327 passed; structure contract 107 passed |
| 18031 second batch semantics | `Tensor.type_as`, reflected scalar stays on the tensor backend, TensorDataset single gather, masked-write gradient, expand scalar dim, torchvision resize/pad/color/crop, Conv1d device/dtype, advanced-index/0-dim device alignment | CPU 373 passed/16 skipped; CUDA targeted 14+2 passed |

| Final gate (after this round's fixes merged) | CPU core and structure | core: native 98 passed/44 skipped/1 xfailed + Torch 229 passed/1 skipped (combined 327 passed/45 skipped/1 xfailed, exit 0, session-completion markers 143/143, 230/230); structure 1548 passed/6 skipped, exit 0 |
| Device combination gate | CUDA core (RTX 4090, sm_89) | native 131 passed/10 skipped/2 xfailed + Torch 230 passed = **361 passed/10 skipped/2 xfailed**, exit 0, session-completion markers 230/230 |
| PR smoke (CPU, cold cache) | whole-tree selection after this round's fixes | native: 4084 passed/2014 skipped/18 failed/239 errors (45m44s); torch: 4775 passed/1878 skipped/16 failed/2 errors. Three contract regressions introduced this round were fixed and re-verified (op-compiler old message, native full-mask scalar dtype, half scalar materialization; 7/15/50 and half sweep+promotion+dtype 70 passed). The remaining failures reproduce identically on the 280d44854 baseline or are verification-environment gaps (venv lacks scipy, safetensors incomplete, no usable independent PyTorch, modelscope `tests` shadowing in `.pth`, real triton/vllm). Among them serialization batch fetch and compile replay SGD are registered as KI-COMPAT-018/017, and bf16 autocast CUDA divergence (79.5%) is registered as KI-COMPAT-019; all three are fixed and closed (see the table above). |

This table does not write partial files or single-case success as whole-library, whole-backend acceptance; NPU/ROCm/310P and multi-machine collective communication still unverified. KI-COMPAT-012's mechanism is fixed (in-process shim publishes torch distribution metadata, metadata regression passes), but downstream Transformers/Diffusers cases still need acceptance in environments with these packages installed. Also: a bare pytest broad sweep in the verification venv resolved `tests._helpers` to the modelscope `tests` package in `.pth` and reported ModuleNotFoundError; that is validation environment shadowing, not a product defect, and the formal gate via `tools/run_test_suite.py` is unaffected.

## 2026-10-26 follow-up: Ascend 310P and bilingual documentation

- The missing parts of PR #18043 are integrated in `5f0a3b6fa`: the 310P SoC
  capability table and kernel-gap fallbacks, the explicit `acl_allow_hf32`
  reduced-precision switch, the ACL inference helpers (FrozenLinear / packed
  weights), the `mapped_concat` / `mapped_packed_matmul` / `mapped_rms_norm`
  composite ops, and the corresponding ACL tests and guides. The 75 files that
  had not diverged from the common base were taken from the PR directly; the
  native fast paths, module call, concat/getitem providers and RMSNorm selector
  were merged by hand, preserving the existing provider-selection and placement
  logic. The DropoutACL gradient manifest row is kept: its CodeOp
  `multi_grad_src` is still counted as a backend gradient by the scanner.
- Device evidence comes from the PR author's 310P3 report; this host has no NPU,
  so ACL device cases were checked for collection and skipping only. Local
  gates: CPU core 327 passed / 45 skipped / 1 xfailed; structure 1569 passed /
  6 skipped; CUDA core 361 passed / 10 skipped / 2 xfailed (RTX 4090, exit 0).
- Documentation is now paired: `X.md` Chinese default plus `X.en.md` English,
  paired indexes, and `README.md` / `README.en.md` at the root. `docs/` holds
  187 pairs, and the link and toctree gates pass.

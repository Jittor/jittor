# Torch default transpose uses real storage views

- Status: verified only for the stated CPU/CUDA contracts; R33 and KI-COMPAT-004 remain open.
- Date: 2026-10-16
- Owner: native tensor/autograd and Torch frontend maintainers.
- Measured clean source: `5ba7f3aff965ea03f91b8040461071d3f2d94fbe`.
- Starting source: `b7c4128b4377218c3123c057002fea47698ba381`.
- Native observer test-only follow-up: `c33967bae79585719bebba459bffa96fe66bbc57`.
- Shared layout prerequisite: `12a9a143d323b41115729d1e8aa1e79028627161`.
- Review when: storage-view provenance, wrapper GC, reshape/copy, recorded history or layout propagation changes.

## Behavior and ownership

The default entry points of `Tensor.transpose/permute/t/T/mT` and the corresponding Torch functions now request the existing native physical permutation view. Values, strides, real shared pointers and write-back are verified together; identity, scalar and unit-axis permutations also retain their actual producer, so they can serve as independent intermediate AD targets. Jittor's global `transpose_storage_view` is not flipped: the native default is still zero, and the layout optimization has not been disabled.

The Torch frontend requests preservation of the actual Python root object through an explicit type attribute. The native view record stores that reference through the existing opaque Python owner/GC traverse/clear seam, without importing Torch and without building a Python ID table. After conversion the final wrapper is registered, not a temporary holder. Allocation ownership is separate from provenance: after `.data` replaces storage, `_base` still points at the genuine original root; ordinary native types keep the existing weak-reference policy. Tests cover nested roots, constructors and factories, temporary objects, three rounds of cyclic release, reentrant destruction, invalid type attributes, threads and exception recovery.

After a no-grad write, if neither the actual allocation nor the source-edge policy changed, history queries on the physical view retain the original producer and the materialized buffer. Source freezing/unfreezing continues to follow the existing recorded-edge rules; a severed edge cannot be reattached. The fast view entry point uses the real `grad_input`; even an invalid `view` observes that history before throwing, and this observation order is proven by an independent four-line Torch comparison.

Reshapes that can share storage prefer the native view. `Tensor.view` raises for incompatible strides; `Tensor.reshape` allows a genuine contiguous copy only under this explicit condition and then attaches the native reshape producer, without exposing a private copy temporary as the public `_base`. First-order, second-order and frozen-source gradients are taken along the real producer. `clone`'s preserve/contiguous formats are likewise executed by the native copy owner according to actual non-overlapping dense strides; expanded/holey layouts are not disguised as dense physical views.

## Provenance and execution protocol

This batch's thirteen production files and three test files were frozen at `5e3492b68`; after merging remote `12a9a143d` the measured source became `5ba7f3aff`. `formal-source-final-v1.json` verifies that the sixteen files of this batch are byte-identical to `5e3492b68`, while the four shared layout files match `12a9a143d`. The logs of the earlier rounds at `0d4e4598e`, `2f7b3a67b` and `5e3492b68` each retain their own identity and were not relabelled.

The independent oracle is genuine Torch `2.11.0+cu128`; its complete official wheel SHA256 is `c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`, and the actual `_C` SHA256 is `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`. The oracle uses a complete private provider; the old shared installation was not patched. Identical maintained test files were copied byte for byte outside the repository and executed there, to avoid the monorepo shim collection policy, without modifying their assertions.

The candidate CPU environment explicitly disables CUDA; genuine CUDA uses a separate device, the cu122 provider and strict kernel math, verifying actual locations/pointers and gradients. Each first JIT was serialized, every phase and the canonical core used separate caches, unit tests did not share the benchmark cache, and no JIT was copied. The new maintained tests, the native owner tests and the original diagnostics use `forbid_backend_fallbacks` and keep them in sync; the eight diagnostic groups additionally recorded fallback counters of zero. Verification time is not a performance conclusion.

## Validation

| Scope | CPU | CUDA-visible environment |
| --- | --- | --- |
| Same new maintained tests, independent Torch | 112 passed / 112 CUDA skips | 224 passed (112 CPU + 112 CUDA) |
| Candidate: full original fifteen-file adjacency + new maintained files | 766 passed / 615 CUDA skips, 390.75 s | 1343 passed / 38 skips for unconfigured source, 1813.50 s |
| Seven native wrapper/GC items + forty-eight shared layout items | 55 passed / 0 skipped, 2.94 s | 55 passed / 0 skipped, 5.03 s |
| Independent sixteen public scalar-promotion lines | 16 lines fully matching / fallback 0 | 16 lines fully matching / fallback 0 |

The native owner/layout table's results on both sides correspond to clean `c33967bae`; the remaining numerical results still correspond to `5ba7f3aff`.

Canonical core: native **98 passed / 41 skipped / 1 existing xfail**, Torch **224 passed / 1 skipped**, totaling **322 passed / 42 skipped / 1 existing xfail**, with both modes exiting zero. The complete structures were run separately: `tests/structure` **1516 passed / 4 skipped** (228.59 s), `compat/tests/structure` **114 passed / 0 skipped** (16.27 s); together **1630 passed / 4 skipped**, not a single combined command. After modifying the test observation order, a separate pre-commit full `tests/structure` run: **1516 passed / 4 skipped** (187.67 s); that workspace's per-file hashes matched the subsequent `c33967bae` commit, and the then-current HEAD is not relabelled to the new SHA. The repository layout check passed; there are no new Ruff diagnostics relative to the shared prerequisite. After adding this report, the targeted documentation check was **18 passed / 1 failed**, the only failure being that the new page is not yet written into the central results index; production gates and documentation registration are separate, and the index is re-verified after central integration.

The independent initial eighty-three-line comparison covers default views, write-back, leaf/grad_fn, no-grad, source freeze/unfreeze and exceptions. Before the fix, eighty-two lines had field differences; that count includes layout, metadata and error contracts and does not represent eighty-two numerical errors. After the fix, the eighty-three lines on CPU and CUDA each fully match, both with fallback 0. Separately, history15, fast-view6, strict-copy2 and invalid-history4 remain field-by-field zero-different. Six of the base9 entries match in the no-cycle scenarios; the other three genuine root/type cycles can be released in the candidate while the oracle still retains them, the complete original differences are preserved, and the requirement is that the candidate introduce no new leak, without claiming fully equal cyclic behavior. The old orphan-root four lines are additionally confirmed by an independent original script/JSON under genuine Torch `2.11.0+cpu`; this retains a separate identity from the main oracle's cu128.

## Preserved failures and boundaries

The full fifteen-file adjacency at the initial clean `0d4e4598e` yielded **6 failed / 648 passed / 503 skipped**. Four of these involved expanded sibling, buffer/producer and old orphan-root assertions; the latter was proven by genuine Torch to keep its root until the final view is released, while the original four leaf-write rejection and value-unchanged assertions are retained. The other two are FP16/BF16 shared layout scalar-promotion errors, fixed by the independent `12a9a143d`. The final CPU and CUDA full adjacencies have no failures; the 38 CUDA skips are all `native flash-attn source not configured` and do not grant that extension a pass.

The first round of the final native owner/layout CUDA run was **54 passed / 1 failed** (143.66 s). The failing reentrant-finalizer test called `.numpy()` first and then checked that the same tensor must still be on the GPU. The pure native two-branch diagnostic `native-finalizer-observation-cuda-v4.json` proves that, without a first host read, the root is device before and after cleanup with the same real device pointer; with a prior `.numpy()` it immediately becomes CPU staging, and cleaning up the root and syncing must not be disguised as GPU execution. Both branches have correct values and destruction order, fallback 0. `c33967bae` only changed the first host read into a sync plus a real location assertion, retaining the final numerical and device assertions; the thirteen production files and all compat tests are unchanged. After the correction, both sides of the clean commit's owner/layout run were each **55 passed / 0 skipped**. The early probe's wrong attribute call, the destruction-cleanup diagnostics and the original single failure are all preserved; production residency behavior was not modified.

The four failures of the original new strict-copy test came from reading the unpublished `_version`; no fabricated attribute was added, and instead the real post-write/inspection behavior of no-grad views was used to check the independent version boundary. The first raw audit also retains one rejection log, because the candidate added two native producer/location observation fields; the final audit explicitly verifies their content without deleting the original differences.

The strides of returned `autograd.grad` tensors still differ: among the three reshape lines the candidate gives `(1,3)` and the oracle `(4,1)`; among the thirty-five copy lines, sixteen returned gradient strides also differ. The corresponding values, shapes and actual forward layouts are identical, and the actual layouts of `.backward` leaf gradients were separately verified as identical. The complete differences are preserved field by field, and these returned-layout differences are not granted full gradient-ABI equivalence. The two candidate-only native observation lines additionally verify that the producer ID and the real physical location are completely unchanged before and after the query.

General saved-input version protection still has a separate genuine gap: after an old square consumer, a no-grad public `copy_` is executed, and genuine Torch rejects the version mismatch while the existing candidate may still accept it. This is the next batch's independent native fix task and is not replaced by this batch's view passes; `.data`'s genuine allowed behavior, constant multiplication, and backward semantics through a saved output must be preserved. Nor does it grant untested accelerator backends, all signature errors, the complete C-extension ABI, all downstream coverage, or completion of R33 maintainer PR handling. Physical storage preservation of differentiable `copy_` also belongs to the next batch's native contracts and cannot be inferred from this batch's ordinary view write-back tests to conclude that all in-place APIs are aligned.

## Reproduction and evidence

```bash
# 每条设备轨使用独立仓外 JITTOR_HOME；首次编译串行。
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 JITTOR_TEST_DEVICE=cpu \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  compat/tests/torch/test_default_transpose_views.py
CUDA_VISIBLE_DEVICES=<gpu> JT_USE_CUDA=1 JITTOR_TEST_DEVICE=cuda \
  JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1 JT_CUDA_KERNEL_MATH=strict \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  compat/tests/torch/test_default_transpose_views.py
JITTOR_TORCH_SHIM=0 <jittor-python> -m pytest -q \
  tests/core/test_view_owner_protocol.py tests/ops/test_storage_layout_dtype.py
<jittor-python> tools/run_test_suite.py --tier core --serial-compile
bash tools/check_repo_layout.sh
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  tests/structure compat/tests/structure
```

Unversioned evidence: `$JITTOR_LAB_ROOT/_state/native-default-view/20261016/`.
The exact complete fifteen-file selection is in `final-adjacency-targets-v1.json`, each round's environment whitelist, command, actual SHA and exit code are in `logs/*.command.json`; the final results are in `logs/final-*.log`.
`formal-source-final-v1.{json,tar}` retains the frozen source, and `final-live-process-platform-v1.json` retains the actual process environment and device identity. The original probe/comparison/audit version files and failure history are all preserved.
`formal-native-observer-source-v1.{json,tar}` binds the test corrections and native production bytes, and `final-owned16-delivery-v2.json` gives the sixteen exact files and the integratable refs in dependency order.
The final `final-default-view-evidence-seal-v2.json` seals **485 files / 287,644,875 bytes**, SHA256 `947944ef0bd9c4103d49b666a4547485fdfff48ec69aa6b33f9cb5429c5a1267`, with an independent re-hashing showing zero differences. The manifest includes the original failures, each phase's source archive and the actually loaded binary identities; no JIT was copied to start verification.

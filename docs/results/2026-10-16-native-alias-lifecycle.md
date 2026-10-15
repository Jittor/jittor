# Native alias history and logical AD inputs

- Status: verified intermediate contracts only; KI-COMPAT-004 and R33 remain open.
- Reviewed: 2026-10-16.
- Owner: native autograd / Torch compatibility maintainers.
- Measurement: `4f95f71195d60b9c60ca3c1830b823dec552eb3a` plus the 28 exact owned-file SHA256 values in `layout-final-source-v1.json`, on remote `00ceb26fd395707fb33649cc43709af045110a9a`.
- Integration source: `0458c28b41643e4b2975845c0a0b366c213e2f8d`, byte-bound to the measured files; earlier e7af/2f/bc/20-file measurements retain their original identities.
- Review when: native view history, edge snapshots, storage rebinding, layout propagation, flags or leaf-gradient layout changes.

## Result and implementation

Native no-grad-created aliases now keep visible trainability, write-version policy and recorded backward edges separate. A source becoming trainable can change an alias's visible `requires_grad` without inventing its accumulator. Old recorded consumers retain their input-edge snapshots. A source edge disabled during a frozen write remains disconnected after the source is enabled again. Empty writes and self-copy advance write history. Public history queries consume pending history; metadata conversion, physical dispatch observation and the `requires_grad_` setter do not.

The native alias policy owns only small policy objects, never Var/Op/Python Tensor owners. Its source chain and own/inherited flags model visible trainability. The write version is shared by detach but separate for data aliases. The creation epoch is an unrecorded view's creation version, or a recorded view's last consumed history version. A real recorded native view producer alone grants recorded-view history; a stale no-grad-created alias cannot acquire it. Source-edge eligibility describes the actual rebase edge, and an epoch change alone grants no edge. Native permanent stop-grad is preserved. Lifecycle tests verify that source Vars and the new forward dependencies are released.

The original recorded-input loop from [819182d1b](2026-10-16-recorded-grad-fn-freeze.md) is byte-identical. The additional recorded-view fallback preserves permanent-stop and disabled-at-construction checks. Native target/traversal eligibility follows actual recorded consumers, and the Torch frontend queries native accumulator eligibility instead of guessing from a Python tensor registry.

`.data` previously replaced a holder's Var with a stopped clone, losing old leaf gradients and nonleaf producers. The fix finishes old readers, pins the right-hand allocation through the existing detached storage owner, and changes the same Var's allocation/shape/dtype/placement through existing allocator operations. Its producer, trainability and input-edge snapshots survive. Old explicit views retain their old allocation and guard provenance. Only an actually recorded graph preserves an old gradient-shape contract; existing native reduction validates compatible gradients and rejects incompatible shapes.

`autograd.grad` returns the recorded graph gradient. `.backward` separately applies the initial leaf-gradient layout contract with existing allocation/view/copy primitives, including real dense permutation storage and higher-order differentiation. The global native transpose policy is unchanged. Torch-style SGD's existing native step executes under no-grad; closures and native/loss-style optimization retain their existing behavior.

The layout optimization had a second, independently reproduced defect: internally transposing or unwrapping saved operands changed their logical identities. After `.data` replacement, multiplication and input-dependent unary derivatives read an old allocation or bypassed a real nonleaf/view target. BinaryOp and UnaryOp now retain their original logical inputs and existing derivative formulas. An explicit stopped dependency computes the optimized forward with the existing kernels; the original operator aliases that real result and owns AD. Saved-output derivatives still use the real saved output. The no-grad/inference fast path remains unchanged. This does not globally disable layout propagation, save stale input values, introduce a parameter-ID table or implement substitute gradient mathematics.

## Validation

Raw evidence is unversioned under `$JITTOR_LAB_ROOT/_state/native-alias-lifecycle/20261016/`. Each command JSON binds command, actual environment, return code, measured HEAD and exact owned-file hashes. CPU hides CUDA, CUDA runs require execution, first JIT is serial, and unit/benchmark caches are separate; JIT files were not copied.

| Check | CPU | CUDA-enabled selection |
| --- | --- | --- |
| Final eleven compatibility files | 524 passed / 400 accelerator skips | 924 passed / 0 skipped |
| Independent genuine Torch contract copy | 399 passed / 400 accelerator skips | 799 passed / 0 skipped |
| Native adjacency, identical 22 production files | 100 passed / 15 skips | 89 passed / 1 preserved expand-memory failure / 0 skips |
| Channels-last layout/gradient/training adjacency | not claimed | 3 passed / 0 skips; fallback forbidden |
| Canonical combined core | 322 passed / 42 skips / 1 existing xfail | not claimed |
| Full structure | 1630 passed / 4 skips | host-only |

A CUDA-enabled parameterized selection includes CPU rows, so these totals do not imply that every row executed CUDA arithmetic. New native/alias semantic tests and dedicated probes synchronize tensors and check physical location under `forbid_backend_fallbacks`; the original 32-case CUDA probe also checks driver pointer attributes. This does not claim a global fallback guard around every unrelated legacy test. The three channels-last nodes additionally run inside a whole-invocation fallback guard. These guards return normally after synchronization; raw JSON does not contain a captured numeric fallback-counter field, and none is added retrospectively.

All three oracle contract/support files are byte-identical to the maintained source (`final-oracle-contract-byte-binding-v1.json`). The independent oracle is official Torch 2.11.0+cu128: wheel SHA256 `c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`, actual `_C` SHA256 `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`. The candidate uses its independent CUDA 12.2 provider. Oracle commands disable bytecode writes and run without the shim. Genuine Torch emits its normal `backward(create_graph=True)` reference-cycle warning; the corresponding test clears accumulated gradients.

The parent independently sealed a real 48-case saved-input/saved-output matrix on old 4f95 production bytes: 12 numerical failures per device, precisely leaf/nonleaf/constructor mutation followed by multiplication, sin, cos or log. Its other 36 rows already matched Torch. The fixed implementation matches all 48 rows on each device. Independent finite/shape/value checks cover 280 arrays per device: maximum absolute error is 2.384185791015625e-7 on CPU and 1.1920928955078125e-7 on CUDA, under the frozen FP32 rtol=2e-6 / atol=2e-7 budget. CPU raw retains its 26-file identity and CUDA its final 28-file identity; both have identical 22 production files. The matrix distinguishes input-dependent derivatives from exp/sqrt/tanh's saved-output derivatives, and preserves explicit user copy/view allocations. A separate lazy 24-case matrix matches on both devices, so materializing an observer does not hide a missing target edge. The final maintained tests include lazy consumers and AD recovery after an intentional constructor error.

The new rank/layout 48-case matrix matches genuine Torch on CPU and CUDA. It includes nonuniform old gradients, rank contraction/expansion, actual physical transpose layouts and broadcast changes. For example, equal-numel 2x2→4 permits the old-shape `autograd.grad` result while `.backward` genuinely rejects the incompatible copy shape; a compatible physical-stride case has its independently observed contract. These outcomes are tested individually rather than replaced by blanket reshape behavior.

Final original32-case reruns: 32 rows and 64 physical observations per device, zero differences; CUDA driver attributes identify every observed pointer on the requested device. Earlier 20-file-source raw evidence remains distinct: extended124, frozen-write6, empty/self-copy6, observation-order48, data/detach16, rebind4, stale-escape4, plus data-orphan guards8/device all matched genuine Torch. The maintained final alias suite retains those boundaries and the original constructor41/device cases. Comparison JSON preserves actual error text; rejection presence is compared, and oracle-only integer version values are retained in raw files without claiming a candidate version API.

The final 28-file freeze changes only tests relative to the successful 26-file phase. `production-source-reuse-binding-v1.json` proves all 22 production file hashes are identical, so the native adjacency and lifetime results above retain their actual 26-file-source identity. Full compatibility/core/structure and the corrected channels-last gate are measured on the final 28-file source. Ruff comparison finds no new diagnostics; existing method/optimizer import diagnostics remain unchanged. Repository layout passes.

## Reproduction

Use separate CPU/CUDA unit homes, a genuine Torch environment for the exact contract copies, and the backend/toolchain settings recorded in the command JSON files. `logs/compat-layout-final-{cpu,cuda}-v2.command.json` lists the full eleven-file selection; `logs/native-layout-full-{cpu,cuda}-v1.command.json` lists native adjacency. Maintained entrypoints include:

```bash
<Jittor Python> tools/run_test_suite.py --tier core --jobs 0 --serial-compile
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <Jittor Python> -m pytest -q tests/structure compat/tests/structure
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <Jittor Python> -m pytest -q compat/tests/torch/test_alias_lifecycle.py compat/tests/torch/test_tensor_constructor_alias.py
```

For CUDA, set both `JITTOR_TEST_DEVICE=cuda` and `JITTOR_TEST_DEVICES=cuda`, require actual execution with `JITTOR_TEST_REQUIRE_CUDA=1` / `JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1`, enable the real provider, and use `JT_CUDA_KERNEL_MATH=strict`. CPU explicitly sets `CUDA_VISIBLE_DEVICES` empty and disables CUDA. The above contract selection is part of the full eleven-file gate; it does not replace its remaining adjacency.

## Preserved failures and remaining scope

Default public transpose layout still differs: native `transpose_storage_view=0` materializes where Torch exposes a view. Local physical-view allocation for leaf gradients does not close default transpose or full view/C-extension ABI compatibility.

`test_expand_does_not_materialize_when_it_is_consumed` retains its original 16 MiB budget and fails with 68,157,440 bytes allocation growth on CUDA. A fresh clean e7af baseline, with an independent cache, reproduces the same result. It remains a failure in the native adjacency total above.

The first channels-last invocation numerically passed its three CUDA-only tests but exited 1 because the old class name was not identified as accelerator execution. Renaming that actually CUDA-only class to `TestChannelsLastCUDA`, with both existing structure selectors updated, preserves all three nodes, budgets and the minimum-one-executed rule. The final guarded invocation exits 0. The original exit-1 log is preserved and is not counted as a successful gate.

Earlier preserved failures include an existing duplicate random-state fixture contract, the exact ledger count for two new user-facing shape checks, and rejected intermediate layout-implementation attempts. The final fixture uses the existing helper, the ledger reflects the actual checks, and no allowlist or test-policy exemption was added. Original logs retain their source identities.

This verifies the stated alias/storage/AD contracts only. It does not close R33 maintainer-PR disposition, certify full default view/ABI behavior, or claim untested accelerators or downstream libraries. The immutable raw inventory is `final-layout-evidence-seal-v1.json`; `final-layout-ready-source-binding-v1.json` binds measured bytes to the clean integration commit and `final-layout-ready-source-v1.tar` preserves its complete source tree.


## Integration recheck

Clean `66db6e44af90718fb732faab10c2f832854b9322` combines these exact 22 production
files with the reviewed AA9 attention and Diffusers source gates. Independent byte
comparison preserves all 22 alias owners and all four AA9 production owners.
Core passes 322 / 42 skipped / one existing xfail; full structure passes
1628 / six skipped. The eleven-file compatibility selection passes CPU
524 / 400 accelerator skips and CUDA-enabled 932 / zero skips, including the eight
new BF16 attention tests. Native alias/attention passes CPU 38; the CUDA-enabled
native alias/attention/full channels-last selection passes 49 / zero skips and
includes explicit CPU math checks. A separate whole-invocation fallback guard
passes all eleven channels-last nodes with the actual counter unchanged at zero.
Host source/nightly tests pass 182. Unit caches are separate from benchmark caches;
first compilation executes locally, without copying JIT artifacts.

Unversioned integration commands, raw logs, actual live CUDA environment/library
observations and the complete 2923-file Git source archive are under
`$JITTOR_LAB_ROOT/_state/refactor-remaining-main-00/20261016/`.
These integration results retain source 66db; they do not relabel the earlier
4f95/0458 evidence, repair the preserved expand-memory failure, or close default
transpose/view and full ABI work.

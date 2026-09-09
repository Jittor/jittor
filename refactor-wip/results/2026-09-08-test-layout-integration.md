# Test ownership and shared policy integration

- Status: Implemented; mapped dynamic collection and package boundaries verified
- Baseline: `0df748c16` (includes `c618d841d` and the local vLLM extraction)
- Reviewed: 2026-09-08
- Owner: test infrastructure and source-domain maintainers
- Recheck when: a test root, process mode, generated case family or package boundary changes

The test tree now mirrors implementation domains. The former compiler bucket
is split into build/codegen/runtime/memory/binding/debug owners; ACL and
communication tests follow their backend directories. Torch, FSDP and Triton
tests move into the compatibility project. vLLM tests remain in the user's
requested adapters project. The protected `tests/core/test_setitem.py` retains
its path. The CPU mock ACL dtype recorder is a structural contract, with an
explicit native-process exception, rather than a falsely advertised NPU test.

All three test roots load one `tests/_helpers/pytest_policy.py` plugin. Thin
conftests do not duplicate hooks. The move exposed two existing duplicate
function definitions: the second `pytest_sessionstart` hid selection checks,
and the second `pytest_sessionfinish` hid xdist-worker state reporting. Their
checks are now connected to single actual entry points. Shared root accounting,
mode selection, hardware markers, execution/skip gates, state reporting and
oracle preload remain active. Both nox and `tools/run_test_suite.py` derive
their native/Torch selections from the same three-root gate scope.

Old-to-new path mapping preserves the generator's historical seed keys for
existing test identities; it does not rewrite pytest nodeids, reports or
selectors. Current commands use physical paths. Historical results and frozen
baseline hashes retain their original evidence paths. Active known-issue links,
runner selectors, skills and lint/format ownership follow the new paths.
See [the layout contract](../testing/test-layout.md).

## Evidence

The coordinator applied the rename/edit patch by three-way merge onto the
current baseline, resolved the two selector conflicts, removed duplicate old
paths for five newly added tests, and preserved the native-preservation,
reduction-policy and device-copy error-boundary assertions from `c618d841d`.

- Against actual `0df748c16` content: 386 mapped files, 3567 original AST test
  identities and 291 mathematical assertion inventories had no loss or
  differences after approved path normalization. This includes signatures and
  parametrization decorators; AST counts are not pytest collection counts.
- The seed audit covers 388 mapped file prefixes and 3585 method keys, including
  the baseline's newly added installer/optional-plugin/provider tests. An earlier
  audit based only on the older development snapshot missed four keys; the
  final audit uses the actual integration baseline.
- The host policy set passed 74 nodes with zero skips. Additional policy/lint
  checks passed ten nodes; tests exercised multiple roots in one process and
  compatibility-directory startup with required execution enabled.
- Actual native collection matched 3282 baseline nodes against 3282 mapped
  final nodes. Torch collection preserved all 5036 baseline nodes, with only
  eight newly added policy guards (5044 at that checkpoint). Neither side had
  collection errors, skips or duplicate nodeids. This comparison includes
  dynamic device/OpInfo parametrization, not just source ASTs. It first exposed
  168 missing native nodes: pytest's default `norecursedirs` ignored the new
  build domain, and an ancestor-directory filter hid the native ACL recorder.
  Both were fixed before the successful comparison.
- Two later host guards distinguish a new class reusing an old method name
  and a new explicit method colliding with a generated-family prefix. Single-file
  collection confirms ten policy nodes after these additions. The final seed
  representation stores scoped class/method identities, device-template aliases
  and bounded generated families. All 4337 actually collected migrated nodeids
  retain their original seed key, without storing the full OpInfo nodeid matrix.
- Core, compat and adapters wheels and sdists built successfully. Each wheel
  rebuilt from its sdist with the same members/content (excluding RECORD).
  The three production wheel path sets have no intersection. Source-byte checks
  covered 1011 core, 166 compat and 11 adapters production files. The core's
  typed bootstrap stub, placement, dtype, frontend scope and contiguous resources
  are present; compat tests are absent from its production wheel. Actual wheel
  discovery loaded and rolled back the vLLM registrar without JIT.

Raw artifacts are unversioned: `$JITTOR_LAB_ROOT/_state/test-layout-map-XuLI67/`
contains `applied-map.json`, `integrated-ast-audit.json`,
`integrated-seed-audit.json`, `integrated-host.log` and the merge record.
`$JITTOR_LAB_ROOT/_state/test-layout-collection-eN4NiT/comparison.json` records
the complete collected-set comparison. Both sides use the same warm core and
compatibility implementation, with source origins recorded; no core was rebuilt
just because the test paths moved.
`$JITTOR_LAB_ROOT/_state/layout-integrated-packaging-o9zkty7o/report.json` records
the wheel/sdist hashes and full packaging checks. No new numerical/backend claim
is inferred from file movement or host-only policy tests.

# Test ownership and shared pytest policy

The native test root is `tests/`; the optional Torch project's root is
`compat/tests/`, and third-party adapters use `adapters/tests/`. All use
`tests/_helpers/pytest_policy.py` through thin
conftest adapters. Helpers remain monorepo development dependencies; no test
package is installed with Jittor or the compat wheel. Running compat tests
requires the monorepo checkout, including its `tests/_helpers` directory.
Standalone adapter sdists keep their own host tests runnable when this optional
monorepo support directory is absent.

Native graph tests live under `tests/core`, generated-code tests under
`tests/codegen`, and Python build tests under `tests/build`. Memory, types,
runtime, bindings and native Python domains have their own directories.
Backend tests mirror `backends/{cpu,cuda,acl,rocm,corex,comm}`. Structural
contracts retain their category under `tests/structure/<source-domain>`;
cross-domain repository governance remains at `tests/structure`.
Compatibility-owned structural tests live under `compat/tests/structure`.
`tests/core/test_setitem.py` intentionally remains at its user-requested path.
The ACL dtype recorder is a CPU mock contract under
`tests/structure/backends/acl`, with an explicit native-process exception; it
does not inherit the real ACL hardware test's fallback fixture or NPU marker.

Process mode belongs to each file, not to its new directory. The authoritative
selection is `tests/_helpers/process_modes.py`; the complete runner derives
native and Torch selections from it through `gate_scope.py`. In particular,
FSDP tests moved from `tests/distributed` keep their native startup mode, while
Torch API tests retain Torch mode. A directory named `acl` retains the `npu`
hardware marker, and communication/FSDP tests retain their `mpi` marker.
No argv-based activation is introduced.

Use `nox -s full` or `tools/run_test_suite.py` for the two-process suite.
`nox -s structure` includes the structural roots and the adapter's structural
contract. It runs the native CPU recorder separately. Direct selectors use the new
physical path; class, method and parameterization suffixes are unchanged.

## Deterministic input compatibility

The reviewed 10.23 move mapping is retained only for input generation in
`tests/_helpers/layout_seed_paths.json`. `layout_seed.seed_nodeid` substitutes
the old file prefix for existing class/method identities before the input generator hashes
its key. Device templates and generated method families are recorded separately;
current explicit method declarations take precedence over a generated-name prefix.
A new class reusing an old method name therefore remains a new case, as does an
explicit new method whose name happens to match an old generator's prefix.
The original class/method/parameter suffix is retained. New method
identities and files use their new canonical path. Pytest nodeids, logs, selectors,
failure reports and coverage are never rewritten by this compatibility map.

The migration compares old and new AST node inventories after mapping only
physical file paths. Existing mathematical assertions are retained. Frozen
result transcripts and approved hash baselines keep the historical paths on
which their evidence was collected; active selectors and commands follow the
new physical paths.

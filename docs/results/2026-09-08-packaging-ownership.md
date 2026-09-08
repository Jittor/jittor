# Repository tools and package-resource ownership

- Status: implementation and offline artifact verification complete; task-state updates belong to the coordinator.
- Reviewed: 2026-09-08.
- Owner: packaging / repository maintainers.
- Baseline: isolated `refactor-packaging-BnbUas` worktree, based on `7e83`.
- Recheck when: resource declarations, package-dir mappings, or build-tool locations change.

Nine commands/self-tests moved out of agent/scripts to tools/build, tools/docs,
tools/release and tools/check_repo_layout.sh. Callers, Nox, AGENTS, type-check
configuration and layout governance use those owners; the old directory has no
forwarders or duplicate implementations. User-facing nvtx, jtune, tracer and
benchmarking remain in jittor.tools. Benchmarking implements jt.benchmark on a
user's model; ASV checkout comparisons remain repository tools.

Core and compat each declare runtime data in pyproject's standard package-data
table. A data-only jittor.src mapping replaces the custom setup.py core-file
copy loop. Compat's setup.py no longer keeps its own resource-directory list.
MANIFEST.in is generated from package-data and the same pyproject's source-only
sdist selections. The packaging gate checks generated files for drift.

Verification did not import or JIT Jittor: builds ran outside the checkout with
an import guard that fails if runtime import is attempted. Build tooling matched
the maintained pins: build 1.3.0, setuptools 83.0.0, wheel 0.45.1.

- Focused packaging/docs/layout/board contracts: 31 passed.
- Sdist checker self-tests: 13 passed.
- Wheel checker self-tests: 14 passed; its pre-existing final-baseline assertion
  still fails because historical approval data predates current required paths.
  Historical manifests and hashes were not refreshed.
- Core sdist contents gate passed, including the four Chinese notebook source
  filenames containing spaces. The initial generated MANIFEST lost them because
  MANIFEST does not parse quoted paths; the generator and a regression fixture
  now cover this case and dirty source caches.
- Core wheel audit: 1,015 members; compat wheel audit: 152 members; overlap zero.
- All 631 declared core resources and 29 compat resources match source, sdist
  and wheel byte-for-byte. All 349 backend files and 291 core / 128 compat Python
  source members also match their wheels. Maintainer tools are absent from wheels;
  user tools remain present.
- Documentation governance, internal links and repository layout passed.

Artifacts and logs are unversioned under
`$JITTOR_LAB_ROOT/_state/packaging-9-19-8r17tW/`. Source snapshots and extracted
sdists live there, not in the checkout. This verification establishes build and
resource consistency; it does not claim an installed runtime/device selftest.

Developer entrypoints:

```bash
python tools/build/generate_manifest.py
python tools/build/generate_manifest.py --check
bash tools/check_repo_layout.sh
python tools/release/check_sdist_contents.py <core-sdist.tar.gz>
python tools/release/check_wheel_contents.py audit <core.whl> --profile core
python tools/release/check_wheel_contents.py audit <compat.whl> --profile compat
```

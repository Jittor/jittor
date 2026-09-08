# Repository Tools

This directory contains maintainer and operational commands that are not part
of the installed `jittor` API. It intentionally has no `__init__.py` files and
must remain absent from wheels.

Most commands write state below
`${JITTOR_LAB_ROOT:-../jittor-lab}/_state`. Review commands under a `legacy/`
directory before use; they preserve old workflows but are not recommended for
new deployments.

- `benchmarks/legacy/`: historical one-off comparisons; maintained benchmarks
  live in the root `benchmarks/` ASV suite.
- `build/`: explicit source and asset builders.
- `distributed/`: operator-facing distributed helpers.
- `docs/`: documentation builders, links and governance checks.
- `install/legacy/`: installers for old environments.
- `release/`: release-only utilities and offline asset packaging.
- `check_repo_layout.sh`: repository layout gate; delegates documentation governance.
- `build/generate_manifest.py`: derive both MANIFEST.in files from their pyproject declarations.

Maintainer checks formerly under `agent/scripts/` now have their sole owners
here. Package resources are declared in each distribution's
`[tool.setuptools.package-data]`; `[tool.jittor.sdist]` adds source-only content.
After changing those declarations or adding source-only files, run
`python tools/build/generate_manifest.py`. The structure gate checks the generated
manifests for drift. Neither build imports Jittor or requires the sibling project.

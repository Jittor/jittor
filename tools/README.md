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
- `docs/legacy/`: documentation tooling pending Stage 8 retirement.
- `install/legacy/`: installers for old environments.
- `release/`: release-only utilities and offline asset packaging.
- `services/legacy/`: explicitly enabled long-running services.

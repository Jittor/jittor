# Repository Tools

This directory contains maintainer and operational commands that are not part
of the installed `jittor` API. It intentionally has no `__init__.py` files and
must remain absent from wheels.

Most commands write state below
`${JITTOR_LAB_ROOT:-../jittor-lab}/_state`.

Benchmark commands live in the root `benchmarks/` tree (ASV regressions,
Torch comparisons, graph profiling, and ACL measurements).

- `build/`: explicit source and asset builders.
- `containers/`: source-development container recipes; build from the repository root.
- `distributed/`: operator-facing distributed helpers.
- `docs/`: documentation builders, links and governance checks.
- `ecosystem/`: live source-binding and oracle tools used by core/compat regression
  gates. Per-library integration instructions and support claims belong to the
  corresponding JittorEcosystem repository; these tools do not establish support
  merely by discovering a case or a repository.
- `lint/`: import-layering and other source checks used by the structure gate.
- `probes/`: maintainer sweeps that print where two equivalent computations,
  devices or round trips disagree. They are not tests; their output is the
  evidence quoted by known-issue entries.
- `release/`: release-only utilities and offline asset packaging.
- `check_repo_layout.sh`: repository layout gate; delegates documentation governance.
- `build/generate_manifest.py`: derive both MANIFEST.in files from their pyproject declarations.

Maintainer checks formerly under `agent/scripts/` now have their sole owners
here. Package resources are declared in each distribution's
`[tool.setuptools.package-data]`; `[tool.jittor.sdist]` adds source-only content.
After changing those declarations or adding source-only files, run
`python tools/build/generate_manifest.py`. The structure gate checks the generated
manifests for drift. Neither build imports Jittor or requires the sibling project.

Development skills are maintained in [Jittor Development Skills](https://github.com/JittorEcosystem/jittor-dev-skills).
Their executable helpers remain here with the source they verify: ACL syntax and binding/import diagnostics
in `build/`, native FSDP evidence tools in `distributed/native_fsdp/`, allocator diagnostics in `probes/`,
and installer inspection in `lint/`. Tests do not need to clone the skills repository.

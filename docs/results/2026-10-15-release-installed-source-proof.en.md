# Installed-source proof for the release scorecard

- Status: the tool-layer gap is fixed; no new model or device coverage is granted.
- Date: 2026-10-15.
- Baseline: `97d7d478a48ecc80f13f2858f0bc8fa78e377480`; this report and the fix are in the same commit.
- Maintainer: Jittor core maintainers.
- Review when: inventory generation, wheel binding, measurement metadata or unified-audit admission changes.

The old audit compared only version strings for ordinary releases and did not verify the actual installed source
in the measurement. In a minimal host counterexample, two records — one with a missing binding and one whose source
did not match — still received L2. After the fix both are rejected; the original counterexample and the fixed
results are kept separately, and the old results are not rewritten.

All registry inventories now record the complete Python source and all tracked package files, with axes and denominators
keeping their original meaning. The unified audit requires source bindings on both sides of base, perturb and training
to cover themselves and the referenced libraries. Non-Python source can serve as a stronger provenance inventory;
declared extra source must match the inventory. When a wheel exists, its metadata and complete installed contents are
checked independently, which can reject records whose Python files are all present but that omit CUDA/data files. The
existing SCM-specific strict bindings and protocol checks are retained.

A read-only re-verification of the real Swift original CPU/CUDA, three seeds and two runtimes — 12 metadata records
in total — passed for 517 Python source files and all 532 package files of the wheel. This wheel was built from the
pinned upstream source and is not claimed as an official PyPI wheel; the original measurement source remains
`606b602e7e79b5bfa5573b9693144f8d224c6a07`, and this read does not re-grant the numerical or device conclusions of that training.

The original counterexample, the read-only re-verification inventory and the test logs are in the unversioned
`$JITTOR_LAB_ROOT/_state/release-source-proof-97d/`. Test entries:

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  tests/tools/test_downstream_coverage.py tests/tools/test_installed_source_proof.py \
  tests/tools/test_component_scorecard.py tests/tools/test_accelerate_protocol_scorecard.py
bash tools/check_repo_layout.sh
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

A historical record lacking the actual installed source cannot enter the numerator under the current strict audit.
Old numerical evidence keeps its original baseline; complete provenance must be collected by a fresh run and cannot be
backfilled from the current installation directory.

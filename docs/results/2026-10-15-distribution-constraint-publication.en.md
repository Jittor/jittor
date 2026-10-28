# Distribution constraint namespace publication

- Status: fixed and verified for frontend publication and bounds checks; no
  general TensorDict compatibility or complete constraint-validation claim.
- Date: 2026-10-15.
- Baseline: `be2ffcff831ca42639525e1c0508ba6ef04d0b77`; implementation is the
  commit containing this report.
- Scope: CPU and real RTX 4090 CUDA; ACL and ROCm unverified.
- Owner: Torch compatibility maintainers.
- Review when: distribution module cloning, native constraint owners, Torch
  namespace publication or the reference version changes.

## Failure and fix

A pure Torch API distribution subclass failed at `constraints.real` before any
TensorDict import. The native namespace already exposed `real`, `interval` and
`dependent_property`, but its frontend clone exposed none of them. Native
constraints are a `ModuleType` subclass with class attributes and static methods;
the frontend copied only `dir(instance)`, which lists module instance attributes.
The installer then saw an existing namespace and correctly avoided replacing it.

The clone now includes attributes from the module's type. This preserves native
constraint and factory ownership, publishes the property decorator, and leaves
the native package untouched. It adds no mathematical implementation and makes
no stronger promise about the native permissive constraints, integer support or
event dimensions. Existing distribution fidelity remains `APPROXIMATE`.

## Validation

Regression cases define a distribution subclass with `arg_constraints` and a
dependent `support` property without importing any downstream library. They
check canonical module identity, inherited owner identity and an unchanged native
namespace. Device cases execute interval, greater-than and less-than predicates,
including a NaN, and require actual CPU/CUDA placement and zero backend fallback.
An independent binary PyTorch 2.11.0+cu128 interpreter, with Jittor absent and no
shim installed, produced exactly the same boolean results on both devices.

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python JITTOR_TEST_DEVICES=cpu,cuda \
  python -m pytest -q \
  compat/tests/torch/test_distribution_package_identity.py \
  compat/tests/torch/test_distribution_frontend_owners.py \
  compat/tests/torch/test_torch_compat_distributions.py

python tools/run_test_suite.py --tier core --serial-compile
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
bash tools/check_repo_layout.sh
```

- CPU distribution selection: 34 passed, no skips.
- CUDA-enabled distribution selection: 35 passed, no skips, including real
  device bounds checks and the existing distribution regressions.
- Complete CPU core runner: 322 passed, 42 skipped, one existing expected failure;
  native and Torch subprocesses both returned zero.
- Complete structure selection with explicit xdist/timeout plugins: 1,432 passed,
  six skipped, no failures. Two skips concern distribution resolution in this
  source-only interpreter; the others are declared or accelerator capability
  skips. Layout and whitespace checks passed.

Python 3.11, GCC 11.5 and CUDA 12.2.140 were used with separate CPU/CUDA caches;
initial compilation was serial. Unversioned logs are under
`$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/`:
`constraints-publication-cpu-recheck.log`, `constraints-publication-cuda.log`,
`constraints-independent-torch.log` and `constraints-publication-core.log`.
The first structure invocation omitted explicit xdist plugin loading while
plugin autoload was disabled; its two subprocess bookkeeping failures do not
establish a source regression. The corrected run is
`constraints-publication-structure-recheck.log`; explicit
`PYTEST_PLUGINS=xdist.plugin,pytest_timeout` supplied the installed plugins.

TensorDict 0.10.0's `TruncatedNormal` class definition exposed this generic
publication bug. End-to-end TensorDict indexing, stacking and training require
their own independent numerical evidence and are outside this report's scope.

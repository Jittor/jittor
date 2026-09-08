# Python Domain Packages

- Status: Domain migration validated against existing failures; 5.26 remains open
- Date: 2026-09-07
- Baseline: `cea23f4cf`
- Owner: coord
- Review when: initialization adapters, domain exports or packaging changes

## Scope

Replace `linalg.py`, `distributions.py` and `init.py` with normal packages and
explicit facades. Linear algebra has eight modules, distributions nine and
initialization five. Each module is below 500 lines. Algorithms retain one owner;
public imports and historical pickle globals continue to resolve. Native
implementation metadata names the new physical modules.

Two composition hazards were checked before acceptance. Torch publication used
to replace the new native `distributions.kl` package attribute with a different
module; the native owner is now `divergence`, leaving `kl` to the adapter.
Kaiming's old late-bound uniform implementation became a fixed native import,
reversing all 12 float32 values in a fixed-seed probe. A shared private Kaiming
helper now accepts the random/writeback implementation explicitly. Native and
Torch callers keep their respective draw behavior without duplicating math.
CPU/CUDA float32/float64 seed, retained-view and generator-error tests cover it.

## Runtime Evidence

Python 3.11, CPU-only `nvcc_path=""`, CUDA `nvcc_path=/usr/local/cuda/bin/nvcc`
(12.2.140, sm89), `PYTHONPATH=python`, `use_mkl=0 use_mpi=0 use_nccl=0`,
`DISABLE_MULTIPROCESSING=0`. CPU and CUDA use separate, unversioned JITTOR_HOME
directories. Torch runs additionally set `JITTOR_TORCH_SHIM=1` and
`JITTOR_TORCH_KEEP_HOME=1`. No shared-cache concurrent JIT.

| Command after `python -m pytest -q --tb=short` | Configuration | Result |
| --- | --- | --- |
| `tests/ops/test_linalg_package.py tests/core/test_init_package.py tests/core/test_distributions_package.py` | Native CPU | 8 passed, 1 accelerator skipped |
| Same three files | Native CUDA-capable | 9 passed, including CPU/CUDA solve values and both gradients |
| `tests/core/test_distributions.py tests/core/test_distributions_grad.py tests/core/test_complex64_linalg.py` | Native CPU | 39 passed, 20 skipped (11 accelerator, 9 Torch) |
| `tests/core/test_init_package.py tests/core/test_distributions.py tests/core/test_distributions_grad.py` | Native, `jt.flags.use_cuda=1` before pytest | 30 passed, 9 Torch skipped |
| `tests/compat/torch/test_distribution_package_identity.py tests/compat/torch/test_torch_compat_distributions.py tests/compat/torch/test_torch_compat_linalg.py` | CPU Torch | 38 passed, 1 failed |
| `tests/compat/torch/test_torch_init_package.py tests/compat/torch/test_distribution_package_identity.py` | CUDA-capable Torch | 11 passed, 0 skipped (CPU/CUDA initialization) |

The CPU Torch failure is
`test_torch_compat_linalg.py::TestLinalg::test_matrix_rank`: scalar arg-reduce
emits an invalid negative-suffix identifier in generated C++. Replacing the
linalg definitions in a fresh process with the exact baseline single-file
implementation reproduces the same node failure. This is not a claim that the
complete baseline suite is green. Native complex tests also reported one live
Var remaining at file teardown; that observation was not suppressed.

## Structure And Packaging

First full structure runs found three new failing nodes: two import-cycle
contracts (180 cyclic modules against the unchanged 164 ceiling), and duplicate
test basenames. The structural tests now have unique names. Runtime-only imports
are now at their call boundaries, and six complex APIs use explicit lazy
re-exports preserving concrete annotations. The unchanged checker reports 164
cyclic modules and all three contracts pass. CPU first run: 1314 collected, 11 failed / 1299 passed /
4 skipped. CUDA first run: 1314 collected, 10 failed / 1302 passed / 2 skipped.
Both used the configuration above plus Torch mode and `--junitxml=<state>/structure-python-packages.xml`.

Final CPU structure (`--junitxml=<state>/structure-python-packages-final.xml`):
1314 collected, 8 failed / 1302 passed / 4 skipped / 0 xfailed, 210.75 seconds.
JUnit comparison to `structure-comm-move.xml` confirms exactly the same eight
failed nodeids, no new or resolved failures. There are 33 added and three removed
nodes: the old single-file environment scans are replaced by package scans;
six new domain structure tests and four previously added ACL/comm contracts
also enter this comparison. The eight baseline failures are listed in the
layout handoff section 1. The full gate is not green.

Final CUDA structure: 1314 collected, 7 failed / 1305 passed / 2 skipped /
0 xfailed, 219.47 seconds. JUnit comparison to
`structure-cuda-final-layout.xml` gives exactly the same seven failures, no new
or resolved failures, 28 added nodes and three removed old-file scans. The
report is `structure-python-packages-final.xml` in the independent CUDA cache.
CPU and CUDA failure sets are compared only to their own configuration.
The layout checker, board contract, unique test names and import-layering
contracts also pass; no failure allowlist or cycle ceiling was relaxed.

After the import fix, the combined six native test files in the table above
(`test_linalg_package`, `test_init_package`, `test_distributions_package`,
`test_distributions`, `test_distributions_grad`, `test_complex64_linalg`)
pass 47 / skip 21 on CPU, and pass 58 / skip 10 with CUDA selected. CUDA skips
are nine Torch-mode prerequisites and the explicit unsupported CuPy general eig.
CPU Torch initialization/package identity/distribution tests pass 33 / skip 0.
The same selection in CUDA-capable Torch mode passes 38 / skip 0.

The first package artifact passed official sdist validation (1421 members),
22 domain files and the initialization adapter were byte-identical across
source/sdist/wheel, and a separate installation passed CPU cold build, domain
values/gradients and all 13-subpackage/three-training-step selftest checks.
Its SHA-256 was `90fe9e4666b94876bcdba5e4d8a3fb81729c8dcde0dd59fe55305f1c8e7990ce`;
this is not the final artifact. After the import-cycle fix and whitespace
cleanup, the rebuilt wheel SHA-256 is
`1be6f8bdd22c05eeb41062bd87e3df46196278234c924b544a950c05cc22d3dc`.
All 22 domain files and two compatibility adapters are byte-identical across
live source/sdist/wheel/installation; official sdist checks still pass 1421
members. The post-cycle-fix installation independently cold-built 205 TUs, then
the final whitespace-only update was installed at the same external target;
its 13-subpackage/three-training-step selftest passes. Artifacts and caches are
unversioned under `_state/python-domains-final-*`, never package-local.

NPU/ROCm/Corex hardware is not validated. No performance claim is made.
This batch does not complete 5.26: root/core API, miscellaneous tensor ops,
pooling, build utilities and contributed resources still need their migrations.

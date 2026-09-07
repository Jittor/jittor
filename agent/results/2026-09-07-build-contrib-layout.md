# Build, Contrib And Package Root Layout

- Status: Native layout and isolated package checkpoint verified; broad gates deferred
- Date: 2026-09-07
- Baseline: `65a4189e6`
- Owner: coord
- Review when: standalone build imports, module aliases or runtime resources change

## Delivered Layout

The package root now has only `__init__.py`, `__init__.pyi` and `selftest.py`.
Compiler and optional-library setup live in `build/`, together with pyjt and
CuPy initialization. Argument policy belongs to `_core`, startup composition
and installation order to `_runtime`, and the public timing helper to `tools`.
Historical module names resolve through the existing fixed alias registry.

Compiler state stays in `build/compiler.py` (1472 lines). Nineteen generation
and compilation functions move to `codegen.py` (626) and `compilation.py` (534).
They read the original compiler state explicitly, including cross-function
lookups, preserving late patching and the two lock decorators. Both files enter
the generator fingerprint, so future edits invalidate the build stamp.

Build utilities have one physical owner under `build/utils`. The standalone
`jittor_utils` entry is a small path loader that does not initialize Jittor;
its module and pickle spelling stay stable. `jittor.build.utils` and all utility
submodules are same-object aliases, preserving one cache, lock and configuration.
Version discovery, cache identity, bootstrap PYTHONPATH and source/header
discovery use the relocated paths. The source package is explicitly exempted
from the generated `build/` Git ignore rule. CPU/CUDA CI cache keys now include
top-level `src`, `backends` and the real build package.

CCL, 3D losses, special math and vendored einops live under `contrib/`.
Old packages/submodules remain aliases. The shared igamma header and classroom
utility resources move with their consumers and MANIFEST declarations.
Same-named function/submodule exports retain their package functions when
aliases are published. EinopsError has a separate owner to remove its parser
cycle; public and old pickle globals still resolve to that class.

Every non-compat Python file under `python/jittor` is now below 1500 lines. Four compatibility files
remain above the threshold: Torch nn/tensor/numerical installers and the shim
FlashAttention bridge. Their architecture and the independent Torch package
are still unfinished; 5.26 is not marked complete.

## Focused Evidence

Python 3.11, `PYTHONPATH=python`, `use_mkl=0 use_mpi=0 use_nccl=0`,
`DISABLE_MULTIPROCESSING=0`; independent CPU/CUDA JITTOR_HOME directories.
CPU uses `nvcc_path=""`; CUDA uses `/usr/local/cuda/bin/nvcc` 12.2/sm89 and
explicit `use_cuda=1`. CPU Torch additionally uses `JITTOR_TORCH_SHIM=1` and
`JITTOR_TORCH_KEEP_HOME=1`.

- Standalone utility import asserts that `jittor` is absent from `sys.modules`.
- CPU, CUDA and CPU Torch processes pass module/lock/config alias identity,
  preserved CCL package functions, old einops error pickle, matrix forward and
  finite gradients, einops transpose and lgamma against explicit NumPy values.
- After compiler extraction, the existing
  `tests/compiler/test_custom_op.py::TestCustomOp::test_compile_custom_op`
  actually compiles and executes a custom operator: 1 passed, 0 skipped.
  The CUDA short probe is also repeated successfully on the extracted compiler.
- The import checker follows the fixed alias registry and scans the real
  utility implementation rather than only the loader. Tool/framework layering
  passes, with 147 cyclic modules in two components; no ceiling is relaxed.
- Existing source-location/fake-compiler contracts are updated without new
  test files or a repeat of the full repository suites.

## Packaging And Remaining Validation

The final sdist has 1496 members. All 1111 production files (402 Python,
349 backend and 360 core) are byte-identical across live source, snapshot,
sdist, wheel and isolated installation. Core/backend/class inventories are
complete; the igamma header is at its canonical contrib location. The installed
package root contains exactly three files, and the standalone utility package
contains just its loader. A separate JITTOR_HOME cold-built 205 translation
units, then passed square/gradient checks and the 13-subpackage, three-step
training selftest. Standalone utility import still does not load Jittor.

Wheel SHA-256:
`b0bd4ecc36203b0de4bdc264a39a0908a8c2ec8ac1e58009339fe205d3ef980d`.
Unversioned artifacts and logs are under `_state/native-final-package-h0M9Ef`.
The historical wheel approval baseline was not refreshed. Complete CPU/CUDA
structure and broad numerical suites remain deferred under the user's requested
faster workflow. No NPU/ROCm/Corex hardware execution or performance improvement
is claimed.

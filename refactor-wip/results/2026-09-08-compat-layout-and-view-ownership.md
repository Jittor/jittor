# Compatibility Layout And View Ownership

- Status: 5.26 layout complete; independent Torch architecture remains open
- Date: 2026-09-08
- Baseline: `4d70f1902`
- Owner: coord
- Review when: installer state, publication, native views or shim resources change

## Layout Completion

The remaining four oversized files become normal packages: Torch nn,
numerical and tensor installers, and the FlashAttention shim backend. Their
children own actual operation families, method installation, source generation,
build integration and adaptation. There is no hidden replacement monolith.
All Python files under `python/jittor` are now at most 1472 lines; the package
root still contains only `__init__.py`, `__init__.pyi` and `selftest.py`.

Installer order, native callable captures, fidelity registration, per-install
attention cache and FlashAttention cache/environment/lock ownership remain
single-source. Moved functions read their original owning facade at call time
where late patching matters. FlashAttention's generated C++/CUDA string values
are unchanged, and the maintained resource manifest lists all five source
files. Pure RNN, normalization and functional helpers become stable module-level
objects. Existing public and historical pickle paths remain resolvable.

## State And Correctness

Native VarView already owns basic-index ancestry and assignment writeback.
Torch now completes that native record for its additional basic-slice spellings,
instead of retaining `_torch_index_parent`/`_torch_index_slices` and recursively
writing the same ancestors again. `out=` and initialization delegate to native
assignment. Advanced indexing remains a copy; `.data` retains its separate
owner/path semantics.

A pre-existing `.data.normal_()` bug was reproduced with the exact baseline
installers loaded in a fresh process: the original parameter became frozen.
The refreshed data alias was assigned the same Var as its owner, allowing the
alias's gradient flag to affect the owner. Assigning a detached RHS to the data
alias preserves their distinct gradient state. The existing test also assumed
that `ones()` enabled gradients; its parameter-specific case now explicitly
enables them, while frozen-tensor cases remain unchanged.

The separate requires-grad dictionary had production writes but no readers.
It and its clearing-only stop_grad wrapper are removed. Native descriptors,
FSDP peer synchronization, and the leaf/retained/optimizer registries actually
used by backward are retained.

## Verification

Python 3.11; `PYTHONPATH=python JITTOR_TORCH_SHIM=1 JITTOR_TORCH_KEEP_HOME=1`,
`use_mkl=0 use_mpi=0 use_nccl=0 DISABLE_MULTIPROCESSING=0`.
CPU uses `nvcc_path=""`; CUDA uses `/usr/local/cuda/bin/nvcc` 12.2/sm89.
CPU and CUDA processes use different JITTOR_HOME directories.

- Existing data/view/initializer/cumulative/state checks: CPU 27 passed;
  final CUDA-capable selection 45 passed, zero skipped, covering CPU and CUDA.
- No-optimizer backward, RNN pack/pad roundtrip, vstack and matrix multiplication
  against explicit NumPy values pass in both CPU and CUDA scopes.
- One complete CPU-configured structure run: 1407 collected, 15 failed,
  1388 passed, 4 skipped, 0 xfailed; 243.18 seconds. The JUnit is unversioned at
  `<layout-cold-state>/structure-complete-python-layout.xml`.
- Eight failures are the previously recorded baseline. Seven new failures
  identified two duplicate implementations, stale source/resource contracts,
  a redundant alias file, and a cold-import test that ignored lazy public
  attributes. These are fixed; the seven nodes plus the related stub contract
  pass together: 8 passed in 6.44 seconds. The full suite was not repeated.
- The duplicate native runtime workaround now re-exports the standalone-tools
  implementation. Two no-op parameter helpers become direct constructor field
  assignments. The obsolete compat/contrib file is removed; the fixed module
  alias remains. Checks were not relaxed with new failure allowlists.

## Remaining Scope

The final sdist has 1528 members. All 1139 production files are byte-identical
across source, snapshot, sdist, wheel and installation. The package root has
three files; every `python/jittor` Python file is within the 1500-line limit;
the four old single files are absent and their full packages are present.
FlashAttention's five files and all core/backend resources are included.
The upgraded installation reuses a previously built core cache (not a new cold
build) and passes CPU Torch view/data/backward checks with the exact expected
gradient `[[0, 4], [0, 8]]` and unchanged input values.

Wheel SHA-256:
`ddb76df749952bf7becfc31c8b0cab05f35c35cbd50833233d4a7d9f1b937244`.
Unversioned artifacts are under `_state/compat-final-package-KdACPf`.
The unrelated historical wheel approval baseline was not refreshed.

5.26's physical organization requirements are satisfied. This does not complete
7.12: TorchNamespace still delegates to an owner, activation still patches the
native Var/NN classes, and full independent Torch semantics remain work ahead.
The original structure failures are not claimed as fixed. A complete CUDA
structure run and broad ecosystem/model validation are not claimed for this
batch; NPU and other unavailable hardware remain unverified.

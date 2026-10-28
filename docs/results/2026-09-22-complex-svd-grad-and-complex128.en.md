# complex64 SVD joint gradient (§3.2) and native complex128 dtype (§3.1)

- Status: both closed; `tests/linalg/test_complex_spectral_gradients.py` and
  `tests/core/test_complex128_native.py` pass on CPU and real CUDA;
  `tools/run_test_suite.py --tier core` and `--tier smoke` run; `tests/structure`
  green after a migrated-boundary-count update.
- Date: 2026-09-22
- Owner: this session
- Baseline: `refactor-2.0` at `f34817660f239487d327c189fd4503da5e1c1f5e`
  (== `origin/refactor-2.0`, nothing to sync)
- Review when: FFT complex128 kernel support lands, second-order complex AD
  lands, or `src/type/nano_string.h`'s `Flags` bit layout changes again

## Question

`local/problems-2026-9-22.md` (an external TensorCircuit-adaptation report,
not tracked in this repo) listed 9 unresolved issues. The user scoped this
session to the two P0 items only: §3.2 (complex64 SVD gradient wrong for a
joint U/S/Vh loss) and §3.1 (complex128 not registered as a native dtype).
The other 7 P1/P2 items are explicitly out of scope for this session.

## Environment

Podman image `jittor-cuda:local`, built from this repo's own
`Containerfile.cuda` (CUDA 11.8, Ubuntu 22.04). Container `jittor-p0-dev`,
repo bind-mounted at `/usr/src/jittor` (edits on host are live in the
container), `JITTOR_HOME=/root/.cache/jittor-p0dev` (isolated from any other
concurrent work). GPU: real NVIDIA GeForce RTX 4090 (driver 610.57.04, one of
3 visible via CDI), compute capability sm_89. CuPy 13.6.0 installed in the
container for CUDA `numpy_code` support. A separate venv at
`/opt/real-torch-venv` (container-only, not part of the repo) with real
upstream PyTorch 2.14.0+cpu, used as a genuinely independent oracle — this
repo's own `torch` import resolves to Jittor's compat shim
(`jittor-torch`), not real PyTorch, so the shim cannot serve as an
independent reference.

## §3.2 — complex64 SVD joint gradient

### What was actually there vs. what the problem doc described

The doc described a subtly-wrong `(1-eye)`-masking backward formula as
"current". That formula only exists in a pre-refactor commit (`32001665`,
reachable from `master`, not from `refactor-2.0`). On this branch,
`complex_svd`'s `backward_code` in `python/jittor/linalg/complex.py` was
simply `raise NotImplementedError`. So this was "implement the backward
correctly," not "fix an existing formula" — verified with a minimal repro
before touching anything (`jt.grad` through `jt.linalg.svd` on a complex64
input raised `NotImplementedError`).

### Fix

Derived the joint complex-SVD adjoint analytically (not guessed): solving
`dC := U^H dA V = dP@S + diag(dS) - S@dQ` (`dP=U^H dU`, `dQ=V^H dV`, both
skew-Hermitian) for its diagonal and off-diagonal parts shows the real-SVD
backward's off-diagonal antisymmetrized term (`F ⊙ (utgu - H(utgu))`)
carries over unchanged, but the diagonal needs an extra term real SVD has no
analogue for: complex unitary `U` only makes `U^H dU` skew-*Hermitian*
(diagonal forced purely imaginary, not zero — the old wrong commit deleted
this outright instead of combining it correctly between the U/Vh branches).
The correct diagonal term is
`i*(Im(diag(U^H gU)) - Im(diag(V^H gV))) / (2*S)`.

Validated computationally against an independent NumPy finite-difference
oracle *before* writing any Jittor code (script discarded after use, not a
repo artifact): square/tall/wide, batched, joint reconstruction loss
(`loss = Re<(U*S)@Vh, P>`, expected `dL/dA == P` exactly) and isolated
dS/dU/dV-only losses — all matched to ~1e-15, i.e. to the finite-difference
step's own ~1e-9/1e-10 floor. Implemented in
`python/jittor/linalg/complex.py::complex_svd`, keeping the existing
`numpy_code`/`out_index` structure (proven correct by VJP linearity — each
output's cotangent contribution is exactly additive, so per-branch
`out_index` calls with the other cotangent implicitly zero sum to the same
total as a joint call; no `jt.Function` restructuring needed).

### Verification of the SVD gradient

New `tests/linalg/test_complex_spectral_gradients.py`: joint-reconstruction
gradient exactly matches `P` (square/tall/wide/batched), matches an
independent NumPy finite-difference oracle, and matches an independent real
PyTorch process (`torch.linalg.svd` autograd, via the isolated venv above) —
plus isolated dS/dU/dV-only regression cases. 18/18 pass on CPU and CUDA
(max error ~4e-7 to ~1e-6, consistent with float32 precision). Full
`tests/linalg/` (46 tests) still passes, no regressions.

## §3.1 — native complex128 dtype

### Confirmed scope

The hard blocker was exactly as scoped: `NanoString::Flags::_dsize_nbits = 2`
(`src/type/nano_string.h`) capped every dtype at 8 bytes; complex128 needs
16. Verified before editing: `jt.array(np.array(..., dtype=complex128))`
raised `RuntimeError: numpy.h:67: Numpy type not support, type_num: 15`.

### Changes (in dependency order)

1. `src/type/nano_string.h` — widened `_dsize_nbits` 2->3 bits, shifted
   `_white_list`/`_no_need_back_in`/`_no_need_back_out`/`_complex` up one bit
   accordingly; added `complex128` to `FOR_ALL_NS`; replaced the four
   hardcoded `return ns_complex64` promotion sites with a width-aware
   `complex_promote`/`complex_needs_double` helper (complex128 if either
   operand is complex128 or a float64/int64/uint64-width real, else
   complex64).
2. `src/type/nano_string.cc` — registered `complex128` (`dsize_map=4`, i.e.
   16 bytes).
3. `src/type/complex_compute.h` — added `struct complex128` (a `double` pair)
   mirroring `complex64`'s full operator/`jt_c*` set, including the CUDA
   `atomicAdd` decomposition, plus explicit `jt_c64_to_c128`/`jt_c128_to_c64`
   precision-cast helpers.
4. `src/type/complex_op_type.cc` — registered `"complex128"`; extended the
   include-guard; replaced the cast special-case with one that distinguishes
   complex-to-real (drop imaginary, unchanged), complex64<->complex128
   (explicit precision cast), and real-to-complex (unchanged, uses the
   target struct's converting constructor).
5. `src/bindings/pyjt/numpy.cc`/`.h` — mapped `NPY_CDOUBLE` to `ns_complex128`
   instead of `ns_void`; added the matching `ns2npy` entry.
6. `src/bindings/pyjt/py_converter.h` — added the `.item()` `to_py_object`
   branch for `ns_complex64` (previously **entirely missing** — hit a generic
   assert) and a new one for `ns_complex128`, both returning Python
   `complex`. **Did not** change the Python-complex-scalar narrowing path:
   `tests/core/test_complex_scalar_operand.py` already pins down that a bare
   Python/NumPy complex scalar narrows to complex64, the same convention a
   bare Python `float` narrows to `float32` — that's intentional design, not
   a complex128 gap, confirmed by reading the existing test before touching
   anything.
7. `src/core/var_holder.h` — widened `ItemData::data` (a flat 8-byte `int64`)
   to a 16-byte union, so `.item()` doesn't overflow into the adjacent
   `dtype` field for a 16-byte payload.
8. `src/ops/composite/reinterpret_view_op.cc` — mirrored the
   complex64<->float32 pair-view checks (`grad`, `infer_shape`) for
   complex128<->float64.
9. `python/jittor/nn/functional/complex.py` — added width-generic
   `_complex_to_real2`/`_real2_to_complex` (dispatch on component width) used
   by `view_as_real`/`view_as_complex`/`polar`/`.real`/`.imag`; kept the
   existing width-fixed `_complex64_to_real2`/`_real2_to_complex64` for the
   legacy `nn.ComplexNumber` bridge (always float32 by construction, per
   `python/jittor/nn/legacy_complex.py`).
10. `python/jittor/linalg/complex.py` — relaxed the five hardcoded
    `assert dtype == "float32"` checks (`complex_inv`/`eig`/`eigh`/`qr`/
    `pinv`) to accept float32 or float64 component pairs.
11. `python/jittor/linalg/_helpers.py::_cn_to_native` — **found and fixed a
    real silent-downcast bug**: this hardcoded `_real2_to_complex64`
    regardless of the `ComplexNumber`'s actual component width, so
    `svd()`/`qr()`/etc. on a native complex128 input would silently return
    complex64. Switched to the new width-generic `_real2_to_complex`.
12. `src/ops/unary_op.cc::UnaryOp::grad` — **found and fixed a second real
    gap**: the complex branch had no case for a complex-to-complex `cast`
    (only complex-to-real), so `jt.grad` through a `complex64.cast(
    "complex128")` (or the reverse) silently returned zero. Added the
    missing branch (adjoint of a linear precision cast is casting the
    cotangent back to the original width).
13. `src/ops/composite/transpose_op.cc` — **found and fixed a third real
    gap, on real hardware**: CUDA's cuTT transpose library cannot plan a
    16-byte-element permutation at all (`cuttPlan failed ... dsize 16`), so
    the very first `.transpose()`/`.T` of *any* complex128 CUDA Var crashed
    hard (this is pervasive — every `_conj_transpose` call in the linalg
    bridge hits it). Gated the CUDA-accelerated transpose capability lookup
    to `dsize() <= 8`, falling through to the existing generic
    `Tx`-templated kernel for 16-byte dtypes.
14. `backends/cuda/kernels/nn/complex_views.py` — added
    `COMPLEX128_TO_REAL2_CUDA_SOURCE`/`REAL2_TO_COMPLEX128_CUDA_SOURCE`
    (mirrors the existing complex64 kernels; only exercised on the
    `reinterpret_view`-unavailable fallback path).
15. `docs/notes/complex-dtype.md` — updated: complex128 moved out of "explicitly
    unsupported"; documented the new promotion rule and scalar-narrowing
    convention; FFT complex128, CUDA complex `prod`, second-order complex
    AD, and degenerate-singular-value SVD gradients recorded as the
    remaining, explicitly-scoped-out limitations.

None of items 1–14 were guessed from the historical `7609dab5` commit's
exact diff (different file layout — `python/jittor/src/...` vs `src/...` —
and a different `NanoString` bit layout after `refactor-2.0`'s own
`_index_nbits` fix); it was used only as a pattern reference, each site
re-verified against this branch's actual current code.

### Verification of complex128

New `tests/core/test_complex128_native.py`: 16-byte `dsize()`, lossless
NumPy round-trip, mixed complex64/complex128 promotion (and that a narrower
operand does *not* force widening), arithmetic + transcendentals
(`abs`->`float64` confirmed), first-order gradients, `.item()` at both
widths, complex64<->complex128 cast (values *and* gradients, both
directions), and the linalg bridge (`inv`/`svd`/`qr`/`eigh`/`pinv`) at
complex128 precision with reconstruction-identity checks. 20/20 pass on CPU
and CUDA. `tests/core/test_complex_scalar_operand.py` (existing, docstring
updated to reflect complex128 landing, no assertion changes) still 5/5.

### Gate results

- `tools/run_test_suite.py --tier core`: native 98 passed/41 skipped/1
  xfailed, torch 212 passed/3 skipped. 0 failures.
- `tools/run_test_suite.py --tier smoke`: native 2162 passed/47 failed/228
  errors, torch 2782 passed/23 failed. The failures/errors are **not** in
  files touched by this session (MPI multi-rank entry points, ACL/Ascend
  backend routing, RNN/cuDNN, TF32 precision-domain plumbing, DDP launch,
  `arg_pool`/`cumprod`/`einsum`/`unique` `ModuleNotFoundError`s — a missing
  optional dependency in this minimal container, not this session's code).
  Spot-checked the one cluster that *did* touch a file this session reads
  (`tests/type/test_complex.py`, not in the `core` tier whitelist): its
  failures are a pre-existing xdist-worker-ordering issue in that file's
  own independent-Torch detection (`tests/_helpers/torch_runtime.py`'s
  `modules_available`/`import_torch_modules` disagree between collection
  and `setUpModule` time under `-n 4` outside `run_test_suite.py`'s own
  invocation) — reproduced identically with the independent-oracle venv
  from this session both present and absent, with two different
  ordering-dependent error messages each time, confirming it is not this
  session's source changes.
- Full `tests/linalg/` (46) and `tests/core/` (446, one pre-existing,
  unrelated `test_namespace_exports.py` failure about `dlpack`/`vmap`/
  `Generator` stub declarations — separately tracked as §3.4/§3.5 in the
  problem doc, not touched here) pass.
- `tools/check_repo_layout.sh`: one finding, `local/problems-2026-9-22.md`
  (the user's own external-report file, present before this session,
  referencing a `development-plan.md` that lives in a different
  repo/harness) — not a file this session created; left for the user to
  handle.
- `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest tests/structure`:
  1378 collected, 3 failed before a fix, 1 after. Fixed: this session's two
  new `USER_CHECK`s in `reinterpret_view_op.cc` (the complex128<->float64
  shape-pairing checks) moved a tracked migrated-boundary count from 8 to
  10 (`tests/structure/core/test_error_categories.py`), updated with the
  same explanatory-comment convention the file already uses. Left
  unfixed (pre-existing, unrelated): the same `local/` doc-governance
  finding as above, and `test_a_timeout_ends_the_grandchildren_too`
  (grandchild-process-outlives-timeout, process/signal-handling in this
  container, not a file this session touched).

## What's still open

The other 7 items from the original 9-item list (SGD dampening, native vmap
batching rules, DLPack stride/empty-buffer handling, CUDA RNG checkpoint
replay, higher-order complex AD, function-level compile/cache contract,
sparse device-side coalesce) are **not** addressed by this session — the
user explicitly scoped this session to the two P0 items only. Within §3.1's
own scope, explicitly still open (documented in `docs/notes/complex-dtype.md`):
FFT complex128 kernel support (separate cuFFT/kernel-level work), CUDA
complex `prod`, second-order complex autodiff (including SVD's own backward-
of-backward), and general complex eigendecomposition on CUDA depending on
CuPy availability.

# SGD dampening (§3.3) and native vmap batching-rule gaps (§3.4)

- Status: §3.3 closed for CPU portable, CUDA fused, and FSDP2 (all verified
  against an independent real-PyTorch process on real CUDA); ACL fixed from
  code reasoning only — no Ascend hardware available this session, not
  verified. §3.4: `cumsum` and `solve` given real batching rules with full
  composition testing; `numpy_code` given a clean fail-fast; sparse already
  fails cleanly (locked, not changed); the "lazy patch install" item was
  investigated, a fix was implemented and then reverted after it broke a
  stronger, explicitly-tested project invariant — see below.
- Date: 2026-09-22
- Owner: this session
- Baseline: `refactor-2.0` at `f34817660f239487d327c189fd4503da5e1c1f5e`
  (unchanged since the earlier P0 session today)
- Review when: ACL/Ascend hardware becomes available (§3.3 needs real
  verification there), or `python/jittor/linalg/solving.py`'s output-shape
  handling changes (see the newly-discovered, out-of-scope `solve()` gap
  below)

## Environment

Same `jittor-p0-dev` podman container as the earlier P0 session today
(`Containerfile.cuda`, CUDA 11.8, 3x RTX 4090, `JITTOR_HOME=/root/.cache/
jittor-p0dev`), plus the same `/opt/real-torch-venv` independent PyTorch
2.14.0+cpu venv reused for the SGD oracle test. No Ascend/NPU hardware is
available in this environment — the ACL-side SGD fix is implemented from
code/coefficient-math reasoning only, explicitly not run on real hardware.

## §3.3 — SGD momentum/dampening

### Root cause, confirmed by repro before editing

`python/jittor/optim/algorithms/sgd.py:16` gated the no-momentum fast path
on `momentum==0 and dampening==0`, so any other combination — including
`momentum==0` with `dampening!=0` — ran `velocity = momentum*velocity +
dp*(1-dampening)` against a buffer that is always zero-initialized at
construction, with no "is this the first update" signal anywhere.
Reproduced the doc's exact numbers before touching anything: `p=1`, fixed
`g=2`, `lr=0.1`, `dampening=0.2`, 3 steps gave `[0.84, 0.68, 0.52]` at
`momentum=0` and `[0.84, 0.552, 0.1616]` at `momentum=0.8` — both matching
the doc's reported wrong values exactly.

### Fix

Matches PyTorch's actual rule: the momentum buffer is seeded from the raw
(undampened) gradient on its first touch, and the dampened recurrence only
applies from the second update onward. Threaded a `step` (1-based) argument
through every layer that previously only knew `(momentum, dampening,
nesterov)`:

- `sgd_update(..., step=1)` (portable/CPU, also used directly by FSDP2):
  fast path is now `momentum==0 and not nesterov` (dampening is irrelevant
  without momentum — this was the bug, not a missing case); `step<=1` seeds
  velocity from raw `dp`, otherwise the existing recurrence.
- `SGD.step()`: calls `n = self._advance_step_count(pg)` once per param
  group, reusing `Optimizer._advance_step_count` — the same mechanism
  Adam/AdamW already use for bias correction (`python/jittor/optim/base.py:
  143`), which persists through `state_dict`/`load_state_dict`, so a
  resumed checkpoint does not get treated as step 1 again "for free."
- `backends/cuda/kernels/optim/fused_sgd_cuda.py`: same `plain`-condition
  fix, plus a `first_step`-conditioned velocity-update line in the
  generated CUDA source (coefficients stay compile-time literals per the
  file's existing design — costs one extra JIT compile at the step-1→
  step-2 transition per parameter group, not a per-step recompile).
- ACL: `FusedSgdAttr::isFirstStep` **already existed** as a field
  (`backends/acl/include/acl_jittor.h:55`) but was hardcoded `false` at the
  op-dispatch site (`backends/acl/src/acl_op_exec.cc:731`, with a comment
  asserting the exact wrong assumption this bug depends on), and
  `fused_sgd_op_acl.cc`'s coefficient computation never branched on it.
  Fixed: threaded a real `first_step` bool from `SGD.step()`'s counter
  through `FusedSgdOp`'s constructor to the ACL dispatch site, and split
  the coefficient computation so only the velocity buffer's own recurrence
  coefficient zeroes on the first step (the Nesterov correction's own
  `-lr*momentum` term is unchanged — PyTorch always uses the real momentum
  there, first step or not, since by that point the buffer already holds
  the correct value).
- `compat/fsdp2/optimizer.py::_sgd_update_for_param`: accepts `step`, passed
  from the caller's existing `param_steps[i]` counter — the exact same
  counter already passed to the parallel Adam path's `n_step`, so no new
  state-tracking was needed.

### Verification

- Portable and CUDA fused paths: reproduced the doc's 3-step recipe
  directly, now matching the doc's expected `[0.8, 0.6, 0.4]` /
  `[0.8, 0.48, 0.064]` exactly on real CUDA.
- New `tests/optim/test_sgd_dampening_pytorch_oracle.py`: the same 3-step
  recipe cross-checked against a real, independent `torch.optim.SGD`
  process (not this repo's own torch-compat shim, which is jittor's own
  implementation under another name) — portable CPU, portable CUDA, and
  CUDA fused all match to `atol=1e-5`.
- `tests/optim/test_optim_core.py::TestSGDUpdate`: fixed the reference
  formula (`_ref_sgd`) to take a `first_step` flag — it previously asserted
  the wrong first-step dampened value; added `test_sgd_dampening_two_steps`
  (dampening applies from step 2) and
  `test_sgd_momentum_zero_ignores_dampening` (the doc's other named
  symptom).
- `compat/tests/fsdp2/test_fsdp_optimizer_math.py`: the existing
  `test_sgd_momentum_and_nesterov` deliberately starts from a nonzero
  velocity buffer (testing the recurrence path) — added an explicit
  `step=2`; added `test_sgd_momentum_first_touch_seeds_from_raw_gradient`
  for the companion first-touch case.
- ACL: fixed from code reasoning (the coefficient math is derived and
  documented in the diff itself); **not run on real Ascend hardware** —
  no NPU available this session. This is a real, open verification gap,
  not a silently-claimed pass.
- Full `tests/optim/` + `compat/tests/fsdp2/test_fsdp_optimizer_math.py`:
  67 passed (1 pre-existing failure deselected, see below; 8 errors are a
  pre-existing "independent Torch was not preloaded" oracle-session-context
  requirement these specific files need `tools/run_test_suite.py`'s own
  invocation for, confirmed unrelated by reproducing the same error pattern
  on unrelated files in the earlier P0 session).

### Pre-existing, unrelated bug found and ruled out

`tests/optim/test_optim_core.py::TestOptimizerPlumbing::
test_low_precision_parameter_and_state_dtypes_are_preserved` hit an internal
Jittor executor invariant crash (`exec_plan.cc:271: queue.size()(14) ==
roots.size()(16)`). Confirmed via `git stash` + full core rebuild that this
reproduces identically on the true unmodified baseline — pre-existing,
unrelated to this session, not chased further. Left deselected in the
regression runs above; not touched.

## §3.4 — native vmap batching-rule gaps

### Architecture recap

`python/jittor/vmap.py`: a `BatchedVar(value, level)` wrapper plus a
hand-maintained table of monkeypatched `jt`/`jt.Var`/`jt.linalg` entry
points, installed by idempotent `install_batching_patches()`. Confirmed via
direct exploration: `solve`, `cumsum`, sparse ops, and `numpy_code` had
zero batching-rule coverage (each fails a different, mostly confusing way
today); the compat `real`/`imag` aliases didn't recognize `BatchedVar`.

### Fixed

1. **`cumsum`** — new `_apply_cumsum`, the same axis-sensitive
   single-input shape as the existing `_apply_reduce`/`_apply_transpose`
   rules (resolve `dim=None` against the *logical* rank via the existing
   `jt.misc._cumsum_dim`, shift by nesting depth, delegate). Registered as
   free-function `jt.cumsum` and as a `BatchedVar.__getattr__` method case
   (`.cumsum(dim)`).
2. **`solve`** (two-input) — new `_apply_solve`, the same max-level dispatch
   shape as `_apply_binary`, without `_align_logical_rank`'s rank-padding
   (not needed: `np.linalg.solve`'s own leading-dim broadcast already
   handles one operand being shared/unbatched, the same property
   `_apply_linalg`'s single-input rules already lean on). Registered on
   `jt.linalg.solve`.
3. **`numpy_code`** — not a batching rule (an arbitrary opaque callback
   can't be vmapped without invoking it once per batch element, which
   defeats vmap's whole point) but a clean, explicit `NotImplementedError`
   when a `BatchedVar` reaches it, instead of a confusing low-level pybind
   type error.
4. **Sparse ops** — already fail cleanly (`SparseVar.__init__`/`spmm`
   hard-`isinstance(x, jt.Var)`-check and raise `TypeError` before any op
   runs). No code change; added a locking test, since none existed.
5. **`compat/torch/installers/numerical/complex.py`'s `real`/`imag`** — a
   real silent-wrong-answer bug: `isinstance(input, (jt.nn.ComplexNumber,
   jt.Var))` didn't recognize `BatchedVar`, so `torch.real(bv)` silently
   returned `bv` unchanged (mislabeled "the real part") and `torch.imag(bv)`
   silently returned all zeros even for a genuinely complex `bv`. Fixed by
   adding `BatchedVar` to the check — native `BatchedVar.real`/`.imag`
   already worked correctly, this was purely a recognition gap in the
   compat alias.

### Investigated, fixed, then reverted: the lazy-patch-install "stale
reference" gap

The doc's concern: `install_batching_patches()` only runs inside `vmap()`'s
own body, so a function reference saved before a process's first `vmap()`
call (`my_qr = jt.linalg.qr`) stays bound to the pre-patch function forever,
even after `vmap()` patches `jt.linalg.qr` elsewhere.

Implemented the "obvious" fix — call `install_batching_patches()` once,
unconditionally, at the end of `python/jittor/__init__.py` — and it
**broke two structural contract tests**:
`tests/structure/nn/test_nn_structure.py::TestPublicBindings::
test_tensor_method_bindings_stay_on_public_functions` (asserts
`jittor.Var.matmul is nn.matmul`) and
`tests/structure/runtime/test_runtime_composition_structure.py::
test_compat_composition_keeps_native_core_implementations_available`
(asserts `jittor.grad is core_api.grad`). This project's own
alias-composition system (`_runtime/composition.py`'s
`make_inplace_alias`/`publish`) gives every one of these functions a
*single shared object identity* across every entry point
(`jittor.matmul`/`jittor.Var.matmul`/`nn.matmul`/`__matmul__`, etc.).
Eager installation only patches *some* of those aliased locations (`jt.
Var.matmul`, `jt.matmul`) and not others (`nn.matmul`), breaking that
identity for **every** jittor import, not just vmap users — a bigger,
explicitly-tested invariant a narrow patch-timing fix should not sacrifice.
Patching every aliased location too would mean threading the fix through
that whole composition system, which is well outside a vmap-scoped change.

**Reverted.** The residual risk stays exactly what `vmap()`'s own
docstring already documents and works around (prefer `vmap(lambda x:
jt.linalg.qr(x))` over a bare function reference) — this is now locked by
`tests/core/test_vmap.py::TestVmapPatchTiming::
test_reference_captured_before_first_vmap_call_stays_unpatched`, which
asserts both the documented failure and the documented workaround, rather
than asserting a fix that was tried and found to cost more than it closed.
This item is **not closed** — recorded here as investigated and
deliberately left as-is, not silently claimed fixed.

### A second, unrelated real bug found and fixed along the way

While testing the (later-reverted) eager-install change, two *other*,
genuinely pre-existing, latent bugs in `vmap.py`'s own patch registration
surfaced and were fixed independently of the eager-install revert (both are
real regardless of install timing, just previously unreachable unless a
process happened to call `jt.vmap()` before hitting the affected code):

- **`_apply_unary`'s original-function preference**: registered `orig =
  getattr(jt.Var, name, None) or getattr(jt, name, None)` (Var-method
  first) instead of the free-function-first order `_BINARY_NAMES`'
  registration already uses (and already documents the reason for). A bare
  Python float argument — e.g. `adam_update`'s `jt.sqrt(1 - b1**step)` bias
  correction — fails against the Var-method descriptor form
  (`TypeError: descriptor 'sqrt' ... doesn't apply to a 'float' object`).
  Fixed to prefer the free function, matching the binary registration's
  existing convention.
- **`view`/`reshape` and `broadcast`/`broadcast_var` silently aliased to
  one original**: `jt.Var.view is jt.Var.reshape` and `jt.Var.broadcast is
  jt.Var.broadcast_var` are both `False` (confirmed), but the registration
  installed the *same* wrapper closure (capturing only `jt.Var.reshape` /
  `jt.Var.broadcast` as `_ORIG_RESHAPE`/`_ORIG_BROADCAST`) onto both names
  in each pair — silently rerouting `.view()`/`.broadcast_var()` calls
  through the *other* method's original implementation. Concretely broke
  `expand().view()`'s storage-aliasing invariant
  (`tests/core/test_storage_strides.py::
  test_expanded_view_and_explicit_dense_library_boundary`, `expand()`
  marks its result `_set_storage_view_of` the input, and only a genuine
  `.view()` call — not `.reshape()` standing in for it — detects and
  preserves that relationship). Found via the *same* full-`tests/core/`
  regression sweep this session was already running; confirmed pre-existing
  (reproduces on the true unmodified baseline via `git stash`) but latent
  until `install_batching_patches()` ran unconditionally during the
  (since-reverted) eager-install experiment. Fixed by giving `_apply_reshape`/
  `_apply_broadcast` an explicit `orig` parameter and capturing/using
  `_ORIG_VIEW`/`_ORIG_BROADCAST_VAR` separately. **This fix is kept** (it's
  correct and needed regardless of install timing — any code path that
  happens to call `.view()`/`.broadcast_var()` after any `vmap()` call
  anywhere in the process was already exposed to it); confirmed all other
  same-shaped multi-name registrations in this file (`transpose`/`permute`,
  `matmul`/`__matmul__`, `jt.X`/`jt.Var.X` free-function pairs) are safe —
  each pair really is the same underlying object.

### Tests (`tests/core/test_vmap.py`)

`test_cumsum_batching_rule`: vs `np.cumsum`, non-default and negative
`dim`, 2-level nested vmap, grad composition. `test_solve_batching_rule`:
both operands batched, shared/unbatched `a` + batched `b` (the doc's
"shared parameter" concern), grad composition — using a proper `(...,M,K)`
matrix RHS (see the newly-discovered `solve()` gap below for why not the
bare `(...,M)` form). `test_numpy_code_raises_clear_error_under_vmap`,
`test_sparse_raises_clear_error_under_vmap`. `test_unsupported_op_raises`
switched from `cumsum` (now supported) to `cumprod` (same op family,
genuinely still unsupported) to keep testing the fail-fast contract
truthfully. New `python/jittor/compat/tests/torch/
test_real_imag_under_vmap.py` (torch-mode) for the compat fix.

### A third, unrelated pre-existing bug found and explicitly NOT fixed (out
of scope)

While shaping the `solve` batching-rule tests, found that `jt.linalg.solve`
itself — independent of vmap entirely, reproduced with a plain non-vmap
call — does not correctly support: (a) a bare `(...,M)` vector-stack `b`
against a batched `a` (this numpy version's `np.linalg.solve` gufunc only
takes the "b is a vector" path when `b.ndim==1` exactly; a batched `(...,M)`
is misparsed as a 2-D `(m,n)` core matrix, raising a core-dimension
mismatch — `solve()`'s own docstring implies `(...,M)` is the supported
vector form, but it isn't for a genuinely batched case with this numpy
version); (b) unbatched `b` against a batched `a` — `numpy_code`'s output
shape is declared as `bt.shape` verbatim
(`python/jittor/linalg/solving.py:316`), not broadcast against `a`'s extra
batch dims, so the correct, larger `np.linalg.solve` result fails to copy
into the too-small declared output buffer
(`ValueError: could not broadcast input array from shape (4,3,1) into
shape (3,1)`). Both are real `solve()` correctness gaps, not vmap gaps —
the vmap batching rule only has to (and does) correctly extend whatever
`solve()` already handles correctly, which is both operands batched (or
neither) with `b` shaped as a proper `(...,M,K)` matrix. Not fixed here —
flagged for separate, dedicated attention to `solving.py`'s own
broadcasting/shape-inference logic.

### Gate results

- `tools/run_test_suite.py --tier core`: native 98 passed/41 skipped/1
  xfailed, torch 212 passed/3 skipped. 0 failures.
- `tools/run_test_suite.py --tier smoke`: same order-of-magnitude
  pre-existing environment noise as the earlier P0 session's smoke run
  (MPI/ACL/DDP/TF32/RNN/pooling/einsum gaps in this minimal container, and
  the same "independent Torch was not preloaded" oracle-session-context
  errors) — a targeted search found nothing new specific to
  SGD/vmap/cumsum/solve/numpy_code beyond the already-known oracle-context
  pattern; `tests/codegen/test_broadcast_tuner.py`'s one failure is the
  same one already seen (and left unfixed) in the P0 session's smoke run.
- Full `tests/core/` + `tests/linalg/` (twice — once catching the view/
  broadcast_var regression, once clean after fixing it): 463 passed, only
  the 2 pre-existing `test_namespace_exports.py` failures remain (dlpack/
  vmap/Generator stub declarations, separately tracked as this doc's own
  §3.4/§3.5, not touched this session).
- `tools/check_repo_layout.sh`: one finding, `local/problems-2026-9-22.md`
  (the user's own external-report file, present before either session,
  referencing a `development-plan.md` in a different repo/harness) — left
  for the user to handle, same as the P0 session's report.
- `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest tests/structure`:
  1378 collected; fixed two real findings along the way (regenerated
  `MANIFEST.in` for the new files this and the earlier P0 session added;
  the two eager-install identity breaks, resolved by reverting eager
  install rather than chasing the alias system) — down to the same 2
  pre-existing failures as the P0 session
  (`test_a_timeout_ends_the_grandchildren_too`,
  `test_documentation_governance_checker`, the latter also the `local/`
  finding above).

## What's still open

Within §3.3: ACL not verified on real hardware (no NPU available this
session). Within §3.4: the lazy-patch-install "stale reference" gap is
investigated and deliberately not fixed (conflicts with a stronger existing
invariant; documented workaround already exists and is now tested); the
`jt.linalg.solve` broadcasting/output-shape gaps found along the way are a
separate, real, unfixed issue. The remaining 5 items from the original
9-item list (native vmap's own JVP/VJP/pytree/shared-parameter composition
beyond what's tested here, DLPack, CUDA RNG checkpoint, higher-order
complex AD, function-level compile/cache contract, sparse device-side
coalesce) are not addressed this session, per the user's own scoping to
§3.3/§3.4.

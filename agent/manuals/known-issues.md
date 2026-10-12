# Known-Issues Ledger

- Status: Maintained
- Last reviewed: 2026-10-06
- Baseline: `1a6e203fc`
- Owner: Jittor core maintainers
- Review when: an entry is fixed or narrowed, a strict expected failure
  XPASSes, a new defect is reproduced, or at quarterly maintenance

This ledger lists the **open** defects and known limitations of the current
tree: problems that are reproduced and not fixed, and constraints a user or a
maintainer will hit, each with its workaround. It is not a history. Environment
outages (a missing driver, a full disk, a killed session) are not framework
defects; they belong in the report of the run that met them. Work that is
written but waits for hardware is listed in
[deferred-hardware.md](deferred-hardware.md); a defect found while doing it is
recorded here.

## Entry format

Each entry is a level-two heading `## KI-<AREA>-<NNN>: <one-line title>`
followed by these fields, in this order:

- **Severity** -- one of the levels in the guide below.
- **Status** -- `Open`, `Open (verification pending: <backend>)`, or
  `Limitation` for an accepted constraint, with the commit or date it was last
  confirmed on.
- **Owner** -- the maintainer group that decides the fix.
- **Symptom** -- what a user or a gate observes, with the smallest reproduction.
- **Cause** -- the mechanism if it is known; otherwise what has been ruled out.
- **Workaround** -- what to do until it is fixed (`none` is a valid answer).
- **Evidence** -- test nodeids, commands or reports that reproduce it. A strict
  expected failure that XPASSes the day the defect is fixed is the preferred
  form.
- **Exit condition** -- what must be true for the entry to be deleted.

Keep an entry to what a reader needs to reproduce, avoid and close the problem.
Investigation narratives, refuted attempts and before/after tables belong in the
commit message, the pull request, or a dated `docs/results/` report; an entry
may name one refuted attempt when repeating it is the likely mistake.

## Ids

- `<AREA>` names the subsystem: `TEST`, `COMPILER`, `CODEGEN`, `TUNER`,
  `BACKEND`, `OPS`, `SEMANTICS`, `DTYPE`, `COMPLEX`, `MEM`, `AUTOGRAD`, `EXEC`,
  `COMPAT`, `DIST`, `LINT`. Add an area only when none of these fits.
- `<NNN>` is one more than the highest number the area has **ever** used,
  deleted entries included:
  `git log -p -- agent/manuals/known-issues.md | grep -o "KI-EXEC-[0-9]*" | sort -u | tail -1`.
  An id is never reused. (`KI-BACKEND-006` was once reused by mistake; the
  later entry was renumbered `KI-BACKEND-014`.)
- Code comments, test messages and documents cite the id. The change that
  deletes or renumbers an entry also rewords those citations
  (`git grep -n "KI-<AREA>-<NNN>"`); dated `docs/results/` reports may keep a
  historical mention.

## Closing an entry

Delete the entry in the change that makes its exit condition true, and name the
id in that commit message or pull request; Git keeps the text. Do not mark an
entry "fixed" and leave it here. If the fix leaves a constraint that users still
hit -- a default that diverges from NumPy or PyTorch, a flag that must be set --
rewrite the entry as a `Limitation` that states only that constraint and its
workaround.

## Severity guide

- **Critical:** silent wrong result, gradient, state, or device placement.
- **High:** a supported operation crashes or fails to compile without a
  practical equivalent path.
- **Medium:** a compatibility divergence with a documented workaround, a
  narrower supported domain, or a gate that cannot report what it is for.
- **Low:** an introspection or test-only defect with no effect on results.
- **Limitation:** an accepted constraint or a measured performance gap.
- **Research:** an intentionally unsupported capability that needs
  architectural work.

## KI-TEST-001: formerly silent test cases expose unresolved contracts

- Severity: Medium
- Status: Open; the cases run as strict expected failures
- Owner: test infrastructure and the affected backend maintainers
- Symptom: these tests used to be disabled with an initial `return` or
  `skipIf(True)`, so pytest reported success without executing them. They now
  run as strict expected failures when their hardware or network prerequisites
  exist; optimizer state-dict coverage fails explicitly until it has an
  implementation.
- Workaround: do not cite these nodeids as passing evidence for reduction dtype
  inference, NHWC cuDNN backward, tensor swapping, repeated dataset RingBuffer
  use, optimizer state restoration, or legacy CUDA pooling.
- Evidence: `TestBF16.test_reduce_dtype_infer`,
  `TestCudnnConvOp.test_backward_nhwc`, `TestCore.test_swap`,
  `TestCore.test_swap_cuda`, `TestRingBuffer.test_dataset`,
  `TestOptStateDict.test_opt_state_dict`, `TestArgPoolOp.test_cuda_old_pool`
  (each carries a `KI-TEST-001` comment).
- Exit condition: fix and independently verify each contract, remove its
  expected-failure marker, and delete the entry when the list is empty.

## KI-TEST-002: the gate does not report a session that died mid-run

- Severity: High
- Status: Open; a detector exists but no gate runs it
- Owner: test infrastructure maintainers
- Symptom: a native crash ends the interpreter, so pytest writes no result line,
  traceback or summary for the case that died and the tests after it never run.
  The log simply stops, and a session that died reads like a session that ran
  fewer tests. Any exclusion taken to work around such a crash silently narrows
  what the gate covers.
- Cause: `tools/run_test_suite.py` and the nox sessions read pytest's exit code
  and summary only. `tests/_helpers/session_completion.py` (registered by the
  pytest policy) prints a `JITTOR-SESSION-COMPLETE` marker and can write a
  sentinel with the collected and executed counts, and
  `tools/check_session_completed.py` turns a missing marker into a failure, but
  neither gate calls the checker.
- Workaround: run `python tools/check_session_completed.py --log <pytest.log>`
  (optionally `--expect-collected N`) on any log a conclusion is drawn from, or
  compare the executed count against the collected count.
- Evidence: `tests/structure/test_session_completion.py` (the detector);
  the last observed case was a CPU-only Torch selection that stopped at 48% with
  exit 1 and no summary.
- Exit condition: a session whose process ends before pytest finishes is
  reported by the maintained gates as a failure naming the case that was running
  and the number of tests that never executed.

## KI-TEST-003: the coverage wrapper is visible to the Torch identity contracts

- Severity: Medium
- Status: Open; measured and excluded rather than hidden
- Owner: test infrastructure maintainers
- Symptom: with `JITTOR_API_COVERAGE=1` on the Torch surface, 86 cases in the
  14 files listed in `tests/_helpers/api_coverage.py::IDENTITY_CONTRACT_FILES`
  fail that pass with it off (`assertIs(torch.addmm, installers.numerical.addmm)`
  and object-keyed fidelity registry lookups).
- Cause: the wrapper records a call by replacing the published object, while
  the Torch frontend contracts that the published object *is* the one its owner
  module holds. Rebinding the defining module (101 failures) or every alias (97)
  made it worse. The native surface states no such contract and is unaffected.
- Workaround: a Torch coverage run excludes those 14 files, and the exclusion is
  written into `tests/structure/torch_api_coverage_baseline.json` so the looser
  set is not read as complete.
- Evidence:
  `tests/structure/test_api_coverage_helper.py::test_the_wrapper_is_visible_to_an_identity_contract`.
- Exit condition: record calls without replacing anything (a `sys.setprofile`
  hook keyed by code object observes the same calls), delete the exclusion list
  and re-take the Torch baseline.

## KI-TEST-007: `test_fused_op_relay_matmul` fails after profiler or graph-replay tests in the same process

- Severity: Low
- Status: Open; reproduced on `1a6e203fc`
- Owner: codegen and test infrastructure
- Symptom: `tests/codegen/test_jit_tests.py::TestJitTests::test_fused_op_relay_matmul`
  passes alone and with its own file, and fails with `[check failed: cm.size()>=2]`
  when `tests/runtime/test_step_profile.py`, `tests/runtime/test_profiler.py` or
  `tests/nn/test_graph_replay_multi_output.py` ran earlier in the same pytest
  process.
- Cause: not isolated; suspected process state (a flag or a JIT cache entry)
  those files leave behind that `src/tests/test_op_relay.cc` depends on without
  setting.
- Workaround: run the file in its own process.
- Evidence: `pytest tests/runtime/test_profiler.py tests/codegen/test_jit_tests.py::TestJitTests::test_fused_op_relay_matmul`.
- Exit condition: the case passes after each of the three files in one process.

## KI-TEST-009: two `test_torch_compat_optim.py` cases fail on device placement

- Severity: Medium
- Status: Open; reproduced on `1a6e203fc` (CUDA)
- Owner: torch compatibility / optimizers
- Symptom: `compat/tests/torch/test_torch_compat_optim.py::TestSGD::test_native_backward_does_not_double_advance_step`
  dies in `method_api.py` `_binary_native` with `device_copy_op.cc: Expected all
  tensor inputs on the same backend and device`, and
  `TestAdam::test_bound_initializers_inside_no_grad_keep_parameter_trainable`
  fails `assert_stays_on_device` with `'cpu' != 'device'`. The other cases in the
  file pass.
- Cause: not isolated; suspected a tensor created on the host (an initializer or
  a split optimizer's state) meeting a device-placed parameter.
- Workaround: none in the tests; real training steps that keep one placement do
  not reach it.
- Evidence: `JITTOR_TORCH_SHIM=1 python -m pytest -q compat/tests/torch/test_torch_compat_optim.py -k "double_advance_step or keep_parameter_trainable"`.
- Exit condition: both cases pass on CUDA.

## KI-TEST-010: three `test_torch_compat_norm.py` LayerNorm fast-path cases fail

- Severity: Low (values elsewhere in the file pass)
- Status: Open; reproduced on `1a6e203fc` (CUDA)
- Owner: torch compatibility / normalization kernels
- Symptom: `compat/tests/torch/test_torch_compat_norm.py::TestLayerNorm::test_ln_no_grad_cuda_fast_path_float32_and_float16`
  (`CUDA no-grad LayerNorm missed its fused path`),
  `::test_ln_no_grad_cuda_dynamic_rows_share_source` (`0 != 2`) and
  `::test_ln_no_grad_cuda_bfloat16_private_opt_in`
  (`'NoneType' object has no attribute 'float32'`).
- Cause: not isolated; suspected the fused LayerNorm kernel selection changed
  under the Torch frontend without these tests following.
- Workaround: none needed for results.
- Evidence: `JITTOR_TORCH_SHIM=1 python -m pytest -q compat/tests/torch/test_torch_compat_norm.py -k ln_no_grad_cuda`.
- Exit condition: the three cases pass, or are rewritten to pin the path the
  frontend is meant to take.

## KI-TEST-011: `test_shared_reduce.py` decodes generated source with the locale encoding

- Severity: Low (a test reads its input wrongly; no product code affected)
- Status: Open; reproduced on `1a6e203fc`
- Owner: CUDA codegen tests
- Symptom: `LC_ALL=C PYTHONCOERCECLOCALE=0 PYTHONUTF8=0 python -m pytest -q tests/backends/cuda/test_shared_reduce.py -k two_stage`
  fails with `UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2`; plain
  `LC_ALL=C` passes only because of Python's locale coercion.
- Cause: the test opens generated JIT source with a bare `open(...)`, and the
  generated source contains non-ASCII (op keys are separated by U+00AB).
- Workaround: run under a UTF-8 locale.
- Evidence: the command above.
- Exit condition: pass `encoding="utf-8"` there, sweep the tree for other
  locale-dependent reads of generated source, and delete the entry.

## KI-TEST-012: the PR smoke tier takes about 390 s, not under five minutes

- Severity: Limitation (gate cost)
- Status: Open; measured 2026-09-06 (warm cache, `-n 4`, 16 cores, load 13-18)
- Owner: test infrastructure
- Symptom: the tier is work-bound, not stuck behind one long file: native
  406.3 s + torch 91.8 s = 498.1 s on that load (native work 1592.9 s over four
  workers). Reaching 300 s needs the native half's work cut to about 550 s.
  `tests/structure` is about 9.4% of the wall time, all in the torch half; the
  native half is `tests/ops` 31.8%, `tests/core` 27.3% (`test_setitem.py` alone
  15.0%), `tests/distributed` 16.2%, `tests/codegen` plus `tests/build` 12.1%,
  `tests/nn` 9.5%. Two runs differ by a jitter floor of about 27 nodeids, so
  "two runs conclude identically" cannot be the acceptance yet.
- Workaround: `python tools/run_test_suite.py --tier core` (~44 s) after an
  edit; the smoke tier before a pull request.
- Evidence: `python tools/run_test_suite.py --tier smoke`;
  `tests/_helpers/tiers.py` (`SLOW_FILES`); `tests/structure/test_gate_tiers.py`.
- Exit condition: make the comparisons themselves cheaper or fewer (not a longer
  exclusion list), or add machines; close when the tier fits five minutes on
  the reference runner.

## KI-TEST-013: the CUDA device-parity gate is compile-bound, and its warm duration is unmeasured

- Severity: Limitation (gate cost)
- Status: Open; measured September 2026
- Owner: test infrastructure
- Symptom: for the same 26 nodeids with one `JITTOR_HOME`, cold took 848.5 s and
  warm 23.6 s (36x), so the cost is JIT compilation, not the comparisons;
  extrapolated, a cold full gate is about two hours. Caching CPU references cut
  cold time by 16.2% with identical conclusions. Splitting across workers was
  only 6% faster and lost conclusions, so the gate stays one process. Nobody has
  measured the full gate on a warm CI cache since the workflow started
  persisting it.
- Workaround: keep a warm `JITTOR_HOME` for local runs.
- Evidence: `tests/backends/parity/test_device_parity.py`;
  `tests/_helpers/reference_cache.py` (`tests/backends/parity/test_reference_cache.py`);
  `tools/gate_conclusion_diff.py compare` (per-nodeid conclusion diff);
  `.github/workflows/cuda.yml` restores and saves the JIT cache.
- Exit condition: measure the nightly gate with a restored cache; if acceptable,
  delete the entry, otherwise reduce cold compilation.

## KI-COMPILER-001: the op-level parallel compiler can corrupt process state

- Severity: High
- Status: Open for the op-level compiler outside Jupyter. The file-level
  compile-pool deadlock (`run_cmds()`) and Jupyter's SIGCHLD exit are fixed;
  both are recorded in the investigation page below.
- Owner: compiler/executor maintainers
- Symptom: with `use_parallel_op_compiler` on (the default), a process that
  compiles many operators can segfault or corrupt state.
- Cause: not established. The op-level compiler is `src/core/parallel_compiler.cc`
  (`std::thread`), unrelated to the file-level `multiprocessing.Pool`.
- Workaround: `jt.flags.use_parallel_op_compiler = 0` for deterministic
  validation workloads; it reaches only the op-level compiler. The maintained
  notebook gate stays serial: a complete notebook smoke also died with eight
  compile workers with Jittor's signal handler disabled.
- Do not re-litigate from intuition: inside one process the op-level compiler is
  worth about 6x (50 distinct CPU kernels: 7 s at the default, 42 s at
  `use_parallel_op_compiler=0`), so turning the default off is not on the table.
  Between processes sharing a cache it is worth nothing, because
  `parallel_compile_all_ops` holds the process-level `jittor.lock` for the whole
  batch; that is the first thing to check when "xdist with eight workers is only
  7% faster than serial".
- Evidence: [investigation and reproduction](../../docs/development/known-issues/parallel-compiler-segfault.md),
  which must agree with this entry on status, workaround and exit condition.
- Exit condition: a sanitizer-backed root cause of the op-level corruption, then
  repeated cold/warm stress, multiprocess-cache and performance gates. A/B
  timings of the flag restore the same cache snapshot before each variant;
  running parallel first and serial second measures cache warming, not the flag.

## KI-COMPILER-007: a CPU-only no-CUDA import has aborted with heap corruption at exit inside the smoke tier

- Severity: High (if it recurs: glibc heap corruption; the functional assertions
  pass and only the child's exit status is wrong)
- Status: Open, intermittent. Reproduced 2026-09-22 only inside
  `--tier smoke --session native` with the tool's default `-n 4`; not reproduced
  in any shape after the core was rebuilt from newer sources (`f8657c8d`).
- Owner: compiler/build maintainers
- Symptom: `tests/build/test_backend_build_config.py::test_explicit_cpu_import_skips_cuda_services`
  fails with `assert -6 == 0`. Its child patches five
  `jittor.build.utils.install_cuda` entry points to raise, imports jittor under
  `JT_BACKEND=cpu nvcc_path=/must/not/probe/nvcc JTCUDA_AUTO_INSTALL=1`, prints
  `CPU_BUILD_CONFIG={"cuda_services": 0, "backend": "cpu"}`, and then aborts in
  teardown with `double free or corruption (!prev)` or
  `corrupted size vs. prev_size while consolidating`.
- Cause: unknown. Every abort ran with four pytest workers sharing one
  `JITTOR_HOME` and its build lock. Ruled out: the child run directly (16/16
  clean, also with `MALLOC_CHECK_=3 MALLOC_PERTURB_=165`), the gate's
  environment variables and thread budget, a stale core, and a teardown-time call
  into the patched entry points (an instrumented sibling saw none).
  `use_parallel_op_compiler=0` was already in force, so KI-COMPILER-001's
  workaround does not cover it.
- Workaround: re-run the case alone; a passing functional line with exit -6 is
  this entry, not a new defect.
- Evidence: `python tools/run_test_suite.py --tier smoke --session native -- -k explicit_cpu_import_skips_cuda_services`.
- Exit condition: the case passes inside the smoke tier on repeated runs; if it
  recurs, capture it under an allocator that reports a stack (ASAN) before
  changing code.

## KI-COMPILER-008: user-reachable JIT source checks still report as internal errors

- Severity: Medium (catchable, but the category and message are wrong: a user
  mistake asks the user to report a framework bug)
- Status: Open; unchanged on `e3c369acb`
- Owner: core error handling / code generation
- Symptom: a misspelt `@out(0)=@nosuchvar` in `jt.code(..., cpu_src=...)` says
  "Something wrong... Could you please report this issue?". Seven bad-`@`-syntax
  `jt.code` calls all raised a catchable `RuntimeError` containing
  `Jit compiler error:`.
- Cause: `precompile` in `src/codegen/op_compiler.cc` wraps its loop in
  `catch (std::exception& e)` and rethrows through `LOGf`, so a `UserError`
  raised inside arrives as a plain error; migrating its `ASSERT`s to
  `USER_CHECK` would change nothing observable while the source-count gate turns
  green. `src/codegen/opt/kernel_ir.cc`, `src/utils/cache_compile.cc` (mostly its
  `#ifdef TEST` self-test), `src/codegen/opt/expr.cc`, `src/ops/op_register.cc`
  and `src/mem/swap.cc` have not been walked for public-argument reachability.
- Workaround: none needed; the exceptions are catchable.
- Evidence: `docs/development/error-categories.md`;
  `tests/structure/core/test_error_categories.py` (source-level gate).
- Exit condition: `precompile`'s catch preserves the `UserError` category, the
  user-reachable checks are migrated with a negative test each, and the listed
  files are classified.

## KI-CODEGEN-001: fused elementwise kernels pay a division and a modulo per element for a broadcast operand

- Severity: Medium (performance; results are correct)
- Status: Open; a documented trade-off
- Owner: codegen maintainers
- Symptom: on CPU, `x + y` where `y` is an expand of a real tensor runs about
  5.2x its dense counterpart (`64x64x64x64 + 1x64x1x1`: 636.5 us against
  122.9 us per add, measured 2026-09-22; NumPy's ratio is 1.04x).
  `contiguous()` of a broadcast view pays the same index recovery. Bias add,
  normalisation scale and attention masks all have this shape.
- Cause: `BroadcastToOp` returns a stride-0 view (`OpType::other`) so a broadcast
  feeding a non-fused consumer never materialises (`src/ops/broadcast_to_op.cc`
  documents the trade-off). In a fused kernel the strided operand's index is
  recovered from the flat index with one division and one modulo per masked
  axis (`binary_op.cc`, `unary_op.cc`, `ternary_op.cc`,
  `composite/contiguous_op.cc`; consecutive divisions are already folded into
  one). `MergeLoopVarPass` does not treat those defines as loop variables and
  merges every loop into one flat loop (`range0_1_2_3`), so the division stays
  and blocks vectorisation.
- Workaround: none.
- Evidence: `tests/codegen/test_merge_loop_var_pass.py::TestMergeLoopVarPass::test3`
  and its CUDA twin assert the nested `range2_3` structure and fail. They are the
  guard for this cost, not stale assertions: the merged kernel is numerically
  right, and rewriting them to accept `range0_1_2_3` would hide the cost.
- Exit condition: in a fused kernel a strided operand's index is expressed in
  loop ids with the masked axes folded at compile time (`YSMASK` is already in
  the jit key) and the masked axis' loop stays nested; `test3` passes with its
  original expectation and a broadcast add costs about what its dense
  counterpart does. A wrong variant fails as silently wrong numbers, so verify
  bit-exactly across broadcast patterns. When a reason appears, give the ungated
  `storage_stride` reads in `src/ops/reindex_op.cc` (`xstride`) and
  `src/ops/reindex_reduce_op.cc` (`ystride`) the same contiguous-input gate the
  elementwise and reduce ops have.

## KI-CODEGEN-002: the `split{i}` and `parallel` loop options cannot be combined

- Severity: Limitation (a compile error, never a wrong result; blocks a tuner
  candidate)
- Status: Open; still true on `e3c369acb`
- Owner: code generation (loop passes, reduce tuner)
- Symptom: with both options set, `ParallelPass` fails `ASSERT(def)`
  (`src/codegen/opt/pass/parallel_pass.cc`).
- Cause: `SplitLoopPass` gives the inner loop the range
  `::min(range{i}-id{i}, stride{i})`, defined inside the outer loop and varying
  with it, which `ParallelPass` cannot evaluate where it sizes the thread grid.
  CUDA always runs `ParallelPass`, so every `split{i}` candidate would break a
  CUDA reduction; the reduce tuner offers none there.
- Workaround: none needed; the tuner does not offer the combination.
- Evidence: `tests/codegen/test_reduce_tuner.py::TestReduceTuner::test_a_split_candidate_would_not_compile_under_parallel`
  (passes while the combination still fails); the guard and its rationale are
  in `src/codegen/opt/tuner/reduce_tuner.cc`.
- Exit condition: `ParallelPass` accepts a split inner range (or hoists it), the
  test above fails, the CUDA guard in `reduce_tuner.cc` is revisited, and the
  entry is deleted.

## KI-CODEGEN-003: CUDA code-generated reductions trail PyTorch on UNet shapes

- Severity: Limitation (performance)
- Status: Open; measured 2026-09-06 on an sm_89 GPU (diffusers UNet2D training
  step). The GroupNorm kernels have been rewritten since (`d99a59e87`,
  `1b6950c35`, `b578dc24e`), so the split below needs re-measuring.
- Owner: code generation (CUDA reduction passes) and CUDA kernels
- Symptom: with call counts matched one to one, the reduction class was
  2279-2297 us (profiler; about 2705 us under nsys) against PyTorch's
  1928-1998 us, 15-36% slower. The generic code-generated sums were 745-753 us
  against 653-682 us; at the time 75% of the gap was in the hand-written
  GroupNorm.
- Cause: partly strategy. `para_opt_level=4` (warp shuffle, one shared value per
  warp, then a first-warp shuffle in `SharedReducePass`) is about 8% faster than
  the default warp reduction on the UNet's reduce shapes but up to 1.39x slower
  on the representative shapes, and it also switches `AtomicTunerPass`, so the
  default stays 3.
- Workaround: none.
- Evidence: `agent/skills/cuda-reduction-strategy-comparison/` (`reduce_ab.py`,
  `group_norm_ab.py`) and `agent/skills/cuda-elementwise-bandwidth-roofline/`
  (`profile_step_torch.py --attribute` pairs each PyTorch kernel with its aten
  op so both sides compare the same work).
- Exit condition: re-measure the class against PyTorch on the same step; close
  when it is no slower, or narrow the entry to what remains.

## KI-CODEGEN-004: fused elementwise kernels of a UNet2D step run about 10% behind PyTorch

- Severity: Limitation (performance)
- Status: Open; measured 2026-09-06 on an sm_89 GPU with TF32, exclusive card
- Owner: code generation, with the Torch-compat division in KI-COMPAT-009
- Symptom: the elementwise class of a `large_diffusers_unet2d` step was 3.37 ms
  (nsys) / 3.29 ms (profiler), 1086 GB/s, roofline ratio 0.84 -- already at the
  measured copy bandwidth -- against PyTorch's 3.04 ms.
- Cause: the positive excess of the 49 fused kernels was about 0.59 ms and none
  of it is code generation: the float64 scalar division of KI-COMPAT-009
  (0.55 ms), a bare `transpose` (`src/ops/composite/transpose_op.cc`, 0.10 ms),
  and about 60 kernels that move almost no data (0.23 ms of launch latency).
- Workaround: none.
- Evidence: `agent/skills/cuda-elementwise-bandwidth-roofline/` (measurement
  scripts, nsys and profiler cross-checked, measured copy roofline).
- Exit condition: re-measure after KI-COMPAT-009 is decided; close when the
  class is no slower than PyTorch's on the same step.

## KI-TUNER-001: the matmul and conv relays never fire, so a hand-written meta-op product runs as a generic kernel

- Severity: High (a supported operation runs orders of magnitude slower than the
  library kernel that exists for it, with no error and no log)
- Status: Open for the hand-written form. `jt.nn.matmul`, `nn.Linear` and
  `nn.Conv2d` reach the library kernels directly and are not affected.
- Owner: compiler/tuner maintainers
- Symptom: with `enable_tuner=1`, a product or convolution written out of
  `broadcast`/`reindex`/`*`/`sum` is never relayed to `mkl_matmul`/`mkl_conv`
  (CPU) or `cublas_matmul`/`cudnn_conv` (CUDA); it runs as a generic kernel.
  Before the public ops were routed around it this measured 99x (1024-cube
  product against NumPy), 35x (`8x64x56x56` conv against torch) and 1874x (the
  same conv transposed, stride 2). The tuner reports
  `Run tuner matmul: confidence(0) candidates({})`, and the
  `Jit op key (not )?found: cudnn_conv...` line the relay emits never appears.
- Cause: `MatmulTuner` and `ConvTuner` require each operand's producer to be a
  `broadcast_to` that is a member of the fused op. An expand is now a stride-0
  view (`OpType::other`, see KI-CODEGEN-001) that never joins the fused op.
  Relaxing the membership test is a trap, not a fix: on a build that registers
  the capability it reaches `add_relay_group`, whose backward BFS requires every
  operand of the relayed op to be a fused-op node
  (`src/codegen/opt/var_relay.cc`, `ASSERT(q.size()==2*group.size())`), and the
  `oprcs` loop's `ASSERT(fnodes.count(v))`; both abort. The relay also hands the
  library op the fused op's vars, which are now views with the broadcast shape.
  Two real fixes: let a view expand join the fused op (`count_fuse` refuses
  every edge touching an `OpType::other` op before it looks at `_force_fuse`,
  which would also help KI-CODEGEN-001), or let the relay carry operands that
  live outside the fused op (a new sentinel in `relayed_members` plus execution
  support).
- Workaround: use the public ops. The CPU rows of the `matmul`, `conv2d` and
  `conv_transpose2d` kernel tables call `mkl_matmul` / `mkl_conv` /
  `mkl_conv_backward_x` directly, the way the CUDA rows call cuBLAS and cuDNN.
  A build without `use_mkl=1` registers no CPU matmul/conv capability at all,
  so the CPU relay tests skip there with `require_library("mkl")`.
- Evidence: `tests/ops/test_matmul.py::TestMatmul::{test_matmul,test_matmul_type,test_matmul_cuda,test_matmul_type_cuda}`
  (their `check_matmul2` builds the product by hand);
  `tests/backends/cpu/test_mkl_conv_op.py::TestMklConvOp::*` (`logs[0][0] == '20'`
  reads the conv tuner's confidence);
  `tests/backends/cuda/test_cudnn_op.py::TestCudnnConvOp::{test,test_backward}`
  (`assert len(logs)==1 and "oihw" in logs[0][0]` finds `logs == []`, reproduced
  on `1a6e203fc`; the value checks next to it pass);
  `tests/backends/cuda/test_cuda_op_capabilities.py::{test_cuda_matmul_capability_relays_meta_operator_graph,test_cuda_conv_capability_relays_meta_operator_graph}`;
  `tests/codegen/{test_conv_tuner,test_matmul_tuner,test_group_conv_tuner}.py`;
  `tests/runtime/test_profiler.py::TestProfiler::{test_profiler,test_marks}`
  (the relayed pair of report rows); `tests/backends/cpu/test_onednn_contract.py`
  (probes oneDNN's declared float32 matmul through the hand-written form).
- Exit condition: with `enable_tuner=1`, a hand-written `broadcast * broadcast ->
  reduce` product emits a `mkl_matmul` (CPU) / `cublas_matmul` (CUDA) jit op key
  and the conv tuner's confidence is 20 again, with the files above green and no
  numerical change.

## KI-BACKEND-001: narrow integer sum/max/min lack NPU atomics

- Severity: High
- Status: Open; NPU skips
- Owner: reduce and ACL backend maintainers
- Symptom: `sum`, `max` and `min` over sub-32-bit integer samples abort on NPU
  because the ACL atomic overloads they need are not implemented. The core
  bool `all_`/`any_` reductions also lack a maintained generic ACL kernel; the
  public `jt.all`/`Tensor.all` and `jt.any` route bool inputs to CANN
  `aclnnAll`/`aclnnAny` (numeric inputs compare nonzero first), which does not
  establish support for the skipped core reduction ops.
- Workaround: promote inputs to a supported width before reducing on NPU; use the
  public CANN truth reductions where their semantics apply. Integer `prod` is
  supported through `aclnnProd`/`aclnnProdDim`.
- Evidence: [`reduce_dtypes.py`](../../tests/opinfo/definitions/reduce_dtypes.py),
  [device parity](../../tests/backends/parity/test_device_parity.py),
  [Ascend 910B validation](../../docs/results/2026-08-28-ascend-910b-validation.md).
- Exit condition: every affected dtype executes and matches the CPU reference on
  a real NPU, turning the skips into passes.

## KI-BACKEND-002: composed atan2 can crash on NPU

- Severity: High
- Status: Open; NPU skip
- Owner: binary operator and ACL backend maintainers
- Symptom: the maintained float32 `atan2` composition can terminate the process
  with an ACL vector-core exception on a real 910B3.
- Workaround: run this operation on a backend with a maintained `atan2` kernel;
  do not mask the process failure with a broad CPU fallback.
- Evidence: [`pointwise_binary.py`](../../tests/opinfo/definitions/pointwise_binary.py),
  [Ascend 910B validation](../../docs/results/2026-08-28-ascend-910b-validation.md).
- Exit condition: the float32 OpInfo reference and a focused crash reproducer
  pass repeatedly on a real NPU without an expected skip.

## KI-BACKEND-003: complex irfft can stall on NPU

- Severity: High
- Status: Open; NPU skip
- Owner: FFT and ACL backend maintainers
- Symptom: the complex-to-real inverse FFT does not complete within 600 s on a
  real 910B3, and pytest's signal timeout does not reliably interrupt the stalled
  native call.
- Workaround: execute `irfft` on a backend with a maintained complex FFT path.
- Evidence: [`fft.py`](../../tests/opinfo/definitions/fft.py),
  [Ascend 910B validation](../../docs/results/2026-08-28-ascend-910b-validation.md).
- Exit condition: forward values match NumPy and the operation exits within the
  maintained timeout on repeated real-NPU runs.

## KI-BACKEND-008: CUDA kernels flush float32 subnormals to zero by default

- Severity: Limitation (documented device divergence)
- Status: Limitation; the default is deliberate and asserted
- Owner: CUDA backend maintainers
- Symptom: on CUDA's default kernel math `1e-45` and `1e-40` read as `0.0`
  while CPU and NumPy keep them. It reaches past the value: `log(1e-45)` is
  `-inf` on CUDA and `-103.28` on CPU, and `count_nonzero([1e-45])` is 0 against 1.
- Cause: nvcc's `--use_fast_math` implies `-ftz=true`.
- Workaround: `jt.flags.cuda_kernel_math = "strict"` restores subnormals. Its cost
  was not measurable on memory-bound 16M-element `divide`/`sqrt`/`exp`/`log`
  (0.1-0.8%, inside the control's spread); the compute-bound cost is unmeasured.
- Evidence: `tests/backends/parity/test_subnormal_contract.py` (both policies
  must differ for it to pass); [numerics contract](../../docs/notes/numerics-contract.md).
- Exit condition: delete only if the default changes; that decision needs a
  compute-bound cost measurement with a pinned fusion shape, and the contract
  test and the numerics page change with it.

## KI-BACKEND-009: CUDA cannot compile a logical or narrow-integer reduction

- Severity: High (a family of reductions is unusable on CUDA, and the error does
  not name the dtype or the operation)
- Status: Open; reproduced 2026-09-11 on CUDA
- Owner: reduce operator and CUDA codegen maintainers
- Symptom: `x.any_()`, `x.all_()` and the bitwise reductions are published API.
  On CUDA they raise `parallel_compiler.cc: Error happened during compilation`
  with an nvcc transcript for `logical_and`/`logical_or`/`logical_xor` over
  `uint8`, `float32` and `float64`, and for `bitwise_and`/`bitwise_or`/
  `bitwise_xor` over `uint8`. CPU answers all 66 operation-dtype combinations;
  `bool`, `int32` and `int64` work on both devices, which is why the tests did
  not notice.
- Cause: not isolated in the CUDA reduction codegen. The float cases are also a
  semantic question: CPU returns a float (`0.0`) for `logical_xor` over floats
  where a bool is the defensible answer, so the CPU dtype should be decided with
  the CUDA fix. The `uint8` cases are well defined and CPU performs them.
- Workaround: cast to `int32` or `bool` before a logical or bitwise reduction on
  CUDA.
- Evidence: every published reduction crossed with six dtypes, one fresh
  process each, with `location()` asserted so a CPU fallback cannot read as a
  pass; [`test_bitwise_dtype_guard.py`](../../tests/ops/test_bitwise_dtype_guard.py)
  already rejects the float rows of the bitwise family on both devices.
- Exit condition: the 66-cell matrix agrees between the two devices in value and
  dtype and a test holds it; the float rows may be resolved by rejecting them on
  both devices with a clear message.

## KI-BACKEND-011: the oneDNN CPU path is float32-only and its per-call cost is unmeasured

- Severity: Limitation (narrower dtype support; performance unverified)
- Status: Open. The oneDNN v3 functional migration is in
  (`backends/cpu/libraries/mkl/onednn_runtime.cc` refuses oneDNN < 3 and keeps a
  bounded per-shape plan cache, 32 plans / 64 MiB scratch).
- Owner: CPU backend (oneDNN)
- Symptom: convolution and matmul through oneDNN accept float32 only; other
  dtypes take the generic kernels (declared, not silent). The plan cache replaced
  the per-call engine/descriptor/primitive rebuild, but the lower per-call
  convolution overhead it was built for has never been measured against the old
  path. oneDNN no longer ships prebuilt v3 binaries, so verification needs a
  source-built installation (`JT_BUILD_MKL_INCLUDE_PATH` / `JT_BUILD_MKL_LIB_PATH`).
- Workaround: none needed for float32.
- Evidence: `tests/backends/cpu/test_onednn_v3_runtime.py`,
  `tests/backends/cpu/test_onednn_contract.py`,
  `tests/backends/cpu/test_mkl_conv_op.py`; capability declarations in
  `python/jittor/nn/backends/onednn.py` (`dtypes={"float32"}`);
  `backends/cpu/libraries/mkl/mkl_matmul_op.cc` ("support float32 only now").
- Exit condition: measure per-call convolution overhead before/after on a fixed
  shape set; widen matmul through `dnnl::matmul` if fp64/fp16/bf16 are wanted;
  delete the entry when both are settled.

## KI-BACKEND-012: the ACL descriptor cache is not wired into any runner, and the ACL launcher and attribute migration never ran on an Ascend device

- Severity: Medium (unvalidated backend paths; host-only evidence)
- Status: Open
- Owner: ACL backend maintainers
- Symptom: every `executeOp` owner goes through the shared
  `BaseOpRunner::launch` (`tests/_helpers/acl_launch_tails.py` finds no
  hand-rolled tail), and the attribute-carrying owners declared in
  `backends/acl/include/aclops/acl_code_attributes.h` receive their attributes
  through the versioned data channel instead of generated source. The descriptor
  identity/cache shell in `acl_data_channel.h`/`acl_data.py` is not used by any
  runner yet, so descriptors are still rebuilt per call; pool descriptors keep
  their own lifetime. All of this has host-only evidence (stub-SDK syntax checks,
  CPU-compiled contract probes); none of it has executed on a 910B3.
- Workaround: validate on hardware with `backend_fallback=error` and the
  fallback-count check before relying on an ACL operator.
- Evidence: [`docs/development/acl-backend-contracts.md`](../../docs/development/acl-backend-contracts.md)
  (migration order and the device acceptance command); host contracts under
  `tests/structure/backends/acl/`; the hardware run in
  [deferred-hardware.md](deferred-hardware.md) (Ascend 910B3 single card).
- Exit condition: wire descriptor address rebinding and invalidation into a real
  runner and pass the 910B3 acceptance runs in the contract page.

## KI-BACKEND-013: seven backend gradients have no gradient test, and one is not implemented

- Severity: Medium (untested gradients on backends this project rarely has)
- Status: Open. Of the 60 backend gradient implementations, 24 run on a CPU+CUDA
  host and were re-checked against CPU references without finding a gradient
  bug; 36 wait for hardware.
- Owner: backend maintainers (HCCL, ROCm, ACL)
- Symptom: `HcclAllGatherOp::grad()` is `LOGf << "not implemented"`;
  `RocprimCumsumOp` has no test at all; `FloorIntACL`, `IndexACL`, `NonzeroACL`,
  `StackACL` and `TriuACL` have forward tests only, so even with a card their
  backward is not exercised.
- Workaround: none; do not cite these backends' gradients as verified.
- Evidence: `tests/structure/test_backend_grad_contract.py`
  (`BACKEND_GRAD_COVERAGE`, equal to the source tree in both directions); the
  per-kind commands and the seven gaps in
  [deferred-hardware.md](deferred-hardware.md) ("后端 `grad()` 的 CPU 参考对拍").
- Exit condition: add the gradient tests (and the HCCL implementation), change
  their `kind`, and run them on the hardware; the structure test then lets the
  manual's list shrink.

## KI-BACKEND-014: cublas_test / cudnn_test are dispatched to the host and have no kernel there

- Severity: Medium (two library self-test ops cannot run; the libraries
  themselves are fine)
- Status: Open; two repair attempts refuted. Renumbered from `KI-BACKEND-006`,
  an id that had already been used.
- Owner: CUDA backend maintainers
- Symptom: `tests/backends/cuda/test_cublas_test_op.py` fails two of its three
  classes with `op.cc:148: No kernel registered for cublas_test on cpu`
  (likewise `cudnn_test`). The third class, `TestCubTestOp`, passes.
- What distinguishes the one that works: `CubTestOp`'s constructor declares
  `set_flag(OpFlags::_cpu, 0); set_flag(OpFlags::_cuda, 1);` and guards its
  body with `#ifdef JIT_cuda`. `CublasTestOp` and `CudnnTestOp` declare no
  backend at all and guard their bodies with `#ifdef JIT_cpu`, so they default
  to the host, where `Op::implementation()` then finds nothing registered.
- Two attempts, both measured and both reverted:
  1. Adding only the two `set_flag` lines moves the op onto the accelerator,
     and the CUDA build then produces a library with no `jit_run` in it:
     `undefined symbol: _ZN6jittor12CublasTestOp7jit_runEv`. Consistent --
     the body is still behind `#ifdef JIT_cpu`.
  2. Adding the flags *and* switching the guard to `#ifdef JIT_cuda` made
     things **worse**: 3 failed instead of 2, taking `TestCubTestOp` -- which
     had been passing -- down with it. So the three ops are coupled through
     something not visible in these three files (a shared generated OpInfo
     unit is the obvious candidate) and the fix is not local to them.
- Note: both entries (`cublas_test_entry`, `cudnn_test_entry`) are host
  functions that drive the libraries through the CUDA runtime, so "host code"
  does not by itself settle which backend the op belongs to.
- Exit condition: all three classes in that file pass, with the mechanism
  behind attempt 2's collateral damage understood rather than worked around.

## KI-VLLM-001: advanced engine modes remain outside verified Jittor support

- Severity: Research
- Status: Reproduced on CUDA, 2026-09-23; vLLM 0.24.0 only
- Baseline: single-card FP8/Graph/compile reverified at `9853c6a`; TP=2 follow-up
  starts from `539ce90` plus the changes recorded in the
  [multi-GPU report](../../docs/results/2026-09-27-vllm-multigpu.md)
- Owner: vLLM adapter and Torch compatibility maintainers
- Evidence: run `agent/skills/vllm-torch-compat/acceptance.py` in isolated
  shim/oracle environments with `--case multigpu`, `fp8`, `compile`,
  `cudagraph`, or `other`; options, current failures and raw-artifact names are
  in [the acceptance report](../../docs/results/2026-09-23-vllm-uva-topk.md).
- Boundaries: TP=2 now has Store-based CPU control groups and passes real
  alternating CPU/GPU collective checks. Qwen TP=2
  previously diverged from token 4 because auto replay omitted foreign writes.
  The retained-graph barrier fixes that case. After Store completion, Event wait
  and scalar-promotion fixes, the no-hook 70-request Qwen TP=2 matrix matches
  oracle, as do 1024/3072-token long cases; explicit shutdown verifies clean
  worker exits/shared-memory release.
  Broader multi-GPU serving and advanced modes remain outside that bounded scope.
  Run-isolated NCCL subgroup files and spawn cache isolation remain harness
  workarounds tracked in the multi-GPU report.
  This is not full libgloo compatibility. FP8 import needs `torch.library.wrap_triton` and its
  calculation kernels remain unimplemented; compilation needs the FX/Inductor
  integration; graph capture lacks a usable graph pool. The later single-card
  follow-up supplies explicit exponential generators and repairs OPT legacy
  host metadata; its current scope is recorded in the
  [single-card report](../../docs/results/2026-09-23-vllm-singlecard-correctness.md).
  The latest FP8 import, graph-pool and lazy-FX failures are retained in the
  [extended acceptance report](../../docs/results/2026-09-23-vllm-singlecard-extended-acceptance.md),
  separately from the accepted bounded eager HTTP service checks.
- Workaround: use the tested Qwen3-0.6B FP16/eager configurations; TP=2 requires
  the documented startup isolation and explicit shutdown. A mode that errors
  is not silently substituted with an accepted configuration. This does
  not claim support for arbitrary models, contexts or production serving.
- Exit condition: implement the relevant public compatibility contract or
  adapter operator, preserve the regression reproducer, and pass that real
  engine mode with an independent oracle and stated correctness criteria.

## KI-SCALAR-001: full_like truncates a large int64 fill scalar

- Severity: High
- Status: Reproduced on CUDA, 2026-09-23
- Baseline: `61294cd14673ba60f4072542a2e73ea2c8f8c509` plus the vLLM acceptance changes
- Owner: Jittor scalar-factory maintainers
- Evidence: with CUDA enabled, `x = jt.zeros((1,), dtype="int64")`, then
  `jt.full_like(x, 2**60 + 17).numpy()` yields `[17]` instead of
  `[1152921504606846993]`. The unversioned `full-like-int64-boundary.log` is
  under `$JITTOR_LAB_ROOT/_state/vllm-acceptance/20260923/`.
- Workaround: construct the fill using `jt.array(value, dtype=x.dtype)` in the
  destination device scope and broadcast explicitly. The repaired native
  `index_fill` uses this path and passes exact-value CPU/CUDA tests.
- Exit condition: fix the generic scalar construction path and verify large
  signed int64 fills on CPU and CUDA, without regressing smaller integer or
  floating factories. The index-fill repair does not resolve this general issue.


## KI-COMPAT-005: empty_like enables gradients by default

- Severity: Medium
- Status: Reproduced on CPU, 2026-09-25; CUDA not yet checked.
- Baseline: `aa6f750` (production metadata optimization `3401bf0`).
- Owner: Torch compatibility factory maintainers.
- Evidence: with CPU float64 input, both source gradient states and both
  inherited/explicit output dtype, `torch.empty_like(source)` returns
  `requires_grad=True` in the shim. Independent binary PyTorch 2.11.0 returns
  False in all four cases; its CUDA state stays uninitialized. This factory
  uses a separate fallback and is outside the constructor-final-cast candidate.
- Minimal reproduction after activating the desired Torch environment:
  `x = torch.tensor([1., 2.], device="cpu"); y = torch.empty_like(x);
  assert not y.requires_grad`. The identical standalone four-case script and
  red/green logs are unversioned under
  `$JITTOR_LAB_ROOT/_state/vllm-performance/20260925/factory-cast/` as
  `empty_like_requires_grad.py`, `empty-like-jittor.log` and
  `empty-like-native.log`.
- Workaround: `torch.empty(tuple(x.shape), dtype=x.dtype, device=x.device,
  requires_grad=False)` passes the two source-gradient cases on CPU. No claim
  is made here about memory-format equivalence or measured performance impact.
- Exit condition: first retain the public-API failure, then fix default and
  explicit gradient semantics without losing dtype/device/shape behavior;
  verify independent CPU/CUDA reference cases and downstream regression.


## KI-VLLM-002: explicit CPU default exposes mixed-device attention

- Severity: Research
- Status: Recorded; repair deferred by user, 2026-09-23. Not a required gate
  for the established single-card CUDA configuration.
- Baseline: `44d7e04` plus uncommitted single-card acceptance changes.
- Owner: vLLM adapter and Torch compatibility maintainers.
- Evidence: in the shim, enabling `jt.flags.use_cuda = 1` and subsequently
  calling `torch.set_default_device("cpu")` resets the global `use_cuda` flag
  to 0. In Qwen request-state acceptance, GPU attention scores then meet a CPU
  causal mask at `python/jittor/nn/paged_attention.py:131`. The observed error
  is `Expected all tensor inputs on the same backend and device`.
- Raw artifacts (unversioned):
  `$JITTOR_LAB_ROOT/_state/vllm-singlecard/20260923/state/jittor.{json,log}`
  from the explicit-CPU-default attempt. Preserve these separately from reruns.
- Reproduction configuration: insert `torch.set_default_device("cpu")` after
  importing Torch in `request_state_acceptance.py`, then run the Qwen case in
  the isolated shim environment. This override is intentionally absent from
  normal acceptance. No attention repair is claimed.
- Workaround: preserve Jittor's established CUDA default and the independent
  PyTorch backend default. Verify actual model parameters, KV cache and
  computation are CUDA; equal global defaults are not a prerequisite.
- Exit condition: if CPU-default compatibility is taken up again, first retain
  a minimal failing test, repair device propagation, then pass that test and
  real-engine acceptance. An old-configuration pass does not close this issue.

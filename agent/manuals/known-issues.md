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
  classes with `op.cc: No kernel registered for cublas_test on cpu` (likewise
  `cudnn_test`); `TestCubTestOp` passes.
- Cause: `CubTestOp` declares `set_flag(OpFlags::_cpu, 0); set_flag(OpFlags::_cuda, 1);`
  and guards its body with `#ifdef JIT_cuda`. `CublasTestOp` and `CudnnTestOp`
  declare no backend and guard their bodies with `#ifdef JIT_cpu`, so they
  default to the host, where `Op::implementation()` finds nothing. Adding only
  the flags produces a CUDA library with no `jit_run`
  (`undefined symbol: _ZN6jittor12CublasTestOp7jit_runEv`); adding the flags and
  switching the guard to `JIT_cuda` also broke `TestCubTestOp` (3 failed instead
  of 2), so the three ops are coupled through something outside their files.
- Workaround: none needed; the libraries are exercised by the real ops.
- Evidence: the test file above.
- Exit condition: all three classes pass, with the coupling behind the second
  attempt understood rather than worked around.

## KI-BACKEND-016: sd15_unet_train intermittently aborts on Ascend with ACL 507035

- Severity: Medium (intermittent crash of one training workload on Ascend)
- Status: Open (verification pending: Ascend 910B3). Non-deterministic -- three
  isolated runs of the same workload gave three outcomes.
- Owner: ACL backend maintainers
- Symptom: `sd15_unet_train` on a 910B3 sometimes aborts with
  `aclrtSynchronizeDevice failed with ACL status 507035`
  (ACL_ERROR_RT_VECTOR_CORE_EXCEPTION); another run instead fails earlier in
  setup with "code requires source for the selected backend", op `code` out
  `int32[1000]` -- the 1000-step DDPM timestep table built at module
  construction (a `torch.arange(1000)` issued before the front end's placement
  scope exists); another run completes cleanly (peak 19.09 GB). No OOM, card
  idle.
- Cause: not isolated. 507035 means an operator received an illegal argument.
  The call-time version of the construction-time arange placement was fixed in
  `c8fffce72`; the module-construction-time path is not covered, which can feed
  a not-yet-placed Var to a vector kernel. Looks like a placement/timing issue
  in graph capture, not numerics; concurrent machine load may contribute.
- Workaround: `jt.flags.auto_graph_replay=0` avoids the capture path (losing the
  replay speedup); re-running sometimes succeeds. Neither is reliable.
- Evidence: `bench/torch_compat --device npu --workloads sd15_unet_train`; peer
  Ascend runs 2026-10-06 under `$JITTOR_LAB_ROOT/_state/npu-verify`.
- Exit condition: a deterministic reproduction, the module-construction-time
  placement path covered, and `sd15_unet_train` completing across repeated 910B3
  runs with `backend_fallback=error` and zero fallbacks.

## KI-BACKEND-017: on Ascend, device-bound single-kernel workloads trail torch_npu

- Severity: Limitation (performance, Ascend)
- Status: Limitation. Measured on a 910B3 at `71eff3105`.
- Owner: ACL backend maintainers
- Symptom: against `torch` + `torch_npu` on the same card, Jittor's ACL backend
  is competitive or faster where graph recompute and operator fusion dominate
  (`qwen3_train` 0.89x, ~11-13% faster, loss agrees to 7.7e-8), but slower where
  a few large aclnn kernels dominate device time (`qwen3_prefill` ~3.2x,
  `vit_b16_train` ~9.9x). `jt.profile` shows the latter are device-bound: host
  launch is a minority of wall time and overlaps device execution, so the gap is
  aclnn single-kernel device time, not host overhead, missing fusion, or CPU
  fallback.
- Cause: individual aclnn kernels on the 910B3 run slower than torch_npu's, and
  Jittor does not yet close that at the kernel level (fusing into fewer/larger
  aclnn calls, HF32/precision modes, op-combo selection are unexplored). Raising
  `auto_graph_replay_retain_bytes` lets more graphs record as device graphs but
  saved only ~4% on `qwen3_prefill`, because the launches it removes already
  overlapped device work.
- Workaround: none; expect torch_npu-level or better throughput on
  training/launch-bound graphs and a gap on single-kernel-bound inference.
- Evidence: `bench/torch_compat --device npu`; peer Ascend runs 2026-10-06 under
  `$JITTOR_LAB_ROOT/_state/npu-verify`. Attributing per kernel needs a CANN
  msprof op-level timeline (jt.profile has no device-side equivalent on ACL).
- Exit condition: close the per-kernel device-time gap on the device-bound
  workloads, or accept and keep this as a documented characteristic.

## KI-OPS-002: integer floor-division ROCm verification incomplete

- Severity: Critical (until verified on the backend)
- Status: Open (verification pending: ROCm). Verified on CPU, CUDA and a real
  Ascend 910B3.
- Owner: binary operator maintainers
- Symptom: C++ integer division truncates negative quotients toward zero; the
  shared CPU/CUDA codegen subtracts one exactly when a nonzero remainder has the
  opposite sign from the divisor. Whether ROCm takes the same path is unverified.
- Workaround: on ROCm, compare representative negative operands against
  `numpy.floor_divide` before relying on the result.
- Evidence: [`test_floor_divide.py`](../../tests/ops/test_floor_divide.py),
  [`sample_floor_divide`](../../tests/opinfo/definitions/pointwise_binary.py),
  [Ascend 910B validation](../../docs/results/2026-08-28-ascend-910b-validation.md).
- Exit condition: the same fixed-vector and OpInfo coverage passes on a real
  ROCm device.

## KI-OPS-006: the NaN-correct CPU max/min reduction is slow on unpredictable data

- Severity: Medium (throughput; the answers are correct)
- Status: Open for random data. Max/min reductions use the blocked shape
  (`BlockedReductionPass` folds with an operation-aware combiner), which gained
  about 4x on monotone or constant input; on random input it gained 1.02x.
  CUDA is unaffected.
- Owner: reduction operator and CPU codegen maintainers
- Symptom: a 16.7M-element float32 `max`/`min` of `randn` data reduces at about
  5 GB/s on one CPU thread, where a `std::max` loop at the kernel's own flags
  (`-O3 -march=native`) reaches about 14 GB/s.
- Cause: the NaN-propagating comparison `((a > b) | (a != a)) ? a : b`
  (`src/type/minmax_compute.h`) is not recognised as a reduction by g++, and the
  reduce kernel's innermost stride is a run-time `storage_stride` value. Eight
  partials of the same expression are level with `std::max` only at unit stride;
  on random data the select is branch-bound, so blocking has nothing to hide.
- Workaround: none needed for correctness. Where throughput matters and the
  input cannot contain NaN, reduce on CUDA, or in an integer dtype (`a != a` is
  constant-false there and the compiler deletes it).
- Evidence: `benchmarks/reductions.py`;
  [`tests/codegen/test_blocked_reduction_pass.py`](../../tests/codegen/test_blocked_reduction_pass.py)
  guards the blocked max/min fold (combiner, seed, NaN, sign of the identity).
- Exit condition: the innermost stride is a compile-time constant when it is one
  (a versioned loop or a unit-stride specialisation), and float32 `max`/`min` on
  random data is within 10% of the `std::max` figure on the same benchmark.
  **Name the input pattern when you measure**: monotone and random data differ by
  up to 4x on the same kernel.

## KI-OPS-012: the CUDA half-precision max/min reduction folds from a finite literal

- Severity: Critical (silent wrong value on finite input as well as infinities,
  half dtypes only)
- Status: Open on CUDA; reproduced by inspection, unverified on a device. The CPU
  half is fixed.
- Owner: reduction operator maintainers
- Symptom: the CUDA reduction identity for float16/bfloat16 is a finite literal,
  so `max([-inf, -inf])` cannot answer `-inf`, and `max([-65504, -65504])`
  (float16's lowest finite value) answers a value that is not in the input; `min`
  mirrors it. Integer and float32/float64 reductions are unaffected.
- Cause: the CUDA map in [`fp16_op_type.cc`](../../src/type/fp16_op_type.cc) sets
  `init_maximum`/`init_minimum` to `-65000.0f`/`-1e38` (and the mirror). It
  should use `-CUDART_INF_F`/`CUDART_INF_F` for both half types, which means
  reaching `type/cuda_limits.h`; that header is pushed only by `parallel_pass`,
  not by the half post-pass, so the change must bring the include with it.
- Workaround: reduce half tensors in float32 on CUDA when the identity matters.
- Evidence: [`test_minmax_reduction_identity.py`](../../tests/ops/test_minmax_reduction_identity.py):
  the CPU class passes; the CUDA class carries the same body as a strict
  expected failure.
- Exit condition: the CUDA class's strict expected failure XPASSes on a device
  (i.e. the CUDA rows answer from an infinity) and the marker is removed.

## KI-OPS-014: a query row whose every key carries `finfo.min` gets float32 attention gradients as if each probability were 1

- Severity: Low (the forward is right; no workload in the benchmark reaches it)
- Status: Open; found 2026-10-01
- Owner: CUDA fused attention maintainers
- Symptom: with `finfo(float32).min` added to every key of a row, the float32
  fused attention backward is off by the row length (2 x 3 x 64 x 64 case:
  relative error 16, 6.6 and 1.8 for `dq`, `dk`, `dv` against float64). The
  output is the uniform average it should be. PyTorch's efficient-attention
  kernel is also wrong on these inputs (26% on output and `dq`). Transformers
  does not produce such rows on its SDPA path (`_unmask_unattended`); a mask
  built by hand can.
- Cause: `backends/cuda/kernels/nn/fused_attention_f32_cuda.py` stores the
  forward's log-sum-exp and the backward recomputes each probability as
  `exp(score - lse)`. Every score in such a row *is* `finfo.min`, and so is
  `lse = max + log(l)` (the `log(l)` is absorbed), so each probability comes back
  as 1 instead of `1/l`.
- Workaround: unmask fully masked rows before attention, as Transformers does.
- Evidence: `test_a_dense_additive_mask` with `mask[1, :, 7] = finfo.min`.
- Exit condition: store the row maximum and `1/l` separately (or `lse` relative
  to the maximum) so the backward never subtracts two equal huge numbers, and
  add the fully masked row to the test.

## KI-SEMANTICS-003: floating-comparison NaN semantics not verified on every backend

- Severity: Critical (until verified on the backend)
- Status: Open (verification pending: NPU dtypes other than float32, ROCm).
  Verified on CPU and CUDA; float32 verified on a real 910B3.
- Owner: compiler and comparison-operator maintainers
- Symptom: a backend whose kernels assume finite math can answer IEEE NaN
  comparisons wrongly (same-object and distinct comparisons, `isnan`/`isinf`/
  `isfinite`). On the verified backends floating and complex comparisons keep
  optimised `-O3` kernels without finite-math assumptions. General ACL float64
  support is unavailable.
- Workaround: for unverified dtype/backend combinations, compare representative
  NaN values against NumPy before relying on comparison masks.
- Evidence: [`test_nan_self_comparisons_across_dtypes`](../../tests/debug/test_kernel_traps.py),
  [`test_float_comparisons_with_nan`](../../tests/ops/test_fusion_correctness.py),
  [Ascend 910B validation](../../docs/results/2026-08-28-ascend-910b-validation.md).
- Exit condition: pass the remaining dtype matrix on a real NPU and the complete
  matrix on a real ROCm device.

## KI-DTYPE-002: implicit array construction narrows 64-bit NumPy values

- Severity: High
- Status: Limitation; the current default, with an explicit escape hatch
- Owner: dtype and compatibility maintainers
- Symptom: `jt.array` can narrow NumPy float64 and int64 inputs to 32-bit
  defaults, invalidating high-precision references or numerical gradient checks.
- Workaround: pass `dtype="float64"` or `dtype="int64"` whenever width is part of
  the contract.
- Evidence: [`test_jt_array_float64_narrowing`](../../tests/debug/test_kernel_traps.py).
- Exit condition: a public dtype-default decision changes the default and its
  explicit-dtype assertion together.

## KI-DTYPE-003: a Python float against a float64 tensor arrives as float32

- Severity: High (silent loss of 29 mantissa bits in float64 arithmetic,
  including in gradcheck)
- Status: Open; found 2026-09-18. Unlike KI-DTYPE-002 there is no escape hatch,
  because the scalar is written inline.
- Owner: dtype and compatibility maintainers
- Symptom: for `v = ones(1, float64)`, `v * 0.1` is `0.10000000149011611938`,
  `v + 0.1` is `1.1000000014901161194`, `v * (2/3)` and `v * 2 ** 0.5` are
  likewise correct to seven digits; `v / 3.0` is exact because 3.0 is exact in
  float32. Real torch answers all five exactly (a Python float is a weak double
  there). The result dtype is float64, which hides the defect.
- Cause: `ArrayOp::ArrayOp(PyObject*)` in `src/bindings/pyjt/py_array_op.cc`
  stores every Python float as `scalar.f32`. That conversion also serves
  `jt.array(0.5)`, so widening it there would make `jt.array(0.5).dtype` float64,
  a visible change to native Jittor's float32 default. The two call sites must be
  told apart first (an argument on the conversion, or a weak-scalar model).
  `auto_convert_64_to_32` is not the discriminator. Half precision is not
  affected: float32 is wider than both half types.
- Workaround: where precision matters, write the scalar as a float64 tensor:
  `v * jt.array(0.1, dtype="float64")` is exact, while `v * np.float64(0.1)` and
  `v * jt.array(np.array(0.1))` still narrow (checked on `1a6e203fc`, CPU).
- Evidence: [`TestScalarPromotion.test_float64_tensor_with_an_inexact_python_float_keeps_its_value`](../../tests/type/test_dtype_promotion.py)
  (an expected failure; its companion `..._with_an_exact_python_float_is_exact`
  must keep passing); `TestHalfPythonScalar` in
  `tests/type/test_half_precision_parity.py` pins the four half-precision
  expressions bit-identical to torch.
- Exit condition: a Python float keeps its value against a float64 operand in
  both modes, the expected failure XPASSes (strict) and its decorator is
  dropped, and the half expressions stay bit-identical to torch.

## KI-COMPLEX-001: native complex capability gaps

- Severity: Research (High for individual operations)
- Status: Open; explicit unsupported contracts
- Owner: dtype, autograd and linear-algebra maintainers
- Symptom: CUDA complex `prod`, second-order complex autograd/JVP, complex128,
  native complex linear-algebra kernels, and some CUDA eig environments are not
  supported; the calls refuse explicitly.
- Workaround: none beyond the supported complex64 surface.
- Evidence: [native complex dtype decision](../../docs/notes/complex-dtype.md).
- Exit condition: remove each sub-item only with focused CPU and accelerator
  tests for its operation and derivative order.

## KI-MEM-002: reading a device tensor relocates it to the host

- Severity: High
- Status: Open; reproduced on real CUDA on `1a6e203fc`
- Owner: memory and executor maintainers
- Symptom: `b.numpy()` moves a device tensor's storage to the host
  (`location()` goes `device` -> `cpu`). The next device operation migrates it
  back: 0.1214 s against 0.0006 s for the same operation on a resident 40 MiB
  tensor (215x), and device memory goes from +40 MiB to +80 MiB because both
  copies are live. `repr()` and `tolist()` take the same path, so printing a
  tensor at a REPL relocates it, and `d[0].item()` moves all of `d`. A reduction
  result is not affected: `u.sum().item()` leaves `u` on the device, so
  `loss.item()` is fine. PyTorch refuses `.numpy()` on a CUDA tensor instead.
- Workaround: read a computed copy, e.g. `(x + 0).numpy()`, which leaves `x` on
  the device (checked on `1a6e203fc`); `x.clone().numpy()` does **not** help,
  because the clone shares `x`'s storage and both move. Avoid `print(x)` and
  element indexing on large device tensors in hot paths.
- Evidence: `tests/core/test_var_residency_contract.py`;
  `tools/probes/side_effect_probe.py` (found the `tolist()` spelling);
  `benchmarks/transfer.py`.
- Exit condition: `numpy()`, `repr()` and element/slice reads leave the source
  Var's `location()` unchanged on real CUDA, a device operation right after such
  a read costs the same as one without it, and the residency contract covers all
  spellings including the reduction case.

## KI-AUTOGRAD-003: register_hook makes the receiver misreport its residency

- Severity: Low
- Status: Open; reproduced on CPU and CUDA
- Owner: autograd maintainers
- Symptom: after `a.register_hook(lambda g: g)` on a materialised Var,
  `a.location()` reads `"none"` (the state of an unmaterialised Var) while `device`,
  shape, dtype, the value and the hook itself are unchanged. Nothing is discarded
  or recomputed (measured: memory unchanged, the next device op costs 0.9x).
- Cause: `register_hook` in `python/jittor/_core/hooks.py` ends with
  `v.swap(hooker(v)[0])`, so the caller's Var carries the state of the hooker's
  not-yet-materialised output node, and `location()` answers for that node.
- Workaround: do not read `location()` immediately after registering a hook.
- Evidence: `tools/probes/side_effect_probe.py --device cpu` reports this as its
  only undeclared mutation (`location cpu->none`).
- Exit condition: `register_hook` leaves `location()` unchanged on CPU and real
  CUDA, and the side-effect probe reports no mutation for it.

## KI-EXEC-002: a profile scope cannot measure work launched before it opened

- Severity: Limitation
- Status: Limitation; inherent to deferred execution
- Owner: executor maintainers
- Symptom: `auto_flush_ops` (default 128, CUDA) launches pending work once that
  many operators have been built, so a graph constructed before
  `with jt.profile_scope()` may already have run when the scope opens and the
  report has no rows for it. The scope sets `auto_flush_ops=0` for its own
  duration (unless the caller overrides it) and raises a `RuntimeWarning` naming
  the cause when it measured no operators.
- Workaround: build the graph inside the scope.
- Evidence: `tests/ops/test_concat_op.py` (`test_concat_perf`,
  `test_concat2_perf` build their graphs inside the scope);
  `python/jittor/_core/diagnostics.py`.
- Exit condition: none planned; delete the entry only if the profiler gains a way
  to attribute work that ran before the scope.

## KI-EXEC-003: cuDNN autotuning is not isolated from execution scheduling

- Severity: High (silent, deterministic change to training numerics)
- Status: Open; cause identified, a small residue unexplained
- Owner: CUDA backend maintainers
- Symptom: the same model, input and build, with only `auto_flush_ops` changed:
  the forward loss is bit-identical at every setting but the gradients are not
  (for example a gradient norm of 33063.719 at 32 against 33076.398 elsewhere,
  about `3.8e-4` relative, the same number on every repeat). Two runs of one
  script with different `auto_flush_ops` train to different weights, and a check
  watching the loss reports nothing.
- Cause: `backends/cuda/kernels/cudnn/cudnn_conv_op.cc` chooses its algorithm by measuring the candidates and
  caches the winner per shape (`max_workspace_ratio` is in the key). What is
  resident during that measurement decides the winner, and `auto_flush_ops`
  changes what is resident. With autotuning off the spread collapses except for a
  `1.1e-6` residue at one setting, in the range a changed accumulation order
  would produce; not chased further.
- Workaround: `jt.cudnn.set_benchmark(0)` when comparing numerics across a flag
  that changes residency (the documented discipline in the numerics contract).
- Evidence: `tests/backends/cuda/test_autotuning_isolation.py`,
  `tests/backends/cuda/test_auto_flush_graph_split.py` (disables autotuning so
  the band does not hide smaller errors);
  [numerics contract](../../docs/notes/numerics-contract.md).
- Exit condition: the algorithm chosen for a shape does not depend on what else
  is resident (measure into a fixed-size scratch buffer, or key the cache on
  something stable); a regression sweeps `auto_flush_ops` and compares gradients,
  not the loss.

## KI-EXEC-004: pipelined execution costs about 43% more peak memory

- Severity: Medium (an undocumented trade-off decides whether a batch size fits)
- Status: Limitation; measured 2026-09-11, accepted and documented
- Owner: executor maintainers
- Symptom: a ResNet-50 training step (batch 32, 224x224, fp32, TF32 off,
  `set_benchmark(0)`, RTX 4090, whole-card `cudaMemGetInfo` peak) peaks at
  3.574 GiB with `auto_flush_ops=0`, 5.104 GiB at the default 128, and
  7.2-7.3 GiB at 8 or 1; PyTorch 2.1.2 on the same step is 3.74 GiB. The whole
  gap appears across the backward and is a steady-state peak, not a leak.
- Cause: a flush boundary keeps some activations alive past their last consumer;
  handed the whole graph, the scheduler can see every tensor's last consumer.
  Which tensors outlive their segment has not been identified (that needs
  `use_stat_allocator` lifetimes).
- Workaround: `jt.flags.auto_flush_ops = 0` when memory is the binding
  constraint; it is the first switch to try on an accelerator OOM, and the OOM
  message says so.
- Evidence: the measurement above (one point, batch 32; the absolute gap grows
  with batch and resolution, the ratio need not).
- Exit condition: the segment scheduler returns an input whose consumers have all
  run within the segment, or a test pins the ratio so a regression past 43% is
  reported.

## KI-EXEC-005: a var released by another thread mid-batch fails the batch

- Severity: High (a supported operation aborts; the pattern is a multi-threaded
  weight loader)
- Status: Open and currently unverifiable: the probe dies in KI-EXEC-007 first
- Owner: executor maintainers
- Symptom: four Python threads, each `jt.array(chunk)`, then
  `param[tid*N:(tid+1)*N] = host`, then `param.sync()`, 30 rounds on one shared
  `param`, failed at phase 7 with `exec_runner.cc: [check failed:
  v->mem_ptr || v->size == 0 || v->flag(_is_swapped) || ...]` on the shared var
  (2 of 15 runs before the batch hold was discounted, 1 of 20 after).
- Cause: thread B's slice assignment rebinds the holder, so the var thread A's
  batch requested becomes garbage while that batch runs and its memory is
  released. The batch's own hold used to count towards the backward liveness
  the assert reads. Phase 7 now tolerates exactly the vars whose storage
  `free_var_mem` released while a batch was in flight (`batch_released_vars`,
  `src/core/var.h`), but that change has not been measured against this failure.
- Workaround: serialise the threads that write the shared parameter.
- Evidence: `agent/skills/jittor-allocator-flag-matrix/probe_shared_param_threads.py`
  (the same pattern). No in-repo regression test: at 1-in-20 it would be a flaky
  gate; `tests/core/test_executor_entry_lock.py` covers a different property.
- Exit condition: close KI-EXEC-007, re-run the loader probe without failures,
  and make the race deterministic enough to gate.

## KI-EXEC-007: four threads writing one parameter free an allocation twice

- Severity: High (a supported operation aborts reproducibly in the pattern a
  multi-threaded weight loader uses)
- Status: Open; found 2026-09-18
- Owner: memory maintainers
- Symptom: the KI-EXEC-005 pattern on CUDA fails with
  `sfrl_allocator.cc: allocation not found: 3 [check failed: block != nullptr]`
  (`op: array in: float32[16384,256,] out: float32[4096,256,]`): 0/5 runs with
  one or two threads, 4/5 with four.
- Cause: thread B's slice assignment makes the old parameter var garbage and
  `free_var_mem` releases its storage while thread A's batch is between planning
  and its migrate loop; `migrate_to_cpu` (and `migrate_to_gpu`) then reads the
  stale `mem_ptr`/`allocation`/`allocator` and frees the block a second time
  (backtrace `VarHolder::sync -> Executor::run_sync -> run_exec_plan ->
  migrate_to_cpu -> SFRLAllocator::free`; the id was already reissued). The
  batch's hold keeps the Var alive but not its storage. The missing invariant is
  that storage the running batch will read is not released -- the in-flight
  counterpart of `_needed_by_backward`. Reordering the migrate closes the window
  after the copy but not before it, so it is not the fix.
- Workaround: serialise the threads that write the shared parameter.
- Evidence: `agent/skills/jittor-allocator-flag-matrix/probe_shared_param_threads.py`
  (`KI007_TRACE=1` turns on the id-space event log and a backtrace at the
  failing free).
- Exit condition: the probe passes repeatedly at four threads on CUDA, and
  KI-EXEC-005 can be measured.

## KI-EXEC-008: the CUDA convolution test files intermittently abort on a forward liveness underflow

- Severity: Medium (a whole pytest process aborts, taking its summary with it)
- Status: Open; found 2026-09-27, reproduced on `a57fb6af`
- Owner: core node liveness (the same counters as KI-EXEC-009, its backward
  counterpart)
- Symptom: `pytest tests/nn/test_*conv*.py tests/backends/cuda/test_cudnn_conv_a*.py tests/backends/cuda/test_cudnn_conv_p*.py`
  on CUDA aborts in about half the runs with `node.h: forward liveness release
  without a matching owner [check failed: value_ > 0]`, either at interpreter
  exit after every test passed or inside the conv-transpose reference of
  `test_cudnn_conv_backward_source.py`. Any single file, any pair, and the five
  files before it together passed every time; only the full selection trips it.
- Cause: not isolated; depends on collection order and timing.
- Workaround: run the files in separate processes.
- Evidence: the command above.
- Exit condition: the full selection passes repeatedly in one process.

## KI-EXEC-009: a taped multi-output `Function` op can release its backward liveness once too often

- Severity: Medium (an abort at interpreter exit, or two Vars leaked per
  occurrence; no wrong value has been observed)
- Status: Open, environment-dependent. Not reproduced on `e3c369acb` with Python
  3.11 (`tests/backends/cuda/test_cudnn_rnn_parity.py`,
  `tests/autograd/test_function.py` and
  `compat/tests/torch/test_torch_numerical_fidelity.py -k split_with_sizes`
  pass), but not fixed: `eaa50ed5f` bisected the disappearance to a docs-only
  commit, so it depends on the order in which `var_holder.cc` tears holders down.
- Owner: core node liveness (`src/core/node.cc`, `src/core/op.cc`); the forward
  counterpart is KI-EXEC-008
- Symptom: a multi-output op built through `Function` (a tape,
  `src/ops/composite/tape_op.h`) with some but not all outputs `stop_grad()`,
  executed as a batch, calls `LivenessCounter<backward>::release()` once too
  often: `node.h` reports `backward liveness release without a matching owner`.
  Where the teardown path catches it, exactly two Vars stay registered with
  `f=0 b=1` (`test_function.py::TestFunctionWithEagerExecution::test_zmem_leak{,2,3}`
  then fail with `2 != 0`); where it does not, the process aborts at exit (134).
  Seen as cuDNN LSTM training with `jt.grad` on CUDA, and as
  `torch.split_with_sizes` plus `Tensor.split` on CPU in Torch mode. Native
  multi-output `jt.code` ops do not leak.
- Cause: suspected, not confirmed: the needed-by-backward guard in `Op::init`
  (`manual_set_vnbb`) special-cases only `_outputs.size()==1 && ... is_stop_grad()`,
  and `Node::release_forward_liveness` enqueues one `release_backward_liveness`
  on the op for every finished, non-stop-grad output while the op's backward
  count may be one.
- Workaround: none needed for values; keep a process that hits it out of a shared
  pytest session.
- Evidence: `tests/core/test_core_invariant_properties.py`
  (`KNOWN_LEAKING_SHAPES`; `test_dropping_a_graph_leaks_nothing_new` fails on any
  new leaking shape, `test_dropping_a_graph_leaks_nothing_at_all` is a non-strict
  xfail, `test_the_leak_is_two_vars_per_occurrence` pins the count where it
  reproduces).
- Exit condition: fix the accounting (do not relax the `node.h` check), make
  `test_dropping_a_graph_leaks_nothing_at_all` pass on an interpreter where
  `KNOWN_LEAKING_SHAPES` reproduces, then empty `KNOWN_LEAKING_SHAPES`.

## KI-EXEC-010: backward graph construction does not take part in pipelined submission

- Severity: Limitation (performance; the device idles during backward graph
  construction)
- Status: Open
- Owner: executor / autograd
- Symptom: `jt.grad` and `Function` backward callbacks build the whole backward
  graph before any of it is submitted, so the device idles while the host
  constructs the backward of Llama- and UNet-sized steps.
- Cause: the explicit partial-graph boundary
  `jt.submit_pending(*vars, device_sync=False)` (`python/jittor/_core/var.py`;
  "动态形状与提交边界" in `docs/development/source-architecture.md`) has one
  production consumer, `compat/fsdp2/shard.py`. Cutting the backward graph inside
  `grad()` was tried and rejected; the intended route is to submit selected roots
  through that boundary (an `ExecPlan` handed to `run_exec_plan`, with timing
  owned by `SubmissionPipeline`).
- Workaround: none.
- Evidence: `tests/core/test_partial_graph_submit.py`.
- Exit condition: `jt.grad`/`Function` callbacks submit completed backward
  segments, and GPU idle time in the backward of a Llama and a UNet step drops
  without the step getting slower.

## KI-COMPAT-001: Torch namespaces publish native-only helpers and imported symbols

- Severity: Medium
- Status: Open; recorded in the Torch API manifest
- Owner: Torch compatibility frontend maintainers
- Symptom: `torch.Tensor` carries all 291 native `Var` methods, 37 of which
  PyTorch's `Tensor` has no equivalent for (`assign`, `start_grad`, `stop_grad`,
  `stop_fuse`, `reindex`, `reindex_reduce`, `reindex_var`, `migrate_to_cpu`,
  `migrate_to_gpu`, `fetch_sync`, `cast`, `float_auto`, `ceil_int`, `floor_int`,
  `round_int`, `safe_clip`, `debug_msg`, `peek`, `tape`, `candidate`, ...);
  `torch.nn.OrderedDict`, `torch.nn.deepcopy` and `torch.nn.partial` are imports
  leaking into a published namespace; and `torch.random` is a module subclass
  with `__call__`, where PyTorch's is a module only. Downstream code can bind to
  these, `dir(torch.nn)` advertises them, and they sit in the Torch coverage
  denominator.
- Workaround: do not treat a name's presence on `torch.*` as evidence that the
  Torch API has it; [`compat/torch/api_manifest.py`](../../compat/torch/api_manifest.py)
  holds the declared set.
- Evidence: [`tests/structure/torch_api_manifest.json`](../../tests/structure/torch_api_manifest.json).
- Exit condition: keep the native-only names and the imported symbols out of the
  published namespaces and regenerate the manifest in the same commit.

## KI-COMPAT-002: `torch.ops.aten` has no general native op surface

- Severity: Medium
- Status: Open. The import-time stop in vLLM-Omni is bridged by the vLLM adapter;
  the general surface is not.
- Owner: Torch compatibility frontend maintainers; vLLM adapter maintainers for
  the names it registers
- Symptom: `torch.ops.<ns>` (`compat/torch/library.py`, `_OpsDispatcher`) holds
  only operators the process registered through `torch.library`; the
  compatibility layer synthesises no aten operators, so code that binds
  `torch.ops.aten.<name>` raises `AttributeError: torch.ops.aten has no op ...`.
  The vLLM adapter (`adapters/jittor_adapters/vllm/aten_ops.py`, registered from
  its bootstrap when `vllm` is imported) registers `reshape` and `reshape.default`
  (forwarding to Jittor) and `_scaled_dot_product_flash_attention` /
  `_scaled_dot_product_efficient_attention`, which exist so module-scope bindings
  in vLLM and vLLM-Omni succeed and raise `NotImplementedError` when called. The
  other aten names vLLM/vLLM-Omni reference (`clone`, `copy`, `copy_`, `view`,
  `permute`, `unsqueeze`, `slice`, `slice_scatter`, `split_with_sizes`,
  `sym_size`, `mm`, `_scaled_mm`, `_scaled_dot_product_attention_flash_musa`,
  `_dyn_quant_matmul_4bit`, `_dyn_quant_pack_4bit_weight`) are not registered,
  and nothing outside the vLLM adapter gets any aten operator.
- Workaround: register the operators a library binds through `torch.library`
  in its adapter, structural ones forwarding to Jittor primitives and kernels
  Jittor lacks refusing with a clear error, as `aten_ops.py` does. Select a
  diffusion attention backend that does not call the refusing operators.
- Evidence: `adapters/jittor_adapters/vllm/aten_ops.py`;
  `adapters/jittor_adapters/vllm/bootstrap.py`.
- Exit condition: a general aten bridge in the compatibility layer (structural
  ops mapped to Jittor primitives, quantized and flash variants refused with a
  clear error), or a recorded decision that aten names are adapter-owned; then
  re-run the vLLM and vLLM-Omni imports and a real generation.

## KI-COMPAT-003: a nested tensor is not a Tensor

- Severity: Low (a stand-in for the few `torch.nested` paths verl reaches)
- Status: Open
- Owner: torch compatibility
- Symptom: `isinstance(torch.nested.as_nested_tensor(...), torch.Tensor)` is
  False here and True in torch 2.13; code that type-checks before branching takes
  the dense path.
- Cause: `compat/torch/nested.py`'s `_NestedTensor` holds a list of Vars plus
  their concatenation and reports `shape == (3, -1)`. A Tensor subclass is a Var
  subclass, and no Var can hold a ragged extent, so this needs real jagged
  storage rather than a cast.
- Workaround: none.
- Evidence: `compat/tests/torch/test_torch_compat_ops.py::TestNestedTensor::test_nested_jagged_basic`
  asserts the gap as it stands, so closing it turns the test red.
- Exit condition: nested tensors are Tensors with jagged storage, and the test is
  updated to the torch answer.

## KI-COMPAT-004: a factory result is not a backward leaf

- Severity: Low (`is_leaf` / `grad_fn` are introspection; gradients are correct)
- Status: Open
- Owner: torch compatibility / core autograd
- Symptom: `torch.ones(3, requires_grad=True).is_leaf` is False here and True in
  torch, and `grad_fn` names `broadcast_to`; code that branches on "is this a
  leaf" (parameter collection, gradient clipping helpers, some checkpointing
  wrappers) takes the wrong branch. From the other side, `torch.Tensor(other)`
  aliases the graph in torch but builds a fresh Var here, so the linkage is lost.
- Cause: `jt.ones` broadcasts a scalar literal (`_constant_scalar` in
  `python/jittor/_core/var.py`). Every float Var requires grad by default, the
  literal included, and `backward_grad_fn` (`src/core/grad.cc`) calls a Var a
  leaf only when no input of its producer requires grad. Making the literal
  `stop_grad()` gives the torch answer but breaks the double-backward `jvp`
  (`tests/autograd/test_autograd_functional_seeds.py::TestAutogradGradSeedAlignment::test_jvp_multi_input_is_unchanged`
  reads a tangent of 0.0 instead of 348.0), so it was reverted.
- Workaround: none.
- Evidence: `compat/tests/torch/test_torch_compat_autograd_semantics.py::TestBackwardLeafAndGradFn::test_leaf_and_intermediate_report_torch_shape`
  and `compat/tests/torch/test_independent_frontend.py::test_independent_tensor_installation_preserves_native_type`
  pin the gap.
- Exit condition: factory results are leaves without breaking `jvp`, and the two
  tests are updated to the torch answer.

## KI-COMPAT-006: `load_state_dict` shares the source's Vars, and an in-place optimizer then moves both models

- Severity: High on CUDA (silent: a second model -- an EMA copy, a reference
  model -- trains along with the first)
- Status: Open; found 2026-09-25
- Owner: torch compatibility / optimizers
- Symptom: after `b.load_state_dict(a.state_dict())` on CUDA, one
  `torch.optim.AdamW(a.parameters()).step()` changes `b`'s weights too; on CPU it
  does not. PyTorch copies in `load_state_dict`.
- Cause: loading binds `b`'s parameters to the Vars `a` holds, which is harmless
  while every update produces a new Var. The fused CUDA AdamW
  (`src/ops/composite/fused_adamw_op.cc`) writes into the parameters' own storage
  (`share_with`), so every holder sees the write.
- Workaround: copy through the host, e.g.
  `p.copy_(torch.tensor(src.cpu().numpy(), device=src.device))`.
- Evidence: two `nn.Linear(2, 2, device="cuda")`, `b.load_state_dict(a.state_dict())`,
  a backward and an AdamW step on `a`: `(a.weight == b.weight).all()` is True.
- Exit condition: `load_state_dict` copies (or the fused optimizer stops writing
  shared storage), with a CUDA regression test for the repro.

## KI-COMPAT-007: `w.copy_(x)` under `no_grad` loses `w`'s gradient when `x` requires grad

- Severity: Medium (silent: `w.grad` stays None after a backward)
- Status: Open; found 2026-09-25
- Owner: torch compatibility / autograd
- Symptom: `w = torch.randn(4, 8, requires_grad=True)`; `with torch.no_grad():
  w.copy_(x)` where `x` requires grad; a later backward through `w` leaves
  `w.grad` None. With an `x` that does not require grad it works. PyTorch keeps
  `w` a leaf that accumulates a gradient in both cases.
- Workaround: `w = x.detach().clone().requires_grad_(True)`.
- Evidence: the repro above.
- Exit condition: `w.grad` is populated in both cases, with a regression test.

## KI-COMPAT-008: `torch.optim.SGD` rejects `foreach=` and `fused=`

- Severity: Low (an immediate `TypeError`, not a silent divergence)
- Status: Open; found 2026-09-25
- Owner: torch compatibility / optimizers
- Symptom: `torch.optim.SGD(params, lr=..., foreach=False)` raises
  `TypeError: initialize_sgd() got an unexpected keyword argument 'foreach'`, and
  the same for `fused=`. Both are ordinary PyTorch arguments.
- Cause: `initialize_sgd` in `compat/torch/optim_frontend.py` does not accept
  them; the native SGD has a `fused` switch they could map onto.
- Workaround: drop the two arguments.
- Evidence: the call above.
- Exit condition: both keywords are accepted (mapped or explicitly ignored with a
  documented reason).

## KI-COMPAT-009: a float32 tensor divided by a Python float is computed in float64

- Severity: Limitation (performance; the result is the float32 PyTorch gives)
- Status: Open; a design decision is pending. Measured 2026-09-06 on an sm_89 GPU.
- Owner: torch compatibility / tensor operators
- Symptom: `x / 2.0` for a float32 `x` on CPU or CUDA casts `x` to float64,
  divides by a float64 0-d array and casts back, to match PyTorch to the last
  ulp. On GPUs whose FP64 rate is 1/64 of FP32 this is the largest single excess
  in the fused elementwise class: about 0.55 ms of a diffusers UNet2D step
  (every `ResnetBlock2D` ends in `/ self.output_scale_factor`); removing the
  widening measured 3.29 -> 2.73 ms for that class (see KI-CODEGEN-004).
- Cause: `use_wide` in the true-division path of
  `compat/torch/installers/tensor/method_api.py` (float32 widens to float64,
  float16/bfloat16 widen only to float32, ACL does not widen); the native
  `_fast_binary` builds the same widened sequence bit for bit.
- Workaround: multiply by the reciprocal (`x * (1 / s)`), which stays float32,
  where 1-ulp agreement with PyTorch is not required.
- Evidence: `compat/tests/torch/test_torch_compat_promotion.py`,
  `compat/tests/torch/test_native_fast_paths.py`.
- Exit condition: decide whether 1-ulp parity is worth FP64 on consumer GPUs (for
  example widen only where FP64 is fast, or compute `x * (1/s)` with a
  correction), update the promotion tests with the decision, and delete the entry.

## KI-DIST-001: FSDP2 flat sharding peaks above the unsharded model

- Severity: Limitation (memory; numerics are correct)
- Status: Open; measured 2026-09-08 on two CUDA ranks
- Owner: torch compatibility / distributed (FSDP2)
- Symptom: allocator high-water mark 19,977,728 B unsharded against 23,123,456 B
  with flat sharding (about 15.7% higher), flat over five steps, so it is a peak,
  not growth. A smaller CUDA Var snapshot is not a peak measurement and must not
  be cited as one: `core.get_peak_allocator_used_memory()` reads the memory
  profiler's executor-checkpoint high-water mark, and
  `get_mem_info().total_cuda_used` includes cached blocks.
- Cause: not isolated; the flat update regrouping and temporaries that are alive
  together are the suspects.
- Workaround: nonflat sharding peaked lower in the diagnostic run; neither is
  below unsharded.
- Evidence: `compat/tests/fsdp2/test_fsdp_memory.py` (four 512x512 Linear layers,
  batch 2, Adam, five steps, fresh ranks per mode); numerics in
  `compat/tests/fsdp2/test_fsdp2_nccl.py` and
  `compat/tests/fsdp2/test_fsdp_optimizer_math.py`. Reproduce in Torch mode
  (`JITTOR_TORCH_SHIM=1`, CUDA with NCCL):
  `JITTOR_FSDP2_MEMORY_MODE=full mpirun -np 2 python -m pytest -q -s compat/tests/fsdp2/test_fsdp_memory.py`,
  then `JITTOR_FSDP2_MEMORY_MODE=shard JITTOR_FSDP2_REFERENCE_PEAK_BYTES=<full peak> mpirun -np 2 ...`;
  the second fails today.
- Exit condition: the reference-peak run passes. NPU/HCCL and multi-node have no
  evidence yet (see [deferred-hardware.md](deferred-hardware.md)).

## KI-LINT-001: mypy covers the build utilities and part of compat only

- Severity: Limitation (static checking coverage)
- Status: Open. Import direction is gated separately
  (`tools/lint/check_import_layering.py` via `tests/structure/test_import_layering.py`
  and `nox -s imports`, with the remaining cycles as a ratchet that may only
  shrink).
- Owner: build and tooling maintainers
- Symptom: `python/jittor` outside `python/jittor/build` and the `backends/`
  overlay are not type-checked; when last measured they carried about 2,500 and
  80 errors respectively (about 2,600 errors in about 200 files).
- Workaround: none.
- Evidence: `[tool.mypy] files` in `pyproject.toml`; `nox -s typing`.
- Exit condition: extend `files` one package at a time with real fixes (no
  `# type: ignore`, no relaxed configuration); delete the entry when
  `python/jittor` and `backends/` are covered.

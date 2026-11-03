# Known-Issues Ledger

- Status: Maintained
- Last reviewed: 2026-11-02
- Baseline: `d8e30464c`
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

This is the main repository's core/compat ledger. A downstream model may be a
reproducer without making its underlying tensor, runtime or API defect an
adapter issue. Version-specific integration bugs and library support matrices
belong to the corresponding JittorEcosystem repository; link the two issues
when a core fix requires downstream revalidation. Moving documentation does
not close either obligation.

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
commit message or the GitHub issue/pull request; an entry
may name one refuted attempt when repeating it is the likely mistake.

## Ids

- `<AREA>` names the subsystem: `TEST`, `COMPILER`, `CODEGEN`, `TUNER`,
  `BACKEND`, `OPS`, `SEMANTICS`, `DTYPE`, `COMPLEX`, `MEM`, `AUTOGRAD`, `EXEC`,
  `COMPAT`, `DIST`, `LINT`. Add an area only when none of these fits.
- `<NNN>` is one more than the highest number the area has **ever** used,
  deleted entries included:
  `git log -p -- agent/manuals/known-issues.md | grep -o "KI-EXEC-[0-9]*" | sort -u | tail -1`.
  An id is never reused. (`KI-BACKEND-006` was once reused by mistake; the
  later entry was given a fresh id.)
- Code comments, test messages and documents cite the id. The change that
  deletes or renumbers an entry also rewords those citations
  (`git grep -n "KI-<AREA>-<NNN>"`); archived reports remain in Git history.

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
  exist. Optimizer state-dict coverage is now a real passing test and is no
  longer part of this entry.
- Workaround: do not cite these nodeids as passing evidence for NHWC cuDNN
  backward, tensor swapping, repeated dataset RingBuffer use, or legacy CUDA
  pooling.
- Evidence: `TestCudnnConvOp.test_backward_nhwc`, `TestCore.test_swap`,
  `TestCore.test_swap_cuda`, `TestRingBuffer.test_dataset`,
  `TestArgPoolOp.test_cuda_old_pool`
  (each carries a `KI-TEST-001` comment).
- Exit condition: fix and independently verify each contract, remove its
  expected-failure marker, and delete the entry when the list is empty.

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

## KI-TEST-019: ten native CPU oracle cases fail on the pre-cleanup baseline

- Severity: Medium (the native oracle gate is not green).
- Status: Open; confirmed 2026-11-03 on clean `0a23c2410cf3571e15200466b4caf4019f90d8cf`
  and the cleanup working tree based on `6c621be80e17cb2f968121bb30b6be278e7a3e72`.
- Owner: test infrastructure, native operators and compatibility maintainers.
- Symptom: with Python 3.11.18, NumPy 1.26.4 and independent CPU Torch 2.7.1+cpu,
  ten exact nodeids fail on both trees:
  `tests/distributed/test_process_store.py::TestProcessGroupOwnership::test_native_and_torch_share_classes_and_legacy_pickle`;
  `tests/core/test_var.py::TestVarFunctions::test_norm`;
  `tests/nn/test_conv_transpose.py::TestConvTranspose::{test,test_function}`;
  `tests/ops/test_unique.py::TestSparse::{test_unique,test_unique_dim}`;
  `tests/type/test_complex.py::TestComplexLinalg::{test_complex_matmul,test_complex_matmul_batch}`;
  `tests/type/test_complex.py::TestTensordot::{test_complex_tensordot_numberdim,test_complex_tensordot_tupledim}`.
- Cause: respectively, a child imports genuine Torch before requesting the shim;
  float32 L1 reduction relative differences around `1.6e-6` exceed `1e-7`;
  a convolution log-count assertion sees zero after numerical checks pass, and
  the functional case passes `bias=False` into a guard expecting a tensor or
  `None`; old positional `unique` arguments return two values where tests unpack
  three; legacy `ComplexNumber` reaches matmul shape validation without `ndim`.
  The log-count symptom overlaps [KI-TUNER-001](#ki-tuner-001-the-matmul-and-conv-relays-never-fire-so-a-hand-written-meta-op-product-runs-as-a-generic-kernel);
  it is not a second independently established tuner defect. All five test files
  are unchanged between the two trees. These failures predate the cleanup.
- Workaround: keep these cases reported as failures; do not replace the oracle
  with the shim or loosen numerical checks without an explicit error analysis.
  Use isolated process modes and current public argument conventions when
  constructing new comparisons; legacy complex-wrapper coverage remains open.
- Evidence: a serial retry of 245 original failed/error nodeids produced
  **187 passed, 48 skipped, 10 failed** in 764.60 s; the former random-concat
  timeout passed. The same ten failures on the clean baseline reproduced in
  70.97 s. The skip count is not passing numerical evidence.
  External, unversioned evidence lives under
  `$JITTOR_LAB_ROOT/_state/cleanup-implementation/`: `native-retry.log`,
  `native-baseline.log`, `native-retry-nodeids.txt`, `run_native_retry.py` and
  `run_native_baseline.py`. Commands were `native-retry-venv/bin/python
  run_native_retry.py` and `native-retry-venv/bin/python run_native_baseline.py`
  from that state directory; each invokes `python -m pytest -q --timeout=900`
  with exact nodeids, CPU-only flags, one runtime thread and serial compilation.
  Both use explicit `REAL_TORCH_SITE` from that environment; the baseline also
  sets `PYTHONPATH` to its isolated checkout's `python/`. Captured source paths
  confirm the baseline was imported. Logs are supporting local evidence, not
  required repository inputs; the listed nodeids and environment reproduce it.
- Exit condition: resolve and independently verify all ten contracts against
  genuine CPU Torch, preserving numerical and gradient coverage; remove this
  entry only when the exact cases pass without hiding failures or substituting
  the candidate as its own oracle. Accelerator support requires separate runs.

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
- Cause: not established in general. The op-level compiler is
  `src/core/parallel_compiler.cc` (`std::thread`), unrelated to the file-level
  `multiprocessing.Pool`. One cause was found and fixed on 2026-10-13: fused
  source generation (`OpCompiler::__get_fused_src`) ran a `std::regex_match`
  whose recursion grew with the source, and a compile thread's stack is glibc's
  2 MB default where `ulimit -s` is unlimited (Slurm nodes); a rank-13 kernel
  overflowed it with SIGSEGV and no backtrace
  (`tests/core/test_high_rank_tensors.py::TestCompileThreadStack`). Whether the
  reports below were all this has not been checked.
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

## KI-CODEGEN-001: fused elementwise kernels pay a division and a modulo per element for a broadcast operand

- Severity: Medium (performance; results are correct)
- Status: Open; a documented trade-off. Both `test3` cases run as strict expected
  failures with their original assertion (a `KI-CODEGEN-001` note on the case).
  A 2026-10-26 prototype on the unversioned `fix-codegen1` worktree reduced the
  recorded broadcast/dense ratio from about 5.2x to 1.77x, but it did not pass
  the original nested-loop contract and was not merged.
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
- Owner: code generation
- Symptom: the elementwise class of a `large_diffusers_unet2d` step was 3.37 ms
  (nsys) / 3.29 ms (profiler), 1086 GB/s, roofline ratio 0.84 -- already at the
  measured copy bandwidth -- against PyTorch's 3.04 ms.
- Cause: the positive excess of the 49 fused kernels was about 0.59 ms and none
  of it is code generation: 0.55 ms was the float64 scalar division that
  KI-COMPAT-009 carried -- that entry closed on 2026-11-01 by dividing in the
  tensor's dtype (see [历史记录](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-11-01-python-float-division.md)), so
  this 0.55 ms needs re-measuring rather than assuming -- 0.10 ms a bare
  `transpose` (`src/ops/composite/transpose_op.cc`), and 0.23 ms of launch
  latency across about 60 kernels that move almost no data.
- Workaround: none.
- Evidence: `agent/skills/cuda-elementwise-bandwidth-roofline/` (measurement
  scripts, nsys and profiler cross-checked, measured copy roofline).
- Exit condition: re-measure the class on the same step now that the division
  no longer widens; close when it is no slower, or narrow the entry to what
  remains.

## KI-TUNER-001: the matmul and conv relays never fire, so a hand-written meta-op product runs as a generic kernel

- Severity: High (a supported operation runs orders of magnitude slower than the
  library kernel that exists for it, with no error and no log)
- Status: Open for the hand-written form; the evidence cases below run as strict
  expected failures (each failing one carries a `KI-TUNER-001` note), so the day
  the relay fires they XPASS and the gate turns red. `test_group_conv_tuner.py`
  and the two `test_profiler.py` cases pass on the CPU gate (2026-10-12) and are
  not marked. `jt.nn.matmul`, `nn.Linear` and
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
  [Ascend 910B validation](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-08-28-ascend-910b-validation.md).
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
  [Ascend 910B validation](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-08-28-ascend-910b-validation.md).
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
  [Ascend 910B validation](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-08-28-ascend-910b-validation.md).
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
  must differ for it to pass); [numerics contract](../../docs/notes/numerics/numerics-contract.md).
- Exit condition: delete only if the default changes; that decision needs a
  compute-bound cost measurement with a pinned fusion shape, and the contract
  test and the numerics page change with it.

## KI-BACKEND-011: the oneDNN CPU path is float32-only and its per-call cost is unmeasured

- Severity: Limitation (narrower dtype support; performance unverified)
- Status: Open. The oneDNN v3 functional migration is in
  (`backends/cpu/libraries/mkl/onednn_runtime.cc` refuses oneDNN < 3 and keeps a
  bounded per-shape plan cache, 32 plans / 64 MiB scratch).
- Owner: CPU backend (oneDNN)
- Symptom: convolution and matmul through oneDNN accept float32 only; other
  convolution dtypes take the generic kernels (declared, not silent). Matmul has
  a CBLAS row behind it (`backends/cpu/libraries/blas`, numpy's bundled
  OpenBLAS by default) that takes float64 and complex64/complex128, and float32
  where oneDNN is not built (no cmake); float16/bfloat16 matmul stays generic.
  The plan cache replaced
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
  shape set; widen matmul through `dnnl::matmul` if fp16/bf16 are wanted;
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
- Evidence: [`docs/development/backends/acl-backend-contracts.md`](../../docs/development/backends/acl-backend-contracts.md)
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

## KI-BACKEND-016: ACL 507035 in batched strided-conv backward (sd15_unet_train, resnet50_train)

- Severity: Medium (crashes two training workloads on Ascend; deterministic
  on one of them)
- Status: Open. A second, deterministic repro found (2026-10-07):
  `resnet50_train` fails the same way on every run, not just sometimes --
  `sd15_unet_train`'s non-determinism (three isolated runs, three outcomes:
  507035, a different construction-time error, or a clean pass) is still
  unexplained on its own, but both workloads share strided convolutions with
  a backward pass, which is now the leading suspect over the original
  construction-time-arange theory below (`resnet50_train` has no DDPM
  timestep table at all).
- Owner: ACL backend maintainers
- Symptom: `sd15_unet_train` on a 910B3 sometimes aborts with
  `aclrtSynchronizeDevice failed with ACL status 507035`
  (ACL_ERROR_RT_VECTOR_CORE_EXCEPTION); another run instead fails earlier in
  setup with "code requires source for the selected backend", op `code` out
  `int32[1000]` -- the 1000-step DDPM timestep table built at module
  construction (a `torch.arange(1000)` issued before the front end's placement
  scope exists); another run completes cleanly (peak 19.09 GB). `resnet50_train`
  fails deterministically with the same 507035, on
  `exec_runner.cc:1093: Execute fused operator(643/979) failed`, op `code`
  (a `Conv2dBackward`) with shapes `float32[64,256,56,56]` (x),
  `float32[512,256,1,1]` (weight), `float32[64,512,28,28]` (dout) x2 -- a
  1x1, stride-2 downsample conv's backward, batch 64, no OOM, card idle.
- Cause: not isolated; one hypothesis tested and refuted. The shape above
  (`Hin=56`, `Hout=28`, stride 2, kernel 1, pad 0) does not satisfy
  `Hin == (Hout-1)*stride - 2*pad + dil*(k-1) + 1` at the stored,
  geometry-only `convOutPads` of `{0,0}` (`conv_op.py`'s
  `_conv_attr_code`) -- the gap is 1, suggesting `aclnnConvolutionBackward`
  needs a nonzero `output_padding` to reconstruct `Hin` from `Hout`. Tested
  directly against this exact shape by computing `output_padding` from the
  actual `out_[0]`/`in_[0]`/`in_[2]` shapes at runtime instead of reading the
  stored `{0,0}`: refuted, not just unhelpful -- aclnn rejects it outright,
  `AclNN_Parameter_Error(EZ1001): The value of outputPadding[1,1] must be 0
  if not transposed`. `output_padding` is not the lever ACL exposes for this
  in non-transposed mode; the actual mechanism behind the 507035 is still
  open. The original construction-time-arange theory (`c8fffce72` fixed the
  call-time version; the module-construction-time path is not covered) may
  still explain `sd15_unet_train`'s DDPM-timestep failure mode specifically,
  but cannot be the whole story now that `resnet50_train` -- no arange, no
  DDPM table -- hits the identical error code deterministically.
- Workaround: `JT_SYNC=1` completes `resnet50_train` cleanly (status ok, zero
  507035) -- serialising execution avoids it, which is the key narrowing below.
  `jt.flags.auto_graph_replay=0` avoids the capture path for `sd15_unet_train`
  (losing the replay speedup); re-running sometimes succeeds for it.
- Evidence: `benchmarks/torch_compat --device npu --workloads sd15_unet_train,
  resnet50_train`; peer Ascend runs 2026-10-06 and 2026-10-07 under
  `$JITTOR_LAB_ROOT/_state/npu-verify`. Narrowed 2026-10-07 and it is NOT the
  conv op or its tensor layout: (a) the exact repro shape run as a standalone
  single-conv backward (both grads, same batch 64) PASSES -- no 507035; the
  real failure is `Execute fused operator(643/979)`, i.e. the conv backward is
  op #643 of a 979-op fused execution batch, and only fails in that batched
  context. (b) `resnet50_train` under `JT_SYNC=1` completes cleanly. (c) a
  layout trace (`is_contiguous`/`storage_offset`/`mem_ptr`/aliasing on dout, x,
  weight, grad_input, grad_weight) before every `Conv2dBackward` in the
  `JT_SYNC=1` run showed every tensor fully contiguous, zero offset, no
  aliasing. So the trigger is the asynchronous/batched execution context --
  in-batch buffer reuse and free ordering across ACL's sequence of aclnn calls
  -- not a per-op kernel defect on a specific layout (cf. the gated-off
  `reuse_dying_inputs` on ACL, which is the same class of hazard: an aclnn op
  in a batch handed a block already reused or freed). This is executor-side.
  Next: capture the layout trace of the actual crashing (`JT_SYNC=0`) call to
  finish ruling out layout, then investigate the executor's per-batch release
  ordering for ACL (`src/core/exec_plan.cc` `release_after`, deferred frees in
  a non-async batch) -- needs a 910B3 to reproduce and verify.
- Exit condition: a deterministic reproduction's real mechanism identified
  (a CANN kernel defect needing a different call shape or an upstream
  report, or a jittor-side parameter/format bug fixable at the call site),
  and `sd15_unet_train` and `resnet50_train` both completing across repeated
  910B3 runs with `backend_fallback=error` and zero fallbacks.

## KI-BACKEND-017: residual Ascend workload performance gaps

- Severity: Limitation (performance, Ascend)
- Status: Limitation; historical device measurements through 2026-10-13.
  Documentation cleanup does not establish performance on the current tree.
- Owner: ACL backend, executor and shared Python dispatch maintainers
- Symptom: paired Ascend measurements still leave individual workloads slower
  than independent `torch` + `torch_npu`, even when the suite's geometric mean
  is below 1. The 12-workload run on `26bf23f9a` measured a mean of 0.98x;
  later worker/dispatch changes improved it further. These are different
  baselines, not current support claims. The latest recorded training A/B
  after async training admission measured `bert_base_train` 1.336x and
  `vit_b16_train` 1.131x; the earlier caller-time-gate A/B measured
  `qwen3_decode` 1.235x and `resnet50_infer` 1.260x. Do not combine those
  separate A/B runs into a new suite result. `sd15_unet_train` and
  `resnet50_train` remain blocked by KI-BACKEND-016.
- Cause and remaining leads:
  - ResNet inference pays ACL convolution's internal format conversions and
    `BNInfer` cost. Reformatting frozen weights is redundant, but the tested
    `aclnnCalculateConvolutionWeightSize` API explicitly rejected Ascend 910B
    (`only support ascend310P`); that API is not an available workaround.
  - Dynamic Qwen decoding still pays shared Python indexing/concatenation and
    graph-construction overhead. Recorded real calls selected fused prefill
    and incremental decode attention correctly, so do not assume an attention
    fallback. Memoized ACL programs, a compat concatenation fast path and ACL
    async execution improved it without eliminating the host cost.
  - BERT/ViT training remains sensitive to matmul and random-number cost.
    `DSARandomUniform` can overlap matmul on a separate queue; replacing the
    composite dropout with individually faster `aclnnDropout` made the actual
    BERT step 8% slower. The unfused path is not automatically the defect
    (see KI-BACKEND-018).
  - Async handover originally regressed three short inference steps by 2–5%.
    The caller-time admission policy improved `resnet50_infer_b1`; the
    recorded `bert_base_infer` and `sd15_vae_decode` regressions were not
    explained by that gate and remain to be measured/attributed. A byte-size
    threshold was ruled out because equally small batches can benefit or lose.
  - Earlier boolean-mask/fp16 dispatch, unused conv input-gradient and
    small-spatial GroupNorm fixes are historical completed work. Do not reopen
    them from obsolete pre-fix ratios; first prove a current regression.
- Workaround: none established for the remaining gaps. Keep same-card,
  same-session paired runs, matched warmup, zero fallback evidence and
  independent output/gradient checks. Isolated kernel improvements do not
  establish an end-to-end improvement.
- Evidence: `benchmarks/torch_compat` paired NPU workload runs and direct
  aclprof captures. The exact chronology, measurements, rejected approaches
  and commands are retained in the
  [pre-cleanup ledger](https://github.com/Jittor/jittor/blob/56cd4b562235697e9f1e09ad508a858936579a4d/agent/manuals/known-issues.md#ki-backend-017-on-ascend-device-bound-workloads-trailed-torch_npu-by-a-modest-margin-six-fixes-closed-the-gap-to-parity-on-average).
  Raw historical artifacts are unversioned under
  `$JITTOR_LAB_ROOT/_state/npu-verify/`; the completed six-fix comparisons
  include `full_bench_verify/results/` and `full_bench_verify2/results/`.
  Later A/B figures in the historical entry are bound to their recorded
  tree/environment, not to this documentation cleanup.
- Exit condition: remeasure each residual workload on a recorded current
  commit and independently attribute or fix the remaining gap, ruling out
  unintended fallback and mismatched host/warmup cost. Keep KI-BACKEND-016
  and unresolved worker correctness entries separate. Narrow this limitation
  only with real-device evidence; an aggregate mean does not close it.

## KI-BACKEND-018: `DropoutACL` reused the same mask forever and under-scaled its gradient, but nothing calls it

- Severity: Low (the class is unreachable from every live dispatch path, so
  no training run observes either defect) with a Medium-severity caveat: the
  coverage ledger that exists specifically to prevent this kind of defect
  from going unnoticed (`tests/structure/test_backend_grad_contract.py`)
  names a test for this class that does not actually exercise it, so if the
  class is ever wired back up, the ledger will not catch a regression.
- Status: Fixed (both defects), 2026-10-07, verified on a 910B3. The ledger
  mismatch is not fixed -- see Exit condition.
- Owner: ACL backend maintainers
- Symptom: `DropoutACL` (`backends/acl/kernels/ops/dropout_op.py`,
  `backends/acl/kernels/native/dropout_op_acl.cc`) is a `jt.Function`
  implementing dropout directly on `aclnnDropout`/`aclnnDropoutBackward`.
  Two independent defects, found while profiling `bert_base_train` for
  KI-BACKEND-017 and chasing why its dropout calls looked expensive (see
  that entry's evidence; this class turned out not to be the one `jt.random`
  uses):
  1. Every call produced the identical mask: `DropoutOpRunner::executeOp`
     read `attr->seed`/`attr->offset`, and the Python wrapper hardcoded both
     to `0`. Four consecutive calls with the same shape and `p` returned
     byte-identical masks.
  2. The gradient was under-scaled by a factor of `(1 - p)` whenever dropout
     was actually active: `DropoutACL.grad` hardcoded `DropoutBackward`'s
     `scale` attribute to `1.0` instead of `1 / (1 - p)` -- consistent with
     `p` never being stored on `self` in the first place, so `grad` had no
     way to compute the right value even if it had tried.
  Neither defect is reachable from a real workload: `jt.nn.dropout`
  (`python/jittor/nn/functional/dropout.py`, what `BertForSequenceClassification`
  hits through the torch-compat shim) is a separate, `jt.random`-based
  composite implementation that never calls `DropoutACL`, and it is correct
  (verified independently: two consecutive calls differ, gradient matches a
  closed-form reference). `DropoutACL` is not registered in
  `backends/acl/kernels/install.py`'s `KERNELS` table, so no dispatch path
  reaches it; a comment already in `dropout.py` explains the composite
  implementation was chosen specifically to keep dropout inside the
  surrounding kernel's fusion boundary, which a `jt.Function` like
  `DropoutACL` cannot do -- this class looks like it predates that design
  and was left behind, not a live alternate path.
- Cause: `DropoutOpRunner`/`MultinomialOpRunner`/`RandomOpRunner` all need a
  position in `aclnnDropout`'s (or the equivalent random op's) Philox-style
  stream; the project's established way to supply one is the global
  `current_seed`/`current_offset` pair (`src/runtime/init.cc`), which
  `RandomOpRunner` and `MultinomialOpRunner` already read and advance.
  `DropoutOpRunner` alone skipped this and used the Python-serialized
  attribute fields instead, which were never populated with anything but
  the literal placeholders `0, 0`.
- Workaround: none needed -- the class is dead code on every path this
  session traced.
- Evidence: a standalone correctness probe (not checked in, per the repo
  boundary) called `DropoutACL` directly: 4 calls, same shape and `p`,
  compared mask bytes -- identical before the fix, distinct after; a second
  probe compared gradients against `2 * out` with and without the correct
  `1 / (1 - p)^2` factor on a constant input. `tests/backends/acl/` 228
  passed (1 known unrelated failure); `tests/structure/` 1378 passed, 8
  failed (known bmm broadcast failures, see KI-BACKEND-017's sibling
  evidence) -- no new failures except
  `test_acl_launcher_contract.py::test_dropout_forward_uses_launcher_and_backward_remains_present`,
  updated in the same change since it asserted the now-removed
  `attr->seed`/`attr->offset` literals instead of the corrected
  `current_seed`/`current_offset` ones.
- Exit condition: either `tests/structure/test_backend_grad_contract.py`'s
  entry for `DropoutACL` points at a test that actually instantiates and
  runs it on real hardware, or the class (and its two unused source files)
  is deleted and the entry removed from that ledger. Whichever happens
  first; this session did neither, to avoid conflating a benchmark-speed
  investigation with a dead-code removal decision that deserves its own
  review.
- Follow-up (2026-10-14): the class was rewritten from a `jt.Function` to
  an `acl_emit` CodeOp with a secondary uint8 mask output (the LayerNorm
  pattern), so a future wire-up would keep `numel/8` bytes of mask and not
  a taped float activation; seed/offset already come from the per-device
  counter (`reserve_acl_random`). It was then wired through
  `try_dispatch("nn.dropout")` and measured on a 910B3 against the
  composite on the same tree: isolated, forward-plus-backward
  `aclnnDropout` cost 0.47-0.77x of `_keep_scale`, but a full
  `bert_base_train` step was 8% slower (96.4 vs 88.7 ms). The composite's
  `DSARandomUniform` runs on the `DSA_SQE` side queue and overlaps MatMul;
  `aclnnDropout` takes the main compute stream instead. The dispatch was
  reverted; the CodeOp rewrite stays so the class is not a footgun if it
  is ever reached again.

## KI-BACKEND-019: `argsort` has no registered constructor on ACL

- Severity: Medium (a supported operation raises), Open
- Status: Open, seen 2026-10-13 on a 910B3
- Owner: ACL backend
- Symptom: eight NPU-gate cases fail with
  `JittorError: op_register.h:64: constructor argsort with requested
  signature not found`, raised from
  `tests/opinfo/definitions/ordering_ops.py`:
  `test_acl.py::TestACL::test_argsort_values_indices_stay_on_acl` and, in
  `tests/ops/test_ops.py::TestCommonNPU`, `test_reference_argsort_float32`
  and `_float64`, `test_reference_sort_float32` and `_float64`,
  `test_reference_topk_float32`, `test_reference_median_float32`,
  `test_reference_kthvalue_float32`.
- Cause: not investigated here. Arrived with the upstream merge; the
  NPU-gate torch group went from 83 to 90 failures and the native group
  from 0 to 1 across that merge, and all eight are this one signature.
- Evidence: the eight cases fail identically on `659e6a47f` with no local
  commits applied -- a worktree at that commit alone, same card, same gate
  script -- so they are not from the ACL async-worker work that was being
  verified when they appeared.
- Exit condition: `argsort`'s ACL constructor is registered with the
  signature `ordering_ops.py` asks for, and the eight cases pass.

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
  [Ascend 910B validation](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-08-28-ascend-910b-validation.md).
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
- Evidence: `benchmarks/asv/reductions.py`;
  [`tests/codegen/test_blocked_reduction_pass.py`](../../tests/codegen/test_blocked_reduction_pass.py)
  guards the blocked max/min fold (combiner, seed, NaN, sign of the identity).
- Exit condition: the innermost stride is a compile-time constant when it is one
  (a versioned loop or a unit-stride specialisation), and float32 `max`/`min` on
  random data is within 10% of the `std::max` figure on the same benchmark.
  **Name the input pattern when you measure**: monotone and random data differ by
  up to 4x on the same kernel.

## KI-OPS-016: small dense linear algebra trails torch on host LAPACK, and QR's backward on CUDA

- Severity: Medium (performance; results are correct)
- Status: Open; measured 2026-10-26 on RTX 4090 / 16 cores of an EPYC 7763
  against torch 2.5.1 in the same Slurm job. The 2-D CUDA forwards are native
  since then (`backends/cuda/kernels/cusolver`, `cublas_trsm`): solve 64x64
  210 us (torch 147), QR 128x64 396 us (torch 370).
- Owner: linalg maintainers
- Symptom: complex64 `jt.linalg.solve` 64x64 takes 254 us on CPU (torch 97);
  from n = 128 up numpy's OpenBLAS LAPACK is 3-5x behind torch's MKL for QR.
  On CUDA, QR 128x64 forward plus backward takes 1.53 ms (torch 0.74). Batched
  (rank > 2) CUDA solve/QR/SVD, and every CUDA linalg op on a NVIDIA-wheel
  CUDA stack (which carries no cuSOLVER), still go through CuPy.
- Cause: the host forwards are `numpy_code` callbacks into numpy's OpenBLAS,
  whose LAPACK does not scale with threads at these sizes (`_host_lapack`
  keeps it on one thread below 2^25 units of work, where its own pool lost to
  Jittor's spinning OpenMP team). QR's backward recomputes QR live, for
  higher-order derivatives (`_reconnect`), and is a dozen small ops.
- Workaround: none.
- Evidence: `tests/linalg/test_lapack_paths.py` pins the current paths.
- Exit condition: CPU solve/QR through LAPACK from the library `setup_blas`
  loads, and a QR backward that reuses the forward, bring the cases above
  within 1.5x of torch.

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
  [Ascend 910B validation](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-08-28-ascend-910b-validation.md).
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

## KI-COMPLEX-001: native complex capability gaps

- Severity: Research (High for individual operations)
- Status: Open; explicit unsupported contracts
- Owner: dtype, autograd and linear-algebra maintainers
- Symptom: CUDA complex `prod`, second-order complex autograd/JVP, complex128,
  native complex linear-algebra kernels, and some CUDA eig environments are not
  supported; the calls refuse explicitly. On Ascend, `jt.fft.rfft` lowers to a
  fused `unary/imag` variant the ACL backend does not register, so the call
  refuses (`unregistered fused operator variant: unary/imag`) instead of
  computing on the device.
- Workaround: none beyond the supported complex64 surface.
- Evidence: [native complex dtype decision](../../docs/notes/numerics/complex-dtype.md);
  `tests/backends/acl/test_acl_indexing.py::test_acl_rfft_keeps_lazy_dft_constants_alive`
  on Ascend 910B3 with CANN 9.0.0 (2026-10-28).
- Exit condition: remove each sub-item only with focused CPU and accelerator
  tests for its operation and derivative order.

## KI-AUTOGRAD-005: a `numpy_code` or `jt.code` gradient cannot be differentiated again

- Severity: Limitation
- Status: Limitation, confirmed 2026-10-29 on `4543b0f2b` plus the change that
  made the `jt.code` case raise (it returned 0 before).
- Owner: autograd maintainers
- Symptom: a second derivative through an op whose backward is a numpy
  callback or a `cpu_grad_src`/`cuda_grad_src` source raises "cannot
  differentiate a ... gradient again". That includes max/min pooling, which
  runs as a `jt.code` op by default, so a gradient penalty through
  `jt.nn.max_pool2d` raises; `complex_pinv` and `complex_eig` have no
  backward at all.
- Cause: the backward is opaque user code; nothing says how its result
  changes with its inputs (`torch.autograd.function.once_differentiable` is
  the same contract).
- Workaround: write the op as a `jt.Function` whose `grad` uses
  differentiable jittor ops. For pooling, `jittor.pool.pool_use_code_op =
  False` selects the reindex implementation, which differentiates twice.
- Evidence: `tests/ops/test_code_op.py::TestCodeOp::test_second_derivative_through_a_code_gradient_raises`,
  `tests/ops/test_numpy_code_op.py::TestCodeOp::test_second_derivative_through_a_numpy_code_gradient_raises`.
- Exit condition: pooling's default path differentiates twice, and the
  remaining callback ops either do or are documented as first-order only.

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
  [numerics contract](../../docs/notes/numerics/numerics-contract.md).
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
- Status: Open (verification): the canonical four-thread probe now passes 35/35
  on this tree; the earlier failure was hidden behind the shared-parameter
  storage race, now fixed. A deterministic gate is still missing.
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
  `src/core/var.h`).
- Workaround: serialise the threads that write the shared parameter.
- Evidence:
  `agent/skills/jittor-allocator-flag-matrix/probe_shared_param_threads.py`
  passes 35/35 at four threads and 6/6 at one on GPU 1 (2026-10-26). The
  `.numpy()` variant of the same shape also passes after the read-path
  ownership and ring-lock fix. No in-repo regression test: the race has not
  been made deterministic;
  `tests/core/test_executor_entry_lock.py` covers a different property.
- Exit condition: make the race deterministic enough to gate, then delete.

## KI-EXEC-008: the CUDA convolution test files intermittently abort on a forward liveness underflow

- Severity: Medium (a whole pytest process aborts, taking its summary with it)
- Status: Open; found 2026-09-27, reproduced on `a57fb6af`
- Owner: core node liveness (the backward counterpart of this forward counter)
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

## KI-EXEC-011: a fork while a batch is in flight deadlocked three ways

- Severity: High (a hang, not a wrong answer), Fixed
- Status: Fixed (2026-10-09)
- Owner: executor / async execution
- Symptom: a Qwen3 decode step on ACL hung with the device idle. Three
  threads, in a cycle: the main Python thread held the graph lock and waited
  for the GIL inside `GraphEntryScope`; a second thread held the GIL inside
  `subprocess.Popen` -> `__libc_fork` -> the `pthread_atfork` prepare handler
  `before_fork` -> `wait_for_worker`; and the worker spun for the graph lock
  in `GraphLockSuspend`'s destructor between its segments.
- Cause: `before_fork` (and `stop_worker`) called `wait_for_worker` bare,
  holding both the GIL and whatever graph lock the calling thread had.
  `async_exec_wait` already released both for the same wait and explained
  why -- the worker takes the graph lock between segments, and compiling an
  operator calls back into Python -- but the two other callers did not. Not
  ACL-specific: the second thread only forks, which a dataloader or an
  `nvidia-smi`/`npu-smi` sampler does, and CUDA was equally exposed. It had
  simply never been exercised, because nothing forked mid-batch in the
  CUDA benchmarks.
- Workaround: none needed; `async_execution=0` avoided it before the fix.
- Evidence: native backtraces of all 211 threads of the hung process
  (`eu-stack`) showing the cycle. Regression test:
  `tests/runtime/test_async_execution.py::...::test_a_second_thread_may_fork_while_a_batch_is_in_flight`,
  which forks from a second thread with a batch in flight.
- Fix: one `wait_for_worker_unlocked` -- graph lock and GIL released, no
  error rethrown so a fork handler can use it -- for all three callers.

## KI-EXEC-012: ACL failed every operator on a second Python thread

- Severity: High (a supported operation aborts), Fixed
- Status: Fixed (2026-10-09)
- Owner: ACL backend
- Symptom: `jt.array(...)` on any thread but the first raised `RuntimeError:
  aclrtGetDevice failed with ACL status 107002` on a 910B3. A multi-threaded
  weight loader, a dataloader thread, or the async-execution worker all hit
  it on their first operator.
- Cause: a thread that has never bound a device has no ACL context, and
  `aclrtGetDevice` reports that as a failure rather than answering "none".
  `acl_runtime_current_device()` and `set_device()` both treated it as an
  error. CUDA never had the problem because `accelerator_current()` binds a
  thread whose `tls_bound_device` does not match the process device.
- Workaround: bind the device on the thread before using jittor from it.
- Evidence:
  `tests/runtime/test_async_execution.py::...::test_a_second_python_thread_may_build_meanwhile`,
  which was CUDA-only-skipped before and now runs on ACL.
- Fix: both ACL entry points treat the status as "unbound" and bind the
  device the process is using, mirroring CUDA.

## KI-EXEC-013: the worker dropped a `jt.Function`'s Python callable without the GIL

- Severity: High (a segfault), Fixed
- Status: Fixed (2026-10-13), verified on a 910B3
- Owner: executor / async execution
- Symptom: `ddpm_unet_train` and `qwen3_train` on ACL died with
  `Caught segfault at address ...` the first time the async-execution worker
  was allowed to take a training forward's batches. The backtrace was the
  same in both: `worker_main` -> `Executor::run_sync`'s batch lambda ->
  `run_exec_plan` -> `Tapes::~Tapes` ->
  `from_py_object<GradCallback>`'s deleter lambda.
- Cause: a tape built from a `jt.Function` owns the Python callable its
  backward calls, and `GradCallback`'s deleter is a bare
  `Py_DECREF(obj); Py_DECREF(frontend_type)`. When the last owner of such a
  tape is a batch the worker is winding down, that runs on a thread with no
  GIL, which is a memory corruption rather than a refcount. Not ACL-specific
  in itself, but only ACL hands a training forward over
  (`async_admits_grad`), and ACL reaches it constantly because its
  convolutions and norms are `jt.Function`s. Latent before that: under
  `no_grad` no tape exists, which is what had kept the worker safe.
- Workaround: `async_execution=0`.
- Evidence: the two crash reports with their resolved frames (`addr2line`
  against the run's `jittor_core` build). Regression test:
  `tests/runtime/test_async_execution.py::...::test_a_python_owned_tape_released_by_the_worker`,
  which puts a `jt.Function` in a chain long enough to be handed over and
  compares the gradient against the serial run.
- Fix: `async_exec_defer_python_release` -- on the worker the two references
  are queued instead of dropped, and the Python thread runs the queue at its
  next executor entry (`async_exec_run_python_releases`, which also checks it
  holds the GIL). Off the worker the deleter still drops them itself.

## KI-EXEC-014: ACL worker intermittently segfaults in `release_forward_liveness` on `qwen3_decode`

- Severity: High (a segfault), Open
- Status: Open, seen once on 2026-10-14 on a 910B3 after rebasing onto
  `659e6a47f`; not reproduced in nine subsequent full-size runs
- Owner: executor / async execution
- Symptom: one `qwen3_decode` process of a 12-workload bench on the
  rebased tree died with `Caught segfault` on the worker thread. Resolved
  frames: `worker_main` -> `run_exec_plan` ->
  `Node::release_forward_liveness` / `Node::release_inputs` /
  `LogFatalVoidify` (so a `CHECK_EXIST` on a freed node, then a secondary
  crash in the fatal path). Not the GradCallback path of KI-EXEC-013:
  decode runs under `no_grad` and builds no tapes.
- Cause: not diagnosed. Appeared after rebasing the ACL async-worker work
  onto upstream `659e6a47f` (352-file Refactor 2.0); the same decode had
  completed cleanly in every pre-rebase run on this card. Nine follow-up
  runs (three targeted, six back-to-back) all passed with the worker on
  (median ~10.5 s/step vs 16.1 s with `async_execution=0`). Timing-sensitive
  use-after-free in the shared liveness queue is the working hypothesis;
  `run_liveness_queue` already documents that two threads draining at once
  would corrupt each other's callbacks.
- Workaround: `async_execution=0` (and loses the ~35% decode gain).
- Evidence: the one crash report under
  `$JITTOR_LAB_ROOT/_state/npu-perf/rbfull/rb_full/logs/qwen3_decode.jittor.log`;
  the nine succeeding runs under `decrep/` and `decstress/`.
- Exit condition: either a deterministic repro and a fix, or enough clean
  runs across cards that the single observation is treated as a one-off
  and this entry closed.

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
  The vLLM adapter ([aten_ops.py](https://github.com/JittorEcosystem/jittor-vllm/blob/main/jittor_vllm/aten_ops.py), registered from
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
- Evidence: [aten_ops.py](https://github.com/JittorEcosystem/jittor-vllm/blob/main/jittor_vllm/aten_ops.py);
  [bootstrap.py](https://github.com/JittorEcosystem/jittor-vllm/blob/main/jittor_vllm/bootstrap.py).
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

## KI-COMPAT-004: Tensor aliases lack mutation guards and dynamic source trainability

- Severity: Medium (autograd view semantics differ from Torch)
- Status: Open; factory leaves, grad-enabled constructor linkage and no-grad
  accumulator eligibility resolved in their verified ranges; recorded producer
  identity now survives a later source freeze (CPU/CUDA verified)
- Owner: native autograd / torch compatibility Tensor constructors
- Symptom: a no-grad-created `torch.Tensor(source)` alias remains usable for
  grad-mode arithmetic after the alias or source is modified with `fill_` or
  `copy_`. Genuine Torch rejects this use and `is_leaf`/`grad_fn` inspection.
  Separately, enabling a previously frozen source does not dynamically update
  its alias's visible `requires_grad`; genuine Torch updates the flag without
  activating an accumulator, in either creation mode.
- Cause: native alias storage and gradient-edge eligibility are represented,
  but creation-mode mutation guards and dynamic source trainability are not.
- Workaround: avoid reusing no-grad-created aliases after writes; create the
  intended alias after setting the source's requires-grad state. Use explicit
  detach/activation only when an independent trainable leaf is intended.
- Evidence: the 41 CPU/CUDA contracts in
  `compat/tests/torch/test_tensor_constructor_alias.py` cover construction,
  source/alias gradients, activation timing, clone/detach/retain and invalid
  roots. Separate genuine Torch and candidate CPU probes confirm 12 mutation
  variants lack the expected rejection, and eight frozen-source/unfreeze
  variants lack dynamic flag propagation. These are remaining failures, not
  covered by the passing constructor matrix. Raw probes are unversioned under
  `$JITTOR_LAB_ROOT/_state/tensor-alias-main-0fb/` and the independent
  `tensor-alias-independent-review` / `tensor-alias-unfreeze-review` states;
  see the [constructor follow-up](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-10-15-size-factory-leaves.md#tensor-source-constructor-follow-up).
- Follow-up: [recorded producer identity](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-10-16-recorded-grad-fn-freeze.md)
  fixes source-freeze grad_fn loss only; the independent lifecycle probe of 32 combinations
  still finds 17 differing combinations. Mutation guards and inherited flags stay open.
- Exit condition: native view creation/mutation metadata and source-state
  propagation agree with independent Torch on CPU and each claimed accelerator,
  including absence of unintended accumulator activation. Preserve the existing
  old/new graph snapshots and storage/gradient contracts when adding this support.

## KI-COMPAT-011: an RNN under the Torch frontend on CUDA does not run through cuDNN

- Severity: Medium (the precision policy it is meant to follow is not exercised)
- Status: Open; reproduced 2026-10-12 on CUDA, with CUPTI available
- Owner: torch compatibility / RNN maintainers
- Symptom: `test_precision_policy_isolation.py`'s two RNN cases find no
  `rnn precision select:` line from `cudnn_rnn_descriptor` ("the RNN must execute
  actual cuDNN descriptors"); the outputs are computed by another path.
- Cause: not isolated.
- Workaround: none needed for results.
- Evidence: `compat/tests/torch/test_precision_policy_isolation.py::test_fp32_rnn_native_default_and_torch_cudnn_policy_match_cpu_recurrence`,
  `::test_grouped_rnn_backward_retains_forward_precision_after_policy_switch`.
- Exit condition: both cases pass on CUDA.

## KI-COMPAT-012: in-process `JITTOR_TORCH_SHIM=1` publishes no `torch` distribution metadata

- Severity: Medium (Transformers and Diffusers decide PyTorch is absent)
- Status: Open (verification pending: downstream install). `c6aef610a` makes
  the in-process shim register a process-local torch distribution finder on
  activation and remove it on release; the metadata-dependent library cases
  still need Transformers/Diffusers installed on the acceptance machine.
- Owner: torch compatibility maintainers
- Symptom: `AutoModel ... requires the PyTorch library but it was not found in
  your environment` / `UNet2DModel requires the PyTorch library` from
  `importlib.metadata`: twelve cases in `test_torch_hf_models.py`,
  `test_torch_hf_alias.py`, `test_torch_hf_cuda_device.py` and
  `test_diffusers.py` raise `ImportError`, and `test_peft.py` fails collection.
- Cause: the shim installs a `torch` module in the process but no `torch`
  distribution, and those libraries ask `importlib.metadata` rather than
  importing. A user who runs `import torch` with the deployed shim does not hit
  it.
- Workaround: deploy the shim (`jittor-torch-shim`, see
  `docs/compatibility/torch-shim.md`) rather than enabling it in-process.
- Evidence: the four test files above on a machine with transformers/diffusers.
- Exit condition: the four evidence files' cases run on an environment with
  Transformers/Diffusers installed (the metadata mechanism itself is covered by
  `compat/tests/torch/test_torch_shim_distribution_metadata.py`).

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

## KI-BACKEND-020: the NPU work in #18036 and #18039 was only partly on this branch

- Severity: Medium (supported operations failed or answered wrongly), Fixed
- Status: Fixed (2026-10-14), verified on a 910B3
- Owner: ACL backend / Torch compatibility
- Symptom: the maintainer's cherry-picks from #18035, #18036, #18039 and
  #18031 took their backend-agnostic parts, so the Ascend-specific halves
  were missing. What that cost, each confirmed on hardware before the fix:
  `labels[:, mask]` raised `NotImplementedError: ACL getitem does not support
  Var slice indices`; a boolean slice assignment had no lowering because CANN
  refuses a bool payload; `x[None]` raised a TypeError; `roll` declined int64;
  `nonzero` counted with a bool sum, so more than one match shaped the output
  for a single row, and an all-false mask reached a kernel that rejects a
  zero-row output. `torch.npu.is_available()` answered False and
  `device_count()` 0 on a machine with eight 910Bs, because `torch.npu` got
  the same placeholders as `torch.mps` and `torch.xpu`; a downstream library
  that asks before choosing a device believed it. `torch.use_deterministic_
  algorithms` was a no-op setter and the query a constant False. A
  device-to-device copy between two cards was refused outright. The ACL RNG
  counter was one process-wide pair, not one per device, and could not be
  checkpointed.
- Cause: the cherry-picks were deliberately scoped to backend-agnostic
  changes; nothing recorded which Ascend-specific parts were left, so they
  were not picked up later either.
- Workaround: none.
- Evidence: six commits, each with its own hardware verification, from
  `a48b4e828` to `1df5bef15`. New tests, all run on a 910B3: four ACL index
  cases in `tests/backends/acl/test_acl_torch_compat.py`, five in
  `compat/tests/torch/test_npu_device_protocol.py` (and its CPU half, which
  asserts the facade still refuses rather than inventing a device), the
  deterministic-policy case, ten in
  `tests/backends/comm/hccl/test_hccl_backend_status.py`,
  `compat/tests/torch/test_npu_peer_copy.py` on two visible cards, and four
  per-device RNG cases including one that runs the draws on the
  async-execution worker. Gates after the merge: core tier 322 passed; the
  NPU gate's failure set byte-identical to the pre-merge baseline (91 both
  ways, `comm` empty) with passes up from 466+241 to 471+245;
  `tests/structure` green except the upstream ratchet below.
- Not merged, and why: #18030's getitem stride fix and #18031's ACL changes
  are already on this branch; #18028 shares #18031's head against a different
  base. #18036's `truth_reduce` keepdims push-down duplicates the reshape
  `_truth_reduce_keepdims` already does, and its `getitem_op.py` and
  `tensor.py` are behind this branch's memoization work, which was kept.
  `torch.npu`'s memory metrics were left refusing rather than answered,
  because the memory-pool bindings they read (`device_pool_memory_used` and
  the rest) are not on this branch; a zero peak reads as a measurement.
- Open alongside this: ACL `all()`/`any()` with no dim returns shape (1,)
  where torch returns a scalar -- `test_reference_all_*`/`test_reference_any_*`
  in the NPU gate's baseline failures. Neither branch nor PR fixes it.


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

## KI-CODEGEN-006: a rank-1 full sum still materializes a pending reshaped producer

- Severity: Medium (allocation and performance; values are correct)
- Status: Open; reproduced 2026-10-26 on current `2.0-refactor` (real CUDA,
  RTX 4090). The multi-dimensional spelling was fixed by `c56b4c2bc` and is
  pinned by `tests/backends/cuda/test_full_reduce.py`.
- Owner: CUDA reduction and codegen maintainers.
- Symptom: `(jt.ones((1,)).expand(1 << 24) * 2).sum()` allocates 68,157,440
  bytes cold because the two-stage CUB path materializes the multiply. The 2-D
  `(4096,1).expand(4096,4096)` spelling allocates 1 MiB.
- Cause: the rank-1 route needs a reshape/view above the pending producer;
  `ReshapeOp` is `OpType::other` and `edge_fusable` refuses it, so the fused
  route stops at that barrier. The only non-materializing alternative measured
  is a single-block `use_shared_reduce=2` fold at ~3.8 GB/s versus ~757 GB/s
  CUB (131 ms vs 0.66 ms at 512 MB); the atomic generated reduce is
  nondeterministic (15 distinct values in 20 runs).
- Workaround: reduce an explicit axis or materialize the producer when the
  allocation matters.
- Evidence: unversioned probe/logs under
  `$JITTOR_LAB_ROOT/_state/fix-codegen5/`; the multi-dimensional allocation
  gate is `tests/backends/cuda/test_full_reduce.py`.
- Exit condition: codegen fuses through the reshape above a pending producer
  deterministically and near CUB throughput, or a justified deterministic
  strategy is selected; the rank-1 allocation gate passes.

## KI-TEST-016: the ACL Torch-compat suite is red under the shim for placement reasons

- Severity: Medium (fourteen pre-existing failures in one file would hide a real
  regression in the same file)
- Status: Open; reproduced 2026-10-29 on an Ascend 910B3 at `1a750dbee` and
  `d3ae2bbd6` with identical results.
- Owner: test infrastructure maintainers
- Symptom: in native mode `pytest tests/backends/acl` reports 1 failed (the
  rfft case) / 441 passed / 11 skipped. Under `JITTOR_TORCH_SHIM=1` the same
  directory reports 15 failed / 455 passed / 12 skipped, and 14 of the failures
  are in `test_acl_torch_compat.py`: relu, silu, pad, roll, rms_norm,
  group_norm, batch_norm and fused_adamw among them. Their assertions
  (`_assert_acl_device`, `output.is_cuda is False`) see `'cpu' != 'device'`.
- Cause: under the shim on this machine a tensor made by `torch.tensor(...)`
  lands on the host, so the ACL provider is never asked and the recorders and
  placement checks see nothing. Those cases rely on `jt.flag_scope(use_acl=1,
  use_cuda=1)` making the accelerator the default device, which does not happen
  in this mode on this machine. Not yet distinguished: the flag scope's default
  device against the shim's device resolution.
- Workaround: run the ACL gate in native mode, where the numbers above hold; a
  case that needs shim placement states `device="npu"` itself, as
  `test_sdpa_boolean_mask_layouts_stay_on_acl` now does.
- Evidence: the two directory runs above on a 910B3.
- Exit condition: the file passes under the shim on a 910B3, or every placement
  case states its device explicitly.


## KI-COMPAT-023: a SiLU module's first call and its later calls round differently

- Severity: Low (one half-precision ulp, about a quarter of the elements)
- Status: Open; reproduced 2026-11-01 at `3253801e8`, and with the day's
  division changes reverted and the core rebuilt (`d8e30464c`) too.
- Owner: torch compatibility / half-precision numerics
- Symptom: `compat/tests/torch/test_native_fast_paths.py::TestNativeModuleDispatch::test_an_inference_group_norm_and_the_silu_after_it`
  fails on CUDA where it compares `torch.nn.Sequential(silu)(plain)` with
  `jt.nn.silu(plain)` for half `plain`: 52-77 of 256 elements differ by one half
  ulp. The smaller reproduction is the module alone:

  ```python
  silu = torch.nn.SiLU()
  plain = torch.randn(2, 8, 4, 4, device="cuda").half()
  with torch.no_grad():
      first = silu(plain)       # == torch.nn.functional.silu(plain) and
      second = silu(plain)      #    == (plain.float32() * plain.float32().sigmoid()).half()
  # first and second differ in ~68 of 256 elements
  ```

  The **first** call on a fresh module is the one that matches real PyTorch 2.11
  bit for bit (0 of 256 over several seeds) and the widened float32 path; every
  later call on that module answers as if the operands had stayed half
  (`x * sigmoid(x)` in half precision), which is the 1-ulp side. `silu(plain)`
  and `silu(plain.clone())` differ the same way, so it is a property of the
  module/process after the first call, not of the tensor.
- Cause: not isolated. Not the tuner (`enable_tuner=0` reproduces), not graph
  replay (`auto_graph_replay=0` reproduces), not the dispatch cache
  (`_prefer_forward` is class-level and stable), and not the ACL kernel (not
  used on CUDA). The remaining suspect is the widening inside `nn.silu` --
  `wide = x.float32(); (wide * wide.sigmoid()).cast(dtype)` -- losing its
  float32 operands on the path the second call takes.
- Workaround: none needed for accuracy. A test that asserts bit parity with
  PyTorch should compare against a spelling that has not already run in the
  same process, or compare with real PyTorch rather than between two Jittor
  spellings.
- Evidence: the snippet above and the nodeid, one process each, on an RTX 4090
  with `JITTOR_TORCH_SHIM=1`; the values are also recorded in
  [历史记录](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-11-01-python-float-division.md).
- Exit condition: repeated calls on one module answer the same bits, and the
  case is green.


## KI-COMPAT-024: storage façades and some constructed views retain alias gaps

- Severity: Medium (storage identity or Python writes can differ from Torch).
- Status: Open; observed during CPU/CUDA work based on `f995bd1f4` through
  `40d4414f45b89b7cb7f5efdc06ca622b674d772a` on 2026-10-30. The relevant
  storage/gather implementations remain at integration baseline `88b48a1f1`;
  this entry does not claim a fresh runtime reproduction on that baseline.
- Owner: native storage / Torch Tensor compatibility maintainers.
- Symptom: `tensor.untyped_storage().data_ptr()` reports Python object identity,
  not allocation identity. General gap-containing `as_strided` requests use a
  gather/copy. Historical probes found some `split` results with correct
  physical pointers whose Python indexed assignment did not update their
  source. This is a bounded historical observation, not a claim that every
  split layout fails or that split reads are wrong.
- Cause: `_Storage.data_ptr` returns `id(self._var)`; unsupported as-strided
  layouts use the gather branch. The historical split issue concerned raw
  getitem paths without the same VarHolder write provenance as registered view
  APIs. Direct foreign-kernel writes through a physical pointer are a distinct
  path and do not prove Python assignment ownership.
- Workaround: check materialized Tensor.data_ptr and byte offsets for physical
  sharing; construct supported views with slice/reshape/permute and perform
  Python writes through their registered parent view. Do not infer general
  alias support from correct read values.
- Evidence: `compat/torch/installers/tensor/method_api.py` owns `_Storage` and
  `_as_strided_as_view`. The historical pointer probe used
  `x=torch.arange(12,dtype=torch.float32); x.data_ptr(); y=x.reshape(3,4).t()`
  and compared Tensor.data_ptr with Storage.data_ptr. Separate CPU probes
  exposed gather/split writeback boundaries. See the
  [integration report](https://github.com/Jittor/jittor/blob/866d5d3e721e9f1d101742b8fe4f97106651491e/docs/results/2026-11-01-vllm-runtime-integration.md)
  for the source/validation distinction. Raw historical artifacts remain under
  `$JITTOR_LAB_ROOT/_state/vllm-product/20261030/`, unversioned.
- Exit condition: allocation identity and valid strided/split write ownership
  match independent Torch on CPU and each claimed accelerator, including
  lifetime/offset tests; unsupported layouts are rejected or clearly declared
  instead of being silently represented as fully aliasing views.

## KI-MEM-004: scalar boolean assignment retains one live operation

- Severity: Low (bounded native CPU lifetime observation; growth is unmeasured).
- Status: Open; reproduced on 2026-11-02 at
  `9391a815dd6cc67b2d2fbe61fd9b4a1fad1d345a` in native CPU mode.
- Owner: Jittor core memory / indexing maintainers.
- Symptom: in a fresh native process, `live_ops` increases from 0 to 1 after
  `value = jt.array(1.0); value[jt.array(True)] = jt.array(2.0);`
  `assert value.item() == 2; del value`. It remains 1 after `gc.collect()`,
  `jt.sync_all()`, and `jt.gc()`. This records one scalar-mask operation only;
  repeated-call growth and accelerator behavior have not been established.
- Cause: the retaining owner is not yet identified. The scalar-mask test leaves
  the operation alive before the later argmax test starts; argmax itself leaves
  the count unchanged. This is not evidence of an argmax/backward leak.
- Workaround: lifetime tests for unrelated operators compare before/after live
  operation counts, or run in a fresh process. This avoids test-order coupling;
  it does not fix the scalar-mask retention or establish general leak freedom.
- Evidence: run
  `python -m pytest -q tests/core/test_setitem.py::TestSetitem::test_scalar_mask_with_var_value_compiles tests/ops/test_misc_issue.py::TestMiscIssue::test_argmax_memleak`
  on the recorded baseline: the scalar-mask test passes, then the historical
  argmax global-zero assertion fails with 1 live operation. Argmax alone passes.
  Both test files are unchanged from `88b48a1f1` on that baseline; runtime
  reproduction on `88b48a1f1` itself was not performed. The argmax regression
  now checks its own count delta and retains its original sync/delete lifecycle.
  Raw evidence: `$JITTOR_LAB_ROOT/_state/vllm-core-merge/20261101/logs/`
  `smoke-argmax-minimal-sequence.log` and `smoke-scalar-mask-liveness.log`.
- Exit condition: identify and justify or release the retained owner, add a
  focused lifetime regression, and establish whether repeated scalar-mask
  assignments accumulate live operations on each claimed backend.

## KI-COMPAT-025: CPU bf16 autocast narrows an explicitly float32 loss

- Severity: Medium (numerical fidelity in bounded autocast training).
- Status: Open; reproduced 2026-11-02 on remote baseline
  `88b48a1f18b2964f05090f65561e07ab64343715` and integration source
  `ee116312ed3448a51c77ab1c9e2abeb45677f7f7`, with separate native caches
  and asserted source paths. This is not introduced by the vLLM prerequisites.
- Owner: Torch compatibility / AMP maintainers.
- Symptom: `TestAutocastTraining::test_bfloat16_autocast` ends at
  `0.158203125` against float32's `0.12238326668739319`, a relative difference of
  `0.2926859143587205` beyond the unchanged 25% gate, on both commits. In CPU bf16 autocast,
  `x = torch.ones(2).bfloat16(); y = torch.ones(2)` followed by
  `((x.float() - y) ** 2).mean()` returns bf16: the explicit cast is float32,
  but subtraction, square, and mean are narrowed. Independent real Torch
  2.13 keeps all four stages float32 on the same CPU input.
- Cause: the region-wide native AMP register lowers operations even after an
  explicit float32 cast, unlike the per-operator autocast policy. The earlier
  LayerNorm fix in `91ff825fc` remains active: all 80 training calls return
  float32 from bf16 inputs with Torch autocast active. Changing only loss
  construction in an untracked diagnostic to disable native AMP gives final
  loss `0.12916548550128937` (5.54% from float32), with finite parameters and
  gradients. This narrows the cause; it does not establish complete autocast
  fidelity. The model uses ReLU, so this is not the SiLU issue KI-COMPAT-023.
- Workaround: use float32 training for this bounded CPU task. Explicit
  `.float()` inside the active region is insufficient to preserve the loss
  arithmetic's dtype. No tolerance relaxation or runtime workaround is merged.
- Evidence: the named case in
  `compat/tests/torch/test_torch_half_precision_numerics.py` fails alone on
  both commits; the candidate file has 10 passed, 1 skipped, and this failure.
  Raw logs and source-identity/dtype probes are under
  `$JITTOR_LAB_ROOT/_state/vllm-core-merge/20261101/`, indexed with hashes in
  `bf16-diagnosis.json`. Historical KI-COMPAT-020 concerned the LayerNorm
  output policy; its fix remains active. This distinct loss-arithmetic issue
  does not reopen that mechanism. CUDA was not tested here.
- Exit condition: a focused float32-loss dtype regression agrees with real
  Torch, this unchanged training gate passes, and the complete half-precision
  file plus related AMP/GradScaler tests pass on CPU and each claimed backend.


## KI-COMPAT-026: lazy module imports invalidate initial namespace contracts

- Severity: Medium (activation ownership and suite isolation).
- Status: Open; reproduced on 2026-11-02 at `9391a815`.
- Owner: Torch compatibility namespace / test maintainers.
- Symptom: importing `torch.distributed.run` after activation adds a lazy module
  that makes a later `compat.install(torch)` reject the completed graph.
  Importing `jittor.compat.fsdp2.activation_checkpoint` also adds a normal Python
  package attribute, so initial-public-surface tests fail in the warmed process.
- Cause: initial namespace expectations and lazy imports are not reconciled.
  A fresh child cannot be compared to a warmed parent's exact module-name set.
- Workaround: run initial activation, alias and FSDP surface checks in clean
  processes; do not disable strict ownership rejection or delete arbitrary
  runtime modules to hide a changed graph.
- Evidence: the four complete activation/alias/compat-structure/FSDP-structure
  files pass 45 cases in a clean source environment. Importing only the two
  modules above after collection reproduces six failures and 35 passes.
  Source/test files involved are unchanged from `88b48a1`; an old-baseline
  runtime reproduction was not performed for this issue. Logs are unversioned
  under `$JITTOR_LAB_ROOT/_state/vllm-core-merge/20261101/logs/`:
  `shim-clean-env-files.log` and `shim-order-repro.log`.
- Exit condition: intentional lazy modules have explicit ownership and
  initial-surface tests isolate their premise; the minimal import-order
  sequence passes without weakening foreign-module rejection.

## KI-COMPAT-027: compiled Accelerate helpers require full-graph capture

- Severity: Medium (compiled training helpers).
- Status: Open; reproduced on 2026-11-03 with core/compat `0a23c241`.
- Owner: Torch compatibility compiler maintainers.
- Symptom: Accelerate 1.10.1's original
  `tests/test_utils.py::UtilsTester::test_convert_to_fp32` fails when it invokes
  `torch.compile(fullgraph=True)`; the independent PyTorch reference passes.
- Cause: `compat/torch/installers/compiler.py` explicitly rejects full-graph
  capture, which is distinct from the native per-operator JIT execution path.
- Workaround: use the eager `Accelerator.prepare` / `backward` training path.
  Its selected three-step CPU configuration passes independent parameter,
  loss and gradient comparisons in the integration repository.
- Evidence: [Accelerate validation records](https://github.com/JittorEcosystem/jittor-accelerate/blob/main/docs/validation.md),
  upstream commit `623cc0ba588d192c77758939a6e340adf07edd4f`.
  The separate nested-tensor failure in `test_pad_across_processes` is tracked
  by KI-COMPAT-003.
- Exit condition: implement full-graph capture semantics and pass the original
  compiled-helper test and graph-break rejection regressions on the declared
  device, while preserving the eager training behavior.

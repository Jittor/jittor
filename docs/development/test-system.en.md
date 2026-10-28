# Test system

- Status: accepted
- Last reviewed: 2026-09-11
- Baseline: `2d716db31`
- Owner: test infrastructure maintainers
- Recheck when: the collection roots, process-mode ownership, markers, the OpInfo contract or
  the backend gates change

Jittor uses pytest as the repository test runner while keeping compatible `unittest.TestCase`
tests. The suite lives in the top-level `tests/` and **is not part of the installed package**.
[`pyproject.toml`](https://github.com/Jittor/jittor/blob/master/pyproject.toml)
is the authoritative source for collection and marker configuration;
[`noxfile.py`](https://github.com/Jittor/jittor/blob/master/noxfile.py)
is the reproducible command surface.

## Goals

- Compare forward behaviour against **independent references**;
- verify analytic gradients numerically, and do so across devices;
- keep structure, CPU and hardware requirements explicit;
- keep valuable historical boundary cases while replacing cross-test imports and ad-hoc
  discovery with shared helpers and registries;
- make unsupported behaviour visible through **specific skips or strict expected failures**;
- keep the collection phase free of compilation, downloads and hardware side effects.

## Directory layout

The repository has three test roots sharing one pytest policy:

```text
tests/                    # 原生测试根
├── conftest.py           # 薄适配层：注册 _helpers.pytest_policy
├── _fixtures/            # 版本化的测试数据
├── _helpers/             # 显式的共享测试工具（进程模式、门禁范围、子进程、能力探测等）
├── opinfo/               # 算子元数据、样本、参考、skip 策略
├── ops/                  # 通用 OpInfo 前向与梯度批测
├── autograd/ core/ type/ mem/ runtime/ bindings/   # 原生图、dtype、内存、运行时与绑定
├── codegen/              # 生成代码、pass 与 tuner
├── build/                # Python 构建与编译工具
├── nn/ optim/ init/ linalg/ distributions/ serialization/ data/ contrib/ models/
├── backends/             # 镜像 backends/{cpu,cuda,acl,rocm,corex,comm}，外加 parity/ 跨设备一致性
├── distributed/          # 原生启动模式下的分布式行为
├── debug/ utils/ tools/  # 诊断、通用工具与仓库工具
├── integration/          # 示例、notebook 与跨组件工作流
└── structure/            # 仓库、打包与静态契约
compat/tests/             # 可选 jittor-torch 项目的测试根
├── torch/                # Torch 模式下真跑张量的行为测试
├── structure/            # 名字归属与命名空间发布的结构契约
├── fsdp2/                # FSDP2
└── triton/               # Triton 桥
adapters/tests/           # 第三方 adapter（含 vllm/）
```

All three roots load `tests/_helpers/pytest_policy.py` through a thin conftest. The helper is a
monorepo development dependency: **neither the Jittor wheel nor the compat wheel installs any
test package**, so running `compat/tests` requires a full checkout that includes
`tests/_helpers`; a standalone adapter sdist can still run its own host tests when that
directory is absent.

Structural contracts are placed under `tests/structure/<源码域>` according to their
source domain, while cross-domain repository governance stays at the `tests/structure` root;
structural tests owned by compat live in `compat/tests/structure`. The ACL dtype recorder
(`tests/structure/backends/acl/test_acl_dtype_preservation.py`) is a CPU mock contract with an
explicit native-process exception, and **does not inherit** the fallback fixture and `npu` marker
of real ACL hardware tests. `tests/core/test_setitem.py` is deliberately kept at that path.

**Process mode belongs to each file, not to the directory it sits in.** The authoritative
selection is in `tests/_helpers/process_modes.py`, and the full runner derives the native and
Torch selection sets from it through `tests/_helpers/gate_scope.py`; there is no argv-activated
mechanism. Hardware markers are derived from the path: a path containing `acl` gets `npu`, one
containing `distributed`, `comm` or `fsdp2` gets `mpi`, and anything under `structure` gets
`structure`.

`nox -s full` or `tools/run_test_suite.py` runs the two-process full suite; `nox -s structure`
includes the structural contracts of every structural root plus the adapters, and runs the
native CPU recorder separately. To select cases directly, use physical paths.

**Determinism of generated inputs survives layout moves.** `tests/_helpers/layout_seed_paths.json`
records the old paths from before the move; `layout_seed.seed_nodeid` swaps the new path back to
the old prefix (keeping the class, method and parametrization suffix) only before the input
generator hashes the key, so a moved case generates the same inputs as before the move. New
files and new methods use their new canonical paths; a new class reusing an old method name, and
a new explicit method whose name happens to match an old generator prefix, both count as new
cases. pytest nodeids, logs, selectors, failure reports and coverage are **never** rewritten by
this mapping.

Test modules may import `tests/_helpers` and `tests/opinfo` through the pytest Python path
configured for the suite. **Another test module must not be imported as an implicit helper API.**
A shared utility containing non-trivial comparison or device logic needs its own targeted test.

## Process-mode isolation

Installing Torch compatibility is **process-level** and changes public methods, dtype promotion,
reduction defaults and lazy execution. Therefore **native tests and Torch-facing tests run in
different pytest processes**.

`tests/_helpers/process_modes.py` owns the `TORCH_MODE_PATHS` list; `tests/conftest.py` ignores
those paths during broad native collection and activates Torch mode when one of them is selected
explicitly. The shared OpInfo and device-parity suites use Torch-facing signatures and therefore
belong to the Torch process.

`tools/run_test_suite.py` is **the entry point for the complete suite**: it runs the native and
Torch sessions with their own separate state, caches and mode variables, then aggregates the
results. Running `python -m pytest tests` directly **is deliberately the native session only**
and must not be reported as full-suite coverage.

A test that asserts `result_type`, Torch cast aliasing, typed tensor names or Torch-specific
defaults belongs to the Torch session -- even if the same file also carries lower-level
contracts and lives under `tests/core/`.

An in-process independent PyTorch oracle requires `REAL_TORCH_SITE` to point at a site-packages
directory containing PyTorch's binary `_C` extension, which pytest preloads before Jittor.
**Without this explicit oracle, tests skip the optional PyTorch comparison**, even when a
deployed Jittor-backed `torch` stub can be found -- **that stub must never be treated as an
independent reference**.

## The three layers of operator evidence

### 1. Independent forward reference

Each `OpInfo` describes the callable, sample constructors, dtypes, autograd support, tolerances
and targeted skip/xfail policy. `tests/ops/test_ops.py` expands this database on the requested
device and compares against an independent NumPy or mathematical reference.

**The reference implementation must not call the Jittor operation under test.** Sample
constructors should cover meaningful shapes, axes, broadcasting, optional arguments and
error-prone dtypes, rather than **piling up quantity without semantic diversity**.

### 2. Numerical gradients

Differentiable OpInfo entries run float64 CPU `gradcheck`; operations that support second-order
differentiation also run `gradgradcheck`. Marking `supports_autograd=False` or
`supports_gradgrad=False` is **a contract decision, not a way to turn a failing test green**, and
the reason must be given in the definition or in the issue ledger.

Numerical checks prove the derivative formula on CPU. **They do not prove that an accelerator
correctly executed the same backward kernel.**

### 3. Device parity

[`tests/backends/parity/test_device_parity.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/parity/test_device_parity.py)
runs the same inputs and cotangents on CPU and on the available accelerator, comparing forward
outputs and gradients with both global and elementwise error measures. This layer catches
device-specific compilation failures, missing gradient contributions and silently biased
kernels.

> **CPU is a valid parity reference only after it has been independently pinned down by the
> forward layer and the numerical gradient layer.** Backend environment failures must be
> reported separately from framework defects.

## Coverage: the question beyond the three layers

The three layers above answer **"is this operator correct"**. They do not answer **"which
operators and which dtypes did we actually ask about"** -- and defects emerge precisely in the
places never asked about.

Four proven shapes:

**The operator surface is far larger than the database.** The public callable surface has 1294
names, and OpInfo covers 179 operators. The difference is not "not done yet"; it is where the
defects live.

**"Covered" is not binary.** `bitwise_not` returns `True` for bool regardless of the input, and it
has an OpInfo entry all along. The entry declares `dtypes=_INT` (`integral_types()` does not
include bool), and the sample generator says `dt = _int_dtype(dtype)`, forcing the passed dtype
to an integer type -- so no test **could** feed it bool. The operator reads as covered. Note also
that `OpInfo.dtypes` defaults to `floating_types()`: an entry that forgets to declare its dtypes
silently tests floats only.

**Counting coverage by text is self-deception.** Every one of the 1294 names appears somewhere
under `tests/` (imports, comments, unrelated identifiers), and that measure gives about 96%.

**The measurement covered only one of the two surfaces.** All 1294 names above are the native
surface. The Torch compatibility surface is a different object graph -- `torch` is a namespace
view of the compatibility owner, `torch.Tensor` is not `jt.Var`, `torch.nn` is composed -- and
the native inventory covers not one name of it. It has 1295 public callable names of its own, as
large as the native surface, and before this system was built it was entirely outside the
measurement.

The corresponding four tools (one inventory and baseline per process mode):

| Tool | Answers |
| --- | --- |
| [`tests/structure/public_api_manifest.json`](https://github.com/Jittor/jittor/blob/master/tests/structure/public_api_manifest.json) and its gate | whether the native 1294 names **still exist** |
| [`tests/structure/torch_api_manifest.json`](https://github.com/Jittor/jittor/blob/master/tests/structure/torch_api_manifest.json) and its gate | whether the Torch surface's 1295 names **still exist** |
| [`tests/_helpers/api_coverage.py`](https://github.com/Jittor/jittor/blob/master/tests/_helpers/api_coverage.py) (`JITTOR_API_COVERAGE=1`) | which entry points a run **actually invoked** |
| [`tools/api_coverage_ratchet.py`](https://github.com/Jittor/jittor/blob/master/tools/api_coverage_ratchet.py) and the two baselines | whether the set of uncalled entry points **has grown** |
| [`tools/probes/opinfo_dtype_gaps.py`](https://github.com/Jittor/jittor/blob/master/tools/probes/opinfo_dtype_gaps.py) | which (operator, dtype) pairs **work but are not declared** |

Coverage measurement is off by default: wrapping every public entry point is a diagnostic, not
the normal state. It is validated with an on/off comparison showing it does not change the
system under test -- native `tests/ops/test_where_op.py` gives 18 passed on both sides, and the
Torch session, after excluding the 14 files below, has exactly the same set of failing nodeids on
both sides.

**This comparison must be repeated once on every surface; passing on one surface does not
extrapolate.** The same line "wrap every callable" has always been harmless on the native
surface, because the native surface has no callable modules; on the Torch surface it replaces
`torch.random`, the frontend's namespace-ownership check then decides the graph has been
modified, and collection fails outright. A diagnostic validated on one object graph is validated
on that object graph only.

The Torch surface has one further limit this wrapping cannot get past, and it is recorded rather
than hidden: the frontend has an explicit contract that "what is published is the same object the
owner module holds", and the fidelity registry is keyed by the objects themselves, so a wrapper
that replaces objects is visible to those cases -- `JITTOR_API_COVERAGE=1` turns 86 of them red.
Two ways of dodging were tried and both were worse (rebinding the defining module: 101 entries;
rebinding every alias: 97 entries, with the failures merely moving to the registry), which shows
this is a property of "a recorder that replaces implementations", not a bug to fix. The list, the
measurements and the way out are recorded in `IDENTITY_CONTRACT_FILES` in
`tests/_helpers/api_coverage.py`: the Torch coverage run excludes those 14 files, the exclusions
are written into the baseline file itself, and the real solution is a recorder that replaces no
objects.

A run measures only the one surface owned by its process mode: `JITTOR_TORCH_SHIM` determines the
process mode, and therefore which inventory to read and which baseline to write; updating one
surface's baseline leaves the other untouched. Wrapping is installed **after collection
finishes** rather than at session start: obtaining a surface means importing it, and in Torch mode
that import is itself the frontend activation some tests measure -- do it one step earlier and
`enable()` is already a no-op by the time the tests execute, so the diagnostic has changed what it
observes.

Entry points that cannot be wrapped are accounted for separately rather than discarded -- one
missing from the denominator flatters the result. A class is an entry point, but wrapping it
replaces the type and breaks `isinstance`; a module is an owner rather than an entry point, and
the Torch surface has callable modules (`torch.random` is a module subclass with `__call__`), so
wrapping it would replace a published namespace binding. Both go into unwrappable rather than
disappearing from the denominator.

The ratchet comparison needs a complete coverage run and is a nightly-level job rather than a PR
gate; what runs on the PR gate is only the shape check of the baselines themselves and the
negative validation of the rules. The dtype probe trial-compiles each (operator, dtype) pair and
is likewise a nightly-level diagnostic.

## Cross-cutting contracts

Some semantics belong to no single operator, and therefore to none of the layers above:
residency and migration, aliasing and in-place writes, and the timing of laziness versus
materialization. Without an owner they have no tests.

[`tests/core/test_var_residency_contract.py`](https://github.com/Jittor/jittor/blob/master/tests/core/test_var_residency_contract.py)
is the first of this kind: an uncomputed Var has no residency; reading the data of a device Var
really migrates the storage back to the host; `device_id` keeps the source device across `cpu()`
while `device`/`location()` follow the data. It runs entirely on CPU, with CUDA cases skipped
individually -- **removing the accelerator only narrows it, it does not empty it**.

This has a history: the device-method contract of Var/Module previously had only four cases, all
requiring two CUDA cards, so they were skipped wholesale on single-card and CPU machines and
looked exactly like four passes in the report. "Not enough devices" is now counted separately as
`insufficient-devices` and named in the summary, distinct from "no accelerator" -- the former is
coverage lost despite the hardware being present, not an environment fact.

## Runtime introspection: what tests should read

Tests and the shared pytest policy observe capabilities, policy and counters through
[`jt.introspection`](../notes/runtime-introspection.en.md), rather than reading native flags or
compiler attributes directly. What the migration preserves is **the question the test was really
asking**:

| Old reading | Supported replacement | Boundary |
| --- | --- | --- |
| `jt.compiler.has_cuda`, skipping accelerator cases based on `jt.flags.use_cuda` | `jt.introspection.capabilities.backend("cuda")` | check failed first, then skip on disabled/absent |
| `core.get_device_count()` called by backend name | `jt.introspection.capabilities.devices(name)` | distinguish FAILED/UNPROBED from 0 |
| library checks such as `compile_extern.cudnn_ops is not None` | `jt.introspection.capabilities.library("cudnn")` | UNPROBED is not the same as unavailable |
| current `jt.flags.use_cuda`, `amp_reg`, `no_grad`, precision reads | `jt.introspection.policy.runtime.<name>` | preserves the semantics of the effective policy |
| `jt.flags.cache_path` and compiler configuration reads | `jt.introspection.policy.startup.<name>` | immutable startup configuration |
| `jt.flags.exec_called` | `jt.introspection.counters.exec_calls` | does not implicitly execute the graph |
| `stat_allocator_total_alloc_byte` and the corresponding call/free counts | `jt.introspection.counters.allocator.*` | instrumentation mode and reset semantics matter |
| `number_of_lived_vars/ops`, `number_of_hold_vars`, `liveness_info()` | `jt.introspection.counters.live_vars/live_ops/held_vars` | no implicit GC |
| assigning to a flag | `jt.runtime.scope(...)` when reversible | writing is not introspection |

**Exceptions are registered explicitly.** Tests that are themselves validating native setters,
registration internals or binding generation keep direct access to the implementation they test;
those files, with their reasons, are registered in `tests/_helpers/introspection_exceptions.json`.
No gate currently reads that registry, so update it when adding an exception -- do not let it
become another one-way list.

## Test categories and markers

Markers are registered in
[`pyproject.toml`](https://github.com/Jittor/jittor/blob/master/pyproject.toml):

| Marker | Contract |
| --- | --- |
| `structure` | runs no device code; layout, packaging and static checks |
| `cpu` | maintained CPU behaviour |
| `cuda` | requires an NVIDIA CUDA environment |
| `rocm` | requires an AMD ROCm environment |
| `npu` | requires an Ascend CANN environment |
| `mpi` | requires an MPI launcher or multiple processes |
| `slow` | excluded from the fast PR gate |
| `network` | requires external network access |
| `manual` | selected explicitly; **never enters an automatic default run** |

**Use the narrowest applicable marker.** Hardware tests must exercise real computation and
**must not pass silently on CPU**. `network` and `manual` tests must state their external
requirements in the module docstring.

## Optional fixtures and manual scripts

Some coverage needs whatever happens to be on the machine -- a dataset, a checkpoint, an
independent PyTorch. These entry points are **always explicit**: if unset they skip for the
specific precondition, and **never quietly switch to a weaker check**.

| Variable | Who reads it | When unset |
| --- | --- | --- |
| `JITTOR_TEST_IMAGENET_TRAIN` | An ImageNet-layout training directory (one subdirectory per class). `tests/data/test_dataset.py` uses it to verify that a multi-worker `Dataset` matches `num_workers=0` batch by batch; `tests/data/test_image_folder.py` additionally uses the optional `torch`/`torchvision` to align Jittor's `ImageFolder` with `torchvision` | the whole test class skips, with the reason naming the variable |
| `REAL_TORCH_SITE` | the in-process independent PyTorch oracle (see "Process-mode isolation" above) | the optional PyTorch comparison is skipped; **a deployed shim does not stand in for it** |
| `QWEN3_0_6B` | the Qwen3-0.6B checkpoint directory for `tests/backends/acl/manual/xcheck/qwen_logits.py`. That script dumps the next-token logits once under real torch and once under the Jittor shim, and `compare.py` then aligns them | the script exits with a `KeyError` -- it is `manual` and not scheduled by any gate |

The scripts under `tests/backends/acl/manual/` (GPT-2 forward/gradient alignment, Qwen3 logits
alignment) need a real Ascend device and are **run manually**; usage and verified results are in
that directory's README, and device-side configuration is in the
[Ascend 910B guide](../guides/ascend-910b.en.md).

## Maintainer probes

[`tools/probes/`](https://github.com/Jittor/jittor/blob/master/tools/README.md) contains **scans**,
not tests: they do not assert a property and turn red, but put two things that should be equal
side by side and print every discrepancy for a human to judge. Their output is the evidence cited
by issue-ledger entries, so they **do not enter the gates** and are run by maintainers on demand.

| Probe | What is compared with what |
| --- | --- |
| `adversarial_device_sweep.py` | the whole OpInfo database on NaN, ±inf, ±0, ties and subnormals, CPU against CUDA |
| `equivalent_form_sweep.py` | two spellings of the same quantity |
| `fusion_consistency_sweep.py` | the same expression evaluated with fusion on and with `no_fuse` |
| `roundtrip_consistency_sweep.py` | the save-reload-recompute round trip, and `no_grad` against an ordinary forward |
| `semantic_divergence_probe.py` | the dtypes and boundary semantics on which two reasonable implementations really diverge |
| `side_effect_probe.py` | what an operation asked only to "read" changed (`a.numpy()` moved `a` itself) |
| `opinfo_dtype_gaps.py` | which (operator, dtype) pairs work but are not declared (see the coverage section above) |
| `error_message_probe.py` | the **verbatim text** a user currently gets for real mistakes; the diff between two runs is the acceptance record for that change |

These scans **need no external reference values** -- which is exactly what lets them point at
expressions with no reference implementation. The kind that needs an external reference
(comparing analytic gradients against difference quotients) was tried and abandoned: every entry
it produced was a wrong reference.

## Skips and known failures

- **skip** means a precondition is unavailable, or a contract is deliberately unsupported; its
  reason must name **the exact precondition or limitation**.
- **expected failure** means a reproduced framework defect. pytest uses strict xfail, so a fix
  produces an XPASS and **forces the ledger to be cleaned up**.
- **Do not** catch arbitrary exceptions outside the test body and turn them into skips. Probe the
  optional environment **narrowly** before execution.
- Every persistent expected failure must be listed in the
  [issue ledger](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md)
  with an owner and exit criteria.

## Gate honesty: three ways of "looking like it passed"

A run can prove nothing at all while never failing. All three shapes have really happened, and
each guard exists because of one of them:

**Entries that only ever skip look exactly like passes.** That is how the backward formulas of
227 operators stayed unverified across three all-green runs. `JITTOR_TEST_REQUIRE_EXECUTION=1`
requires an entry either to execute something or to be listed with a reason in
`gate_scope.EXECUTES_NOTHING`.

**"Not enough hardware" is not "no hardware".** When a test needing two cards skips on a
single-card machine, the reason also mentions CUDA, so it lands in the `accelerator` bucket and
reads like an environment fact -- even though the machine has an accelerator and the coverage was
simply thrown away. **All** of the Var/Module device-method contract sat behind exactly this
skip: four cases that never execute on non-multi-card machines and are indistinguishable from
four passes in the report. `insufficient-devices` is now counted separately and named in the
summary.

**A session that dies mid-way produces a log that says "finished".** A native crash inside a
test does not fail that test -- it takes the whole interpreter with it: pytest prints the nodeid
it is about to execute and the process is gone, with no result line, no traceback, no summary,
and the tests after it never ran. Nowhere in the log does it say failed, so someone scanning for
failures finds nothing, and a script grepping for failure agrees. This is worse than a skip -- a
skip at least occupies a line in the summary.

The maintained CPU gate was itself running only part of its work this way: in a CPU-only build
(`nvcc_path=""`, which `tools/run_test_suite.py` sets), a `scatter_add` segfault made the Torch
session vanish at 48%, while the same session ran to completion on a machine with CUDA -- which is
why it went unnoticed.

[`tests/_helpers/session_completion.py`](https://github.com/Jittor/jittor/blob/master/tests/_helpers/session_completion.py)
makes a session testify that it completed: `pytest_sessionfinish` runs after the last test,
whether passed, failed or interrupted by pytest, and **does not run only when the process dies**
-- precisely the distinction wanted. The sentinel also carries collected/executed counts, so
"completed but quietly collected less" is visible too.
[`tools/check_session_completed.py`](https://github.com/Jittor/jittor/blob/master/tools/check_session_completed.py)
is the consumer, with evidence priority sentinel > log marker > pytest summary line. **A missing
sentinel is not treated as "probably fine"**: this failure mode already looks like nothing
happened.

One shared discipline: **a new gate must be shown to go red**. After writing it, manufacture the
defect it is meant to prevent, confirm it goes red, then restore. A guard that has never been
negatively validated looks the same in a report as a guard that does not exist -- this round alone
fixed two existing guards that had never checked anything (one scanned a directory that a layout
move had deleted, the other read a stale path and never touched runtime behaviour).

## Four disciplines taught by accidents

None of the four below was derived from principle; each was added after a "green report that
proved nothing". Each notes what taught it, and the full record is in the issue ledger
[`agent/manuals/known-issues.md`](https://github.com/Jittor/jittor/blob/master/agent/manuals/known-issues.md).

### 1. Subprocesses always go through `_helpers.child_process`, and must specify the device and prove where they landed

`tests/conftest.py` puts **the checkout under test** on its own `sys.path`, but **does not export
`PYTHONPATH`**. So a bare `subprocess.run([sys.executable, ...])` inside a test hands the child
whichever `jittor` the interpreter environment resolves -- which in a development checkout is
often an editable install pointing at **another tree**.

**Quiet failures are the most expensive**: a child that imported another tree still ran, the test
still went green, and it proved nothing about the code under test. The loud version showed itself
when a rename happened: `[0.08]` changed core's `set_lock_path` to `set_lock_fd`; the child loaded
the core this branch had just built but imported `compiler.py` (with the old name) from the main
tree, and died with an `AttributeError` that matched neither tree.

The rule is mechanical rather than heuristic:
[`tests/structure/test_child_process_contract.py`](https://github.com/Jittor/jittor/blob/master/tests/structure/test_child_process_contract.py)
uses an AST scan over every process launch under `tests/`, and **anything that names an
interpreter directly makes it red**. `tools/run_test_suite.py` and
`tools/gate_conclusion_diff.py` are in scope too, because they launch the same kind of
subprocess. A scenario that deliberately **removes** `PYTHONPATH` ("where would `import jittor`
resolve for someone who just installed it") also goes through the helper, written as
`child_env(..., repo_paths=False)` -- an exception that reads like an oversight cannot be told
apart from an oversight.

**Passing the right tree is not enough; the device must be passed too, and proven.** The
subprocess in `tools/probes/roundtrip_consistency_sweep.py` never set `use_cuda` and relied on
the default -- and **`use_cuda` defaults to 0 in a fresh process, even on a machine with eight
cards**. So the `--device cuda` arm compared "a CUDA parent" with "a CPU child", treating the
difference between two devices as a serialization defect. It kept reporting OK because under
`-Ofast` this model happened to be bit-identical on both devices; switching to `-O3` moved the
CPU result by `1.49e-08`, the luck ran out, and the defect appeared.

The device is now passed as an argument to the child, and the child asserts with
`x.location()` that it really landed on the end it was asked for:

```python
where = x.location()
expected = "device" if device == "cuda" else "cpu"
assert where == expected, "child asked for %s, tensor is on %s" % (device, where)
```

**Passing the flag is easy to get right and easy to have silently ignored** -- fallback, a
missing driver, a flag read too late -- and once it is ignored, the check turns back into
comparing two devices while looking like it compares two processes.

(What taught it: the child-process contract gate, long red with four offenders listed, commit
`51cdefae2`; the roundtrip sweep's device defect, commit `fef56a211`.)

### 2. Cases that can crash run in their own subprocess

A native crash inside a test **does not fail that test** -- it takes the whole interpreter with
it: pytest prints the nodeid it is about to execute, the process is gone, and there is no result
line, no traceback and no summary, **and the tests after it never ran**. Nowhere in the log does
it say failed.

So a case **known to die by signal** must run in its own process and must declare that with
`run_child_script(..., crash_isolated=True)` -- without it, a signal death reaches pytest as a
bare negative return code. Two ready-made examples:

- [`tests/backends/cuda/test_auto_flush_graph_split.py`](https://github.com/Jittor/jittor/blob/master/tests/backends/cuda/test_auto_flush_graph_split.py):
  every value of `auto_flush_ops` runs in its own process, because the failure mode is a
  segfault. It asserts that the loss is bit-identical **and** compares gradients -- a fix that
  "stopped the crash but read the wrong bytes in backward" would pass a check that only asks
  whether it runs, but not this one.
- [`tests/ops/test_index_bounds.py`](https://github.com/Jittor/jittor/blob/master/tests/ops/test_index_bounds.py):
  a device-side trap takes the CUDA context with it, so it cannot share a process with anything
  after it.

(What taught it: `KI-EXEC-001`, five bottleneck blocks segfaulting deterministically at
`auto_flush_ops`'s shipped default of 128; and the maintained CPU gate -- in a CPU-only build a
`scatter_add` segfault made the Torch session vanish at **48%**, while the same session ran to
completion on a machine with CUDA, which is why it went unnoticed.)

### 3. Read the "files this session proved nothing about" report

At the end of every run, `tests/_helpers/pytest_policy.py` prints a block:

```text
========== files this session proved nothing about ==========
tests/.../test_x.py  collected 0 tests
tests/.../test_y.py  12 skipped, 0 executed
Reported only. Set JITTOR_TEST_REQUIRE_EXECUTION=1 (the gates do) to make an
unexplained entry fail the run.
```

Two kinds of entry, with different meanings:

- **`collected 0 tests`**: this file did not even collect any cases. Usually a collection-time
  exception or the whole file being filtered out.
- **`N skipped, 0 executed`**: collected, but not one case really executed.

An entry may carry one of two annotations: `-- expected here: <理由>` means it is listed in the
`EXECUTES_NOTHING` exemption table in `tests/_helpers/gate_scope.py`; `-- explained: <skip 理由>` means the skip reason was recognized as an environment fact (no accelerator, missing
optional dependency). **An entry with no annotation is what this report really wants you to look
at**, and with `JITTOR_TEST_REQUIRE_EXECUTION=1` on, it turns the whole run red.

Why this gets its own section: the mechanism **existed and worked**; what failed was "someone
reading it". `tests/structure/backends/acl/test_acl_python_registration.py` collected 40 cases
and executed **0** for two whole days; the report printed that line faithfully every round, and
one more red line in a red suite is invisible. So -- **when the suite is red, read this block
before the failure list**: failing tests are at least still talking.

(What taught it: `KI-TEST-004`. That same day, the same question -- "if the thing this gate
checks were broken, would the gate go red?" -- found two more: the profiler's empty report in
`KI-EXEC-002`, and the roundtrip sweep's device defect.)

### 4. When comparing numerics across a flag that changes device residency, turn off cuDNN autotuning first

cuDNN picks convolution algorithms from **measured candidates**, cached by shape. What is in
device memory at measurement time decides which algorithm wins, and scheduling flags such as
`auto_flush_ops` change exactly that residency. So **comparing gradients across such a flag
compares two things**: the change under test, and the convolution algorithm that got swapped
(measured `3.8e-4` relative difference, deterministic, with the forward loss identical to the
last digit).

So the first line of such a comparison is:

```python
import jittor as jt
jt.cudnn.set_benchmark(0)     # 强制走确定性的启发式
```

With it off, the cross-setting gradient residual of `KI-EXEC-001` is `2e-6` -- recombination from
different batch boundaries.

(What taught it: the first fix for `KI-EXEC-001` **was wrongly rejected**. The reason given was
that it "turned the crash into 40–60% gradient errors on individual elements", and both halves
were the reviewer's mistakes: the comparison ran with autotuning **on**, and autotuning depends on
residency, which is precisely what the flag under test changes; and "40–60%" was the maximum
absolute difference divided by the gradient's **RMS** rather than by the magnitude of the element
it belongs to. Redone with tuning off, the residual is `2e-6`. The lesson is not "measure a few
more times", but **a comparison across a flag that changes residency must first pin the tuner
down, and a relative error must be divided by the element it is relative to**. See `KI-EXEC-003`
and the [numerics contract](../notes/numerics-contract.en.md).)

## Three tiers: core / smoke / full

A full run takes a dozen-plus minutes at the start, which is too expensive as an answer to "did
what I just changed break", so there are three tiers answering three different questions:

| Tier | Question | Selection | Measured |
| --- | --- | --- | --- |
| `core` | did a single edit break something immediately | a **whitelist** in `tiers.CORE_FILES`, one file per basic surface | 44 s (two-process mode, serial) |
| `smoke` | how long should a PR wait | the whole tree **minus** `tiers.SLOW_FILES` | about 390 s (4 workers) |
| `full` | what state is this tree actually in | the whole tree | a dozen-plus minutes and up |

Only `full` is a statement about this tree; the other two are statements about time -- which is
also why `core` uses a whitelist and `smoke` a blacklist: a file missing from `smoke` is **a hole
in coverage**, so the default must be inclusion; a file missing from `core` is only **found a
little later**, since the other two tiers still go red, so a selection is acceptable.

`core` is deliberately serial: it is small enough that workers buy little, and this way it also
runs in checkouts without pytest-xdist installed. Its budget is checked **arithmetically** by
`tests/structure/test_gate_tiers.py` (summing each file's measured seconds and comparing against
the budget), not by asserting wall-clock time -- the latter fails on a busy machine in a way that
looks exactly like a real regression.

```bash
python tools/run_test_suite.py --tier core    # 改完就跑这个
python tools/run_test_suite.py --tier smoke   # 提 PR 前
python tools/run_test_suite.py                # 全量,按需
```

Adding a file to `core` spends everyone's time on every edit; the entry must state **the measured
seconds** and **which basic surface it guards**, and the structural test checks both.

## Commands

```bash
# 三层,以及仅原生收集、单个模块
python tools/run_test_suite.py --tier core
python tools/run_test_suite.py
python -m pytest --collect-only -q tests
python -m pytest -v tests/ops/test_ops.py

# 选择某个运算或后端 marker
JITTOR_TEST_DEVICES=cpu python -m pytest tests/ops/test_ops.py -k exp
python -m pytest -m structure tests/structure

# 兼容层测试：走符号链接的那条路径，不要直接指 compat/
JITTOR_TORCH_SHIM=1 PYTHONPATH=python \
    python -m pytest python/jittor/compat/tests/torch

# 可复现门禁
python -m nox -s structure
python -m nox -s cpu
python -m nox -s optional
python -m nox -s cuda
python -m nox -s npu
python -m nox -s rocm
python -m nox -s mpi
python -m nox -s nccl
```

nox sessions establish isolated state and caches. **Runs started concurrently by hand must also
use different `JITTOR_HOME` or `cache_name` values; the first build of a new JIT operation or
extension should run serially.**

Why the compatibility command takes a detour: `compat/` carries its own `__init__.py`, so
`pytest compat/tests/torch` directly makes pytest import it as a **top-level** package `compat`,
while `compat/_aliases.py`'s `from .._runtime import ...` requires it to be `jittor.compat`, so
every case fails with
`ImportError: attempted relative import beyond top-level package` -- collection looks fine (1888
collected), and the errors all land in setup, which is easily misread as "cannot collect".
`python/jittor/compat` is a symlink to `compat/`, and entering through it resolves the package
name to `jittor.compat.tests.torch`, so the relative import holds. The gate does not have this
problem because `_run_pytest_once` first runs `pip install -e compat/`, installing `jittor-torch`
as a real package.

## JIT cache: the cost of a cold start, and when it is paid for nothing

A cold-cache run is an order of magnitude slower than a hot one, and the difference is **entirely
the cache**. Measured on an 8-GPU machine with an exclusive `JITTOR_HOME` and `cache_name`, the
same command once cold and once hot:

| | Time | Result |
| --- | --- | --- |
| `tests/nn` cold | **37:03** | 360 passed, 71 skipped, 948 subtests |
| `tests/nn` hot | **1:02** | 360 passed, 71 skipped, 948 subtests |

### The cold-cache duration ranking is not a "heavy files" ranking

`--durations=40` accounts for 78.3% of the cold run's total time, and the top two items alone for
34%. But look at the same cases under a hot cache and the ranking falls apart:

| Case | Cold | Hot |
| --- | --- | --- |
| `test_bmm.py::TestBMM::test_bmm_cuda` | 466.00s | **1.16s** |
| `test_depthwise_conv.py::TestDepthwiseConv::test_data` | 291.08s | 10.46s |
| `test_norm_unification.py::…::test_module_and_functional_agree` | 98.50s | 1.98s |
| `test_pool3d.py::…::test_cuda_backward_matches_cpu_backward` | 59.88s | 1.51s |
| `test_knn.py::TestKnnOp::test_knn` | 54.82s | 1.68s |

**There are no heavy files in `tests/nn`.** The cold-run ranking is a table of "whoever runs
first pays the compilation bill for everyone", and the top of it depends on collection order
rather than on the cases themselves. Pick "tests to optimize" from cold-cache durations and you
pick collection order.

### What warmup cannot warm: 72% of kernels are keyed by the subgraph

A cold run of `tests/nn` produces 604 JIT kernels:

| Category | Count | Share | Can a generic warmup compile it ahead of time |
| --- | --- | --- | --- |
| fused kernels (`__opkey…`) | 436 | 72% | **no** |
| library-provided `code` kernels | 111 | 18% | yes |
| other single-operator kernels | 57 | 9% | yes |

A fused kernel's key is assembled from **the sequence of operators fused into it** -- each
operator's dtype, rank, whether it is strided, its broadcast shape, in fusion order -- rather than
from operator names. So "compile the common operator combinations once, then exit" reaches only
about 28% by measurement, and **the remaining 72% only hits if you produce the same graph**. This
is a negative conclusion: what amortizes the cold cost is not "warming up more operators" but
**making the cache hits that should happen actually happen**.

### What really makes "a hot cache look cold": the locale ended up in the cache directory name

The cache path contains an `archXXXXXXXXXX` segment, a hash of the output text of
`cc -march=native -Q --help=target` -- keying on the instruction set the compiler actually
expanded, which in itself is right. But in that output, every feature's on/off status is
**translated by gettext**:

```text
LC_ALL=C           -m128bit-long-double  [enabled]   -> arch0fe147db07
LC_ALL=zh_CN.UTF-8 -m128bit-long-double  [启用]      -> arch6fe5b27f7d
```

Same machine, same compiler, same CPU, same flags, **two completely unrelated cache trees**:

```text
…/Linux-5.4.241-x7a/arch0fe147db07/da0673f5cfe0/demo/cfgd436a523
…/Linux-5.4.241-x7a/arch6fe5b27f7d/da0673f5cfe0/demo/cfgd436a523
```

This is not theoretical, and this repository holds both sides: `noxfile.py`'s `_session_env` sets
`LC_ALL=C`, and **nowhere else in the repository** does. So the `nox` gates use one tree, while
every hand-run `pytest` (and `tools/run_test_suite.py`, which does not set the locale) uses the
other on non-English machines, **each paying again, on the same machine, for cold compilation the
other already paid for**. `_shared_jittor_cache()` sharing `JITTOR_HOME` was working -- it was
merely undercut by an `LC_ALL=C` 12 lines away in the same function.

Measured cost (empty `JITTOR_HOME`, `JT_PROBE_CACHE=0`, running only `import jittor`):

| | 1st run `LC_ALL=C` | 2nd run `LC_ALL=zh_CN.UTF-8` | 3rd run, same |
| --- | --- | --- | --- |
| Before the fix | 60s (cold) | **59s (fully cold again)** | 3s |
| After the fix | 60s (cold) | **2s (hit)** | 2s |

Before the fix two `arch*` trees were produced; after it, only one. And this is only the core
build of `import jittor`; a suite would then compile all its JIT kernels again in the second
tree.

Both trees are sitting side by side on this machine right now:

```text
arch0fe147db07   1.1G   7801 个 kernel
arch6fe5b27f7d   113M    196 个 kernel
```

`probe.json` makes this intermittent rather than steadily visible: it caches the probe result by
compiler, not by locale, so **whichever locale fills it in first decides the whole machine's cache
directory**, until the compiler's size/mtime changes -- at which point the whole machine quietly
moves to the other tree and recompiles everything.

The fix has two steps. First, make the probe itself ask under the C locale
(`jittor_utils.c_locale_environment()`) rather than requiring every caller to export it; that
converges on exactly the tree the nox gates already use, so the gates' hot cache is unaffected and
hand-run runs join it. Second, change the memo slot name from `target_arch:<cc>` to
`target_arch:c-locale:<cc>` -- **changing only the probe and not the slot is not enough**: a
`probe.json` that already stored translated text keeps answering, so `target_arch_key()` no longer
agrees with its own cache, and that machine stays forever in the directory chosen by the first
locale that asked. With the slot name changed, every home re-probes once and they all land on the
same key (measured here: before the change the process resolved to `arch6fe5b27f7d`, after it to
`arch0fe147db07`, the 1.1G hot tree with 7801 kernels). The gate is
[`tests/build/test_build_config_cache.py`](https://github.com/Jittor/jittor/blob/master/tests/build/test_build_config_cache.py)
`::test_the_instruction_set_key_does_not_depend_on_the_message_locale`, which was first shown to
go red (`{'C': 'arch0fe147db07', 'zh_CN.UTF-8': 'arch6fe5b27f7d'}`) and only then fixed.

The other probes that enter the cache key (`version:`, `cuda_archs`, `nvcc_archs:`,
python-config, mpicc) were audited -- they all take structured numbers or flags only, and
**`target_arch` is the only one that fed translated free text into the hash**.

### Second item: oneDNN is built from source, four-way parallel, and nobody owned it

`mkl` (oneDNN v3) is **lazy**: `import jittor` does not touch it, and
`tools/run_test_suite.py`'s original warmup probe (`(jt.array([1.,2.])*2).sum()`) did not either.
With a cold cache it builds ~1200 TUs with cmake from source, and `--parallel` used to be hard
coded to `min(4, cpus)`. So that build landed on **the first case in collection order to reach a
CPU convolution/matmul**, and appeared as a long pause with no output at all.

Isolated A/B (same source tree, run serially one after the other, machine idle):

| `cmake --build --parallel` | Time |
| --- | --- |
| 4 (the old value) | **443s** / **448s** |
| 32 (the current default cap) | **87s** / **87s** |

About **5x**, saving roughly 6 minutes per cold cache. The current value converges on both core
count and memory (`JT_ONEDNN_BUILD_JOBS` can override it directly), and still falls back to the
core count on small runners.

End-to-end confirmation (per-library timing of the warmup probe, same machine):

```text
mkl       available  105.3s      <- oneDNN 源码构建，并行度已放开
cutt      available   10.6s
cub/cudnn/cublas/curand/cufft/cusparse   各 0.0s
```

In the original cold run the same oneDNN build took about 480s at `--parallel 4`, and was charged
to `test_bmm_cuda`.

The warmup probe now also builds the backend libraries explicitly
(`probe_library(..., load=True)`): **this does not reduce the total work**, it only makes that
cost attributable in the warmup step and gives it a timeout, instead of disguising it as some bmm
case being 466 seconds slow. The communication libraries (`mpi`/`nccl`/`hccl`) are **not** in the
warmup list -- their loaders initialize communicators and wait for other ranks, and there are no
other ranks here; a warmup that might block forever is worse than the pause it would replace.

### Cache hits do not take the global lock (they once did, and that was the parallelism bottleneck)

`Op::jit_run` (`src/core/op.cc`) checks the **in-process** `jit_ops` map first and runs directly
on a hit; only a miss goes through `OpCompiler::do_compile`. `jit_ops` only helps "the second use
of the same key in the same process" -- with a fully hot disk cache, each process still goes
through `do_compile` once for each distinct JIT key.

**In the past, the first line of `do_compile` was `jittor::lock_guard lg`**, so the whole section
(including pure hits) ran inside the global `jittor.lock`. What happens inside the lock differs
enormously between the two cases -- cold is a whole nvcc/g++ run (seconds), hot is a key
comparison plus one `dlopen` (milliseconds) -- but the hit path queued up just the same.

`do_compile` no longer takes the lock; the lock moved down into `jit_compiler::compile()`, and is
**taken only when compilation really happens**. A hit follows an unlocked fast path: compute the
cache key this compile command would produce now, compare it with the `<product>.key` recorded
next to the artifact, and if they match, `dlopen` directly.

The only basis for that path being safe is **publication order**: `cache_compile()` first
`rename()`s the finished product into place and **only afterwards** writes `.key`, both being
same-directory `rename()`s and therefore atomic. So "a matching key is visible" implies "the
directory entry next to it already points at that build's complete product" -- there is no
intermediate state with the key published and the product not. Generated source is the only input
that exists solely in memory; the fast path **does not write it** (writing is a build action) but
hashes that string directly, which is exactly equivalent to hashing the file it would be written
as.

`tests/build/test_jit_cache_concurrency.py` guards these points: it samples a slow rebuild from
the outside and fails as soon as "a new key with an old product / half a product" appears
(changing `install_file` to an in-place copy reproduces the failure); it also has 8 processes
concurrently cold-compile overlapping kernel sets into one cache, for 3 rounds, requiring every
process's numerics to agree entry by entry.

Measured (8-GPU machine, exclusive `JITTOR_HOME` and `cache_name`, each number run twice):

| | Before the fix | After the fix |
| --- | --- | --- |
| same batch of fully hot kernels, nobody holding the lock | 0.226s / 0.225s | 0.239s / 0.223s |
| same batch of fully hot kernels, another process holds the lock for 30s | **30.231s / 30.231s** | **0.225s / 0.225s** |
| `tests/nn` single-process hot run | 59.7s / 61.1s / 61.2s | 60.9s / 59.9s |
| `tests/nn` split into 8 files, 8 processes sharing one hot cache | **76.9s / 85.9s** | **26.4s / 30.1s** |

The second row is the mechanism itself: before the fix it waited the full 30 seconds, after the
fix it is completely unaffected. The fourth row is the result: with the same 8 processes sharing
one hot cache, it used to be 1.3–1.4x **slower** than a single process, and is now more than 2x
**faster**. All four `tests/nn` runs are 360 passed, 0 failed, exactly matching serial.

(Note the third row: a single process contends with nobody, so it should not get faster, and it
did not -- this row guards against regression, it is not a benefit.)

Two sources of interference must be removed before measuring this, or the numbers measure
something else: `import jittor` itself runs entirely inside `lock_scope()`, so a lock-holding
process started **before** the import gets waited out by the import; and the first
`transpose()` lazily builds the `cutt` backend library through Python's `compile_custom_ops`,
which is another (still lock-holding) once-per-process path rather than a per-JIT-key cost. Both
must be done before the table starts.

**Not done yet:** on the Python side, `@lock.lock_scope()` still wraps all of
`compile_custom_ops` / `compile_module` / `download_url_to_local`, even on a plain cache hit. It
is once per process rather than once per key and is no longer the main term in the fourth row
above, but it still serializes multi-process cold starts.

Previously this section recommended "give each worker its own `JITTOR_HOME` and seed a warm cache
with `cp -al` hard links". **That is no longer needed**, and that route always had a cost: a new
home's first `import jittor` still rebuilds `jit_utils` and exits with code 3 asking to be re-run,
and that rebuild also invalidates part of the seeded kernels (measured: about 84/875 recompiled
per worker home). Sharing one hot cache is now both cheaper on disk and faster.

(The same data also says one thing: this 384-core machine has a `user/real` of only **1.5** while
cold-running `tests/nn`, and **4.9** while hot-running it -- the bottleneck was never compute.)

### Two things to be careful about when reading these numbers

**"Hot cache" must be proven, not assumed.** In the 37:03 / 1:02 table above, both runs have
identical passed/skipped/subtest counts; one run took 35 minutes longer and the report looks the
same. So before attributing a "slow" observation to anything else, first confirm the cache really
is hot -- the cheapest confirmation is **running the same command again**, because the second
run's time is the suite's own cost.

**Use `-v`, not `-q`, on large directories.** `-q`'s failure list is **printed only in the final
summary**: a `pytest tests/ -q` that ran for 4 hours and was killed by a timeout at 76% left not a
single failure in the log. `-v` writes a `…::test_x PASSED/FAILED` line as each case finishes, so
a killed run still keeps the full record up to that moment. Measured on the same file: `-v` has 23
result lines before the summary, `-q` has 0. This is the same flaw as "a session that dies
mid-way produces a log that says finished", seen from the other side; use it together with the
sentinel from `tools/check_session_completed.py`.

**When running concurrently against a shared cache, distinguish cold from hot.** The **hot** case
can now run in parallel against a shared cache: cache hits no longer take `jittor.lock` (see
"Cache hits do not take the global lock" above). The **cold** case is still serial: a process that
really needs to compile still has to take the same global lock, so "start more processes" to
parallelize the first compilation still does not work -- **the first build of a new JIT operation
or extension should run serially**, or be given its own `JITTOR_HOME` / `cache_name`. The
in-process parallel compiler (`use_parallel_op_compiler`) remains the only parallel path for cold
compilation.

## CI support matrix

Workflow status is **declared explicitly**, because "a nox session can run" does not mean CI has
the required hardware or dependencies. "Manual" means a maintainer must run the specified
fail-closed session on a configured machine before the interface may be claimed as verified --
**it must not be described as an automatic check**.

| Session | CI status | Runner and trigger |
| --- | --- | --- |
| `cuda` | automatic | Pushes and the weekly schedule use the declared RTX 4090 / CUDA 12.2 runner. Maintainers may add the `ci:cuda` label to a PR; the label, reopening and subsequent synchronize events all run the same gate. |
| `optional` | manual | No CUDA runner with complete dependencies is declared. Run `python -m nox -s optional` in the preset environment described below. |
| `rocm` | manual | No AMD/ROCm runner is declared. Run `python -m nox -s rocm` on a real supported AMD GPU. |
| `mpi` | manual | No multi-process MPI runner is declared. Run `python -m nox -s mpi` in an environment with a working launcher and compiler wrappers. |
| `nccl` | manual | The declared CUDA runner guarantees only one RTX 4090, while this gate needs two visible GPUs. Run `python -m nox -s nccl` on a multi-GPU host. |

**This matrix describes current scheduling, not the degree of backend support.** Manual results
must record the commit tested, the toolchain, the device topology, the command and the pytest
result; **an unavailable runner is not a pass.**

The maintained CUDA session runs the complete CUDA backend directory, dtype coverage, CPU/CUDA
device parity, the Torch TF32 controls and the strict CUDA OpInfo suite.

The `nccl` session needs two visible NVIDIA GPUs by default. It first warms an isolated JIT cache
serially for each rank, then uses `jittor.distributed.launch` to verify real flat FSDP2 parameter
sharding, NCCL all-gather and reduce-scatter, and sharded optimizer updates against independent
NumPy results. Setting `JITTOR_NCCL_WORLD_SIZE` and exposing at least that many devices runs more
than two ranks.

The `optional` session is a fail-closed offline CUDA gate against preset TorchMetrics,
mmcv-lite/MMEngine, PEFT, Safetensors, TensorDict and the deployed FlashAttention adapter. It
probes each package before pytest, enables the Jittor Torch shim explicitly, and treats **a PEFT
import failure as an error rather than an optional skip**. When `JITTOR_FLASH_ATTN_JITTOR_SRC`
points at the official FlashAttention checkout, the session runs in two phases: the ordinary
optional tests run with the deployed math adapter, then a phase **requiring the native
implementation** runs fused fp16 forward, dense/varlen/packed backward, dropout RNG replay, GQA
and the float32 opt-in tests. The native phase defaults to head dimension 32 and fp16; the
FlashAttention capability environment variables **extend** this base set rather than replacing
it. **The native phase cannot be satisfied by fallback.**

## Adding coverage

Use an **OpInfo definition** when an operation can share the standard sample, reference, dtype,
gradient and device-parity machinery. Use a **targeted test** when the contract involves state,
mutation, serialization, error behaviour, module lifetime, import order, distributed coordination,
or a specific regression that cannot be expressed as an OpInfo.

A new operation typically needs:

1. independent forward samples and a reference;
2. applicable dtype, shape, axis, empty-tensor, broadcast and non-contiguous cases;
3. gradient and second-order-gradient declarations backed by tests;
4. device parity on **every accelerator claimed as supported**;
5. explicit error-contract tests;
6. an OpInfo report entry or targeted test name, so that **missing coverage can be discovered**.

## Acceptance

A test-system change is complete when collection has no device side effects, new harness logic
has its own targeted tests, the structural gates pass, at least one mandatory CPU case really
executes, and every hardware result distinguishes pass, framework failure and environment
unavailability.

**Quantity alone is not an acceptance criterion** -- the evidence must really exercise the
claimed semantics.

# Numerical, gradient and coverage evidence

[Section contents](../index.en.md) · [中文](test-evidence.md)

Prerequisite: a source checkout and the [test system](test-system.en.md) process-mode rules. This page explains what a numerical result and a coverage count prove.

## The three layers of operator evidence

## 1. Independent forward reference

Each `OpInfo` describes the callable, sample constructors, dtypes, autograd support, tolerances
and targeted skip/xfail policy. `tests/ops/test_ops.py` expands this database on the requested
device and compares against an independent NumPy or mathematical reference.

**The reference implementation must not call the Jittor operation under test.** Sample
constructors should cover meaningful shapes, axes, broadcasting, optional arguments and
error-prone dtypes, rather than **piling up quantity without semantic diversity**.

## 2. Numerical gradients

Differentiable OpInfo entries run float64 CPU `gradcheck`; operations that support second-order
differentiation also run `gradgradcheck`. Marking `supports_autograd=False` or
`supports_gradgrad=False` is **a contract decision, not a way to turn a failing test green**, and
the reason must be given in the definition or in the issue ledger.

Numerical checks prove the derivative formula on CPU. **They do not prove that an accelerator
correctly executed the same backward kernel.**

## 3. Device parity

[`tests/backends/parity/test_device_parity.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/backends/parity/test_device_parity.py)
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
| [`tests/structure/public_api_manifest.json`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/structure/public_api_manifest.json) and its gate | whether the native 1294 names **still exist** |
| [`tests/structure/torch_api_manifest.json`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/structure/torch_api_manifest.json) and its gate | whether the Torch surface's 1295 names **still exist** |
| [`tests/_helpers/api_coverage.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/_helpers/api_coverage.py) (`JITTOR_API_COVERAGE=1`) | which entry points a run **actually invoked** |
| [`tools/api_coverage_ratchet.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/api_coverage_ratchet.py) and the two baselines | whether the set of uncalled entry points **has grown** |
| [`tools/probes/opinfo_dtype_gaps.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/probes/opinfo_dtype_gaps.py) | which (operator, dtype) pairs **work but are not declared** |

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

[`tests/core/test_var_residency_contract.py`](https://github.com/Jittor/jittor/blob/2.0-refactor/tests/core/test_var_residency_contract.py)
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
[`jt.introspection`](../../notes/runtime/runtime-introspection.en.md), rather than reading native flags or
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

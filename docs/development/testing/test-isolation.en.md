# Test isolation, skips and completion

[Section contents](../index.en.md) · [中文](test-isolation.md)

Prerequisite: a source checkout and the [test system](test-system.en.md) process-mode rules. This page explains how to isolate tests and interpret skips, crashes and completion.

## Test categories and markers

Markers are registered in
[`pyproject.toml`](https://github.com/Jittor/jittor/blob/2.0-refactor/pyproject.toml):

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
| `REAL_TORCH_SITE` | the in-process independent PyTorch oracle (see [process-mode isolation](test-system.en.md#process-mode-isolation)) | the optional PyTorch comparison is skipped; **a deployed shim does not stand in for it** |

Native GPT forward/gradient probes live in `tools/probes/acl/` and require explicit real-device setup. Qwen3-specific probes live in [jittor-transformers testing](https://github.com/JittorEcosystem/jittor-transformers/blob/main/docs/testing.en.md), with pinned versions and explicit model arguments. See the [Ascend 910B guide](../../guides/hardware/ascend-910b.en.md) for device configuration. Migration grants no new hardware acceptance.

## Maintainer probes

[`tools/probes/`](https://github.com/Jittor/jittor/blob/2.0-refactor/tools/README.md) contains **scans**,
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
| `opinfo_dtype_gaps.py` | which (operator, dtype) pairs work but are not declared (see [coverage evidence](test-evidence.en.md)) |
| `error_message_probe.py` | the **verbatim text** a user currently gets for real mistakes; the diff between two runs is the acceptance record for that change |

These consistency scans need no external reference values and can inspect expressions without
an independent reference. They do not replace OpInfo forward references or numerical gradient checks.

## Skips and known failures

- **skip** means a precondition is unavailable, or a contract is deliberately unsupported; its
  reason must name **the exact precondition or limitation**.
- **expected failure** means a reproduced framework defect. pytest uses strict xfail, so a fix
  produces an XPASS and **forces the ledger to be cleaned up**.
- **Do not** catch arbitrary exceptions outside the test body and turn them into skips. Probe the
  optional environment **narrowly** before execution.
- Every persistent expected failure must be listed in the
  [historical record](https://github.com/Jittor/jittor/blob/3679de22c02c83b982432a0e1bdb3da148411c50/agent/manuals/known-issues.md)
  with an owner and exit criteria.

## Check execution and completion before interpreting failures

A green summary is useful only if the selected cases ran and the process completed.
Check these independently:

| Observation | Required interpretation |
| --- | --- |
| No cases executed | Explain the prerequisite, or record the intentional exception in `gate_scope.EXECUTES_NOTHING` |
| Too few devices | Report `insufficient-devices`, separately from an absent accelerator |
| Process ended without a completion sentinel | Treat the run as incomplete; do not infer success from the absence of `FAILED` |

`JITTOR_TEST_REQUIRE_EXECUTION=1` makes unexplained non-execution fail the gate.
At session end, `tests/_helpers/pytest_policy.py` prints:

```text
========== files this session proved nothing about ==========
tests/.../test_x.py  collected 0 tests
tests/.../test_y.py  12 skipped, 0 executed
```

`collected 0 tests` means no cases were collected; inspect collection errors and filters.
`N skipped, 0 executed` means collection succeeded but execution did not. An
`expected here` annotation names an explicit `EXECUTES_NOTHING` exception; `explained`
names a recognized environment reason. Read unannotated entries before the failure list.

`tests/_helpers/session_completion.py` records completion from `pytest_sessionfinish`,
including collected/executed counts. A native process death cannot write that sentinel.
`tools/check_session_completed.py` uses sentinel > log marker > pytest summary as its
evidence order. Preserve the sentinel and return code with the log. Use `-v` for long
runs so a timeout still leaves individual results; quiet output alone is not a completion check.

A new gate needs a negative control: introduce the specific defect it should detect,
confirm rejection, then restore it. A scan over an obsolete or empty path is not coverage.

## Pin the child process to its source and device

Use `_helpers.child_process` for Python children. Parent pytest configuration adds the
checkout to `sys.path`, but that alone does not export a matching `PYTHONPATH` to children.
A bare interpreter launch can load another editable checkout and report an irrelevant pass.
The AST gate `tests/structure/test_child_process_contract.py` covers test launches and
`tools/run_test_suite.py` / `tools/gate_conclusion_diff.py`.

For an installation-resolution test that deliberately excludes repository paths, use
`child_env(..., repo_paths=False)`. Check the imported source, process-mode variables and
requested device inside the child. Do not infer residency from a flag or available GPU:

```python
where = x.location()
expected = "device" if device == "cuda" else "cpu"
assert where == expected, "child asked for %s, tensor is on %s" % (device, where)
```

This example checks CPU/CUDA placement; other backends need their corresponding actual-device
assertion and zero-fallback checks. The child must execute and synchronize the selected work.

## Isolate crashes and stateful initialization

A known signal-death reproducer runs in its own process through
`run_child_script(..., crash_isolated=True)`. A native crash otherwise kills pytest itself,
including every later case. Device traps can also invalidate a CUDA context.

Examples are `tests/backends/cuda/test_auto_flush_graph_split.py` and
`tests/ops/test_index_bounds.py`. The former compares loss and gradients as well as survival:
stopping the crash does not prove backward correctness. Initial namespace/activation assertions
also require a clean process when prior lazy imports would change their premise.

## Control autotuning in residency comparisons

When comparing a scheduling or residency flag such as `auto_flush_ops`, fix cuDNN algorithm
selection before measuring both sides:

```python
import jittor as jt
jt.cudnn.set_benchmark(0)
```

Autotuning otherwise chooses an algorithm based on available memory, so the comparison changes
both the intended flag and the convolution algorithm. Compare forward and gradients, and compute
elementwise relative error against the corresponding element, not a whole-gradient RMS.
See the [numerics contract](../../notes/numerics/numerics-contract.en.md); KI-EXEC-001/003 and KI-TEST-004
in the [historical record](https://github.com/Jittor/jittor/blob/3679de22c02c83b982432a0e1bdb3da148411c50/agent/manuals/known-issues.md)
provide the historical motivation. This protocol does not grant new hardware acceptance.

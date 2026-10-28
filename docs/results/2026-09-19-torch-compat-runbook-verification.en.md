# Downstream repository torch-compat runbooks and four-axis measurements

- Status: accepted within the measured scope; conclusions for several repositories remain report-derived
- Date: 2026-09-19
- Baseline commit: `90fe0b9d` (only partial re-verification after merging the remote; see "Re-verification after merging the remote")
- Verification scope: CUDA, same device on both sides; the four axes (support list / accuracy / memory / speed) for 13 downstream repositories, with the execution tool
  `verify_repo.py` delegating to `compat/tests/torch/_ecosystem_runner.py`; the reference is real torch
  2.13.0+cu129. verl and trellis have no environment on this machine, so their conclusions are report-derived; the four axes were not re-run on the merged tree
- Maintainer: Jittor maintainer
- Review conditions: any target library in `agent/skills/<lib>-torch-compat/` moves to a new major version, the device branch of `_ecosystem_runner`
  changes again, or the measurement convention of `verify_repo.py` changes

## What was done this time

For each of the 13 downstream repositories I wrote a runbook for "running on the jittor shim + comparing against native torch"
(`agent/skills/<lib>-torch-compat/`), and separated the **template and four-axis acceptance protocol** into
[`torch-compat-repo-runbook`](https://github.com/Jittor/jittor/blob/master/agent/skills/torch-compat-repo-runbook/SKILL.md).
A runbook is not a documentation draft: every entry was run, and the results were written back into its own `## 实测（2026-09-19）` section.

The four axes = **support list / accuracy / memory / speed**. The execution tool is
`agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py`, and single-case execution is delegated to
the project's own `compat/tests/torch/_ecosystem_runner.py`, so the numbers share the same source as the ecosystem gate.

## A gate defect that was fixed

**Symptom**: `_ecosystem_runner.py --device cuda --runtime jittor` always fails on the first
numpy-backed tensor:

```
RuntimeError: dispatch_context.cc:52: Expected all tensor inputs on the same backend
and device ... first input backend=0 index=0 but another input has backend=1 ...
```

**Root cause**: the jittor branch of `_select_device` returns the identity function, with the comment rationale that "Jittor moves the
graph, not the tensor". But that statement only holds for tensors Jittor **creates itself**: `use_cuda=1` places parameters on the
device, while tensors produced by `from_numpy` still reside on the host, and putting the two into the same operator triggers the same-backend check.
Real PyTorch's `from_numpy` likewise returns a CPU tensor, and the caller must move it itself — the harness moved it in the torch branch
but not in the jittor branch.

**Fix**: the cuda branch of `compat/tests/torch/_ecosystem_runner.py` was changed to explicitly move host-backed
tensors to the device. This is not a way around Jittor: the torch side behaves identically, and Jittor itself has no problem. The NPU branch is left
as is, because there is no ACL device on this machine to verify against, and unverified changes should not enter the tree.

**Scope**: the ecosystem gate runs on CPU by default (`JITTOR_TEST_DEVICES=cpu`), so this fix does not affect the existing
nightly; what it unlocks is four-axis measurement on CUDA.

## A second gate defect fixed along the way: the structure gate was reddened by build residue

`tests/structure` is the regression gate for this batch of changes. After the changes it reported 2 extra failures locally, both in
`test_import_layering.py`; on a pristine HEAD worktree the same code has only 3 (all 3 unrelated to
this change). Tracking them down one by one, neither of the 2 was a code problem.

**Symptom**: `python tools/lint/check_import_layering.py` reported
`cycle-surface FAIL (1) — 4 import-time cycles, up from 3`, while nothing in the diff could explain it.

**Root cause**: under `compat/` there was a gitignored setuptools build artifact
`compat/build/lib/jittor/compat/...` (148 `.py` files, a whole-tree copy from `setup.py build`).
`scan_roots` recursed the **entire** `compat/ -> jittor.compat` tree by `package-dir`, so this copy was read in as
source: the module count went 539 → 687, and the copy's own paired `__init__` files formed cycles that do not exist in the source.
Along the way, a onetime suspect was cleared: the newly added `python/jittor/nn/functional/_amp.py` merely joined
the large cycle that already existed (146 → 147 modules), `MAX_CYCLIC_MODULES = 164` is far from being reached,
and `CYCLIC_SUBPACKAGES` already contained `jittor.nn` — it triggers no contract by itself, and both failures came entirely from the residue.

**Fix**: `discover_modules` now skips `build/` and `dist/` directories that are **not packages** — the criterion is "the directory
has no `__init__.py` of its own", because `jittor.build` is a real package and judging by directory name alone would cause false positives.
These two names are not a new convention: `[tool.jittor.sdist]` in `compat/pyproject.toml` already said
`exclude = ["build/**", "dist/**", "tests/**"]`. The regression test
`test_build_output_inside_a_package_root_is_not_source` uses a temporary directory to lock both sides at once
(the artifact is skipped, `jittor.build` is kept).

**The residue was not deleted**: it was something already present in someone else's environment, and deleting it would only postpone the same problem to the next build.
After the fix the residue is still in the tree, the gate is green, and the numbers are character-for-character identical to those without it (539 modules / 1424 edges / 3 cycles).

**Structure gate status**: `tests/structure/test_import_layering.py` 16 passed. The other 3 failures
(the two find_packages and MANIFEST ones in `test_packaging_structure.py`, and the collection side-effect one in `test_pytest_contract.py`)
fail in the same way on pristine `90fe0b9d`, are unrelated to this change, and were left unaddressed.

**Re-verification (evening of 2026-09-19, the grad-identity fix)**: the same `compat/build/` residue also reddens two other
scanners — `tests/structure/test_cleanup_structure.py::test_cross_file_duplicate_implementations_are_reviewed`
(which mirrors every implementation in `compat/` as an "unreviewed duplicate implementation", adding 1262 entries, each one
`compat/x` against `compat/build/lib/jittor/compat/x`) and
`compat/tests/structure/test_compat_write_entry_points.py::test_every_process_global_write_in_compat_is_classified`
(all 54 unclassified process-global writes come from the residue; the only entries not under `compat/build/` are pytest's own node names).
Together with `tests/structure/backends/comm/test_comm_resource_layout.py::test_legacy_runtime_resource_trees_are_absent`,
caused by `python/jittor/extern`
(a directory kept on this machine; it passes on a fresh checkout — see the "Gates and reproducing" section of the [vLLM-Omni report](2026-09-14-vllm-omni-h3-enablement.en.md)).
All three red items are caused by local artifacts **not tracked by git** (`build/` is in `.gitignore`, `extern/` is ignored),
which do not exist on a pristine checkout and are therefore green there; none is related to the source changes.

## Four-axis measurement summary

CUDA, same device on both sides, speed taken as the minimum over repeats, see each runbook for `fallback_count`.

| Repository | Cases that passed | Accuracy (worst, relative to overall tensor magnitude) | Speed jt/torch | Notes |
| --- | --- | --- | --- | --- |
| transformers | tiny 5/6, large 5/5 | ≤3.6e-4 / ≤1.51e-3 | 1.71–3.21x / 1.03–1.19x | `transformers_t5` on CUDA fails in apex on the **oracle side** |
| diffusers | 3/3 | 4.93e-5–6.77e-4 | 1.30–2.89x | Large case memory: jt 3.78 GiB > torch 1.10 GiB (**old convention**, see below) |
| peft | 1/1 | 2.745e-4 (abs 1.209e-2) | 2.04x | Requires a transformers 4.x package site |
| ms-swift | 1/1 | 2.745e-4 | 2.05x | Same as above |
| mmcv | 1/1 | 2.514e-7 | 0.998x | Uses mmcv-lite (full mmcv needs CUDA compilation) |
| mmengine | 1/1 | 3.983e-7 | 0.984x | |
| torchmetrics | 0 (no cases) | — | — | See "Real defects" below |
| tensordict | 0 (no cases) | — | — | The shim lacks `torch.distributions.constraints` |
| vllm | 0 (no cases) | — | — | Adapter lifecycle tests 9 passed |
| vllm-omni | No cases | — | — | Small VAE decoding runs end to end (including cold JIT 167.5 s) |
| flash-attention | No cases | — | — | First observation of a fused hit |
| verl | — | — | — | No environment on this machine, report-derived |
| trellis | — | — | — | No environment on this machine, report-derived |

The speed ratio of small cases (2–3x) is dispatch-bound and does not represent real sizes; the transformers large
tier (1.03–1.19x) is the citable form.

## Memory: the convention was wrong twice, and the remaining gap is rooted in the shim's grad object identity

Originally the tool asked each side its own question, and they were not the same quantity: the oracle side `torch.cuda.max_memory_allocated`
(**live** bytes, single card), the Jittor side `get_mem_info().total_cuda_used` — which is **live + cached free**
and, per `mem_info.cc:316-322`, sums over **all** devices. Reserved against allocated.

The tool has been changed so both sides report both the live and pool numbers: torch uses `max_memory_allocated` /
`max_memory_reserved`, Jittor uses `device_memory_used(N)` / `device_memory_reserved(N)`.
When the jittor under test is too old and lacks these two APIs, it reports `jittor_peak_bytes = -1` with a `memory_note`
explaining why, **rather than** substituting reserved — the deployed copy from 2026-09-11 falls into this branch.

After the convention change the gap remained (2.88x), so the root cause was investigated phase by phase. Along the way a second convention trap was hit, recorded here as well:

`jt.core.get_peak_allocator_used_memory()` looks like it should be the counterpart of `max_memory_allocated`,
but `MemoryProfiler::get_memory_info` accumulates `SFRLAllocator::sfrl_allocators` **without filtering for CUDA**
(`memory_profiler.cc:58`), i.e. **host + device**. Measured: a 512 MiB pure-CPU Var pushed it up by
512 MiB, and adding a 256 MiB CUDA Var pushed it to 768 MiB; it also has no reset interface. **It cannot be put side by side with torch's
`max_memory_allocated`**; the only comparable ones are `device_memory_used` and the CUDA-only
`total_cuda_used`.

### Addendum (2026-09-20, `f1a498c7`): the missing counterpart above has been added

The statement above that "the only comparable one is `device_memory_used`" was the state at that time — jittor had no
`max_memory_allocated` of its own. Now it does: `jt.core.get_peak_device_used_memory(N)`
(`src/core/memory_profiler.h`), where `MemoryProfiler::check()` records, after every op, the high-water mark of the sum of `used_memory` across that device's
pools. 8-layer llama, seq 512, batch 4, fp32:

| | Reading |
| --- | --- |
| torch `max_memory_allocated` high-water mark | 6655 MiB |
| jittor `get_peak_device_used_memory(0)` | **7272.6 MiB (1.09x)** |
| jittor `device_memory_reserved(0)` | 7479 MiB |
| The "step peak" reported by python polling `device_memory_used` | 3406.5 MiB ← **not the peak** |

The last row is the lesson of this episode: **the peak can only be asked of the runtime**. The sampling thread only runs when the interpreter releases the GIL (inside a step,
that is, while waiting on the device), so a 48 s first step lets it sample the real peak, while a 47 ms subsequent step can only sample an intermediate
value — hence the same step yields two readings, 3406.5 / 7144.6 (2:2 across 4 runs), and a
"first-step 2x transient" is conjured out of thin air. The four-axis tool's `jittor_peak_bytes` used to be sampled this way, so that number is not the peak;
from `f1a498c7` it reads the interface above instead, and reports it as unmeasurable when the interface is missing. The 1.09x in the table above was measured directly with the same shape on both sides;
the individual cases of `verify_repo.py` have not yet been re-run under the new convention.

Like its host+device sibling, it **has no reset interface**: it reads "the high-water mark since profiling was
enabled", so to measure a given phase, turn `profile_memory_enable` on at the start of that phase. The regression is at
`tests/runtime/test_profiler.py::TestMemoryHighWater`.

The same cause also bit the pool column: when the sampling thread cannot get the GIL it under-reads, and in a 5-step small probe it read
`device_memory_reserved` as 512 MiB while the runtime held 1536 MiB at that moment — a threefold under-report (from `f2ec8c5f`
the tool merges in one more runtime reading at wrap-up; sampling can only miss, never invent, and for the high-water semantics of the torch
side this is an identity transformation). The conclusion is the same: **a memory reading must either ask the runtime or be declared unmeasurable**; nobody should use "the maximum
a python thread saw" as a peak.

The root cause is in the shim: `p.grad` is a new object every step. torch's `zero_grad(set_to_none=False)` zeroes
the existing tensor in place and `AccumulateGrad` accumulates in place, so within one training loop `p.grad` is always the same object;
the shim's `_zero_grad` rebuilds with `zeros_like`, and the optimizer-free accumulation is written as `prev + gr`, both of which replace
the object. So any code that holds `[p.grad for p in model.parameters()]` across steps (gradient clipping, the four-axis
memory pass) pins a complete set of gradients per step on jittor. `large_transformers_bert` measured 198 extra
Vars and about 415 MiB per step — 198 ≈ the parameter count, 415 MiB ≈ the parameter byte count. After changing both places to in-place operations, live memory fell back between steps
to about 840 MiB (839–845 MiB measured at phase boundaries), the Var count stayed constant at 403–411 and no longer grew with the step count; in the same run, the relative error of the 198 outputs and gradients
against torch was identical item by item to before the fix (still in the ≤3.6e-4 bracket of the table above), so what was fixed was object identity, not the numerics.

After the fix the same case was re-measured (repository checkout, device-only counts only):

| Phase | torch live peak | jittor live peak |
| --- | --- | --- |
| First backward | 1640.3 MiB | 4048.2 MiB |
| First warmup step | 2122.5 MiB | 4469.7 MiB |
| Steady state per step, warmup / timed | 2083.6–2122.5 MiB | **2366.4 MiB** |

So the whole-run 2.11x (live) / 2.01x (pool) **comes almost entirely from the first-step transient**: jittor's first
backward needs 4048 MiB and the first training step 4470 MiB, then it settles at 2366 MiB, while torch has no such transient;
the steady state differs by only **1.11x** per step. jittor's pool is high-water and never returns memory to CUDA, so the transient is permanently recorded in the
pool, and the whole-run peak reported by the four-axis tool carries it all along.

When doing a memory budget, both numbers must be stated: the steady state is about **1.11x**, but **the first-step peak of about 2.1x** is the one that must fit.
The 1.10 -> 3.78 GiB for diffusers in the table above is still the **old convention** and has not been re-measured under the new one.

(External sampling is still unusable: `nvidia-smi --query-compute-apps` cannot see the process (container pid mapping),
and per-GPU `memory.used` includes co-tenants. So we can only ask the runtime itself, which makes it all the more important to ensure we are asking for the same quantity.)

## Real defects found along the way

These are not problems with the runbooks but with the shim/compat layer, and deserve items in their own right:

1. `torchmetrics`: `Reduce dim out of range` (`src/op/reduce_op.cc:291`), the classification
   case cannot be produced; plus an nvcc return-512 error for `setitem`.
2. `tensordict`: the shim's `torch.distributions.constraints` is empty (attributes such as `real` are missing),
   and `import tensordict` fails outright.
3. `peft 0.20.0`: the shim lacks `torch.distributions.wishart`, so import fails.
4. `ms-swift 4.5.3 + peft 0.20.0`: `Linear` lacks the `config` parameter, TypeError.
5. `torchmetrics` helper function marker names are inconsistent: `_jittor_fast_*` vs `_jittor_orig_*`.
6. `transformers_t5` fails on the **oracle** side (real torch 2.13.0+cu129) in apex
   `fused_rms_norm_affine_fwd: input must be contiguous` — not the shim's fault, but it will block
   all subsequent t5 CUDA comparisons using that oracle venv.

## Reproducing

```bash
cd <repo>
source $JITTOR_LAB_ROOT/minimax-h3/env-jittor.sh
export JITTOR_HOME=<本任务独占的缓存目录>
REAL_TORCH_PYTHON=$JITTOR_LAB_ROOT/_state/h3/venv-oracle-cu129/bin/python \
"$VENV/bin/python" agent/skills/torch-compat-repo-runbook/scripts/verify_repo.py \
  --repo <lib> --device cuda --repeats 5 --out <lab>/_state/<topic>/verify/<lib>
```

`--list-only` first shows which cases the library registers, without starting a process. When a library is missing, build a package site with
`pip install --target <site>` and point `JITTOR_ECOSYSTEM_PACKAGE_SITE` /
`JITTOR_ECOSYSTEM_REFERENCE_PACKAGE_SITE` at it; **do not install it into the shim interpreter itself**.

The package sites used this time: `$JITTOR_LAB_ROOT/_state/verify-ml/site` (peft 0.20.0, ms-swift 4.5.3,
mmcv-lite 2.2.0, mmengine 0.10.7) and `site-peft17` (transformers 4.56.2, peft 0.17.1,
ms-swift 4.5.2); `$JITTOR_LAB_ROOT/_state/verify-misc/site` (torchmetrics, tensordict).
The raw `verify-report.json` files are in the respective lab output directories and do not enter the main repository.

## Re-verification after merging the remote

After this report was written, the branch diverged from the remote: 10 local commits against 59 on the remote (merge-base
`4bd42443`), and the remote line carried a whole autocast / GradScaler / half-precision rework that sits right on the objects this document
quantifies. Following the collaboration rules, the local work was committed first, then `git merge origin/2.0-refactor`, merge commit
`5fca26ff`. Four files overlapped, and only `src/core/var_holder.h` conflicted — both sides added the same
`#include "runtime/device.h"`, differing only in comment wording, so the remote was taken per "keep the remote at overlaps"; the other
auto-merged files were checked one by one and both sides' content is present.

**Therefore the baseline for all the numbers above is still `90fe0b9d`, not the merged tree.** After the merge only the following were re-verified:

| Item | Result |
| --- | --- |
| Import-direction gate (3 contracts) | All pass: 543 modules / 1441 edges / 3 cycles, no new cycles |
| `tools/check_repo_layout.sh` | OK, 221 Markdown files |
| The three affected files in `tests/structure` | **55 passed, 8 subtests** |
| dtype behavior of bias | 5/5 consistent with torch |

Those three files had 3 failures before the merge, all fixed on the remote by `f0bd4279`; when I added this report I forgot to
regenerate `MANIFEST.in` (`test_manifest_covers_runtime_trees_without_cache_payloads` is exactly what
measures this), and I have since added it per `tools/build/generate_manifest.py`; the whole file gained only one line.

The bias item was re-verified separately, because its three test cases live in `compat/tests/torch/` and cannot be collected in this lab
(see "Limitations"). Running `probe_bias_dtype.py` directly: under autocast,
`conv3d(f32, f32, f32)`, the `Conv2d` module, and `linear` all return **float16**, outside an amp region they return
**float32**, and the conv3d control group without bias is likewise float16 — matching real torch 2.13's dtype item by item.

**Not yet done**: the four axes have not been re-run on the merged tree. The remote changed the dtype / half-precision / autocast and generator
streams, which is exactly what those transformers/diffusers/peft/ms-swift cases measure, so the accuracy and
speed numbers in the table above need re-measuring after the merge. Another obstacle is that the four axes go through the deployed tree in the lab (the 2026-09-11 copy, see
the copy-deploy trap); to measure the merged tree the repository must first be redeployed into `venv-jittor` — that would touch the H3 lab,
so it must first be confirmed that nobody is using it.

## Limitations

- `compat/tests/torch/*.py` cannot be collected by pytest in the local lab venv
  (`compat/__init__.py` → `_aliases.py:8` relative import goes out of bounds); the torchmetrics/tensordict
  conclusions come from a standalone runner, not a verbatim reproduction by the gate.
- peft/ms-swift/mmcv/mmengine have only tiny cases, with no speed tier.
- flash-attention has no native baseline (importing real 2.8.1 in the oracle fails), so there is only the dispatch fact,
  no speed comparison. On the shim side what resolves is the stub in the deployed tree.
- verl / trellis have no environment on this machine; all conclusions are report-derived and are marked as such in the runbooks.

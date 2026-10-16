# Optimizer checkpoint and deterministic Accelerate continuation

Status: Historical single-process numerical acceptance. The later
[RNG and FSDP2 FULL continuation report](2026-10-14-checkpoint-rng-fsdp2-continuation.md)
supersedes the RNG and multi-rank limitations recorded below. ACL remains
unverified; this report preserves the original baseline and evidence.

Baseline: `cded3102037edc3663b8f2054a6ebc06da6ebfd2` (`origin/2.0-refactor`, fetched
before implementation). Implementation: the commit containing this report.
Maintainer: Jittor compatibility maintainers.
Review when: checkpoint FQN transforms, optimizer state ownership, FSDP provider
transforms, serialization allowlists, RNG owners, or the downstream versions change.

The baseline cannot import `get_optimizer_state_dict` from
`torch.distributed.checkpoint.state_dict`; its combined setter also ignores the
optimizer checkpoint. The implementation adds FQN-based optimizer get/set,
parameter-group restoration independent of live parameter/group ordering,
multiple optimizers, flattening, FULL FSDP provider transforms, host offload, and
rank-zero broadcasts. Model loading honors `strict`; model checkpoints can omit
frozen parameters. Nonempty submodule selection explicitly raises
`NotImplementedError`; this is not a complete implementation of all DCP options.

Actual Accelerate loading additionally exposed the missing
`torch.serialization.get_safe_globals`, `add_safe_globals`, `clear_safe_globals`,
and `safe_globals` APIs. Their implementation controls the restricted unpicklers;
custom classes fail before registration and after the temporary allowlist ends.
No downstream adapter or alternate optimizer mathematics was introduced.

The same seven test cases ran under independent binary PyTorch 2.11.0+cu128 and
the Jittor Torch frontend, with Accelerate 1.10.1 on both sides. The oracle identity
was checked independently and its package used the original binary libraries.
Missing environment dependencies were supplied in isolated external overlays;
shared environments were not repaired. CPU and CUDA used distinct JIT caches.
Both backends ran genuine tensor computation with fallback forbidden and observed
fallback count zero.

| Check | CPU | CUDA |
| --- | --- | --- |
| Independent PyTorch / shim cases | 7 / 7 passed | 7 / 7 passed |
| Maximum absolute error against independent reference | 1.49e-8 | 3.73e-9 |
| Backend fallback count | 0 | 0 |

Comparisons include every model parameter, optimizer moment and step, parameter
group learning rate, resumed loss, all parameter gradients, and the input
gradient. Inputs and initial weights are fixed local arrays; the model has no
stochastic layers. PyTorch 2.11 adds `decoupled_weight_decay=True` to AdamW group
metadata; the shim does not expose that newer field. It was checked explicitly
and excluded from cross-runtime structural equality; all shared metadata and
numerical entries were compared. Independent PyTorch currently raises a KeyError
when loading the combined FULL-plus-flatten option, so the portable flatten gate
uses `flatten_optimizer_state_dict=True, cpu_offload=True` and tests FULL offload
separately. This does not validate the unsupported upstream option combination.

Snapshots use public `torch.clone`: native `Var.clone` is a graph/storage alias,
so it cannot establish immutable Torch checkpoint evidence. The setter also uses
the public copy operation when installing a loaded tensor.

Reproduce the functional selection with the deployed shim and isolated CPU/CUDA
environments, then run the identical file in an independent PyTorch interpreter:

```bash
CHECKPOINT_DEVICE=cpu JITTOR_REQUIRE_ACCELERATE=1 \
  python compat/tests/torch/test_optimizer_checkpoint_state.py -v
CHECKPOINT_DEVICE=cuda JITTOR_REQUIRE_ACCELERATE=1 \
  python compat/tests/torch/test_optimizer_checkpoint_state.py -v
```

The implementation's focused serialization, checkpoint, and FSDP structure
selection produced 63 passed and one existing accelerator skip. Integration onto
`7ed9b3b52b658d2a3c1ac390eba09087c5e80f95` then produced 59 CPU passes and one
failure caused by missing `requests.HTTPError` in the isolated environment. After
supplying the dependency in that environment, the affected Accelerate case passed
on its own. All seven CUDA checkpoint cases passed on the integrated tree.

Repository layout validation passed. The integration pre-push full
`tests/structure` run produced 1429 passed, four skipped, and one failure in
`test_backend_grad_contract.py::test_the_manifest_is_exactly_the_gradients_in_the_tree`.
Its existing `DropoutACL` entry is not recognized by the direct-`grad`-method
scanner after the ACL implementation moved its backward to a CodeOp. This change
does not modify ACL or that manifest; the complete structure gate remains failing.
The earlier isolated-run ASV and xdist dependency failures were resolved in the
validation environment and did not recur in this run.

The integrated CPU `core` runner passed both process modes: native produced
98 passed, 41 skipped and one existing expected failure; Torch produced
224 passed and one accelerator skip. Combined: 322 passed, 42 skipped and
one expected failure. An initial invocation omitted the explicitly loaded
pytest-timeout plugin while plugin autoload was disabled; it stopped before
native test execution. The corrected complete invocation above exited zero.

An additional real Accelerate RNG continuation probe failed: Python and NumPy
streams restore, but the Torch CPU stream resumes from its seed instead of its
saved position. The compatibility API explicitly advertises seed-only behavior;
this implementation does not fix it. The native CPU/ACL snapshot API exists and
needs a separately verified frontend adaptation. CUDA already has a cuRAND
seed/offset interface, but this report does not accept complete RNG restoration.
The failing probe and raw logs are retained under the external task state tree
(`checkpoint-optimizer/rng_checkpoint_repro.py` and
`checkpoint-optimizer/logs/targeted-rng-failure.log`). Claims here cover deterministic
fixed-input numerical continuation only, not complete Accelerate state recovery.

## 2026-10-16: gradients published before optimizer construction

Status: CPU/CUDA behavior verified for the contracts below. Owner: Torch
optimizer compatibility maintainers. Baseline: `4fd2a82951d2a5300774862685a0e661e364e93f`.
Implementation: `dc6a178064c22bb5a31fc76bd55a879cc9497f5e`. Review when: Tensor gradient publication,
optimizer initialization, parameter groups or native lazy gradient creation changes.

A Parameter may already have `.grad` from a public assignment or `backward()`
when an optimizer is constructed. On the baseline, all eight combinations of
SGD/Adam/AdamW/RMSprop and these two gradient sources silently skipped the
update; moving the same gradient creation after the constructor worked.
Independent binary Torch 2.11 updated in all sixteen before/after controls.

The constructor now connects existing published gradient objects to the native
parameter groups, including missing gradients in other groups. It preserves
object/storage identity so another optimizer, in-place gradient edits and
`zero_grad(set_to_none=True)` observe the same state. When no gradient exists,
the original native lazy initialization remains active. An intermediate attempt
to prepopulate all groups with None caused nine native backward/step(loss)
regressions; that attempt was rejected and its failed log retained.

The new maintained file `compat/tests/torch/test_optimizer_preexisting_grad.py`
has nineteen contracts per device: both gradient sources and constructor orders,
two consecutive updates, Adam step state, two optimizers sharing a gradient,
missing gradients across groups, zero gradients with weight decay, and no-gradient
steps. Every inspected candidate tensor checks its actual backend location;
fallback is forbidden. The exact file also runs in independent binary Torch.

| Scope | CPU | CUDA configuration |
| --- | --- | --- |
| Candidate, new file plus four neighboring optimizer/gradient files | 57 passed / 19 CUDA skips | 76 passed / 0 skipped |
| Independent Torch, exact copy of new file | 19 passed / 19 CUDA skips | 38 passed / 0 skipped |

The CUDA configuration runs both CPU and CUDA parameters from the new file;
its nineteen CUDA cases execute on the accelerator. Some neighboring tests
explicitly exercise CPU, so the total 76 is not a count of CUDA-only tests.
The full candidate CUDA invocation observed fallback counter 0 before and after.
Independent Torch versions were 2.11.0+cpu and 2.11.0+cu128 respectively.
The candidate native CUDA binary SHA-256 was
`03b66666626c7fef5b1a2a9a645c5e80972cc9e8c8a316d106971db74c7d71ef`.

Core validation: 322 passed / 42 skips / one existing xfail. The final production
file is byte-identical to that core run; only the test device parametrization
was subsequently simplified. Structure/layout/manifest results: `1628 passed / 6 skipped; layout and manifest checks passed`.
The new test passes Ruff; the production file retains the same eight pre-existing
Ruff diagnostics as the fetched baseline, with no new diagnostics.

Reproduce the five-file selection in isolated CPU/CUDA environments:

```bash
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_optimizer_preexisting_grad.py \
  compat/tests/torch/test_torch_compat_grad_management.py \
  compat/tests/torch/test_optimizer_parameter_rebinding.py \
  compat/tests/torch/test_optimizer_scheduler_api_owners.py \
  compat/tests/torch/test_torch_compat_optim.py
```

Raw evidence is unversioned under
`$JITTOR_LAB_ROOT/_state/optimizer-existing-grad/20261016/`: baseline and fixed
sixteen-row probes, rejected intermediate implementation, frozen measured source,
exact-copy oracle binding, CPU/CUDA logs, whole-call fallback guard and core gates.
An initial probe passed RMSprop(weight_decay=0), an unsupported keyword on this
frontend, and failed loudly; the corrected probe uses the default zero decay
without that keyword. This change does not claim that separate keyword gap is fixed.

## 2026-10-16: native RMSprop coupled weight decay

Status: verified for the stated FP32/FP64 CPU/CUDA contracts. Baseline:
`8333abbbb59776acde9fd622293353193a45c7eb`. Implementation: the commit containing
this follow-up; its exact seven-owner Git binding is retained with the raw evidence.
Owner: native optimizer / Torch compatibility maintainers. Review when: RMSprop
arithmetic, group overrides, gradient ownership or checkpoint metadata changes.

This closes the weight-decay keyword gap recorded above. The baseline rejected
`weight_decay=0` as an unknown keyword and silently ignored decay supplied only
through a parameter group. The final baseline selection had fifteen failing CPU
contracts, including both FP32/FP64 group-only numerical failures. The update now
adds the L2 term to the gradient before accumulating the square average and
updating the parameter, in the existing native RMSprop owner. Published `.grad`
objects and bytes remain intact. The Torch state export records the actual
global/group decay, allowing a saved nonzero-decay optimizer to resume in an
optimizer initially constructed with zero decay. Negative and NaN global decay
raise ValueError, matching the independent binary Torch controls.

Native positional argument order remains `(params, lr, eps, alpha, weight_decay)`;
the first four arguments retain their old order. This is a keyword-decay acceptance,
not a claim that every Torch RMSprop option or positional convention is supported.
New contracts cover three successive updates and square-average states, global
and per-group overrides including zero, absent versus zero gradients, exact
checkpoint continuation, gradient identity, dtype and actual backend location.
The portable optimizer contracts share one fallback guard rather than copying it.

| Final scope | CPU | CUDA configuration |
| --- | --- | --- |
| Six neighboring Torch optimizer/gradient files | 73 passed / 33 CUDA skips | 106 passed / 0 skipped |
| Native optimizer core plus explicit RMSprop device contracts | 39 passed / 7 skips | 46 passed / 0 skipped |
| Independent Torch exact copies of both portable contract files | 35 passed / 33 CUDA skips | 68 passed / 0 skipped |

The CUDA configuration includes explicitly selected CPU controls. The new RMSprop
contracts contain fourteen public CUDA cases and six native CUDA cases, all
checking actual allocation location. Both final CUDA invocations were wrapped
in fallback-error scopes, with counters zero before and after. The initial native
unittest-only invocation had 35 successful assertions but exited 1 because its
internal device loops supplied no statically identifiable accelerator cases.
The maintained native regression now parameterizes the device explicitly and
checks actual parameter/state storage; the final 46-case gate passes unchanged.
A separate first structure run rejected the intentionally extended native
signature. Its exact signature contract now includes the appended default-zero
argument while preserving legacy order; the gate is not bypassed.

Core: 322 passed / 42 skips / one existing xfail. Final structure, layout and
manifest results are recorded in the delivery packet. New test/support files
pass Ruff; the edited production owners retain their baseline diagnostics with
no additions. ACL and ROCm were not executed in this follow-up; this does not
grant those devices or full optimizer ABI support.

Reproduce by adding `compat/tests/torch/test_rmsprop_weight_decay.py` to the
five-file command above and running native
`tests/optim/test_optim_core.py tests/optim/test_rmsprop_device_contract.py`
separately with the Torch shim disabled. Use isolated CPU/CUDA unit caches and
an independent Torch interpreter for exact copies of the portable files and
`_optimizer_contract_support.py`.

Raw logs, rejected intermediate validation, source snapshots, exact-copy bindings,
whole-call CUDA guards, native binary identities and the delivery packet are
unversioned under `$JITTOR_LAB_ROOT/_state/rmsprop-weight-decay/20261016/`.

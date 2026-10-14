# Optimizer checkpoint and deterministic Accelerate continuation

Status: CPU and CUDA single-process numerical validation complete; complete
Accelerate state restoration remains incomplete because CPU RNG stream positions
are not restored. Multi-rank FSDP2 and ACL are not accepted by this report.

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

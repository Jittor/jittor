# Optional Torch library adapters

Install this distribution only when the Jittor Torch frontend needs its
Transformers, TorchMetrics or vLLM integration:

```bash
python -m pip install ./adapters
```

The distribution is `jittor-torch-adapters`; its Python package is
`jittor_adapters`. It is independent of the compatibility package and registers
through the `jittor.module_patches` entry-point group. Registration does not
import these third-party libraries. The normal module loader applies each patch
after its owning module executes; `builtins.__import__` is never replaced.

Transformers and TorchMetrics versions are explicit:

| Adapter | Versions | Behavior |
| --- | --- | --- |
| Transformers | 4.56.2, 5.5.3 | Its native `torch_npu` availability probe returns false in the Jittor frontend; an installed Jittor `torchvision` facade is reported as available so public image/video `AutoProcessor` paths can initialize. |
| TorchMetrics | 1.7.4 | Private bounded bincount, concatenation and safe-divide helpers retain the established Jittor behavior. |

These versions come from the maintained integration evidence and local
distribution metadata. An unrecognized version, missing version, or changed
required helper raises `UnsupportedAdapterVersion` when the module patch is
applied. Supporting another version requires checking those contracts and
updating this table and the adapter's version set together.

The generic Torch frontend does not import or depend on this package. Missing
adapter entry points appear as `unavailable` in
`jittor.compat.module_patcher.last_module_patch_report()`, not as applied
patches. Patches use the shared method-patching API and its runtime lifetime
ledger, so supported cleanup preserves external replacements and reports
ownership conflicts.

Run the light source tests without initializing Jittor:

```bash
python adapters/tests/test_adapters.py -q
```

They import controlled fake packages through the real finder, execute the NPU
guard, recognize a deployed torchvision facade, reject an unsupported version,
check missing-adapter reports and retain TorchMetrics' original fallback. They
do not establish new hardware or performance claims.

## Explicit Trainer dataloader checkpoints

`jittor_adapters.trainer_stateful.stateful_trainer_class()` returns an explicit
Trainer subclass. Importing it does not register a patch or change the default
Trainer. This entry point is separately validated for Transformers **4.56.2**
and Accelerate **1.10.1**; other versions fail closed. Accelerate also requires
installed `torchdata` metadata (the acceptance environments use **0.11.0**).
Install that dependency without replacing the environment's Torch runtime.

```python
from jittor_adapters.trainer_stateful import stateful_trainer_class

StatefulTrainer = stateful_trainer_class()  # optionally pass a custom Trainer class
trainer = StatefulTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
    stateful_dataset_id="fixed-dataset-and-transform-config-v1",
)
trainer.train()
# In a new process, recreate the same configuration and immutable dataset:
trainer.train(resume_from_checkpoint="output/checkpoint-5")
```

The subclass enables `accelerator.dataloader_config.use_stateful_dataloader`,
saves the **prepared** loader's public state in `stateful_dataloader.pt` beside
each Trainer checkpoint, and restores it before the next training iterator.
The prepared state accounts for Accelerate's one-batch lookahead. Explicit
loader/sampler generators are saved too. `ignore_data_skip=True` prevents
Trainer from advancing the restored cursor again; the callback corrects its
public epoch fraction before saving. Missing state, changed configuration,
inconsistent Trainer metadata, or unsupported versions raise an error.
Checkpoint files contain Python/NumPy state; load trusted checkpoints only.

For caller-owned loading, pass `stateful_dataloader_factory=lambda trainer: ...`.
It must return an **unprepared** `torchdata.stateful_dataloader.StatefulDataLoader`
using `trainer.train_dataset` and the intended collator, batches, workers and
generators. The adapter passes it to the public `accelerator.prepare` method.
Native PyTorch/torchdata requires checkpointable sampler/batch-sampler and
dataset transform state: the acceptance fixture uses torchdata's `RandomSampler`
and `BatchSampler`, and a dataset exposing its Python/NumPy RNG state through
`state_dict`/`load_state_dict`. An arbitrary native dataset or ordinary random
sampler does not acquire a complete state contract by enabling this flag.

The current supported Trainer schedule is deliberately bounded: **one process,
one device, an explicit positive `max_steps` within the first epoch, complete
gradient-accumulation windows, and a checkpoint strictly before epoch end**.
Multiple epochs, accumulation tails, HPO trials, automatic batch-size search,
model-only checkpoints, distributed training and Seq2SeqTrainer are outside
this entry point's acceptance. The factory and dataset identity are the caller's
contract: use the same data, transform code and sampling configuration on resume.

Jittor's explicit StatefulDataLoader snapshots a finite epoch batch plan, with
a default limit of **100000 actual sample indices**; direct users can choose
`max_stateful_indices`. Ordinary DataLoader keeps streaming without this history.
Jittor workers remain threads: the supported worker RNG scope is Python `random`
and NumPy's legacy global RNG under the loader's lock. Custom RNG objects,
worker Torch RNG, mutable/stateful datasets, out-of-order loading and IterableDataset
are not covered. This integration is a Transformers checkpoint concern and lives
in the removable adapters package; generic loader state lives in Torch compat.

Run the fast host contracts with:

```bash
PYTHONPATH=adapters python -m pytest -q adapters/tests/test_trainer_stateful.py
```

Real-library acceptance and the exact supported device scope are recorded in
[the Stateful Trainer acceptance report](../refactor-wip/results/2026-10-17-trainer-stateful-resume.md).
Revalidate when Trainer/Accelerate/torchdata versions, prepared-loader semantics,
or the compat public runtime hooks (`owned_runtime_hook`, `transaction`,
`module_patcher`) change. This entry point makes no performance claim.

## vLLM

The vLLM implementation lives in this repository at `jittor_adapters/vllm` and
ships in this existing distribution. Neither core nor jittor-torch bundles it.
The named `jittor_vllm` entry point exports
`register(register_module_patch) -> None`. The optional Torch stage selects
only that name, leaving unrelated adapters unimported. Missing installation is
reported as unavailable without breaking the frontend.

There are two phases: a before-import finder supplies vLLM's compiled-extension
surface, and after-import callbacks patch its fully defined layer classes.
Both preserve the shared transaction/runtime-hook ownership and rollback rules.
Operator mathematics delegates to public Jittor primitives on the same Var/Op
graph. Importing `jittor_adapters.vllm` alone activates nothing. Installation
does not pull vLLM or PyTorch, and no separate adapter repository is required.

Historical CUDA evidence refers to vLLM checkout
`51a99565c398c8320de8131e07731c75c52eb87c`, whose locally generated 0.11.0
metadata was not authoritative. Distinct Ascend notes refer to vLLM 0.20.2
and an external platform/worker package whose sources are unavailable here.
No `vllm.platform_plugins` entry point or new hardware/version-support claim
is made by this extraction. The Transformers/TorchMetrics version rejection
policy above does not imply a newly validated vLLM version matrix.

Run the controlled lifecycle tests without JIT from the repository root:

```bash
JITTOR_VLLM_HOST_ONLY=1 PYTHONPATH=adapters python -m pytest -q \
  adapters/tests/vllm/test_plugin_lifecycle.py \
  adapters/tests/vllm/test_arming_ownership.py \
  adapters/tests/vllm/test_runtime_ownership.py
```

In an extracted adapters sdist, set `JITTOR_COMPAT_SOURCE` to the separately
installed compatibility source directory. Numerical layer/attention tests
remain under `tests/vllm` for a working backend; lifecycle and wheel checks
do not establish CUDA/NPU serving correctness or performance.

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
| Transformers | 4.56.2, 5.5.3 | Its native `torch_npu` availability probe returns false in the Jittor frontend. |
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
guard, reject an unsupported version, check missing-adapter reports and retain
TorchMetrics' original fallback. They do not establish new hardware or
performance claims.

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

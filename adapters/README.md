# Optional Torch library adapters

Install this distribution only when the Jittor Torch frontend needs its
Transformers or TorchMetrics integration:

```bash
python -m pip install ./adapters
```

The distribution is `jittor-torch-adapters`; its Python package is
`jittor_adapters`. It is independent of the compatibility package and registers
through the `jittor.module_patches` entry-point group. Registration does not
import either third-party library. The normal module loader applies each patch
after its owning module executes; `builtins.__import__` is never replaced.

Supported versions are explicit:

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

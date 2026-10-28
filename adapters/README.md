# Optional Torch library adapters

Install this distribution only when the Jittor Torch frontend needs its
Transformers, TorchMetrics, TensorDict, verl or vLLM integration:

```bash
python -m pip install ./adapters
```

The distribution is `jittor-torch-adapters`; its Python package is
`jittor_adapters`. It is independent of the compatibility package and registers
through the `jittor.module_patches` entry-point group. Registration does not
import these third-party libraries. The normal module loader applies each patch
after its owning module executes; `builtins.__import__` is never replaced.

Third-party interface versions are explicit:

| Adapter | Versions | Behavior |
| --- | --- | --- |
| Transformers | 4.56.2, 5.5.3, 5.9.0 | Its native `torch_npu` availability probe returns false in the Jittor frontend. |
| TorchMetrics | 1.7.4 | Private bounded bincount, concatenation and safe-divide helpers retain the established Jittor behavior. |
| TensorDict | 0.10.0, selected source-pinned helpers | Private accepted-leaf registration, validators and consolidation leaf dispatch preserve jagged objects; private packed reconstruction uses public construction with upstream storage-alias/detach semantics. |
| verl | 0.9.1, source-pinned | Selected private TensorDict helper predicates accept the actual Jittor jagged class; private NPU helper discovery uses upstream's strict availability probe, not portable namespace presence. |

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
python adapters/jittor_adapters/tensordict/tests/test_adapter.py -q
python adapters/jittor_adapters/verl/tests/test_adapter.py -q
```

They import controlled fake packages through the real finder, execute the NPU
guard, reject an unsupported version, check missing-adapter reports and retain
TorchMetrics' original fallback. They do not establish new hardware or
performance claims.

For Transformers 5.9.0 Qwen3.5, adapter retirement is partial: both upstream
gated-delta fallback sources are still checked, but the chunk function is no
longer replaced. Only the functional recurrent replacement remains installed;
the old functional chunk helper is retained for historical controls, not armed.
The normal loader, import guards and patch registration remain unchanged.
This route change affects every device on which the adapter is activated.
NPU, cached recurrence, stored callables on actual model instances and complete
PPO must be revalidated separately; the stub contracts establish none of them.
Models created before the changed import/installation are not repaired by this
change. No all-original-mathematics or new L1-L5 acceptance is claimed.

TensorDict's adapter consumes the public, allocation-free
`jittor.compat.torch.nested_tensor_type()` accessor. Registration targets the
fully imported TensorDict root (its version is only published near the end of
import). The optional installer only registers the adapter; the pre-existing
TensorDict index-compatibility installer retains its historical import behavior.
Accepted-class aliases and validators share the module-patcher rollback ledger.
Consolidation's private helper and leaf predicate also use source guards and
the same ownership ledger. Packed reconstruction uses the public
`jittor.compat.torch.nested_tensor_rebuild()` helper with detached values and
the template's canonical axis. For the pinned TensorDict 0.10/Torch extraction
contract, it copies the current template metadata-key identity only when the
new lengths are present; explicit lengths=None does not copy that identity.
The compatibility owner controls this registry binding, and existing nested
objects keep their construction-time symbols. Revalidate on either public
nested helper changing. The CUDA-only regression is
`adapters/jittor_adapters/tensordict/tests/test_jagged_rebuild_cuda.py`;
its four outer cases also cover chained rebinding and holey-to-None rebuild.
Its source-pinned consolidated transport rewrite replaces only
the exact private jagged import/call; upstream storage/metadata logic and live
globals are preserved. That serialized format reconstructs jagged dimension
one; no new multi-axis serialization support is claimed.
It does not make jagged objects subclasses of `torch.Tensor`, add
missing nested operations, claim memmap support, or validate verl's own private
tensor-type assertions. Host lifecycle tests are not device evidence; the real
CPU/CUDA regression is `compat/tests/torch/test_tensordict_consolidate.py`.

The separate verl adapter pins both the full upstream source file and selected
helper source hashes. Its limited AST transformation expands only documented
`isinstance` predicates; it does not change `torch.Tensor`, builtins, payloads,
mathematics, defaults, or the original functions' dynamic globals. A changed
version, missing source, hash mismatch or unexpected predicate shape fails
explicitly. Unsupported helpers, Ray, offload, rollout and full training are
not covered by the adapter's host tests. Run its source-fixture tests with
Python 3.10 or newer; an older interpreter's skips are not verification.

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

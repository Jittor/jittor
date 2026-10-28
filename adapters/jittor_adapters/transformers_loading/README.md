# Explicit materialized Transformers loading

Status: opt-in; restricted tiny Qwen3 CPU loading and single-rank NPU worker
validation passed on 2026-09-24. General meta support remains unavailable.
Owner: Jittor Torch compatibility maintainers.
Review when: Transformers initialization sources, the public module-patcher
lifecycle, Torch initialization functions, or factory/device semantics change.

| Breakpoint | Owner | Selected behavior |
| --- | --- | --- |
| HF 5.9 assumes construction on meta, replaces nonpersistent buffers with empty storage, and skips already-initialized modules | Version-pinned downstream loading policy | Use HF's own construction-only `no_init_weights()` with real CPU storage; retain upstream checkpoint loading and subsequent missing-key initialization |

Enable explicitly with `JITTOR_TRANSFORMERS_MATERIALIZED_INIT=1` before adapter
registration. Disabled by default. This is **not meta support** and does not
alter Jittor/Torch device properties, tensors, model mathematics, or buffers.
The entire model needs CPU storage during construction; native constructors
may still allocate temporary weights and consume random numbers. Do not infer
memory savings or identical missing-random-weight RNG trajectories from this
policy. The controlled worker uses a complete shared checkpoint.

Only Transformers 5.9.0 is admitted, with exact source hashes for its selector
and `no_init_weights` dependency. Quantization, DeepSpeed, remote model code,
hub kernels, changed context factories/layouts, and overrides entering the
selector via `super()` are rejected. A custom loader overriding the selector
without ever calling it bypasses this adapter and is **outside its scope**;
callers must verify the active selector marker for their actual model class.
Multi-rank initialization, offload/TP and arbitrary custom loaders are unverified.
Construction must be serialized within a process: the upstream no-init context
temporarily replaces initialization callables process-wide, not thread-locally.

The adapter changes only the third-party classmethod using the shared public
patch lifecycle. It never imports Transformers during registration. The
original first three construction contexts and final safe-creation context
are preserved; the meta context becomes the upstream no-init context followed
by `torch.device("cpu")`. All construction contexts exit before the upstream
load/finalize phase. Production registration is routed by the parent adapter.

Host contracts (no Jittor import or JIT):

```bash
python adapters/jittor_adapters/transformers_loading/tests/test_adapter.py -v
```

These tests execute exact pinned HF excerpts with fake host leaves and the
actual compatibility initialization guard and rollback mechanism. They are
not evidence of CPU tensor or NPU model correctness. Real validation must
check all checkpoint parameters, both persistent/nonpersistent buffers,
device placement, full gradients and zero candidate backend fallback.

Real validation: CPU shared-checkpoint loading matched native Torch exactly
for all 15 critic parameters and both nonpersistent RoPE buffers. Single-rank
NPU critic/actor TrainingWorkers completed three SGD steps with every parameter
gradient/update captured; 116/115 snapshots matched torch_npu within 1e-4,
maximum absolute errors 5.96e-8/1.19e-7 and candidate fallback 0. These are
controlled FP32 tiny-model tests, not full trainers, memory or speed acceptance.
Native Torch jagged conversion warned of CPU fallback; its count is unknown.

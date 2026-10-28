# Verification Results

This section holds **reproducible maintainer verification conclusions**: a concrete
phenomenon, its root cause, the fix, and the numbers measured on real devices after the
fix. It records "what we checked and how it turned out", not project history and not a
tutorial—consult it when you want to get a model or backend running, or to check the
conclusion for a known issue.

Every conclusion states its status, date, baseline commit, verification scope, maintainer
and review conditions up front. Stale conclusions are deleted or archived and are not
maintained in several files at once; condensed reports keep only the conclusion, final
numbers, reproduction commands and open items, with the step-by-step record left in Git
history.

```{toctree}
:maxdepth: 1

2026-10-25-recent-pr-integration.en
2026-10-24-datatrove-cpu-compatibility.en
2026-10-16-diffusers-maintained-cpu-nightly.en
2026-10-16-h3-upstream-gradient-components.en
2026-10-16-native-alias-lifecycle.en
2026-10-16-default-torch-storage-views.en
2026-10-16-sgd-cuda-opmath.en
2026-10-16-sgd-checkpoint-continuation.en
2026-10-16-sgd-execution-options.en
2026-10-16-module-load-independent-storage.en
2026-10-16-h3-sdpa-gradient-rounding.en
2026-10-16-h3-default-sdpa-forward.en
2026-10-16-metrics-tensordict-maintained-cpu-nightly.en
2026-10-16-openmmlab-maintained-cpu-nightly.en
2026-10-16-transformers-maintained-cpu-nightly.en
2026-10-16-diffusers-official-installation.en
2026-10-16-metrics-tensordict-installed-source.en
2026-10-16-verl-maintained-cpu-nightly.en
2026-10-16-recorded-grad-fn-freeze.en
2026-10-16-transformers-main-source-training.en
2026-10-15-maintained-ecosystem-nightly.en
2026-10-15-openmmlab-installed-source-training.en
2026-10-15-transformers-installed-source-training.en
2026-10-15-verl-installed-source-training.en
2026-10-15-size-factory-leaves.en
2026-10-15-tensor-rng-inplace.en
2026-10-15-swift-trainer-installed-source.en
2026-10-15-peft-installed-source-training.en
2026-10-15-nn-init-trainability-copy.en
2026-10-15-nn-init-generator.en
2026-10-15-release-installed-source-proof.en
2026-08-28-ascend-910b-validation.en
2026-08-30-qwen3-ascend-training.en
2026-08-31-vllm-ascend-jittor-bootstrap.en
2026-09-02-verl-ascend-core-algorithms.en
2026-09-04-import-jittor-cost-attribution.en
2026-09-12-minimax-h3-torch-compat.en
2026-09-12-cuda-metaop-launch-index-scalar.en
2026-09-12-batched-linear-graph-nodes.en
2026-09-13-host-path-per-node-cost.en
2026-09-14-autocast-conv-mixed-dtype.en
2026-09-14-exit-heap-corruption.en
2026-09-14-jittor-vs-pytorch.en
2026-09-14-vllm-omni-h3-enablement.en
2026-09-19-torch-compat-runbook-verification.en
2026-09-21-flowey-merge-verification.en
2026-09-22-complex-svd-grad-and-complex128.en
2026-09-22-qwen3-ascend310p-fp16.en
2026-09-22-sgd-dampening-and-native-vmap-gaps.en
2026-09-24-torch-compat-real-models.en
2026-09-25-profiling-tools.en
2026-10-05-high-rank-and-downstream-core-gaps.en
2026-10-09-downstream-p1-allocator-protocol-transpose.en
2026-10-12-qwen3-ascend310p-performance.en
2026-10-13-qwen3-8b-ascend310p-performance.en
2026-10-14-benchmark-coverage.en
2026-10-14-downstream-coverage-audit.en
2026-10-14-optimizer-checkpoint-continuation.en
2026-10-14-checkpoint-rng-fsdp2-continuation.en
2026-10-14-amp-bfloat16-gradscaler.en
2026-10-15-distribution-constraint-publication.en
2026-10-15-training-registry-denominators.en
2026-10-14-cuda-sdpa-mask-coverage.en
2026-10-15-sdpa-opmath.en
2026-10-15-like-factory-leaves.en
2026-10-15-diffusers-component-scorecards.en
2026-10-15-diffusers-installed-source.en
2026-10-15-mpi-bfloat16-process-groups.en
2026-10-15-cpu-bfloat16-accumulators.en
2026-10-15-h3-silu-contracts.en
2026-10-15-accelerate-protocol-scorecards.en
2026-10-15-deepspeed-paired-protocol.en
2026-10-14-graph-replay-retain-budget.en
2026-10-14-downstream-oracle-scorecards.en
2026-10-15-cuda13-toolchain-packaging.en
2026-10-15-bf16-fsdp2-checkpoint-continuation.en
2026-10-15-metric-state-and-tensordict-oracles.en
2026-10-15-metrics-tensordict-scorecards.en
2026-10-15-vllm-component-scorecards.en
2026-10-15-checkpoint-state-options.en
2026-10-15-integration-nccl-coverage.en
2026-10-15-upstream-pr-disposition.en
2026-10-15-verl-registered-component-oracles.en
2026-10-15-transformers-model-family-matrix.en
2026-10-15-native-sharded-checkpoint.en
2026-10-15-native-multinode-control-plane.en
2026-10-15-native-multihost-runner.en
2026-10-15-accelerate-fresh-process.en
2026-10-15-cuda-seed-without-devices.en
2026-10-15-h3-autograd-boundary.en
2026-10-16-omni-installed-source-diagnostics.en
2026-10-15-longformer-32768-context-oracles.en
2026-10-15-deepspeed-engine-prerequisites.en
2026-10-15-deepspeed-zero-engine.en
2026-10-15-native-zero-optimizer.en
2026-10-15-native-tensor-parallel.en
2026-10-15-native-fsdp.en
2026-10-15-bfloat16-linear-bias-accumulation.en
2026-10-15-native-tensor-parallel-adamw-bf16.en
2026-10-15-fsdp-compat-lifecycle.en
2026-10-15-openmmlab-training-oracles.en
2026-10-15-tensor-detach-and-factory-leaf.en
2026-10-15-cuda-extension-philox.en
2026-10-15-flash-attention-training-oracles.en
2026-10-16-flash-attention-installed-source-training.en
2026-10-16-accelerate-installed-source-protocols.en
2026-10-16-accelerate-maintained-cpu-nightly.en
2026-10-15-deepspeed-bfloat16-multirank.en
2026-10-15-deepspeed-maintained-gate.en
2026-10-15-tanh-fsdp-bfloat16.en
2026-10-15-trellis-model-validation.en
2026-10-15-pytree-transformers-513.en
2026-10-15-rng-native-generator-contract.en
2026-10-15-default-generator-state-contract.en
2026-10-15-rmsnorm-triton-alias.en
```

## Index by Theme

| Result | Status | Date |
| --- | --- | --- |
| [Ascend 910B3: cold-cache startup, ACL execution and NPU gate](2026-08-28-ascend-910b-validation.en.md) | NPU gate 397 passed / 9 skipped; the capability boundaries behind the skips remain open | 2026-08-28, reviewed 2026-09-01 |
| [Qwen3-0.6B training on Ascend 910B3](2026-08-30-qwen3-ascend-training.en.md) | FP32 training and BF16 single-step alignment accepted; the exact BF16 path is about 6% slower and the training trajectory is not yet accepted | 2026-08-30, 2026-09-02 |
| [vLLM on Ascend 910B3 through Jittor](2026-08-31-vllm-ascend-jittor-bootstrap.en.md) | Single-request Qwen3-0.6B is correct and 6.8% faster; the external NPU plugin is not in the repo and multi-request was not tested | 2026-08-31, 2026-09-02 |
| [verl core algorithms on Ascend 910B3](2026-09-02-verl-ascend-core-algorithms.en.md) | Six loss kinds and gradients bitwise identical; five are 1.25–1.75x slower, NPU end-to-end PPO not run | 2026-09-02 |
| [Cost attribution for a warm-cache `import jittor`](2026-09-04-import-jittor-cost-attribution.en.md) | Attribution accepted, warm cache now below 1 s; cold cache still compiles the core during import | 2026-09-04 |
| [MiniMax-H3 runs under the Torch compatibility layer](2026-09-12-minimax-h3-torch-compat.en.md) | End-to-end pass, parity speed | 2026-09-12, re-verified 2026-09-14 |
| [autocast requested a convolution with mixed dtypes](2026-09-14-autocast-conv-mixed-dtype.en.md) | Fixed, verified on a real CUDA device | 2026-09-14 |
| [Exit-time "corrupted double-linked list"](2026-09-14-exit-heap-corruption.en.md) | Fixed, verified over multiple real runs | 2026-09-14 |
| [jittor + torch-compat + vLLM-Omni: shim defects exposed by H3 enablement](2026-09-14-vllm-omni-h3-enablement.en.md) | Single-card and TP2 requests end-to-end correct (710.7 s → 39.6 s), deployment-side video noise fixed; reference VAE decoding still 1.33x slower than PyTorch | 2026-09-14 to 2026-09-22 |
| [CUDA meta-ops: launch configuration, strided indexing and scalar-constant fusion](2026-09-12-cuda-metaop-launch-index-scalar.en.md) | Partially complete, measured on H20 | 2026-09-12 |
| [Host-bound steps: removing per-operator Python overhead from graph construction](2026-09-13-host-path-per-node-cost.en.md) | Python path complete | 2026-09-13 |
| [The two flatten nodes of batched Linear](2026-09-12-batched-linear-graph-nodes.en.md) | Landed, measured on H20; comparison conclusion not attached | 2026-09-12, archived 2026-09-23 |
| [Jittor vs real PyTorch 2.9.1: where the gap stands now](2026-09-14-jittor-vs-pytorch.en.md) | Snapshot: 6 wins, 5 ties, 1 loss against eager; 2 wins, 2 ties, 5 losses against `torch.compile` | 2026-09-14 |
| [Torch compatibility layer on real models vs PyTorch: gap table, memory and performance fixes](2026-09-24-torch-compat-real-models.en.md) | All 11 pass, geometric mean 1.26x, process peak memory 0.87–1.45x of PyTorch; the remaining gap is on the host side | 2026-09-24 |
| [Performance/memory profiling tools: audit and rewrite](2026-09-25-profiling-tools.en.md) | Implemented, verified on RTX 4090; not merged | 2026-09-25 |
| [Tensor rank limit 10 → 32 and core gaps in downstream evaluation](2026-10-05-high-rank-and-downstream-core-gaps.en.md) | Fixed, verified on CPU and RTX 4090 | 2026-10-05 |
| [Downstream P1: allocator coexistence limit, numerical protocol and transpose's runtime permutation](2026-10-09-downstream-p1-allocator-protocol-transpose.en.md) | Fixed, verified on CPU; transpose/protocol additionally verified on RTX 4090 | 2026-10-09 |
| [Qwen3-0.6B: 310P3 FP16 inference performance](2026-10-12-qwen3-ascend310p-performance.en.md) | Under identical pre-converted weights, 502/1998-token decode is about 6%/26% faster than native torch_npu; token-by-token identical, zero Jittor fallback | 2026-10-12 |
| [Qwen3-8B: 310P3 FP16 inference performance](2026-10-13-qwen3-8b-ascend310p-performance.en.md) | 502-token decode roughly at parity, 1998-token 20.1% faster than native torch_npu; token-by-token identical, zero Jittor fallback | 2026-10-13 |
| [Benchmark coverage and comparison protocol completed](2026-10-14-benchmark-coverage.en.md) | Convolution CPU/CUDA, Accelerate CPU and model CPU harness verified; other device and scenario gaps remain | 2026-10-14 |
| [Downstream registry, gates and coverage audit](2026-10-14-downstream-coverage-audit.en.md) | 13 runbooks and five upstream registries audited; no new device verification level granted | 2026-10-14 |
| [Optimizer FQN checkpoint and deterministic continuation](2026-10-14-optimizer-checkpoint-continuation.en.md) | Single-process deterministic numerical continuation verified; RNG/dual-card acceptance in a later report, ACL pending | 2026-10-14 |
| [Full RNG and FSDP2 FULL checkpoint continuation](2026-10-14-checkpoint-rng-fsdp2-continuation.en.md) | CPU/CUDA random streams and dual-card FULL continuation verified; sharded I/O and ACL pending | 2026-10-14 |
| [bf16 autocast and GradScaler boundaries](2026-10-14-amp-bfloat16-gradscaler.en.md) | CPU/CUDA actual dtype and boundary comparisons pass; fidelity remains APPROXIMATE | 2026-10-14 |
| [Distribution-constraint namespace publication](2026-10-15-distribution-constraint-publication.en.md) | Native attribute and factory publication fixed; CPU/CUDA boundary-check comparisons pass | 2026-10-15 |
| [Training registry denominators completed](2026-10-15-training-registry-denominators.en.md) | ms-swift/verl training axes and vLLM/Omni denominators added, SCM wheel source identity verified; enumeration does not grant numerical coverage | 2026-10-15 |
| [CUDA SDPA mask coverage](2026-10-14-cuda-sdpa-mask-coverage.en.md) | 180 real CUDA scenarios compared; fused path extended, boundaries kept explicit | 2026-10-14 |
| [Graph replay retention budget](2026-10-14-graph-replay-retain-budget.en.md) | Dynamic budget constrains cached replays; default budget retained, measured on CPU/CUDA | 2026-10-14 |
| [Downstream independent oracles and training scorecards](2026-10-14-downstream-oracle-scorecards.en.md) | Representative CPU/CUDA scenarios for four libraries verified; LoRA/component scope limited, no L4/L5 granted | 2026-10-14 |
| [CUDA 13 toolchain and uv installation](2026-10-15-cuda13-toolchain-packaging.en.md) | Linux/SM89 toolchain, CUB, matmul and convolution measured; Cu122 regression passes | 2026-10-15 |
| [Dual-card bf16 FSDP2 continuation](2026-10-15-bf16-fsdp2-checkpoint-continuation.en.md) | bf16 real computation and exact FULL/RNG restore verified; cross-framework rounding differences remain | 2026-10-15 |
| [TorchMetrics state and TensorDict comparison](2026-10-15-metric-state-and-tensordict-oracles.en.md) | Both libraries pass CPU/CUDA full gradients; TensorDict three-step training passes, component scope limited | 2026-10-15 |
| [vLLM component CPU/CUDA scorecards](2026-10-15-vllm-component-scorecards.en.md) | Three parameter-independent components at L2, each 3/40 CustomOp; models 0/376, whole-engine blocking listed separately | 2026-10-15 |
| [TorchMetrics/TensorDict unified scorecards](2026-10-15-metrics-tensordict-scorecards.en.md) | New CPU/CUDA raw evidence accepts 4; bounded top-level APIs each 2/105 L2 and 2/60 L3 | 2026-10-15 |
| [Checkpoint options and submodule contracts](2026-10-15-checkpoint-state-options.en.md) | CPU/CUDA single-process options and real Torch checkpoint interop verified; multi-rank options and ACL pending | 2026-10-15 |
| [Integration regression and NCCL defect closure](2026-10-15-integration-nccl-coverage.en.md) | NCCL 2.18.3 dual-card exit condition passes, KI-DIST-002/003 closed; targeted scope only | 2026-10-15 |
| [Upstream PR disposition and NumPy ABI](2026-10-15-upstream-pr-disposition.en.md) | Ten PRs checked; NumPy 1/2 true-copy and shape regression pass, Windows needs real hardware | 2026-10-15 |
| [VERL registered algorithm component comparison](2026-10-15-verl-registered-component-oracles.en.md) | GAE/vanilla PPO and k3 component CPU/CUDA full gradients and three-step trajectories pass; rollout/Ray/Hydra excluded | 2026-10-15 |
| [Transformers model-family matrix](2026-10-15-transformers-model-family-matrix.en.md) | Nine CPU/CUDA L3 scenarios; 8/374 model families, 2/46 tasks, including CLIP/Mixtral/2048-token RoPE | 2026-10-15 |
| [DeepSpeed Engine prerequisite ownership and gradient contracts](2026-10-15-deepspeed-engine-prerequisites.en.md) | Native module registration and backward-callback verification; real Engine acceptance in a dedicated report | 2026-10-15 |
| [DeepSpeed single-rank FP32 training and recovery](2026-10-15-deepspeed-zero-engine.en.md) | 12 configurations each on CPU and CUDA pass; dual-rank bf16 not yet accepted | 2026-10-15 |
| [Native ZeRO1 state sharding](2026-10-15-native-zero-optimizer.en.md) | CPU and dual-card NCCL training/recovery verified; limited to Stage 1, excluding FSDP/TP or multinode | 2026-10-15 |
| [OpenMMLab real training comparison](2026-10-15-openmmlab-training-oracles.en.md) | Four L3 evidence points for MMEngine/MMCV on CPU and CUDA; limited to real train_step scenarios | 2026-10-15 |
| [Tensor detach and factory gradient boundaries](2026-10-15-tensor-detach-and-factory-leaf.en.md) | CPU/CUDA in-place detach, default gradient publication and bf16 new-leaf contracts pass; dual-card Engine still pending | 2026-10-15 |
| [CUDA extension Philox state](2026-10-15-cuda-extension-philox.en.md) | Real CUDA RNG replay and fused FlashAttention training regression; independent Torch comparison still pending | 2026-10-15 |
| [FlashAttention public training API comparison](2026-10-15-flash-attention-training-oracles.en.md) | Six-entry historical 48 trajectories and formal 12 CUDA scenarios pass; unified L3 record verified, missing CPU ladder keeps breadth at 0/7 | 2026-10-15 |
| [DeepSpeed dual-card BF16 training and recovery](2026-10-15-deepspeed-bfloat16-multirank.en.md) | ZeRO1/2 AdamW three-step training, full gradients and strict continuation verified; limited to small models and real dual-card CUDA | 2026-10-15 |
| [DeepSpeed maintained gate and CPU state rounding](2026-10-15-deepspeed-maintained-gate.en.md) | Dual-card BF16 four cells and 12 training/continuation items each on CPU/CUDA pass; fixed a CPU Adam state value-ordering difference | 2026-10-15 |
| [tanh backward and native FSDP BF16 comparison](2026-10-15-tanh-fsdp-bfloat16.en.md) | Fixed differentiation through low-precision saved outputs; fixed dual-card BF16 80 parameter gradients bitwise aligned, 368 continuation arrays bitwise identical | 2026-10-15 |
| [TRELLIS.2 real-model training comparison](2026-10-15-trellis-model-validation.en.md) | Three real models' CPU/CUDA three-step training, all gradients and independent perturbation pass; unified models breadth each 3/9 | 2026-10-15 |
| [Custom pytree and Transformers 5.13.0](2026-10-15-pytree-transformers-513.en.md) | Custom nodes and key-path fixes; fixed GPT2 CPU/CUDA three-step, all parameters and input gradients pass | 2026-10-15 |
| [Native RNG and full-width seed](2026-10-15-rng-native-generator-contract.en.md) | CPU/CUDA native streams, 64-bit seed, explicit generators and dual-device checkpoint recovery pass; no promise of bitwise-identical samples across frameworks | 2026-10-15 |
| [RMSNorm optional contracts and Triton in-place aliasing](2026-10-15-rmsnorm-triton-alias.en.md) | Preserved the module's original normalization contract, fixed real CUDA in-place writes being overwritten, and integrated the vLLM component gate audit | 2026-10-15 |
| [Native TP linear-layer training and recovery](2026-10-15-native-tensor-parallel.en.md) | FP32 evenly-sharded linear layers, full gradients and fixed-world continuation on CPU and dual-card NCCL verified; bf16 and multinode pending | 2026-10-15 |
| [Native FSDP sharding and fresh-process recovery](2026-10-15-native-fsdp.en.md) | Native dual-card training and strict recovery verified; bf16 small-gradient error and legacy compatibility failure listed separately | 2026-10-15 |
| [CUDA BF16 Linear bias rounding](2026-10-15-bfloat16-linear-bias-accumulation.en.md) | Real BF16 GEMM keeps FP32 accumulation after the bias; existing CPU path and second-order gradient regression verified | 2026-10-15 |
| [Native TP AdamW and BF16 continuation](2026-10-15-native-tensor-parallel-adamw-bf16.en.md) | Dual-card BF16 full gradients, state sharding and new-object recovery pass a fixed error budget; excludes multinode | 2026-10-15 |
| [FSDP compatibility layer optimizer lifecycle](2026-10-15-fsdp-compat-lifecycle.en.md) | Optimizer parameter replacement and native module checkpoint dispatch fixed, targeted re-verification on CPU/CUDA and dual-rank | 2026-10-15 |
| [jtrun fixed-node control plane](2026-10-15-native-multinode-control-plane.en.md) | Real CUDA communication, FSDP fresh-process continuation and failure exit with two agents on one machine pass; real multi-host pending | 2026-10-15 |
| [H3 upstream CUDA gradient boundaries](2026-10-15-h3-autograd-boundary.en.md) | Real Torch and the candidate side both lack full gradients in eval/train; traced to upstream inference kernels, no full-model L2/L3 granted | 2026-10-15 |
| [Omni full installed-source diagnostics](2026-10-16-omni-installed-source-diagnostics.en.md) | Real SCM wheel/full-install binding and generated-file substitution rejection; the original forward's four items exceed budget, full gradients on both sides still fail | 2026-10-16 |
| [Default generator state and device isolation](2026-10-15-default-generator-state-contract.en.md) | CPU state swap, private copies and real CUDA stream isolation pass; native frames and legacy snapshot reads retained | 2026-10-15 |
| [Per-host jtrun verification entry point](2026-10-15-native-multihost-runner.en.md) | Explicit source, interpreter, device and shared-artifact protocol; full execution with two agents on one machine, real multi-host still pending | 2026-10-15 |
| [Seeding without visible CUDA devices](2026-10-15-cuda-seed-without-devices.en.md) | CPU training can call the CUDA seeding interface; parameter conversion and real CUDA stream replay retained | 2026-10-15 |
| [Accelerate fresh-process continuation](2026-10-15-accelerate-fresh-process.en.md) | CPU/CUDA fresh-process recovery, full state and real CUDA FSDP2 pass; the upstream CPU dual-rank path is actually DDP, so the pair does not grant a pass | 2026-10-15 |
| [Accelerate full installed source and continuation](2026-10-16-accelerate-installed-source-protocols.en.md) | New CPU/CUDA 2,516 array pairs and full fresh-process recovery pass; common 2 accepted, 2 pairs blocked, real METADATA/RECORD bound to the original wheel | 2026-10-16 |
| [FlashAttention full installed-source training](2026-10-16-flash-attention-installed-source-training.en.md) | Original 12 CUDA scenarios, 48 workers and 936 numerical checks pass; the common 12 are blocked by the missing CPU counterpart, and the pinned local build artifact rejects synchronized-payload tampering | 2026-10-16 |

| [PEFT / Swift independent CPU nightly](2026-10-15-maintained-ecosystem-nightly.en.md) | Full dependency and source verification, both maintained gates executed locally; no GitHub scheduled run observed | 2026-10-15 |
| [OpenMMLab installed-source training](2026-10-15-openmmlab-installed-source-training.en.md) | MMEngine/MMCV four limited L3; 462 arrays, three input kinds and three-step real train_step, full installed-source binding | 2026-10-15 |

| [Transformers main-source re-verification](2026-10-16-transformers-main-source-training.en.md) | Independent 09e9 source, CPU/CUDA 160 workers total, 7854 array pairs, 20 limited L3; original B4 measurements not relabeled | 2026-10-16 |

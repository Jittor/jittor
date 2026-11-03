---
name: trl-torch-compat
description: Run the pinned official TRL text Trainers through the Torch frontend, with independent CPU/CUDA trajectories and explicit precision and checkpoint limits.
---

# TRL through the Torch frontend

- Status: bounded FP32 text training gate and measured CUDA Qwen3-0.6B FP16/BF16 LoRA parity; historical CUDA generation first-fault attribution remains open.
- Reviewed: 2026-10-26.
- Owner: Torch compatibility and downstream gate maintainers.
- Review when: TRL/library pins, native arithmetic, AMP, optimizer, checkpoint or gate producers change.

Follow [downstream-library-adaptation](../downstream-library-adaptation/SKILL.md).
TRL consumes public Torch APIs; no TRL adapter or replacement Trainer is needed.
Missing arithmetic belongs to native Jittor; Torch policy/signature/state changes
belong to `jittor.compat.torch`.

The maintained entry is `nox -s ecosystem_trl`. Prepare separate Python 3.11
candidate/oracle environments with [the pinned stack](../../../requirements/trl-compat.txt),
genuine Torch 2.8.0 in the oracle, and a clean official TRL checkout at
`46d09bd2408f17605409fb3ee8ba12705add7faa`. Set `JITTOR_CI_PYTHON`,
`REAL_TORCH_PYTHON`, `JITTOR_TRL_CHECKOUT` and `JITTOR_TEST_DEVICES=cpu` or `cuda`.
CUDA additionally requires one explicitly allocated `CUDA_VISIBLE_DEVICES` and
NVCC. Only `--output <fresh-external-path>` and `--jittor-home <exclusive-cache>`
may override the canonical three-step sixteen-method matrix.

Full provisioning, standalone commands and trace contracts are maintained once
in the [shared runbook](../torch-compat-repo-runbook/SKILL.md#maintained-sixteen-text-trainer-gate-and-extra-five-selectors).
The [CPU workflow](../../../.github/workflows/trl-compat-cpu.yml) runs that same
protocol nightly. Failures and incomplete trajectories fail the gate; a later
passing retry does not erase an earlier failure.

The denominator is **18 public Trainer exports** including optional diffusion
APIs, derived by `registry_inventory.py --repo trl` through `trl_inventory.py`.
The selected text subset has 16 methods; inventory/discovery grants no numerical
level or scorecard numerator. Each measurement retains its actual model, device,
precision, trained tensors, source fingerprints and independent oracle evidence.
Do not label a passed tiny FP32 matrix as whole-library support.

Read the text/LoRA results (historical validation record; see PR evidence)
and full PPO, repeatability and precision limits (historical validation record; see PR evidence)
before making a support claim. The full-checkpoint numerical and validation report (historical validation record; see PR evidence)
binds exact ten-step CUDA FP16/BF16 LoRA training and fresh-process continuation
for its pinned protocol. Historical CUDA generation first-fault attribution
remains open in the [issue ledger](../../manuals/known-issues.md). Checkpoint evidence is scenario
specific; the sixteen-method training gate alone does not establish resume,
distributed, diffusion, generation-engine, other-version or other-hardware support.

# Existing ecosystem regression tools

These tools are still called by the main repository's regression tests and CI.
They collect installation identity, registry inventories and numerical evidence
for diagnosing shared core/compat behavior. They are not a downstream support
matrix, and moving them here does not establish new hardware acceptance.

Library-specific adapters, version combinations, official tests, tutorials,
benchmarks and support tables belong to the corresponding
[JittorEcosystem repository](https://github.com/JittorEcosystem). Add new library
acceptance workflows there; reduce shared framework bugs to focused main-repo
regressions. The triage boundary is documented in
[downstream-library-adaptation](https://github.com/JittorEcosystem/jittor-dev-skills/blob/main/skills/downstream-library-adaptation/SKILL.md).

## Entry points

| Need | Entry |
| --- | --- |
| Inspect existing regression coverage and integration ownership | `audit_coverage.py audit --json` |
| Enumerate a pinned upstream registry without importing it | `registry_inventory.py --help` |
| Validate fresh source-bound evidence | `audit_coverage.py scorecard --help` |
| Run a registered case with two interpreters | `verify_repo.py --help` |
| Capture multi-step parameter and gradient trajectories | `training_probe.py --help` |
| Use existing component protocols | `collect_*scorecard*.py --help` |

`flash_attention/`, `trellis/` and `vllm_omni/` hold probes still referenced by
existing main-repo gates. They do not duplicate the installation or support
guides in the independent integration repositories. Files beginning with `_`
are helpers, not standalone commands.

The audit lists registered cases and dedicated gates separately. Its library
inventory is explicit and does not depend on agent skills or old reports being
present. `verified_level` stays empty until current execution evidence is
validated. See the [evidence protocol](../../docs/development/testing/downstream-scorecards.en.md)
for platform, precision, provenance and performance requirements.

## Multi-step diagnostic example

Use separate, already provisioned reference and candidate interpreters. Run the
reference first, then load its exact initial weights in the candidate. Set
`JITTOR_LAB_ROOT` to an external workspace and use a fresh output directory.

```bash
RUN="$JITTOR_LAB_ROOT/_state/ecosystem/<run>"
mkdir -p "$RUN"
"$ORACLE_PYTHON" tools/ecosystem/training_probe.py \
  --case peft_lora_llama --runtime torch --device cpu \
  --output "$RUN/reference.npz"
JITTOR_TORCH_SHIM=1 "$CANDIDATE_PYTHON" tools/ecosystem/training_probe.py \
  --case peft_lora_llama --runtime jittor --device cpu \
  --weights "$RUN/reference.weights.npz" --output "$RUN/candidate.npz"
```

Both interpreters need the same pinned downstream version; the reference must
be real PyTorch, and the candidate must load the intended Jittor and compat
build. Use a separate compilation cache for each concurrent process. Compare
all requested arrays and metadata, not just the final loss. This diagnostic
captures host snapshots and is not a performance benchmark or accelerator
acceptance run.

## Historical evidence

`_source_layout.py` resolves moved tool paths against the actual measured Git
commit. Old evidence retains old filenames and bytes; a current path never
substitutes today's source for a historical measurement. Ambiguous ownership
is rejected. Re-auditing historical evidence does not promote current support.

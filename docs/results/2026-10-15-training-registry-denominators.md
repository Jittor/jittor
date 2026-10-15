# Training registry denominators

- Status: source enumeration verified; enumeration grants no numerical or device
  support level.
- Date: 2026-10-15.
- Baseline: `9d9310e3c9a4af737702e12d2d6d47062fa015cd`; implementation is the
  commit containing this report.
- Owner: downstream compatibility maintainers.
- Review when: upstream registry layouts or SHAs, registry extraction or
  scorecard member accounting changes.

## Missing axes and extraction

The maintained extractor previously reported ms-swift model/tuner registrations
and verl advantage-estimator/policy-loss registrations. It omitted their trainer,
CLI, declared task/RLHF and distillation axes even when those were present in the
same clean source checkout. The new extraction reads tracked Python ASTs without
importing downstream libraries or a tensor runtime. Each inventory still records
the full upstream SHA, remote, version and hashes of the source files it uses.

Ordinary and Megatron trainers and CLI routes have distinct axes. Declared
`rlhf_type` aliases remain a separate axis: SimPO's presence in arguments does
not invent an additional trainer class. Only fields owned by argument modules
contribute to that axis; model metadata fields do not define CLI choices.
verl Hydra entries include training, evaluation and generation and are identified
by source path, function and config name. They are not algorithm acceptance.

Missing or dynamic registries, unsupported Literal declarations, ambiguous axes
and invalid distillation names raise errors rather than silently shrinking the
denominator. No manually maintained list of model or algorithm names was added.

## Pinned source results

All five inventories were rederived from clean tracked source checkouts:

| Repository | Version | Full upstream SHA | Separate denominator axes |
| --- | --- | --- | --- |
| Transformers | 4.56.2 | `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc` | 374 model families; 46 task mappings |
| PEFT | 0.17.1 | `53c25fe4fdd7c6aa4b40db0560815ea570d32303` | 26 methods; 6 tasks |
| Diffusers | 0.35.1 | `0f252be0ed42006c125ef4429156cb13ae6c1d60` | 79 pipeline module families; 249 pipeline exports; 37 AutoPipeline families; 3 AutoPipeline task mappings |
| ms-swift | 4.5.2 | `ff5128777dd3dc5538f1a95fbcc29442c327d7d2` | 222 model families; 11 tuners; 13 ordinary trainers; 5 Megatron trainers; 12 ordinary CLI routes; 4 Megatron CLI routes; 5 declared tasks; 9 RLHF choices |
| verl | 0.10.0.dev | `e5a627cf55c94d85406556854c2c1a22db4ff5e7` | 14 advantage estimators; 12 policy losses; 5 Hydra entries; 8 distillation losses |

These counts are independent denominators, not numbers to sum into a support
rate. In particular, Diffusers' smaller AutoPipeline matrix does not replace its
full pipeline denominator. A scorecard generated with these inventories and no
numerical evidence retained every denominator and assigned every numerator zero.

## Reproduction and checks

```bash
python -S agent/skills/torch-compat-repo-runbook/scripts/registry_inventory.py \
  --repo ms-swift --checkout <clean-pinned-ms-swift-checkout> \
  --output "$JITTOR_LAB_ROOT/_state/downstream-coverage/ms-swift-registry.json"
python -S tests/tools/test_training_registries.py -v
python -S tests/tools/test_downstream_coverage.py -v
```

Eight source-only regression cases passed, including dynamic and ambiguous
registry counterexamples, invalid names, and upstream changes that add a method
alias without changing trainer coverage. The 18 existing coverage/audit contracts
also passed. An unavailable tensor-runtime import in the source fixtures proves
enumeration does not execute upstream code. The full five-source scorecard CLI
completed successfully with no accepted evidence rows and all zero numerators.
The complete structure selection passed with 1,432 passed, six skipped and no
failures. Two skips concern distribution resolution in the source-only
interpreter; the others are declared or accelerator capability skips. An initial
run caught collection-time `exec_module` in the new host fixture; replacing it
with ordinary import fixed the test contract without relaxing the gate. Layout
and whitespace checks passed. The unversioned structure log is
`$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/training-registry-structure-recheck.log`.
The CPU core runner also passed: 322 passed, 42 skipped and one existing expected
failure, with both subprocesses returning zero. Its unversioned log is
`$JITTOR_LAB_ROOT/_state/refactor-goal/distributed/logs/training-registry-core.log`.

Python 3.11 was used. Unversioned baseline and final inventories, source hashes
and the empty scorecard remain under
`$JITTOR_LAB_ROOT/_state/refactor-goal/r21-registry-audit/`. Files are named
`baseline-ms-swift.json`, `baseline-verl.json`, `final-<repo>.json` and
`empty-scorecard.json`. These are registry evidence, not model benchmark data.

## vLLM and Omni central inventory follow-up

The follow-up based on `1eaf2678d` connects the host-only vLLM extractor to the
same `registry_inventory.py` CLI and schema. Actual clean upstream checkouts
were re-enumerated; no tensor runtime or downstream package was imported:

| Repository | Source identity | Full upstream SHA | Separate denominator axes |
| --- | --- | --- | --- |
| vLLM | exact release tag `v0.29.0` | `98dff2a81d747d1dba01a47f939f48c3526d4206` | 376 model architectures; 40 CustomOp layer declarations |
| vLLM Omni | `source@446c2b5474dabffccf9a0034836eeab77c62b4b8` | `446c2b5474dabffccf9a0034836eeab77c62b4b8` | 92 owned model architectures; 68 diffusion models; 64 pipeline model types |

The inventories bind 589 and 172 scanned source hashes respectively and retain
dynamic registration sites and missing implementation targets. Omni's inherited
vLLM registry requires a separately pinned vLLM inventory; it is not silently
added to the owned-model denominator. Each axis remains separate. CustomOp
membership or a layer numerical result cannot certify an engine or model.

When generated package-version files are absent, only an exact final-release
tag at the selected commit supplies a release version. A nearby tag or release
candidate is insufficient; source-only identity remains `source@<full SHA>`.
Multiple distinct exact release versions fail closed. This source label does
not replace independently recorded installed-distribution versions, and these
enumerations grant no numerical scorecard rows or adapter version support.

Reproduce with the existing CLI, using `--repo vllm` or `--repo vllm-omni` and
the corresponding clean pinned checkout. The helper's 14 contracts and central
coverage/identity/documentation checks passed together: 77 passed, no skips.
They cover dirty source, unknown mutations, missing targets, inherited scope,
release-tag ambiguity and upstream re-enumeration. Raw inventories and logs are
unversioned under `$JITTOR_LAB_ROOT/_state/leaf-view-integration/`:
`vllm029-inventory.json`, `omni-inventory.json` and
`vllm-registry-contracts-corrected.log`.

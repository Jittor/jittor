# Downstream registry, gating and scorecard coverage audit

- Status: Source-code coverage audit executed; independent oracle numerical rerun blocked by environment. This report grants no new L2–L5.
- Date: 2026-10-14 (Asia/Shanghai).
- Baseline commit: `cded3102037edc3663b8f2054a6ebc06da6ebfd2`, plus the commit containing this report.
- Scope: all per-library runbooks, shared ecosystem CASES/test nodes, referenced reports, and the automatic denominators of five upstreams.
- Owner: Torch compatibility layer and downstream ecosystem maintainers.
- Review when: upstream registry/version, shared ecosystem tests, runbook, scorecard protocol or verification environment change.

Subsequent real execution is in [four-library CPU/CUDA oracle and three-step training](2026-10-14-downstream-oracle-scorecards.en.md): eight device samples of fixed releases passed, and the registry scorecard accepted six of the L3 entries; the two Diffusers component evidence entries are not counted in the pipeline denominator. This page preserves the original baseline and environment state of the initial audit.

## Source entry facts

The table distinguishes registration from execution. Report counts only include `docs/results/` files referenced by runbooks that still exist; mentions of missing (including explicitly retired) reports are also recorded by the machine audit; mentions of library names across all results are listed separately as weak associations and cannot count as dedicated verification. Independent adapters/dedicated unit tests are not part of the shared ecosystem gate here, and 0 does not mean "no other tests at all". The L levels mentioned in skills are textual facts; this run does not treat them as passing evidence.

| Repo / skill | registered / gate case count | existing / missing report mentions | L level mentioned in skill (not verified) | grantable level this run |
| --- | ---: | ---: | --- | --- |
| diffusers | 3 / 3 | 1 / 3 | not explicitly graded | not established |
| flash-attention | 0 / 0 | 1 / 0 | not explicitly graded | not established |
| mmcv | 1 / 1 | 0 / 2 | not explicitly graded | not established |
| mmengine | 1 / 1 | 0 / 2 | not explicitly graded | not established |
| ms-swift | 1 / 1 | 0 / 1 | not explicitly graded | not established |
| peft | 1 / 1 | 0 / 1 | not explicitly graded | not established |
| tensordict | 0 / 0 | 0 / 1 | not explicitly graded | not established |
| torchmetrics | 0 / 0 | 0 / 1 | not explicitly graded | not established |
| transformers | 11 / 11 | 1 / 0 | not explicitly graded | not established |
| trellis | 0 / 0 | 0 / 1 | L2 | not established |
| verl | 0 / 0 | 1 / 2 | not explicitly graded | not established |
| vllm-omni | 0 / 0 | 3 / 0 | not explicitly graded | not established |
| vllm | 0 / 0 | 1 / 2 | not explicitly graded | not established |

Transformers' 11 entries include 6 tiny cases and 5 realistic speed cases; Diffusers' 3 entries include UNet/DiT modules and realistic UNet, not equivalent to full pipeline acceptance. ms-swift has only `ms_swift_lora_llama`, cannot extrapolate to full/distributed training; verl has a runbook and historical report entry, but no case entering the shared ecosystem gate. The number of newly accepted compute evidence entries in core/broad/tail this run is **0** for all.

## Automatically derived complete denominators

Enumerated from clean Git checkouts of each upstream; no downstream libraries imported, no denominator reduced due to missing dependencies. X=0 below means no evidence satisfying the new scorecard protocol this run, **does not deny models historically run**. CPU, CUDA, NPU, ROCm, Corex output per device; newly accepted numerators this run are 0 for each device.

| Upstream | complete commit / version | automatic breadth axes and denominators |
| --- | --- | --- |
| transformers | `1ad1ad2f2691ce4f867590d2319bcd12fc2a02cc` / `5.19.0.dev0` | model_family **0 / 564**; task **0 / 53** |
| peft | `20f82b52cb690b249cb90cbab43af49d005ab510` / `0.21.3.dev0` | method **0 / 45**; task **0 / 6** |
| diffusers | `ea981b8b7d80f10deaff369bcda8cbf2c90428dd` / `0.41.0.dev0` | model_family **0 / 93**; pipeline_class **0 / 350**; auto_family **0 / 66**; task **0 / 8** |
| ms-swift | `4dc1e2add536088a3dee85e7568b4022330a2049` / `4.7.0.dev0` | model_family **0 / 237**; method **0 / 11** |
| verl | `e5a627cf55c94d85406556854c2c1a22db4ff5e7` / `0.10.0.dev` | advantage_estimator **0 / 14**; policy_loss **0 / 12** |

Upstream remotes are the GitHub SSH repositories `huggingface/transformers`, `huggingface/peft`, `huggingface/diffusers`, `modelscope/ms-swift`, `volcengine/verl`; actual URLs are preserved in the raw snapshot. Diffusers all conditional branches yield 93 pipeline module families and 350 pipeline export APIs; the AutoPipeline 66 families are only a sub-matrix, not used to replace the full-library denominator. verl's 14/12 are the advantage-estimator/policy-loss registries, not pretending to be a complete training algorithm or a support list of all models. PEFT's model families require binding to the Transformers registry reference separately; 45 methods cannot be treated as 45 models.

## Actual rerun and environment boundaries

CPU ① explicitly set `CUDA_VISIBLE_DEVICES` empty, `nvcc_path`/`JT_BUILD_NVCC_PATH` empty, `use_cuda=0`, and set independent JIT state, fixed seeds, offline and thread configuration. Selecting `transformers_gpt2` to run the independent PyTorch oracle failed at startup: `ModuleNotFoundError: No module named typing_extensions`. Raw output and failure state preserved, command returned nonzero. Two independent external venv preparation attempts then hit the base interpreter missing `_sysconfigdata__linux_x86_64-linux-gnu`, so the shared environment was not modified nor numeric/speed ratios published.

Currently discoverable downstream installed versions are Transformers 4.56.2, Diffusers 0.35.1; PEFT, ms-swift, verl absent, and different from the source-code denominator versions above. Old installed versions' numbers cannot be used to count coverage for the latest upstream SHA. CUDA③ and NPU③ not yet rerun; ROCm/Corex not verified; no new L4 or L5 conclusion covering all platforms.

`verify_repo.py` has been tightened: nonzero exit (including printing a success marker first), non-finite/wrong-shape tensors, missing tensors, device/dependency version/precision/thread configuration mismatch all fail; failures generate no comparable ratio, and complete metadata and logs from both sides are kept. This single-step diagnostic is fixed at `scorecard_eligible=false`, not pretending to be multi-step training, input perturbation or whole-process memory sampling.

## Current re-verification matrix

| Repo / specific scenario | CPU① | CUDA③ | NPU③ | current level / speed ratio |
| --- | --- | --- | --- | --- |
| Transformers / tiny GPT-2 | attempted, oracle environment blocked | not run | not run | not established / not published |
| Diffusers / tiny UNet2D | attempted, same oracle environment blocked | not run | not run | not established / not published |
| PEFT / tiny Llama LoRA | attempted, same oracle environment blocked; PEFT also absent | not run | not run | not established / not published |
| ms-swift / tiny Llama LoRA | attempted, same oracle environment blocked; swift/PEFT also absent | not run | not run | not established / not published |
| verl / algorithm registry | no shared ecosystem case, not wired to this gate | not run | not run | not established / not published |

The four registered libraries each returned nonzero and saved `verify-report.json`/both-side stage logs. No historical reports were used to fill gaps, nor was the registry denominator reduced for absent libraries. When the structure gate was actually attempted, the current interpreter reported `No module named pytest.__main__`, so the structure gate cannot be claimed to pass. Host-side unittest, Ruff, layout and new active documentation link checks passed.

## Protocol and runnable verification

Unified platform/speed blocks, JSON schema and report templates are in the [downstream scorecard protocol](../development/downstream-scorecards.en.md). 18 host-side targeted and negative tests passed: covering denominator growth after upstream registrations, dirty source/empty/dynamic registry, L1/skip/xfail not scored, stale SHA/version, shim oracle, wrong device, missing gradient/perturbation, NaN, fake gate, artifact hash tampering, missing CPU/zero fallback, single-step pretending L3, and tiny/wrong regime/missing agreement/wrong memory protocol pretending L4.

```bash
python -m unittest discover -s tests/tools -p test_downstream_coverage.py -v
python -S agent/skills/torch-compat-repo-runbook/scripts/audit_coverage.py audit --json
```

The complete reproduction commands for upstream denominators and scorecard are in the protocol; all five `--upstream <repo>=<checkout>` can be derived repeatedly. An empty evidence list does not automatically pass computation, only outputs auditable 0/N. Commands with illegal evidence keep rejected reasons and return nonzero.

Raw snapshots, scorecard JSON, audit JSON, failure logs and caches are in `$JITTOR_LAB_ROOT/_state/downstream-evidence/`, upstream checkouts in `$JITTOR_LAB_ROOT/downstream-coverage/upstream-ssh/`, all **unversioned**. This report keeps only reproducible maintainer conclusions, not raw registry data or environment paths.

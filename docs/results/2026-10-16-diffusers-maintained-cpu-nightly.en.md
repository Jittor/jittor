# Independent-environment acceptance of the complete Diffusers CPU nightly

- Status: local complete maintained gate passed; this is not a record of a GitHub scheduled job execution.
- Date: 2026-10-16.
- Maintainer: Jittor core maintainers.
- Measured source: `d63976115fb6a31d94c4474871b07f2aba69d7e7`, clean worktree.
- Scope: the existing complete tiny UNet CPU three-phase protocol for official Diffusers 0.35.1.
- Review when: the nightly installation closure, provenance validation, the UNet protocol, or the pinned Torch or Diffusers version changes.

## Maintained entry and environment

`ecosystem.yml` gains a `diffusers` profile; `tools/provision_ecosystem_nightly.py`
creates three brand-new prefixes (candidate, oracle and controller) and then invokes
`nox -s ecosystem_diffusers_source`. The maintained entry ultimately runs `tools/run_diffusers_gate.py`;
no import/preflight check or shrunken model was substituted for the complete protocol. The ASTs of the existing 74 nox functions are unchanged.

This run bootstrapped from an independent healthy Python 3.11, and all dependencies resolved and installed normally. The actual `pip check` passed in all three prefixes; the oracle actually imported `_C` from Torch 2.11.0+cpu, with no shim and with CUDA invisible.
What is reusable locally is the hashed official wheel and the clean upstream Git inputs; no JIT cache or installation directory was copied.
Unit-test and benchmark HOME/JITTOR_HOME are separate, and the first benchmark JIT actually compiled.
The original controller host command failure (pytest not installed) is preserved, and host validation used an independent healthy oracle environment.

## Complete protocol and independent recomputation

The original seeds 0/7, 144 parameters and sample input gradients, and three SGD steps (lr=0.001) are unchanged.
Six independent workers completed base, perturb and train; nox exited naturally with 0, taking about eight minutes including the first compilation.
Independently reading the original NPZ/JSON values and recomputing gives:

| Evidence | Actual result |
| --- | --- |
| Complete array pairs on both sides | 1,165 |
| Actual CPU tensor observations | 2,618 |
| Per-worker source file closure | 1,612 |
| Three-step parameter updates recomputed on both sides | 864 |
| Maximum absolute error / maximum global normalized error | 4.959106e-5 / 9.408136e-7 |
| Maximum SGD replay rounding error | 3.725290e-9 |

The original per-phase `.002 absolute OR .01 global-reference-scale` budget is retained; SGD replay uses the rounding bound `2*float32_eps*(abs(old)+abs(lr*grad))`. Parameter and input gradients are complete and nonzero, the perturbation really changes the output, and every worker's fallback count is 0.

Per side, the actual installed files were compared with the original official wheel: 11,945 non-RECORD files for each Torch, 702 for each Diffusers; all hashed rows of the real pip RECORD were verified at 11,949 and 706 respectively.
Archive RECORD was not written over the installation receipt; generated pyc is only verified against its corresponding source and is not passed off as wheel payload.
The complete source was additionally verified by Git mode/blob/content across 2,916 files; the candidate JSON's binary field is unusable, so no claim is made that its native binary SHA was captured. The raw hash of the genuine oracle `_C` was verified separately.

## Evidence and scope

The raw artifacts are unversioned and located under `$JITTOR_LAB_ROOT/_state/diffusers-nightly/20261016/`:
`fresh-v1/provision.json`, `fresh-v1/logs/024.log`, `fresh-v1/maintained-output/`,
`fresh-nightly-terminal-v1.json`, `nightly-cpu-independent-v1.json` and
`nightly-installed-independent-v1.json` preserve the actual installation, commands, terminal state and independent recomputation.
`nightly-evidence-seal-v1.json` seals 395 files / 82,710,906 bytes, SHA256
`3abfe7fe0acba477941a5d9c4694a5252a10c69fb3366678eef446f214a30179`; the full second hashing pass succeeded.

This accepts only the existing complete CPU component nightly. For the existing CPU/CUDA paired-source conclusions see the
[official installation validation](2026-10-16-diffusers-official-installation.en.md); its original b212 source is not relabelled as this run's d639.
No whole-library coverage numerator is added, and no pipeline, new CUDA test, L4 performance or "GitHub scheduled job executed" conclusion is granted.

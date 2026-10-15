# Accelerate checkpoint restoration in new processes

- Status: verified bounded protocols; world-two CUDA paired admission remains blocked
- Date: 2026-10-15
- Baseline commit: `93d2cfa7dd6c69ce6f5cb0e2d310dcf80f06e1e0` (clean final-v6 measurement source)
- Upstream: Accelerate 1.10.1, official clean commit `623cc0ba588d192c77758939a6e340adf07edd4f`
- Owner: Jittor Torch compatibility maintainers
- Review when: checkpoint, RNG, optimizer, scheduler, FSDP2, launcher, protocol capture or independent-auditor ownership changes; rerun on other hardware/toolkit combinations.

The maintained Accelerate gate now saves with an actual producer process, waits for that process and launcher to terminate, and starts a distinct consumer with newly constructed/prepared model, optimizer and scheduler. The consumer reads only persisted files through `Accelerator.load_state`; it does not read producer observations or receive live Python objects. PID, nonce, monotonic process/launcher boundaries, complete checkpoint hashes and both phases' original arrays are audited separately.

The existing fixtures and tolerances are retained: Linear(3,2), AdamW plus StepLR for world one; CUDA world-two FSDP2 FULL state with Linear(4,3), AdamW and no scheduler, in FP32 and BF16. Each fixture uses original seed-0 inputs and seed-7 perturbation, three pre-save updates and two continuation updates. Every parameter gradient and input gradient is compared to an independent genuine Torch process with `atol=2e-6`, `rtol=2e-5`; restoration and own-runtime continuation remain exact. Python MT, NumPy MT, Torch CPU and every CUDA RNG state are actually advanced before saving, then restored completely, with subsequent samples also verified. The consumer deliberately has a distinct pre-load seed/state. Each visible CUDA generator is sampled before saving and after restoration, including the oracle process's non-current card; advancement is checked per generator, rather than comparing only the aggregate list of states.

Historical b7 measurements remain same-prepared-object evidence. They are not relabelled by the new protocol IDs `accelerate-checkpoint-fresh-process` and `accelerate-fsdp2-full-fresh-process`. CLI defaults still support the legacy protocol; maintained nox defaults select `--fresh-process`.

## Actual environment

The oracle is genuine Torch 2.11.0+cu128 with a binary `_C` extension, separate from the source-bound Jittor Torch shim. Both consume the identical clean Accelerate 1.10.1 source/package proof. CUDA hardware is NVIDIA RTX 4090 SM89 with driver 580.178.04. Native provider uses actual NVCC 12.2.140 with ten real pinned CUDA component wheels, including cuDNN 8.9.7.29 and NCCL 2.18.3. These wheels were downloaded into an isolated task prefix; official PyPI SHA256 and wheel RECORD checks verify the actual installed headers/libraries. The real Torch environment retains its own CUDA 12.8 component stack. Actual per-process loaded-library observations distinguish the two stacks; no metadata or library is re-labelled.

CPU execution uses an AMD EPYC 9554 under Linux 6.1/glibc 2.34, Python 3.11.16 and GCC 11.5.0. CPU and CUDA each use the maintained nox entrypoint. Native two-rank CUDA uses the public `python -m jittor.distributed.launch` entrypoint (equivalent to console `jtrun`); genuine Torch uses its own torchrun. Serial warm-up compiles each rank's real native core and NCCL extension before formal launch. Every rank owns HOME/JITTOR_HOME/TMPDIR, XDG/shim and extension caches, while communicator control is shared only within its actual launch.

## Reproduction

Prepare independent candidate and genuine Torch interpreters, the official pinned Accelerate checkout and a coherent native CUDA installation. Set the interpreters and checkout through the existing gate inputs:

```bash
export JITTOR_CI_PYTHON=<candidate-python>
export REAL_TORCH_PYTHON=<binary-torch-python>
export JITTOR_ACCELERATE_CHECKOUT=<official-pinned-checkout>
export JITTOR_LAB_ROOT=<external-lab-root>
export JITTOR_CUDA_WHEEL_STRICT=1
JITTOR_TEST_DEVICES=cpu python -m nox -s ecosystem_accelerate -- \
  --output <empty-external>/cpu --cache-root <private-benchmark-cache>
CUDA_VISIBLE_DEVICES=<allocated-gpus> JITTOR_TEST_DEVICES=cuda \
  python -m nox -s ecosystem_accelerate -- \
  --output <empty-external>/cuda --cache-root <private-benchmark-cache>
```

The gate refuses a nonempty fresh-process output directory, including partial failures. A failed producer never starts its consumer. Session-level explicit strict provider/NVCC/NCCL inputs survive nox isolation. Measurements and unit tests use distinct caches; raw logs, checkpoints, component wheels and compile products are external and unversioned.

## Results and limits

| Device | World | Precision | Array pairs | Param/input gradients | Max abs | Max scaled | Paired admission |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| cpu | 1 | no | 438 | 28/14 | 5.9604645e-08 | 1.7745026e-07 | L3 accepted |
| cuda | 1 | no | 438 | 28/14 | 2.9802322e-08 | 1.9864779e-07 | L3 accepted |
| cuda | 2 | bf16 | 820 | 56/28 | 1.4901161e-08 | 6.4892491e-08 | Blocked: no CPU world-two counterpart |
| cuda | 2 | no | 820 | 56/28 | 1.4901161e-08 | 8.9234059e-08 | Blocked: no CPU world-two counterpart |

The common auditor accepts two world-one L3 records, blocks the two independently verified CUDA world-two L3 records, and rejects none. It checks 2,516 cross-runtime array pairs, all 168 requested parameter-gradient fields and 84 input-gradient fields. All 48 actual phase workers completed with zero skipped tests; the 24 native workers report error-on-fallback and counter zero. Genuine Torch has no Jittor fallback counter and records it as not applicable. There are 64 persisted checkpoint files.

The BF16 FSDP2 cases observe actual BF16 Linear inputs/weights/outputs while retaining FP32 master parameters and optimizer state. Every own-runtime saved model, optimizer, scheduler when present, complete RNG state and next-sample/continuation comparison is exact; independent numerical budgets remain unchanged. Native CUDA world two uses public jtrun. This protocol does not use the older CPU MPI track.

The final source passes 60 host admission contracts and full structure plus those contracts: **1,568 passed, 6 skipped**. The checkpoint-owner CPU selection passed 33 tests with one optional-Accelerate skip in its separate unit interpreter, and the added rotation selection passed 5 tests. The maintained formal interpreters both have required Accelerate and execute every requested node without skips. CPU core at the unchanged checkpoint-owner source `0823f271d` passed **322**, skipped **42**, and retained **1 xfail**; subsequent changes affect capture/auditor fields only. Seven disposable copies of actual raw artifacts are rejected for same PID, changed moment, wrong complete RNG, initial-offset CPU RNG, nonzero fallback, inactive second CUDA generator and overlapping launcher lifetimes. Historical b7 recomputation retains 2 accepted/2 blocked/0 rejected with its original same-object semantics.

External, unversioned evidence is under `$JITTOR_LAB_ROOT/_state/accelerate-fresh-process/20261015/`: `formal-v6/{cpu,cuda}`, `final-scorecards-v6`, `preflight-v6a`, `provider-watch-v6.jsonl` and `final-audit-negative-controls-v6.json`. The total `final-v6-evidence.json` SHA256 is `8646545e61bcda2d7df7bd4235fa1360f5fdf514951d128fd03a4fffd32b0591`; it binds 1,567 source files, 342 raw files, all 48 phase identities and 38 actually mapped library files. Every CUDA phase PID has an actual map observation. Sixteen native NVIDIA library payloads match the official wheel RECORD. All native phase metadata reports the resolved exact wheel pins.

Native actual runtime/NCCL versions are 12020/21803, and genuine Torch versions are 12080/22809. Launchers and serial warm workers additionally map the external NVCC toolchain's separate `libcudart.so.12.2.140` (also API version 12020); none of the 48 formal phase PIDs maps that extra file. Its actual path/hash is retained separately, and it is not labelled as a wheel payload.

World-one CPU and matching CUDA records can be admitted by the common full-semantic protocol pairing. CUDA world-two FP32/BF16 has no same-semantic CPU FSDP2 counterpart and must remain blocked for paired admission, even when independent CUDA agreement and strict fresh-process restoration succeed. CPU world one is not a substitute for CPU world two. These are bounded protocol records with no registry denominator or whole-Accelerate coverage claim. This work establishes no L4 speed/memory claim and no NPU/ROCm evidence.

Recompute independent agreement in an isolated checkout containing this report and the code refs, with prepared NumPy/stdlib audit dependencies:

```bash
<audit-python> agent/skills/torch-compat-repo-runbook/scripts/collect_accelerate_scorecards.py \
  --run <run-state>/formal-v6/cpu --run <run-state>/formal-v6/cuda \
  --upstream <official-pinned-checkout> \
  --baseline 93d2cfa7dd6c69ce6f5cb0e2d310dcf80f06e1e0 \
  --report docs/results/2026-10-15-accelerate-fresh-process.md \
  --command "independent raw recheck" --output <new-external-audit-output>
```

## Retained failures and scope changes

- formal-v2 native CUDA failed before training because nox dropped the caller's strict wheel option and the older candidate environment selected CUDA 12.8 metadata instead of native cu122. The genuine Torch side's successful runs do not admit that setup.
- formal-v3 tensor tests exposed inherited nox XDG/shim cache routing: preflight rewrote native HOME into a shared temporary session directory. The strict external auditor refused this evidence. A later source changes only the benchmark environment routing, with the failed raw files preserved.

- The initial v3 two-rank consumer also exposed a DCP initialization bug: a zero-learning-rate AdamW initialization step rotates native FSDP public shard views, invalidating an FQN identity map cached before that step. The checkpoint owner now resolves names after initialization and state extraction. A local actual-tensor rotation test preserves weights, zero moments, step and learning rate; the final two-rank public API run verifies the real FSDP case.
- v4 already passed all bounded tensor/strict-restoration scenarios, but the oracle's non-current visible CUDA generator had remained at its initial offset. v5 adds actual per-device advancement and subsequent-sample verification; v4 is not relabelled as this stronger protocol.

- v5 passed the bounded mathematics and restoration checks, but its package inventory comprehension let a later shadowed site overwrite the actually selected CUDA distribution version. The capture now records `importlib.metadata.version(name)` for each package, with real competing `dist-info` precedence tests. A six-process preflight checked both environments, real mapped libraries against wheel RECORD, private HOME/XDG paths and all RNG streams before freezing final v6. v5 metadata remains unchanged and is not evidence of consistent provider provenance.

All earlier phases and failed setups retain their own source identities. Final evidence binds only the explicitly listed final clean source and terminal runs.

# Accelerate maintained CPU fresh-process nightly

- Status: maintained provisioning and strict complete gate implemented; fresh execution pending.
- Reviewed: 2026-10-16.
- Owner: downstream verification maintainers.
- Implementation base: `c47033ed0a6bb692922efa98ecbbf7896bfe867f`.
- Review when: upstream/wheel pins, installed-source proofs, training or recovery owners,
  phase launchers, platform capture, provisioning or source-closure validation changes.

The `accelerate` CPU nightly profile provisions three independent environments and
runs the complete original world-one FP32 fresh-process protocol. It requires both
input seeds 0/7, three AdamW/StepLR updates before checkpointing, terminated producers,
new consumers and two continuation updates. All 438 array pairs, 28 parameter-gradient
and 14 input-gradient fields remain mandatory. The original elementwise
`atol=2e-6, rtol=2e-5` budget is unchanged. A successful local run requires the
common auditor to accept exactly one L3 protocol record with no blocked/rejected rows,
and verifies eight distinct phase-worker PIDs. It creates no registry denominator.

```bash
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 use_cuda=0 \
python tools/provision_ecosystem_nightly.py --library accelerate \
  --state-root "${JITTOR_LAB_ROOT:-../jittor-lab}/_state/accelerate-nightly/fresh-run" --run
```

The source pin remains official Accelerate 1.10.1 at
`623cc0ba588d192c77758939a6e340adf07edd4f`; the official wheel SHA256 is
`3621cff60b9a27ce798857ece05e2b9f56fcc71631cfb31ccf71f0359c311f11`.
Normal pip resolves the complete runtime Requires-Dist closure into each private
candidate/oracle prefix; the nox controller has its own complete tool closure.
Each prefix must pass `pip check`. The candidate alone activates the official shim;
the oracle asserts its real Torch binary. Actual pip METADATA/RECORD and the full
pinned package payload remain worker-bound; this profile never restores archive
RECORD over installed receipts. Source editable installs reuse the resolved closure.

The maintained `ecosystem_accelerate` nox session selects the strict profile only
when `JITTOR_ACCELERATE_COMPLETE_CPU=1`; its existing CPU/CUDA entrypoints and optional
arguments remain available otherwise. Complete nightly mode rejects protocol/device
substitutions and runs the existing gate followed by the existing collector with
`--require-complete-cpu`. The public producer/consumer launcher and training math
are reused. Persistent output uses `JITTOR_ACCELERATE_EVIDENCE_DIR`; benchmark caches
remain separate from unit caches, with first compilation serialized.

New workers capture the actually called collector, nox and provisioner bytes in
addition to the original measurement sources. The validator reads that static
control closure from the specified baseline's committed instrumentation, never
from present-day file existence. Historical c7 maps retain their original fields;
replay of its 48 phase maps and four original records still yields two accepted,
two CPU-unpaired blocked and zero rejected. No old raw file receives a new field.

The [original source-bound CPU/CUDA results](2026-10-16-accelerate-installed-source-protocols.md)
retain their measured commit and original scope. This maintenance profile does not
claim CPU world-two FSDP2, substitute DDP as its counterpart, grant a new CUDA run,
prove sharded/multihost checkpoint support or establish whole-library coverage.
Workflow wiring alone does not prove GitHub's hosted schedule executed. Fresh
execution, complete source/install/platform audits and sealed evidence will be
recorded only after the actual complete profile terminates.

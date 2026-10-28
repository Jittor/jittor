# Native jtrun fixed-node control plane: single-host two agents and real CUDA verification

- Status: the Linux/NCCL fixed-node software control plane is implemented; real multi-host acceptance awaits hardware.
- Review date: 2026-10-15.
- Owner: Native distributed maintainers.
- Measurement source: `bc07df0c079fcaa6ffcefce7ede1423ca599305e`; upstream baseline `4ccd72fafe51496190671dc93548c852be2974c1`.
- Later synchronization: official remote `bd17d080ee31c8cb84f3ac2617662f423c343fec`, merged into this tree as `ee7a5ce6d697ca4a39d37e3d672cc85f7d16b925`; the production owner is byte-for-byte identical to the measurement source.
- Independent recomputation tool: `abb4f302c434a7ea3251bfdc6949ae1dfe22e442`; test-organization fix only `5844a4581`. The tool and report do not change the identity of the historical measurement.
- Recheck condition: the launcher, Store, NCCL bootstrap/subgroup/watchdog, the FSDP fixture or the actual network/device topology changes.

## Confirmed problem and implementation

Minimal reproduction on the actual host at the original baseline `4ce19215cba8fb15d7b347f53fda012ced9b4143`:
`--nnodes/--node-rank` were not recognized; non-master agents tried to bind the remote master address; the two agents
each allocated rank 0/1. The old subgroup and watchdog also relied on files under the same root-info path.
The original reproduction is kept in the unversioned `baseline-control-plane-probes.json`.

The public `jtrun` and the equivalent module entry `python -m jittor.distributed.launch` gained fixed-node arguments.
The global rank is derived from the node rank and the local process index; the default single-node behavior and the one-card-per-worker mask are preserved.
Multi-node is enabled only for Linux/NCCL; HCCL is rejected explicitly before device probing and worker startup.

The launcher on node 0 holds the existing native TCPStore service; other agents/workers connect to the same service.
The job id isolates key spaces, node count/process count/backend must agree, and a duplicate node rank is rejected.
The worker does not hold the server side, so that GPU collectives holding the GIL do not block other workers' Store requests.
WORLD and subgroup exchange the NCCL opaque unique id; the subgroup checks the rank order of all WORLD participants,
then reuses the original C++ communicator owner to create an independent communicator.

The watchdog only reads the heartbeat files of workers on its own node; agents monitor other agents using the Store sequence number
and local monotonic time, and relay worker failures. The Linux parent-death signal constrains the worker's agent lifetime.
No FSDP/TP/ZeRO/optimizer/checkpoint mathematics was copied; this batch only extends the launch and communication control plane.

## Actual execution

Used two RTX 4090 cards, the real Cu122 wheel stack, NVCC 12.2.140; each worker's CUDA mask
exposed only one card, with an actual device ordinal of 0. External consumer processes were left running; no throughput, L4 or peak-memory conclusion is drawn.
Each rank used task-private HOME/JITTOR_HOME/TMPDIR; unit tests and the earlier benchmark cache were kept separate.
The first core, extension and maintained-fixture warm-up ran serially per rank; real NCCL was called directly so that the singleton
public identity path could not mask an unloaded library. The candidate was always in native mode, fallback forbidden, count delta 0.

The reference interpreter actually loaded binary Torch `2.11.0+cu128`, rejected the shim, disabled TF32, and used highest precision.
The candidate and reference reused the maintained fixture's fixed weights and inputs; the five-step dense reference ran independently.

| Check | Actual result | Scope |
| --- | --- | --- |
| CPU core | 322 passed, 42 skipped, 1 xfailed | native 98 + Torch 224; clean `abb4f302c` |
| New and existing host adjacency tests | 46 passed, 0 skipped | loopback agent, Store, legacy ownership, evidence negative cases |
| Two agents / two CUDA cards | 3 passed per rank, 0 skipped | reverse-order root, two independent subgroups, direct singleton NCCL, WORLD, 75 mixed sets, out-of-order rejection |
| Original single agent / two CUDA cards | 2 passed per rank, 0 skipped | same numerical nodes and original file subgroup path; extra Store out-of-order node explicitly deselected |
| Independent Torch / native FSDP parity | 4 passed per rank, 0 skipped | SGD/AdamW × FP32/BF16, five-step full gradients, parameters, moments, original budgets |
| Production-side fresh launcher | 4 passed per rank, 0 skipped | save at step three, dropout 0.25, continue two steps |
| Another fresh launcher on the resume side | 4 passed per rank, 0 skipped | new PID, new model/optimizer, complete arrays exact for two steps after loading |
| Real CUDA worker failure | agent returned `[1,137]` | killed only this task's rank 1 after both sides' STEP 5; exit propagated in about 0.717 seconds, no worker left behind |

The production-side candidate PIDs were `594850/594848` and the resume side `604357/604349`; the launcher PIDs also differed.
The raw worker identity records also contain the measurement SHA, world/local-world, the external Store flag and the private cache paths.
This fresh-process run is the native FSDP/checkpoint fixture and cannot be used to relabel the historical Accelerate
same-prepared-object record.

The FP32 per-element budget stays `atol=2e-6, rtol=2e-5`; BF16 stays `atol=4e-4, rtol=3e-2`.
The per-parameter update L2 budgets are `2e-6 + 2e-5 * oracle_norm` and
`1e-6 + 0.03 * oracle_norm` respectively, not relaxed according to this round's error. Every real gradient and parameter update has a zero
negative case; the actual Linear input/weight BF16, FP32 master/gradient/moments, BF16 all-gather and
FP32 reduce-scatter are all checked by the original fixture. Each rank owns 35 storage elements, corresponding to 66
elements for the full model plus padding; this owned-storage fact is not proof of peak-memory savings.

## Raw evidence and independent recomputation

Unversioned root directory: `$JITTOR_LAB_ROOT/_state/native-multihost/20261015/`.
`raw-bindings-bc07.json` binds 299 raw files (2,166,662 bytes), with SHA256
`9ee73c750a46ddf6194328ef1c38f67dd37e57d57ed46edd203114dcba95d1cb`.
`source-bc07.json` records the SHA256 of 2,804 regular files and did not include the target text of the directory symlink in that scan;
`source-bc07.bundle` keeps the complete historical Git source, so the regular-file list is not called proof of a complete Git tree.

The actual argv/env are in `cuda-unit/commands-{warm,oracle,multi,legacy,parity,baseline,resume,kill}.json`;
per-rank compile/run logs, raw NPZ/JSON, the four checkpoints, actual PIDs and scripts are in the same directory.
The formal results were executed through the public module launcher; private two-process or MPI results are not relabelled as jtrun.
The raw executor is `cuda-unit/run.py` and the worker environment entry is `cuda-unit/rank_entry.py`; their current
file hashes are in the raw bindings, and the historical warm failures keep only diagnostic identity and grant no pass.

The maintained entry [independent recomputation tool](../../agent/skills/jittor-distributed-verification/recheck_native_fsdp.py)
uses NumPy to check all bindings, the complete parameter list and the metadata, then recomputes the 680 oracle arrays,
the 368 fresh-resume arrays, the 160 update budgets and the 160 zero-gradient rejections.
The four RNG kinds are the actual subsequent draws of Python/NumPy/native CPU/native CUDA, bit-identical to the production side after resume.
The tool imports neither Torch nor Jittor, and recomputation did not replace the original GPU execution.

Recomputation usage is in the [distributed verification skill](https://github.com/Jittor/jittor/blob/2.0-refactor/agent/skills/jittor-distributed-verification/SKILL.md).
This round's result is `recheck-bc07.json`; maintainers also re-derived the same 680/368/160
results with an independent self-written NumPy audit, and matched the 2,804 regular files in the measurement snapshot against the corresponding historical Git blobs one by one. The independent audit is recorded in
`$JITTOR_LAB_ROOT/_state/jtrun-control-integration-bd17/independent-source-raw-audit.json`
and `independent-array-audit.json` in the same directory.

In addition, after copying the real raw data, the full parameter gradients were removed, the real gradients were zeroed, the actual compute
metadata of the BF16 run was changed to FP32, and the resume side was made to reuse the producer PID; all four negative cases were rejected. The copies' modifications were rebound separately,
and the original 299 files were not rewritten. Recorded as `negative-recheck-abb4.json`.

The initial Torch-first legacy-pickle failure of the installation environment, the compile-lock wait caused by an incorrect eager sitecustomize,
and the warm failure where world-one does not auto-load NCCL are all retained. The final CPU private prefix deploys only ordinary package entries through the repository's
`compat/shim/deploy.py --target`, keeping no self-made finder or eager startup hook, and changing neither the shared environment nor the genuine Torch reference environment.

## Follow-up gates and boundaries

The CPU core result on clean `abb4f302c` was 322 passed, 42 skipped, 1 xfailed. The full structure check
was 1,507 passed, 6 skipped, 1 failed: the new tool test loaded modules dynamically during collection, violating the existing
collection side-effect contract. `5844a4581` defers the load to fixture execution; the tool and pytest/child
structure contracts total 46 passed, 0 skipped; the original failure is retained and not rewritten as a full structure pass.
The integrator re-verifies the full structure after the tool, the report and the central index/MANIFEST wiring are in place.
Logs are `unit-cpu/core-final.log`, `unit-cpu/structure-final.log` and
`unit-cpu/evidence-fixture-followup.log`; other hashes and layout are recorded in `final-verification-manifest.json`.
The CPU core runner uses native/Torch session caches separately; the runner overrides the outer `JITTOR_HOME`,
and the actual path is kept in the logs, separate from this round's CUDA rank caches; it cannot be claimed as the outer-specified unit home.
After the report was added, the targeted docs/private-path check was 18 passed, 1 failed; the only failure was that the new results page
was not yet wired into the central toctree; per file ownership this is handed to the integrator to add the results index/MANIFEST and re-verify.
Layout, Ruff on the changed Python files and diff whitespace checks pass.
The first full structure check in fact found the new `JT_NCCL_LOCAL_WORLD_RANKS` unregistered; it was registered and the targeted re-verification passed.

This round had only one physical host. The host's four global ranks prove only rank arithmetic and the protocol; real CUDA was two
agents with one card each. Genuine cross-host TCP/NCCL routing, firewalls, different filesystems and clocks, network partitions,
multi-host training and recovery still require real multi-machine acceptance. The control plane does not transfer checkpoints; the existing shared-file-path requirement is retained.
Multi-node HCCL, elastic membership/restart and end-to-end deadlines for arbitrary collectives are outside this round's accepted scope.
R17 cannot be closed overall on the strength of this batch of single-host software and device evidence.

## Independent integration re-verification in the main tree

The main tree first synchronized to remote `bd17d080ee31c8cb84f3ac2617662f423c343fec`, and after integrating the production changes all computation
was done on the frozen source of the clean `bce0f96f1adf42ae1eabd5ceabdb014ef1902839`.
The original historical measurement remains bound to `bc07df0c079fcaa6ffcefce7ede1423ca599305e`.
The main tree also keeps raw evidence in the unversioned `$JITTOR_LAB_ROOT/_state/jtrun-control-integration-bd17/`.

Each of the two ranks serially completed a real NCCL FP32/BF16 warm-up; then the public jtrun two-agent runs passed
3 items each, and the original single-agent two-card runs passed 2 items each. SGD/AdamW × FP32/BF16 parity,
baseline and resume in a separate process passed 4 items/rank each, with a fallback delta of 0 in all cases.
The production workers were `849104/849100` and the resume workers `853567/853565`.
After STEP 5 in practice, this task's rank 1 was killed, the agent returned `[1,137]`, exiting in about 0.666 seconds;
all 12 formal worker PIDs no longer existed. The elapsed time only describes that failure exit and is not a performance conclusion.

The main tree's independent NumPy audit and the maintained recomputation tool separately re-checked the 680 oracle arrays, the 368 bit-exact resume
fields, the 160 update budgets and the 160 zero-gradient rejections, all passing the original budgets.
`raw-bindings-main-bce0.json` seals 273 raw/source/log files;
the 2,811 Git blobs in `source-main-bce0-complete.json` were matched one by one, including the target text of the directory symlink.
The initial regular-file snapshot is kept separately; the historical snapshot and raw arrays were not rewritten.

Host regression on the same frozen production source was 46 passed/1 skipped, core 322 passed/42 skipped/
1 xfailed, and full structure 1,508 passed/6 skipped. The final check that then integrated the pure audit tool and the report
is recorded separately in `integration-manifest.json` and `final-structure.log` in the same directory.
Source hashes were verified at each stage; existing xfails/skips do not count as actual CUDA execution.
This run again had only one physical host, and R17's real cross-host acceptance stays retained.

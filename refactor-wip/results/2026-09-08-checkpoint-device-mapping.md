# Checkpoint device mapping

- Status: saved-device mapping and indexed CUDA restoration implemented
- Baseline: `1667f34dc`
- Date: 2026-09-08
- Owner: coord
- Review when: checkpoint transport, placement or Parameter storage updates change

Portable checkpoints record source devices. Torch zip storage locations feed a
per-load map, so map_location dictionaries and callbacks receive the actual
saved location instead of an unconditional cpu label. Unmapped/default targets
restore that location; explicit cuda:N retains its index. A capable accelerator
can be enabled when explicitly requested rather than rejected just because its
runtime flag was initially off. Old portable files without device metadata
retain their CPU default.

The two-device check first exposed tensor/as_tensor/from_numpy ignoring their
CUDA index; their constructors now forward it through the existing device-copy
path. It then exposed clone().numpy() moving the original sharing group to CPU
during save, corrupting the saved location too. Supported checkpoint tensor
containers now collect unique tensors and fetch host snapshots in one batch;
the live tensors retain their residency, and fetch/synchronization failures
propagate. Parameter CPU restoration migrates its holder directly. Combined
Module dtype/CPU conversion also preserves the intended host residency after
detaching the replacement storage.

Verification: twenty focused CPU checkpoint cases passed in 5.73 s; three CUDA
integration cases passed in 7.89 s. Two visible CUDA devices were used: construct
and save on device 1, assert the source remains device-resident, explicitly
reload on device 1, restore its saved location by default, and remap it to CPU
while preserving Parameter type. Module dtype-plus-CPU conversion was exercised
as well. Zip fixtures separately verify source-location callbacks and dictionary
mapping without accelerator hardware. The legacy serialization file reported
retained runtime objects (14 holders, 12 Vars); this was not a full liveness
audit. No full suite, wheel rebuild or NPU run was performed.

7.12 remains open. Shared storage/stride reconstruction, arbitrary object graphs
and full external-model coverage are not established by these focused cases.
Per-load device bookkeeping does not create a process-global tensor-id registry.
